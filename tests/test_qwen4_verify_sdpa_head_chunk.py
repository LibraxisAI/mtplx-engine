"""CPU tests for the small-q_len verify SDPA head-chunk lever.

MLX's fused vector-attention kernel serves at most q_len * (n_q_heads /
n_kv_heads) <= 32 rows per dispatch; a multi-row speculative verify over a long
KV (q_len 3-8 x GQA 12) exceeds it and falls to an unfused path that
materializes the [n_q_heads, q_len, T] score plane and GQA-expands k/v -- the
term that OOMs a 262K-context decode. The lever splits the query heads (each
chunk paired with its own single kv head, never GQA-expanding k/v) so every
dispatch stays on the bounded fused kernel. These tests cover the reader
(read-at-use), the chunk arithmetic, numerical equivalence to an unchunked
fused call, and that no [.., q_len, T] plane is ever formed -- all on CPU.
"""

import importlib
import os

import mlx.core as mx
import pytest

from mtplx.models import qwen4_exp as q4


GQA = 12
N_Q = 24
N_KV = 2
D = 64  # head_dim (small for CPU); exactness is dim-independent


def _clear_env():
    os.environ.pop("MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK", None)


@pytest.fixture(autouse=True)
def _cpu_and_reset():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    q4.reset_verify_sdpa_head_chunk_engagement()
    _clear_env()
    try:
        yield
    finally:
        q4.reset_verify_sdpa_head_chunk_engagement()
        _clear_env()
        mx.set_default_device(prev)


# ---------------------------------------------------------------------------
# Reader: default ON, resolved at use (served-order safe).
# ---------------------------------------------------------------------------


def test_reader_default_on_and_opt_out():
    assert q4._verify_sdpa_head_chunk_enabled() is True  # default ON
    for off in ("0", "off", "false", "no", "OFF"):
        os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = off
        assert q4._verify_sdpa_head_chunk_enabled() is False
    os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = "1"
    assert q4._verify_sdpa_head_chunk_enabled() is True


def test_served_order_reader_not_frozen():
    # Reproduce the served order: the model + server modules import before any
    # auto-arm/setdefault stamps env. A reader frozen at import would miss a
    # value set only now; read-at-use sees it.
    importlib.import_module("mtplx.server.openai")
    os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = "0"
    assert q4._verify_sdpa_head_chunk_enabled() is False
    os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = "1"
    assert q4._verify_sdpa_head_chunk_enabled() is True


# ---------------------------------------------------------------------------
# Chunk arithmetic for q_len 1..8 (and the prefill regime) at GQA 12.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_len,expect",
    [
        (1, None),
        (2, None),  # 2*12=24 <= 32: already fused
        (3, (10, 4)),
        (4, (8, 4)),
        (5, (6, 4)),
        (6, (5, 6)),
        (7, (4, 6)),
        (8, (4, 6)),
        (32, (1, 24)),
        (33, None),  # prefill regime: MLX routes to flash
        (256, None),
    ],
)
def test_chunk_plan(q_len, expect):
    assert q4._verify_sdpa_head_chunk_plan(q_len, N_Q, N_KV) == expect


def test_chunk_plan_invariant_holds_for_band():
    # Every chunked case keeps q_len * heads_per_chunk <= 32 (the fused
    # vector-kernel limit) and covers all heads.
    for q_len in range(3, 33):
        plan = q4._verify_sdpa_head_chunk_plan(q_len, N_Q, N_KV)
        assert plan is not None, q_len
        hpc, chunks = plan
        assert q_len * hpc <= q4._VECTOR_SDPA_QLEN_HEADS_LIMIT
        assert 1 <= hpc <= GQA
        chunks_per_kv = (GQA + hpc - 1) // hpc
        assert chunks == N_KV * chunks_per_kv


def test_chunk_plan_non_divisible_geometry_passes_through():
    # If n_q_heads is not a multiple of n_kv_heads, GQA slicing is undefined;
    # the helper must decline rather than corrupt the head mapping.
    assert q4._verify_sdpa_head_chunk_plan(6, 25, 2) is None


# ---------------------------------------------------------------------------
# Numerical equivalence: chunked fused == unchunked fused reference.
# ---------------------------------------------------------------------------


def _reference_mha(q, k, v, scale, mask):
    # Unchunked fused reference: GQA-expand k/v to MHA (n_kv == n_q), so the
    # fused kernel serves it directly for q_len <= 32. Mathematically identical
    # to GQA; head i attends to kv head i // gqa in both.
    gqa = q.shape[1] // k.shape[1]
    k_exp = mx.repeat(k, gqa, axis=1)
    v_exp = mx.repeat(v, gqa, axis=1)
    return mx.fast.scaled_dot_product_attention(q, k_exp, v_exp, scale=scale, mask=mask)


@pytest.mark.parametrize("S", [3, 4, 5, 6, 7, 8])
def test_chunked_matches_unchunked_fused(S):
    mx.random.seed(1234 + S)
    T = 200
    pos_start = T - S
    q = mx.random.normal((1, N_Q, S, D)).astype(mx.float32)
    k = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    v = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    scale = D**-0.5
    qpos = pos_start + mx.arange(S)
    tpos = mx.arange(T)
    mask = (tpos[None, :] <= qpos[:, None])[None, None]

    os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = "1"
    chunked = q4._verify_sdpa(q, k, v, scale=scale, mask=mask)
    ref = _reference_mha(q, k, v, scale, mask)
    mx.eval(chunked, ref)
    assert chunked.shape == (1, N_Q, S, D)
    max_abs = float(mx.max(mx.abs(chunked - ref)))
    assert max_abs < 1e-6, f"S={S} max_abs={max_abs}"


def test_opt_out_uses_single_unchunked_call(monkeypatch):
    calls = _record_sdpa_calls(monkeypatch)
    S, T = 6, 128
    q = mx.random.normal((1, N_Q, S, D)).astype(mx.float32)
    k = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    v = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = "0"
    out = q4._verify_sdpa(q, k, v, scale=D**-0.5, mask=None)
    mx.eval(out)
    assert len(calls) == 1
    # opt-out: one call with the full head count (the unfused band)
    assert calls[0][1] == N_Q  # n_q_heads
    assert q4.verify_sdpa_head_chunk_report() is None


# ---------------------------------------------------------------------------
# No [.., q_len, T] plane: every dispatch stays within the fused-kernel bound.
# ---------------------------------------------------------------------------


def _record_sdpa_calls(monkeypatch):
    real = mx.fast.scaled_dot_product_attention
    calls = []

    def _spy(q, k, v, *args, **kwargs):
        # record (heads*S, n_q_heads, S, n_kv_heads)
        calls.append((q.shape[1] * q.shape[2], q.shape[1], q.shape[2], k.shape[1]))
        return real(q, k, v, *args, **kwargs)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", _spy)
    return calls


def test_chunked_path_never_exceeds_vector_bound(monkeypatch):
    calls = _record_sdpa_calls(monkeypatch)
    S, T = 6, 4096  # long KV: an unfused path would form a [24, 6, 4096] plane
    q = mx.random.normal((1, N_Q, S, D)).astype(mx.float32)
    k = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    v = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = "1"
    out = q4._verify_sdpa(q, k, v, scale=D**-0.5, mask=None)
    mx.eval(out)
    assert out.shape == (1, N_Q, S, D)
    # Every dispatch is a single-kv-head fused call within the vector bound;
    # none is the full-24-head call that would materialize the O(T) plane.
    assert len(calls) == 6  # q_len 6 -> heads_per_chunk 5 -> 2*ceil(12/5)=6
    for heads_x_s, n_q, s, n_kv in calls:
        assert heads_x_s <= q4._VECTOR_SDPA_QLEN_HEADS_LIMIT
        assert n_kv == 1  # k/v sliced to one head, never GQA-expanded
        assert n_q <= 5


def test_decode_single_row_not_chunked(monkeypatch):
    calls = _record_sdpa_calls(monkeypatch)
    S, T = 1, 512  # S=1 decode: 1*12=12 <= 32, already fused
    q = mx.random.normal((1, N_Q, S, D)).astype(mx.float32)
    k = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    v = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    out = q4._verify_sdpa(q, k, v, scale=D**-0.5, mask=None)
    mx.eval(out)
    assert len(calls) == 1
    assert calls[0][1] == N_Q  # single unchunked GQA call
    assert q4.verify_sdpa_head_chunk_report() is None


# ---------------------------------------------------------------------------
# Engagement receipt (for /health).
# ---------------------------------------------------------------------------


def test_engagement_report_and_reset():
    S, T = 6, 64
    q = mx.random.normal((1, N_Q, S, D)).astype(mx.float32)
    k = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    v = mx.random.normal((1, N_KV, T, D)).astype(mx.float32)
    os.environ["MTPLX_QWEN4_VERIFY_SDPA_HEAD_CHUNK"] = "1"
    assert q4.verify_sdpa_head_chunk_report() is None
    mx.eval(q4._verify_sdpa(q, k, v, scale=D**-0.5, mask=None))
    rep = q4.verify_sdpa_head_chunk_report()
    assert rep is not None
    assert rep["engaged"] is True
    assert rep["q_len"] == 6
    assert rep["heads_per_chunk"] == 5
    assert rep["chunks"] == 6
    assert rep["n_kv_heads"] == N_KV
    q4.reset_verify_sdpa_head_chunk_engagement()
    assert q4.verify_sdpa_head_chunk_report() is None
