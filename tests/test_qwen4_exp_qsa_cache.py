"""QSACache must be a full citizen of the cache contract.

The QSA indexer keeps its own raw-key stream (and derived pooled block keys)
next to the attention KV. The serve loop rolls caches back after every
speculative verify round (``rollback_after_verify``: trim for trimmable
entries, snapshot-restore for the rest) and resumes banked sessions through
``state``. A raw-key stream that only ever appends desyncs from the KV on the
first rollback; once the context crosses the indexer's engage threshold the
selection mask is built from the raw-stream length while attention keys come
from the KV — the ``broadcast_shapes (1,1,4,3719) vs (1,24,4,3715)`` crash
OpenCode hit live at 3.7k ctx (2026-08-27). Below the threshold the same
desync corrupts pooled blocks silently instead of crashing.

All runs are CPU (M-series GPU fp32 matmul is reduced-precision; CPU is the
parity surface).
"""

import mlx.core as mx
import pytest

import mtplx.graphbank as graphbank
from mtplx.cache_state import (
    rollback_after_verify,
    snapshot_untrimmable_cache,
)
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_tiny_args())
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.float32)


PREFILL = 12  # engage threshold with budget=8/ratio=2 is >8 visible tokens
STEP = 4  # a depth-3 verify round: 1 committed + 3 drafts


def test_rollback_then_forward_matches_fresh_run(attn):
    """A rejected verify round must leave the QSA layer exactly where a run
    that never saw the rejected tokens would be."""
    x_pre = _hidden(PREFILL, seed=1)
    x_rejected = _hidden(STEP, seed=2)
    x_next = _hidden(STEP, seed=3)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    assert cache[0].offset == PREFILL
    out = attn(x_next, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert out.shape == golden.shape
    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_state_roundtrip_resumes_identically(attn):
    """Bank restore: ``state`` must carry everything the layer needs — a
    resumed session past the engage threshold selects the same blocks and
    produces the same output as the uninterrupted run."""
    x_pre = _hidden(PREFILL, seed=4)
    x_next = _hidden(STEP, seed=5)

    live = QSACache()
    attn(x_pre, live)
    golden = attn(x_next, live)

    donor = QSACache()
    attn(x_pre, donor)
    resumed = QSACache()
    resumed.state = donor.state
    assert resumed.offset == PREFILL
    out = attn(x_next, resumed)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_trim_contract(attn):
    """QSACache is trimmable: trim rolls the layer back token-exactly,
    including through a pooled-block boundary."""
    cache = QSACache()
    assert cache.is_trimmable()

    x_pre = _hidden(PREFILL, seed=6)
    x_tail = _hidden(3, seed=7)  # odd length: trims back through a block edge
    x_next = _hidden(STEP, seed=8)

    attn(x_pre, cache)
    attn(x_tail, cache)
    assert cache.trim(3) == 3
    assert cache.offset == PREFILL
    out = attn(x_next, cache)

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_rollback_below_engage_threshold_still_exact(attn):
    """The desync is silent below the engage threshold (dense mask hides it);
    the pooled stream must still be positionally correct once the session
    grows past it."""
    x_pre = _hidden(4, seed=9)
    x_rejected = _hidden(STEP, seed=10)
    # two accepted rounds carry the session across the threshold
    x_a = _hidden(STEP, seed=11)
    x_b = _hidden(STEP, seed=12)
    x_c = _hidden(STEP, seed=13)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    for chunk in (x_a, x_b, x_c):
        out = attn(chunk, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    for chunk in (x_a, x_b, x_c):
        golden = attn(chunk, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_tensor_offset_qsa_cache_trim_matches_stock(attn):
    """The compiled-verifier cache owns fixed banks without changing QSA math."""
    x_pre = _hidden(PREFILL, seed=14)
    x_rejected = _hidden(STEP, seed=15)
    x_next = _hidden(STEP, seed=16)

    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, cache[0])
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache,
        reserve_tokens=STEP,
        initial_reserve_tokens=16,
    )

    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL

    attn(x_rejected, cache[0])
    assert cache[0].trim(STEP) == STEP
    out = attn(x_next, cache[0])

    fresh = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_compiled_verify_bank_threads_qsa_state_without_fallback(attn):
    class TinyQSARuntime:
        def __init__(self):
            mx.random.seed(17)
            self.attn = attn
            self.embed = mx.random.normal((32, 64)).astype(mx.float32)
            self.head = mx.random.normal((64, 32)).astype(mx.float32)

        def forward_ar_capture(
            self,
            input_ids,
            *,
            cache,
            return_hidden=True,
            hidden_variant=None,
            capture_backend=None,
        ):
            del hidden_variant, capture_backend
            hidden = self.attn(self.embed[input_ids], cache[0])
            logits = hidden @ self.head
            return logits, hidden, {}

    rt = TinyQSARuntime()
    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    rt.forward_ar_capture(
        mx.arange(PREFILL, dtype=mx.int32).reshape(1, -1), cache=cache
    )
    bank = graphbank.CompiledVerifyBank(rt, request_max_tokens=16)

    bank.forward_ar_capture(mx.array([[1, 2, 3, 4]]), cache=cache)
    bank.forward_ar_capture(mx.array([[5, 6, 7, 8]]), cache=cache)

    assert bank.stats["fallback_calls"] == 0, bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["traces"] == 1
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL + 8


# ---------------------------------------------------------------------------
# Image requests on the compiled verifier.
#
# The verify bank owns the request's rotary delta, every fixed QSA bank it
# promotes rotates at offset + delta, and the compiled step takes the delta as
# a graph INPUT. Two image requests with different deltas therefore replay one
# trace, and a text request replays the trace it always had.
# ---------------------------------------------------------------------------

import mlx.utils  # noqa: E402

import mtplx.models.qwen4_exp as qwen4_exp  # noqa: E402
from mtplx import qwen4_verify_glue, runtime_options  # noqa: E402
from mtplx.attention_context import vision_rope  # noqa: E402
from mtplx.vision.mrope import build_mrope_positions  # noqa: E402

VOCAB = 64
IMAGE_PAD = 63
PROMPT_TOKENS = 40
VERIFY_WINDOWS = ([1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12])


def _vision_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )


class TinyVisionQSARuntime:
    """One bf16 QSA layer between an embedding and a head.

    ``traces`` counts executions of the Python forward body while ANY verify
    bank is tracing, so it sees a retrace no matter which bank owns it.
    """

    def __init__(self, layer):
        mx.random.seed(17)
        self.attn = layer
        # Every id of the prompts below is a row of this table: an id past
        # it would be an out-of-bounds gather, whose garbage differs per call.
        self.embed = mx.random.normal((VOCAB, 64)).astype(mx.bfloat16)
        self.head = mx.random.normal((64, VOCAB)).astype(mx.bfloat16)
        mx.eval(self.embed, self.head)

    def forward_ar_capture(
        self,
        input_ids,
        *,
        cache,
        return_hidden=True,
        hidden_variant=None,
        capture_backend=None,
    ):
        del hidden_variant, capture_backend
        hidden = self.attn(self.embed[input_ids], cache[0])
        logits = hidden @ self.head
        return logits, hidden, {}


@pytest.fixture(params=["glue_off", "glue_on"])
def vision_rt(request, monkeypatch):
    glue = request.param == "glue_on"
    if glue and not mx.metal.is_available():
        pytest.skip("the rope glue kernels are Metal kernels")
    env = {"MTPLX_QWEN4_VERIFY_GLUE": "1"} if glue else {}
    prev = mx.default_device()
    mx.set_default_device(mx.gpu if glue else mx.cpu)
    runtime_options.reset_qwen4_verify_glue_cache(env=env)
    qwen4_verify_glue.reset_for_tests()
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    try:
        mx.random.seed(0)
        layer = Attention(_vision_args())
        layer.update(
            mlx.utils.tree_map(lambda p: p.astype(mx.bfloat16), layer.parameters())
        )
        mx.eval(layer.parameters())
        if glue:
            report = qwen4_verify_glue.install([(0, layer)], rows=4)
            assert report["items"]["qsa_rope"]["installed"], report
            assert report["items"]["qsa_rope_idx"]["installed"], report
        yield TinyVisionQSARuntime(layer)
    finally:
        runtime_options.reset_qwen4_verify_glue_cache(env={})
        qwen4_verify_glue.reset_for_tests()
        mx.set_default_device(prev)


def _image_request(image_tokens: int, grid):
    """A prompt of PROMPT_TOKENS ids holding one image: ids, table, delta.

    Both requests below have the same length (so the same bank shapes) and
    different images (so different deltas).
    """

    text = list(range(1, PROMPT_TOKENS - image_tokens + 1))
    ids = text[:6] + [IMAGE_PAD] * image_tokens + text[6:]
    assert len(ids) == PROMPT_TOKENS and IMAGE_PAD not in text
    assert max(ids) < VOCAB
    table, delta = build_mrope_positions(
        ids, image_token_id=IMAGE_PAD, image_grids=[grid], spatial_merge_size=2
    )
    return ids, mx.array(table), int(delta)


def _same_bits(a: mx.array, b: mx.array) -> bool:
    return a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def _serve(rt, ids, table, delta, *, poison=None):
    """One request: prefill, then compiled verify windows against eager.

    Returns the verify bank. ``poison`` is a request context opened around the
    COMPILED calls only: the fixed lane must not notice it.
    """

    from contextlib import nullcontext

    scope = (lambda: vision_rope(table, delta)) if table is not None else nullcontext
    prompt = mx.array([ids])
    cache = [QSACache(compress_ratio=rt.attn.indexer.ratio)]
    golden = [QSACache(compress_ratio=rt.attn.indexer.ratio)]
    with scope():
        rt.forward_ar_capture(prompt, cache=cache)
        rt.forward_ar_capture(prompt, cache=golden)
    bank = graphbank.CompiledVerifyBank(rt, request_max_tokens=16, rope_delta=delta)
    for window in VERIFY_WINDOWS:
        window = mx.array([window])
        with poison() if poison is not None else nullcontext():
            logits, hidden, _ = bank.forward_ar_capture(window, cache=cache)
        with scope():
            want_logits, want_hidden, _ = rt.forward_ar_capture(window, cache=golden)
        assert _same_bits(logits, want_logits)
        assert _same_bits(hidden, want_hidden)
    assert bank.stats["fallback_calls"] == 0, bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == len(VERIFY_WINDOWS)
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    end = cache[0].size()
    assert end == golden[0].offset == PROMPT_TOKENS + 4 * len(VERIFY_WINDOWS)
    assert _same_bits(cache[0].kv.keys[:, :, :end], golden[0].kv.keys[:, :, :end])
    assert _same_bits(
        cache[0].pooled[:, : end // 4], golden[0].pooled[:, : end // 4]
    )
    return bank


def test_two_image_requests_with_different_deltas_replay_one_trace(vision_rt):
    rt = vision_rt
    ids_a, table_a, delta_a = _image_request(16, (1, 8, 8))  # 4 x 4 rows
    ids_b, table_b, delta_b = _image_request(4, (1, 4, 4))  # 2 x 2 rows
    assert (delta_a, delta_b) == (-12, -2)

    def poison():
        # Another request's table and delta, open while the fixed lane runs.
        return vision_rope(table_a + 7, 4242)

    bank_a = _serve(rt, ids_a, table_a, delta_a, poison=poison)
    bank_b = _serve(rt, ids_b, table_b, delta_b, poison=poison)
    # A delta closed over as a Python value would have been folded into the
    # trace: request B would replay request A's positions and fail above, or
    # retrace and fail here.
    assert bank_a.stats["traces"] == 1
    assert bank_b.stats["traces"] == 0
    assert bank_a.to_dict()["compiled_keys"] == ["m4:default:b0:rope_delta"]
    assert bank_b.to_dict()["compiled_keys"] == ["m4:default:b0:rope_delta"]


def test_the_fixed_lane_never_reads_the_request_context(vision_rt, monkeypatch):
    """A context read on this lane bakes one request into every request's trace."""

    rt = vision_rt
    ids, table, delta = _image_request(16, (1, 8, 8))
    prompt = mx.array([ids])
    cache = [QSACache(compress_ratio=rt.attn.indexer.ratio)]
    with vision_rope(table, delta):
        rt.forward_ar_capture(prompt, cache=cache)
    bank = graphbank.CompiledVerifyBank(rt, request_max_tokens=16, rope_delta=delta)

    def read_is_a_bug():
        raise AssertionError("vision_rope_state() was read on the fixed lane")

    monkeypatch.setattr(qwen4_exp, "vision_rope_state", read_is_a_bug)
    with vision_rope(table, delta):  # the scope the generation loop has open
        logits, _hidden, _ = bank.forward_ar_capture(
            mx.array([VERIFY_WINDOWS[0]]), cache=cache
        )  # traced
        mx.eval(logits)
        # Eager forwards over the promoted bank: a short window and a copy
        # block width, the rounds the generation loop runs this way.
        for width in (2, 9):
            out = rt.forward_ar_capture(
                mx.array([list(range(1, width + 1))]), cache=cache
            )
            mx.eval(out[0])
    assert bank.stats["traces"] == 1 and bank.stats["compiled_calls"] == 1
    # The trap is live: the same forward over a STOCK cache does read the
    # context (that is how the eager verifier gets its positions).
    with pytest.raises(AssertionError, match="read on the fixed lane"):
        rt.forward_ar_capture(
            mx.array([VERIFY_WINDOWS[0]]),
            cache=[QSACache(compress_ratio=rt.attn.indexer.ratio)],
        )


def test_text_request_keeps_its_trace_and_its_bits(vision_rt):
    """A text request compiles the step it compiled before image requests
    could reach this lane: same key, same inputs, the offset object itself as
    the rotary origin, and the eager lane's bits."""

    rt = vision_rt
    ids = list(range(1, PROMPT_TOKENS + 1))
    calls = []
    real_step = graphbank.CompiledVerifyBank._make_verify_step

    def counting_step(self, *args, **kwargs):
        step = real_step(self, *args, **kwargs)

        def counted(input_ids, *leaves):
            calls.append(len(leaves))
            return step(input_ids, *leaves)

        return counted

    import unittest.mock

    with unittest.mock.patch.object(
        graphbank.CompiledVerifyBank, "_make_verify_step", counting_step
    ):
        bank = _serve(rt, ids, None, None)
    assert bank.to_dict()["compiled_keys"] == ["m4:default:b0"]
    assert calls == [5]  # traced once: K, V, offset, raw keys, pooled keys. No delta.
    image_ids, table, delta = _image_request(16, (1, 8, 8))
    calls.clear()
    with unittest.mock.patch.object(
        graphbank.CompiledVerifyBank, "_make_verify_step", counting_step
    ):
        _serve(rt, image_ids, table, delta)
    assert calls == [6]  # the delta, then the same five leaves
