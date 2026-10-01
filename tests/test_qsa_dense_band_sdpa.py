"""Dense-band QSA attention with the mask applied in the score GEMM (2026-09-23).

``mtplx.kernels.qsa_dense_band_sdpa`` must be bit-identical to
``mx.fast.scaled_dot_product_attention`` wherever it is eligible (head dim 256,
where MLX runs its unfused fallback): on both of MLX's softmax kernels (one
pass up to 4,096 columns, looped above), across the 4,096 boundary, for sparse
and dense masks and for a fully masked row, and it must not peak higher than
the stock call by more than its ``[S, T]`` mask plane.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.kernels.qsa_dense_band_sdpa import dense_band_eligible, dense_band_sdpa


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16))


def _case(S, T, density, seed, hq=24, hkv=2, d=256):
    mx.random.seed(seed)
    q = mx.random.normal((1, hq, S, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, T, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, T, d)).astype(mx.bfloat16)
    qpos = (T - S) + mx.arange(S)
    tpos = mx.arange(T)
    causal = tpos[None, :] <= qpos[:, None]
    picked = mx.random.uniform(shape=(S, T)) < density
    tail = tpos[None, :] >= qpos[:, None] - 63
    mask = (causal & (picked | tail))[None, None]
    return q, k, v, mask


@pytest.mark.parametrize(
    "S,T,density",
    [(64, 4096, 0.25),     # one-pass softmax, exactly the limit
     (37, 4082, 0.25),     # one-pass, ragged last thread
     (16, 1000, 0.5),
     (40, 4097, 0.25),     # looped softmax, one column past the limit
     (64, 8192, 0.25),
     (48, 12288, 0.17),
     (32, 16384, 0.125),
     (512, 4608, 0.25)],   # a wider row block
)
def test_bit_identical_to_the_stock_sdpa(S, T, density):
    q, k, v, mask = _case(S, T, density, seed=S * 7 + T)
    scale = 256 ** -0.5
    assert dense_band_eligible(q, k, mask)
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    got = dense_band_sdpa(q, k, v, scale=scale, mask=mask)
    assert got.shape == want.shape
    assert np.array_equal(_bits(got), _bits(want))


@pytest.mark.parametrize("T", [3000, 9000])
def test_a_fully_masked_row_attends_uniformly_like_the_fallback(T):
    q, k, v, mask = _case(12, T, 0.3, seed=T)
    mask = mx.array(np.array(mask).copy())
    mask[..., 0, :] = False
    scale = 256 ** -0.5
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    got = dense_band_sdpa(q, k, v, scale=scale, mask=mask)
    assert not np.isnan(np.array(got.astype(mx.float32))).any()
    assert np.array_equal(_bits(got), _bits(want))


def _peak_bytes(fn):
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    return mx.get_peak_memory() - base


def test_peak_memory_is_the_stock_calls_plus_the_mask_plane():
    S, T = 512, 4608
    q, k, v, mask = _case(S, T, 0.25, seed=5)
    mx.eval(q, k, v, mask)
    scale = 256 ** -0.5
    stock = _peak_bytes(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask))
    ours = _peak_bytes(lambda: dense_band_sdpa(q, k, v, scale=scale, mask=mask))
    plane = 24 * S * T * 2
    assert stock >= plane  # the fallback does hold one score plane
    assert ours <= stock + S * T * 2 + (1 << 20)


def test_eligibility_is_the_fallback_regime_only():
    q, k, v, mask = _case(16, 512, 0.5, seed=1)
    assert dense_band_eligible(q, k, mask)
    assert not dense_band_eligible(q[:, :, :8], k, mask[:, :, :8])          # vector-kernel band
    assert not dense_band_eligible(q[..., :128], k[..., :128], mask)        # MLX has a fused kernel
    assert not dense_band_eligible(q.astype(mx.float16), k.astype(mx.float16), mask)
    assert not dense_band_eligible(q, k, None)
    assert not dense_band_eligible(q, k, mask.astype(mx.bfloat16))          # additive masks stay stock
    assert not dense_band_eligible(q, k, mx.broadcast_to(mask, (1, 24, 16, 512)))


def test_the_model_switch_is_opt_in_and_prefill_only(monkeypatch):
    from mtplx.attention_context import attention_phase
    from mtplx.models import qwen4_exp

    q, k, v, mask = _case(64, 512, 0.5, seed=2)
    monkeypatch.delenv("MTPLX_QSA_DENSE_BAND_SDPA", raising=False)
    with attention_phase("prefill"):
        assert not qwen4_exp._qsa_dense_band_sdpa_applies(q, k, mask)
        monkeypatch.setenv("MTPLX_QSA_DENSE_BAND_SDPA", "1")
        assert qwen4_exp._qsa_dense_band_sdpa_applies(q, k, mask)
        assert not qwen4_exp._qsa_dense_band_sdpa_applies(q[:, :, :31], k, mask[:, :, :31])
    for phase in (None, "verify", "decode"):
        with attention_phase(phase):
            assert not qwen4_exp._qsa_dense_band_sdpa_applies(q, k, mask)
