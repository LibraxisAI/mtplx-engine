"""Flash-Next GDN prefill prework as one kernel (2026-09-23).

``mtplx.kernels.gdn_prefill_prework`` replaces the prefill-width chain
concat -> depthwise conv1d -> silu -> split -> q/k l2norm -> q scale.  It must
be bit-identical to that chain (q, k, v and the conv-state tail), through a
whole family-geometry GatedDeltaNet forward with a cache, and stay off for
decode/verify widths and with ``MTPLX_QWEN4_GDN_PREFILL_PREWORK=0``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mtplx.attention_context import attention_phase
from mtplx.kernels.gdn_prefill_prework import gdn_prefill_prework, prework_eligible
from mtplx.models.qwen4_exp import GatedDeltaNet, TextArgs


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a)


def _eager(qkv, conv_state, conv):
    B, S, _ = qkv.shape
    conv_out = nn.silu(conv(mx.concatenate([conv_state, qkv], axis=1)))
    q, k, v = [
        t.reshape(B, S, h, 128)
        for t, h in zip(mx.split(conv_out, [2048, 4096], -1), [16, 16, 48])
    ]

    def l2(x):
        xf = x.astype(mx.float32)
        return (xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + 1e-6)).astype(x.dtype)

    return (128 ** -0.5) * l2(q), l2(k), v


@pytest.mark.parametrize(
    "S,scale,zero_state",
    [(4096, 1.0, False), (300, 3.0, False), (64, 1.0, True), (37, 0.05, False)],
)
def test_the_kernel_is_bit_identical_to_the_eager_chain(S, scale, zero_state):
    mx.random.seed(S)
    conv = nn.Conv1d(10240, 10240, kernel_size=4, groups=10240, bias=False)
    conv.set_dtype(mx.bfloat16)
    in_proj_out = (mx.random.normal((1, S, 16480)) * scale).astype(mx.bfloat16)
    qkv = in_proj_out[..., :10240]  # the strided view the model passes
    conv_state = (
        mx.zeros((1, 3, 10240), mx.bfloat16)
        if zero_state
        else (mx.random.normal((1, 3, 10240)) * scale).astype(mx.bfloat16)
    )
    assert prework_eligible(qkv, conv_state, conv.weight)
    got = gdn_prefill_prework(qkv, conv_state, conv.weight, 128 ** -0.5)
    for a, b in zip(got, _eager(qkv, conv_state, conv)):
        assert a.shape == b.shape
        assert np.array_equal(_bits(a), _bits(b))


def _family_gdn():
    args = TextArgs(
        hidden_size=256,
        num_hidden_layers=2,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    mx.random.seed(11)
    layer = GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    layer.eval()  # a loaded model runs in eval mode (the recurrence kernel needs it too)
    mx.eval(layer.parameters())
    return layer


def _cache():
    from mlx_lm.models.cache import ArraysCache

    return ArraysCache(size=2)


def test_the_module_forward_and_its_cache_are_bit_identical(monkeypatch):
    import mtplx.kernels.gdn_prefill_prework as pw

    layer = _family_gdn()
    x1 = (mx.random.normal((1, 96, 256)) * 0.5).astype(mx.bfloat16)
    x2 = (mx.random.normal((1, 64, 256)) * 0.5).astype(mx.bfloat16)
    calls = []
    real = pw.gdn_prefill_prework
    monkeypatch.setattr(pw, "gdn_prefill_prework", lambda *a: calls.append(1) or real(*a))

    def run():
        cache = _cache()
        with attention_phase("prefill"):
            y1 = layer(x1, cache=cache)
            y2 = layer(x2, cache=cache)  # second chunk: conv state carried over
        mx.eval(y1, y2, cache[0], cache[1])
        return y1, y2, cache[0], cache[1]

    monkeypatch.delenv("MTPLX_QWEN4_GDN_PREFILL_PREWORK", raising=False)
    fused = run()
    assert len(calls) == 2
    monkeypatch.setenv("MTPLX_QWEN4_GDN_PREFILL_PREWORK", "0")
    stock = run()
    assert len(calls) == 2
    for a, b in zip(fused, stock):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(_bits(a), _bits(b))


@pytest.mark.parametrize("rows,phase", [(1, "prefill"), (31, "prefill"), (64, "verify"), (64, None)])
def test_other_widths_and_phases_keep_the_staged_chain(monkeypatch, rows, phase):
    import mtplx.kernels.gdn_prefill_prework as pw

    def refuse(*args):  # pragma: no cover - must not be reached
        raise AssertionError("the prefill prework ran outside a prefill-width forward")

    monkeypatch.setattr(pw, "gdn_prefill_prework", refuse)
    monkeypatch.delenv("MTPLX_QWEN4_GDN_PREFILL_PREWORK", raising=False)
    layer = _family_gdn()
    with attention_phase(phase):
        y = layer((mx.random.normal((1, rows, 256)) * 0.5).astype(mx.bfloat16), cache=_cache())
    mx.eval(y)
    assert y.shape == (1, rows, 256)
