"""The GDN norm's sigmoid output gate as one kernel (2026-09-23).

Bit-identical to ``(sigmoid(gate.astype(f32)) * x.astype(f32)).astype(bf16)``
for bf16 and float32 norm outputs, reached only by prefill-width GDN norms,
and off with ``MTPLX_QWEN4_GDN_GATED_NORM=0``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.attention_context import attention_phase
from mtplx.kernels.gdn_gated_norm import gated_eligible, sigmoid_gate
from mtplx.models import qwen4_exp


def _bits(a):
    mx.eval(a)
    return np.array(a.view(mx.uint16))


@pytest.mark.parametrize("rows,weight_dtype", [(4096, mx.bfloat16), (300, mx.float32), (64, mx.bfloat16)])
def test_bit_identical_to_the_stock_gate(rows, weight_dtype):
    mx.random.seed(rows)
    out = (mx.random.normal((1, rows, 48, 128)) * 2).astype(mx.bfloat16)
    weight = (1 + 0.1 * mx.random.normal((128,))).astype(weight_dtype)
    z = (mx.random.normal((1, rows, 16480)) * 3).astype(mx.bfloat16)[..., 10240:16384].reshape(1, rows, 48, 128)
    x = mx.fast.rms_norm(out, weight, 1e-6)
    want = (mx.sigmoid(z.astype(mx.float32)) * x.astype(mx.float32)).astype(mx.bfloat16)
    assert gated_eligible(x, z)
    assert np.array_equal(_bits(sigmoid_gate(x, z)), _bits(want))


def test_the_module_uses_it_only_for_prefill_width(monkeypatch):
    import mtplx.kernels.gdn_gated_norm as gn

    norm = qwen4_exp.SigmoidRMSNormGated(128)
    norm.set_dtype(mx.bfloat16)
    calls = []
    real = gn.sigmoid_gate
    monkeypatch.setattr(gn, "sigmoid_gate", lambda *a: calls.append(1) or real(*a))
    monkeypatch.delenv("MTPLX_QWEN4_GDN_GATED_NORM", raising=False)
    wide = mx.random.normal((1, 64, 48, 128)).astype(mx.bfloat16)
    gate = mx.random.normal((1, 64, 48, 128)).astype(mx.bfloat16)
    with attention_phase("prefill"):
        fused = norm(wide, gate)
    monkeypatch.setenv("MTPLX_QWEN4_GDN_GATED_NORM", "0")
    with attention_phase("prefill"):
        stock = norm(wide, gate)
    assert calls == [1]
    assert np.array_equal(_bits(fused), _bits(stock))
    monkeypatch.delenv("MTPLX_QWEN4_GDN_GATED_NORM", raising=False)
    with attention_phase("prefill"):
        norm(wide[:, :31], gate[:, :31])
    with attention_phase("verify"):
        norm(wide, gate)
    assert calls == [1]
