"""The hyper-connection write at prefill width runs as one fused kernel, bit
for bit the eager result (Flash-Next, 2026-09-18).

Synthetic tensors, no model. The compiled write is the eager write op for op
on the same views, so every output bit must match at every width, and the
switch must restore the eager pair of kernels.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models import qwen4_exp
from mtplx.models.qwen4_exp import _hyper_residual_write


def _bits(array: mx.array) -> np.ndarray:
    return np.array(array.view(mx.uint16))


def _inputs(rows, dtype, hc=4, hidden=64, seed=0):
    mx.random.seed(seed)
    hyper = (3 * mx.random.normal((1, rows, hc * hidden))).astype(dtype)
    block_out = mx.random.normal((1, rows, hidden)).astype(dtype)
    inject = (2 * mx.sigmoid(mx.random.normal((1, rows, hc)))).astype(dtype)
    return hyper, block_out, inject


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [32, 33, 100, 2048])
def test_compiled_write_equals_the_eager_write_bit_for_bit(monkeypatch, dtype, rows):
    hyper, block_out, inject = _inputs(rows, dtype)
    monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_COMPILE", "0")
    assert not qwen4_exp._hc_write_compile_applies(hyper)
    eager = _hyper_residual_write(hyper, block_out, inject)
    monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_COMPILE", "1")
    assert qwen4_exp._hc_write_compile_applies(hyper)
    compiled = _hyper_residual_write(hyper, block_out, inject)
    mx.eval(eager, compiled)
    assert compiled.shape == hyper.shape and compiled.dtype == dtype
    assert np.array_equal(_bits(eager), _bits(compiled))


def test_one_shapeless_trace_serves_every_width(monkeypatch):
    # A last chunk can be any width: the same compiled function must serve a
    # width it has never seen, after having served another one.
    monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_COMPILE", "1")
    assert qwen4_exp._hc_compiled_write(4, 64) is qwen4_exp._hc_compiled_write(4, 64)
    for seed, rows in enumerate((64, 4096, 33, 2972, 64)):
        hyper, block_out, inject = _inputs(rows, mx.bfloat16, seed=seed)
        compiled = _hyper_residual_write(hyper, block_out, inject)
        monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_COMPILE", "0")
        eager = _hyper_residual_write(hyper, block_out, inject)
        monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_COMPILE", "1")
        mx.eval(compiled, eager)
        assert compiled.shape == hyper.shape
        assert np.array_equal(_bits(eager), _bits(compiled)), rows


def test_decode_and_verify_widths_and_float32_stay_eager(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_COMPILE", "1")
    for rows in (1, 4, 8, 25, 31):
        assert not qwen4_exp._hc_write_compile_applies(
            mx.zeros((1, rows, 256), dtype=mx.bfloat16)
        ), rows
    assert qwen4_exp._hc_write_compile_applies(mx.zeros((1, 32, 256), dtype=mx.bfloat16))
    # Nothing rounds between the ops in float32, so a fused kernel may
    # contract them; the float32 path is the exact bisection path.
    assert not qwen4_exp._hc_write_compile_applies(mx.zeros((1, 64, 256), dtype=mx.float32))
    calls = []
    monkeypatch.setattr(qwen4_exp, "_hc_compiled_write", lambda *a: calls.append(a))
    hyper, block_out, inject = _inputs(64, mx.float32)
    _hyper_residual_write(hyper, block_out, inject)
    assert calls == []


def test_the_stock_spelling_is_untouched_when_the_op_diet_is_off(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_HC_PREFILL_COMPILE", "1")
    monkeypatch.setattr(qwen4_exp, "qwen4_opdiet_enabled", lambda item=None: False)
    calls = []
    monkeypatch.setattr(qwen4_exp, "_hc_compiled_write", lambda *a: calls.append(a))
    hyper, block_out, inject = _inputs(64, mx.bfloat16)
    out = _hyper_residual_write(hyper, block_out, inject)
    mx.eval(out)
    assert calls == [] and out.shape == hyper.shape
