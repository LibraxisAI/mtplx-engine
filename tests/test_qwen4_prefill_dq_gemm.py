"""Flash-Next's wide prefill projections as dequantize + dense GEMM (2026-09-23).

On by default on tensor-unit (M5-class) GPUs for prefill forwards of 2,048 rows
or more; decode and verify widths keep the quantized matmul everywhere.  An
M1 to M4 keeps the quantized matmul unless ``MTPLX_QWEN4_PREFILL_DQ_GEMM=1``
forces the lane, and ``=0`` turns it off everywhere.  The lane only ships
because, on a tensor-unit GPU at these widths, MLX's quantized matmul and a
dense GEMM over the dequantized weight are bit-identical: the tests below pin
that on two of the model's own projection shapes, so an MLX upgrade that
changes either kernel's accumulation order fails here first.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mtplx import nax_detect
from mtplx.attention_context import attention_phase
from mtplx.models import qwen4_exp


@pytest.fixture
def tensor_units(monkeypatch):
    """The one detector's M5 answer (hardware truth patched, switch unset)."""

    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", raising=False)
    monkeypatch.setattr(nax_detect, "nax_hardware_available", lambda: True)


@pytest.fixture
def no_tensor_units(monkeypatch):
    """The M1 to M4 answer, through the real rehearsal switch."""

    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")


def _bits_equal(a: mx.array, b: mx.array) -> bool:
    mx.eval(a, b)
    return a.dtype == b.dtype and np.array_equal(
        np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16))
    )


def _quantized(n, k, bits, group, seed=0):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=group, bits=bits)


def _refuse(*args, **kwargs):  # pragma: no cover - must not be reached
    raise AssertionError("the stock path dequantized its weight")


def test_the_lane_is_on_by_default_on_tensor_units_and_the_switch_turns_it_off(
    monkeypatch, tensor_units
):
    x = mx.zeros((1, 4096, 64), dtype=mx.bfloat16)
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    with attention_phase("prefill"):
        assert qwen4_exp._prefill_dq_gemm_applies(x)
        for off in ("0", "false", "no", "off", " OFF "):
            monkeypatch.setenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", off)
            assert not qwen4_exp._prefill_dq_gemm_applies(x)


def test_without_tensor_units_the_quantized_matmul_runs(monkeypatch, no_tensor_units):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    wq, sc, bi = _quantized(1280, 2560, 4, 32)
    x = (mx.random.normal((1, 2048, 2560)) * 0.5).astype(mx.bfloat16)
    with attention_phase("prefill"):
        assert not qwen4_exp._prefill_dq_gemm_applies(x)
        with monkeypatch.context() as m:
            m.setattr(qwen4_exp.mx, "dequantize", _refuse)
            got = qwen4_exp._projection(x, wq, sc, bi, group_size=32, bits=4, mode="affine")
    want = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=32, bits=4)
    assert _bits_equal(got, want)


def test_without_tensor_units_linear_is_the_module_call(monkeypatch, no_tensor_units):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    layer = nn.QuantizedLinear(256, 64, bias=False, group_size=32, bits=4)
    layer.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 2048, 256)).astype(mx.bfloat16)
    with attention_phase("prefill"):
        with monkeypatch.context() as m:
            m.setattr(qwen4_exp.mx, "dequantize", _refuse)
            got = qwen4_exp._linear(layer, x)
    assert _bits_equal(got, layer(x))


def test_an_explicit_on_forces_the_lane_without_tensor_units(monkeypatch, no_tensor_units):
    x = mx.zeros((1, 2048, 64), dtype=mx.bfloat16)
    for on in ("1", "true", "yes", "on", " ON "):
        monkeypatch.setenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", on)
        with attention_phase("prefill"):
            assert qwen4_exp._prefill_dq_gemm_applies(x)
        # The width and phase rules still hold for a forced lane.
        with attention_phase("decode_verify"):
            assert not qwen4_exp._prefill_dq_gemm_applies(x)
        with attention_phase("prefill"):
            assert not qwen4_exp._prefill_dq_gemm_applies(x[:, :2047])


@pytest.mark.parametrize(
    "rows,phase,expected",
    [(4096, "prefill", True), (2048, "prefill", True), (2047, "prefill", False),
     (4096, None, False), (4096, "verify", False), (1, "prefill", False)],
)
def test_the_lane_needs_a_wide_prefill_forward(monkeypatch, tensor_units, rows, phase, expected):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    x = mx.zeros((1, rows, 64), dtype=mx.bfloat16)
    with attention_phase(phase):
        assert qwen4_exp._prefill_dq_gemm_applies(x) is expected


@pytest.mark.skipif(
    not nax_detect.nax_hardware_available(),
    reason="bit-identity is an M5 (tensor-unit) receipt; the lane is off on other GPUs",
)
@pytest.mark.parametrize(
    "n,k,bits,group",
    [(1280, 2560, 4, 32),   # routed/shared gate+up geometry, 4-bit g32
     (2560, 640, 8, 64)],   # shared-expert down projection, 8-bit g64
)
def test_the_dense_gemm_is_bit_identical_to_the_quantized_matmul(monkeypatch, n, k, bits, group):
    # Forced on, so this pins the dense GEMM itself and cannot pass by
    # comparing the quantized matmul with itself.
    monkeypatch.setenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", "1")
    wq, sc, bi = _quantized(n, k, bits, group)
    x = (mx.random.normal((1, 2048, k)) * 0.5).astype(mx.bfloat16)
    with attention_phase("prefill"):
        got = qwen4_exp._projection(x, wq, sc, bi, group_size=group, bits=bits, mode="affine")
    want = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=group, bits=bits)
    assert _bits_equal(got, want)


def test_narrow_forwards_keep_the_quantized_matmul(monkeypatch):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    wq, sc, bi = _quantized(64, 256, 4, 32)
    x = mx.random.normal((1, 8, 256)).astype(mx.bfloat16)

    def refuse(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a narrow forward dequantized its weight")

    monkeypatch.setattr(qwen4_exp.mx, "dequantize", refuse)
    with attention_phase("prefill"):
        got = qwen4_exp._projection(x, wq, sc, bi, group_size=32, bits=4, mode="affine")
    monkeypatch.undo()
    want = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=32, bits=4)
    assert _bits_equal(got, want)


def test_linear_routes_only_a_bias_free_quantized_linear(monkeypatch, tensor_units):
    monkeypatch.delenv("MTPLX_QWEN4_PREFILL_DQ_GEMM", raising=False)
    layer = nn.QuantizedLinear(256, 64, bias=False, group_size=32, bits=4)
    layer.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 2048, 256)).astype(mx.bfloat16)
    dense = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=32, bits=4)
    with attention_phase("prefill"):
        assert _bits_equal(qwen4_exp._linear(layer, x), mx.matmul(x, dense.T))
    # A layer with a bias, a plain Linear, and a verify-width call are the module call.
    biased = nn.QuantizedLinear(256, 64, bias=True, group_size=32, bits=4)
    biased.set_dtype(mx.bfloat16)
    plain = nn.Linear(256, 64, bias=False)
    plain.set_dtype(mx.bfloat16)
    with attention_phase("prefill"):
        assert _bits_equal(qwen4_exp._linear(biased, x), biased(x))
        assert _bits_equal(qwen4_exp._linear(plain, x), plain(x))
    with attention_phase("verify"):
        assert _bits_equal(qwen4_exp._linear(layer, x[:, :4]), layer(x[:, :4]))
