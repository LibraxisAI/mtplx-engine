"""A quantized projection's bf16 cast must not change eager verify numerics."""

import mlx.core as mx
import pytest

from mtplx.attention_math import attention_gate
from mtplx.attention_context import attention_phase


def test_bf16_eager_gate_matches_the_existing_compiled_projection_gate():
    # The real 27B first differed at this input, after identical Q/K/V and
    # attention outputs. Include both tails and the ordinary operating range.
    raw = mx.array([-30.0, -10.0, -6.85, -6.84, -1.0, 0.0, 1.0, 10.0])
    output = mx.array([0.5, -2.0, 1.0, 4.0, -0.25, 2.0, 0.1, -0.5], dtype=mx.bfloat16)
    mx.eval(raw, output)

    def existing_trace(raw, output):
        return output * mx.sigmoid(raw.astype(mx.bfloat16))

    def candidate_trace(raw, output):
        return attention_gate(output, raw.astype(mx.bfloat16))

    with attention_phase("decode_verify"):
        reference = mx.compile(existing_trace)(raw, output)
        eager = candidate_trace(raw, output)
        compiled = mx.compile(candidate_trace)(raw, output)
    assert mx.array_equal(eager, reference).item()
    assert mx.array_equal(compiled, reference).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_other_gate_dtypes_keep_the_stock_expression(dtype):
    gate = mx.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=dtype)
    output = mx.array([0.5, -2.0, 1.0, 4.0, -0.25], dtype=dtype)
    assert mx.array_equal(attention_gate(output, gate), output * mx.sigmoid(gate)).item()


@pytest.mark.parametrize("phase", ["prefill", "ar_decode"])
def test_bf16_prefill_and_plain_decode_keep_the_stock_expression(phase):
    gate = mx.array([-6.85, -6.84, 0.0, 1.0], dtype=mx.bfloat16)
    output = mx.ones(gate.shape, dtype=mx.bfloat16)
    with attention_phase(phase):
        assert mx.array_equal(attention_gate(output, gate), output * mx.sigmoid(gate)).item()
