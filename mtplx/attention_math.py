"""The dense attention gate's numerical contract, eager and compiled alike."""

import mlx.core as mx

from .attention_context import current_attention_phase


@mx.compile
def _bf16_attention_gate(output: mx.array, gate: mx.array) -> mx.array:
    # Quantized projections reduce in fp32 and cast to bf16. In MLX 0.32.2,
    # fusing that cast with sigmoid differs from the standalone bf16 sigmoid
    # at some negative inputs (for example -6.84375). Use the same lowering
    # on eager forwards. The round trip is value-preserving and folds into
    # the existing cast inside an outer verify trace, preserving its result.
    # Computing sigmoid in fp32 and then casting would change that contract.
    return output * mx.sigmoid(gate.astype(mx.float32).astype(mx.bfloat16))


def attention_gate(output: mx.array, gate: mx.array) -> mx.array:
    if gate.dtype == mx.bfloat16 and current_attention_phase() == "decode_verify":
        return _bf16_attention_gate(output, gate)
    return output * mx.sigmoid(gate)
