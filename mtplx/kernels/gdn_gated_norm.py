"""The sigmoid output gate of Flash-Next's GDN norm as one elementwise kernel.

``SigmoidRMSNormGated`` finishes every GDN layer with

    g = sigmoid(gate.astype(float32))
    y = (g * x.astype(float32)).astype(bfloat16)     # x = rms_norm(out)

which at prefill width is five passes over ``[rows, 48, 128]`` tensors, three
of them float32.  This kernel does the gate in one pass.  ``gate`` is bf16, so
its float32 sigmoid takes only 65,536 distinct values: they come from a table
the installed MLX computed with its own ``mx.sigmoid`` (so JIT and metallib
transcendentals can never disagree), and the product and the bf16 round are
the stock ones.  The result is bit-identical to the stock expression
(tests/test_gdn_gated_norm.py).

Every GPU generation: the kernel is plain SIMD (a table read, one float
multiply, 256-thread groups) with no tensor units, so it runs on M1 to M5, but
only an M5 has measured it.  The load-time self-check
(``mtplx.kernel_selfcheck``, lane ``qwen4_gdn_gated_norm``) compares its bits
with the stock expression on this GPU and turns it off for the process on any
difference or a build or launch failure.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..kernel_selfcheck import lane_disabled

ENV = "MTPLX_QWEN4_GDN_GATED_NORM"
LANE = "qwen4_gdn_gated_norm"

_SOURCE = """
    const uint i = thread_position_in_grid.x;
    const float g = sigmoid_table[as_type<ushort>(gate[i])];
    out[i] = bfloat(g * float(x[i]));
"""


@lru_cache(maxsize=1)
def sigmoid_table() -> mx.array:
    """``sigmoid(float32(b))`` for every bf16 bit pattern ``b``, by the installed MLX."""

    bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
    table = mx.sigmoid(bits.view(mx.bfloat16).astype(mx.float32))
    mx.eval(table)
    return table


@lru_cache(maxsize=4)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gdn_sigmoid_gate",
        input_names=["x", "gate", "sigmoid_table"],
        output_names=["out"],
        source=_SOURCE,
    )


def switched_on() -> bool:
    """The user's switch alone (default on)."""

    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def enabled() -> bool:
    """Switched on, and not turned off by the load-time self-check on this GPU."""

    return switched_on() and not lane_disabled(LANE)


def gated_eligible(x: mx.array, gate: mx.array) -> bool:
    if gate.dtype != mx.bfloat16 or x.dtype not in (mx.bfloat16, mx.float32):
        return False
    if tuple(x.shape) != tuple(gate.shape) or x.size == 0:
        return False
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def sigmoid_gate(x: mx.array, gate: mx.array) -> mx.array:
    """``(sigmoid(gate.astype(f32)) * x.astype(f32)).astype(bf16)`` in one pass."""

    (out,) = _kernel()(
        inputs=[x, gate, sigmoid_table()],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )
    return out
