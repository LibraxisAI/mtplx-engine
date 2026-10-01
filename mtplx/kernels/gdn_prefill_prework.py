"""Flash-Next GDN prefill prework in one kernel: conv, silu and the q/k l2norm.

Between the fused in_proj and the gated-delta recurrence a prefill-width GDN
forward runs an eager chain over the ``[rows, 10240]`` q|k|v stream:

    conv_input = concat([conv_state, qkv])            # a copy of the stream
    conv_out   = silu(depthwise_conv1d_k4(conv_input))  # two more
    q, k, v    = split(conv_out)                       # q/k 16 x 128, v 48 x 128
    q = inv_scale * l2norm(q);  k = l2norm(k)          # f32 round trips

This kernel reads the stream once and writes q, k and v.  Every rounding step
is the eager chain's:

* the convolution accumulates its four taps in float in tap order and rounds
  once to bf16, which is MLX's depthwise conv1d (the products of two bf16
  values are exact in float, so fused and separate multiply-adds agree);
* ``silu`` is looked up in a 65,536-entry table that MLX's own ``nn.silu``
  computed for every bf16 bit pattern, so JIT and metallib transcendentals
  can never disagree;
* each head's sum of squares follows MLX's ``row_reduce_simple`` for a
  128-wide float row (32 lanes, four consecutive values per lane added in
  order, then ``simd_sum``), then ``precise::rsqrt`` as MLX's Rsqrt, the float
  multiply and the bf16 round, and ``inv_scale`` multiplies q as a bf16 scalar.

The output is bit-identical to the eager chain (tests/test_gdn_prefill_prework.py).
Geometry is Flash-Next's: conv_dim 10240 = q 2048 | k 2048 | v 6144, head dim
128, kernel 4, no conv bias.

Every GPU generation: the kernel is plain SIMD (float arithmetic, ``simd_sum``,
one 32-thread simdgroup per row and head, no threadgroup memory) with no
tensor units, so it runs on M1 to M5, but only an M5 has measured it.  The
load-time self-check (``mtplx.kernel_selfcheck``, lane
``qwen4_gdn_prefill_prework``) compares q, k and v with the eager chain's bits
on this GPU and turns it off for the process on any difference or a build or
launch failure.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..kernel_selfcheck import lane_disabled

ENV = "MTPLX_QWEN4_GDN_PREFILL_PREWORK"
LANE = "qwen4_gdn_prefill_prework"

_CONV_DIM = 10240
_KEY_DIM = 2048
_HEAD = 128
_KERNEL = 4
_HEADS = _CONV_DIM // _HEAD  # 16 q + 16 k + 48 v
_QK_HEADS = 2 * _KEY_DIM // _HEAD

_SOURCE = """
    // One simdgroup per (row, head); lane i owns channels 4i..4i+3 of the head.
    const uint head = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_position_in_threadgroup.x;
    const uint c0 = head * 128 + lane * 4;
    const size_t qkv_row_stride = size_t(qkv_strides[0]);
    const size_t qkv_col_stride = size_t(qkv_strides[1]);

    float vals[4];
    for (int j = 0; j < 4; ++j) {
        const uint c = c0 + j;
        float acc = 0.0f;
        for (int t = 0; t < 4; ++t) {
            // conv_input row (row + t) of [conv_state (3 rows), qkv (S rows)]
            const int src = int(row) + t - 3;
            const float x = (src < 0)
                ? float(conv_state[size_t(src + 3) * 10240 + c])
                : float(qkv[size_t(src) * qkv_row_stride + size_t(c) * qkv_col_stride]);
            acc += x * float(conv_w[c * 4 + t]);
        }
        const T conv = T(acc);
        vals[j] = float(silu_table[as_type<ushort>(conv)]);
    }

    if (head >= QK_HEADS) {
        device T* dst = v_out + size_t(row) * 6144 + (c0 - 4096);
        for (int j = 0; j < 4; ++j) dst[j] = T(vals[j]);
        return;
    }
    float part = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const float sq = vals[j] * vals[j];
        part = sq + part;
    }
    const float total = simd_sum(part);
    const float inv = metal::precise::rsqrt(total + 1e-6f);
    if (head < QK_HEADS / 2) {
        const T scale = T(inv_scale);
        device T* dst = q_out + size_t(row) * 2048 + c0;
        for (int j = 0; j < 4; ++j) dst[j] = scale * T(vals[j] * inv);
    } else {
        device T* dst = k_out + size_t(row) * 2048 + (c0 - 2048);
        for (int j = 0; j < 4; ++j) dst[j] = T(vals[j] * inv);
    }
"""


def switched_on() -> bool:
    """The user's switch alone (default on)."""

    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def enabled() -> bool:
    """Switched on, and not turned off by the load-time self-check on this GPU."""

    return switched_on() and not lane_disabled(LANE)


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_gdn_prefill_prework",
        input_names=["qkv", "conv_state", "conv_w", "silu_table", "inv_scale"],
        output_names=["q_out", "k_out", "v_out"],
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def silu_table() -> mx.array:
    """``nn.silu`` of every bf16 bit pattern, computed by the installed MLX."""

    import mlx.nn as nn

    bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
    table = nn.silu(bits.view(mx.bfloat16))
    mx.eval(table)
    return table


def prework_eligible(qkv, conv_state, conv_w) -> bool:
    """Metadata-only checks for the family geometry; anything else stays eager."""

    if qkv.ndim != 3 or qkv.shape[0] != 1 or qkv.shape[-1] != _CONV_DIM:
        return False
    if qkv.dtype != mx.bfloat16 or conv_state.dtype != mx.bfloat16:
        return False
    if tuple(int(d) for d in conv_state.shape) != (1, _KERNEL - 1, _CONV_DIM):
        return False
    if conv_w.dtype != mx.bfloat16 or conv_w.size != _CONV_DIM * _KERNEL:
        return False
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def gdn_prefill_prework(qkv, conv_state, conv_w, inv_scale: float):
    """``(q, k, v)`` as ``[1, S, heads, 128]`` bf16, exactly the eager chain's.

    ``qkv`` is the ``[1, S, 10240]`` q|k|v slice of the in_proj output (a view
    with any row stride), ``conv_state`` the ``[1, 3, 10240]`` tail before
    this forward, ``conv_w`` the depthwise weight.  The conv-state update is
    the caller's (the last three rows of ``[conv_state, qkv]``).
    """

    S = int(qkv.shape[1])
    q, k, v = _kernel()(
        inputs=[
            qkv[0],
            mx.contiguous(conv_state),
            mx.contiguous(conv_w.reshape(-1)),
            silu_table(),
            float(inv_scale),
        ],
        template=[("T", mx.bfloat16), ("QK_HEADS", _QK_HEADS)],
        grid=(_HEADS * 32, S, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(S, _KEY_DIM), (S, _KEY_DIM), (S, _CONV_DIM - 2 * _KEY_DIM)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )
    return (
        q.reshape(1, S, _KEY_DIM // _HEAD, _HEAD),
        k.reshape(1, S, _KEY_DIM // _HEAD, _HEAD),
        v.reshape(1, S, (_CONV_DIM - 2 * _KEY_DIM) // _HEAD, _HEAD),
    )
