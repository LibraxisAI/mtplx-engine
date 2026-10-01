"""Flash-Next prefill MoE combine: unsort, weight, reduce and add the shared expert.

At prefill width the routed experts run expert-sorted (mlx_lm's
``_gather_sort`` feeding ``gather_qmm(sorted_indices=True)``).  The stock tail
then materializes three ``[rows, top_k, hidden]`` BF16 tensors in a row: the
unsorted expert outputs (``_scatter_unsort``), their product with the routing
scores, and the column sum over ``top_k``.  At 4,096 rows that is about 1 GB of
memory traffic per MoE layer for a 21 MB result.

This kernel reads each expert output once, straight from the sorted buffer
through the inverse permutation, and writes the combined row:

    out[t] = sum_r bf16(y_sorted[inv_order[t * top_k + r]] * scores[t, r]) + shared[t]

Every rounding step is the stock chain's: the product rounds to BF16 as the
materialized ``y * scores[..., None]`` does, the reduction reproduces MLX's
``col_reduce_small`` order for a K-deep BF16 column reduction (``TY = min(8, K)``
partial accumulators, partial y summing rows ``y, y + TY, ...`` in ascending
order, partials combined in ascending y, all in BF16; the same order
``laguna_prefill_moe_combine.py`` pins), and the shared-expert add is the one
BF16 add ``y + shared_y`` performs.  The result is bit-identical to the stock
tail.

Every GPU generation: the kernel is plain SIMD (BF16 arithmetic, 256-thread
groups, no threadgroup memory) with no tensor units, so it runs on M1 to M5,
but only an M5 has measured it.  The load-time self-check
(``mtplx.kernel_selfcheck``, lane ``qwen4_moe_prefill_combine``) compares its
bits with the stock tail on this GPU and turns it off for the process on any
difference or a build or launch failure.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..kernel_selfcheck import lane_disabled

ENV = "MTPLX_QWEN4_MOE_PREFILL_COMBINE"
LANE = "qwen4_moe_prefill_combine"
_MAX_TOP_K = 32


def switched_on() -> bool:
    """The user's switch alone (default on)."""

    raw = (os.environ.get(ENV) or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def enabled() -> bool:
    """Switched on, and not turned off by the load-time self-check on this GPU."""

    return switched_on() and not lane_disabled(LANE)


def _on_metal_device() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def combine_eligible(y_sorted, inv_order, scores, shared) -> bool:
    """Metadata-only checks; anything else keeps the stock tail."""

    if not _on_metal_device():
        return False
    if y_sorted.ndim != 2 or inv_order.ndim != 1 or scores.ndim != 2 or shared.ndim != 2:
        return False
    if y_sorted.dtype != mx.bfloat16 or scores.dtype != mx.bfloat16 or shared.dtype != mx.bfloat16:
        return False
    if inv_order.dtype not in (mx.uint32, mx.int32):
        return False
    rows, top_k = (int(d) for d in scores.shape)
    hidden = int(y_sorted.shape[1])
    if not 0 < top_k <= _MAX_TOP_K:
        return False
    if int(y_sorted.shape[0]) != rows * top_k or int(inv_order.shape[0]) != rows * top_k:
        return False
    return tuple(int(d) for d in shared.shape) == (rows, hidden)


@lru_cache(maxsize=None)
def _kernel(top_k: int, hidden: int):
    ty = min(8, top_k)
    header = f"""
        using namespace metal;
        constant constexpr int TOP_K = {top_k};
        constant constexpr int HIDDEN = {hidden};
        constant constexpr int TY = {ty};
    """
    source = """
        uint idx = thread_position_in_grid.x;
        uint row = idx / uint(HIDDEN);
        uint c = idx - row * uint(HIDDEN);

        T totals[TY];
        for (int y = 0; y < TY; ++y) {
            totals[y] = T(0);
        }
        for (int r = 0; r < TOP_K; ++r) {
            size_t src = (size_t)inv_order[(size_t)row * TOP_K + r];
            T prod = y_sorted[src * (size_t)HIDDEN + c] * scores[(size_t)row * TOP_K + r];
            totals[r % TY] = prod + totals[r % TY];
        }
        T total = totals[0];
        for (int y = 1; y < TY; ++y) {
            total = totals[y] + total;
        }
        out[idx] = total + shared[idx];
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_qwen4_moe_prefill_combine_k{top_k}_h{hidden}",
        input_names=["y_sorted", "inv_order", "scores", "shared"],
        output_names=["out"],
        header=header,
        source=source,
    )


def moe_prefill_combine_reference(y_sorted, inv_order, scores, shared):
    """The stock tail this kernel replaces, as mlx_lm writes it."""

    rows, top_k = (int(d) for d in scores.shape)
    y = y_sorted[inv_order].reshape(rows, top_k, -1)
    return (y * scores[..., None]).sum(axis=-2) + shared


def moe_prefill_combine(y_sorted, inv_order, scores, shared):
    """``[rows, hidden]`` combined MoE output (see the module docstring).

    ``y_sorted`` is ``[rows * top_k, hidden]`` in expert-sorted order,
    ``inv_order`` maps each (row, slot) to its sorted position, ``scores`` is
    ``[rows, top_k]`` and ``shared`` the gated shared-expert output
    ``[rows, hidden]``.  Shapes the kernel does not cover take the reference.
    """

    if not combine_eligible(y_sorted, inv_order, scores, shared):
        return moe_prefill_combine_reference(y_sorted, inv_order, scores, shared)
    return _launch(y_sorted, inv_order, scores, shared)


def _launch(y_sorted, inv_order, scores, shared):
    """The kernel itself, for inputs the caller has checked (the self-check probes this)."""

    rows, top_k = (int(d) for d in scores.shape)
    hidden = int(y_sorted.shape[1])
    total = rows * hidden
    (out,) = _kernel(top_k, hidden)(
        inputs=[y_sorted, inv_order, scores, shared],
        template=[("T", mx.bfloat16)],
        grid=(total, 1, 1),
        threadgroup=(256 if total >= 256 else 32, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[mx.bfloat16],
    )
    return out
