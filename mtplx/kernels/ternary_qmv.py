"""Small-M GEMV for Prism's ternary 2-bit packs (Ternary Bonsai 2), float16.

Prism stores every rotated projection in MLX's affine 2-bit container (group
128) with ``biases == -scales``, so a weight is ``scale * (code - 1)`` with
``code`` in {0, 1, 2}: the bias term folds into the code and needs no second
multiply or activation sum. Stock ``mx.quantized_matmul`` does not know that
and runs its generic affine kernels; at 2 bits those are ALU-bound, not
bandwidth-bound, on the M5 Max (2026-09-21 receipt: 250-420 GB/s on the 27B
shapes against a 614 GB/s bus).

This kernel reads each packed weight word once for all ``M`` activation rows
(verify windows, M = 1..4). Codes decode with the half2 magic-number trick
(one AND + OR per two codes turns a 2-bit field into an exact half integer),
products are exact in float32 (a small integer times a float16 value) and
every sum is float32. The result is therefore the same mathematical dot
product as stock with a different float32 summation order; it rounds to the
same float16 value in the vast majority of outputs and never differs by more
than float32 rounding before the final float16 cast. Measured on the real pack
over 1,630 teacher-forced verify positions (2026-09-23): mean KL to Prism's
float32 logits 2.92e-6 against stock's 3.02e-6 (the pack stamps 4.0e-6), one
top-1 change at an exact float16 tie that Prism itself separates by 0.0024
logits, and identical greedy continuations on three prompts. On by default;
``MTPLX_BONSAI_TERNARY_QMV=0`` restores stock ``mx.quantized_matmul``.

The decode trick and the row layout follow mlx-serve's ``qmv2.zig``
(ddalcu/mlx-serve, Apache-2.0), re-implemented here as an
``mx.fast.metal_kernel``.

Contract (``ternary_qmv`` returns None on any miss, the caller keeps stock):
  * ``x``: float16, ``[..., K]`` with at most ``MAX_ROWS`` rows in total;
  * ``w``: uint32 ``[N, K/16]``; ``scales``: float16 ``[N, K/128]``;
  * ``K % 512 == 0`` and ``N`` a multiple of the rows one threadgroup covers;
  * the caller has proven ``biases == -scales`` for this matrix once.

Geometry (2026-09-23 sweep on the pack's own matrices, M5 Max, 16-call
bursts): two rows per simdgroup and eight simdgroups per threadgroup beat
stock by 1.15-1.25x on the MLP and linear-attention projections at M = 1
and 2 (four rows per simdgroup spills registers and runs at half of
stock); the vocabulary head prefers four simdgroups (1.16x at M = 1, 1.39x
at M = 2). Matrices with fewer than ``MIN_N`` outputs (the 1024-row key and
value projections) have too few threadgroups to fill the GPU and stay on
stock; the model arms the kernel only for ``worthwhile`` shapes.

Every GPU generation: the kernel is plain SIMD (half and float arithmetic,
``simd_sum``, no threadgroup memory, 128 or 256 threads, 32-lane simdgroups)
with no tensor units, so it runs on M1 to M5; the geometry above was tuned on
an M5 Max only. The load-time self-check (``mtplx.kernel_selfcheck``, lane
``bonsai_ternary_qmv``) runs it against stock on this GPU and turns it off for
the process on a mismatch or a build or launch failure.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..kernel_selfcheck import lane_disabled

ENV = "MTPLX_BONSAI_TERNARY_QMV"
LANE = "bonsai_ternary_qmv"
MAX_ROWS = 4
GROUP_SIZE = 128
BITS = 2
ROWS_PER_SIMDGROUP = 2
SIMDGROUPS = 8
HEAD_SIMDGROUPS = 4
HEAD_MIN_N = 65536
MIN_N = 2048

_COUNTS = {"served": 0, "declined": 0}

_HEADER = """
inline void ternary_h2dec(uint u, thread half2* q) {
  // Codes j (low half) and j + 8 (high half) of one packed word as exact
  // half integers minus one: a weight is scale * (code - 1).
  uint u6 = u >> 6;
  q[0] = as_type<half2>((u  & 0x00030003u) | 0x64006400u) - half2(1025.0h);
  q[1] = as_type<half2>((u  & 0x000C000Cu) | 0x5C005C00u) - half2(257.0h);
  q[2] = as_type<half2>((u  & 0x00300030u) | 0x54005400u) - half2(65.0h);
  q[3] = as_type<half2>((u  & 0x00C000C0u) | 0x4C004C00u) - half2(17.0h);
  q[4] = as_type<half2>((u  & 0x03000300u) | 0x44004400u) - half2(5.0h);
  q[5] = as_type<half2>((u6 & 0x00300030u) | 0x54005400u) - half2(65.0h);
  q[6] = as_type<half2>((u6 & 0x00C000C0u) | 0x4C004C00u) - half2(17.0h);
  q[7] = as_type<half2>((u6 & 0x03000300u) | 0x44004400u) - half2(5.0h);
}
"""

_SOURCE = """
  constexpr int R = ROWS;
  constexpr int SG = SGROUPS;
  const int K = x_shape[x_ndim - 1];
  const int N = w_shape[0];
  const int KW = K / 16;
  const int KG = K / 128;
  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const int row0 = ((int)threadgroup_position_in_grid.y * SG + (int)sgi) * R;
  const device uint* wq = w + (size_t)row0 * KW + lane;
  const device half* sp = scales + (size_t)row0 * KG + lane / 8;
  const device half* xp = x + lane * 16;
  float acc[R][M];
  for (int r = 0; r < R; ++r) {
    for (int m = 0; m < M; ++m) {
      acc[r][m] = 0.0f;
    }
  }
  for (int k = 0; k < K; k += 512) {
    uint wv[R];
    float sc[R];
    for (int r = 0; r < R; ++r) {
      wv[r] = wq[(size_t)r * KW];
      sc[r] = float(sp[(size_t)r * KG]);
    }
    // Decode and widen each code once per block; every activation row
    // reuses it (one float2 FMA per code pair per row after this).
    float2 qf[R][8];
    for (int r = 0; r < R; ++r) {
      half2 q[8];
      ternary_h2dec(wv[r], q);
      for (int j = 0; j < 8; ++j) {
        qf[r][j] = float2(q[j]);
      }
    }
    for (int m = 0; m < M; ++m) {
      const device half* xr = xp + (size_t)m * K + k;
      float2 xf[8];
      for (int j = 0; j < 8; ++j) {
        xf[j] = float2(float(xr[j]), float(xr[j + 8]));
      }
      for (int r = 0; r < R; ++r) {
        float2 a = qf[r][0] * xf[0];
        for (int j = 1; j < 8; ++j) {
          a = fma(qf[r][j], xf[j], a);
        }
        acc[r][m] += sc[r] * (a.x + a.y);
      }
    }
    wq += 32;
    sp += 4;
  }
  for (int r = 0; r < R; ++r) {
    for (int m = 0; m < M; ++m) {
      float v = simd_sum(acc[r][m]);
      if (lane == 0) {
        y[(size_t)m * N + row0 + r] = static_cast<half>(v);
      }
    }
  }
"""


def switched_on() -> bool:
    """The user's switch alone (default on)."""

    return (os.environ.get(ENV, "1").strip().lower()) not in {"0", "false", "no", "off"}


def enabled() -> bool:
    """Switched on, and not turned off by the load-time self-check on this GPU."""

    return switched_on() and not lane_disabled(LANE)


def counters() -> dict[str, int]:
    return dict(_COUNTS)


def _geometry(n: int) -> tuple[int, int]:
    """(rows per simdgroup, simdgroups per threadgroup) for ``n`` outputs; env override for tuning."""

    raw = (os.environ.get("MTPLX_TERNARY_QMV_GEOMETRY") or "").strip()
    if raw:
        try:
            r, sg = (int(v) for v in raw.lower().split("x"))
            if r in (1, 2, 4, 8) and sg in (1, 2, 4, 8):
                return r, sg
        except ValueError:
            pass
    if n >= HEAD_MIN_N:
        return ROWS_PER_SIMDGROUP, HEAD_SIMDGROUPS
    return ROWS_PER_SIMDGROUP, SIMDGROUPS


def worthwhile(n: int) -> bool:
    """True when a matrix with ``n`` outputs is large enough to beat stock."""

    return int(n) >= MIN_N


@lru_cache(maxsize=None)
def _kernel(rows: int, r: int, sg: int):
    source = _SOURCE.replace("ROWS", str(r)).replace("SGROUPS", str(sg))
    return mx.fast.metal_kernel(
        name=f"mtplx_ternary_qmv_m{rows}_r{r}_sg{sg}",
        input_names=["x", "w", "scales"],
        output_names=["y"],
        source=source,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


def ternary_layout(scales: mx.array, biases: mx.array) -> bool:
    """True when every bias is exactly ``-scale`` (Prism's ternary layout)."""

    if scales.dtype != mx.float16 or biases.dtype != mx.float16:
        return False
    if tuple(scales.shape) != tuple(biases.shape):
        return False
    return bool(mx.array_equal(biases, -scales).item())


def eligible(x: mx.array, w: mx.array, scales: mx.array) -> bool:
    if x.dtype != mx.float16 or scales.dtype != mx.float16 or w.dtype != mx.uint32:
        return False
    if x.ndim < 1 or w.ndim != 2:
        return False
    k = int(x.shape[-1])
    n = int(w.shape[0])
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    if rows < 1 or rows > MAX_ROWS:
        return False
    if k % 512 or int(w.shape[1]) * 16 != k:
        return False
    r, sg = _geometry(n)
    if n % (r * sg):
        return False
    if tuple(scales.shape) != (n, k // GROUP_SIZE):
        return False
    return True


class Plan:
    """Launch parameters of one proven-ternary matrix, validated once when it is armed.

    A verify round runs the kernel on about 370 matrices; the model calls
    ``run`` with the plan so each call checks only the activation (dtype, row
    count, width) instead of re-deriving the whole contract: about 2 us less
    host time per launch (lazy graph build, measured 5.8 -> 3.7 us on a busy
    machine), about 0.8 ms per round of GPU-idle host time. The geometry
    (and its ``MTPLX_TERNARY_QMV_GEOMETRY`` override) is fixed when the plan
    is made.
    """

    __slots__ = ("n", "k", "r", "sg", "grid", "threadgroup", "calls")

    def __init__(self, n: int, k: int, r: int, sg: int) -> None:
        self.n = n
        self.k = k
        self.r = r
        self.sg = sg
        self.grid = (32 * sg, n // (r * sg), 1)
        self.threadgroup = (32 * sg, 1, 1)
        # rows -> (kernel, template, output shapes), filled on first use.
        self.calls: list = [None] * (MAX_ROWS + 1)


def plan(w: mx.array, scales: mx.array) -> Plan | None:
    """Launch plan for a ternary matrix, or None when its shape is out of contract."""

    if w.dtype != mx.uint32 or scales.dtype != mx.float16 or w.ndim != 2:
        return None
    n = int(w.shape[0])
    k = int(w.shape[1]) * 16
    if k % 512:
        return None
    r, sg = _geometry(n)
    if n % (r * sg):
        return None
    if tuple(scales.shape) != (n, k // GROUP_SIZE):
        return None
    return Plan(n, k, r, sg)


def run(p: Plan, x: mx.array, w: mx.array, scales: mx.array) -> mx.array | None:
    """``ternary_qmv`` for a planned matrix: same kernel, same bits, fewer host checks."""

    if not enabled() or x.dtype != mx.float16:
        _COUNTS["declined"] += 1
        return None
    shape = x.shape
    if not shape or shape[-1] != p.k:
        _COUNTS["declined"] += 1
        return None
    rows = x.size // p.k
    if rows < 1 or rows > MAX_ROWS:
        _COUNTS["declined"] += 1
        return None
    call = p.calls[rows]
    if call is None:
        call = p.calls[rows] = (_kernel(rows, p.r, p.sg), [("M", rows)], [(rows, p.n)])
    kernel, template, out_shapes = call
    out = kernel(
        inputs=[x.reshape(rows, p.k), w, scales],
        template=template,
        grid=p.grid,
        threadgroup=p.threadgroup,
        output_shapes=out_shapes,
        output_dtypes=[mx.float16],
    )[0]
    _COUNTS["served"] += 1
    return out.reshape(*shape[:-1], p.n)


def ternary_qmv(x: mx.array, w: mx.array, scales: mx.array) -> mx.array | None:
    """``x @ dequant(w).T`` for a ternary 2-bit/g128 matrix, or None."""

    if not enabled() or not eligible(x, w, scales):
        _COUNTS["declined"] += 1
        return None
    k = int(x.shape[-1])
    n = int(w.shape[0])
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    x2 = x.reshape(rows, k)
    r, sg = _geometry(n)
    out = _launch(x2, w, scales, rows, r, sg)
    _COUNTS["served"] += 1
    return out.reshape(*x.shape[:-1], n)


def _launch(x2: mx.array, w: mx.array, scales: mx.array, rows: int, r: int, sg: int) -> mx.array:
    """The kernel on ``[rows, K]`` activations the caller has checked (the self-check probes this)."""

    n = int(w.shape[0])
    return _kernel(rows, r, sg)(
        inputs=[x2, w, scales],
        template=[("M", rows)],
        grid=(32 * sg, n // (r * sg), 1),
        threadgroup=(32 * sg, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float16],
    )[0]
