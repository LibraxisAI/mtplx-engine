"""One-dispatch blockwise Walsh-Hadamard rotation, bit-identical to the MLX chain.

The Prism loader rotates every activation that enters a packed projection with
``T(x) = H(s * x)`` (and the embedding rows with ``s * H(y)``), computed in
float32 and returned in the input dtype. With stock MLX ops that is four
dispatches per rotation (cast, sign multiply, ``mx.hadamard_transform``, cast)
and three float32 round trips through memory; Ternary Bonsai 2 runs 256 of
them per verify round.

This kernel does the whole rotation in one dispatch, one threadgroup of 128
threads per 1024-wide block, and returns the same bits as the MLX chain:

* the sign multiply is exact (a multiply by +/-1);
* the butterflies run in the order ``mx.hadamard_transform`` runs them for a
  1024 block (radix-16, radix-16, radix-4: element distance 1, 2, 4, ..., 512
  in increasing order), every add and subtract in float32, so every
  intermediate value is the same float32 number;
* the 1/32 scale is a power of two (exact) and the float16 cast rounds to
  nearest even, like ``astype``.

Element ``e`` of a block lives in register ``p`` of lane ``l`` of simdgroup
``g`` with ``e = (32 g + l) * 8 + p``: distances 1-4 are register
butterflies, 8-128 are ``simd_shuffle_xor`` butterflies and 256-512 go
through one threadgroup exchange. The exactness claim is tested against the
MLX chain (``tests/test_prism_bonsai_kernels.py``) and on the real pack: every
one of 1,630 teacher-forced verify positions (code, prose, reasoning) returns
bit-identical logits. On by default; ``MTPLX_PRISM_FUSED_ROTATION=0`` restores
the four-op MLX chain.

Every GPU generation: the kernel is plain SIMD (float and half arithmetic,
``simd_shuffle_xor``, a 4 KB threadgroup array, 128 threads) with no tensor
units, so it runs on M1 to M5. Only the M5 has measured it; the load-time
self-check (``mtplx.kernel_selfcheck``, lane ``prism_fused_rotation``) compares
its bits with the MLX chain on this GPU and turns it off for the process on
any difference or build failure.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..kernel_selfcheck import lane_disabled

ENV = "MTPLX_PRISM_FUSED_ROTATION"
LANE = "prism_fused_rotation"
BLOCK = 1024

_SOURCE = """
  constexpr int B = 1024;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const int K = x_shape[x_ndim - 1];
  const size_t base = (size_t)threadgroup_position_in_grid.x * B;
  const int col0 = (int)(base % (size_t)K);
  const int e0 = (int)tid * 8;
  float v[8];
  for (int p = 0; p < 8; ++p) {
    float e = float(x[base + e0 + p]);
    v[p] = INV ? e : e * signs[col0 + e0 + p];
  }
  for (int h = 1; h < 8; h <<= 1) {
    for (int p = 0; p < 8; ++p) {
      if ((p & h) == 0) {
        float a = v[p];
        float b = v[p + h];
        v[p] = a + b;
        v[p + h] = a - b;
      }
    }
  }
  for (uint m = 1; m < 32; m <<= 1) {
    const bool upper = (lane & m) != 0;
    for (int p = 0; p < 8; ++p) {
      float o = simd_shuffle_xor(v[p], (ushort)m);
      v[p] = upper ? (o - v[p]) : (v[p] + o);
    }
  }
  threadgroup float tg[B];
  for (int p = 0; p < 8; ++p) {
    tg[e0 + p] = v[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = 0.03125f;
  for (int g = (int)tid; g < 256; g += 128) {
    float w0 = tg[g];
    float w1 = tg[g + 256];
    float w2 = tg[g + 512];
    float w3 = tg[g + 768];
    float a0 = w0 + w1;
    float a1 = w0 - w1;
    float a2 = w2 + w3;
    float a3 = w2 - w3;
    float r[4];
    r[0] = a0 + a2;
    r[2] = a0 - a2;
    r[1] = a1 + a3;
    r[3] = a1 - a3;
    for (int q = 0; q < 4; ++q) {
      const int e = g + 256 * q;
      float o = r[q] * scale;
      if (INV) {
        o = o * signs[col0 + e];
      }
      y[base + e] = static_cast<T>(o);
    }
  }
"""

_COUNTS = {"served": 0, "declined": 0}


def switched_on() -> bool:
    """The user's switch alone (default on)."""

    return (os.environ.get(ENV, "1").strip().lower()) not in {"0", "false", "no", "off"}


def enabled() -> bool:
    """Switched on, and not turned off by the load-time self-check on this GPU."""

    return switched_on() and not lane_disabled(LANE)


def counters() -> dict[str, int]:
    return dict(_COUNTS)


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_prism_hadamard_rotate_1024",
        input_names=["x", "signs"],
        output_names=["y"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def rotate(x: mx.array, signs: mx.array, block: int, *, inverse: bool = False) -> mx.array | None:
    """``H(s * x)`` (or ``s * H(x)`` when ``inverse``), or None when out of contract."""

    if not enabled():
        _COUNTS["declined"] += 1
        return None
    if (
        block != BLOCK
        or x.dtype != mx.float16
        or signs.dtype != mx.float32
        or x.ndim < 1
        or int(x.shape[-1]) % BLOCK
        or tuple(signs.shape) != (int(x.shape[-1]),)
        or x.size == 0
    ):
        _COUNTS["declined"] += 1
        return None
    out = _launch(x, signs, inverse)
    _COUNTS["served"] += 1
    return out


def _launch(x: mx.array, signs: mx.array, inverse: bool) -> mx.array:
    """The kernel itself, for an input the caller has checked (the self-check probes this)."""

    blocks = x.size // BLOCK
    return _kernel()(
        inputs=[x, signs],
        template=[("T", x.dtype), ("INV", bool(inverse))],
        grid=(128 * blocks, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
