"""Flash-Next dense-band attention without the separate ``where`` pass.

Below the block-sparse crossover a wide QSA prefill forward attends through a
boolean mask (the top-512 blocks, the visible tail and the causal edge).  At
head dim 256 MLX has no fused kernel for that call, so
``mx.fast.scaled_dot_product_attention`` runs its fallback:

    scores = (bf16(scale) * q) @ k^T              # [B, Hkv, rep, S, T] bf16
    scores = where(mask, scores, finfo(bf16).min)  # a full pass over the plane
    probs  = softmax(scores, precise=True)
    out    = probs @ v

At 4,096 rows and 16K of history the score plane is 3.2 GB, so the ``where``
reads and writes 6.4 GB per layer.  Here the mask rides in the score matmul
instead: ``addmm`` adds a ``[S, T]`` bf16 plane of 0 (visible) and
``finfo(bf16).min`` (masked) in the GEMM epilogue.  A visible score is
``bf16(acc + 0) == bf16(acc)``; a masked one is ``bf16(acc + min) == min``,
the value ``where`` writes (attention scores are far below the rounding step
at ``min``).  The softmax and the second matmul are the fallback's own, and
MLX still donates the score buffer to the softmax, so the output is
bit-identical to the stock call at the same peak memory plus the ``[S, T]``
plane (tests/test_qsa_dense_band_sdpa.py).
"""

from __future__ import annotations

import mlx.core as mx

#: The fallback's masked value: ``finfo(bfloat16).min`` (a fully masked row
#: attends uniformly, it does not produce NaN).
_MASKED = -3.3895313892515355e38


def dense_band_eligible(q: mx.array, k: mx.array, mask) -> bool:
    """The calls whose stock SDPA is MLX's unfused fallback, and nothing else.

    Head dim 256 (no fused MLX kernel), more than 8 query rows (the vector
    kernel's band), bf16, GQA with equal q/v head dims, and a boolean
    ``[1, 1, S, T]`` mask.  Metadata only.
    """

    if not isinstance(mask, mx.array) or mask.dtype != mx.bool_:
        return False
    if q.ndim != 4 or k.ndim != 4 or q.dtype != mx.bfloat16 or k.dtype != mx.bfloat16:
        return False
    B, Hq, S, D = (int(d) for d in q.shape)
    Hkv, T = int(k.shape[1]), int(k.shape[2])
    if D != 256 or int(k.shape[3]) != D or S <= 8 or B != 1:
        return False
    if Hkv <= 0 or Hq % Hkv != 0:
        return False
    if tuple(int(d) for d in mask.shape) != (1, 1, S, T):
        return False
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def dense_band_sdpa(q, k, v, *, scale: float, mask: mx.array) -> mx.array:
    """MLX's SDPA fallback with the boolean mask applied inside the score GEMM.

    Callers check :func:`dense_band_eligible` first.
    """

    B, Hq, S, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    rep = Hq // Hkv
    q = mx.array(scale, dtype=q.dtype) * q
    if rep > 1:
        q = q.reshape(B, Hkv, rep, S, D)
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)
    additive = mx.where(
        mask.reshape(S, T),
        mx.array(0.0, dtype=q.dtype),
        mx.array(_MASKED, dtype=q.dtype),
    )
    scores = mx.addmm(additive, q, mx.swapaxes(k, -1, -2))
    probs = mx.softmax(scores, axis=-1, precise=True)
    out = mx.matmul(probs, v)
    if rep > 1:
        out = out.reshape(B, Hq, S, -1)
    return out
