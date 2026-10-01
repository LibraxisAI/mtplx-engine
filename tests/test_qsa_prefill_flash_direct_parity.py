"""The direct source of the block-sparse QSA prefill consumer equals the
staged source bit for bit, and both equal an fp32 reference within bf16 noise.

Small synthetic tensors at the production geometry; no model. Needs the
tensor-unit route (GPU generation 17), so it skips everywhere else.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.kernels.qsa_prefill_flash as flash
from mtplx.kernels.qsa_indexer_select import qsa_indexer_select_nax_available

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available()
    or mx.default_device() != mx.gpu
    or not qsa_indexer_select_nax_available(),
    reason="the block-sparse prefill consumer needs the tensor-unit route",
)

BLOCK, TOP_K, SCALE = 4, 512, 0.0625


def _fixture(rows, total, *, dtype, holes=0.0, future=0.0, views=False, seed=0):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    pos_start = total - rows
    q = (mx.random.normal((1, 24, rows, 256)) * 0.5).astype(dtype)
    k = (mx.random.normal((1, 2, total + 5, 256)) * 0.5).astype(dtype)
    v = mx.random.normal((1, 2, total + 5, 256)).astype(dtype)
    ids = np.empty((rows, TOP_K), dtype=np.int32)
    for r in range(rows):
        complete = (pos_start + r + 1) // BLOCK
        ids[r] = np.sort(rng.choice(complete, size=TOP_K, replace=False))
    valid = rng.random((rows, TOP_K)) >= holes
    if future:
        ids = np.where(rng.random((rows, TOP_K)) < future, ids + 1_000_000, ids)
    block_ids = mx.array(ids.astype(np.int32))
    if views:
        q = mx.transpose(mx.transpose(q, (0, 2, 1, 3)) + 0, (0, 2, 1, 3))
        wide = (mx.random.normal((1, 2, total + 5, 512)) * 0.5).astype(dtype)
        k, v = wide[..., ::2], (wide * 2)[..., 1::2]
        block_ids = mx.concatenate([block_ids, block_ids], axis=1)[:, :TOP_K]
    return q, k, v, block_ids, mx.array(valid), pos_start


def _run(monkeypatch, variant, q, k, v, ids, valid, pos_start, total):
    monkeypatch.setenv("MTPLX_QSA_PREFILL_FLASH_KERNEL", variant)
    out = flash.qsa_prefill_flash(
        q, k, v, ids, valid, pos_start=pos_start, total_tokens=total, scale=SCALE
    )
    mx.eval(out)
    return out


def _reference(q, k, v, ids, valid, pos_start, row):
    qn = np.array(q[0].astype(mx.float32))
    kn = np.array(k[0].astype(mx.float32))
    vn = np.array(v[0].astype(mx.float32))
    ids_n, valid_n = np.array(ids), np.array(valid)
    pos = pos_start + row
    complete = (pos + 1) // BLOCK
    blocks = ids_n[row][valid_n[row] & (ids_n[row] >= 0) & (ids_n[row] < complete)]
    tokens = (blocks[:, None] * BLOCK + np.arange(BLOCK)[None, :]).reshape(-1)
    tokens = np.concatenate([tokens, np.arange(complete * BLOCK, pos + 1)])
    out = np.empty((24, 256), dtype=np.float32)
    for head in range(24):
        kv = head // 12
        scores = (kn[kv, tokens] @ qn[head, row]) * SCALE
        p = np.exp(scores - scores.max())
        out[head] = (p / p.sum()) @ vn[kv, tokens]
    return out


@pytest.mark.parametrize(
    "name,rows,total,kwargs",
    [
        ("just-past-the-boundary", 9, 2061, {}),
        ("every-tail-length", 37, 4099, {}),
        ("float16", 17, 3000, {"dtype": mx.float16}),
        ("validity-holes", 21, 5000, {"holes": 0.4}),
        ("nearly-empty-selection", 11, 5000, {"holes": 0.995}),
        ("ids-past-the-visible-blocks", 13, 5000, {"future": 0.2}),
        ("strided-views", 19, 3500, {"views": True, "holes": 0.2}),
    ],
)
def test_direct_equals_staged_bit_for_bit(monkeypatch, name, rows, total, kwargs):
    dtype = kwargs.pop("dtype", mx.bfloat16)
    q, k, v, ids, valid, pos_start = _fixture(rows, total, dtype=dtype, **kwargs)
    staged = _run(monkeypatch, "staged", q, k, v, ids, valid, pos_start, total)
    direct = _run(monkeypatch, "direct", q, k, v, ids, valid, pos_start, total)
    assert direct.dtype == staged.dtype == dtype
    assert np.array_equal(
        np.array(direct.view(mx.uint16)), np.array(staged.view(mx.uint16))
    ), name
    got = np.array(direct[0].astype(mx.float32))
    assert np.isfinite(got).all()
    for row in (0, rows // 2, rows - 1):
        ref = _reference(q, k, v, ids, valid, pos_start, row)
        assert np.abs(got[:, row, :] - ref).max() <= 2e-3 + 2e-2 * np.abs(ref).max(), (name, row)


def test_an_unknown_switch_value_serves_the_direct_source(monkeypatch):
    monkeypatch.setenv("MTPLX_QSA_PREFILL_FLASH_KERNEL", "something-else")
    assert flash._kernel_variant() == "direct"
    monkeypatch.setenv("MTPLX_QSA_PREFILL_FLASH_KERNEL", " Staged ")
    assert flash._kernel_variant() == "staged"
    monkeypatch.delenv("MTPLX_QSA_PREFILL_FLASH_KERNEL")
    assert flash._kernel_variant() == "direct"
