"""The simd top-k selector equals the network selector element for element.

Synthetic float32 score planes, no model. The simd selector is gated on the
tensor-unit route, so the numeric half skips everywhere else; the routing
rules are host-only.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.kernels.qsa_indexer_prefill as backend
from mtplx.kernels.qsa_indexer_select import qsa_indexer_select_nax_available

RATIO, TOPK = 4, 512
needs_tensor_units = pytest.mark.skipif(
    not mx.metal.is_available()
    or mx.default_device() != mx.gpu
    or not qsa_indexer_select_nax_available(),
    reason="the simd selector is gated on the tensor-unit route",
)


def _scores(rows: int, blocks: int, kind: str, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    plane = mx.random.normal((rows, blocks))
    if kind == "relu":  # what the scorer emits: about half exact zeros
        plane = mx.maximum(plane, 0)
    elif kind == "ties":  # the block-id tiebreak decides most of the row
        plane = mx.round(plane * 2) / 2
    elif kind == "constant":
        plane = mx.zeros((rows, blocks))
    elif kind == "specials":
        for value in (float("inf"), -float("inf"), -0.0):
            mask = mx.random.uniform(shape=(rows, blocks)) < 0.01
            plane = mx.where(mask, mx.array(value), plane)
    return plane.astype(mx.float32)


def _both(monkeypatch, scores, total, **kw):
    rows = int(scores.shape[0])
    args = dict(
        pos_start=total - rows, total_tokens=total, block_topk=TOPK,
        compress_ratio=RATIO, mode="blocks", **kw,
    )
    monkeypatch.setenv("MTPLX_QSA_PREFILL_TOPK_KERNEL", "network")
    network = backend.qsa_indexer_prefill_topk_metal(scores, **args)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_TOPK_KERNEL", "simd")
    simd = backend.qsa_indexer_prefill_topk_metal(scores, **args)
    mx.eval(network, simd)
    return network, simd


@needs_tensor_units
@pytest.mark.parametrize("kind", ["normal", "relu", "ties", "constant", "specials"])
def test_simd_selector_equals_the_network_selector(monkeypatch, kind):
    total = 12_001
    network, simd = _both(monkeypatch, _scores(97, total // RATIO, kind), total)
    for a, b in zip(network, simd):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(np.array(a), np.array(b)), kind
    ids, valid = np.array(simd[0]), np.array(simd[1])
    # Born ascending, one winner per slot.
    assert valid.all()
    assert (np.diff(ids, axis=1) > 0).all()


@needs_tensor_units
@pytest.mark.parametrize("rows,total", [(7, 2060), (64, 2049), (300, 3000), (33, 40_001)])
def test_rows_at_and_below_the_sparse_boundary(monkeypatch, rows, total):
    # Rows that see fewer than 512 complete blocks select all of them and pad.
    network, simd = _both(monkeypatch, _scores(rows, total // RATIO, "relu", seed=rows), total)
    for a, b in zip(network, simd):
        assert np.array_equal(np.array(a), np.array(b))


@needs_tensor_units
def test_logical_blocks_smaller_than_the_backing_plane(monkeypatch):
    total, backing = 8_000, 4_096
    scores = _scores(40, backing, "normal", seed=9)
    network, simd = _both(monkeypatch, scores, total, logical_blocks=total // RATIO)
    for a, b in zip(network, simd):
        assert np.array_equal(np.array(a), np.array(b))
    assert int(np.array(simd[0]).max()) < total // RATIO


def test_routing_rules(monkeypatch):
    monkeypatch.setattr(backend, "qsa_indexer_select_nax_available", lambda: True)
    monkeypatch.delenv("MTPLX_QSA_PREFILL_TOPK_KERNEL", raising=False)
    limit = backend._SIMD_TOPK_MAX_BLOCKS
    assert backend._simd_topk_applies("blocks", 4_096)
    assert backend._simd_topk_applies("blocks", limit)
    # Long rows keep the network selector: the simd one reads the row ten times.
    assert not backend._simd_topk_applies("blocks", limit + 1)
    # The other two output contracts keep the network selector.
    assert not backend._simd_topk_applies("row_tokens", 4_096)
    assert not backend._simd_topk_applies("dense_mask", 4_096)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_TOPK_KERNEL", "network")
    assert not backend._simd_topk_applies("blocks", 4_096)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_TOPK_KERNEL", "simd")
    assert backend._simd_topk_applies("blocks", limit + 1)
    assert not backend._simd_topk_applies("row_tokens", 4_096)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_TOPK_KERNEL", "junk")
    assert backend._prefill_topk_variant() == "auto"
    # No tensor-unit route (an M1 to M4, or the rehearsal switch): network.
    monkeypatch.setattr(backend, "qsa_indexer_select_nax_available", lambda: False)
    monkeypatch.setenv("MTPLX_QSA_PREFILL_TOPK_KERNEL", "simd")
    assert not backend._simd_topk_applies("blocks", 4_096)


def test_simd_source_has_no_threadgroup_memory_no_barrier_no_atomic():
    source = backend._SIMD_TOPK_SOURCE
    for forbidden in ("threadgroup ", "threadgroup_barrier", "atomic_", "exchange_"):
        assert forbidden not in source, forbidden
    assert "simd_sum(" in source and "simd_prefix_exclusive_sum(" in source
    # Same adjusted score and the same order key as the network selector.
    assert "scores[score_base + block] - float(block) * 1.0e-12f" in source
    assert "qsa_float_order_key(adjusted)" in source
    assert "const uint valid_count = metal::min(logical, complete);" in source
    # Ties: the larger block id ranks higher, as in the composite key.
    assert "first_winning_tie = ties - need;" in source
    assert "wins = tie_index >= first_winning_tie;" in source
