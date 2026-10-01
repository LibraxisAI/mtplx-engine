"""The prefill top-k selectors read the block count at run time (2026-09-20).

The score plane's width is context // 4, so it is different for every prefill
chunk of every prompt.  Baked into the kernel source it compiled a fresh Metal
pipeline per chunk: 40 ms (simd selector) or 56 ms (network selector) on an M5
Max, with the GPU idle behind the encoder.  The kernels use the width as a row
stride and a clamp only, so it now comes from the score plane's shape.

Pinned here: one kernel serves every block count, its outputs equal the
compile-time variant's element for element on both selectors (normal, ReLU,
heavy-tie, constant and inf / -inf / -0.0 planes, a logical count below the
backing width, rows that see fewer than K blocks), and the rollback switch
restores the old specialization.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mtplx.kernels.qsa_indexer_prefill as backend
from mtplx.kernels.qsa_indexer_select import qsa_indexer_select_nax_available

RATIO, TOPK = 4, 512
STATIC = "MTPLX_QSA_PREFILL_TOPK_STATIC_BLOCKS"
VARIANT = "MTPLX_QSA_PREFILL_TOPK_KERNEL"

needs_metal = pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="the selectors are Metal kernels",
)
needs_tensor_units = pytest.mark.skipif(
    not mx.metal.is_available()
    or mx.default_device() != mx.gpu
    or not qsa_indexer_select_nax_available(),
    reason="the simd selector is gated on the tensor-unit route",
)


def _scores(rows: int, blocks: int, kind: str, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    plane = mx.random.normal((rows, blocks))
    if kind == "relu":
        plane = mx.maximum(plane, 0)
    elif kind == "ties":
        plane = mx.round(plane * 2) / 2
    elif kind == "constant":
        plane = mx.zeros((rows, blocks))
    elif kind == "specials":
        for value in (float("inf"), -float("inf"), -0.0):
            mask = mx.random.uniform(shape=(rows, blocks)) < 0.01
            plane = mx.where(mask, mx.array(value), plane)
    return plane.astype(mx.float32)


def _select(monkeypatch, scores, total, *, variant, static, mode="blocks", **kw):
    rows = int(scores.shape[0])
    monkeypatch.setenv(VARIANT, variant)
    if static:
        monkeypatch.setenv(STATIC, "1")
    else:
        monkeypatch.delenv(STATIC, raising=False)
    out = backend.qsa_indexer_prefill_topk_metal(
        scores,
        pos_start=total - rows,
        total_tokens=total,
        block_topk=TOPK,
        compress_ratio=RATIO,
        mode=mode,
        **kw,
    )
    out = out if isinstance(out, tuple) else (out,)
    mx.eval(*out)
    return [np.array(leaf) for leaf in out]


def _assert_equal(left, right, note=""):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b, equal_nan=True), note


@needs_metal
@pytest.mark.parametrize("kind", ["normal", "relu", "ties", "constant", "specials"])
def test_network_selector_runtime_blocks_equal_the_compile_time_variant(monkeypatch, kind):
    total = 12_001
    scores = _scores(97, total // RATIO, kind)
    static = _select(monkeypatch, scores, total, variant="network", static=True)
    runtime = _select(monkeypatch, scores, total, variant="network", static=False)
    _assert_equal(static, runtime, kind)


@needs_tensor_units
@pytest.mark.parametrize("kind", ["normal", "relu", "ties", "constant", "specials"])
def test_simd_selector_runtime_blocks_equal_the_compile_time_variant(monkeypatch, kind):
    total = 12_001
    scores = _scores(97, total // RATIO, kind)
    static = _select(monkeypatch, scores, total, variant="simd", static=True)
    runtime = _select(monkeypatch, scores, total, variant="simd", static=False)
    _assert_equal(static, runtime, kind)


@needs_metal
@pytest.mark.parametrize("variant", ["network", "simd"])
@pytest.mark.parametrize("rows,total", [(7, 2060), (64, 2049), (300, 3000), (33, 40_001)])
def test_every_block_count_gives_the_compile_time_answer(monkeypatch, variant, rows, total):
    if variant == "simd" and not qsa_indexer_select_nax_available():
        pytest.skip("the simd selector is gated on the tensor-unit route")
    scores = _scores(rows, total // RATIO, "relu", seed=rows)
    static = _select(monkeypatch, scores, total, variant=variant, static=True)
    runtime = _select(monkeypatch, scores, total, variant=variant, static=False)
    _assert_equal(static, runtime, f"{variant} {rows}x{total}")


@needs_metal
@pytest.mark.parametrize("variant", ["network", "simd"])
def test_a_logical_count_below_the_backing_width(monkeypatch, variant):
    if variant == "simd" and not qsa_indexer_select_nax_available():
        pytest.skip("the simd selector is gated on the tensor-unit route")
    total, backing = 8_000, 4_096
    scores = _scores(40, backing, "normal", seed=9)
    kw = {"logical_blocks": total // RATIO}
    static = _select(monkeypatch, scores, total, variant=variant, static=True, **kw)
    runtime = _select(monkeypatch, scores, total, variant=variant, static=False, **kw)
    _assert_equal(static, runtime, variant)
    assert int(runtime[0].max()) < total // RATIO


@needs_metal
def test_row_tokens_mode_runtime_blocks_equal_the_compile_time_variant(monkeypatch):
    total = 9_001
    scores = _scores(21, total // RATIO, "relu", seed=4)
    static = _select(monkeypatch, scores, total, variant="network", static=True, mode="row_tokens")
    runtime = _select(monkeypatch, scores, total, variant="network", static=False, mode="row_tokens")
    _assert_equal(static, runtime, "row_tokens")


def test_one_kernel_serves_every_block_count(monkeypatch):
    monkeypatch.delenv(STATIC, raising=False)
    assert backend._topk_static_blocks() is False
    simd = backend._prefill_topk_simd_kernel(TOPK, RATIO)
    assert simd is backend._prefill_topk_simd_kernel(TOPK, RATIO)
    network = backend._prefill_topk_kernel("blocks", TOPK, RATIO, 512, 0)
    assert network is backend._prefill_topk_kernel("blocks", TOPK, RATIO, 512, 0)
    # The compile-time variant is still one kernel per count: that is the
    # rollback, and the reference the numeric tests compare against.
    assert backend._prefill_topk_simd_kernel(TOPK, RATIO, 4_100) is not simd
    assert backend._prefill_topk_simd_kernel(
        TOPK, RATIO, 4_100
    ) is not backend._prefill_topk_simd_kernel(TOPK, RATIO, 4_101)


def test_the_rollback_switch_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv(STATIC, "1")
    assert backend._topk_static_blocks() is True
    monkeypatch.setenv(STATIC, "0")
    assert backend._topk_static_blocks() is False
    monkeypatch.delenv(STATIC, raising=False)
    assert backend._topk_static_blocks() is False


def test_the_block_count_is_used_as_a_stride_and_a_clamp_only():
    """Why reading it at run time cannot change an output: no loop bound, no
    threadgroup array and no dispatch size depends on it."""

    for source in (backend._SIMD_TOPK_SOURCE,):
        uses = [line.strip() for line in source.splitlines() if "BACKING_BLOCKS" in line]
        assert uses == [
            "? metal::min(uint(logical_value), BACKING_BLOCKS)",
            "const size_t score_base = (size_t)row * BACKING_BLOCKS;",
        ]
    assert "scores_shape[1]" in backend._RUNTIME_BLOCKS_PREAMBLE
    assert backend._blocks_constant(0).startswith("//")
    assert backend._blocks_constant(4_100) == "constant constexpr uint BACKING_BLOCKS = 4100;"
