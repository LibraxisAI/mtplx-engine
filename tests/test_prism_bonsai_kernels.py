"""The two Bonsai decode kernels against the stock MLX computations they replace.

* The fused Hadamard rotation must return the same bits as the four-op MLX
  chain in ``prism_hadamard_qwen35.hadamard_rotate`` (forward and inverse).
* The ternary GEMV must compute the same dot product as stock
  ``mx.quantized_matmul`` on a ternary (biases == -scales) matrix: its error
  against a float32 reference stays within stock's own error, and it declines
  every shape or layout outside its contract.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")


def _chain_rotate(x, signs, block, inverse=False):
    import math

    dtype = x.dtype
    y = x.astype(mx.float32)
    if not inverse:
        y = y * signs
    width = int(x.shape[-1])
    y = mx.hadamard_transform(mx.unflatten(y, -1, (width // block, block)), scale=1.0 / math.sqrt(block))
    y = mx.flatten(y, -2, -1)
    if inverse:
        y = y * signs
    return y.astype(dtype)


def _activations(shape, seed):
    mx.random.seed(seed)
    x = mx.random.normal(shape) * 2.0
    # Heavy tails and tiny values, as a residual stream has.
    spikes = (mx.random.uniform(shape=shape) < 0.01) * mx.random.normal(shape) * 300.0
    small = (mx.random.uniform(shape=shape) < 0.05) * 1e-4
    return (x + spikes + small).astype(mx.float16)


@pytest.mark.parametrize("shape", [(1, 1, 1024), (1, 2, 5120), (1, 4, 6144), (3, 17408), (2, 7, 5120)])
@pytest.mark.parametrize("inverse", [False, True])
def test_fused_rotation_is_bit_identical_to_the_mlx_chain(monkeypatch, shape, inverse):
    from mtplx.kernels import hadamard_rotate as hr

    monkeypatch.setenv(hr.ENV, "1")
    width = shape[-1]
    x = _activations(shape, seed=width + len(shape))
    signs = mx.where(mx.random.uniform(shape=(width,)) < 0.5, -1.0, 1.0).astype(mx.float32)
    expected = _chain_rotate(x, signs, 1024, inverse=inverse)
    actual = hr.rotate(x, signs, 1024, inverse=inverse)
    assert actual is not None
    assert actual.dtype == mx.float16 and actual.shape == x.shape
    assert bool(mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item())


def test_fused_rotation_declines_outside_its_contract(monkeypatch):
    from mtplx.kernels import hadamard_rotate as hr

    signs = mx.ones((2048,), dtype=mx.float32)
    monkeypatch.setenv(hr.ENV, "1")
    assert hr.rotate(mx.zeros((1, 2048), dtype=mx.bfloat16), signs, 1024) is None
    assert hr.rotate(mx.zeros((1, 2048), dtype=mx.float16), signs, 512) is None
    assert hr.rotate(mx.zeros((1, 2048), dtype=mx.float16), signs.astype(mx.float16), 1024) is None
    monkeypatch.setenv(hr.ENV, "0")
    assert hr.rotate(mx.zeros((1, 2048), dtype=mx.float16), signs, 1024) is None


def test_prism_rotation_uses_the_fused_kernel_when_enabled(monkeypatch):
    from mtplx.kernels import hadamard_rotate as hr
    from mtplx.models.prism_hadamard_qwen35 import hadamard_rotate

    x = _activations((1, 2, 5120), seed=3)
    signs = mx.where(mx.arange(5120) % 3 == 0, -1.0, 1.0).astype(mx.float32)
    monkeypatch.setenv(hr.ENV, "0")
    off = hadamard_rotate(x, signs, 1024)
    before = hr.counters()["served"]
    monkeypatch.setenv(hr.ENV, "1")
    on = hadamard_rotate(x, signs, 1024)
    assert hr.counters()["served"] == before + 1
    assert bool(mx.array_equal(off.view(mx.uint16), on.view(mx.uint16)).item())


def _ternary_matrix(n, k, seed):
    mx.random.seed(seed)
    codes = mx.random.randint(0, 3, (n, k)).astype(mx.uint32)
    shifts = (mx.arange(16, dtype=mx.uint32) * 2)[None, None, :]
    words = (codes.reshape(n, k // 16, 16) << shifts).sum(axis=-1).astype(mx.uint32)
    scales = (0.004 + 0.02 * mx.random.uniform(shape=(n, k // 128))).astype(mx.float16)
    return words, scales, -scales


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("n,k", [(256, 1024), (1024, 5120), (512, 17408)])
def test_ternary_gemv_matches_stock_within_float32_rounding(monkeypatch, rows, n, k):
    from mtplx.kernels import ternary_qmv as tq

    monkeypatch.setenv(tq.ENV, "1")
    w, s, b = _ternary_matrix(n, k, seed=n + k + rows)
    assert tq.ternary_layout(s, b)
    x = _activations((1, rows, k), seed=rows)
    stock = mx.quantized_matmul(x, w, scales=s, biases=b, transpose=True, group_size=128, bits=2)
    mine = tq.ternary_qmv(x, w, s)
    assert mine is not None and mine.shape == stock.shape and mine.dtype == mx.float16
    wf = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=128, bits=2)
    truth = x.astype(mx.float32) @ wf.T
    err_stock = mx.abs(stock.astype(mx.float32) - truth)
    err_mine = mx.abs(mine.astype(mx.float32) - truth)

    def rms(e):
        return float(mx.sqrt(mx.mean(e * e)).item())

    # Same numerical class as stock: no worse RMS or worst-case error against
    # the float32 reference (the bar mlx-serve's qmv2 test sets), and at the
    # verify widths (M >= 2, stock's qmv_wide order) the same float16 bits on
    # almost every output. At M = 1 stock's qmv_fast order is the less
    # accurate one, so only the error bound applies.
    assert rms(err_mine) <= 1.05 * rms(err_stock)
    assert float(mx.max(err_mine).item()) <= 1.05 * float(mx.max(err_stock).item()) + 1e-6
    if rows >= 2:
        assert float(mx.mean((mine == stock).astype(mx.float32)).item()) >= 0.99


@pytest.mark.parametrize("rows", [1, 2])
def test_ternary_gemv_head_geometry_matches_stock(monkeypatch, rows):
    from mtplx.kernels import ternary_qmv as tq

    monkeypatch.setenv(tq.ENV, "1")
    n, k = tq.HEAD_MIN_N, 1024
    w, s, b = _ternary_matrix(n, k, seed=rows)
    x = _activations((1, rows, k), seed=10 + rows)
    stock = mx.quantized_matmul(x, w, scales=s, biases=b, transpose=True, group_size=128, bits=2)
    mine = tq.ternary_qmv(x, w, s)
    assert mine is not None and mine.shape == stock.shape
    wf = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=128, bits=2)
    truth = x.astype(mx.float32) @ wf.T
    err_stock = mx.abs(stock.astype(mx.float32) - truth)
    err_mine = mx.abs(mine.astype(mx.float32) - truth)
    assert float(mx.max(err_mine).item()) <= 1.05 * float(mx.max(err_stock).item()) + 1e-6


@pytest.mark.parametrize("geometry", ["1x8", "4x8", "2x4", "8x2"])
def test_ternary_gemv_geometry_never_changes_the_bits(monkeypatch, geometry):
    # Every output row is summed in the same order whatever the rows per
    # simdgroup or simdgroups per threadgroup, so tuning is free of numerics.
    from mtplx.kernels import ternary_qmv as tq

    monkeypatch.setenv(tq.ENV, "1")
    w, s, _ = _ternary_matrix(1024, 5120, seed=4)
    x = _activations((1, 2, 5120), seed=5)
    monkeypatch.delenv("MTPLX_TERNARY_QMV_GEOMETRY", raising=False)
    default = tq.ternary_qmv(x, w, s)
    monkeypatch.setenv("MTPLX_TERNARY_QMV_GEOMETRY", geometry)
    tuned = tq.ternary_qmv(x, w, s)
    assert default is not None and tuned is not None
    assert bool(mx.array_equal(default.view(mx.uint16), tuned.view(mx.uint16)).item())


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_planned_run_returns_the_same_bits_as_the_checked_call(monkeypatch, rows):
    from mtplx.kernels import ternary_qmv as tq

    monkeypatch.setenv(tq.ENV, "1")
    w, s, _ = _ternary_matrix(2048, 5120, seed=20 + rows)
    x = _activations((1, rows, 5120), seed=30 + rows)
    p = tq.plan(w, s)
    assert p is not None and (p.n, p.k) == (2048, 5120)
    a = tq.ternary_qmv(x, w, s)
    b = tq.run(p, x, w, s)
    assert b is not None and b.shape == a.shape
    assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())
    # Out of contract activations decline, like the checked call.
    assert tq.run(p, x.astype(mx.bfloat16), w, s) is None
    assert tq.run(p, _activations((1, 5, 5120), seed=1), w, s) is None
    assert tq.run(p, _activations((1, 2, 1024), seed=1), w, s) is None
    monkeypatch.setenv(tq.ENV, "0")
    assert tq.run(p, x, w, s) is None


def test_plan_refuses_shapes_outside_the_contract():
    from mtplx.kernels import ternary_qmv as tq

    w, s, _ = _ternary_matrix(256, 1024, seed=3)
    assert tq.plan(w, s) is not None
    w2, s2, _ = _ternary_matrix(256, 1152, seed=3)
    assert tq.plan(w2, s2) is None  # K not a multiple of 512
    assert tq.plan(w, s.astype(mx.float32)) is None
    assert tq.plan(w, s[:, :4]) is None


def test_only_matrices_large_enough_to_win_are_worthwhile():
    from mtplx.kernels import ternary_qmv as tq

    assert not tq.worthwhile(1024)  # Bonsai's key and value projections
    assert tq.worthwhile(5120) and tq.worthwhile(17408) and tq.worthwhile(248320)


def test_both_kernels_are_on_by_default_and_zero_turns_them_off(monkeypatch):
    from mtplx.kernels import hadamard_rotate as hr
    from mtplx.kernels import ternary_qmv as tq

    for module in (hr, tq):
        monkeypatch.delenv(module.ENV, raising=False)
        assert module.enabled()
        for off in ("0", "false", "off", "no"):
            monkeypatch.setenv(module.ENV, off)
            assert not module.enabled()
        monkeypatch.setenv(module.ENV, "1")
        assert module.enabled()


def test_ternary_gemv_declines_outside_its_contract(monkeypatch):
    from mtplx.kernels import ternary_qmv as tq

    monkeypatch.setenv(tq.ENV, "1")
    w, s, b = _ternary_matrix(256, 1024, seed=1)
    assert tq.ternary_qmv(mx.zeros((1, 5, 1024), dtype=mx.float16), w, s) is None  # M > 4
    assert tq.ternary_qmv(mx.zeros((1, 2, 1024), dtype=mx.bfloat16), w, s) is None
    w2, s2, _ = _ternary_matrix(256, 1152, seed=2)  # K not a multiple of 512
    assert tq.ternary_qmv(mx.zeros((1, 2, 1152), dtype=mx.float16), w2, s2) is None
    assert not tq.ternary_layout(s, b + mx.array(1e-3, dtype=mx.float16))
    monkeypatch.setenv(tq.ENV, "0")
    assert tq.ternary_qmv(mx.zeros((1, 2, 1024), dtype=mx.float16), w, s) is None
