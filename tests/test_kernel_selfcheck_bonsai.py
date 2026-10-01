"""Load-time self-check for the two Ternary Bonsai (Prism) kernels.

The fused Hadamard rotation and the ternary GEMV are plain SIMD and run on
every GPU generation with no family gate, but only an M5 has measured them.
The self-check runs both on this GPU at load: a mismatch, a build failure or a
launch MLX refuses turns that lane off for the process and the stock MLX
path serves instead, lane by lane.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx import kernel_selfcheck
from mtplx.kernel_selfcheck import (
    lane_disabled,
    report_for_health,
    run_kernel_selfcheck,
    selfcheck_enabled,
)
from mtplx.kernels import hadamard_rotate as hr
from mtplx.kernels import ternary_qmv as tq

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")

_TURBO_ENVS = (
    "MTPLX_NAX_VERIFY",
    "MTPLX_GQA_PACKED_SDPA",
    "MTPLX_QWEN_ROW_OWNED_ROUTER",
    "MTPLX_QWEN_COMBINE_TAIL",
    "MTPLX_FUSE_GDN_POST_CONV",
    "MTPLX_A3B_WHOLE_MOE_FUSION",
    "MTPLX_KERNEL_SELFCHECK",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (hr.ENV, tq.ENV, *_TURBO_ENVS):
        monkeypatch.delenv(env, raising=False)
    kernel_selfcheck._reset_for_tests()
    yield
    kernel_selfcheck._reset_for_tests()


def test_both_bonsai_lanes_pass_on_this_machine():
    report = run_kernel_selfcheck(mx.float16, 2, 128, prism_ternary=True)
    assert report["lanes"][hr.LANE] == "ok"
    assert report["lanes"][tq.LANE] == "ok"
    assert report["dmax"][hr.LANE] == 0.0  # bitwise, not merely close
    assert report["dmax"][tq.LANE] <= 2e-3  # a float16 step or two
    assert hr.enabled() and tq.enabled()


def test_lanes_are_skipped_for_other_models_and_when_switched_off(monkeypatch):
    report = run_kernel_selfcheck(mx.float16, 2, 128)
    assert report["lanes"][hr.LANE] == "skipped"
    assert report["lanes"][tq.LANE] == "skipped"
    monkeypatch.setenv(hr.ENV, "0")
    report = run_kernel_selfcheck(mx.float16, 2, 128, prism_ternary=True)
    assert report["lanes"][hr.LANE] == "skipped"
    assert report["lanes"][tq.LANE] == "ok"


def test_a_wrong_rotation_turns_off_only_its_lane_and_the_chain_serves(monkeypatch):
    from mtplx.models.prism_hadamard_qwen35 import hadamard_rotate, mlx_chain_rotate

    real = hr._launch
    monkeypatch.setattr(
        hr, "_launch", lambda x, s, inv: real(x, s, inv) + mx.array(1.0, dtype=mx.float16)
    )
    report = run_kernel_selfcheck(mx.float16, 2, 128, prism_ternary=True)
    assert report["lanes"][hr.LANE] == "fallback"
    assert lane_disabled(hr.LANE) and not hr.enabled() and hr.switched_on()
    assert report["lanes"][tq.LANE] == "ok"  # per lane, not global
    assert report_for_health()[hr.LANE] == "fallback"

    x = (mx.random.normal((1, 2, 2048)) * 2.0).astype(mx.float16)
    signs = mx.where(mx.arange(2048) % 3 == 0, -1.0, 1.0).astype(mx.float32)
    assert hr.rotate(x, signs, 1024) is None
    served = hadamard_rotate(x, signs, 1024)
    chain = mlx_chain_rotate(x, signs, 1024)
    assert bool(mx.array_equal(served.view(mx.uint16), chain.view(mx.uint16)).item())


def test_a_ternary_kernel_that_fails_to_build_falls_back_to_stock(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("[metal::Device] Unable to build metal library from source")

    monkeypatch.setattr(tq, "_launch", broken)
    report = run_kernel_selfcheck(mx.float16, 2, 128, prism_ternary=True)
    assert report["lanes"][tq.LANE] == "fallback"
    assert not tq.enabled()
    assert report["lanes"][hr.LANE] == "ok"
    # The armed matrices' planned calls decline, so HadamardQuantizedLinear
    # runs stock mx.quantized_matmul.
    w = mx.zeros((2048, 1024 // 16), dtype=mx.uint32)
    s = mx.ones((2048, 1024 // 128), dtype=mx.float16)
    p = tq.plan(w, s)
    assert p is not None
    assert tq.run(p, mx.zeros((1, 2, 1024), dtype=mx.float16), w, s) is None


def test_a_lane_turned_off_once_is_probed_again_on_the_next_load(monkeypatch):
    real = tq._launch
    monkeypatch.setattr(tq, "_launch", lambda *a: real(*a) + mx.array(1.0, dtype=mx.float16))
    run_kernel_selfcheck(mx.float16, 2, 128, prism_ternary=True)
    assert lane_disabled(tq.LANE)
    monkeypatch.setattr(tq, "_launch", real)
    report = run_kernel_selfcheck(mx.float16, 2, 128, prism_ternary=True)
    assert report["lanes"][tq.LANE] == "ok" and tq.enabled()


def test_a_prism_model_is_checked_under_every_profile(monkeypatch):
    assert selfcheck_enabled() is False
    assert selfcheck_enabled(prism_ternary=True) is True
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "0")
    assert selfcheck_enabled(prism_ternary=True) is False


def test_model_entry_probes_the_bonsai_lanes_for_a_prism_model(monkeypatch):
    monkeypatch.setattr(kernel_selfcheck, "_prism_ternary_model", lambda model: True)
    monkeypatch.setattr(
        kernel_selfcheck, "_model_quant_signature", lambda model: (mx.float16, 2, 128)
    )
    report = kernel_selfcheck.maybe_run_model_selfcheck(object())
    assert report is not None
    assert report["lanes"][hr.LANE] == "ok"
    assert report["lanes"][tq.LANE] == "ok"


def test_other_models_are_not_mistaken_for_prism():
    assert kernel_selfcheck._prism_ternary_model(object()) is False
