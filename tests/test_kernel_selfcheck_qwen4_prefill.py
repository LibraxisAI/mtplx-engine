"""Load-time self-check for the four Flash-Next (qwen4_exp) prefill kernels.

The hyper-connection read, the GDN gated norm, the GDN prefill prework and the
MoE prefill combine are plain SIMD, on by default on every Mac, and built to
return the stock MLX chain's exact bits, but only an M5 has measured them.
The self-check runs each on this GPU at load: a bit difference, a build
failure or a launch MLX refuses turns that lane off for the process and the
model's call site takes the stock path instead, lane by lane.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx import kernel_selfcheck
from mtplx.attention_context import attention_phase
from mtplx.kernel_selfcheck import (
    lane_disabled,
    report_for_health,
    run_kernel_selfcheck,
    selfcheck_enabled,
)
from mtplx.kernels import gdn_gated_norm as gn
from mtplx.kernels import gdn_prefill_prework as pw
from mtplx.kernels import hc_prefill as hc
from mtplx.kernels import qwen4_moe_prefill_combine as mc
from mtplx.models import qwen4_exp

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")

_KERNELS = (hc, gn, pw, mc)
_LANES = tuple(module.LANE for module in _KERNELS)
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
    for env in (*(module.ENV for module in _KERNELS), *_TURBO_ENVS):
        monkeypatch.delenv(env, raising=False)
    kernel_selfcheck._reset_for_tests()
    yield
    kernel_selfcheck._reset_for_tests()


def _bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a)


def _same(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(_bits(a), _bits(b))


def _check(**kwargs):
    return run_kernel_selfcheck(mx.bfloat16, 4, 64, **kwargs)


def _raise(*args, **kwargs):
    raise RuntimeError("[metal::Device] Unable to build metal library from source")


def test_all_four_flash_next_lanes_pass_on_this_machine():
    report = _check(flash_next=True)
    for module in _KERNELS:
        assert report["lanes"][module.LANE] == "ok"
        assert report["dmax"][module.LANE] == 0.0  # bitwise, not merely close
        assert module.enabled()
    health = report_for_health()
    assert all(health[lane] == "ok" for lane in _LANES)


def test_lanes_are_skipped_for_other_models_and_when_switched_off(monkeypatch):
    report = _check()
    assert all(report["lanes"][lane] == "skipped" for lane in _LANES)
    assert all(lane in report_for_health() for lane in _LANES)
    monkeypatch.setenv(pw.ENV, "0")
    report = _check(flash_next=True)
    assert report["lanes"][pw.LANE] == "skipped"
    assert not pw.enabled() and not lane_disabled(pw.LANE)
    assert all(report["lanes"][m.LANE] == "ok" for m in (hc, gn, mc))


def test_a_hc_read_that_fails_to_build_falls_back_and_the_eager_chain_serves(monkeypatch):
    mx.random.seed(71)
    owner = qwen4_exp.GatedResidual(
        SimpleNamespace(hc_count=4, hidden_size=2560, hc_lowrank=320, rms_norm_eps=1e-6),
        use_combine=True,
    )
    owner.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 33, 10240)).astype(mx.bfloat16)
    monkeypatch.setenv(hc.ENV, "0")
    stock = owner(x)
    monkeypatch.delenv(hc.ENV)

    monkeypatch.setattr(hc, "_normalize", _raise)
    report = _check(flash_next=True)
    assert report["lanes"][hc.LANE] == "fallback"
    assert lane_disabled(hc.LANE) and not hc.enabled() and hc.switched_on()
    assert all(report["lanes"][m.LANE] == "ok" for m in (gn, pw, mc))  # per lane
    assert report_for_health()[hc.LANE] == "fallback"
    # The call site never reaches the raising kernel: the eager chain serves.
    served = owner(x)
    assert served[1] is x
    assert _same(served[0], stock[0]) and _same(served[2], stock[2])


def test_a_gated_norm_with_wrong_bits_falls_back_and_the_stock_gate_serves(monkeypatch):
    norm = qwen4_exp.SigmoidRMSNormGated(128)
    norm.set_dtype(mx.bfloat16)
    wide = mx.random.normal((1, 64, 48, 128)).astype(mx.bfloat16)
    gate = mx.random.normal((1, 64, 48, 128)).astype(mx.bfloat16)
    monkeypatch.setenv(gn.ENV, "0")
    with attention_phase("prefill"):
        stock = norm(wide, gate)
    monkeypatch.delenv(gn.ENV)

    real = gn.sigmoid_gate
    monkeypatch.setattr(
        gn, "sigmoid_gate", lambda x, g: real(x, g) + mx.array(1.0, dtype=mx.bfloat16)
    )
    report = _check(flash_next=True)
    assert report["lanes"][gn.LANE] == "fallback"
    assert report["dmax"][gn.LANE] > 0.0
    assert not gn.enabled()
    with attention_phase("prefill"):
        assert not qwen4_exp._gdn_gated_norm_fused_applies(wide, wide, gate)
        assert _same(norm(wide, gate), stock)


def _family_gdn():
    args = qwen4_exp.TextArgs(
        hidden_size=256,
        num_hidden_layers=2,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    mx.random.seed(11)
    layer = qwen4_exp.GatedDeltaNet(args)
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def test_a_prework_kernel_that_raises_falls_back_and_the_staged_chain_serves(monkeypatch):
    from mlx_lm.models.cache import ArraysCache

    layer = _family_gdn()
    x = (mx.random.normal((1, 96, 256)) * 0.5).astype(mx.bfloat16)

    def run():
        cache = ArraysCache(size=2)
        with attention_phase("prefill"):
            y = layer(x, cache=cache)
        mx.eval(y, cache[0], cache[1])
        return y, cache[0], cache[1]

    monkeypatch.setenv(pw.ENV, "0")
    stock = run()
    monkeypatch.delenv(pw.ENV)

    monkeypatch.setattr(pw, "gdn_prefill_prework", _raise)
    report = _check(flash_next=True)
    assert report["lanes"][pw.LANE] == "fallback"
    assert not pw.enabled()
    for served, want in zip(run(), stock):
        assert _same(served, want)


def _moe_block(hidden=256, experts=16, top_k=4, inter=64, bits=4, group=32):
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        norm_topk_prob=True,
        num_experts=experts,
        num_experts_per_tok=top_k,
    )
    block = qwen4_exp.SparseMoeBlock(args)
    mx.random.seed(7)
    gu = (mx.random.normal((experts, 2 * inter, hidden)) * 0.05).astype(mx.bfloat16)
    gu_w, gu_s, gu_b = mx.quantize(gu, group_size=group, bits=bits)
    down = QuantizedSwitchLinear(inter, hidden, experts, bias=False, group_size=group, bits=bits)
    down.set_dtype(mx.bfloat16)
    block.switch_mlp = qwen4_exp._FusedGateUpSwitchGLU(
        down, gu_w, gu_s, gu_b, group, bits, "affine"
    )
    block.set_dtype(mx.bfloat16)
    return block


def test_a_moe_combine_that_raises_falls_back_and_the_stock_tail_serves(monkeypatch):
    block = _moe_block()
    x = mx.random.normal((1, 96, 256)).astype(mx.bfloat16)
    monkeypatch.setenv(mc.ENV, "0")
    stock = block(x)
    monkeypatch.delenv(mc.ENV)
    assert qwen4_exp._moe_prefill_combine_applies(block, x)

    monkeypatch.setattr(mc, "_launch", _raise)
    report = _check(flash_next=True)
    assert report["lanes"][mc.LANE] == "fallback"
    assert not mc.enabled()
    assert not qwen4_exp._moe_prefill_combine_applies(block, x)
    assert _same(block(x), stock)


def test_a_lane_turned_off_once_is_probed_again_on_the_next_load(monkeypatch):
    real = mc._launch
    monkeypatch.setattr(mc, "_launch", _raise)
    _check(flash_next=True)
    assert lane_disabled(mc.LANE)
    monkeypatch.setattr(mc, "_launch", real)
    report = _check(flash_next=True)
    assert report["lanes"][mc.LANE] == "ok" and mc.enabled()


def test_a_flash_next_model_is_checked_under_every_profile(monkeypatch):
    assert selfcheck_enabled() is False
    assert selfcheck_enabled(flash_next=True) is True
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "0")
    assert selfcheck_enabled(flash_next=True) is False


def test_selfcheck_off_skips_the_lanes_and_leaves_the_kernels_on(monkeypatch):
    monkeypatch.setattr(kernel_selfcheck, "_flash_next_model", lambda model: True)
    monkeypatch.setattr(mc, "_launch", _raise)  # would fail if it were probed
    monkeypatch.setenv("MTPLX_KERNEL_SELFCHECK", "0")
    assert kernel_selfcheck.maybe_run_model_selfcheck(object()) is None
    assert report_for_health() == {"ran": False}
    assert all(module.enabled() for module in _KERNELS)


def test_model_entry_probes_the_lanes_for_a_flash_next_model_and_its_mtp_subclass(
    monkeypatch,
):
    monkeypatch.setattr(
        kernel_selfcheck, "_model_quant_signature", lambda model: (mx.bfloat16, 4, 64)
    )
    # MTP injection swaps the loaded model's class for a subclass.
    mtp_class = type("_MTPLXOuterModel", (qwen4_exp.Model,), {})
    for model in (qwen4_exp.Model.__new__(qwen4_exp.Model), mtp_class.__new__(mtp_class)):
        assert kernel_selfcheck._flash_next_model(model)
        kernel_selfcheck._reset_for_tests()
        report = kernel_selfcheck.maybe_run_model_selfcheck(model)
        assert report is not None
        assert all(report["lanes"][lane] == "ok" for lane in _LANES)


def test_other_models_are_not_mistaken_for_flash_next():
    assert kernel_selfcheck._flash_next_model(object()) is False
    assert kernel_selfcheck._flash_next_model(qwen4_exp.TextModel.__new__(qwen4_exp.TextModel)) is False
    report = kernel_selfcheck.maybe_run_model_selfcheck(object())
    assert report is None  # no turbo env, not Prism, not Flash-Next
