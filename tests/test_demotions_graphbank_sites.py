"""PX.2 call sites in mtplx/graphbank.py: a verify round that runs eager
through the compiled bank records a demotion. No model, no GPU: the bank is
built bare and its runtime forward is a stub."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx import demotions, graphbank


@pytest.fixture(autouse=True)
def _clean_ledger():
    demotions.reset()
    yield
    demotions.reset()


def _bare_bank(*, fixed_m4: bool):
    bank = object.__new__(graphbank.CompiledVerifyBank)
    bank._fixed_m4_dispatch = {"installed": True} if fixed_m4 else None
    bank.stats = {"calls": 0, "fallback_calls": 0, "fallback_reasons": {}}
    bank.strict_no_fallback = False
    bank._growth_budget_fallback_reported = False
    bank.reserve_fixed_m4_window = lambda *_a, **_k: None
    bank._runtime_forward = lambda ids, **_kw: ("logits", "hidden", {"eager": True})
    return bank


def _ids(width: int):
    return SimpleNamespace(shape=(1, width), ndim=2)


@pytest.mark.parametrize("width", [2, 3, 5, 6])
def test_a_flash_next_round_at_an_uncompiled_width_is_counted(width):
    bank = _bare_bank(fixed_m4=True)
    out = bank.forward_ar_capture(_ids(width), cache=[])
    assert out[2] == {"eager": True}
    assert bank.last_fallback_reason == "fixed_m4_short_window"
    snap = demotions.snapshot()
    assert snap["counts"]["fixed_m4_uncompiled_round"] == 1
    assert "no compiled route" in snap["reasons"]["fixed_m4_uncompiled_round"]


def test_a_width_4_call_without_host_inputs_is_counted():
    bank = _bare_bank(fixed_m4=True)
    bank.forward_ar_capture(_ids(4), cache=[])
    assert bank.last_fallback_reason == "fixed_m4_host_inputs_missing"
    snap = demotions.snapshot()
    assert snap["counts"]["fixed_m4_uncompiled_round"] == 1
    assert "host-owned n-gram inputs" in snap["reasons"]["fixed_m4_uncompiled_round"]


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("growth_budget_exhausted", "compiled_verify_growth_demotion"),
        ("block_window_capacity", "compiled_verify_growth_demotion"),
        ("context_above_threshold", "compiled_verify_context_fence"),
        ("capacity_overflow", "compiled_verify_other_fallback"),
    ],
)
def test_the_27b_fallback_funnel_feeds_the_ledger(reason, kind):
    bank = _bare_bank(fixed_m4=False)
    for _round in range(3):
        bank._fallback(
            _ids(4), cache=[], return_hidden=True, hidden_variant=None, reason=reason
        )
    # Every eager round counts: the number a user reads is "how many verify
    # rounds ran eager", while the bank's own per-request stats keep counting
    # a growth exhaustion as one transition.
    assert demotions.counts()[kind] == 3
    assert bank.stats["fallback_calls"] == 3
    if reason == "growth_budget_exhausted":
        assert bank.stats["fallback_reasons"][reason] == 1


def test_per_round_reasons_are_constants():
    assert isinstance(graphbank._FIXED_M4_OTHER_WIDTH_REASON, str)
    assert isinstance(graphbank._FIXED_M4_NO_HOST_INPUTS_REASON, str)
