"""PX.2 call sites in mtplx/models/qwen4_exp.py: a Flash-Next prefill chunk
that should have been sparse and ran dense records a demotion."""

from __future__ import annotations

import pytest

from mtplx import demotions
from mtplx.models import qwen4_exp


@pytest.fixture(autouse=True)
def _clean_ledger():
    demotions.reset()
    yield
    demotions.reset()


@pytest.fixture
def prefill_phase(monkeypatch):
    monkeypatch.setattr(qwen4_exp, "current_attention_phase", lambda: "prefill")
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_min_rows", lambda: 32)
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_min_context", lambda: 32_768)


def test_an_eligible_chunk_with_the_lane_off_is_counted(monkeypatch, prefill_phase):
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_enabled", lambda: False)
    assert qwen4_exp._qsa_large_prefill_enabled(2_048, 40_000) is False
    snap = demotions.snapshot()
    assert snap["counts"]["qsa_prefill_lane_off"] == 1
    assert "no tensor units" in snap["reasons"]["qsa_prefill_lane_off"]


def test_an_eligible_chunk_with_the_lane_on_records_nothing(monkeypatch, prefill_phase):
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_enabled", lambda: True)
    assert qwen4_exp._qsa_large_prefill_enabled(2_048, 40_000) is True
    assert demotions.snapshot()["total"] == 0


def test_chunks_below_the_floor_are_a_choice_not_a_demotion(monkeypatch, prefill_phase):
    calls = []
    monkeypatch.setattr(
        qwen4_exp, "_qsa_prefill_enabled", lambda: calls.append(1) or False
    )
    # Below the context floor, below the row floor, and outside prefill.
    assert qwen4_exp._qsa_large_prefill_enabled(2_048, 16_384) is False
    assert qwen4_exp._qsa_large_prefill_enabled(4, 40_000) is False
    monkeypatch.setattr(qwen4_exp, "current_attention_phase", lambda: "decode_verify")
    assert qwen4_exp._qsa_large_prefill_enabled(2_048, 40_000) is False
    assert demotions.snapshot()["total"] == 0
    # The capability probe is still never paid on those calls.
    assert calls == []


def test_a_selected_chunk_nobody_consumes_is_counted():
    out = qwen4_exp._qsa_prefill_dispatch_tier(
        flash_supported=lambda: False,
        flash_call=lambda: "flash",
        direct_supported=lambda: False,
        direct_call=lambda: "direct",
        gather_enabled=False,
        gather_call=lambda: "gather",
    )
    assert out is None
    snap = demotions.snapshot()
    assert snap["counts"]["qsa_prefill_dense_mask"] == 1
    assert "dense mask was rebuilt" in snap["reasons"]["qsa_prefill_dense_mask"]


def test_a_consumed_chunk_records_nothing():
    out = qwen4_exp._qsa_prefill_dispatch_tier(
        flash_supported=lambda: False,
        flash_call=lambda: "flash",
        direct_supported=lambda: True,
        direct_call=lambda: "direct",
        gather_enabled=False,
        gather_call=lambda: "gather",
    )
    assert out == "direct"
    assert demotions.snapshot()["total"] == 0
