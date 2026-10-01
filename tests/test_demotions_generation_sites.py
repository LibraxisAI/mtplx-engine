"""PX.2 call sites in mtplx/generation.py: each Flash-Next request or round
that leaves the compiled verifier records a demotion."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from mtplx import demotions, generation


@pytest.fixture(autouse=True)
def _clean_ledger():
    demotions.reset()
    yield
    demotions.reset()


def _flash_next_rt(limit_bytes: int = 0):
    args = SimpleNamespace(
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 12,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
    )
    return SimpleNamespace(
        qwen4_fixed_m4_compiled_verify=True,
        model=SimpleNamespace(args=args),
        metal_memory_limit_bytes=limit_bytes or None,
    )


def test_depth_below_the_compiled_window_is_counted():
    receipt: dict = {}
    admitted = generation._qwen4_fixed_m4_compiled_verify_requested(
        _flash_next_rt(),
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=64,
        cached_tokens=0,
        prompt_tokens=100,
        speculative_depth=2,
        receipt=receipt,
    )
    assert admitted is False
    assert receipt["reason"] == "depth_below_compiled_window"
    snap = demotions.snapshot()
    assert snap["counts"]["fixed_m4_lane_skipped"] == 1
    assert "depth below 3" in snap["reasons"]["fixed_m4_lane_skipped"]


def test_a_family_without_the_lane_records_nothing():
    rt = SimpleNamespace(qwen4_fixed_m4_compiled_verify=False)
    assert not generation._qwen4_fixed_m4_compiled_verify_requested(
        rt,
        verify_strategy="capture_commit",
        compiled_mode="on",
        max_tokens=64,
        cached_tokens=0,
        prompt_tokens=100,
        speculative_depth=2,
    )
    assert demotions.snapshot()["total"] == 0


def test_memory_gate_skip_is_counted_with_its_reason(monkeypatch, capsys):
    gib = 1024**3
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT", raising=False)
    monkeypatch.setattr(generation, "_mlx_live_memory_bytes", lambda: 97 * gib)
    monkeypatch.setattr(generation, "_mlx_release_allocator_cache", lambda: 0)
    fits = generation._qwen4_fixed_m4_lane_fits(
        _flash_next_rt(limit_bytes=96 * gib), prompt_tokens=162_000
    )
    assert fits is False
    snap = demotions.snapshot()
    assert snap["counts"]["fixed_m4_lane_skipped"] == 1
    assert "prompt 162000 tokens" in snap["reasons"]["fixed_m4_lane_skipped"]
    # The stderr line stays: the ledger is an addition, not a replacement.
    assert "[qwen4-fixed-M4] lane skipped" in capsys.readouterr().err


def test_memory_gate_admission_records_nothing(monkeypatch):
    gib = 1024**3
    monkeypatch.delenv("MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT", raising=False)
    monkeypatch.setattr(generation, "_mlx_live_memory_bytes", lambda: 80 * gib)
    assert generation._qwen4_fixed_m4_lane_fits(
        _flash_next_rt(limit_bytes=96 * gib), prompt_tokens=8_000
    )
    assert demotions.snapshot()["total"] == 0


def test_operator_ceiling_skip_is_counted(monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT", "4096")
    assert not generation._qwen4_fixed_m4_lane_fits(
        _flash_next_rt(), prompt_tokens=5_000
    )
    assert demotions.snapshot()["counts"]["fixed_m4_lane_skipped"] == 1


def test_generation_stats_carry_the_request_delta():
    fields = {f.name for f in generation.GenerationStats.__dataclass_fields__.values()}
    assert "demotions" in fields
    source = inspect.getsource(generation.generate_mtpk)
    assert "_demotions_at_start = _demotion_mark()" in source
    assert "demotions=_demotions_since(_demotions_at_start)" in source
    # The mark is taken before the prefill, so a sparse-prefill demotion is
    # attributed to the request that paid for it.
    assert source.index("_demotions_at_start = _demotion_mark()") < source.index(
        "restore_or_prefill_prompt_state("
    )


def test_copy_round_and_vision_sites_are_wired():
    source = inspect.getsource(generation.generate_mtpk)
    forward = source.index("_cb_logits, _cb_hidden = rt.forward_ar(")
    note = source.index('_note_demotion("copy_round_eager", _COPY_ROUND_EAGER_REASON)')
    assert 0 < note - forward < 400
    # The reason on the per-round path is a constant: no formatting per round.
    assert isinstance(generation._COPY_ROUND_EAGER_REASON, str)
    # An image request that stays on the eager verifier is counted where it is
    # decided: the fixed-M4 admission, which generate_mtpk calls, notes it with
    # the reason the request record carries (the refusals themselves are
    # exercised in tests/test_qwen4_fixed_m4_vision_admission.py).
    assert "_qwen4_fixed_m4_admission(" in source
    admission = inspect.getsource(generation._qwen4_fixed_m4_admission)
    note = admission.index('"vision_request_eager_verify"')
    refusal = admission.index("if refusal is not None:")
    assert 0 < note - refusal < 400
    # Every reason is a constant line: nothing is formatted per request.
    assert all(
        isinstance(reason, str) and reason
        for reason in generation._VISION_COMPILED_VERIFY_REFUSALS.values()
    )
