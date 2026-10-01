"""Nothing goes slow in silence (PX.2, 2026-09-18): the demotion ledger and
its three readers: /health, the request log, and `mtplx doctor --explain`."""

from __future__ import annotations

import json
import sys
import types

import pytest

from mtplx import demotions, lane_explain


@pytest.fixture(autouse=True)
def _clean_ledger():
    demotions.reset()
    yield
    demotions.reset()


def test_note_counts_and_keeps_the_last_reason():
    demotions.note("fixed_m4_lane_skipped", "prompt 9000 tokens over the line")
    demotions.note("fixed_m4_lane_skipped", "prompt 12000 tokens over the line")
    demotions.note("copy_round_eager")
    snap = demotions.snapshot()
    assert snap["counts"]["fixed_m4_lane_skipped"] == 2
    assert snap["counts"]["copy_round_eager"] == 1
    assert snap["total"] == 3
    assert snap["reasons"]["fixed_m4_lane_skipped"].startswith("prompt 12000")
    assert set(snap["meanings"]) == {"fixed_m4_lane_skipped", "copy_round_eager"}
    # Every declared kind is present with a zero, so a reader can tell
    # "never happened" from "this build does not count it".
    assert set(demotions.KINDS) <= set(snap["counts"])


def test_note_never_raises_on_an_undeclared_kind():
    demotions.note("a_kind_nobody_declared", "why")
    assert demotions.counts()["a_kind_nobody_declared"] == 1


def test_since_reports_only_what_moved():
    demotions.note("copy_round_eager", count=5)
    before = demotions.mark()
    demotions.note("copy_round_eager", count=2)
    demotions.note("vision_request_eager_verify", "image turn")
    assert demotions.since(before) == {
        "copy_round_eager": 2,
        "vision_request_eager_verify": 1,
    }
    assert demotions.since(demotions.mark()) == {}


def test_bank_fallback_labels_map_to_kinds():
    demotions.note_bank_fallback("growth_budget_exhausted")
    demotions.note_bank_fallback("block_window_capacity")
    demotions.note_bank_fallback("context_above_threshold")
    demotions.note_bank_fallback("python_cache_offsets")
    counts = demotions.counts()
    assert counts["compiled_verify_growth_demotion"] == 2
    assert counts["compiled_verify_context_fence"] == 1
    assert counts["compiled_verify_other_fallback"] == 1
    assert demotions.snapshot()["reasons"]["compiled_verify_other_fallback"] == (
        "python_cache_offsets"
    )


def test_tensor_unit_bails_are_read_from_the_kernel_counters(monkeypatch):
    # The kernels already count their gate bails per call; the ledger reads
    # them at snapshot time and adds nothing to the verify path.
    fake = types.ModuleType("mtplx.kernels.sdpa_nax_flash")
    fake.nax_flash_bail_counts = {"gpu_family_or_os": 96, "other": 3}
    monkeypatch.setitem(sys.modules, "mtplx.kernels.sdpa_nax_flash", fake)
    snap = demotions.snapshot()
    assert snap["counts"]["tensor_unit_route_bail"] == 96
    assert "no tensor units" in snap["reasons"]["tensor_unit_route_bail"]


def test_note_is_a_plain_increment_with_no_imports():
    # The cost contract: importing the ledger must not import MLX or the
    # server, and a note must not allocate a new key for a declared kind.
    import importlib

    module = importlib.reload(demotions)
    assert not any(
        name == "mlx" or name.startswith("mlx.")
        for name in getattr(module, "__dict__", {})
    )
    before_keys = set(module._COUNTS)
    module.note("fixed_m4_uncompiled_round", "width 3")
    assert set(module._COUNTS) == before_keys


def test_health_degradation_payload_carries_the_ledger():
    from mtplx.server import openai

    demotions.note("qsa_prefill_lane_off", "no tensor units and no Steel extension")
    payload = openai._health_degradation_payload(types.SimpleNamespace())
    assert payload["demotions"]["counts"]["qsa_prefill_lane_off"] == 1
    assert payload["demotions"]["total"] == 1
    json.dumps(payload["demotions"])


def _envelope(stats):
    from mtplx.server import openai

    return openai._metrics_envelope(
        stats=stats,
        prompt_tokens=10,
        completion_tokens=4,
        request_elapsed_s=1.0,
        token_times=[0.1, 0.2, 0.3, 0.4],
        request_started_s=0.0,
        lock_wait_time_s=0.0,
        session_id=None,
        session_cache_hit=False,
        cache_miss_reason=None,
        session_restore_mode="none",
        mtp_depth=3,
        generation_limits={},
    )


def test_request_log_envelope_carries_the_request_delta():
    envelope = _envelope({"demotions": {"fixed_m4_uncompiled_round": 28}})
    assert envelope["demotions"] == {"fixed_m4_uncompiled_round": 28}
    assert "demotions" not in _envelope({})


def test_explain_with_no_server_says_so():
    report = lane_explain.build_explain_report(
        health={"ok": False, "error": "refused", "url": "x"},
        server_url="http://127.0.0.1:8000",
        tensor_units={"architecture": "applegpu_g16s", "hardware": False,
                      "route": False, "fallback_forced": False},
    )
    assert report["server_reachable"] is False
    text = "\n".join(lane_explain.render_explain_lines(report))
    assert "No MTPLX server answered" in text
    assert "tensor units: not available (applegpu_g16s)" in text


def test_explain_prints_counts_reasons_and_sources():
    demotions.note("fixed_m4_uncompiled_round", "verify width other than 4", 2160)
    demotions.note("fixed_m4_lane_skipped", "prompt 162000 tokens: live 97.0 GB")
    health = {
        "degradation": {"demotions": demotions.snapshot()},
        "settings": {"model": "m", "depth": 3, "adaptive_policy": "expected_value"},
    }
    report = lane_explain.build_explain_report(
        health=health,
        server_url="http://127.0.0.1:8000",
        tensor_units={"architecture": "applegpu_g17s", "hardware": True,
                      "route": True, "fallback_forced": False},
        lane={
            "choices": {
                "depth": {"value": 3, "source": lane_explain.SOURCE_DEMOTED,
                          "reason": "adaptive depth has no effect on this model yet"},
                "prefill_chunk": {"value": 2048, "source": lane_explain.SOURCE_FAMILY},
            },
            "notes": ["inherited, unmeasured"],
        },
    )
    text = "\n".join(lane_explain.render_explain_lines(report))
    assert "tensor units: available (applegpu_g17s)" in text
    assert "   2,160  fixed_m4_uncompiled_round" in text
    assert "last reason: prompt 162000 tokens" in text
    assert "depth: 3 (demoted). adaptive depth has no effect on this model yet" in text
    assert "prefill_chunk: 2048 (family)" in text
    assert "adaptive_policy: expected_value" in text


def test_explain_says_when_an_older_server_has_no_counters():
    report = lane_explain.build_explain_report(
        health={"degradation": {"nax": {}}},
        server_url="http://127.0.0.1:8000",
        tensor_units={"architecture": "applegpu_g17s", "hardware": True,
                      "route": True, "fallback_forced": False},
    )
    assert report["server_reachable"] is True
    assert "predates" in "\n".join(lane_explain.render_explain_lines(report))


def test_doctor_explain_flag_parses_and_renders(monkeypatch, capsys):
    from mtplx.cli import build_parser
    from mtplx.commands import public

    args = build_parser().parse_args(["doctor", "--explain"])
    assert args.explain is True
    demotions.note("copy_round_eager", "copy-block rounds run the eager forward")
    monkeypatch.setattr(
        public,
        "_http_json",
        lambda url, **_kw: {"degradation": {"demotions": demotions.snapshot()}},
    )
    report = {"explain": public._doctor_explain_report(args, set())}
    assert report["explain"]["server_url"] == "http://127.0.0.1:8000"
    assert public._render_doctor_report(args, report) == 0
    out = capsys.readouterr().out
    assert "copy_round_eager" in out and "Explain" in out
