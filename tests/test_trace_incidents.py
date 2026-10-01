import json

import pytest

from mtplx.commands.trace import _join_session, _match_receipt
from mtplx.commands.trace_analysis import request_diagnostics, scope_since
from mtplx.commands.trace_clients import load_pi_session
from mtplx.commands.trace_record import system_sample


def test_pi_compaction_is_a_request_and_only_joins_matching_usage_and_clock(tmp_path):
    records = [
        {"type": "session", "id": "pi", "cwd": str(tmp_path)},
        {"type": "message", "id": "old", "parentId": None,
         "message": {"role": "user", "timestamp": 1000, "content": "Old history"}},
        {"type": "compaction", "id": "compact", "parentId": "old",
         "timestamp": "2026-09-20T23:23:26Z", "summary": "Keep the screenshots",
         "tokensBefore": 170000, "firstKeptEntryId": "old",
         "usage": {"input": 129050, "output": 1964, "cacheRead": 0}},
    ]
    path = tmp_path / "pi.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    session, messages = load_pi_session(path)
    compaction = messages[-1]
    completed = compaction["time"]["completed"] / 1000
    receipt = {"session_id": "pi", "request_id": "compact-request", "logged_at_s": completed - 1,
               "prompt_tokens": 129050, "completion_tokens": 1964, "request_elapsed_s": 180}
    joined = _join_session(None, "pi", [receipt], [], session=session, messages=messages)
    assert joined["turns"][-1]["receipt"] == receipt
    assert joined["turns"][-1]["message"]["_request_kind"] == "compaction"
    assert joined["turns"][-1]["message"]["_compaction"]["first_kept_entry_id"] == "old"
    assert joined["turns"][-1]["message"]["time"]["created"] == (completed - 181) * 1000
    assert _match_receipt(compaction, [receipt, dict(receipt)], set()) is None
    assert _match_receipt(compaction, [{**receipt, "prompt_tokens": 40}], set()) is None
    assert _match_receipt(compaction, [{**receipt, "logged_at_s": completed - 10}], set()) is None


def test_incident_scope_keeps_cross_boundary_request_and_does_not_mutate_session():
    joined = {"turns": [
        {"kind": "assistant", "turn": 8, "message": {"time": {"completed": 1000}},
         "receipt": {"logged_at_s": 1}},
        {"kind": "assistant", "turn": 9, "message": {"time": {"created": 1000}},
         "receipt": {"logged_at_s": 5, "request_elapsed_s": 4}},
    ], "unmatched_receipts": [{"logged_at_s": 1}, {"logged_at_s": 6}]}
    scoped = scope_since(joined, "1970-01-01T00:00:03Z")
    assert len(scoped["turns"]) == 1
    assert scoped["turns"][0]["turn"] == 1
    assert scoped["turns"][0]["session_turn"] == 9
    assert joined["turns"][1]["turn"] == 9
    assert scoped["unmatched_receipts"] == [{"logged_at_s": 6}]
    with pytest.raises(ValueError, match="timezone"):
        scope_since(joined, "2026-09-20T23:00:00")


def test_diagnostics_do_not_treat_absent_routes_or_late_pressure_as_a_cause():
    receipt = {"request_vision_images": 2, "completion_tokens": 100, "decode_elapsed_s": 5,
               "verify_time_s": 4, "draft_time_s": .8, "accepted_by_depth": [30, 20, 4],
               "drafted_by_depth": [40, 40, 8], "logged_at_s": 10, "request_elapsed_s": 7}
    samples = [{"ev": "s", "ts": i, "gen": (i - 4) * 10, "vt": (i - 4) * .4,
                "ctx": 1000 + i} for i in range(4, 10)]
    result = request_diagnostics(receipt, samples, [{"ev": "system", "ts": 100, "memory_pressure_level": 4}])
    assert result["depth_cycle_shares"] == [0, .8, .2]
    assert result["cost_ms_per_token"]["verify"] == 40
    assert result["recorded_verify_routes"] == []
    assert result["system_sample_count"] == 0
    assert result["slowest_windows"][0]["tok_s"] == 10
    assert result["slowest_windows"][0]["vt_ms_per_token"] == 40
    assert result["slowest_windows"][0]["system"] is None
    assert any("Image context" in text for text in result["warnings"])
    result = request_diagnostics(receipt, samples, [{"ev": "system", "ts": 8, "memory_pressure_level": 2}])
    assert result["system_sample_count"] == 1
    assert result["intervals"][-1]["system_age_s"] == 1


def test_system_record_preserves_pressure_source_and_excludes_prompt_content():
    result = system_sample({"ts": 20, "in_flight": [{"request_id": "r", "prompt_preview": "private"}],
                            "memory_pressure_level": 2, "memory_pressure_source": "allocator",
                            "mem": {"phys_footprint_bytes": 50},
                            "session_bank": {"entries": 2, "cold_tier": {"restore_hits": 1}}},
                           {"ok": False, "error": "not available"})
    assert result["request_ids"] == ["r"]
    assert result["memory_pressure_source"] == "allocator"
    assert result["thermal"]["ok"] is False
    assert "private" not in json.dumps(result)


def test_request_inspector_omits_image_payloads_in_tool_results():
    from mtplx.commands.trace_report import _sec_inspector

    joined = {"session": {}, "turns": [{"kind": "assistant", "turn": 1,
              "message": {"_parts": [{"type": "tool", "tool": "read",
                                       "state": {"input": {"path": "screenshot.png"},
                                                 "output": [{"type": "image", "data": "SECRET_IMAGE_BYTES"}]}}]}}]}
    html = _sec_inspector(joined, None, None)
    assert "SECRET_IMAGE_BYTES" not in html
    assert "screenshot.png" in html
    assert "inspect-diagnosis" in html


def test_compiled_share_uses_bank_dispatches_not_copy_verifies_or_route_samples():
    receipt = {"completion_tokens": 120, "decode_elapsed_s": 3,
               "drafted_by_depth": [40, 40, 35], "accepted_by_depth": [30, 25, 20],
               "verify_calls": 48, "verify_time_s": 2.4,
               "compiled_verify": {"calls": 40, "compiled_calls": 35,
                                   "traces": 2, "fixed_m4_capacity_transitions": 1}}
    result = request_diagnostics(receipt, [{"ev": "s", "ts": 1, "route": "compiled_bank"}])
    assert result["verifier"]["compiled_share"] == 35 / 40
    assert result["economics"]["verify_ms_per_round"] == pytest.approx(50)
    assert result["economics"]["cycle_ms"] == pytest.approx(75)
    assert result["economics"]["tokens_per_cycle"] == 3
    assert request_diagnostics({}, [])["verifier"]["compiled_share"] is None
    receipt["compiled_verify"].pop("compiled_calls")
    assert request_diagnostics(receipt, [])["verifier"]["compiled_share"] is None
