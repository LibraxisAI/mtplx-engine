"""Read-only request attribution. Missing telemetry is never a zero or a cause."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from .trace_metrics import mtp_economics, sample_intervals


def scope_since(joined: dict, value: str | None) -> dict:
    if not value:
        return joined
    stamp = datetime.datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("--since needs a timezone, for example 2026-09-20T23:19:00Z")
    since = stamp.timestamp()
    turns = []
    number = 0
    for turn in joined["turns"]:
        receipt = turn.get("receipt") or {}
        time = turn["message"].get("time") or {}
        end = receipt.get("logged_at_s") or (time.get("completed") or time.get("created") or 0) / 1000
        if end < since:
            continue
        turn = dict(turn)
        if turn["kind"] == "assistant":
            number += 1
            turn.update(session_turn=turn["turn"], turn=number)
        turns.append(turn)
    return {**joined, "turns": turns, "since": value,
            "unmatched_receipts": [r for r in joined.get("unmatched_receipts", [])
                                   if float(r.get("logged_at_s") or 0) >= since]}


def load_system_log(path: str | None) -> list[dict]:
    if not path:
        return []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def request_diagnostics(receipt: dict, flight: list[dict], system: list[dict] = ()) -> dict:
    economics = mtp_economics(receipt)
    samples = [e for e in flight if e.get("ev") == "s"]
    intervals = sample_intervals(flight)
    drafted = receipt.get("drafted_by_depth") or []
    cycles = drafted[0] if drafted else 0
    depth_mix = [(n - (drafted[i + 1] if i + 1 < len(drafted) else 0)) / cycles
                 for i, n in enumerate(drafted)] if cycles else []
    end = receipt.get("logged_at_s") or 0
    start = end - (receipt.get("request_elapsed_s") or 0)
    system = [s for s in system if s.get("ev") == "system" and start <= s.get("ts", 0) <= end]
    for interval in intervals:
        recent = [s for s in system if 0 <= interval["end_s"] - s["ts"] <= 5]
        if recent:
            observed = max(recent, key=lambda s: s["ts"])
            interval["system"] = observed
            interval["system_age_s"] = interval["end_s"] - observed["ts"]
    routes = sorted({e["route"] for e in samples if e.get("route")})
    tokens = receipt.get("completion_tokens") or 0
    costs = {name: 1000 * receipt[key] / tokens if tokens and receipt.get(key) is not None else None
             for name, key in (("verify", "verify_time_s"), ("draft", "draft_time_s"),
                               ("accept", "accept_time_s"))}
    bank = receipt.get("compiled_verify") or {}
    bank_calls, compiled_calls = bank.get("calls"), bank.get("compiled_calls")
    compiled_share = (compiled_calls / bank_calls
                      if isinstance(bank_calls, (int, float)) and bank_calls > 0
                      and isinstance(compiled_calls, (int, float))
                      and 0 <= compiled_calls <= bank_calls else None)
    verifier = {"bank_calls": bank_calls, "compiled_calls": compiled_calls,
                "compiled_share": compiled_share,
                "admission": receipt.get("fixed_m4_admission"),
                "capacity_transitions": bank.get("fixed_m4_capacity_transitions"),
                "traces": bank.get("traces")}
    images = receipt.get("request_vision_images")
    warnings = []
    if images and not routes:
        warnings.append("Image context present; verifier route was not recorded. Missing counters do not mean zero verify work.")
    if not system:
        warnings.append("No contemporaneous system samples: thermal, swap and memory-pressure causation is unmeasured.")
    if any(i["observation_gap"] for i in intervals):
        warnings.append("Flight sampling has gaps; those gaps are not invented zero-speed intervals.")
    # Five-second windows make sustained drops visible without treating one
    # asynchronously published counter delta as an instantaneous causal proof.
    windows, bucket = [], []
    for interval in intervals:
        if interval["observation_gap"] or interval["gen"] is None:
            bucket = []
            continue
        bucket.append(interval)
        duration = sum(i["duration_s"] for i in bucket)
        if duration < 5:
            continue
        generated = sum(i["gen"] for i in bucket)
        window = {"start_s": bucket[0]["start_s"], "end_s": bucket[-1]["end_s"],
                  "decode_offset_s": bucket[0]["start_s"] - samples[0]["ts"],
                  "duration_s": duration, "tok_s": generated / duration,
                  "context_tokens": bucket[-1]["context_tokens"],
                  "active_memory_bytes": bucket[-1]["active_memory_bytes"],
                  "system": bucket[-1].get("system")}
        for key in ("vt", "dt"):
            window[key + "_ms_per_token"] = (
                1000 * sum(i[key] for i in bucket) / generated
                if generated and all(i[key] is not None for i in bucket) else None)
        windows.append(window)
        bucket = []
    return {"prompt_tokens": receipt.get("prompt_tokens"), "end_context_tokens": receipt.get("context_len"),
            "images": images, "image_rows": receipt.get("request_vision_rows"),
            "depth_cycle_shares": depth_mix, "recorded_verify_routes": routes,
            "route_sample_coverage": sum(bool(s.get("route")) for s in samples) / len(samples) if samples else None,
            "cost_ms_per_token": costs, "economics": economics, "verifier": verifier,
            "compared_workload": {k: receipt.get(k) for k in (
                "served_model_id", "resolved_reasoning_effort", "chat_template_profile",
                "effective_temperature", "effective_top_p", "effective_top_k", "request_tool_count")},
            "slowest_windows": sorted(windows, key=lambda w: w["tok_s"])[:5],
            "system_sample_count": len(system), "warnings": warnings, "intervals": intervals}
