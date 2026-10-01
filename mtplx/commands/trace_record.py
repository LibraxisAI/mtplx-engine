"""Capture lightweight daemon/system evidence without generating or changing settings."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _get(port: int, endpoint: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{endpoint}", timeout=5) as response:
        return json.load(response)


def system_sample(snapshot: dict, thermal: dict) -> dict:
    bank = snapshot.get("session_bank") or {}
    cold = bank.get("cold_tier") or {}
    return {"ev": "system", "ts": snapshot.get("ts", time.time()),
            "request_ids": [r.get("request_id") for r in snapshot.get("in_flight", [])],
            "mem": snapshot.get("mem"), "thermal": thermal,
            "memory_pressure_level": snapshot.get("memory_pressure_level"),
            "memory_pressure_source": snapshot.get("memory_pressure_source"),
            "allocator_fraction": snapshot.get("allocator_fraction"),
            "memory_guard_events": snapshot.get("memory_guard_events"),
            "bank": {k: bank.get(k) for k in ("entries", "total_nbytes", "effective_max_bytes", "last_miss_reason")},
            "cold_tier": {k: cold.get(k) for k in (
                "writes_completed", "restore_hits", "restore_misses", "entries_evicted",
                "writer_foreground_pauses", "writer_foreground_pause_s", "writer_backlog_bytes")}}


def cmd_trace_record(args) -> int:
    if not all(math.isfinite(n) and n > 0 for n in (args.duration, args.interval)) or args.interval < 0.5:
        raise ValueError("Use a positive duration and an interval of at least 0.5 seconds")
    port = args.port or 8000
    path = Path(args.out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Evidence is append-only in spirit: refuse to replace an earlier run.
    with path.open("x", encoding="utf-8") as handle:
        def write(value):
            handle.write(json.dumps(value) + "\n")
            handle.flush()

        write({"ev": "runtime", "ts": time.time(), "port": port,
               "health": _get(port, "/health")})
        deadline = time.monotonic() + args.duration
        captured, errors = 0, 0
        while time.monotonic() < deadline:
            started = time.monotonic()
            try:
                snapshot = _get(port, "/v1/mtplx/snapshot")
                try:
                    thermal = _get(port, "/v1/mtplx/thermal/status")
                except (OSError, ValueError, urllib.error.URLError) as exc:
                    thermal = {"ok": False, "error": str(exc)}
                sample = system_sample(snapshot, thermal)
                sysctl = shutil.which("sysctl") if sys.platform == "darwin" else None
                if sysctl:
                    swap = subprocess.run([sysctl, "-n", "vm.swapusage"], capture_output=True,
                                          text=True, timeout=2, check=False)
                    sample["host_swap"] = {"raw": swap.stdout.strip(), "error": swap.stderr.strip(),
                                           "exit_code": swap.returncode}
                write(sample)
                captured += 1
            except (OSError, ValueError, urllib.error.URLError) as exc:
                errors += 1
                write({"ev": "observation_error", "ts": time.time(), "error": str(exc)})
            time.sleep(max(0, min(deadline - time.monotonic(), args.interval - (time.monotonic() - started))))
    print(f"wrote {path}: {captured} samples, {errors} observation errors")
    return 0 if captured and not errors else 1
