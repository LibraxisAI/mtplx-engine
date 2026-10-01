"""`mtplx doctor` carries the app's last failed start (issue #504).

The report on #504 was a screenshot of a one-line banner that cut the reason
off. The daemon's output lived only in the app's memory, so there was nothing
to attach. The app now writes that output to
``~/.mtplx/logs/last-failed-start.log`` when a start fails, and the bug report
template already asks for ``mtplx doctor --json``, so this is what makes a
"will not start" report carry its own cause.
"""

from __future__ import annotations

import os
import time

from mtplx import diagnostics

REPORT = """MTPLX failed start
when: 2026-09-17T11:40:02Z
app: 2.11.3 (2011050)
macos: Version 27.2 (Build 26C5045f)
reason: daemon exited before /health became ready
--- daemon output (last 3 of 3 lines) ---
11:40:01.100 [system] launched python -m mtplx.server.openai --api-key ******
11:40:01.900 [stderr] libc++abi: terminating due to uncaught exception of type std::runtime_error
11:40:01.901 [stderr] [metal::Device] Unable to build metal library from source
"""


def test_no_report_is_a_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_START_FAILURE_REPORT", str(tmp_path / "absent.log"))

    check = diagnostics.last_failed_start_check()

    assert check.id == "app.last_failed_start"
    assert check.status == "pass"
    assert check.observed is None


def test_a_recent_report_is_a_warning_that_carries_the_output(tmp_path, monkeypatch):
    path = tmp_path / "last-failed-start.log"
    path.write_text(REPORT)
    monkeypatch.setenv("MTPLX_START_FAILURE_REPORT", str(path))

    check = diagnostics.last_failed_start_check()

    assert check.status == "warn" and check.severity == "warning"
    assert check.observed["path"] == str(path)
    assert check.observed["reason"] == "daemon exited before /health became ready"
    assert check.observed["macos"] == "Version 27.2 (Build 26C5045f)"
    # The line the banner cut off.
    assert check.observed["tail"][-1].endswith(
        "[metal::Device] Unable to build metal library from source"
    )
    assert check.observed["age_hours"] < 1
    # The masked launch line stays masked; doctor adds nothing to it.
    assert "******" in "\n".join(check.observed["tail"])


def test_the_tail_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "last-failed-start.log"
    path.write_text(REPORT + "".join(f"12:00:00.000 [stderr] line {i}\n" for i in range(500)))
    monkeypatch.setenv("MTPLX_START_FAILURE_REPORT", str(path))

    check = diagnostics.last_failed_start_check()

    assert len(check.observed["tail"]) == diagnostics.FAILED_START_TAIL_LINES
    assert check.observed["tail"][-1].endswith("line 499")


def test_an_old_report_is_not_todays_problem(tmp_path, monkeypatch):
    path = tmp_path / "last-failed-start.log"
    path.write_text(REPORT)
    a_month_ago = time.time() - 30 * 24 * 3600
    os.utime(path, (a_month_ago, a_month_ago))
    monkeypatch.setenv("MTPLX_START_FAILURE_REPORT", str(path))

    check = diagnostics.last_failed_start_check()

    assert check.status == "pass"
    assert check.observed == {"path": str(path), "stale_days": 30}


def test_doctor_includes_the_check(tmp_path, monkeypatch):
    path = tmp_path / "last-failed-start.log"
    path.write_text(REPORT)
    monkeypatch.setenv("MTPLX_START_FAILURE_REPORT", str(path))

    _host, checks = diagnostics.build_diagnostic_checks(
        mlx_info={"mlx": "0.32.2"}, thermal_control={"available": False}
    )

    by_id = {check.id: check for check in checks}
    assert by_id["app.last_failed_start"].status == "warn"
