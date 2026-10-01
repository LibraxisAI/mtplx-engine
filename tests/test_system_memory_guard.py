"""The desktop's share of memory: the guard that looks past the engine's own limit.

Receipt (2026-09-19): a 129,050-token Pi compaction prompt on a fresh daemon,
128 GB Mac, a game editor and two browsers open. The engine stayed under its
Metal limit the whole time, macOS reported "normal" pressure, and the Mac froze
about a minute into the prefill. These tests pin the three places the kernel's
available-memory figure now acts: the reading itself, prefill admission, and the
guard loop's level.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mtplx.server.openai as srv
import mtplx.system_memory as sm

GIB = 1024**3
RAM = 128 * GIB
LIMIT = 96 * GIB
KV_PER_TOKEN = 24576
AUX_PER_TOKEN = 7872
PER_TOKEN = KV_PER_TOKEN + AUX_PER_TOKEN


def _reading(available_gib: float, total: int = RAM) -> sm.SystemMemory:
    available = int(available_gib * GIB)
    return sm.SystemMemory(
        available_bytes=available,
        total_bytes=total,
        level_percent=int(available * 100 // total),
    )


def _install(monkeypatch, available_gib: float | None, total: int = RAM) -> None:
    monkeypatch.setattr(
        sm,
        "_reader",
        (lambda: None)
        if available_gib is None
        else (lambda: _reading(available_gib, total)),
    )


class TestReading:
    def test_the_real_kernel_answers_on_this_mac(self):
        reading = sm._read_kernel()
        assert reading is not None
        assert reading.total_bytes >= 8 * GIB
        assert 0 <= reading.available_bytes <= reading.total_bytes
        assert reading.available_bytes == reading.total_bytes * reading.level_percent // 100

    def test_kill_switch_reads_unknown(self, monkeypatch):
        _install(monkeypatch, 40)
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_GUARD", "0")
        assert sm.read_system_memory() is None

    def test_a_reader_that_raises_reads_unknown(self, monkeypatch):
        def boom():
            raise OSError("sysctl refused")

        monkeypatch.setattr(sm, "_reader", boom)
        assert sm.read_system_memory() is None

    def test_rehearsal_replaces_the_available_figure_only(self, monkeypatch):
        _install(monkeypatch, 40)
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES", "2G")
        reading = sm.read_system_memory()
        assert reading is not None
        assert reading.available_bytes == 2 * GIB
        assert reading.total_bytes == RAM

    @pytest.mark.parametrize(
        "raw, expected",
        [("1073741824", GIB), ("2G", 2 * GIB), ("2GiB", 2 * GIB), ("512M", 512 * 1024**2),
         ("1.5g", int(1.5 * GIB)), ("", None), ("lots", None)],
    )
    def test_byte_sizes_parse(self, raw, expected):
        assert sm._parse_bytes(raw) == expected


class TestFloorsAndLevel:
    def test_floors_scale_with_the_machine_and_never_vanish(self):
        shed, abort = sm.system_memory_floors(128 * GIB)
        assert abort == int(128 * GIB * 0.025)
        assert shed == 2 * abort
        shed_small, abort_small = sm.system_memory_floors(16 * GIB)
        assert abort_small == 1 * GIB
        assert shed_small == 2 * GIB

    def test_floor_overrides(self, monkeypatch):
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_ABORT_FLOOR_BYTES", "4G")
        assert sm.system_memory_floors(RAM) == (8 * GIB, 4 * GIB)
        monkeypatch.setenv("MTPLX_SYSTEM_MEMORY_SHED_FLOOR_BYTES", "5G")
        assert sm.system_memory_floors(RAM) == (5 * GIB, 4 * GIB)

    def test_level_ladder(self):
        assert sm.system_pressure_level(None) == 1
        assert sm.system_pressure_level(_reading(30)) == 1
        assert sm.system_pressure_level(_reading(6.3)) == 2
        assert sm.system_pressure_level(_reading(3.1)) == 4

    def test_unknown_never_reports_a_shortfall(self):
        assert sm.admission_shortfall_bytes(None, growth_bytes=50 * GIB, reclaimable_bytes=0) == 0

    def test_shortfall_counts_the_engines_own_pool(self):
        shed, _abort = sm.system_memory_floors(RAM)
        reading = _reading(10)
        assert sm.admission_shortfall_bytes(reading, growth_bytes=8 * GIB, reclaimable_bytes=0) == (
            8 * GIB + shed - 10 * GIB
        )
        assert sm.admission_shortfall_bytes(reading, growth_bytes=8 * GIB, reclaimable_bytes=6 * GIB) == 0


class _Bank:
    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)
        self.shrink_calls: list[tuple[int, str, bool]] = []
        self.cleared_sessions: list[str | None] = []

    @property
    def total_nbytes(self):
        return self.nbytes

    def longest_prefix(self, token_ids):
        return None

    def longest_shared_prefix_tokens(self, token_ids, *, session_id=None):
        return 0

    def clear(self, *, session_id=None):
        self.cleared_sessions.append(session_id)
        return 0

    def touch_sessions(self, session_ids):
        pass

    def shrink_to_bytes(self, target_bytes, *, reason="", protect_active=False):
        self.shrink_calls.append((int(target_bytes), reason, protect_active))
        evicted = 1 if self.nbytes > int(target_bytes) else 0
        self.nbytes = min(self.nbytes, int(target_bytes))
        return evicted


def _state(allow_swap: bool = False):
    return SimpleNamespace(
        metal_memory_caps={"memory_limit_bytes": LIMIT},
        memory_plan=SimpleNamespace(
            kv_bytes_per_token_effective=KV_PER_TOKEN,
            aux_bytes_per_token=AUX_PER_TOKEN,
            prefill_transient_bytes_per_token=0,
        ),
        dashboard=SimpleNamespace(),
        allow_swap=allow_swap,
    )


def _pin_engine(monkeypatch, *, active_gib: float, cache_gib: float) -> None:
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": int(active_gib * GIB),
            "cache_memory_bytes": int(cache_gib * GIB),
        },
    )
    monkeypatch.setattr(srv, "_record_guard_event", lambda state, payload: None)


PROMPT = list(range(129_050))


class TestAdmission:
    """The engine is healthy against its own limit in every case here."""

    def test_a_quiet_desktop_admits_the_saturday_prompt_untouched(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 40)
        bank = _Bank(0)
        assert srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=bank, session_id="pi"
        ) is None
        assert bank.shrink_calls == []

    def test_an_unreadable_machine_changes_nothing(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, None)
        assert srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=_Bank(0), session_id="pi"
        ) is None

    def test_a_full_desktop_refuses_before_prefill(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 9)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=_Bank(0), session_id="pi"
        )
        assert receipt is not None
        assert receipt["refused"] is True
        assert receipt["refusal_reason"] == "system_memory_short_after_reclamation"
        growth = len(PROMPT) * PER_TOKEN * 2 + srv_transients()
        shed_floor, _ = sm.system_memory_floors(RAM)
        assert receipt["system_shortfall_bytes"] == growth + shed_floor - (9 + 1) * GIB
        detail = srv._prefill_admission_refusal(_state(), receipt).detail
        assert "other apps" in detail and "9.0 GiB free" in detail and "--allow-swap" in detail

    def test_the_bank_is_shed_for_the_desktop_too(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 9)
        bank = _Bank(7 * GIB)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=bank, session_id="pi"
        )
        assert receipt is not None
        assert bank.cleared_sessions == ["pi"]
        assert bank.shrink_calls and bank.shrink_calls[0][1] == "prefill_admission"

    def test_a_desktop_that_recovers_after_the_shed_is_admitted(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        readings = iter([_reading(9), _reading(30)])
        last = [_reading(30)]

        def reader():
            try:
                last[0] = next(readings)
            except StopIteration:
                pass
            return last[0]

        monkeypatch.setattr(sm, "_reader", reader)
        receipt = srv._prefill_admission_shed(
            _state(), prompt_ids=PROMPT, session_bank=_Bank(7 * GIB), session_id="pi"
        )
        assert receipt is not None
        assert receipt.get("refused") is not True
        assert receipt["system_shortfall_bytes_after"] == 0

    def test_allow_swap_keeps_the_operators_choice(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 9)
        receipt = srv._prefill_admission_shed(
            _state(allow_swap=True), prompt_ids=PROMPT, session_bank=_Bank(0), session_id="pi"
        )
        assert receipt is not None
        assert receipt.get("refused") is not True

    def test_a_short_prompt_is_never_refused_by_a_full_desktop(self, monkeypatch):
        _pin_engine(monkeypatch, active_gib=78, cache_gib=1)
        _install(monkeypatch, 1)
        assert srv._prefill_admission_shed(
            _state(), prompt_ids=list(range(1024)), session_bank=_Bank(0), session_id="pi"
        ) is None


def srv_transients() -> int:
    from mtplx.memory_plan import RUNTIME_TRANSIENTS_BYTES

    return int(RUNTIME_TRANSIENTS_BYTES)


class _LoopBank:
    def __init__(self, total, max_bytes):
        self.total_nbytes = total
        self.max_bytes = max_bytes
        self.calls = []

    def shrink_to_bytes(self, target, *, reason, protect_active=False):
        self.calls.append((target, reason))
        self.total_nbytes = min(self.total_nbytes, target)
        return 1


def _loop_state(bank):
    return SimpleNamespace(
        sessions=SimpleNamespace(bank=bank),
        dashboard=SimpleNamespace(last_memory_pressure_level=0),
    )


def _run_loop(state, monkeypatch, *, seconds: float):
    import asyncio

    monkeypatch.setattr(srv, "_memory_pressure_level", lambda: 1)

    async def run():
        task = asyncio.ensure_future(srv._memory_pressure_loop(state, interval_s=3600))
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


class TestGuardLoop:
    """macOS says normal and the allocator is far from its limit in every case."""

    def test_a_short_desktop_raises_the_level_and_names_itself(self, monkeypatch):
        _install(monkeypatch, 5)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert state.dashboard.last_memory_pressure_level == 2
        assert state.dashboard.last_memory_pressure_source == "system_available"
        assert state.dashboard.last_system_available_bytes == 5 * GIB
        assert bank.calls == [(4 * GIB, "memory_pressure_warning")]
        event = list(state.dashboard.memory_guard_events)[-1]
        assert event["level_source"] == "system_available"
        assert event["system_available_bytes"] == 5 * GIB

    def test_under_the_abort_floor_is_critical_and_empties_the_bank(self, monkeypatch):
        _install(monkeypatch, 2)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert state.dashboard.last_memory_pressure_level == 4
        assert bank.calls == [(0, "memory_pressure_critical")]

    def test_a_busy_engine_under_the_abort_floor_is_aborted_within_seconds(self, monkeypatch):
        _install(monkeypatch, 2)
        monkeypatch.setattr(srv, "_SYSTEM_SHORT_INTERVAL_S", 0.01)
        monkeypatch.setattr(srv, "_engine_busy_signal", lambda state: True)
        state = _loop_state(_LoopBank(total=0, max_bytes=8 * GIB))
        _run_loop(state, monkeypatch, seconds=0.3)
        assert srv._pressure_abort_requested(state)
        actions = [e["action"] for e in state.dashboard.memory_guard_events]
        assert "pressure_abort_armed" in actions

    def test_a_roomy_desktop_keeps_the_slow_tick_and_does_nothing(self, monkeypatch):
        _install(monkeypatch, 40)
        bank = _LoopBank(total=8 * GIB, max_bytes=8 * GIB)
        state = _loop_state(bank)
        _run_loop(state, monkeypatch, seconds=0.05)
        assert state.dashboard.last_memory_pressure_level == 1
        assert state.dashboard.last_system_available_bytes == 40 * GIB
        assert bank.calls == []
