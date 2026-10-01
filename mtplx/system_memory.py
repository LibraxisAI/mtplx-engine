"""What the rest of the Mac has left: the kernel's own available-memory reading.

The allocator guard measures the engine against its own Metal limit. That limit
is sized from total RAM when the daemon starts, so it says nothing about the
other apps on the desktop. A long cold prefill adds 10 to 15 GiB on top of the
resident weights. When an editor and two browsers already hold the rest, macOS
compresses and swaps the desktop until the UI stops answering, and nothing in
the engine notices: the engine itself is still under its limit.

Receipt (2026-09-19): a 129,050-token Pi compaction prompt, fresh daemon, 128 GB
Mac with a game editor open. The Mac froze about a minute into the prefill and
needed a hard power-off. The same prefill on a quiet desktop wires about 100 GB
and leaves the kernel reporting 23 to 25 percent available.

``kern.memorystatus_level`` is the kernel's percentage of memory it can still
hand out (free, purgeable and reclaimable file cache). It is the number behind
``memory_pressure -Q`` and it moves before ``kern.memorystatus_vm_pressure_level``
leaves "normal". Everything here degrades to "unknown" and never to a refusal:
a guard that cannot read the machine takes no action.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
from dataclasses import dataclass
from typing import Callable

GIB = 1024**3

# Below the abort floor the desktop is one allocation away from a swap storm:
# 2.5 percent of RAM, never under 1 GiB (16 GB Mac: 1 GiB, 128 GB Mac: 3.2 GiB).
# The shed floor is twice that: the engine gives back its own reusable memory
# (allocator pool, idle session snapshots) before anyone is refused.
_ABORT_FLOOR_FRACTION = 0.025
_ABORT_FLOOR_MIN_BYTES = 1 * GIB
_SHED_FLOOR_MULTIPLE = 2


@dataclass(frozen=True)
class SystemMemory:
    """One reading of the kernel's available-memory accounting."""

    available_bytes: int
    total_bytes: int
    level_percent: int


def system_memory_guard_enabled() -> bool:
    raw = os.environ.get("MTPLX_SYSTEM_MEMORY_GUARD", "1").strip().lower()
    return raw not in {"0", "off", "false", "no"}


@functools.cache
def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(ctypes.util.find_library("c"))


def _sysctl_int(name: bytes, width: int) -> int | None:
    value = ctypes.c_uint64(0) if width == 8 else ctypes.c_int32(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    rc = _libc().sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0)
    if rc != 0:
        return None
    return int(value.value)


def _read_kernel() -> SystemMemory | None:
    level = _sysctl_int(b"kern.memorystatus_level", 4)
    total = _sysctl_int(b"hw.memsize", 8)
    if level is None or total is None or total <= 0 or not 0 <= level <= 100:
        return None
    return SystemMemory(
        available_bytes=int(total * level // 100),
        total_bytes=int(total),
        level_percent=int(level),
    )


# Swappable for tests and for the rehearsal switch below; production reads the
# kernel.
_reader: Callable[[], SystemMemory | None] = _read_kernel


def _parse_bytes(raw: str) -> int | None:
    text = raw.strip().upper()
    if not text:
        return None
    scale = 1
    for suffix, factor in (("K", 1024), ("M", 1024**2), ("G", GIB)):
        if text.endswith(suffix + "IB"):
            text, scale = text[:-3], factor
            break
        if text.endswith(suffix + "B"):
            text, scale = text[:-2], factor
            break
        if text.endswith(suffix):
            text, scale = text[:-1], factor
            break
    try:
        return int(float(text) * scale)
    except ValueError:
        return None


def read_system_memory() -> SystemMemory | None:
    """The kernel's reading, or None when it cannot be read or the guard is off.

    ``MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES`` replaces the available
    figure with a fixed value so the shed, refuse and abort paths can be
    exercised on a machine that is not actually short of memory.
    """

    if not system_memory_guard_enabled():
        return None
    try:
        reading = _reader()
    except Exception:
        return None
    if reading is None:
        return None
    rehearsal = _parse_bytes(
        os.environ.get("MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES", "")
    )
    if rehearsal is not None:
        available = max(0, min(int(rehearsal), reading.total_bytes))
        return SystemMemory(
            available_bytes=available,
            total_bytes=reading.total_bytes,
            level_percent=int(available * 100 // reading.total_bytes),
        )
    return reading


def system_memory_floors(total_bytes: int) -> tuple[int, int]:
    """(shed_floor_bytes, abort_floor_bytes) for a machine of this size."""

    abort = _parse_bytes(os.environ.get("MTPLX_SYSTEM_MEMORY_ABORT_FLOOR_BYTES", ""))
    if abort is None or abort <= 0:
        abort = max(_ABORT_FLOOR_MIN_BYTES, int(total_bytes * _ABORT_FLOOR_FRACTION))
    shed = _parse_bytes(os.environ.get("MTPLX_SYSTEM_MEMORY_SHED_FLOOR_BYTES", ""))
    if shed is None or shed <= 0:
        shed = abort * _SHED_FLOOR_MULTIPLE
    return int(max(shed, abort)), int(abort)


def system_pressure_level(reading: SystemMemory | None) -> int:
    """Map a reading onto the guard's scale: 1 normal, 2 warning, 4 critical."""

    if reading is None:
        return 1
    shed_floor, abort_floor = system_memory_floors(reading.total_bytes)
    if reading.available_bytes < abort_floor:
        return 4
    if reading.available_bytes < shed_floor:
        return 2
    return 1


def admission_shortfall_bytes(
    reading: SystemMemory | None, *, growth_bytes: int, reclaimable_bytes: int
) -> int:
    """How far a prefill's growth would push the desktop under the shed floor.

    ``growth_bytes`` is what the prefill will add; ``reclaimable_bytes`` is the
    engine's own allocator pool, which the growth reuses before it asks the
    system for anything. Zero means the request fits with the floor intact, and
    an unreadable machine never reports a shortfall.
    """

    if reading is None:
        return 0
    shed_floor, _abort_floor = system_memory_floors(reading.total_bytes)
    have = int(reading.available_bytes) + max(0, int(reclaimable_bytes))
    return max(0, int(growth_bytes) + shed_floor - have)


__all__ = [
    "SystemMemory",
    "admission_shortfall_bytes",
    "read_system_memory",
    "system_memory_floors",
    "system_memory_guard_enabled",
    "system_pressure_level",
]
