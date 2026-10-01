"""The one tensor-unit (NAX) detector.

Two detectors used to answer "does this Mac have Metal 4 tensor units":
``nax_verify`` matched the GPU architecture string against the exact prefix
``applegpu_g17`` and honored the rehearsal switch; ``qsa_indexer_select``
parsed the generation number (17 or newer, 18 for the phone class) and
ignored the switch. A generation-18 GPU would have received the Flash-Next
prefill lanes and lost every 27B verify lane, and an M5 could not rehearse
the Flash-Next prefill path an M1 to M4 runs.

Two questions, two functions:

* ``nax_hardware_available()`` is the hardware truth (GPU generation and
  the macOS floor), memoized for the life of the process. Code that mirrors
  what MLX itself does on this machine (the TF32 numerics of its float32
  GEMM) must read this one: MLX does not know about our rehearsal switch.
* ``nax_available()`` is the route gate every lane reads: the hardware
  truth unless ``MTPLX_FORCE_GPU_FAMILY_FALLBACK=1`` asks this Mac to take
  the code path an M1 to M4 takes. The switch is read per call.

The parser mirrors MLX 0.31.2 and 0.32.2 ``is_nax_available``: macOS 26.2 or
newer; the last two digits of the architecture are the generation and the
final letter the device class; phone class ``p`` needs generation 18, every
other class 17. Unknown formats fail closed.
"""

from __future__ import annotations

import os
import platform
import re
from functools import lru_cache

import mlx.core as mx

FORCE_FALLBACK_ENV = "MTPLX_FORCE_GPU_FAMILY_FALLBACK"
_TRUTHY = {"1", "true", "on", "yes"}


def nax_available_for_platform(
    macos_version: str | None, architecture: str | None
) -> bool:
    """Pure parser: is this (macOS version, GPU architecture) a NAX machine?"""

    if not isinstance(macos_version, str) or not isinstance(architecture, str):
        return False
    version_match = re.match(r"^\s*(\d+)\.(\d+)(?:\.\d+)?(?:\D.*)?$", macos_version)
    if version_match is None:
        return False
    version = (int(version_match.group(1)), int(version_match.group(2)))
    if version < (26, 2):
        return False

    arch_match = re.search(r"(\d{2})([pgsd])$", architecture.lower())
    if arch_match is None:
        return False
    generation = int(arch_match.group(1))
    suffix = arch_match.group(2)
    return generation >= (18 if suffix == "p" else 17)


def gpu_architecture() -> str | None:
    """The Metal GPU's architecture string, for example ``applegpu_g17s``."""

    info = None
    device_info = getattr(mx, "device_info", None)
    if callable(device_info):
        try:
            # Capability is about the Metal GPU even if a CPU parity test has
            # temporarily changed MLX's default device.
            info = device_info(mx.gpu)
        except TypeError:
            try:
                info = device_info()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                info = None
        except (AttributeError, RuntimeError, ValueError):
            info = None
    if not isinstance(info, dict):
        # Compatibility fallback for an MLX release without mx.device_info.
        metal_device_info = getattr(getattr(mx, "metal", None), "device_info", None)
        if callable(metal_device_info):
            try:
                info = metal_device_info()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                info = None
    architecture = info.get("architecture") if isinstance(info, dict) else None
    return architecture if isinstance(architecture, str) else None


@lru_cache(maxsize=1)
def nax_hardware_available() -> bool:
    """GPU generation + macOS floor. Immutable for the process: memoized."""

    try:
        macos_version = platform.mac_ver()[0]
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return nax_available_for_platform(macos_version, gpu_architecture())


def gpu_family_fallback_forced() -> bool:
    """The rehearsal switch: pretend this GPU has no tensor units.

    Read per call. Memoizing it froze the value at the first probe, so
    setting the switch after import (profiles, tests) silently did nothing.
    """

    return str(os.environ.get(FORCE_FALLBACK_ENV, "")).strip().lower() in _TRUTHY


def nax_available() -> bool:
    """The route gate: tensor units present AND not rehearsing their absence."""

    if gpu_family_fallback_forced():
        return False
    return nax_hardware_available()


nax_available.cache_clear = nax_hardware_available.cache_clear  # type: ignore[attr-defined]
