"""MLX process settings MTPLX needs in place before the Metal device exists.

MLX reads ``MLX_MAX_MB_PER_BUFFER`` once, when it creates its Metal device, and
from then on closes the open command buffer every time the distinct buffers the
buffer has touched add up to more than that many MiB (50 on a Max).  The rule
is there to bound the temporaries one command buffer can pin.  It also counts
persistent state.  A sparse-attention layer's key bank and value bank are
135 MB each at a 128K context, so every write into them and every gather out
of them closes a command buffer: about four per layer, twelve layers, every
decode round.

Receipt (2026-09-20, Flash-Next, M5 Max, 128K context, same tree, same night):
verify forward 45 to 52 ms per round with MLX's 50 MiB rule, 30.0 ms with the
rule lifted (27.5 ms at 4K); decode 45.6 to 54.3 tok/s against 65.9.  A Metal
timeline with one op per command buffer shows the cost as one stalled
submission of about 1 ms per sparse-attention layer per round.  At 4K and 16K
the banks are under the limit and nothing changes.

It is NOT on by default, because prefill temporaries are exactly what the rule
bounds.  Same night, founder's cell order (4K, 16K, 64K, 128K in one process),
rule lifted: process peak 103.7 GB at 16K against 92.0 GB, 103.9 GB at 64K
against 95.6 GB, and the decode gain at 128K shrinks to 45.6 -> 49.4 tok/s
once the process is that close to its memory limit.  With 1,024 MiB: 128K
decode 59.2 tok/s from a fresh process, peak 103.2 GB.  MLX reads the value
once per process, so it cannot be raised for decode and lowered for prefill;
until it can, this is an operator's choice for decode-heavy long-context
serving on a Mac with memory to spare, and the engine leaves MLX's default
alone.

``MTPLX_MLX_COMMAND_BUFFER_MB`` sets it: a number of MiB (``mlx``, ``0`` or
``off`` and an unset variable all leave MLX's own default alone).  A value the operator set for
MLX directly (``MLX_MAX_MB_PER_BUFFER``) always wins.  This module imports
nothing from MLX and must stay that way: it runs from ``mtplx/__init__.py``
so that it is in place before the first GPU call of the process.
"""

from __future__ import annotations

import os

MLX_ENV = "MLX_MAX_MB_PER_BUFFER"
OVERRIDE_ENV = "MTPLX_MLX_COMMAND_BUFFER_MB"
#: None leaves MLX's own default in place (see the receipts above).
DEFAULT_COMMAND_BUFFER_MB: int | None = None

_LEAVE_ALONE = frozenset({"0", "off", "false", "no", "mlx", "default"})

#: What this process was told, for /health: MLX has no getter for the value.
_APPLIED: dict[str, object] = {"value_mb": None, "source": "unset"}


def resolve_command_buffer_mb(env=None) -> tuple[int | None, str]:
    """``(MiB or None, source)`` for this environment; None leaves MLX alone."""

    source = os.environ if env is None else env
    direct = str(source.get(MLX_ENV, "")).strip()
    if direct:
        try:
            return int(direct), "operator:" + MLX_ENV
        except ValueError:
            return None, "operator:" + MLX_ENV + ":unparsed"
    raw = str(source.get(OVERRIDE_ENV, "")).strip().lower()
    if raw in _LEAVE_ALONE and raw:
        return None, "override:mlx_default"
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return DEFAULT_COMMAND_BUFFER_MB, "default:unparsed_override"
        if value <= 0:
            return None, "override:mlx_default"
        return value, "override:" + OVERRIDE_ENV
    if DEFAULT_COMMAND_BUFFER_MB is None:
        return None, "default:mlx_default"
    return DEFAULT_COMMAND_BUFFER_MB, "default"


def apply_mlx_process_defaults(env=None) -> dict[str, object]:
    """Put the command-buffer bound in the environment MLX will read.

    Idempotent, and it never overwrites a value the operator gave MLX.
    """

    target = os.environ if env is None else env
    value, source = resolve_command_buffer_mb(target)
    if value is not None and not source.startswith("operator:"):
        target[MLX_ENV] = str(int(value))
    if env is None:
        _APPLIED.update({"value_mb": value, "source": source})
    return {"value_mb": value, "source": source}


def applied_command_buffer_mb() -> dict[str, object]:
    return dict(_APPLIED)


__all__ = [
    "DEFAULT_COMMAND_BUFFER_MB",
    "MLX_ENV",
    "OVERRIDE_ENV",
    "applied_command_buffer_mb",
    "apply_mlx_process_defaults",
    "resolve_command_buffer_mb",
]
