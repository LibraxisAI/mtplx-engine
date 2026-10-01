"""MTPLX: native Qwen3.6 MTP experiments on MLX."""

from __future__ import annotations

from typing import Any

from .mlx_process_env import apply_mlx_process_defaults
from .version import DISPLAY_VERSION, __version__

# Before anything in this package can reach the GPU: MLX reads this setting
# once, when it creates its Metal device (see mtplx/mlx_process_env.py).
apply_mlx_process_defaults()

__all__ = ["MTPLXRuntime", "load", "__version__", "DISPLAY_VERSION"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .runtime import MTPLXRuntime, load

        exports = {"MTPLXRuntime": MTPLXRuntime, "load": load}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
