"""One tensor-unit (NAX) detector for every lane (PX.3e, 2026-09-18).

``nax_verify`` matched the exact prefix ``applegpu_g17`` and honored the
rehearsal switch; ``qsa_indexer_select`` parsed the generation and ignored
it. Both now read ``mtplx.nax_detect``.
"""

from __future__ import annotations

import platform

import mlx.core as mx
import pytest

from mtplx import nax_detect, nax_verify
from mtplx.kernels import qsa_indexer_select


@pytest.fixture
def fake_machine(monkeypatch):
    def install(architecture: str, macos: str = "26.2.1") -> None:
        monkeypatch.setattr(
            mx, "device_info", lambda *_device: {"architecture": architecture}
        )
        monkeypatch.setattr(platform, "mac_ver", lambda: (macos, ("", "", ""), "arm64"))
        nax_detect.nax_hardware_available.cache_clear()

    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", raising=False)
    yield install
    nax_detect.nax_hardware_available.cache_clear()


def test_both_modules_read_the_same_detector():
    assert nax_verify.nax_available is nax_detect.nax_available
    assert nax_verify._nax_hardware_available is nax_detect.nax_hardware_available
    assert qsa_indexer_select._mlx_nax_available is nax_detect.nax_hardware_available
    assert (
        qsa_indexer_select._nax_available_for_platform
        is nax_detect.nax_available_for_platform
    )


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("applegpu_g13s", False),  # M1
        ("applegpu_g14s", False),  # M2
        ("applegpu_g15s", False),  # M3
        ("applegpu_g16s", False),  # M4 (issue #504 doctor output)
        ("applegpu_g17s", True),  # M5
        ("applegpu_g17d", True),
        ("applegpu_g17p", False),  # phone class needs generation 18
        ("applegpu_g18s", True),  # the old exact-prefix rule said False here
        ("applegpu_g18p", True),
    ],
)
def test_every_lane_gets_the_same_answer_per_chip(fake_machine, architecture, expected):
    fake_machine(architecture)
    assert nax_verify.nax_available() is expected
    assert qsa_indexer_select.qsa_indexer_select_nax_available() is expected


def test_macos_floor_applies_to_both(fake_machine):
    fake_machine("applegpu_g17s", macos="26.1")
    assert nax_verify.nax_available() is False
    assert qsa_indexer_select.qsa_indexer_select_nax_available() is False


def test_rehearsal_switch_reaches_the_flash_next_prefill_gate(fake_machine, monkeypatch):
    fake_machine("applegpu_g17s")
    assert qsa_indexer_select.qsa_indexer_select_nax_available() is True
    # Read per call: no cache_clear between the flips.
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    assert nax_verify.nax_available() is False
    assert qsa_indexer_select.qsa_indexer_select_nax_available() is False
    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK")
    assert nax_verify.nax_available() is True
    assert qsa_indexer_select.qsa_indexer_select_nax_available() is True


def test_rehearsal_switch_never_moves_the_hardware_truth(fake_machine, monkeypatch):
    # The TF32 mirror must match what MLX's own float32 GEMM does on this
    # machine, and MLX does not know the switch.
    fake_machine("applegpu_g17s")
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    qsa_indexer_select._mlx_tf32_enabled.cache_clear()
    try:
        assert nax_detect.nax_hardware_available() is True
        assert qsa_indexer_select._mlx_tf32_enabled() is True
    finally:
        qsa_indexer_select._mlx_tf32_enabled.cache_clear()


def test_flash_next_prefill_auto_gate_follows_the_switch(fake_machine, monkeypatch):
    from mtplx.kernels import qsa_prefill_direct
    from mtplx.models import qwen4_exp

    fake_machine("applegpu_g17s")
    monkeypatch.setattr(qsa_prefill_direct, "qsa_prefill_direct_ready", lambda: False)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is True
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    # No tensor units and no Steel extension: the auto lane is off, which is
    # what a pip or Homebrew install on an M1 to M4 resolves.
    assert qwen4_exp.qsa_prefill_lane_auto_supported() is False
