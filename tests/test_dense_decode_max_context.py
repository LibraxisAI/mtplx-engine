"""The dense-decode context ceiling (the 147.4k decode-cliff constant).

Receipts: MEASUREMENTS 2026-08-26 07:58 (root cause) and 08:24 (fix
verified: 12.0 -> 16.3/18.4 tok/s at 147.4k once decode stays dense).
"""

import os

import pytest

from mtplx.generation import (
    _dense_decode_max_context,
    _sustained_prefill_layout,
)

CEILING = "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        CEILING,
        "MTPLX_DENSE_KV_BYTES_PER_TOKEN",
        "MTPLX_DENSE_KV_BYTES_PER_TOKEN_DERIVED",
        "MTPLX_DENSE_DECODE_RAM_PERCENT",
        "MTPLX_MEMORY_BUDGET",
        "MTPLX_CONTEXT_WINDOW_TOKENS",
        "MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT",
        "MTPLX_KV_QUANT",
        "MTPLX_PAGED_KV_QUANT",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


def _fake_sysconf(total_bytes):
    def sysconf(name):
        if name == "SC_PAGE_SIZE":
            return 4096
        if name == "SC_PHYS_PAGES":
            return total_bytes // 4096
        raise ValueError(name)

    return sysconf


def test_default_is_the_shipped_literal():
    assert _dense_decode_max_context() == 131072


def test_numeric_env_wins(monkeypatch):
    monkeypatch.setenv(CEILING, "262144")
    assert _dense_decode_max_context() == 262144


def test_garbage_env_falls_back(monkeypatch):
    monkeypatch.setenv(CEILING, "lots")
    assert _dense_decode_max_context() == 131072


def test_auto_budgets_from_ram(monkeypatch):
    # 96 GiB machine, Qwen3.8 geometry, 15% budget:
    # 96 GiB * 0.15 / 65536 B = 235929 tokens.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 235929


def test_auto_clamps_to_context_window(monkeypatch):
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_CONTEXT_WINDOW_TOKENS", "200000")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 200000


def test_auto_never_regresses_below_shipped_literal(monkeypatch):
    # 36 GiB machine at 15% = 88473 tokens < 131072: auto must not make
    # the product SLOWER than today's default anywhere.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(36 * 1024**3))
    assert _dense_decode_max_context() == 131072


def test_auto_survives_missing_sysconf(monkeypatch):
    monkeypatch.setenv(CEILING, "auto")

    def broken(_name):
        raise ValueError("no sysconf here")

    monkeypatch.setattr(os, "sysconf", broken)
    assert _dense_decode_max_context() == 131072


def test_auto_honours_geometry_env(monkeypatch):
    # A model with half the KV bytes per token affords twice the tokens.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN", "32768")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 471859


def test_layout_stays_dense_below_ceiling(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv(CEILING, "262144")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "147434")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"


def test_layout_repages_above_ceiling(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv(CEILING, "131072")
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "147434")
    assert _sustained_prefill_layout() == "contiguous_then_repage"


# --- PX.3(b), 2026-09-18: geometry from the model config, seat-aware floor ---


def test_auto_reads_the_geometry_the_loader_derived(monkeypatch):
    # Flash-Next: 12 full-attention layers x K+V x 2 KV heads x 256 x bf16
    # = 24,576 B/token, exported by runtime.load from config.json. Before the
    # export every model was budgeted with the 27B's 65,536.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN_DERIVED", "24576")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == int(96 * 1024**3 * 15 / 100) // 24576


def test_operator_geometry_beats_the_derived_value(monkeypatch):
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN", "32768")
    monkeypatch.setenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN_DERIVED", "24576")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 471859


def test_loader_exports_the_derived_geometry_and_clears_it(monkeypatch):
    from mtplx.runtime import DERIVED_DENSE_KV_BYTES_ENV, _export_derived_model_geometry

    flash_next = {
        "text_config": {
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
            "num_key_value_heads": 2,
            "head_dim": 256,
        }
    }
    dense_27b = {
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
        "num_key_value_heads": 4,
        "head_dim": 256,
    }
    assert _export_derived_model_geometry(flash_next) == 24_576
    assert os.environ[DERIVED_DENSE_KV_BYTES_ENV] == "24576"
    assert _export_derived_model_geometry(dense_27b) == 65_536
    assert os.environ[DERIVED_DENSE_KV_BYTES_ENV] == "65536"
    # A config that does not describe its attention must not inherit the
    # previous model's value.
    assert _export_derived_model_geometry({"model_type": "mystery"}) is None
    assert DERIVED_DENSE_KV_BYTES_ENV not in os.environ


@pytest.mark.parametrize(
    ("ram_gib", "expected"),
    [
        # 27B geometry. From 32 GB up the 131072 floor holds (8 GiB slab is
        # within a quarter of RAM): nothing moves, the 48 GB seat included.
        (128, 314572),
        (64, 157286),
        (48, 131072),
        (36, 131072),
        (32, 131072),
        # Small seats lose the flat floor: a quarter of RAM bounds the slab.
        (24, 98304),
        (16, 65536),
    ],
)
def test_floor_is_seat_aware_on_the_27b_geometry(monkeypatch, ram_gib, expected):
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN_DERIVED", "65536")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(ram_gib * 1024**3))
    assert _dense_decode_max_context() == expected


def test_floor_holds_for_flash_next_on_every_seat_that_loads_it(monkeypatch):
    # 131072 x 24,576 B = 3 GiB: inside a quarter of RAM from 12 GiB up.
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_DENSE_KV_BYTES_PER_TOKEN_DERIVED", "24576")
    monkeypatch.setenv("MTPLX_CONTEXT_WINDOW_TOKENS", "86016")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(96 * 1024**3))
    assert _dense_decode_max_context() == 131072


def test_simulated_small_seat_matches_a_real_one(monkeypatch):
    monkeypatch.setenv(CEILING, "auto")
    monkeypatch.setenv("MTPLX_MEMORY_BUDGET", "24G")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(128 * 1024**3))
    assert _dense_decode_max_context() == 98304
