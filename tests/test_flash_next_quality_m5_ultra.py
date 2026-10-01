"""Flash-Next Optimized-Quality on the 256 GB and 512 GB M5 Ultra: CPU-only pins.

Every figure here is arithmetic on the shipped catalog entry and the memory
planner. Nothing loads a model or touches Metal. The weight-file byte counts
come from the published Quality pack at revision
3908bbfcfde1c331f3c29b3ad04228e5f00fbaff, verified against its Hub inventory.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.hardware import classify_apple_silicon_generation
from mtplx.memory_plan import (
    ENGINE_RAM_CAP_BYTES,
    dense_kv_bytes_per_token_from_config,
    engine_envelope_bytes,
    plan_memory,
    qsa_aux_bytes_per_token_from_config,
    qsa_prefill_transient_bytes_per_token_from_config,
    system_reserve_bytes,
)
from mtplx.model_catalog import (
    MODERN_TIER,
    catalog_model_with_id,
    default_catalog_model,
    evaluate_feasibility,
    recommended_catalog_ids,
    recommended_models,
)

GIB = 1024**3
QUALITY_ID = "flash-next-optimized-quality"
QUALITY_HF = "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality"

# Same file-size accounting as engine_session.model_weights_bytes and
# ngram_table_bytes: 37 body/vision shards, the MTP head, and the separate
# table, which streams from SSD on every Mac and is never weight.
BODY_MTP_VISION_BYTES = 137_934_585_655
NGRAM_TABLE_BYTES = 32_000_153_976
# openai._resident_floor_margin_bytes at 112 GB and up.
RESIDENT_FLOOR_MARGIN_BYTES = 6 * GIB
MODEL_MAX_CONTEXT = 262_144

# Real Flash-Next geometry (mirrors tests/test_qsa_memory_plan_393.py).
FLASH_NEXT_CONFIG = {
    "text_config": {
        "indexer_n_heads": 4,
        "indexer_head_dim": 128,
        "indexer_compress_ratio": 4,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12,
    }
}

M5_ULTRA_TIERS = (256, 512)


def test_m5_ultra_brand_string_classifies_as_modern_m5():
    # sysctl machdep.cpu.brand_string on an M5 Ultra reads "Apple M5 Ultra".
    assert classify_apple_silicon_generation("Apple M5 Ultra", system="Darwin", machine="arm64") == "m5"
    assert classify_apple_silicon_generation("Apple M3 Ultra", system="Darwin", machine="arm64") == "m3"


@pytest.mark.parametrize("ram_gib", M5_ULTRA_TIERS)
def test_catalog_leads_with_optimized_speed_then_quality_on_m5_ultra_tiers(ram_gib):
    # Optimized Speed leads and Quality follows, as the 27B trio does.
    lead = ["flash-next-optimized-speed", QUALITY_ID]
    raw = recommended_catalog_ids(memory_gib=ram_gib, chip_tier=MODERN_TIER)
    assert raw[:2] == lead
    assert raw.count(QUALITY_ID) == 1
    visible = recommended_models(memory_gib=ram_gib, chip_tier=MODERN_TIER)
    assert [m.id for m in visible[:2]] == lead
    assert default_catalog_model(memory_gib=ram_gib, chip_tier=MODERN_TIER).id == lead[0]
    pack = catalog_model_with_id(QUALITY_ID)
    assert pack.hf_model_id == QUALITY_HF
    assert "mtplx-flash-next-optimized-quality" in pack.aliases
    # The badge: 136.4 GiB peak x 1.5 safety = 204.6 GiB <= 256 GiB.
    assert evaluate_feasibility(pack, chip_tier=MODERN_TIER, ram_gib=ram_gib, disk_free_gib=1000).verdict == "recommended"


def test_quality_is_offered_but_not_first_below_256_gib():
    for ram_gib in (192, 255.9):
        raw = recommended_catalog_ids(memory_gib=ram_gib, chip_tier=MODERN_TIER)
        assert raw[0] == "qwen38-27b-optimized-speed"
        assert QUALITY_ID in raw
        assert QUALITY_ID in [m.id for m in recommended_models(memory_gib=ram_gib, chip_tier=MODERN_TIER)]
    pack = catalog_model_with_id(QUALITY_ID)
    assert evaluate_feasibility(pack, chip_tier=MODERN_TIER, ram_gib=192, disk_free_gib=1000).verdict == "tight_fit"
    # The 96 GB base M5 Ultra never sees the pack.
    assert QUALITY_ID not in [m.id for m in recommended_models(memory_gib=96, chip_tier=MODERN_TIER)]


@pytest.mark.parametrize("ram_gib", M5_ULTRA_TIERS)
def test_mtplx_start_default_is_optimized_speed_on_m5_ultra(ram_gib, monkeypatch):
    from mtplx import default_models as defaults

    monkeypatch.delenv(defaults.QWEN38_OPTIMIZED_SPEED_MODEL_ENV, raising=False)
    monkeypatch.delenv(defaults.SPEED_MODEL_ENV, raising=False)
    monkeypatch.delenv(defaults.DEFAULT_MODEL_VARIANT_ENV, raising=False)
    monkeypatch.delenv("MTPLX_MODEL_DIR", raising=False)
    # No local library pack: the Hub repo is the answer.
    monkeypatch.setattr(defaults, "_QWEN38_OPTIMIZED_SPEED_LOCAL_CANDIDATES", (), raising=False)
    # The brand string path, not a pre-classified generation.
    hardware = {"system": "Darwin", "machine": "arm64", "chip": "Apple M5 Ultra", "memory_gib": float(ram_gib)}
    selection = defaults.select_default_model(hardware=hardware)
    assert selection.hf_model == "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    assert selection.display_name == "Qwen 3.8 Flash-Next Optimized Speed"
    assert selection.chip_generation == "m5"
    assert defaults.public_model_id_for_ref(selection.hf_model) == "mtplx-flash-next-optimized-speed"


def test_served_id_and_hf_repo_resolve_through_cli_paths():
    from mtplx.artifacts import _hf_repo_id_from_ref
    from mtplx.backends.descriptors import model_family_from_inspection
    from mtplx.commands.public import _model_ref_from_public_model_id

    for ref in ("mtplx-flash-next-optimized-quality", QUALITY_ID, QUALITY_HF, "Qwen3.8-Flash-Next-MTPLX-Optimized-Quality"):
        assert _model_ref_from_public_model_id(ref) == QUALITY_HF
        assert _hf_repo_id_from_ref(ref) == QUALITY_HF
        assert model_family_from_inspection(model_ref=ref) == "qwen4_exp"


def _quality_plan(ram_gib: int, *, sparse_prefill: bool, requested_context: int | None = None):
    total = ram_gib * GIB
    kv = dense_kv_bytes_per_token_from_config(FLASH_NEXT_CONFIG)
    aux = qsa_aux_bytes_per_token_from_config(FLASH_NEXT_CONFIG)
    transient = 0 if sparse_prefill else qsa_prefill_transient_bytes_per_token_from_config(FLASH_NEXT_CONFIG)
    assert (kv, aux) == (24_576, 7_872)
    floor = BODY_MTP_VISION_BYTES + RESIDENT_FLOOR_MARGIN_BYTES
    return plan_memory(
        total_ram_bytes=total,
        model_weights_bytes=BODY_MTP_VISION_BYTES,
        ngram_table_streamed_bytes=NGRAM_TABLE_BYTES,
        kv_bytes_per_token=kv,
        model_max_context=MODEL_MAX_CONTEXT,
        requested_context=requested_context,
        aux_bytes_per_token=aux,
        prefill_transient_bytes_per_token=transient,
        resident_floor_bytes=floor,
    )


@pytest.mark.parametrize("ram_gib", M5_ULTRA_TIERS)
def test_planner_grants_full_window_with_sparse_prefill_on_m5_ultra(ram_gib):
    plan = _quality_plan(ram_gib, sparse_prefill=True)
    assert plan.available and plan.model_fits and not plan.tight_machine
    # The allocator envelope is the 192 GiB cap on both tiers; the floor
    # (128.5 + 6 GiB) sits under it, so the 75% rule is not lifted.
    assert plan.usable_bytes == ENGINE_RAM_CAP_BYTES == 192 * GIB
    assert plan.usable_source == "formula"
    assert plan.context_window_fit == MODEL_MAX_CONTEXT
    assert plan.context_window_resolved == MODEL_MAX_CONTEXT
    assert not plan.context_machine_bound
    # Weights 128.46 GiB leave 59.5 GiB of KV budget under the cap; the
    # 29.8 GiB table streams and is reported, not budgeted.
    assert 59 * GIB < plan.usable_bytes - plan.model_weights_bytes - 4 * GIB < 60 * GIB
    assert plan.ngram_table_streamed_bytes == NGRAM_TABLE_BYTES


@pytest.mark.parametrize("ram_gib", M5_ULTRA_TIERS)
def test_planner_dense_prefill_fallback_still_covers_the_full_window_on_m5_ultra(ram_gib):
    # M3 Ultra from pip or Homebrew: the dense indexer lane prices 104,448
    # extra bytes per token, and 59.5 GiB still funds 466,944 tokens.
    plan = _quality_plan(ram_gib, sparse_prefill=False)
    assert plan.model_fits and not plan.tight_machine
    assert plan.context_window_fit == MODEL_MAX_CONTEXT
    assert not plan.context_machine_bound
    # 128K requested explicitly is not overcommitted even on the dense lane.
    explicit = _quality_plan(ram_gib, sparse_prefill=False, requested_context=131_072)
    assert explicit.context_window_resolved == 131_072 and not explicit.context_overcommitted


def test_512_gib_plan_is_identical_to_256_gib_plan():
    for sparse in (True, False):
        a = _quality_plan(256, sparse_prefill=sparse).to_dict()
        b = _quality_plan(512, sparse_prefill=sparse).to_dict()
        for key in ("usable_bytes", "context_window_fit", "bank_idle_max_bytes", "bank_steady_bytes", "model_fits"):
            assert a[key] == b[key]


@pytest.mark.parametrize("ram_gib", M5_ULTRA_TIERS)
def test_metal_load_gate_admits_quality_on_m5_ultra(ram_gib, monkeypatch):
    from mtplx.server import openai

    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    calls: list[tuple[str, int]] = []
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        set_memory_limit=lambda v: calls.append(("memory", int(v))),
        set_wired_limit=lambda v: calls.append(("wired", int(v))),
    )
    total = ram_gib * GIB
    floor = BODY_MTP_VISION_BYTES + openai._resident_floor_margin_bytes(total)
    assert openai._resident_floor_margin_bytes(total) == RESIDENT_FLOOR_MARGIN_BYTES
    result = openai._apply_metal_memory_caps(mx_module=mx, total_ram_bytes=total, minimum_resident_bytes=floor)
    assert result["applied"] is True, result
    assert result["memory_limit_bytes"] == 192 * GIB
    # The floor (134.5 GiB) sits under the 60% wired default (153.6 GiB at
    # 256, the 160 GiB cap at 512), so the default wires the weights.
    assert result["wired_limit_bytes"] == min(int(total * 0.60), 160 * GIB)
    assert result["wired_limit_bytes"] > floor
    assert floor + system_reserve_bytes(total) < total
    assert engine_envelope_bytes(total, resident_floor_bytes=floor) == 192 * GIB


def test_metal_load_gate_refuses_quality_on_96_gib_m5_ultra(monkeypatch):
    from mtplx.server import openai

    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_WIRED_LIMIT_BYTES", raising=False)
    mx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True), set_memory_limit=lambda v: None, set_wired_limit=lambda v: None)
    total = 96 * GIB
    # The table streams, so the floor is body+MTP+vision + margin.
    floor = BODY_MTP_VISION_BYTES + openai._resident_floor_margin_bytes(total)
    result = openai._apply_metal_memory_caps(mx_module=mx, total_ram_bytes=total, minimum_resident_bytes=floor)
    assert result["applied"] is False and result["reason"] == "insufficient_ram"


def test_swift_catalog_matches_python_on_the_quality_pack():
    from pathlib import Path

    swift = Path("apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXModelOption.swift").read_text()
    pack = catalog_model_with_id(QUALITY_ID)
    assert f"sizeBytes: {pack.size_bytes:_}" in swift
    assert f"peakMemoryGiB: {pack.peak_memory_gib}" in swift
    assert 'hfModelID: "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality"' in swift
    assert "hardware.unifiedMemoryGiB >= 256" in swift and "hardware.unifiedMemoryGiB >= 96" in swift
    feasibility = Path("apps/MTPLXApp/Sources/MTPLXAppCore/Onboarding/ModelFeasibility.swift").read_text()
    assert "memorySafetyFactor: Double = 1.5" in feasibility
