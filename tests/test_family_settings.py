"""Every model family owns its model-tuned settings (PX.0, 2026-09-18)."""

from __future__ import annotations

import argparse
import json

import pytest

from mtplx.backends import family_settings as fs
from mtplx.backends.descriptors import family_settings_for_model

FLASH_NEXT_CONFIG = {
    "model_type": "qwen4_exp",
    "text_config": {
        "model_type": "qwen4_exp_text",
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
        "num_key_value_heads": 2,
        "num_attention_heads": 24,
        "head_dim": 256,
        "indexer_n_heads": 4,
    },
}
DENSE_27B_CONFIG = {
    "model_type": "qwen3_5",
    "num_hidden_layers": 64,
    "full_attention_interval": 4,
    "num_key_value_heads": 4,
    "num_attention_heads": 24,
    "head_dim": 256,
}


@pytest.mark.parametrize("family", sorted(fs.FAMILY_SETTINGS))
def test_no_model_tuned_key_is_served_without_a_receipt_or_an_unmeasured_tag(family):
    """The PX.0 gate: walk the list of model-tuned keys. A family block must
    carry every key, each with an accounted source (measured, family-own,
    derived, or the explicit "inherited-27B-era, unmeasured") and a receipt
    line. A key that silently rides the shared profile fails here."""

    block = fs.family_settings(family)
    assert set(block) == set(fs.MODEL_TUNED_KEYS)
    for key in fs.MODEL_TUNED_KEYS:
        setting = block[key]
        assert setting.source in fs.ACCOUNTED_SOURCES, (family, key, setting.source)
        assert setting.source != fs.SHARED_DEFAULT
        assert len(setting.receipt.strip()) >= 20, (family, key)
        if setting.source == fs.MEASURED:
            # A measured value names when or where it was measured.
            assert any(ch.isdigit() for ch in setting.receipt), (family, key)


def test_flash_next_inherited_values_say_so_and_name_their_cell():
    block = fs.family_settings("qwen4_exp")
    inherited = {k for k, v in block.items() if v.source == fs.INHERITED}
    assert inherited == {
        "prefill_chunk_tokens",
        "qsa_prefill_compile_rows",
        "qsa_prefill_score_mb",
        "prefill_cleanup_every",
        "decode_clear_every",
        "decode_clear_context_threshold",
        "depth_policy_priors",
        "copy_lane",
    }
    for key in inherited:
        assert "Cell" in block[key].receipt or "P2.2" in block[key].receipt, key


def test_blocks_change_nothing_on_a_mac_without_tensor_units():
    # Every value that is not capability-bound equals the engine default, so
    # the stamp is empty for both families on an M1 to M4: nothing moves until
    # a measured number is written for that class of Mac.
    assert fs.family_env_stamp("qwen4_exp") == {}
    assert fs.family_env_stamp("qwen3_8") == {}
    assert fs.family_env_stamp("some_other_family") == {}
    assert fs.resolved_value("qwen4_exp", "prefill_chunk_tokens") == 2048
    assert fs.resolved_value("qwen4_exp", "qsa_prefill_compile_rows") == 2048
    assert fs.resolved_value("qwen3_8", "first_verify_reserve_tokens") == 512
    assert fs.resolved_value("qwen4_exp", "first_verify_reserve_tokens") == 1024


def test_flash_next_prefill_width_is_stamped_on_tensor_unit_gpus_only():
    # Measured on an M5 Max on 2026-09-18, so only a tensor-unit GPU gets it.
    block = fs.family_settings("qwen4_exp")
    for key in ("prefill_wide_chunk_tokens", "qsa_prefill_wide_min_context"):
        assert block[key].source == fs.MEASURED
        assert block[key].requires == fs.TENSOR_UNIT_GPU
        assert block[key].engine_default == 0
    assert fs.family_env_stamp("qwen4_exp", capabilities=(fs.TENSOR_UNIT_GPU,)) == {
        "MTPLX_QWEN4_PREFILL_WIDE_CHUNK": "4096",
        "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT": "16384",
    }
    assert fs.family_env_stamp("qwen4_exp", capabilities=("something_else",)) == {}
    # The base chunk stays the plan every Mac runs and the refusal falls back to.
    assert fs.resolved_value("qwen4_exp", "prefill_chunk_tokens") == 2048
    # The 27B has no wide-chunk receipt and no QSA layers: nothing to stamp.
    assert fs.family_env_stamp("qwen3_8", capabilities=(fs.TENSOR_UNIT_GPU,)) == {}
    assert fs.resolved_value("qwen3_8", "prefill_wide_chunk_tokens") is None


def test_one_number_edit_moves_the_chunk_and_its_compiled_width_together(monkeypatch):
    from dataclasses import replace

    block = dict(fs.QWEN4_EXP_SETTINGS)
    block["prefill_chunk_tokens"] = replace(
        block["prefill_chunk_tokens"], value=8192, source=fs.MEASURED,
        receipt="cells/p2-2-chunk-8192 (2026-09-18)",
    )
    monkeypatch.setitem(fs.FAMILY_SETTINGS, "qwen4_exp", block)
    assert fs.family_env_stamp("qwen4_exp") == {
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE": "8192",
        "MTPLX_PREFILL_CHUNK_SIZE_REPAGE": "8192",
        "MTPLX_QSA_PREFILL_COMPILE_ROWS": "8192",
    }
    # The 27B block did not move.
    assert fs.family_env_stamp("qwen3_8") == {}


def test_copy_lane_fields_stamp_one_by_one(monkeypatch):
    from dataclasses import replace

    block = dict(fs.QWEN4_EXP_SETTINGS)
    lane = dict(block["copy_lane"].value)
    lane["block_k"] = 16
    block["copy_lane"] = replace(block["copy_lane"], value=lane)
    monkeypatch.setitem(fs.FAMILY_SETTINGS, "qwen4_exp", block)
    assert fs.family_env_stamp("qwen4_exp") == {"MTPLX_CONTEXT_COPY_K": "16"}


def test_geometry_is_derived_never_typed():
    flash = fs.derived_geometry(FLASH_NEXT_CONFIG)
    assert flash == {
        "kv_bytes_per_token": 24_576,
        "kv_heads": 2,
        "query_heads": 24,
        "head_dim": 256,
    }
    dense = fs.derived_geometry(DENSE_27B_CONFIG)
    assert dense["kv_bytes_per_token"] == 65_536
    assert (dense["kv_heads"], dense["query_heads"], dense["head_dim"]) == (4, 24, 256)
    for family in fs.FAMILY_SETTINGS:
        block = fs.family_settings(family)
        assert block["kv_bytes_per_token"].value == fs.DERIVED
        assert block["prewarm_geometry"].value == fs.DERIVED


def test_compiled_verify_depths_is_a_family_capability():
    assert fs.compiled_verify_depths("qwen4_exp") == frozenset({3})
    assert fs.compiled_verify_depths("qwen3_8") is None
    assert fs.compiled_verify_depths("unknown") is None


def test_explain_rows_name_value_source_and_receipt():
    rows = {row["key"]: row for row in fs.explain_rows("qwen4_exp", FLASH_NEXT_CONFIG)}
    assert rows["prefill_chunk_tokens"]["value"] == 2048
    assert rows["prefill_chunk_tokens"]["source"] == fs.INHERITED
    assert rows["kv_bytes_per_token"]["value"] == 24_576
    assert rows["prewarm_geometry"]["value"] == {
        "kv_heads": 2, "query_heads": 24, "head_dim": 256,
    }
    assert rows["compiled_verify_depths"]["value"] == [3]
    json.dumps(list(rows.values()))
    unknown = fs.explain_rows("mystery_family")
    assert {row["source"] for row in unknown} == {fs.SHARED_DEFAULT}


def test_descriptor_lookup_returns_the_family_block():
    family, block = family_settings_for_model(
        model_ref="Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    )
    assert family == "qwen4_exp" and block is fs.QWEN4_EXP_SETTINGS
    family, block = family_settings_for_model(
        model_ref="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    )
    assert family == "qwen3_8" and block is fs.QWEN3_8_SETTINGS


def _serve_args(tmp_path, config):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return argparse.Namespace(
        model=str(model), verify_strategy="batched", generation_mode="mtp",
        scheduler_mode="serial",
    )


def test_server_stamps_the_family_block_and_an_operator_export_wins(
    monkeypatch, tmp_path
):
    from dataclasses import replace

    from mtplx.server import openai

    block = dict(fs.QWEN4_EXP_SETTINGS)
    block["prefill_chunk_tokens"] = replace(block["prefill_chunk_tokens"], value=4096)
    monkeypatch.setitem(fs.FAMILY_SETTINGS, "qwen4_exp", block)
    for key in (
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
        "MTPLX_PREFILL_CHUNK_SIZE_REPAGE",
        "MTPLX_QSA_PREFILL_COMPILE_ROWS",
    ):
        monkeypatch.delenv(key, raising=False)
    args = _serve_args(tmp_path, FLASH_NEXT_CONFIG)
    assert openai._served_model_family(args) == "qwen4_exp"
    overrides = openai._server_runtime_env_overrides(args, None)
    assert overrides["MTPLX_PREFILL_CHUNK_SIZE_DENSE"] == "4096"
    assert overrides["MTPLX_PREFILL_CHUNK_SIZE_REPAGE"] == "4096"
    assert overrides["MTPLX_QSA_PREFILL_COMPILE_ROWS"] == "4096"
    # An operator export beats the family block.
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "1024")
    overrides = openai._server_runtime_env_overrides(args, None)
    assert "MTPLX_PREFILL_CHUNK_SIZE_DENSE" not in overrides
    assert overrides["MTPLX_PREFILL_CHUNK_SIZE_REPAGE"] == "4096"


def test_server_stamp_is_empty_without_tensor_units(monkeypatch, tmp_path):
    from mtplx.server import openai

    monkeypatch.setattr(openai, "_qwen4_tensor_unit_gpu", lambda: False)
    stamp_keys = {
        "MTPLX_QWEN4_PREFILL_WIDE_CHUNK",
        "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT",
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
        "MTPLX_PREFILL_CHUNK_SIZE_REPAGE",
        "MTPLX_QSA_PREFILL_COMPILE_ROWS",
        "MTPLX_QSA_PREFILL_SCORE_MB",
        "MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY",
        "MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT",
        "MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD",
        "MTPLX_COMPILED_VERIFY_GROWTH_RESERVE",
        "MTPLX_CONTEXT_COPY_K",
    }
    for key in stamp_keys:
        monkeypatch.delenv(key, raising=False)
    args = _serve_args(tmp_path, FLASH_NEXT_CONFIG)
    assert openai._served_family_env_stamp(args) == {}
    assert not (stamp_keys & set(openai._server_runtime_env_overrides(args, None)))


def test_server_stamps_the_prefill_width_on_a_tensor_unit_mac(monkeypatch, tmp_path):
    from mtplx.server import openai

    monkeypatch.setattr(openai, "_qwen4_tensor_unit_gpu", lambda: True)
    for key in ("MTPLX_QWEN4_PREFILL_WIDE_CHUNK", "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    args = _serve_args(tmp_path, FLASH_NEXT_CONFIG)
    assert openai._served_family_env_stamp(args) == {
        "MTPLX_QWEN4_PREFILL_WIDE_CHUNK": "4096",
        "MTPLX_QSA_PREFILL_WIDE_MIN_CONTEXT": "16384",
    }
    # The dense 27B on the same Mac gets nothing.
    dense = tmp_path / "dense"
    dense.mkdir()
    assert openai._served_family_env_stamp(_serve_args(dense, DENSE_27B_CONFIG)) == {}


def test_the_user_flag_still_beats_the_family_block():
    # generation._prefill_chunk_size: the request-local override (the launch
    # flag --prefill-chunk-tokens or the live setting) wins before any env.
    from mtplx import generation

    with generation.prefill_chunk_size_override(512):
        assert generation._prefill_chunk_size() == 512


def test_doctor_explain_lists_each_setting_with_its_source(capsys):
    from mtplx import lane_explain
    from mtplx.commands import public

    section = public._doctor_explain_family_settings(
        {"settings": {"model": "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"}}
    )
    assert section["family"] == "qwen4_exp"
    report = lane_explain.build_explain_report(
        health=None,
        server_url="http://127.0.0.1:8000",
        tensor_units={"architecture": "applegpu_g17s", "hardware": True,
                      "route": True, "fallback_forced": False},
        family_settings=section,
    )
    text = "\n".join(lane_explain.render_explain_lines(report))
    assert "prefill_chunk_tokens: 2048 (inherited-27B-era, unmeasured)" in text
    assert "prefill_wide_chunk_tokens: 4096 (measured)" in text
    assert "needs tensor units: this Mac has them" in text
    portable = lane_explain.build_explain_report(
        health=None,
        server_url="http://127.0.0.1:8000",
        tensor_units={"architecture": "applegpu_g15s", "hardware": False,
                      "route": False, "fallback_forced": False},
        family_settings=section,
    )
    text = "\n".join(lane_explain.render_explain_lines(portable))
    assert "needs tensor units: this Mac does not, so the engine default serves" in text
    assert "compiled_verify_depths: [3]" in text


def test_flash_next_owns_its_own_copies_of_the_inherited_tables():
    # Editing a Flash-Next table must never move the 27B (or the engine
    # default the stamp compares against).
    flash = fs.family_settings("qwen4_exp")
    dense = fs.family_settings("qwen3_8")
    for key in ("depth_policy_priors", "copy_lane"):
        assert flash[key].value == dense[key].value
        assert flash[key].value is not dense[key].value
    assert flash["copy_lane"].engine_default is dense["copy_lane"].value
