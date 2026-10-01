"""The import-frozen runtime gates follow the env the server stamps."""

from __future__ import annotations

import os

import pytest

from mtplx import generation, profiles, qwen4_block_verify, qwen4_draft_k20_prescatter, runtime_options

KEYS = (
    "MTPLX_QWEN4_OPDIET",
    "MTPLX_QWEN4_VERIFY_GLUE",
    "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
    "MTPLX_QWEN4_BLOCK_VERIFY",
)


@pytest.fixture
def clean_flags(monkeypatch):
    for key in KEYS + ("MTPLX_QWEN4_OPDIET_ITEMS", "MTPLX_QWEN4_VERIFY_GLUE_ITEMS"):
        monkeypatch.delenv(key, raising=False)
    yield
    runtime_options.refresh_env_flags({})


def test_refresh_reads_every_frozen_gate_from_the_given_env(clean_flags):
    armed = runtime_options.refresh_env_flags({key: "1" for key in KEYS})
    assert armed == {key: True for key in KEYS}
    assert runtime_options.qwen4_opdiet_enabled() is True
    assert runtime_options.qwen4_verify_glue_enabled() is True
    assert qwen4_block_verify.is_enabled() is True
    assert qwen4_draft_k20_prescatter.is_enabled() is True
    assert generation._QWEN4_BLOCK_VERIFY is True
    assert generation._QWEN4_DRAFT_K20_PRESCATTER is True

    disarmed = runtime_options.refresh_env_flags({})
    assert disarmed == {key: False for key in KEYS}
    assert qwen4_block_verify.is_enabled() is False
    assert generation._QWEN4_BLOCK_VERIFY is False


def test_apply_profile_env_refreshes_the_gates_it_just_wrote(clean_flags, monkeypatch):
    # apply_profile_env writes the WHOLE profile block into os.environ; the
    # cleanup below only knows the four gates, so the other 38 keys used to
    # stay exported for every later test file.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(profiles, "_profile_env_previous_state", {}, raising=False)
    profiles.apply_profile_env(
        "turbo", runtime_env_overrides={"MTPLX_QWEN4_BLOCK_VERIFY": "1", "MTPLX_QWEN4_OPDIET": "1"}
    )
    try:
        assert os.environ.get("MTPLX_QWEN4_BLOCK_VERIFY") == "1"
        assert qwen4_block_verify.is_enabled() is True
        assert generation._QWEN4_BLOCK_VERIFY is True
        assert runtime_options.qwen4_opdiet_enabled() is True
    finally:
        # apply_profile_env wrote these straight into os.environ; a
        # monkeypatch.delenv here would record that value and put it back at
        # teardown, leaking the gates into the next test file.
        for key in KEYS:
            os.environ.pop(key, None)
        runtime_options.refresh_env_flags({})


def test_a_test_mapping_does_not_touch_module_state(clean_flags):
    sink: dict[str, str] = {}
    profiles.apply_profile_env("turbo", environ=sink, runtime_env_overrides={"MTPLX_QWEN4_BLOCK_VERIFY": "1"})
    assert sink.get("MTPLX_QWEN4_BLOCK_VERIFY") == "1"
    assert qwen4_block_verify.is_enabled() is False
