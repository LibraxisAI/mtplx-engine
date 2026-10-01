"""What a pack's mtplx_runtime.json says reaches the user as the real reason.

An official pack whose exactness measurement is still pending is neither
verified nor broken, and a pack that needs a newer engine is refused before
any weight is read.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from mtplx import artifacts, runtime
from mtplx.artifacts import MTPInspection, inspect_model
from mtplx.backends import registry
from mtplx.commands import public
from mtplx.ui import onboarding
from mtplx.version import __version__

BONSAI_ID = "mtplx-bonsai-2-27b-optimized-speed"


def _contract(**fields):
    return {
        "mtplx_version": "2.12.0",
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 3,
        "recommended_profile": "stable",
        "exactness_baseline": {"max_abs_diff": 0.0},
        "verified_on": {"hardware": "test"},
        **fields,
    }


def _qwen_pack(path, monkeypatch, **contract_fields):
    (path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": 1,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 248320,
    }))
    monkeypatch.setattr(artifacts, "inspect_mtp_tensors", lambda *_a, **_k: MTPInspection(
        mtp_file=str(path / "mtp.safetensors"), exists=True, tensor_count=15,
    ))
    (path / "mtplx_runtime.json").write_text(json.dumps(_contract(**contract_fields)))
    return inspect_model(path).compatibility


def test_official_pack_with_a_pending_measurement_is_qualification_pending(monkeypatch, tmp_path):
    verdict = _qwen_pack(
        tmp_path, monkeypatch,
        public_model_id=BONSAI_ID,
        exactness_baseline={"status": "pending_measurement"},
    )

    assert verdict["tier"] == "family-compatible-unverified"
    assert (verdict["can_run"], verdict["exit_code"], verdict["unverified_model"]) == (True, 0, True)
    assert verdict["support_level"] == registry.SUPPORT_QUALIFICATION_PENDING
    assert "qualification pending" in verdict["message"]
    assert "pending_measurement" in verdict["message"]
    assert "Forge" not in verdict["message"]


@pytest.mark.parametrize("contract_fields", [
    {"public_model_id": BONSAI_ID, "exactness_baseline": {"status": "failed"}},
    {"public_model_id": BONSAI_ID,
     "exactness_baseline": {"status": "pending", "public_release_blocker": True}},
    {"public_model_id": BONSAI_ID, "exactness_baseline": {"status": "pending"},
     "speed_evidence": {"verdict": "no_mtp_depth_beat_ar"}},
    {"public_model_id": "someone/custom-pack",
     "exactness_baseline": {"status": "pending_measurement"}},
])
def test_contract_damage_and_third_party_pending_read_as_before(monkeypatch, tmp_path, contract_fields):
    verdict = _qwen_pack(tmp_path, monkeypatch, **contract_fields)

    assert verdict["runtime_compatibility"] == "runtime-contract-unverified"
    assert verdict["support_level"] == "native-backend-needs-contract-repair"
    assert (verdict["can_run"], verdict["exit_code"]) == (True, 3)
    assert "regenerate the contract with Forge" in verdict["message"]


def test_picker_labels_an_official_pending_pack_honestly(tmp_path):
    pack = tmp_path / "Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
    pack.mkdir()
    (pack / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "mtp_num_hidden_layers": 1,
    }))
    (pack / "mtp.safetensors").write_bytes(b"placeholder")
    (pack / "mtplx_runtime.json").write_text(json.dumps(_contract(
        public_model_id=BONSAI_ID, exactness_baseline={"status": "pending_measurement"},
    )))

    scanned = onboarding._classify_scanned_model(pack)

    assert scanned.tier == "qualification-pending"
    assert onboarding._tier_badge(scanned.tier)[0] == "Official pack, qualification pending"


def test_serve_gate_prints_the_verdicts_own_reason(monkeypatch, capsys):
    def gate(compatibility):
        inspection = {"compatibility": {"can_run": True, "unverified_model": True, **compatibility}}
        monkeypatch.setattr(public, "inspect_model", lambda _model: SimpleNamespace(to_dict=lambda: inspection))
        return public._model_gate("pack")

    assert gate({
        "support_level": registry.SUPPORT_QUALIFICATION_PENDING,
        "message": "Official MTPLX pack, qualification pending (x).",
    })[1] is None
    assert capsys.readouterr().err == "NOTE: Official MTPLX pack, qualification pending (x).\n"

    gate({"message": "Runtime contract is not verified: y."})
    assert capsys.readouterr().err == "WARNING: Runtime contract is not verified: y.\n"


@pytest.mark.parametrize("floor", [None, "", __version__, "2.12", "2.11.9"])
def test_an_engine_floor_at_or_below_this_engine_passes(monkeypatch, tmp_path, floor):
    fields = {} if floor is None else {"min_engine_version": floor}

    verdict = _qwen_pack(tmp_path, monkeypatch, **fields)

    assert (verdict["tier"], verdict["can_run"]) == ("verified", True)


def test_a_pack_that_needs_a_newer_engine_is_refused_before_any_weight_is_read(monkeypatch, tmp_path):
    message = (
        f"This model needs MTPLX 99.0 or later (you have {__version__}). "
        "Update MTPLX, then try again."
    )

    verdict = _qwen_pack(tmp_path, monkeypatch, min_engine_version="99.0")

    assert verdict["message"] == message
    assert (verdict["can_run"], verdict["exit_code"]) == (False, 4)
    # No unsafe-force flag starts it, and the refusal is one plain line.
    inspection, exit_code = public._model_gate(str(tmp_path), unsafe_force_unverified=True, yes=True)
    assert exit_code == 4
    assert public._model_gate_error_lines(inspection) == [f"error: {message}"]
    # The loader refuses too, before it reads the config.
    monkeypatch.setattr(runtime, "load_config", lambda *_a: pytest.fail("the loader read the pack"))
    with pytest.raises(registry.ModelCompatibilityError, match="needs MTPLX 99.0"):
        runtime.load(tmp_path)


def test_a_malformed_engine_floor_is_ignored_with_a_warning(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="mtplx.backends.registry"):
        verdict = _qwen_pack(tmp_path, monkeypatch, min_engine_version="2.12.x")

    assert (verdict["tier"], verdict["can_run"]) == ("verified", True)
    assert "malformed min_engine_version '2.12.x'" in caplog.text
