"""MiMo resolves to its own family and tunes at depth 1 only.

``mimo-mtp`` has shipped a native backend with ``can_run_verified=True``, but
``model_family_from_inspection`` had no branch returning "mimo", so every MiMo
artifact resolved to "unknown" and ``tune_policy_for_model`` refused it.
``forge build`` exited 1 at the tune gate after a successful convert, extract
and calibrate.

Depth is capped at D1: ``mimo_mtp_patch.mtp_forward`` raises on ``mtp_depth``
above 1, and vLLM's proposer is single-token as well, so offering D2+ would
advertise depths the backend refuses.  Measured on a forged MiMo-7B-RL pack on
an M3 Max: AR 69.69 tok/s, D1 91.38 tok/s (1.311x) at 69.2% acceptance.
"""

from __future__ import annotations

from mtplx.backends.descriptors import (
    model_family_from_inspection,
    reasoning_policy_for_model,
    tune_policy_for_model,
)

MIMO_INSPECTION = {
    "model_type": "mimo",
    "architecture": "MiMoForCausalLM",
    "mtp_arch": "mimo-mtp",
    "num_hidden_layers": 36,
    "mtp_num_hidden_layers": 1,
}


def test_mimo_resolves_to_the_mimo_family():
    assert model_family_from_inspection(MIMO_INSPECTION) == "mimo"


def test_tune_is_supported_for_mimo():
    assert tune_policy_for_model(inspection=MIMO_INSPECTION).supported is True


def test_mimo_offers_depth_one_only():
    policy = tune_policy_for_model(inspection=MIMO_INSPECTION)
    assert policy.candidates == ("AR", "D1")
    assert not any(c in policy.candidates for c in ("D2", "D3"))


def test_the_family_is_read_from_the_artifact_not_the_folder_name():
    # model_type and arch_id both feed the marker text, so a MiMo artifact is
    # recognised even when its directory has been renamed.
    assert model_family_from_inspection({"model_type": "mimo"}) == "mimo"
    assert model_family_from_inspection({"mtp_arch": "mimo-mtp"}) == "mimo"


# XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B is a Qwen3.5-9B fine-tune: "MiMo" is in
# its name, not in its architecture, so it keeps the Qwen 3.5 contract.
DISTILL_REF = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
DISTILL_INSPECTION = {
    "model_dir": "/Users/example/.mtplx/models/XiaomiMiMo--MiMo-V2.6-Distill-Qwen-9B",
    "runtime_model": DISTILL_REF,
    "model_type": "qwen3_5",
    "architecture": "Qwen3_5ForConditionalGeneration",
    "num_hidden_layers": 32,
    "mtp_num_hidden_layers": 1,
}


def test_a_qwen35_checkpoint_named_mimo_resolves_to_qwen35():
    assert model_family_from_inspection(DISTILL_INSPECTION) == "qwen3_5"
    assert model_family_from_inspection(DISTILL_INSPECTION, model_ref=DISTILL_REF) == "qwen3_5"


def test_a_qwen35_checkpoint_named_mimo_keeps_depth_and_reasoning():
    assert tune_policy_for_model(DISTILL_REF, DISTILL_INSPECTION).candidates == ("AR", "D1", "D2", "D3")
    assert reasoning_policy_for_model(DISTILL_REF, DISTILL_INSPECTION).supported is True


def test_a_mimo_checkpoint_in_a_mimo_named_folder_still_resolves_to_mimo():
    inspection = dict(
        MIMO_INSPECTION,
        model_dir="/Users/example/.mtplx/models/XiaomiMiMo--MiMo-7B-RL",
        runtime_model="XiaomiMiMo/MiMo-7B-RL",
    )
    assert model_family_from_inspection(inspection, model_ref="XiaomiMiMo/MiMo-7B-RL") == "mimo"
