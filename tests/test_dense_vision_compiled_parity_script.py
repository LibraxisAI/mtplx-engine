"""The dense parity driver's verdict, tested without a server.

``scripts/dense_vision_compiled_parity.py`` reads the dense request record
(``compiled_verify_admission``, ``compiled_verify.parity2``) and refuses to
call a run exact unless every image request took the compiled route with
its delta, dispatched compiled rounds, and had every one of them compared
without a difference.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dense_vision_compiled_parity.py"


def _load():
    spec = importlib.util.spec_from_file_location("dense_vision_compiled_parity", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = _load()

NATIVE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def _parity_record(*, image: bool, rounds=12, divergent=0, missing=0, logits=0.0, kl=0.0):
    return {
        "positions": "vision_delta" if image else "text",
        "rope_delta": -990 if image else None,
        "rounds": rounds,
        "rounds_by_width": {"4": rounds},
        "divergent_rounds": divergent,
        "divergent_rounds_by_width": {"4": divergent} if divergent else {},
        "reference_scope_missing_rounds": missing,
        "reference_scope_missing_by_width": {"4": missing} if missing else {},
        "compared_leaves_per_round": 40,
        "logits_max_abs_diff": logits,
        "logits_max_kl": kl,
        "hidden_max_abs_diff": 0.0,
        "state_max_abs_diff": 0.0,
        "capture_max_abs_diff": 0.0,
        "first_divergence": None if not divergent else {"leaf": "logits", "round": 3},
    }


def _entry(
    *,
    image: bool,
    mode="on",
    engaged=True,
    reason="admitted",
    sha="a",
    compiled_calls=12,
    cached_tokens=0,
    prompt_tokens=1200,
    **parity_kwargs,
):
    admission = {
        "family": "dense",
        "positions": "vision_delta" if image else "text",
        "rope_delta": -990 if image else None,
        "images": 1 if image else 0,
        "engaged": engaged,
        "reason": reason,
    }
    bank = {
        "mode": mode,
        "compiled_calls": compiled_calls,
        "fallback_calls": 0,
        "parity2_calls": compiled_calls if mode == "parity2" else 0,
        "rope_delta_input": image and engaged,
        "rope_delta": -990 if image and engaged else None,
    }
    if mode == "parity2":
        bank["parity2"] = _parity_record(image=image, **parity_kwargs)
    entry = {
        "kind": "image" if image else "text",
        "finish_reason": "stop",
        "output_sha256": sha,
        "compiled_verify_admission": admission,
        "cached_tokens": cached_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 256,
    }
    if engaged:
        entry["compiled_verify"] = bank
    return entry


def _arms(**instrument_image):
    def requests(mode, image_kwargs=None, *, eager=False):
        image_kwargs = dict(image_kwargs or {})
        if eager:
            image_kwargs.update(engaged=False, reason="vision_kill_switch")
        return {
            "image_turn1": _entry(image=True, mode=mode, **image_kwargs),
            "image_turn2": _entry(
                image=True, mode=mode, cached_tokens=1200, **image_kwargs
            ),
            "text_control": _entry(image=False, mode=mode),
        }

    return {
        "instrument": {
            "health": {"sampler": dict(NATIVE)},
            "requests": requests("parity2", instrument_image),
        },
        "product": {"health": {"sampler": dict(NATIVE)}, "requests": requests("on")},
        "eager": {"health": {"sampler": dict(NATIVE)}, "requests": requests("on", eager=True)},
    }


def _check(outcome, name):
    return next(c for c in outcome["checks"] if c["name"] == name)


def test_every_round_equal_and_every_check_held_is_exact():
    outcome = parity.evaluate(_arms(), native_sampler=NATIVE)
    assert outcome["verdict"] == "exact" and outcome["exit_code"] == 0
    assert outcome["failed_checks"] == []
    assert all(c["ok"] is True for c in outcome["checks"])
    assert "24 image rounds compared over 24 compiled dispatches" in _check(
        outcome, "instrument_image_rounds_exact"
    )["detail"]
    restored = _check(outcome, "product_second_turn_restored_the_image_turn")
    assert restored["ok"] is True and restored["required"] is False


def test_incomplete_image_turns_cannot_pass_even_with_exact_rounds():
    arms = _arms()
    arms["product"]["requests"]["image_turn1"]["finish_reason"] = "length"
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert outcome["exit_code"] == 1
    assert "image_turns_completed" in outcome["failed_checks"]


@pytest.mark.parametrize("thinking,expected", [("default", None), ("on", True), ("off", False)])
def test_explicit_request_thinking_control(thinking, expected):
    args = parity.parse_args(["--pack", "/unused", "--image", "/unused.png", "--thinking", thinking])
    body = parity._body([{"role": "user", "content": "Describe the image."}], args)
    if expected is None:
        assert "enable_thinking" not in body
    else:
        assert body["enable_thinking"] is expected


def test_one_divergent_image_round_is_never_exact():
    outcome = parity.evaluate(_arms(divergent=1, logits=3e-3, kl=1e-6), native_sampler=NATIVE)
    assert outcome["verdict"] == "image_route_diverges" and outcome["exit_code"] == 1
    assert "instrument_image_rounds_exact" in outcome["failed_checks"]


def test_a_divergence_the_size_of_the_text_controls_is_named_as_such():
    arms = _arms(divergent=2, logits=1e-4, kl=1e-9)
    text = arms["instrument"]["requests"]["text_control"]["compiled_verify"]
    text["parity2"] = _parity_record(image=False, divergent=5, logits=1.5e-4, kl=2.0e-9)
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert outcome["verdict"] == "diverges_like_text" and outcome["exit_code"] == 1
    text["parity2"] = _parity_record(image=False, divergent=5, logits=1.0e-6, kl=1.0e-12)
    assert parity.evaluate(arms, native_sampler=NATIVE)["verdict"] == "image_route_diverges"


def test_rounds_without_a_reference_are_said_and_not_exit_0():
    outcome = parity.evaluate(_arms(missing=2), native_sampler=NATIVE)
    assert outcome["verdict"] == "exact_on_compared_rounds" and outcome["exit_code"] == 1
    assert "instrument_every_compiled_image_round_was_compared" in outcome["failed_checks"]


def test_compiled_rounds_left_uncompared_are_not_exact():
    """An exact verdict needs every compiled dispatch behind a comparison."""
    # Both image requests dispatched 12 rounds and compared 10: four short.
    outcome = parity.evaluate(_arms(rounds=10), native_sampler=NATIVE)
    assert outcome["verdict"] == "exact_on_compared_rounds" and outcome["exit_code"] == 1
    assert "4 compiled rounds were not compared" in _check(
        outcome, "instrument_every_compiled_image_round_was_compared"
    )["detail"]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda e: e["compiled_verify_admission"].update(engaged=False, reason="vision_kill_switch"), "kept off the compiled route"),
        (lambda e: e.pop("compiled_verify_admission"), "no_admission_record"),
        (lambda e: e["compiled_verify"].pop("parity2"), "no_instrument_record"),
        (lambda e: e["compiled_verify"]["parity2"].update(rounds=0), "no verify round was compared"),
        (lambda e: e["compiled_verify"].update(compiled_calls=0), "dispatched no compiled round"),
        (lambda e: e["compiled_verify_admission"].update(positions="vision_sequential"), "did not take the image trace"),
        (lambda e: e["compiled_verify"].update(rope_delta_input=False), "did not take the image trace"),
    ],
)
def test_an_image_route_that_was_not_exercised_is_not_proven(mutate, fragment):
    arms = _arms()
    mutate(arms["instrument"]["requests"]["image_turn1"])
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert outcome["verdict"] == "not_proven" and outcome["exit_code"] == 1
    assert any(fragment in reason for reason in outcome["reasons"])


def test_the_product_arm_must_take_the_route_and_dispatch():
    arms = _arms()
    arms["product"]["requests"]["image_turn2"]["compiled_verify"].update(compiled_calls=0)
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert outcome["verdict"] == "exact"  # the instrument arm was fine
    assert "product_image_requests_take_the_compiled_route" in outcome["failed_checks"]
    assert outcome["exit_code"] == 1
    arms = _arms()
    arms["product"]["requests"]["image_turn2"]["output_sha256"] = "other"
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert "instrument_did_not_change_the_output" in outcome["failed_checks"]


def test_the_eager_arm_binds_the_free_running_check():
    arms = _arms()
    arms["eager"]["requests"]["image_turn2"]["output_sha256"] = "other"
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert "compiled_output_equals_eager_output" in outcome["failed_checks"]
    assert outcome["exit_code"] == 1
    arms = _arms()
    arms["eager"]["requests"]["image_turn1"]["compiled_verify_admission"]["reason"] = "admitted"
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert "eager_arm_ran_on_the_eager_verifier" in outcome["failed_checks"]


def test_a_server_that_does_not_apply_the_native_sampler_fails_the_run():
    arms = _arms()
    arms["product"]["health"]["sampler"] = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
    outcome = parity.evaluate(arms, native_sampler=NATIVE)
    assert "sampled_at_the_packs_native_settings" in outcome["failed_checks"]


def test_the_second_turn_carries_the_first_reply_not_a_canned_line(tmp_path):
    class Args:
        question = "What is in the picture?"
        follow_up = "And the upper left?"
        text_prompt = "Explain bridges."
        max_tokens = 64
        seed = 7

    plan = parity.request_plan("data:image/png;base64,AAAA", Args())
    assert plan["image_turn2"]["follows"] == "image_turn1"
    assert "body" not in plan["image_turn2"]
    first = {"body": plan["image_turn1"]["body"], "output_text": "A cat on a chair."}
    body = parity._second_turn(first, plan["image_turn2"], Args())
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert body["messages"][1]["content"] == "A cat on a chair."
    assert body["messages"][2]["content"] == "And the upper left?"
    assert body["seed"] == 7 and body["max_tokens"] == 64
    assert not any(key in body for key in ("temperature", "top_p", "top_k"))
