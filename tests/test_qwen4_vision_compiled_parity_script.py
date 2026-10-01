"""scripts/qwen4_vision_compiled_parity.py without a model.

Two halves. The verdict is a pure function of request records, so it is fed
records of every shape. The run (boot, ask, read the record, stop) is driven
against a stand-in server: a small HTTP process that accepts the same command
line, answers /health and /v1/chat/completions, and appends request-log lines
shaped like the product's, from the same environment switches. The instrument
records themselves are real in tests/test_qwen4_vision_compiled_route_e2e.py,
which feeds what the tiny pack produced through this same verdict.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import socket
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "qwen4_vision_compiled_parity.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("qwen4_vision_compiled_parity", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = _load()


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def _parity_record(*, image: bool, rounds=12, divergent=0, missing=0, logits=0.0, kl=0.0):
    return {
        "positions": "vision_delta" if image else "text",
        "rope_delta": -990 if image else None,
        "rounds": rounds,
        "compiled_rounds": rounds,
        "compiled_exact_rounds": rounds - divergent,
        "eager_rounds": 0,
        "eager_exact_rounds": 0,
        "rounds_by_width": {"4": rounds},
        "divergent_rounds": divergent,
        "divergent_rounds_by_width": {"4": divergent} if divergent else {},
        "reference_scope_missing_rounds": missing,
        "reference_scope_missing_by_width": {"4": missing} if missing else {},
        "compared_leaves_per_round": 90,
        "logits_max_abs_diff": logits,
        "logits_max_kl": kl,
        "hidden_max_abs_diff": logits / 2,
        "state_max_abs_diff": logits / 4,
        "capture_max_abs_diff": 0.0,
        "first_divergence": (
            {"round": 1, "width": 4, "leaf": "state[3:qsa].0", "report": ["state[3:qsa].0 ..."]}
            if divergent
            else None
        ),
    }


def _entry(*, image: bool, mode="on", engaged=True, reason="admitted", sha="a", **parity_kwargs):
    admission = {
        "requested_depth": 3,
        "engaged": engaged,
        "reason": reason,
        "positions": "vision_delta" if image else "text",
        "rope_delta": -990 if image else None,
        "images": 1 if image else 0,
    }
    entry = {
        "kind": "image" if image else "text",
        "output_sha256": sha,
        "finish_reason": "stop",
        "output_text": "A completed response.",
        "fixed_m4_admission": admission,
        "prompt_tokens": 1200 if image else 40,
        "completion_tokens": 48,
        "cached_tokens": 0,
        "session_cache_hit": False,
        "session_restore_mode": "cold",
    }
    if engaged:
        bank = {
            "mode": mode,
            "calls": 12,
            "compiled_calls": 12,
            "fallback_calls": 0,
            "fixed_m4": {"installed": True, "rope_delta_input": image, "rope_delta": admission["rope_delta"]},
        }
        if mode == "parity2":
            bank["fixed_m4_parity2"] = _parity_record(image=image, **parity_kwargs)
        entry["compiled_verify"] = bank
    return entry


def _arms(**instrument_image):
    """Three healthy arms; keyword arguments reshape the instrument's image rounds."""

    def requests(mode, image_kwargs=None, *, eager=False):
        image_kwargs = image_kwargs or {}
        if eager:
            image = dict(engaged=False, reason="vision_kill_switch")
        else:
            image = dict(mode=mode, **image_kwargs)
        records = {
            "image_turn1": _entry(image=True, sha="one", **image),
            "image_turn2": _entry(image=True, sha="two", **image),
            "text_control": _entry(image=False, mode=mode, sha="text"),
        }
        records["image_turn2"].update(
            prompt_tokens=1260, cached_tokens=1248,
            session_cache_hit=True, session_restore_mode="clone",
        )
        return records

    sampler = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    return {
        "instrument": {"health": {"sampler": sampler}, "requests": requests("parity2", instrument_image)},
        "product": {"health": {"sampler": sampler}, "requests": requests("on")},
        "eager": {"health": {"sampler": sampler}, "requests": requests("on", eager=True)},
    }


NATIVE = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def _evaluate(arms, *, fix=True):
    return parity.evaluate(arms, eager_scope_fix_present=fix, native_sampler=NATIVE)


def _check(outcome, name):
    (item,) = [c for c in outcome["checks"] if c["name"] == name]
    return item


def test_every_round_equal_and_every_check_held_is_exact():
    outcome = _evaluate(_arms())
    assert outcome["verdict"] == "exact"
    assert outcome["failed_checks"] == [] and outcome["exit_code"] == 0
    assert all(c["ok"] is True for c in outcome["checks"])
    assert {c["name"] for c in outcome["checks"]} == {
        "image_turns_completed",
        "sampled_at_the_packs_native_settings",
        "instrument_image_rounds_exact",
        "instrument_every_image_round_had_a_reference",
        "instrument_text_control_exact",
        "product_image_requests_take_the_compiled_route",
        "instrument_did_not_change_the_output",
        "eager_arm_ran_on_the_eager_verifier",
        "compiled_output_equals_eager_output",
        "compiled_arms_executed_compiled_verification",
        "follow_up_restored_generated_tokens",
    }
    assert outcome["instrument"]["image_turn1"]["state"] == "exact"
    assert all(r["restored_generated_tokens"] == 48 for r in outcome["generated_restore"].values())


@pytest.mark.parametrize("arm", ["instrument", "product"])
@pytest.mark.parametrize("request_name", ["image_turn1", "image_turn2", "text_control"])
def test_admission_and_eager_only_rounds_do_not_prove_a_compiled_dispatch(arm, request_name):
    arms = _arms()
    bank = arms[arm]["requests"][request_name]["compiled_verify"]
    bank["compiled_calls"] = 0
    if arm == "instrument":
        bank["fixed_m4_parity2"].update(
            compiled_rounds=0, compiled_exact_rounds=0,
            eager_rounds=12, eager_exact_rounds=12, rounds_by_width={"2": 12},
        )
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "no_compiled_dispatch"
    assert outcome["exit_code"] == 1
    assert "compiled_arms_executed_compiled_verification" in outcome["failed_checks"]


@pytest.mark.parametrize("request_name", ["image_turn1", "text_control"])
def test_eager_width_four_comparisons_are_not_compiled_comparisons(request_name):
    arms = _arms()
    record = arms["instrument"]["requests"][request_name]["compiled_verify"]["fixed_m4_parity2"]
    record.update(compiled_rounds=0, compiled_exact_rounds=0, eager_rounds=12, eager_exact_rounds=12)
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "no_compiled_comparisons"
    assert outcome["exit_code"] == 1


@pytest.mark.parametrize("arm", ["instrument", "product", "eager"])
@pytest.mark.parametrize("cached,hit,mode,reason", [
    (0, False, "cold", "cold_prefill"),
    (1200, True, "clone", "prompt_only_restore"),
    (1199, True, "clone", "prompt_only_restore"),
    (1248, False, "cold", "cold_prefill"),
])
def test_a_follow_up_must_restore_generated_rows_in_every_arm(arm, cached, hit, mode, reason):
    arms = _arms()
    arms[arm]["requests"]["image_turn2"].update(
        cached_tokens=cached, session_cache_hit=hit, session_restore_mode=mode,
    )
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "generated_state_not_restored"
    assert outcome["exit_code"] == 1
    assert outcome["generated_restore"][arm]["reason"] == reason


def test_the_reviewers_all_cold_records_cannot_be_exact():
    arms = _arms()
    for arm in arms.values():
        for entry in arm["requests"].values():
            entry.update(cached_tokens=0, session_cache_hit=False, session_restore_mode="cold")
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "generated_state_not_restored" and outcome["exit_code"] == 1


def test_one_generated_row_is_enough_but_missing_restore_evidence_is_not():
    arms = _arms()
    follow = arms["instrument"]["requests"]["image_turn2"]
    follow["cached_tokens"] = 1201
    assert _evaluate(arms)["verdict"] == "exact"
    follow.pop("cached_tokens")
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "generated_state_not_restored" and outcome["exit_code"] == 1
    assert outcome["generated_restore"]["instrument"]["reason"] == "missing_generated_restore_evidence"


def test_one_divergent_image_round_is_never_exact():
    outcome = _evaluate(_arms(divergent=1, logits=0.4375, kl=1.03e-2))
    assert outcome["verdict"] == "image_route_diverges"
    assert outcome["exit_code"] == 1
    assert "instrument_image_rounds_exact" in outcome["failed_checks"]
    assert outcome["instrument"]["image_turn1"]["first_divergence"]["leaf"] == "state[3:qsa].0"


def test_a_divergence_the_size_of_the_text_controls_is_named_as_such():
    arms = _arms(divergent=4, logits=2.0e-4, kl=3.0e-9)
    text = arms["instrument"]["requests"]["text_control"]["compiled_verify"]
    text["fixed_m4_parity2"] = _parity_record(image=False, divergent=5, logits=1.5e-4, kl=2.0e-9)
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "diverges_like_text"
    assert outcome["exit_code"] == 1  # a reading aid, never a pass
    assert _check(outcome, "instrument_text_control_exact")["ok"] is False

    # An image divergence far beyond the text control's is the image route's.
    text["fixed_m4_parity2"] = _parity_record(image=False, divergent=5, logits=1.0e-6, kl=1.0e-12)
    assert _evaluate(arms)["verdict"] == "image_route_diverges"


def test_rounds_without_a_reference_are_said_not_hidden():
    outcome = _evaluate(_arms(missing=3), fix=False)
    assert outcome["verdict"] == "exact_on_compared_rounds"
    assert outcome["exit_code"] == 1
    item = _check(outcome, "instrument_every_image_round_had_a_reference")
    # Three in each of the two image requests.
    assert item["ok"] is False and item["detail"].startswith("6 image rounds")
    assert "expected before the eager scope fix" in item["detail"]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda e: e["fixed_m4_admission"].update(engaged=False, reason="vision_tail_block_in_image"),
         "kept off the compiled route (vision_tail_block_in_image)"),
        (lambda e: e["fixed_m4_admission"].update(engaged=False, reason="memory_gate"),
         "kept off the compiled route (memory_gate)"),
        (lambda e: e.pop("fixed_m4_admission"), "no_admission_record"),
        (lambda e: e["compiled_verify"].pop("fixed_m4_parity2"), "no_instrument_record"),
        (lambda e: e["compiled_verify"]["fixed_m4_parity2"].update(rounds=0), "no verify round was compared"),
        (lambda e: e.update(error="HTTP 500: boom"), "request_failed"),
        (lambda e: e["fixed_m4_admission"].update(positions="vision_sequential"),
         "did not take the image trace"),
        (lambda e: e["compiled_verify"]["fixed_m4"].update(rope_delta_input=False),
         "did not take the image trace"),
    ],
)
def test_an_image_route_that_was_not_exercised_is_not_proven(mutate, fragment):
    arms = _arms()
    mutate(arms["instrument"]["requests"]["image_turn1"])
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "not_proven" and outcome["exit_code"] == 1
    assert any(fragment in reason for reason in outcome["reasons"]), outcome["reasons"]


def test_the_product_arm_must_take_the_route_and_say_the_same_thing():
    arms = _arms()
    arms["product"]["requests"]["image_turn2"]["fixed_m4_admission"].update(
        engaged=False, reason="memory_gate"
    )
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "exact" and outcome["exit_code"] == 1
    assert outcome["failed_checks"] == ["product_image_requests_take_the_compiled_route"]

    arms = _arms()
    arms["product"]["requests"]["text_control"]["output_sha256"] = "other"
    outcome = _evaluate(arms)
    assert outcome["failed_checks"] == ["instrument_did_not_change_the_output"]
    assert "text_control" in _check(outcome, "instrument_did_not_change_the_output")["detail"]


def test_the_eager_comparison_binds_only_on_a_tree_with_the_eager_scope_fix():
    arms = _arms()
    arms["eager"]["requests"]["image_turn1"]["output_sha256"] = "parted"
    with_fix = _evaluate(arms, fix=True)
    assert with_fix["failed_checks"] == ["compiled_output_equals_eager_output"]
    assert with_fix["exit_code"] == 1

    without_fix = _evaluate(arms, fix=False)
    item = _check(without_fix, "compiled_output_equals_eager_output")
    assert item["ok"] is False and item["required"] is False
    assert "not a fair reference" in item["detail"]
    assert without_fix["failed_checks"] == [] and without_fix["exit_code"] == 0

    # An eager arm that was not eager proves nothing about the eager route.
    arms = _arms()
    arms["eager"]["requests"]["image_turn1"] = _entry(image=True, sha="one")
    assert "eager_arm_ran_on_the_eager_verifier" in _evaluate(arms)["failed_checks"]


def test_a_server_that_does_not_apply_the_native_sampler_fails_the_run():
    arms = _arms()
    arms["product"]["health"]["sampler"] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    outcome = _evaluate(arms)
    assert outcome["failed_checks"] == ["sampled_at_the_packs_native_settings"]
    assert "product applies" in _check(outcome, "sampled_at_the_packs_native_settings")["detail"]


@pytest.mark.parametrize("finish,content", [("length", "partial"), ("stop", ""), (None, "partial")])
def test_exact_rounds_and_a_cache_hit_do_not_prove_a_completed_image_turn(finish, content):
    arms = _arms()
    arms["product"]["requests"]["image_turn1"].update(
        finish_reason=finish, output_text=content,
    )
    outcome = _evaluate(arms)
    assert outcome["failed_checks"] == ["image_turns_completed"]
    assert outcome["exit_code"] == 1


def test_the_instrument_arm_alone_is_enough_for_a_verdict():
    arms = {"instrument": _arms()["instrument"]}
    outcome = _evaluate(arms)
    assert outcome["verdict"] == "exact" and outcome["exit_code"] == 0
    assert "product_image_requests_take_the_compiled_route" not in {
        c["name"] for c in outcome["checks"]
    }


@pytest.mark.parametrize("first_fails", [False, True])
def test_follow_up_uses_each_arms_actual_response_without_mutating_the_plan(
    monkeypatch, tmp_path, first_fails
):
    """Exercise run_arm's request dependency without a subprocess or socket."""
    args = parity.parse_args(["--pack", str(tmp_path), "--image", "/unused.png"])
    plan = parity.build_requests("data:image/png;base64,cGl4ZWxz", args)
    original = json.dumps(plan)
    monkeypatch.setattr(parity, "memory_guard", lambda _: {"checked": False})
    monkeypatch.setattr(parity, "_port_answers", lambda *_: False)
    monkeypatch.setattr(parity, "_start_server", lambda *_: SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(parity, "_stop_server", lambda _: None)
    monkeypatch.setattr(parity, "_wait_for_health", lambda *_: {"profile": {"sampler": NATIVE}})
    monkeypatch.setattr(parity, "_read_record", lambda *_: {})
    for arm in parity.ARMS:
        sent = []
        message = {"role": "assistant", "content": f"actual {arm} reply",
                   "reasoning_content": f"actual {arm} reasoning"}

        def respond(method, url, body, timeout):
            sent.append(body)
            if len(sent) == 1 and first_fails:
                raise RuntimeError("first response failed")
            return {"choices": [{"message": message, "finish_reason": "stop"}]}

        monkeypatch.setattr(parity, "_http_json", respond)
        result = parity.run_arm(arm, args, plan, tmp_path)
        if first_fails:
            assert len(sent) == 2  # first image and independent text control only
            assert result["requests"]["image_turn2"]["error"] == (
                "the first response is unavailable for the follow-up"
            )
        else:
            assert len(sent) == 3
            assert sent[1]["messages"][-2] == message
            assert sent[1]["messages"][-1] == {"role": "user", "content": args.follow_up}
            assert result["requests"]["image_turn1"]["response_message"] == message
        assert json.dumps(plan) == original


# ---------------------------------------------------------------------------
# The run, against a stand-in server
# ---------------------------------------------------------------------------

_STUB = textwrap.dedent(
    '''
    """Stand-in for `mtplx serve`: same command line, same record shapes."""
    import argparse, hashlib, json, os, time, uuid
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--no-auth", action="store_true", required=True)
    parser.add_argument("--profile")
    args = parser.parse_args()

    LOG = os.environ["MTPLX_REQUEST_LOG_JSONL"]
    MODE = os.environ.get("MTPLX_COMPILED_VERIFY") or "1"  # the product pins 1
    KILL = os.environ.get("MTPLX_QWEN4_VISION_COMPILED_VERIFY") == "0"
    DIVERGENT = int(os.environ.get("STUB_DIVERGENT", "0"))
    first_reply = None

    def log(row):
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"logged_at_s": time.time(), **row}) + "\\n")

    log({"warmup": True, "fixed_m4_admission": {"engaged": True, "positions": "text"}})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {})
            self._send(200, {"ok": True, "model": "stub", "model_path": args.model,
                             "runtime_mode": "Turbo", "profile": {"name": args.profile or "turbo",
                             "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}})

        def do_POST(self):
            global first_reply
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if {"temperature", "top_p", "top_k"} & set(body) or "seed" not in body or body.get("stream"):
                return self._send(400, {"error": "native sampler, a seed and no stream expected"})
            image = any(
                isinstance(m.get("content"), list)
                and any(p.get("type") == "image_url" for p in m["content"])
                for m in body["messages"]
            )
            rid = "chatcmpl-" + uuid.uuid4().hex
            digest = hashlib.sha256(json.dumps([body["messages"], body["seed"]]).encode()).hexdigest()
            if os.environ.get("STUB_SALT_BY_MODE"):
                digest = hashlib.sha256((digest + MODE).encode()).hexdigest()
            admission = {"requested_depth": 3, "positions": "vision_delta" if image else "text",
                         "rope_delta": -990 if image else None, "images": 1 if image else 0}
            record = {"request_id": rid, "prompt_tokens": 1200 if image else 40,
                      "completion_tokens": body["max_tokens"], "server_seed": body["seed"],
                      "cached_tokens": 0, "session_cache_hit": False, "session_restore_mode": "cold",
                      "fixed_m4_admission": admission}
            if image and len(body["messages"]) > 1:
                if body["messages"][-2] != first_reply:
                    return self._send(400, {"error": "follow-up did not use the actual first response"})
                record.update(prompt_tokens=1200 + body["max_tokens"] + 12,
                              cached_tokens=1200 + body["max_tokens"],
                              session_cache_hit=True, session_restore_mode="clone")
            if image and KILL:
                admission.update(engaged=False, reason="vision_kill_switch")
                record["demotions"] = {"vision_request_eager_verify": 1}
            else:
                admission.update(engaged=True, reason="admitted")
                bank = {"mode": "parity2" if MODE == "parity2" else "on", "calls": 12,
                        "compiled_calls": 12, "fallback_calls": 0,
                        "compiled_keys": ["m4:post_norm:b0" + (":rope_delta" if image else "")],
                        "fixed_m4": {"installed": True, "rope_delta_input": image,
                                     "rope_delta": admission["rope_delta"]}}
                if MODE == "parity2":
                    bad = DIVERGENT if image else 0
                    bank["fixed_m4_parity2"] = {
                        "positions": admission["positions"], "rope_delta": admission["rope_delta"],
                        "rounds": 12, "rounds_by_width": {"4": 12}, "divergent_rounds": bad,
                        "compiled_rounds": 12, "compiled_exact_rounds": 12 - bad,
                        "eager_rounds": 0, "eager_exact_rounds": 0,
                        "reference_scope_missing_rounds": 0, "logits_max_abs_diff": 0.5 if bad else 0.0,
                        "logits_max_kl": 0.01 if bad else 0.0, "hidden_max_abs_diff": 0.0,
                        "state_max_abs_diff": 0.0, "capture_max_abs_diff": 0.0,
                        "first_divergence": {"round": 1, "leaf": "logits"} if bad else None}
                record["compiled_verify"] = bank
            reply = {"role": "assistant", "content": "echo " + digest[:16], "reasoning_content": "hm"}
            if image and len(body["messages"]) == 1:
                first_reply = reply
            self._send(200, {"id": rid, "choices": [{"finish_reason": os.environ.get("STUB_FINISH", "stop"), "message": reply}],
                "usage": {"completion_tokens": body["max_tokens"]}})
            time.sleep(0.3)  # the product writes its line as the request finishes
            log(record)

    HTTPServer((args.host, args.port), Handler).serve_forever()
    '''
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def stand_in(tmp_path, monkeypatch):
    stub = tmp_path / "stub_serve.py"
    stub.write_text(_STUB, encoding="utf-8")
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "mtplx_runtime.json").write_text(json.dumps({"sampler": NATIVE}), encoding="utf-8")
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"pixels")
    # What an operator's shell might hold: the product arm must not inherit it.
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "parity")
    monkeypatch.setenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "0")
    monkeypatch.setenv("MTPLX_API_KEY", "not-for-the-report")
    port = _free_port()

    def run(*extra: str):
        out = tmp_path / "run" / "report.json"
        argv = [
            "--pack", str(pack), "--image", str(image), "--out", str(out),
            "--port", str(port), "--serve-command", shlex.join([sys.executable, str(stub)]),
            "--boot-timeout", "60", "--record-timeout", "20", *extra,
        ]
        code = parity.main(argv)
        report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        return code, report, port

    return run


def test_a_full_run_boots_each_arm_asks_the_same_three_things_and_stops(stand_in):
    code, report, port = stand_in()
    assert code == 0 and report["verdict"] == "exact" and report["exit_code"] == 0
    assert report["schema"] == parity.SCHEMA
    assert list(report["arms"]) == ["instrument", "product", "eager"]
    assert report["failed_checks"] == [] and report["arm_errors"] == {}
    assert report["pack"]["native_sampler"] == NATIVE

    # Each boot got its arm's switches and none of the shell's.
    env = {arm: data["env"] for arm, data in report["arms"].items()}
    assert env["instrument"]["MTPLX_COMPILED_VERIFY"] == "parity2"
    assert "MTPLX_QWEN4_VISION_COMPILED_VERIFY" not in env["instrument"]
    assert "MTPLX_COMPILED_VERIFY" not in env["product"]
    assert "MTPLX_QWEN4_VISION_COMPILED_VERIFY" not in env["product"]
    assert env["eager"]["MTPLX_QWEN4_VISION_COMPILED_VERIFY"] == "0"
    assert "MTPLX_COMPILED_VERIFY" not in env["eager"]
    assert all("MTPLX_API_KEY" not in values for values in env.values())

    for arm, data in report["arms"].items():
        assert list(data["requests"]) == ["image_turn1", "image_turn2", "text_control"]
        assert data["health"]["sampler"] == NATIVE
        for name, entry in data["requests"].items():
            assert entry["request_id"] == entry["response_id"], (arm, name)
            assert entry["output_text"].startswith("echo ") and entry["reasoning_text"] == "hm"
            assert entry["fixed_m4_admission"]["positions"] == (
                "text" if name == "text_control" else "vision_delta"
            )
    instrument = report["arms"]["instrument"]["requests"]
    assert instrument["image_turn1"]["compiled_verify"]["fixed_m4_parity2"]["rounds"] == 12
    assert report["arms"]["eager"]["requests"]["image_turn1"]["fixed_m4_admission"]["reason"] == (
        "vision_kill_switch"
    )
    # The same bytes went to every arm, and the second turn is not the first.
    shas = {arm: [e["output_sha256"] for e in d["requests"].values()] for arm, d in report["arms"].items()}
    assert shas["instrument"] == shas["product"] == shas["eager"]
    assert len(set(shas["product"])) == 3
    # Nothing it started is still up.
    assert not parity._port_answers("127.0.0.1", port)


def test_divergent_image_rounds_fail_the_run(stand_in):
    code, report, _ = stand_in("--env", "STUB_DIVERGENT=3", "--arms", "instrument")
    assert code == 1 and report["verdict"] == "image_route_diverges"
    assert report["instrument_summary"]["image_turn1"]["divergent_rounds"] == 3
    assert report["instrument_summary"]["text_control"]["state"] == "exact"


def test_truncated_reply_is_not_sent_as_a_generated_state_follow_up(stand_in):
    code, report, _ = stand_in("--env", "STUB_FINISH=length", "--arms", "instrument")
    assert code == 1
    assert "image_turns_completed" in report["failed_checks"]
    follow = report["arms"]["instrument"]["requests"]["image_turn2"]
    assert "increase --max-tokens" in follow["error"]
    assert "response_id" not in follow


def test_an_instrument_that_changes_the_output_fails_the_run(stand_in):
    code, report, _ = stand_in("--env", "STUB_SALT_BY_MODE=1", "--arms", "instrument,product")
    assert code == 1 and report["verdict"] == "exact"
    assert report["failed_checks"] == ["instrument_did_not_change_the_output"]


def test_it_never_boots_onto_a_port_that_answers(stand_in, tmp_path):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        busy = str(listener.getsockname()[1])
        code, report, _ = stand_in("--port", busy)
    assert code == 1 and report["verdict"] == "not_proven"
    assert "already answers" in report["arm_errors"]["instrument"]
    assert report["arms_not_run"] == ["product", "eager"]
    assert not (tmp_path / "run" / "instrument.server.log").exists()


def test_it_refuses_to_load_a_pack_that_cannot_fit(stand_in, monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(
        parity,
        "memory_guard",
        lambda pack: {"checked": True, "ok": False, "weights_bytes": 100 * gib,
                      "free_bytes": 54 * gib, "needed_bytes": 75 * gib, "measured_by": "test"},
    )
    code, report, _ = stand_in()
    assert code == 1
    assert "another engine probably holds a model" in report["arm_errors"]["instrument"]
    assert report["arms"]["instrument"]["requests"] == {}

    code, report, _ = stand_in("--skip-memory-guard", "--arms", "instrument")
    assert code == 0 and report["verdict"] == "exact"


def test_the_memory_guard_reads_the_weight_files(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "model-00001-of-00002.safetensors").write_bytes(b"x" * 4000)
    (pack / "mtp.safetensors").write_bytes(b"x" * 1000)
    (pack / "config.json").write_text("{}", encoding="utf-8")
    import mtplx.ple_row_gather as ple_row_gather

    monkeypatch.setattr(ple_row_gather, "free_memory_bytes", lambda: (3749, "test"))
    guard = parity.memory_guard(pack)
    assert guard["weights_bytes"] == 5000 and guard["needed_bytes"] == 3750
    assert guard["ok"] is False
    monkeypatch.setattr(ple_row_gather, "free_memory_bytes", lambda: (3750, "test"))
    assert parity.memory_guard(pack)["ok"] is True
    assert parity.memory_guard(tmp_path / "not-a-pack")["checked"] is False


def test_dry_run_prints_the_plan_and_starts_nothing(stand_in, capsys, tmp_path):
    code, report, port = stand_in("--dry-run", "--profile", "turbo", "--serve-arg=--context-window", "--serve-arg=65536")
    assert code == 0 and report is None and not (tmp_path / "run").exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["serve_command"][-7:] == [
        "--port", str(port), "--no-auth", "--profile", "turbo", "--context-window", "65536",
    ]
    assert plan["arms"]["instrument"]["set"] == {"MTPLX_COMPILED_VERIFY": "parity2"}
    assert plan["requests"]["image_turn2"]["messages"] == 3
    assert all(r["sampler_fields_sent"] == [] for r in plan["requests"].values())


def test_the_default_command_is_the_products_own_server():
    args = parity.parse_args(["--pack", "/p", "--image", "/i.png"])
    command = parity.serve_command(args)
    assert command[1:4] == ["-m", "mtplx.cli", "serve"]
    assert command[4:] == ["--model", "/p", "--host", "127.0.0.1", "--port", "18473", "--no-auth"]
    assert args.arm_list == ["instrument", "product", "eager"]
    with pytest.raises(SystemExit):
        parity.parse_args(["--pack", "/p", "--image", "/i.png", "--arms", "product"])
