#!/usr/bin/env python3
"""Real-pack proof that dense Qwen3.8 image requests verify exactly on the compiled route.

One command, the 27B pack and an image:

    PYTHONPATH=$PWD python scripts/dense_vision_compiled_parity.py \\
        --pack ~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed \\
        --image photo.jpg

The dense sibling of ``scripts/qwen4_vision_compiled_parity.py``, on the same
server harness: it starts the product server (``mtplx serve``) once per arm,
sends the same requests to each boot, reads every request's record from the
request log and writes one JSON report. Exit code 0 means every check held.

The arms (one boot each, one at a time; ``ARMS`` of the Flash-Next script):

  instrument  ``MTPLX_COMPILED_VERIFY=parity2``. The compiled verifier stays
              authoritative. Every compiled verify round is also run on the
              eager verifier, on a copy of the same pre-round buffers whose
              rows are positioned from the request's own image position table
              (the dense adapter's host-plan scope), and the logits, the
              hidden state, the captures and every cache leaf are compared bit
              for bit on the device (``compiled_verify.parity2`` in the request
              record).
  product     The shipping configuration with nothing overridden: an image
              request is admitted to the compiled route as shipped, and the
              instrument did not change what the lane generates.
  eager       ``MTPLX_QWEN4_VISION_COMPILED_VERIFY=0``, the kill switch (one
              knob for both families): the same image requests on the eager
              verifier, same seed, so the text must be the product arm's text.
              This free-running check covers what a per-round instrument
              cannot see: repair forwards, the final commit, and the rows the
              compiled route banked, which the second turn restores.

The requests (identical bytes in every arm, except where the model's own
words are carried forward):

  image_turn1   the image and a question;
  image_turn2   a second turn with the image in the history and the model's
                OWN first reply as the assistant message (not a canned line),
                so the rows the compiled route generated and banked are the
                rows the restore hands the second turn;
  text_control  a text-only request: what the compiled step and the eager
                forward agree to on this pack when no image is involved.

Sampling: no sampler field is sent, so the server applies the pack's native
settings (``mtplx_runtime.json``), with one fixed seed. Greedy is not used.
Use ``--thinking on`` or ``--thinking off`` for explicit request controls,
and a request budget large enough to finish both image turns. An image turn
whose finish reason is not ``stop`` fails the gate. Dense image compilation
is currently opt-in: testing that candidate requires
``--env MTPLX_DENSE_VISION_COMPILED_VERIFY=1``. Without it, the default eager
admission is reported and does not qualify as compiled-route proof.

How to read the verdict:

  exact                     every image request took the compiled route with
                            its delta, dispatched compiled rounds, every one of
                            them was compared and none differed, and every
                            other check held.
  exact_on_compared_rounds  as above, but some compiled rounds had no reference
                            (their caller had no position scope open); they
                            are counted in ``reference_scope_missing_rounds``.
                            Not exit 0: the proof is incomplete.
  diverges_like_text        image rounds differ, the text control differs too,
                            and by a similar size: the compiled step is not bit
                            exact on this pack with or without an image.
  image_route_diverges      image rounds differ and text rounds do not (or
                            differ by much less): look at ``first_divergence``.
  not_proven                the image route was not exercised: refused, no
                            compiled dispatch, no record. The reasons are
                            listed.

Safety: as the Flash-Next script. It never stops a process it did not start,
refuses a port that answers, and refuses to load the pack when free memory is
below three quarters of its weight files. It takes no speed measurement and
none of its numbers is one. ``--dry-run`` prints the plan and starts nothing.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen4_vision_compiled_parity as harness

ARMS = harness.ARMS
DEFAULT_ARMS = harness.DEFAULT_ARMS
SCHEMA = "dense_vision_compiled_parity/1"
SAME_CLASS_FACTOR = harness.SAME_CLASS_FACTOR
_PARITY_MAXIMA = harness._PARITY_MAXIMA
_RECORD_KEYS = tuple(dict.fromkeys((*harness._RECORD_KEYS, "compiled_verify_admission")))


# ---------------------------------------------------------------------------
# The verdict: pure functions over request records, tested without a server.
# ---------------------------------------------------------------------------


def request_status(entry: dict[str, Any]) -> dict[str, Any]:
    """What one request's record says about the dense compiled verify route."""

    admission = entry.get("compiled_verify_admission") or {}
    bank = entry.get("compiled_verify") or {}
    parity = bank.get("parity2")
    compiled_calls = int(bank.get("compiled_calls") or 0)
    status: dict[str, Any] = {
        "engaged": bool(admission.get("engaged")),
        "admission_reason": admission.get("reason"),
        "positions": admission.get("positions"),
        "rope_delta": admission.get("rope_delta"),
        "images": admission.get("images"),
        "rope_delta_input": bool(bank.get("rope_delta_input")),
        "bank_mode": bank.get("mode"),
        "compiled_calls": compiled_calls,
        "fallback_calls": bank.get("fallback_calls"),
        "cached_tokens": entry.get("cached_tokens"),
        "prompt_tokens": entry.get("prompt_tokens"),
        "completion_tokens": entry.get("completion_tokens"),
    }
    if entry.get("error"):
        status["state"] = "request_failed"
    elif not admission:
        status["state"] = "no_admission_record"
    elif not admission.get("engaged"):
        status["state"] = "not_engaged"
    elif compiled_calls == 0:
        # Engaged means the bank was built; a verdict needs it to have run.
        status["state"] = "no_compiled_dispatch"
    elif not isinstance(parity, dict):
        status["state"] = "no_instrument_record"
    else:
        rounds = int(parity.get("rounds") or 0)
        divergent = int(parity.get("divergent_rounds") or 0)
        status.update(
            rounds=rounds,
            divergent_rounds=divergent,
            reference_scope_missing_rounds=int(parity.get("reference_scope_missing_rounds") or 0),
            rounds_by_width=parity.get("rounds_by_width") or {},
            first_divergence=parity.get("first_divergence"),
            **{name: float(parity.get(name) or 0.0) for name in _PARITY_MAXIMA},
        )
        if rounds == 0:
            status["state"] = "no_compared_rounds"
        else:
            status["state"] = "divergent" if divergent else "exact"
    return status


def _worst(statuses: list[dict[str, Any]], name: str) -> float:
    return max((float(item.get(name) or 0.0) for item in statuses), default=0.0)


def evaluate(
    arms: dict[str, Any],
    *,
    native_sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The verdict, the checks behind it, and the exit code.

    ``arms`` maps an arm name to ``{"requests": {name: entry}}``; an entry
    holds ``kind`` (``image`` | ``text``), ``output_sha256`` and the record
    fields. ``native_sampler`` is the pack's own sampler block; each arm's
    ``health.sampler`` is what its server applies to a request that sends no
    sampler field.
    """

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool | None, detail: str, *, required: bool = True):
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})

    def requests_of(arm: str) -> dict[str, Any]:
        return (arms.get(arm) or {}).get("requests") or {}

    incomplete = [
        f"{arm}/{name}: {entry.get('finish_reason')!r}"
        for arm in arms
        for name, entry in requests_of(arm).items()
        if entry.get("kind") == "image" and entry.get("finish_reason") != "stop"
    ]
    check(
        "image_turns_completed",
        not incomplete,
        "; ".join(incomplete) or "every image turn finished with stop",
    )

    applied = {
        arm: data["health"].get("sampler")
        for arm, data in arms.items()
        if isinstance(data.get("health"), dict)
    }
    if native_sampler and applied:
        wrong = [
            arm for arm, sampler in applied.items()
            if not harness._same_sampler(sampler, native_sampler)
        ]
        check(
            "sampled_at_the_packs_native_settings",
            not wrong,
            f"the pack declares {native_sampler}; "
            + (
                "every arm's server applies it"
                if not wrong
                else "; ".join(f"{arm} applies {applied[arm]}" for arm in wrong)
            ),
        )

    instrument = {name: request_status(e) for name, e in requests_of("instrument").items()}
    kinds = {name: e.get("kind") for name, e in requests_of("instrument").items()}
    image = {n: s for n, s in instrument.items() if kinds.get(n) == "image"}
    text = {n: s for n, s in instrument.items() if kinds.get(n) == "text"}

    reasons: list[str] = []
    if not image:
        reasons.append("the instrument arm holds no image request")
    for name, status in image.items():
        if status["state"] in {"request_failed", "no_admission_record", "no_instrument_record"}:
            reasons.append(f"{name}: {status['state']}")
        elif status["state"] == "not_engaged":
            reasons.append(f"{name}: kept off the compiled route ({status['admission_reason']})")
        elif status["state"] == "no_compiled_dispatch":
            reasons.append(f"{name}: the bank was built but dispatched no compiled round")
        elif status["state"] == "no_compared_rounds":
            reasons.append(f"{name}: no verify round was compared")
        elif status["positions"] != "vision_delta" or not status["rope_delta_input"]:
            reasons.append(
                f"{name}: positions {status['positions']!r}, rope_delta_input "
                f"{status['rope_delta_input']}: the request did not take the image trace"
            )

    image_list = list(image.values())
    text_list = [s for s in text.values() if s["state"] in {"exact", "divergent"}]
    unreferenced = sum(int(s.get("reference_scope_missing_rounds") or 0) for s in image_list)
    uncompared = sum(
        max(0, int(s.get("compiled_calls") or 0) - int(s.get("rounds") or 0) - int(s.get("reference_scope_missing_rounds") or 0))
        for s in image_list
    )
    if reasons:
        verdict = "not_proven"
    elif all(s["state"] == "exact" for s in image_list):
        verdict = "exact_on_compared_rounds" if (unreferenced or uncompared) else "exact"
    else:
        text_divergent = [s for s in text_list if s["state"] == "divergent"]
        same_class = bool(text_divergent) and all(
            _worst(image_list, name) <= SAME_CLASS_FACTOR * _worst(text_divergent, name)
            for name in ("logits_max_abs_diff", "logits_max_kl")
        )
        verdict = "diverges_like_text" if same_class else "image_route_diverges"

    check(
        "instrument_image_rounds_exact",
        verdict in {"exact", "exact_on_compared_rounds"},
        "; ".join(reasons)
        or (
            f"{sum(int(s.get('rounds') or 0) for s in image_list)} image rounds compared over "
            f"{sum(int(s.get('compiled_calls') or 0) for s in image_list)} compiled dispatches, "
            f"{sum(int(s.get('divergent_rounds') or 0) for s in image_list)} divergent, "
            f"largest logit difference {_worst(image_list, 'logits_max_abs_diff'):.3g}, "
            f"largest KL {_worst(image_list, 'logits_max_kl'):.3g}"
        ),
    )
    check(
        "instrument_every_compiled_image_round_was_compared",
        unreferenced == 0 and uncompared == 0,
        f"{unreferenced} image rounds had no position scope to take a reference from, "
        f"{uncompared} compiled rounds were not compared",
    )
    if text:
        check(
            "instrument_text_control_exact",
            bool(text_list) and all(s["state"] == "exact" for s in text_list),
            (
                f"{sum(int(s.get('rounds') or 0) for s in text_list)} text rounds compared, "
                f"{sum(int(s.get('divergent_rounds') or 0) for s in text_list)} divergent, "
                f"largest logit difference {_worst(text_list, 'logits_max_abs_diff'):.3g}"
            )
            if text_list
            else "the text control produced no compared round: "
            + ", ".join(f"{n}: {s['state']}" for n, s in text.items()),
        )

    def outputs_equal(left_arm: str, right_arm: str, kind: str | None):
        left, right = requests_of(left_arm), requests_of(right_arm)
        names = [
            name
            for name in left
            if name in right and (kind is None or left[name].get("kind") == kind)
        ]
        different = [
            name
            for name in names
            if not left[name].get("output_sha256")
            or left[name].get("output_sha256") != right[name].get("output_sha256")
        ]
        return names, different

    def restored_second_turn(arm: str) -> tuple[bool | None, str]:
        turn1 = request_status(requests_of(arm).get("image_turn1") or {})
        turn2 = request_status(requests_of(arm).get("image_turn2") or {})
        cached = turn2.get("cached_tokens")
        prompt1 = turn1.get("prompt_tokens")
        if cached is None or prompt1 is None:
            return None, f"{arm}: no token counts in the records"
        generated = int(turn1.get("completion_tokens") or 0)
        restored_generated = max(0, int(cached) - int(prompt1))
        detail = (
            f"{arm}: the second turn restored {cached} tokens over a first-turn prompt of "
            f"{prompt1}; {restored_generated} of the {generated} generated rows came back "
            "from the bank (a thinking model's reply is carried without its reasoning, "
            "so the restore ends where the templates diverge)"
        )
        return int(cached) >= int(prompt1), detail

    if "product" in arms:
        product = {n: request_status(e) for n, e in requests_of("product").items()}
        product_kinds = {n: e.get("kind") for n, e in requests_of("product").items()}
        off_route = [
            f"{name}: {status['state']}"
            + (f" ({status['admission_reason']})" if status.get("admission_reason") else "")
            + f", positions {status['positions']!r}, rope_delta_input {status['rope_delta_input']}"
            + f", compiled_calls {status['compiled_calls']}"
            for name, status in product.items()
            if product_kinds.get(name) == "image"
            and not (
                status["engaged"]
                and status["positions"] == "vision_delta"
                and status["rope_delta_input"]
                and status["bank_mode"] == "on"
                and int(status["compiled_calls"] or 0) >= 1
            )
        ]
        has_image = any(kind == "image" for kind in product_kinds.values())
        check(
            "product_image_requests_take_the_compiled_route",
            has_image and not off_route,
            "; ".join(off_route)
            or (
                "every image request was admitted with its rotary delta as a trace input "
                "and dispatched compiled rounds"
                if has_image
                else "the product arm holds no image request"
            ),
        )
        names, different = outputs_equal("instrument", "product", None)
        check(
            "instrument_did_not_change_the_output",
            bool(names) and not different,
            f"same text in both arms for {len(names) - len(different)} of {len(names)} requests"
            + (f"; different: {', '.join(different)}" if different else ""),
        )
        ok, detail = restored_second_turn("product")
        check("product_second_turn_restored_the_image_turn", ok, detail, required=False)

    if "eager" in arms:
        eager = {n: request_status(e) for n, e in requests_of("eager").items()}
        eager_kinds = {n: e.get("kind") for n, e in requests_of("eager").items()}
        not_eager = [
            f"{name}: {status['admission_reason']!r}"
            for name, status in eager.items()
            if eager_kinds.get(name) == "image"
            and status["admission_reason"] != "vision_kill_switch"
        ]
        check(
            "eager_arm_ran_on_the_eager_verifier",
            not not_eager,
            "; ".join(not_eager) or "every image request was kept eager by the kill switch",
        )
        reference = "product" if "product" in arms else "instrument"
        names, different = outputs_equal(reference, "eager", "image")
        check(
            "compiled_output_equals_eager_output",
            bool(names) and not different,
            f"same text on the compiled and the eager route for "
            f"{len(names) - len(different)} of {len(names)} image requests"
            + (f"; different: {', '.join(different)}" if different else ""),
        )

    failed = [c["name"] for c in checks if c["required"] and c["ok"] is not True]
    return {
        "verdict": verdict,
        "reasons": reasons,
        "checks": checks,
        "failed_checks": failed,
        "instrument": instrument,
        "exit_code": 0 if verdict == "exact" and not failed else 1,
    }


# ---------------------------------------------------------------------------
# The run: boot, ask (the second turn from the first answer), read, stop.
# ---------------------------------------------------------------------------


def _body(messages: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    # No temperature, top_p or top_k: the server applies the pack's native
    # sampler. One seed for every request of every arm.
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed),
        "stream": False,
    }
    thinking = getattr(args, "thinking", "default")
    if thinking != "default":
        body["enable_thinking"] = thinking == "on"
    return body


def request_plan(image_url: str, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """The requests of one arm; ``image_turn2`` is completed from the first reply."""

    image_part = {"type": "image_url", "image_url": {"url": image_url}}
    first_turn = [{"role": "user", "content": [image_part, {"type": "text", "text": args.question}]}]
    return {
        "image_turn1": {"kind": "image", "body": _body(first_turn, args)},
        "image_turn2": {"kind": "image", "follows": "image_turn1", "follow_up": args.follow_up},
        "text_control": {
            "kind": "text",
            "body": _body([{"role": "user", "content": args.text_prompt}], args),
        },
    }


def _second_turn(first: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reply = str(first.get("output_text") or "")
    messages = list(first["body"]["messages"]) + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": spec["follow_up"]},
    ]
    return _body(messages, args)


def run_arm(
    arm: str, args: argparse.Namespace, plan: dict[str, dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    request_log = run_dir / f"{arm}.request-log.jsonl"
    server_log = run_dir / f"{arm}.server.log"
    env = harness._child_env(arm, request_log, dict(args.env or []))
    result: dict[str, Any] = {
        "env": harness._reportable_env(env),
        "env_unset": list(ARMS[arm]["unset"]),
        "server_log": str(server_log),
        "request_log": str(request_log),
        "requests": {},
    }
    guard = harness.memory_guard(Path(args.pack).expanduser())
    result["memory_guard"] = guard
    if guard.get("checked") and not guard.get("ok") and not args.skip_memory_guard:
        result["error"] = (
            f"free memory {guard['free_bytes'] / 1024**3:.1f} GiB is below the "
            f"{guard['needed_bytes'] / 1024**3:.1f} GiB this pack needs: another engine "
            "probably holds a model. Stop it first (this script never stops a process it "
            "did not start), or pass --skip-memory-guard."
        )
        return result
    if harness._port_answers(args.host, args.port):
        result["error"] = f"something already answers on {args.host}:{args.port}; choose another --port"
        return result

    proc = None
    try:
        started = time.monotonic()
        proc = harness._start_server(harness.serve_command(args), env, server_log)
        health = harness._wait_for_health(base_url, proc, args.boot_timeout)
        result["boot_s"] = round(time.monotonic() - started, 1)
        profile = health.get("profile") if isinstance(health.get("profile"), dict) else {}
        result["health"] = {
            "model": health.get("model"),
            "model_path": health.get("model_path"),
            "runtime_mode": health.get("runtime_mode"),
            "profile": profile.get("name"),
            "sampler": profile.get("sampler"),
        }
        for name, spec in plan.items():
            print(f"[{arm}] {name} ...", flush=True)
            entry: dict[str, Any] = {"kind": spec["kind"]}
            result["requests"][name] = entry
            if "follows" in spec:
                first = result["requests"].get(spec["follows"]) or {}
                if not first.get("output_text"):
                    entry["error"] = f"{spec['follows']} produced no text to carry forward"
                    continue
                body = _second_turn({**first, "body": plan[spec["follows"]]["body"]}, spec, args)
                entry["carried_reply_sha256"] = hashlib.sha256(
                    str(first.get("output_text")).encode("utf-8")
                ).hexdigest()
            else:
                body = spec["body"]
            entry["messages"] = len(body["messages"])
            sent_at = time.time()
            try:
                response = harness._http_json(
                    "POST", base_url + "/v1/chat/completions", body, args.request_timeout
                )
            except urllib.error.HTTPError as error:
                entry["error"] = f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:2000]}"
                continue
            except Exception as error:  # noqa: BLE001 - reported, and the arm goes on
                entry["error"] = f"{type(error).__name__}: {error}"
                if proc.poll() is not None:
                    break
                continue
            content, reasoning, finish = harness._message_text(response)
            entry.update(
                response_id=response.get("id"),
                finish_reason=finish,
                usage=response.get("usage") or {},
                output_sha256=hashlib.sha256(
                    (reasoning + "\x00" + content).encode("utf-8")
                ).hexdigest(),
                output_text=content,
                reasoning_text=reasoning,
            )
            record = harness._read_record(request_log, response.get("id"), sent_at, args.record_timeout)
            if record is None:
                entry["record_error"] = "no line for this request in the request log"
            else:
                entry.update({key: record[key] for key in _RECORD_KEYS if key in record})
    except Exception as error:  # noqa: BLE001 - the report carries it
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        harness._stop_server(proc)
        if result.get("error") or any(e.get("error") for e in result["requests"].values()):
            result["server_log_tail"] = harness._tail(server_log)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="The module docstring explains the arms, the requests and the verdicts.",
    )
    parser.add_argument("--pack", required=True, help="the dense Qwen3.8 pack (a local directory)")
    parser.add_argument("--image", required=True, help="an image file (png, jpeg, webp)")
    parser.add_argument("--out", default=None, help="report path (default: outputs/dense-vision-compiled-parity/<UTC time>/report.json under the repository root)")
    parser.add_argument("--arms", default=DEFAULT_ARMS, help=f"comma separated, from {', '.join(ARMS)} (default: {DEFAULT_ARMS})")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="default", help="explicit per-request thinking control; default leaves the server setting unchanged")
    parser.add_argument("--question", default=harness.QUESTION)
    parser.add_argument("--follow-up", default=harness.FOLLOW_UP)
    parser.add_argument("--text-prompt", default=harness.TEXT_PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18474)
    parser.add_argument("--profile", default=None, help="serve profile (default: the product's own choice for the pack)")
    parser.add_argument("--serve-arg", action="append", default=[], help="extra argument for `mtplx serve`, repeatable (use --serve-arg=--flag)")
    parser.add_argument("--serve-command", default=None, help="replace `python -m mtplx.cli serve` (tests use a stand-in server)")
    parser.add_argument("--env", action="append", type=harness._env_pair, default=[], help="extra KEY=VALUE for every boot, repeatable")
    parser.add_argument("--boot-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--record-timeout", type=float, default=60.0)
    parser.add_argument("--skip-memory-guard", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and start nothing")
    args = parser.parse_args(argv)
    args.arm_list = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in args.arm_list if name not in ARMS]
    if unknown or not args.arm_list:
        parser.error(f"unknown arm(s) {unknown}; choose from {', '.join(ARMS)}")
    if "instrument" not in args.arm_list:
        parser.error("the instrument arm carries the verdict; it cannot be left out")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pack = Path(args.pack).expanduser()
    image = Path(args.image).expanduser()
    if not image.is_file():
        raise SystemExit(f"no image at {image}")
    image_bytes = image.read_bytes()
    mime = mimetypes.guess_type(image.name)[0] or "image/png"
    image_url = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
    plan_requests = request_plan(image_url, args)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = (
        Path(args.out).expanduser()
        if args.out
        else ROOT / "outputs" / "dense-vision-compiled-parity" / stamp / "report.json"
    ).resolve()
    run_dir = out.parent

    native_sampler = None
    try:
        native_sampler = json.loads((pack / "mtplx_runtime.json").read_text(encoding="utf-8")).get("sampler")
    except (OSError, ValueError):
        pass

    plan = {
        "serve_command": harness.serve_command(args),
        "arms": {
            arm: {"set": ARMS[arm]["set"], "unset": ARMS[arm]["unset"]} for arm in args.arm_list
        },
        "requests": {
            name: {
                "kind": spec["kind"],
                "messages": len(spec["body"]["messages"]) if "body" in spec else "the first turn, its reply, the follow-up",
                "max_tokens": int(args.max_tokens),
                "seed": int(args.seed),
                "sampler_fields_sent": [],
                "thinking": args.thinking,
            }
            for name, spec in plan_requests.items()
        },
        "report": str(out),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        "This loads the full pack once per arm and keeps the GPU busy for a few "
        "minutes each: set the fans first.",
        flush=True,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": stamp,
        "tree": {
            "root": str(ROOT),
            "commit": harness._git("rev-parse", "HEAD"),
            "branch": harness._git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(harness._git("status", "--porcelain")),
        },
        "pack": {
            "path": str(pack),
            "weights_bytes": harness._weights_bytes(pack),
            "native_sampler": native_sampler,
        },
        "image": {
            "path": str(image),
            "bytes": len(image_bytes),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        },
        "settings": {
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "thinking": args.thinking,
            "sampler": "native: no sampler field is sent, the server applies the pack's own",
            "question": args.question,
            "follow_up": args.follow_up,
            "text_prompt": args.text_prompt,
        },
        "plan": plan,
        "arms": {},
    }
    for arm in args.arm_list:
        print(f"[{arm}] starting the server ...", flush=True)
        report["arms"][arm] = run_arm(arm, args, plan_requests, run_dir)
        if report["arms"][arm].get("error"):
            print(f"[{arm}] {report['arms'][arm]['error']}", flush=True)
            break
        deadline = time.monotonic() + 60
        while harness._port_answers(args.host, args.port) and time.monotonic() < deadline:
            time.sleep(1.0)

    outcome = evaluate(
        report["arms"],
        native_sampler=native_sampler if isinstance(native_sampler, dict) else None,
    )
    arm_errors = {arm: data["error"] for arm, data in report["arms"].items() if data.get("error")}
    missing_arms = [arm for arm in args.arm_list if arm not in report["arms"]]
    if arm_errors or missing_arms:
        outcome["exit_code"] = 1
    report.update(
        verdict=outcome["verdict"],
        reasons=outcome["reasons"],
        checks=outcome["checks"],
        failed_checks=outcome["failed_checks"],
        instrument_summary=outcome["instrument"],
        arm_errors=arm_errors,
        arms_not_run=missing_arms,
        exit_code=outcome["exit_code"],
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nverdict: {report['verdict']}")
    for reason in report["reasons"]:
        print(f"  - {reason}")
    for item in report["checks"]:
        mark = "ok  " if item["ok"] is True else ("n/a " if item["ok"] is None else "FAIL")
        note = "" if item["required"] else " (reported only)"
        print(f"  [{mark}] {item['name']}{note}: {item['detail']}")
    for arm, error in arm_errors.items():
        print(f"  [FAIL] arm {arm}: {error}")
    print(f"report: {out}")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
