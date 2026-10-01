#!/usr/bin/env python3
"""Real-pack proof that Flash-Next image requests verify exactly on the compiled route.

One command, a pack and an image:

    PYTHONPATH=$PWD python scripts/qwen4_vision_compiled_parity.py \\
        --pack ~/.mtplx/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \\
        --image photo.png

It starts the product server (``mtplx serve``) once per arm, sends the same
three-turn workflow to each boot, reads every request's record from the request log
and writes one JSON report. Exit code 0 means every check below held.

The arms (one boot each, one at a time):

  instrument  ``MTPLX_COMPILED_VERIFY=parity2``. The compiled verifier stays
              authoritative. Every verify round is also run on the eager
              verifier, from a stock copy of the same pre-round cache, inside
              the request's own position scope, and the logits, the hidden
              state, the captures and every cache leaf are compared bit for bit
              (``compiled_verify.fixed_m4_parity2`` in the request record:
              rounds, divergent rounds, largest absolute difference, largest
              KL, and the first divergent leaf if there is one).
  product     The shipping configuration with nothing overridden. Shows that an
              image request is admitted to the compiled route as shipped, and
              that the instrument did not change what the lane generates.
  eager       ``MTPLX_QWEN4_VISION_COMPILED_VERIFY=0``, the kill switch: the
              same image requests on the eager verifier. Same seed, so the text
              must be the product arm's text. This is the free-running check.
              It covers what a per-round instrument cannot see: repair
              forwards, the final commit, and the banked state a second turn
              restores. It is only a fair reference on a tree that holds the
              eager position scope fix (``_decode_trunk_scope`` in
              mtplx/generation.py); on an older tree the eager route ropes its
              repair, copy-round and final-commit rows without the image delta,
              so a difference there is reported and does not fail the run.

The requests (the follow-up carries that arm's actual first response):

  image_turn1   the image and a question;
  image_turn2   a second turn with that image in the history, which restores
                the image session from the bank before it decodes;
  text_control  a text-only request. It says what the compiled step and the
                eager forward agree to on this pack when no image is involved.

Sampling: no sampler field is sent, so the server applies the pack's native
settings (``mtplx_runtime.json``), with one fixed seed. Greedy is not used.
Both image turns must finish with ``stop`` and visible content. A truncated
reasoning-only answer cannot establish a real follow-up or cache restoration;
increase this diagnostic's ``--max-tokens`` when the report names that failure.

How to read the verdict:

  exact                     every compared round of every image request is bit
                            for bit the eager verifier, and every other check
                            held.
  exact_on_compared_rounds  as above, but some rounds had no reference: their
                            caller had no position scope open (before the eager
                            scope fix, a copy round routed through the bank by
                            MTPLX_CCOPY_BANK_ROUTE=1 is such a caller). They
                            are counted in ``reference_scope_missing_rounds``.
  diverges_like_text        image rounds differ, the text control differs too,
                            and by a similar size: the compiled step is not bit
                            exact on this pack with or without an image.
  image_route_diverges      image rounds differ and text rounds do not (or
                            differ by much less): look at ``first_divergence``.
  not_proven                the image route was not exercised; the reasons are
                            listed (refused shape, memory gate, no record).
  no_compiled_dispatch      an arm claiming compiled verification never ran it.
  no_compiled_comparisons   only eager dispatches had parity comparisons.
  generated_state_not_restored  the follow-up did not restore generated rows.

Safety: the script never stops a process it did not start, refuses to boot on a
port that answers, and refuses to load the pack when free memory is below three
quarters of the pack's weight files (another engine probably holds a model).
It loads the full pack and keeps the GPU busy for a few minutes per arm: set
the fans first. It takes no speed measurement and none of its numbers is one.

``--dry-run`` prints the plan (commands, environment changes, requests) and
starts nothing.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "qwen4_vision_compiled_parity/1"

QUESTION = "Describe this image in detail, then list three things a careful viewer would notice."
FOLLOW_UP = "Look again at the upper left quarter of the image and describe only that part."
TEXT_PROMPT = (
    "Explain how a suspension bridge carries its load, from the deck to the "
    "anchorages, then list three things an inspector would check first."
)

# What each arm changes in the server's environment. Everything else is the
# operator's shell, untouched: the product's own defaults are the subject.
ARMS: dict[str, dict[str, Any]] = {
    "instrument": {
        "set": {"MTPLX_COMPILED_VERIFY": "parity2"},
        "unset": ["MTPLX_QWEN4_VISION_COMPILED_VERIFY"],
    },
    "product": {
        "set": {},
        "unset": ["MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4_VISION_COMPILED_VERIFY"],
    },
    "eager": {
        "set": {"MTPLX_QWEN4_VISION_COMPILED_VERIFY": "0"},
        "unset": ["MTPLX_COMPILED_VERIFY"],
    },
}
DEFAULT_ARMS = "instrument,product,eager"

# A divergence counts as "like the text control" when its largest logit
# difference and largest KL are within this factor of the text control's. A
# reading aid for a lane that is not bit exact on text either, not a tolerance:
# the verdict is never "exact" unless every compared leaf was equal.
SAME_CLASS_FACTOR = 10.0

_PARITY_MAXIMA = (
    "logits_max_abs_diff",
    "logits_max_kl",
    "hidden_max_abs_diff",
    "state_max_abs_diff",
    "capture_max_abs_diff",
)
_RECORD_KEYS = (
    "request_id",
    "mode",
    "generation_mode",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "new_prefill_tokens",
    "finish_reason",
    "server_seed",
    "mtp_depth",
    "verify_calls",
    "accepted_drafts",
    "rejected_drafts",
    "drafted_tokens",
    "accepted_by_depth",
    "drafted_by_depth",
    "context_copy_rounds",
    "session_cache_hit",
    "session_restore_mode",
    "cache_miss_reason",
    "demotions",
    "fixed_m4_admission",
    "compiled_verify",
)
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


# ---------------------------------------------------------------------------
# The verdict: pure functions over request records, tested without a server.
# ---------------------------------------------------------------------------


def request_status(entry: dict[str, Any]) -> dict[str, Any]:
    """What one request's record says about the compiled verify lane."""

    admission = entry.get("fixed_m4_admission") or {}
    bank = entry.get("compiled_verify") or {}
    fixed = bank.get("fixed_m4") or {}
    parity = bank.get("fixed_m4_parity2")
    status: dict[str, Any] = {
        "engaged": bool(admission.get("engaged")),
        "admission_reason": admission.get("reason"),
        "positions": admission.get("positions"),
        "rope_delta": admission.get("rope_delta"),
        "images": admission.get("images"),
        "rope_delta_input": bool(fixed.get("rope_delta_input")),
        "bank_mode": bank.get("mode"),
        "compiled_calls": int(bank.get("compiled_calls") or 0),
        "fallback_calls": bank.get("fallback_calls"),
    }
    if entry.get("error"):
        status["state"] = "request_failed"
    elif not admission:
        # Not the Flash-Next fixed-M4 lane, a build without the request-record
        # fields, or the record never arrived.
        status["state"] = "no_admission_record"
    elif not admission.get("engaged"):
        status["state"] = "not_engaged"
    elif not isinstance(parity, dict):
        status["state"] = "no_instrument_record"
    else:
        rounds = int(parity.get("rounds") or 0)
        divergent = int(parity.get("divergent_rounds") or 0)
        status.update(
            rounds=rounds,
            compiled_rounds=int(parity.get("compiled_rounds") or 0),
            compiled_exact_rounds=int(parity.get("compiled_exact_rounds") or 0),
            eager_rounds=int(parity.get("eager_rounds") or 0),
            eager_exact_rounds=int(parity.get("eager_exact_rounds") or 0),
            divergent_rounds=divergent,
            reference_scope_missing_rounds=int(
                parity.get("reference_scope_missing_rounds") or 0
            ),
            rounds_by_width=parity.get("rounds_by_width") or {},
            first_divergence=parity.get("first_divergence"),
            **{name: float(parity.get(name) or 0.0) for name in _PARITY_MAXIMA},
        )
        if rounds == 0:
            status["state"] = "no_compared_rounds"
        elif status["compiled_calls"] <= 0:
            status["state"] = "no_compiled_dispatch"
        elif status["compiled_rounds"] <= 0:
            status["state"] = "no_compiled_comparisons"
        else:
            status["state"] = "divergent" if divergent else "exact"
    return status


def _worst(statuses: list[dict[str, Any]], name: str) -> float:
    return max((float(item.get(name) or 0.0) for item in statuses), default=0.0)


def _same_sampler(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        return all(
            float(left[name]) == float(right[name]) for name in ("temperature", "top_p", "top_k")
        )
    except (KeyError, TypeError, ValueError):
        return False


def generated_restore_status(requests: dict[str, Any]) -> dict[str, Any]:
    """A warm hit must reach beyond turn one's prompt into its generated rows."""

    first, follow = requests.get("image_turn1"), requests.get("image_turn2")
    if not first or not follow:
        return {"ok": False, "reason": "missing_image_turn"}
    prompt = int(first.get("prompt_tokens") or 0)
    generated = int(first.get("completion_tokens") or 0)
    cached = int(follow.get("cached_tokens") or 0)
    mode = follow.get("session_restore_mode")
    if first.get("error") or follow.get("error"):
        reason = "request_failed"
    elif prompt <= 0 or generated <= 0 or "cached_tokens" not in follow:
        reason = "missing_generated_restore_evidence"
    elif follow.get("session_cache_hit") is not True or mode in (None, "", "cold"):
        reason = "cold_prefill"
    elif cached <= prompt:
        reason = "prompt_only_restore"
    else:
        reason = "generated_tokens_restored"
    return {
        "ok": reason == "generated_tokens_restored", "reason": reason,
        "first_prompt_tokens": prompt, "first_completion_tokens": generated,
        "cached_tokens": cached, "session_restore_mode": mode,
        "restored_generated_tokens": max(0, min(generated, cached - prompt)),
    }


def evaluate(
    arms: dict[str, Any],
    *,
    eager_scope_fix_present: bool,
    native_sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The verdict, the checks behind it, and the exit code.

    ``arms`` maps an arm name to ``{"requests": {name: entry}}``; an entry
    holds ``kind`` (``image`` | ``text``), ``output_sha256`` and the record
    fields (``fixed_m4_admission``, ``compiled_verify``). ``native_sampler`` is
    the pack's own sampler block; each arm's ``health.sampler`` is what its
    server applies to a request that sends no sampler field.
    """

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool | None, detail: str, *, required: bool = True):
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})

    def requests_of(arm: str) -> dict[str, Any]:
        return (arms.get(arm) or {}).get("requests") or {}

    incomplete = [
        f"{arm}/{name}: finish={entry.get('finish_reason')!r}, "
        f"visible_content={bool(str(entry.get('output_text') or '').strip())}"
        for arm in arms
        for name, entry in requests_of(arm).items()
        if entry.get("kind") == "image" and (
            entry.get("finish_reason") != "stop"
            or not str(entry.get("output_text") or "").strip()
        )
    ]
    check(
        "image_turns_completed", not incomplete,
        "; ".join(incomplete) or "every image turn finished with stop and visible content",
    )

    applied = {
        arm: data["health"].get("sampler")
        for arm, data in arms.items()
        if isinstance(data.get("health"), dict)
    }
    if native_sampler and applied:
        wrong = [arm for arm, sampler in applied.items() if not _same_sampler(sampler, native_sampler)]
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
            reasons.append(
                f"{name}: kept off the compiled route ({status['admission_reason']})"
            )
        elif status["state"] == "no_compared_rounds":
            reasons.append(f"{name}: no verify round was compared")
        elif status["state"] in {"no_compiled_dispatch", "no_compiled_comparisons"}:
            reasons.append(f"{name}: {status['state']}")
        elif status["positions"] != "vision_delta" or not status["rope_delta_input"]:
            reasons.append(
                f"{name}: positions {status['positions']!r}, rope_delta_input "
                f"{status['rope_delta_input']}: the request did not take the image trace"
            )

    image_list = list(image.values())
    text_list = [s for s in text.values() if s["state"] in {"exact", "divergent"}]
    unreferenced = sum(int(s.get("reference_scope_missing_rounds") or 0) for s in image_list)
    if any(s["state"] == "no_compiled_dispatch" for s in image_list):
        verdict = "no_compiled_dispatch"
    elif any(s["state"] == "no_compiled_comparisons" for s in image_list):
        verdict = "no_compiled_comparisons"
    elif reasons:
        verdict = "not_proven"
    elif all(s["state"] == "exact" for s in image_list):
        verdict = "exact_on_compared_rounds" if unreferenced else "exact"
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
            f"{sum(int(s.get('rounds') or 0) for s in image_list)} image rounds compared, "
            f"{sum(int(s.get('compiled_rounds') or 0) for s in image_list)} compiled, "
            f"{sum(int(s.get('eager_rounds') or 0) for s in image_list)} eager, "
            f"{sum(int(s.get('divergent_rounds') or 0) for s in image_list)} divergent, "
            f"largest logit difference {_worst(image_list, 'logits_max_abs_diff'):.3g}, "
            f"largest KL {_worst(image_list, 'logits_max_kl'):.3g}"
        ),
    )
    check(
        "instrument_every_image_round_had_a_reference",
        unreferenced == 0,
        f"{unreferenced} image rounds were not compared because their caller had no "
        "position scope open"
        + ("" if eager_scope_fix_present else " (expected before the eager scope fix)"),
    )
    if text:
        check(
            "instrument_text_control_exact",
            all(s["state"] == "exact" for s in text.values()),
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

    if "product" in arms:
        product = {n: request_status(e) for n, e in requests_of("product").items()}
        product_kinds = {n: e.get("kind") for n, e in requests_of("product").items()}
        off_route = [
            f"{name}: {status['state']}"
            + (f" ({status['admission_reason']})" if status.get("admission_reason") else "")
            + f", positions {status['positions']!r}, rope_delta_input {status['rope_delta_input']}"
            for name, status in product.items()
            if product_kinds.get(name) == "image"
            and not (
                status["engaged"]
                and status["positions"] == "vision_delta"
                and status["rope_delta_input"]
                and status["bank_mode"] == "on"
                and status["compiled_calls"] > 0
            )
        ]
        has_image = any(kind == "image" for kind in product_kinds.values())
        check(
            "product_image_requests_take_the_compiled_route",
            has_image and not off_route,
            "; ".join(off_route)
            or (
                "every image request ran compiled verification with its rotary delta as a trace input"
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
            + (f"; different: {', '.join(different)}" if different else "")
            + (
                ""
                if eager_scope_fix_present
                else " (this tree has no eager position scope fix, so the eager "
                "route is not a fair reference and this check is reported only)"
            ),
            required=eager_scope_fix_present,
        )

    no_dispatch = [
        f"{arm}/{name}"
        for arm in ("instrument", "product") if arm in arms
        for name, entry in requests_of(arm).items()
        if int((entry.get("compiled_verify") or {}).get("compiled_calls") or 0) <= 0
    ]
    check(
        "compiled_arms_executed_compiled_verification", not no_dispatch,
        "no compiled dispatch: " + ", ".join(no_dispatch) if no_dispatch
        else "every request in the compiled arms executed compiled verification",
    )
    restores = {arm: generated_restore_status(requests_of(arm)) for arm in arms}
    missing_restore = [f"{arm}: {status['reason']}" for arm, status in restores.items()
                       if not status["ok"]]
    check(
        "follow_up_restored_generated_tokens", bool(restores) and not missing_restore,
        "; ".join(missing_restore) or "every arm restored past the first prompt into generated tokens",
    )
    if verdict in {"exact", "exact_on_compared_rounds"}:
        if no_dispatch:
            verdict = "no_compiled_dispatch"
        elif any(s["state"] == "no_compiled_comparisons" for s in text.values()):
            verdict = "no_compiled_comparisons"
        elif missing_restore:
            verdict = "generated_state_not_restored"
    failed = [c["name"] for c in checks if c["required"] and c["ok"] is not True]
    return {
        "verdict": verdict,
        "reasons": reasons,
        "checks": checks,
        "failed_checks": failed,
        "instrument": instrument,
        "generated_restore": restores,
        "exit_code": 0 if verdict == "exact" and not failed else 1,
    }


# ---------------------------------------------------------------------------
# The run: boot, ask, read the record, stop.
# ---------------------------------------------------------------------------


def _http_json(method: str, url: str, payload: dict[str, Any] | None, timeout_s: float):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _port_answers(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _weights_bytes(pack: Path) -> int | None:
    if not pack.is_dir():
        return None
    return sum(path.stat().st_size for path in pack.glob("*.safetensors"))


def memory_guard(pack: Path) -> dict[str, Any]:
    """Refuse to load a pack that cannot fit next to what is already resident."""

    weights = _weights_bytes(pack)
    if not weights:
        return {"checked": False, "why": "the pack is not a local directory of safetensors"}
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from mtplx.ple_row_gather import free_memory_bytes

    free, how = free_memory_bytes()
    need = int(weights * 0.75)
    return {
        "checked": True,
        "weights_bytes": weights,
        "free_bytes": free,
        "needed_bytes": need,
        "measured_by": how,
        "ok": free >= need,
    }


def _tail(path: Path, lines: int = 40) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _child_env(arm: str, request_log: Path, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    for name in ARMS[arm]["unset"]:
        env.pop(name, None)
    env.update(ARMS[arm]["set"])
    env.update(extra)
    env["PYTHONPATH"] = str(ROOT)
    env["MTPLX_REQUEST_LOG_JSONL"] = str(request_log)
    return env


def _reportable_env(env: dict[str, str]) -> dict[str, str]:
    """The MTPLX_* settings of a boot, without anything that looks like a secret."""

    return {
        name: value
        for name, value in sorted(env.items())
        if name.startswith("MTPLX_")
        and not any(marker in name.upper() for marker in _SECRET_MARKERS)
    }


def serve_command(args: argparse.Namespace) -> list[str]:
    command = shlex.split(args.serve_command) if args.serve_command else [
        sys.executable,
        "-m",
        "mtplx.cli",
        "serve",
    ]
    command += ["--model", str(args.pack), "--host", args.host, "--port", str(args.port), "--no-auth"]
    if args.profile:
        command += ["--profile", args.profile]
    return command + list(args.serve_arg or [])


def _start_server(command: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _stop_server(proc: subprocess.Popen | None) -> None:
    """Stop the server this script started (its own process group, nothing else)."""

    if proc is None or proc.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGTERM, 45), (signal.SIGKILL, 15)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the server exited during startup (rc {proc.returncode})")
        try:
            health = _http_json("GET", base_url + "/health", None, 10.0)
            if health:
                return health
        except Exception as error:  # noqa: BLE001 - any failure means "not up yet"
            last = f"{type(error).__name__}: {error}"
        time.sleep(2.0)
    raise TimeoutError(f"no answer from /health within {timeout_s:.0f} s ({last})")


def _read_record(request_log: Path, request_id: str | None, sent_at: float, timeout_s: float):
    """This request's line of the request log (the server appends it as it finishes).

    Matched by the response id. A line that does not carry the id is still
    this request's when it was written after the request was sent, because
    this script is the only client of the server it started; such a line is
    accepted a few seconds after it appears, in case the matching one follows.
    """

    deadline = time.monotonic() + timeout_s
    first_fresh_at: float | None = None
    while True:
        rows: list[dict[str, Any]] = []
        try:
            for line in request_log.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and not row.get("warmup"):
                    rows.append(row)
        except OSError:
            pass
        if request_id:
            for row in reversed(rows):
                if row.get("request_id") == request_id:
                    return row
        fresh = [row for row in rows if float(row.get("logged_at_s") or 0.0) >= sent_at]
        now = time.monotonic()
        if fresh and first_fresh_at is None:
            first_fresh_at = now
        settled = first_fresh_at is not None and now - first_fresh_at >= 3.0
        if fresh and (not request_id or settled or now >= deadline):
            return fresh[-1]
        if now >= deadline:
            return None
        time.sleep(0.5)


def _message_text(response: dict[str, Any]) -> tuple[str, str, str | None]:
    choice = ((response.get("choices") or [{}])[0]) or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return str(content or ""), str(reasoning or ""), choice.get("finish_reason")


def build_requests(image_url: str, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    image_part = {"type": "image_url", "image_url": {"url": image_url}}
    first_turn = [
        {"role": "user", "content": [image_part, {"type": "text", "text": args.question}]}
    ]
    # run_arm inserts the actual assistant message after turn one finishes.
    second_turn = first_turn + [{"role": "user", "content": args.follow_up}]

    def body(messages: list[dict[str, Any]]) -> dict[str, Any]:
        # No temperature, top_p or top_k: the server applies the pack's native
        # sampler. One seed for every request of every arm.
        return {
            "model": "default",
            "messages": messages,
            "max_tokens": int(args.max_tokens),
            "seed": int(args.seed),
            "stream": False,
        }

    return {
        "image_turn1": {"kind": "image", "body": body(first_turn)},
        "image_turn2": {"kind": "image", "after_response": "image_turn1",
                        "body": body(second_turn)},
        "text_control": {
            "kind": "text",
            "body": body([{"role": "user", "content": args.text_prompt}]),
        },
    }


def run_arm(
    arm: str, args: argparse.Namespace, requests: dict[str, dict[str, Any]], run_dir: Path
) -> dict[str, Any]:
    base_url = f"http://{args.host}:{args.port}"
    request_log = run_dir / f"{arm}.request-log.jsonl"
    server_log = run_dir / f"{arm}.server.log"
    env = _child_env(arm, request_log, dict(args.env or []))
    result: dict[str, Any] = {
        "env": _reportable_env(env),
        "env_unset": list(ARMS[arm]["unset"]),
        "server_log": str(server_log),
        "request_log": str(request_log),
        "requests": {},
    }
    guard = memory_guard(Path(args.pack).expanduser())
    result["memory_guard"] = guard
    if guard.get("checked") and not guard.get("ok") and not args.skip_memory_guard:
        result["error"] = (
            f"free memory {guard['free_bytes'] / 1024**3:.1f} GiB is below the "
            f"{guard['needed_bytes'] / 1024**3:.1f} GiB this pack needs: another engine "
            "probably holds a model. Stop it first (this script never stops a process it "
            "did not start), or pass --skip-memory-guard."
        )
        return result
    if _port_answers(args.host, args.port):
        result["error"] = f"something already answers on {args.host}:{args.port}; choose another --port"
        return result

    proc = None
    try:
        started = time.monotonic()
        proc = _start_server(serve_command(args), env, server_log)
        health = _wait_for_health(base_url, proc, args.boot_timeout)
        result["boot_s"] = round(time.monotonic() - started, 1)
        profile = health.get("profile") if isinstance(health.get("profile"), dict) else {}
        result["health"] = {
            "model": health.get("model"),
            "model_path": health.get("model_path"),
            "runtime_mode": health.get("runtime_mode"),
            "profile": profile.get("name"),
            "sampler": profile.get("sampler"),
        }
        for name, spec in requests.items():
            print(f"[{arm}] {name} ...", flush=True)
            entry: dict[str, Any] = {"kind": spec["kind"]}
            result["requests"][name] = entry
            body = spec["body"]
            if spec.get("after_response"):
                first = result["requests"].get(spec["after_response"]) or {}
                if first.get("error") or not first.get("response_message"):
                    entry["error"] = "the first response is unavailable for the follow-up"
                    continue
                if first.get("finish_reason") != "stop" or not first.get("output_text", "").strip():
                    entry["error"] = (
                        "the first image response did not complete with visible content; "
                        "increase --max-tokens before testing generated-state restoration"
                    )
                    continue
                body = dict(body, messages=[
                    *body["messages"][:-1], first["response_message"], body["messages"][-1],
                ])
            sent_at = time.time()
            try:
                response = _http_json(
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
            content, reasoning, finish = _message_text(response)
            entry.update(
                response_message=response["choices"][0]["message"],
                response_id=response.get("id"),
                finish_reason=finish,
                usage=response.get("usage") or {},
                output_sha256=hashlib.sha256(
                    (reasoning + "\x00" + content).encode("utf-8")
                ).hexdigest(),
                output_text=content,
                reasoning_text=reasoning,
            )
            record = _read_record(request_log, response.get("id"), sent_at, args.record_timeout)
            if record is None:
                entry["record_error"] = "no line for this request in the request log"
            else:
                entry.update({key: record[key] for key in _RECORD_KEYS if key in record})
    except Exception as error:  # noqa: BLE001 - the report carries it
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        _stop_server(proc)
        if result.get("error") or any(e.get("error") for e in result["requests"].values()):
            result["server_log_tail"] = _tail(server_log)
    return result


def _git(*argv: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *argv], cwd=str(ROOT), capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def eager_scope_fix_present() -> bool:
    try:
        source = (ROOT / "mtplx" / "generation.py").read_text(encoding="utf-8")
    except OSError:
        return False
    return "def _decode_trunk_scope(" in source


def _env_pair(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return name, value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="The module docstring explains the arms, the requests and the verdicts.",
    )
    parser.add_argument("--pack", required=True, help="the Flash-Next pack (a local directory)")
    parser.add_argument("--image", required=True, help="an image file (png, jpeg, webp)")
    parser.add_argument("--out", default=None, help="report path (default: outputs/qwen4-vision-compiled-parity/<UTC time>/report.json under the repository root)")
    parser.add_argument("--arms", default=DEFAULT_ARMS, help=f"comma separated, from {', '.join(ARMS)} (default: {DEFAULT_ARMS})")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--question-file", default=None, help="text appended to the question (a long file makes a long-context image request)")
    parser.add_argument("--follow-up", default=FOLLOW_UP)
    parser.add_argument("--text-prompt", default=TEXT_PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18473)
    parser.add_argument("--profile", default=None, help="serve profile (default: the product's own choice for the pack)")
    parser.add_argument("--serve-arg", action="append", default=[], help="extra argument for `mtplx serve`, repeatable (use --serve-arg=--flag)")
    parser.add_argument("--serve-command", default=None, help="replace `python -m mtplx.cli serve` (tests use a stand-in server)")
    parser.add_argument("--env", action="append", type=_env_pair, default=[], help="extra KEY=VALUE for every boot, repeatable")
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
    if args.question_file:
        extra = Path(args.question_file).expanduser().read_text(encoding="utf-8")
        args.question = f"{args.question}\n\n{extra}"
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
    requests = build_requests(image_url, args)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = (
        Path(args.out).expanduser()
        if args.out
        else ROOT / "outputs" / "qwen4-vision-compiled-parity" / stamp / "report.json"
    ).resolve()
    run_dir = out.parent

    native_sampler = None
    try:
        native_sampler = json.loads((pack / "mtplx_runtime.json").read_text(encoding="utf-8")).get("sampler")
    except (OSError, ValueError):
        pass

    plan = {
        "serve_command": serve_command(args),
        "arms": {
            arm: {"set": ARMS[arm]["set"], "unset": ARMS[arm]["unset"]} for arm in args.arm_list
        },
        "requests": {
            name: {
                "kind": spec["kind"],
                "messages": len(spec["body"]["messages"]) + bool(spec.get("after_response")),
                "after_response": spec.get("after_response"),
                "max_tokens": spec["body"]["max_tokens"],
                "seed": spec["body"]["seed"],
                "sampler_fields_sent": [],
            }
            for name, spec in requests.items()
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
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "eager_scope_fix_present": eager_scope_fix_present(),
        },
        "pack": {
            "path": str(pack),
            "weights_bytes": _weights_bytes(pack),
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
            "sampler": "native: no sampler field is sent, the server applies the pack's own",
            "question": args.question if not args.question_file else f"{QUESTION} (+ {args.question_file})",
            "follow_up": args.follow_up,
            "text_prompt": args.text_prompt,
        },
        "plan": plan,
        "arms": {},
    }
    for arm in args.arm_list:
        print(f"[{arm}] starting the server ...", flush=True)
        report["arms"][arm] = run_arm(arm, args, requests, run_dir)
        if report["arms"][arm].get("error"):
            print(f"[{arm}] {report['arms'][arm]['error']}", flush=True)
            break
        deadline = time.monotonic() + 60
        while _port_answers(args.host, args.port) and time.monotonic() < deadline:
            time.sleep(1.0)

    outcome = evaluate(
        report["arms"],
        eager_scope_fix_present=report["tree"]["eager_scope_fix_present"],
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
        generated_restore=outcome["generated_restore"],
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
