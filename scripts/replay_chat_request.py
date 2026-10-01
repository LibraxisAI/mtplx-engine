#!/usr/bin/env python3
"""Replay a captured local chat request without rerunning its agent tool loop.

This is a bounded decode diagnostic, not a task-quality benchmark. Tool calls
are recorded, never executed. Sampling and prompts remain those of the capture
unless explicitly overridden. Start the daemon with verified max fans first.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--messages", type=int, help="Replay this prefix of messages")
    parser.add_argument("--max-tokens", type=int, required=True,
                        help="Explicit output budget for this diagnostic only")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--client", default="opencode")
    parser.add_argument("--session-id", default="decode-replay")
    parser.add_argument("--thermalforge", default=shutil.which("thermalforge") or
                        str(pathlib.Path.home() / ".mtplx/bin/thermalforge"))
    parser.add_argument("--request-log", type=pathlib.Path)
    args = parser.parse_args()
    body = json.loads(args.request.read_text())
    if args.max_tokens < 1 or (args.messages is not None and
                             not 1 <= args.messages <= len(body["messages"])):
        parser.error("Token budget and message prefix must be positive and in range")
    if args.messages is not None:
        body["messages"] = body["messages"][:args.messages]
    body.update(max_tokens=args.max_tokens, stream=True,
                stream_options={"include_usage": True})
    body.pop("max_completion_tokens", None)
    if args.seed is not None:
        body["seed"] = args.seed
    base = f"http://127.0.0.1:{args.port}"

    def read(path):
        with urllib.request.urlopen(base + path, timeout=10) as response:
            return json.load(response)

    def command(*argv):
        return subprocess.check_output(argv, text=True, timeout=10)

    def thermal():
        return json.loads(command(args.thermalforge, "status"))

    health = read("/health")
    if not health["ok"] or read("/v1/mtplx/flight")["active"]:
        raise RuntimeError("Daemon is unavailable or another inference is active")
    fans = thermal().get("fans", [])
    if not fans or not all(f["mode"] == "manual" and f["max_rpm"] > 0 and
                           f["actual_rpm"] >= .95 * f["max_rpm"] for f in fans):
        raise RuntimeError("Actual max-fan ramp has not been verified")
    args.out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")

    save("request.json", body)
    save("health-before.json", health)
    rid = None
    request_tag = "replay-" + uuid.uuid4().hex
    headers = {"Content-Type": "application/json", "x-mtplx-client": args.client,
               "x-mtplx-session-id": args.session_id,
               "x-mtplx-request-id": request_tag}
    save("headers.json", headers)
    log = args.request_log or pathlib.Path.home() / f".mtplx/logs/request-log-{args.port}.jsonl"
    log_offset = log.stat().st_size if log.exists() else 0
    stop = threading.Event()
    errors = []

    def monitor():
        try:
            with (args.out / "monitor.jsonl").open("w") as output:
                while not stop.is_set():
                    row = {"ts": time.time(), "thermal": thermal(),
                           "vm_stat": command("vm_stat"),
                           "swap": command("sysctl", "vm.swapusage"),
                           "flight": read("/v1/mtplx/flight"), "health": read("/health")}
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    stop.wait(1)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            errors.append(repr(exc))

    worker = threading.Thread(target=monitor, daemon=True)
    macmon = None
    gpu_output = (args.out / "gpu.jsonl").open("w")
    if shutil.which("macmon"):
        macmon = subprocess.Popen(["macmon", "pipe", "--samples", "3600", "--interval", "1000"],
                                  stdout=gpu_output, stderr=subprocess.STDOUT)
    worker.start()
    try:
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=900) as response, \
                (args.out / "stream.jsonl").open("w") as output:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                rid = event.get("id", rid)
                output.write(json.dumps({"ts": time.time(), "event": event}) + "\n")
                if "error" in event:
                    raise RuntimeError(event["error"])
    finally:
        stop.set()
        worker.join(45)
        if macmon:
            macmon.terminate()
            macmon.wait(timeout=10)
        gpu_output.close()
        save("monitor-errors.json", errors)
    receipt = None
    for _ in range(30):
        if log.exists():
            with log.open() as source:
                source.seek(log_offset)
                for line in source:
                    if line.endswith("\n"):
                        row = json.loads(line)
                        if row.get("request_id") == rid:
                            receipt = row
        if receipt:
            break
        time.sleep(.2)
    if receipt is None:
        raise RuntimeError(f"Missing request receipt for {rid}; streaming evidence was saved")
    save("receipt.json", receipt)
    save("health-after.json", read("/health"))
    print(json.dumps({key: receipt.get(key) for key in
                      ("request_id", "prompt_tokens", "cached_tokens", "completion_tokens",
                       "decode_tok_s", "ttft_s", "verify_calls", "verify_time_s",
                       "draft_time_s", "active_memory_bytes", "peak_memory_bytes")}))
    if errors:
        raise RuntimeError(f"Incomplete monitoring: {errors}")


if __name__ == "__main__":
    main()
