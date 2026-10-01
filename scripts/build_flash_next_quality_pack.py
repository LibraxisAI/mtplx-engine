"""One-command BF16 -> Quality release candidate. --dry-run has no side effects."""
from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from mtplx.commands.forge_qwen4_exp import (
    OFFICIAL_SOURCE_REPO, QUALITY_RECIPE, QUALITY_REPO, QUALITY_SERVED_ID,
)
from mtplx.commands.forge_qwen4_audit import checksum

# Shape-derived estimates from R1-quality-pack.astra.md, Q4 table alternative.
SOURCE_BYTES = 360_000_000_000
PACK_BYTES = 170_000_000_000
HEADROOM_BYTES = 40_000_000_000


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def disk_plan(source: str, output: Path, download_dir: Path) -> dict:
    local = Path(source).expanduser().is_dir()
    # Downloads use HF --local-dir, not a second full Hub cache copy.
    download = 0 if local else SOURCE_BYTES
    devices = {}
    for target, needed, label in ((output.parent, PACK_BYTES + HEADROOM_BYTES, "output + headroom"),
                                  (download_dir, download, "BF16 download")):
        ancestor = target.absolute()
        while not ancestor.exists():
            ancestor = ancestor.parent
        dev = str(ancestor.stat().st_dev)
        entry = devices.setdefault(dev, {"path": str(ancestor), "additional_bytes": 0,
                                         "free_bytes": shutil.disk_usage(ancestor).free, "uses": []})
        entry["additional_bytes"] += needed
        entry["uses"].append(label)
    return {"source_estimate_bytes": SOURCE_BYTES, "output_estimate_bytes": PACK_BYTES,
            "headroom_bytes": HEADROOM_BYTES, "download_estimate_bytes": download,
            "additional_bytes": download + PACK_BYTES + HEADROOM_BYTES, "filesystems": list(devices.values()),
            "note": "Conservative free-space check; resumed downloads still reserve a full source. APFS free space is never added twice."}


def run_step(run: Path, label: str, command: list[str], *, env=None) -> None:
    print(f"{label}: {' '.join(command)}", flush=True)
    with (run / f"{label}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}); read {run / (label + '.log')}")


def preflight(plan: dict) -> dict:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This build requires an Apple Silicon Mac running macOS")
    if int(platform.mac_ver()[0].split(".")[0]) < 14:
        raise RuntimeError("macOS 14 or later is required")
    ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    if ram < 256 * 2**30:
        raise RuntimeError("The complete build and full-load smoke require at least 256 GB unified memory")
    mlx = importlib.metadata.version("mlx")
    if mlx != "0.32.2":
        raise RuntimeError(f"MLX 0.32.2 is required; found {mlx}")
    import mlx.core as mx
    import psutil

    if not mx.metal.is_available():
        raise RuntimeError("Metal is unavailable; full-load verification cannot run")
    for fs in plan["filesystems"]:
        if fs["free_bytes"] < fs["additional_bytes"]:
            raise RuntimeError(f"Insufficient disk at {fs['path']}: need {fs['additional_bytes']/1e9:.1f} GB free, have {fs['free_bytes']/1e9:.1f} GB")
    return {"macos": platform.mac_ver()[0], "ram_bytes": ram, "available_ram_bytes": psutil.virtual_memory().available,
            "mlx": mlx, "disk": plan,
            "ram_estimates": {"conversion": "Whole-body Q8 peak unmeasured; per-tensor evaluation and 4 GiB output shards",
                              "resident_weights_gib": 158.264, "128k_sparse_gib": 166.22, "128k_dense_gib": 178.97}}


def http_json(base: str, endpoint: str, payload=None, timeout=30):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + endpoint, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def serve_smoke(pack: Path, run: Path, *, max_tokens: int, startup_timeout: int, request_timeout: int, env=None) -> dict:
    import psutil
    from PIL import Image
    from io import BytesIO

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "mtplx.cli", "serve", "--model", str(pack),
               "--model-id", QUALITY_SERVED_ID, "--host", "127.0.0.1", "--port", str(port),
               "--no-auth", "--context-window", "131072"]
    write_json(run / "serve-command.json", command)
    stop = threading.Event()
    receipt = {"passed": False, "peak_process_tree_rss_bytes": 0, "peak_listener_rss_bytes": 0,
               "peak_mlx_bytes_from_responses": None, "artifact_path": str(pack), "served_id": QUALITY_SERVED_ID,
               "listener_pid": None, "rss_sample_interval_seconds": 0.5,
               "max_tokens": max_tokens, "startup_timeout_seconds": startup_timeout,
               "request_timeout_seconds": request_timeout, "context_window": 131072,
               "note": "Short chat/tool/image smoke; does not certify 128K prompt quality or peak at 128K."}
    with (run / "serve.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        receipt["launcher_pid"] = proc.pid

        def monitor():
            while not stop.is_set():
                try:
                    processes = [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
                    rss = 0
                    for process in processes:
                        amount = process.memory_info().rss
                        rss += amount
                        if process.pid == receipt["listener_pid"]:
                            receipt["peak_listener_rss_bytes"] = max(receipt["peak_listener_rss_bytes"], amount)
                    receipt["peak_process_tree_rss_bytes"] = max(receipt["peak_process_tree_rss_bytes"], rss)
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as exc:
                    receipt["memory_error"] = str(exc)
                    return
                stop.wait(0.5)

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic() + startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("mtplx serve exited before readiness; see serve.log")
                try:
                    models = http_json(base, "/v1/models", timeout=5)
                    if QUALITY_SERVED_ID not in {m["id"] for m in models.get("data", [])}:
                        raise RuntimeError("The smoke server exposes the wrong served model id")
                    listeners = [p for p in [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
                                 if any(c.status == psutil.CONN_LISTEN and c.laddr.port == port for c in p.net_connections(kind="tcp"))]
                    if len(listeners) != 1:
                        raise RuntimeError("Could not identify the smoke server's actual listener PID")
                    receipt["listener_pid"] = listeners[0].pid
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("mtplx serve readiness timed out; see serve.log")
                    time.sleep(1)
            common = {"model": QUALITY_SERVED_ID, "max_tokens": max_tokens, "stream": False,
                      "temperature": 1.0, "top_p": 0.95, "top_k": 20}
            png = BytesIO()
            Image.new("RGB", (224, 224), (255, 0, 0)).save(png, format="PNG")
            requests = {
                "chat": {"messages": [{"role": "user", "content": "What is 2 + 2? Reply with the number only."}]},
                "tool": {"messages": [{"role": "user", "content": "Use report_number to report the number 7."}],
                         "tools": [{"type": "function", "function": {"name": "report_number", "description": "Report a number",
                             "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}}],
                         "tool_choice": {"type": "function", "function": {"name": "report_number"}}},
                "image": {"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What is the dominant color of this image? Reply with one color word."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()}}]}]},
            }
            for label, request in requests.items():
                write_json(run / f"smoke-{label}-request.json", {**common, **request})
                response = http_json(base, "/v1/chat/completions", {**common, **request}, timeout=request_timeout)
                write_json(run / f"smoke-{label}-response.json", response)
                peak = response.get("mtplx_stats", {}).get("peak_memory_bytes")
                if isinstance(peak, (float, int)):
                    receipt["peak_mlx_bytes_from_responses"] = max(receipt["peak_mlx_bytes_from_responses"] or 0, peak)
                choice = response["choices"][0]
                message = choice["message"]
                if choice.get("finish_reason") == "length":
                    raise RuntimeError(f"{label} smoke exhausted its explicit {max_tokens}-token budget")
                if label == "tool":
                    calls = message.get("tool_calls", [])
                    if len(calls) != 1 or calls[0]["function"]["name"] != "report_number" or json.loads(calls[0]["function"]["arguments"]) != {"number": 7}:
                        raise RuntimeError("Tool smoke did not produce the expected structured call")
                else:
                    content = (message.get("content") or "").strip().lower().rstrip(".! ")
                    if content != ("4" if label == "chat" else "red"):
                        raise RuntimeError(f"{label} smoke returned an incorrect answer; see its response JSON")
                write_json(run / f"smoke-{label}-snapshot.json", http_json(base, "/v1/mtplx/snapshot"))
            if receipt.get("memory_error") or not receipt["peak_listener_rss_bytes"]:
                raise RuntimeError("Smoke memory telemetry failed; see serve-smoke.json")
            receipt["passed"] = True
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            stop.set()
            watcher.join(timeout=2)
            write_json(run / "serve-smoke.json", receipt)
    return receipt


def generate_card(pack: Path) -> str:
    runtime = json.loads((pack / "mtplx_runtime.json").read_text())
    meta = runtime["quality_pack"]
    actual_bytes = sum(p.stat().st_size for p in pack.iterdir() if p.is_file())
    rows = [f"| {name.replace('_', ' ')} | {info['stored_precision']} | {info['bytes'] / 1e9:.6f} |"
            for name, info in sorted(meta["stored_precision"].items())]
    license = meta["license"]
    credits = meta["credits"]
    return f"""---
license: {license['id']}
license_name: {license['name']}
license_link: {license['file']}
library_name: mtplx
pipeline_tag: text-generation
base_model: {credits['base_model']}
base_model_relation: quantized
tags:
- mtplx
- mlx
- apple-silicon
- macos
- speculative-decoding
- multi-token-prediction
- qwen
- qwen3.8
- qwen3.8-flash-next
- flash-next
- moe
- mtp
- 8-bit
- vision
- mac-studio
---

# Qwen 3.8 Flash-Next Optimized Quality

**The 8-bit build of Qwen 3.8 Flash-Next, for Macs with 256 GB or 512 GB. Requires MTPLX {meta['min_engine_version']} or later.**

Qwen's 125B-A6B Flash-Next, the Qwen4-generation hybrid mixture of experts with
Qwen Sparse Attention and a 51B-parameter n-gram table, packed for
[MTPLX](https://mtplx.com) with its multi-token prediction head. The main model
and the draft head are 8-bit with group size 64, the structural weights stay in
BF16, and the n-gram table is 4-bit with group size 32. On a Mac with 128 GB or
more, [Optimized Speed](https://huggingface.co/{credits['carried_from']}) is the
recommended build.

## Memory

The model weights, the draft head and the vision tower need about 128.5 GiB,
and the 32 GB n-gram table streams from SSD.

- **128 GB**: Cannot load. The weights, the draft head and the vision tower alone need about 128.5 GiB.
- **256 GB and 512 GB**: the Macs this pack is for, with about 59.5 GiB left for context and the session cache. The MTPLX app and CLI list it second there, after Optimized Speed.

## Speed

Speed on 256 GB and 512 GB Macs is not measured yet. The 8-bit weights move
twice the bytes per token of Optimized Speed, so expect slower decoding.

## What is in the pack

| Tensor class | Stored precision | Size (GB) |
|---|---|---:|
{chr(10).join(rows)}

The download is {actual_bytes / 1e9:.2f} GB. `size-checksums.json` lists the
size and SHA-256 of every other file.

## Use it

In the Mac app, pick Qwen 3.8 Flash-Next Optimized Quality. From the command line:

```bash
pip install mtplx
mtplx serve --model {meta['repo']} --model-id {meta['served_id']}
```

MTPLX samples at the official Qwen 3.8 settings (temperature 1.0, top-p 0.95,
top-k 20), and drafts are accepted with exact speculative sampling, so the
output follows the model's own distribution.

Built with the `{meta['recipe']['name']}` recipe from
[{credits['base_model']}](https://huggingface.co/{credits['base_model']}) at
revision `{meta['source']['revision']}`. Qwen Community License, preserved in
`{license['file']}`. The upstream model card is preserved as
`{credits['upstream_card']}`. License and credits are carried from
[{credits['carried_from']}](https://huggingface.co/{credits['carried_from']}).
Conversion and serving: {credits['engine']}.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help=f"BF16 directory or Hub repo id (official: {OFFICIAL_SOURCE_REPO})")
    parser.add_argument("output", type=Path, help="New final artifact directory")
    parser.add_argument("--download-dir", type=Path, help="Resumable BF16 source directory")
    parser.add_argument("--revision", default="main", help="Hub revision; resolved to an immutable commit before download")
    parser.add_argument("--upload", action="store_true", help=f"Use existing hf login to upload the finished pack to {QUALITY_REPO}")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and disk arithmetic without writing, networking or loading MLX")
    parser.add_argument("--smoke-max-tokens", type=int, default=2048)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=600)
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    download_dir = (args.download_dir or output.parent / "Qwen3.8-Flash-Next-BF16").expanduser().absolute()
    if min(args.smoke_max_tokens, args.startup_timeout, args.request_timeout) <= 0:
        parser.error("Smoke budgets and timeouts must be positive")
    plan = disk_plan(args.source, output, download_dir)
    print(json.dumps({"recipe": QUALITY_RECIPE, "source": args.source, "official_bf16_repo": OFFICIAL_SOURCE_REPO,
                      "output": str(output), "disk": plan, "ram_required_gib": 256,
                      "steps": ["preflight", "resume download if needed", "forge build + full-load verification",
                                "serve chat/tool/image smoke + peak RSS", "metadata-derived card", "size/checksum manifest"]
                               + ([f"hf upload {QUALITY_REPO}"] if args.upload else []),
                      "smoke_limits": {"max_tokens": args.smoke_max_tokens, "startup_seconds": args.startup_timeout,
                                       "request_seconds": args.request_timeout}}, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not Path(args.source).expanduser().is_dir() and not re.fullmatch(r"[\w.-]+/[\w.-]+", args.source):
        raise RuntimeError(f"Source directory does not exist and is not a Hub repo id: {args.source}")
    if output == download_dir or output in download_dir.parents:
        raise RuntimeError("The BF16 download directory must be outside the final output directory")
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}; choose a new path (nothing is overwritten)")
    run = output.parent / f"{output.name}.runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=False)
    try:
        write_json(run / "preflight.json", preflight(plan))
        source = Path(args.source).expanduser()
        if not source.is_dir():
            if not shutil.which("hf"):
                raise RuntimeError("The hf CLI must already be installed for downloading")
            from huggingface_hub import HfApi

            sha = HfApi().model_info(args.source, revision=args.revision).sha
            if not sha:
                raise RuntimeError("Hub did not return an immutable source revision")
            write_json(run / "source-resolution.json", {"repo": args.source, "revision": sha})
            run_step(run, "download", ["hf", "download", args.source, "--revision", sha, "--local-dir", str(download_dir)])
            source = download_dir
            write_json(source / ".mtplx-source.json", {"repo_id": args.source, "resolved_sha": sha})
        source = source.absolute()
        for required in ("LICENSE", "README.md"):
            if not (source / required).is_file():
                raise RuntimeError(f"BF16 source is missing {required}; preserve upstream license and credits before building")
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        # A clean named output is required; Forge may never silently select -1.
        command = [sys.executable, "-m", "mtplx.cli", "forge", "build", "--repo", str(source),
                   "--model-root", str(output.parent), "--branded-name", output.name,
                   "--out", str(run), "--run-id", "forge", "--recipe", QUALITY_RECIPE,
                   "--verification", "full-load", "--json"]
        run_step(run, "forge-build", command, env=env)
        runtime = json.loads((output / "mtplx_runtime.json").read_text())
        if runtime.get("verification", {}).get("status") != "verified":
            raise RuntimeError("Forge did not produce a full-load verified artifact")
        smoke = serve_smoke(output, run, max_tokens=args.smoke_max_tokens,
                            startup_timeout=args.startup_timeout, request_timeout=args.request_timeout, env=env)
        runtime["quality_pack"]["serve_smoke"] = smoke
        write_json(output / "mtplx_runtime.json", runtime)
        shutil.copy2(source / "LICENSE", output / "LICENSE")
        if (source / "NOTICE").is_file():
            shutil.copy2(source / "NOTICE", output / "NOTICE")
        upstream = output / runtime["quality_pack"]["credits"]["upstream_card"]
        if not upstream.exists():
            shutil.copy2(source / "README.md", upstream)
        if (output / "README.md").exists():
            (output / "README.md").rename(output / "README-before-quality-card.md")
        (output / "README.md").write_text(generate_card(output))
        (run / "card.log").write_text("Generated README.md from mtplx_runtime.json and actual file sizes.\n")
        manifest = {p.name: {"bytes": p.stat().st_size, "sha256": checksum(p)}
                    for p in sorted(output.iterdir()) if p.is_file()}
        write_json(output / "size-checksums.json", {"files": manifest, "total_bytes_excluding_manifest": sum(i["bytes"] for i in manifest.values())})
        write_json(run / "size-checksums.json", manifest)
        if args.upload:
            run_step(run, "login-check", ["hf", "auth", "whoami"])
            run_step(run, "upload", ["hf", "upload", QUALITY_REPO, str(output), "."])
        print(f"Built {output}\nLogs: {run}")
        return 0
    except Exception as exc:
        (run / "FAILED.txt").write_text(str(exc) + "\n")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Build stopped: {exc or 'interrupted'}", file=sys.stderr)
        raise SystemExit(1)
