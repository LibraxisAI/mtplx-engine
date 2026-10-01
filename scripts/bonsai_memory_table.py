#!/usr/bin/env python3
"""Measure Bonsai memory, never speed. --out is a NEW report directory.

RAM classes use GiB (1024**3 bytes); 4K/8K/16K mean 4096/8192/16384 tokens.
Each class gets one process, one model load, and fresh caches for every case.
The planner receives RAM explicitly. Refused cases are still measured, with
no admission override in the product and no shrinking of the requested work.

The full pack (including the draft head and vision weights) stays resident.
Text prefill and greedy AR decode use the engine's turbo prefill/cache path;
MTP verification, image activations and session-bank occupancy are NOT tested.
MLX limits are guidelines, not hard ceilings or a physical small-Mac emulator.
Completion alone therefore never establishes a RAM recommendation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from mtplx.memory_plan import (
    GIB,
    dense_kv_bytes_per_token_from_config,
    detect_total_ram_bytes,
    plan_memory,
)

SCHEMA = "mtplx.bonsai-memory.v1"
DECODE_TOKENS = 1024
EVENT_PREFIX = "BONSAI_MEMORY "
PROMPT_TEXT = "Explain the following code carefully, including its inputs and results.\n"


def positive_int(value: str) -> int:
    raw = value.strip().lower()
    try:
        result = int(raw[:-1]) * 1024 if raw.endswith("k") else int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer or K tokens: {value}") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def pack_metadata(pack: Path) -> dict[str, Any]:
    """Read local metadata and file sizes only; dry-run never imports MLX."""
    config_bytes = (pack / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("model_type") != "prism_hadamard_qwen35":
        raise ValueError("--pack must be a prism_hadamard_qwen35 pack")
    for name in ("model.safetensors", "mtp.safetensors", "mtplx_runtime.json"):
        if not (pack / name).is_file():
            raise ValueError(f"pack is missing {name}")
    # Same accounting as engine_session.model_weights_bytes; no MLX import.
    weights = {str(p.relative_to(pack)): p.stat().st_size
               for p in sorted(pack.rglob("*.safetensors"))}
    kv = dense_kv_bytes_per_token_from_config(config)
    maximum = config.get("text_config", {}).get("max_position_embeddings")
    if not kv or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("pack must declare its attention geometry and maximum context")
    runtime = json.loads((pack / "mtplx_runtime.json").read_text())
    return {
        "path": str(pack),
        "model_type": config["model_type"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        # Identity only: an already stamped pack may carry speed evidence.
        # Never copy those figures into a memory-only report.
        "identity": {key: runtime[key] for key in (
            "pack_name", "hf_repo", "public_model_id", "model_family",
            "base_trunk", "min_engine_version", "precision_variant",
        ) if key in runtime},
        "weight_files_bytes": weights,
        "weights_bytes": sum(weights.values()),
        "weights_gib": sum(weights.values()) / GIB,
        "kv_bytes_per_token": kv,
        "model_max_context": maximum,
    }


def make_report(metadata: dict[str, Any], classes: list[int], contexts: list[int]) -> dict[str, Any]:
    rows = []
    for ram, context, quant, decode in itertools.product(classes, contexts, ("off", "q8"), (0, DECODE_TOKENS)):
        plan = plan_memory(
            total_ram_bytes=ram * GIB,
            model_weights_bytes=metadata["weights_bytes"],
            kv_bytes_per_token=metadata["kv_bytes_per_token"],
            kv_quantization=quant,
            model_max_context=metadata["model_max_context"],
            # This table is the measurement the tight-machine rule requires,
            # so its verdicts show what the rule admits once it is stamped.
            tight_machine_measured=True,
        )
        rows.append({
            "ram_gib": ram, "prompt_tokens": context, "kv_quantization": quant,
            "decode_tokens": decode, "total_tokens": context + decode,
            "planner_verdict": "admit" if plan.model_fits else "refuse",
            "planner": plan.to_dict(),
            "request_fits_planner": plan.model_fits and context + decode <= plan.context_window_fit,
            "engine_budget_bytes": plan.usable_bytes, "engine_budget_gib": plan.usable_bytes / GIB,
            "wired_limit_bytes": wired_limit_bytes(ram * GIB, plan.usable_bytes),
            "status": "not_run", "completed": False, "allocation_failure": None,
            "peak_memory_bytes": None, "peak_memory_gib": None,
        })
    return {
        "schema": SCHEMA,
        "pack": metadata,
        "method": {
            "ram_unit": "GiB", "token_k": 1024, "decode_mode": "greedy_ar",
            "mtp_head_resident": True, "vision_weights_resident": True,
            "mtp_verification_measured": False, "image_activations_measured": False,
            "session_bank_measured": False, "profile": "turbo",
            "prefill_chunk_tokens": 2048, "prompt_text": PROMPT_TEXT,
            "inherited_mtplx_overrides": "ignored; clean product profile per class",
            "eos_policy": "ignored to measure the complete requested decode",
            "peak_includes_load": True,
            "limitation": "Metal limits are guidelines. Check peak <= budget; physical small-Mac validation is still required.",
        },
        "class_runs": [], "rows": rows,
    }


def wired_limit_bytes(ram_bytes: int, engine_budget: int) -> int:
    # mtplx serve: server/openai.py::_apply_metal_memory_caps, no Bonsai
    # resident-floor override. Allocation budget comes from the real planner.
    return min(engine_budget, max(4 * GIB, int(ram_bytes * 0.60)), 160 * GIB)


def set_limits(mx: Any, ram_bytes: int, budget: int) -> dict[str, int]:
    wired = wired_limit_bytes(ram_bytes, budget)
    mx.set_memory_limit(budget)
    mx.set_wired_limit(wired)
    return {"memory_limit_bytes": budget, "wired_limit_bytes": wired}


def exact_prompt(tokenizer: Any, count: int) -> list[int]:
    seed = list(tokenizer.encode(PROMPT_TEXT, add_special_tokens=False))
    if not seed:
        raise ValueError("tokenizer returned no prompt tokens")
    return (seed * ((count + len(seed) - 1) // len(seed)))[:count]


def error_details(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    allocation = isinstance(exc, MemoryError) or any(
        phrase in message.lower() for phrase in (
            "out of memory", "failed to allocate", "allocation failed", "insufficient memory",
            "memory limit exceeded", "resource exhausted", "unable to allocate",
            "attempting to allocate", "bad_alloc", "failed to create buffer",
        )
    )
    return {"error_type": type(exc).__name__, "error": message, "allocation_failure": allocation}


def cache_quantization(cache: list[Any], requested: str) -> dict[str, Any]:
    attention = [entry for entry in cache if callable(getattr(entry, "is_trimmable", None))
                 and entry.is_trimmable()]
    modes = [entry.kv_quant_config.normalized_mode if getattr(entry, "kv_quant", False) else "off"
             for entry in attention]
    if not modes or any(mode != requested for mode in modes):
        raise RuntimeError(f"requested KV {requested}, observed full-attention cache modes {modes}")
    return {"observed_kv_quantization": requested, "full_attention_cache_count": len(modes)}


def summarize_case(row: dict[str, Any], measured: dict[str, Any], load_peak: int) -> dict[str, Any]:
    result = {**row, **measured}
    peak = max(load_peak, measured["request_peak_memory_bytes"])
    result.update(
        load_peak_memory_bytes=load_peak, load_peak_memory_gib=load_peak / GIB,
        peak_memory_bytes=peak, peak_memory_gib=peak / GIB,
        within_engine_budget=peak <= row["engine_budget_bytes"],
    )
    return result


class MetalRunner:
    """Only this adapter needs MLX. It deliberately has no clock or sampler stats."""

    def __init__(self) -> None:
        import mlx.core as mx

        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable")
        self.mx = mx

    def load(self, pack: Path, row: dict[str, Any]) -> dict[str, Any]:
        from mtplx.profiles import get_profile

        # Child processes start without inherited MTPLX knobs. Use the actual
        # product profile, then explicitly state the one measurement override.
        os.environ.update(get_profile("turbo").env_dict())
        os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "2048"
        os.environ["MTPLX_PRISM_AUX_DTYPE"] = "float16"
        limits = set_limits(self.mx, row["ram_gib"] * GIB, row["engine_budget_bytes"])
        self.mx.set_default_device(self.mx.gpu)
        self.mx.reset_peak_memory()
        from mtplx import runtime
        from mtplx.vision import load_vision_tower

        self.rt = runtime.load(pack, mtp=True)
        self.tower = load_vision_tower(pack)
        self.mx.eval(self.rt.model.parameters(), self.tower.parameters())
        self.mx.synchronize()
        self.load_peak = int(self.mx.get_peak_memory())
        return {**limits, "load_peak_memory_bytes": self.load_peak,
                "load_peak_memory_gib": self.load_peak / GIB,
                "loaded_active_memory_bytes": int(self.mx.get_active_memory()),
                "mlx_version": version("mlx"),
                "metal_device": self.mx.metal.device_info(),
                "runtime_env": {k: v for k, v in os.environ.items() if k.startswith("MTPLX_")}}

    def run_case(self, row: dict[str, Any]) -> dict[str, Any]:
        self.progress = {"phase": "setup", "decoded_tokens": 0, "prefilled_tokens": 0}
        from mtplx.generation import _eval_cache_roots, _prefill
        from mtplx.attention_context import attention_phase

        mx = self.mx
        quant = row["kv_quantization"]
        os.environ["MTPLX_PAGED_KV_QUANT"] = quant
        os.environ["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] = quant
        os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] = str(row["prompt_tokens"])
        os.environ["MTPLX_DYNAMIC_PAGED_KV_TOKENS"] = str(row["total_tokens"])
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        result = {"completed": False, "status": "failed", "decoded_tokens": 0,
                  "prefilled_tokens": 0, "allocation_failure": False}
        self.progress = result
        result["phase"] = "prefill"
        # The helper's timing return is discarded; it is never recorded.
        cache, logits, _hidden, _timing = _prefill(
            self.rt, exact_prompt(self.rt.tokenizer, row["prompt_tokens"]), return_hidden=False,
        )
        result["prefilled_tokens"] = row["prompt_tokens"]
        result.update(cache_quantization(cache, quant))
        result["prefill_peak_memory_bytes"] = int(mx.get_peak_memory())
        result["phase"] = "decode" if row["decode_tokens"] else "prefill"
        for _ in range(row["decode_tokens"]):
            token = mx.argmax(logits, axis=-1).reshape(1, 1)
            with attention_phase("decode"):
                logits = self.rt.forward_ar(token, cache=cache, logits_keep=1)
            mx.eval(logits)
            _eval_cache_roots(cache)
            result["decoded_tokens"] += 1
        mx.synchronize()
        result.update(completed=True, status="completed", phase="complete",
                      request_peak_memory_bytes=int(mx.get_peak_memory()),
                      active_memory_bytes=int(mx.get_active_memory()),
                      allocator_cache_bytes=int(mx.get_cache_memory()))
        return result


def measure_class(pack: Path, rows: list[dict[str, Any]], emit: Any, runner_factory=MetalRunner) -> None:
    runner = None
    try:
        runner = runner_factory()
        loaded = runner.load(pack, rows[0])
        emit({"kind": "load", **loaded})
    except Exception as exc:
        failure = {"kind": "load", "status": "failed", **error_details(exc)}
        if runner is not None:
            failure["load_peak_memory_bytes"] = int(runner.mx.get_peak_memory())
            failure["load_peak_memory_gib"] = failure["load_peak_memory_bytes"] / GIB
        emit(failure)
        for row in rows:
            failed_row = {**row, **{k: v for k, v in failure.items() if k != "kind"}, "status": "load_failed"}
            if "load_peak_memory_bytes" in failure:
                peak = failure["load_peak_memory_bytes"]
                failed_row.update(peak_memory_bytes=peak, peak_memory_gib=peak / GIB,
                                  within_engine_budget=peak <= row["engine_budget_bytes"])
            emit({"kind": "row", "row": failed_row})
        return
    for index, row in enumerate(rows):
        try:
            result = runner.run_case(row)
        except Exception as exc:
            result = {**getattr(runner, "progress", {}),
                      "status": "failed", "completed": False, **error_details(exc),
                      "request_peak_memory_bytes": int(runner.mx.get_peak_memory())}
            emit({"kind": "row", "row": summarize_case(row, result, loaded["load_peak_memory_bytes"])})
            # After failed graph evaluation, cached shared rotations may retain
            # the failed graph. Do not label subsequent contaminated runs valid
            # or silently reload (the contract is one load per class).
            for remaining in rows[index + 1:]:
                emit({"kind": "row", "row": {**remaining, "status": "not_run_after_failure",
                      "error": "class process stopped after a failed case"}})
            return
        emit({"kind": "row", "row": summarize_case(row, result, loaded["load_peak_memory_bytes"])})


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "| RAM (GiB) | Engine budget (GiB) | Planner | Planner context | Prompt | Decode | KV | Completed | Peak (bytes) | Peak (GiB) | Within budget | Allocation failure | Status |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | :--- | :--- |",
    ]
    for row in report["rows"]:
        peak = row.get("peak_memory_bytes")
        lines.append("| " + " | ".join(str(v) for v in (
            row["ram_gib"], f'{row["engine_budget_gib"]:.2f}', row["planner_verdict"],
            row["planner"]["context_window_resolved"], row["prompt_tokens"], row["decode_tokens"],
            row["kv_quantization"], row["completed"], peak if peak is not None else "pending",
            f'{peak / GIB:.4f}' if peak is not None else "pending",
            row.get("within_engine_budget", "pending"),
            row.get("allocation_failure") if row.get("allocation_failure") is not None else "unknown",
            row["status"],
        )) + " |")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out: Path) -> None:
    (out / "memory.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "memory.md").write_text(
        "# Bonsai memory measurements\n\n" + report["method"]["limitation"] + "\n\n"
        "Peaks include model loading, MTP head and vision weights. Text AR only; "
        "MTP verification, image activations and populated session banks remain unmeasured. "
        "A refused planner's 4096-token field is a fallback, not an admitted window.\n\n"
        + markdown_table(report)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/bonsai-memory"))
    parser.add_argument("--classes", nargs="+", type=positive_int, default=[16, 18, 24])
    parser.add_argument("--contexts", nargs="+", type=positive_int, default=[4096, 8192, 16384])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-class", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        pack = args.pack.expanduser().resolve(strict=True)
        classes, contexts = list(dict.fromkeys(args.classes)), list(dict.fromkeys(args.contexts))
        report = make_report(pack_metadata(pack), classes, contexts)
        if args.dry_run:
            print(markdown_table(report))
            print(json.dumps({"pack": report["pack"], "method": report["method"],
                              "planner_verdicts": [r for r in report["rows"] if r["prompt_tokens"] == contexts[0] and r["decode_tokens"] == 0]}, indent=2))
            return 0
        if args.worker_class is not None:
            rows = [r for r in report["rows"] if r["ram_gib"] == args.worker_class]
            measure_class(pack, rows, lambda event: print(EVENT_PREFIX + json.dumps(event), flush=True))
            return 0
        out = args.out.expanduser().resolve()
        if out == pack or pack in out.parents or out in pack.parents:
            raise ValueError("--out must be separate from the source pack")
        out.mkdir(parents=True, exist_ok=False)
        report["host_ram_bytes"] = detect_total_ram_bytes()
        write_report(report, out)
        # Keep product traces/configuration out of the user's live state.
        env = {k: v for k, v in os.environ.items() if not k.startswith("MTPLX_")}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        for ram in classes:
            class_run: dict[str, Any] = {"ram_gib": ram}
            report["class_runs"].append(class_run)
            indices = [i for i, r in enumerate(report["rows"]) if r["ram_gib"] == ram]
            command = [sys.executable, str(Path(__file__).resolve()), "--pack", str(pack),
                       "--classes", str(ram), "--contexts", *map(str, contexts), "--worker-class", str(ram)]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as process:
                for line in process.stdout:
                    if line.startswith(EVENT_PREFIX):
                        event = json.loads(line[len(EVENT_PREFIX):])
                        if event["kind"] == "row":
                            report["rows"][indices.pop(0)] = event["row"]
                        else:
                            class_run.update(event)
                        write_report(report, out)
                    else:
                        # Loader diagnostics are visible, but no unstructured
                        # output (which might contain timing) goes in the data.
                        print(line, end="", file=sys.stderr)
                class_run["exit_code"] = process.wait()
            for index in indices:
                report["rows"][index].update(status="worker_failed", error=f"worker exited {process.returncode}; no measurement returned")
            write_report(report, out)
        print(f"Wrote {out / 'memory.json'} and {out / 'memory.md'}")
        return 0 if all(r["completed"] for r in report["rows"]) else 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
