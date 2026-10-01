"""Real-shape random BF16 -> Q8/g64 component matmul parity, not model quality.

One projection is resident at a time. Routed projections use all 512 experts
with top-10 indices (including duplicates); both sides use the same routing.
The reference multiplies by dequantized weights. BF16-source error is reported
separately. No Speed-pack weights or fabricated BF16 reference are used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPERTS = {"gate": (512, 640, 2560), "up": (512, 640, 2560), "down": (512, 2560, 640)}
QSA = {"q": (12288, 2560), "k": (512, 2560), "v": (512, 2560),
       "o": (2560, 6144), "index_qk": (640, 2560)}
GDN = {"qkv": (10240, 2560), "z": (6144, 2560), "a": (48, 2560),
       "b": (48, 2560), "out": (2560, 6144)}
SHARED = {"router": (512, 2560), "shared_gate": (1, 2560),
          "shared_gate_proj": (640, 2560), "shared_up": (640, 2560), "shared_down": (2560, 640)}
COMPONENTS = {"expert_bank": EXPERTS, "qsa": QSA, "gdn": GDN,
              "mtp": {**{f"expert_{k}": v for k, v in EXPERTS.items()}, **QSA, **SHARED}}
WIDTHS = {"decode": 1, "verify": 4, "prefill": 2048}


def projection_checks(mx, shape, *, atol: float, rtol: float) -> list[dict]:
    weight = (mx.random.normal(shape) / math.sqrt(shape[-1])).astype(mx.bfloat16)
    mx.eval(weight)
    q, s, b = mx.quantize(weight, group_size=64, bits=8)
    mx.eval(q, s, b)
    dequant = mx.dequantize(q, s, b, group_size=64, bits=8)
    mx.eval(dequant)
    results = []
    for phase, width in WIDTHS.items():
        x = mx.random.normal((width, shape[-1])).astype(mx.bfloat16)
        if len(shape) == 3:
            x = x[:, None, :]
            lhs = mx.arange(width, dtype=mx.uint32)[:, None]
            rhs = (mx.arange(width * 10, dtype=mx.uint32).reshape(width, 10) * 13) % shape[0]
            # Repeated IDs are legal and exercise the gather contract.
            rhs = mx.where(mx.arange(10)[None, :] == 9, rhs[:, :1], rhs)
            got = mx.gather_qmm(x, q, s, b, lhs_indices=lhs, rhs_indices=rhs, transpose=True, bits=8, group_size=64)
            reference = mx.gather_mm(x.astype(mx.float32), dequant.astype(mx.float32).swapaxes(-1, -2), lhs_indices=lhs, rhs_indices=rhs)
            bf16 = mx.gather_mm(x.astype(mx.float32), weight.astype(mx.float32).swapaxes(-1, -2), lhs_indices=lhs, rhs_indices=rhs)
        else:
            got = mx.quantized_matmul(x, q, s, b, transpose=True, bits=8, group_size=64)
            reference = x.astype(mx.float32) @ dequant.astype(mx.float32).T
            bf16 = x.astype(mx.float32) @ weight.astype(mx.float32).T
        mx.eval(got, reference, bf16)
        error = got.astype(mx.float32) - reference
        denominator = mx.maximum(mx.sum(reference * reference), 1e-20)
        results.append({"phase": phase, "M": width, "weight_shape": list(shape),
                        "output_shape": list(got.shape), "max_abs_error": float(mx.max(mx.abs(error))),
                        "relative_l2": float(mx.sqrt(mx.sum(error * error) / denominator)),
                        "bf16_source_relative_l2": float(mx.sqrt(mx.sum((got.astype(mx.float32) - bf16) ** 2) / mx.maximum(mx.sum(bf16 * bf16), 1e-20))),
                        "passed": bool(mx.all(mx.abs(error) <= atol + rtol * mx.abs(reference)))})
        del got, reference, bf16, x, error
        mx.clear_cache()
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--components", nargs="+", choices=tuple(COMPONENTS), default=list(COMPONENTS))
    parser.add_argument("--seed", type=int, default=2114)
    parser.add_argument("--atol", type=float, default=0.05, help="Explicit BF16-output absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.02, help="Explicit BF16-output relative tolerance")
    parser.add_argument("--dry-run", action="store_true", help="Print geometry; no files or MLX work")
    args = parser.parse_args(argv)
    report = {"status": "planned", "reference": "random BF16 tensors at official shapes; dequantized Q8/g64 float32 matmul",
              "seed": args.seed, "device": args.device, "bits": 8, "group_size": 64,
              "widths": WIDTHS, "atol": args.atol, "rtol": args.rtol,
              "scope": "Projection/operator parity, not full attention/GDN recurrence, end-to-end MTP, or checkpoint quality",
              "structural_tensors": "FC, norms, hyper-connections, GDN conv/A_log/dt_bias remain BF16 in the pack; not quantized by this probe",
              "components": {k: COMPONENTS[k] for k in args.components}, "results": []}
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    try:
        import mlx.core as mx
        from importlib.metadata import version

        report["mlx_version"] = version("mlx")
        if args.device == "gpu" and not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable; no component measurements were made")
        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
        mx.random.seed(args.seed)
        mx.reset_peak_memory()
        for component in args.components:
            for name, shape in COMPONENTS[component].items():
                print(f"Checking {component}.{name} {shape}", flush=True)
                for result in projection_checks(mx, shape, atol=args.atol, rtol=args.rtol):
                    report["results"].append({"component": component, "projection": name, **result})
                mx.clear_cache()
        report["peak_mlx_bytes"] = mx.get_peak_memory()
        report["status"] = "passed" if all(row["passed"] for row in report["results"]) else "failed"
    except (ImportError, RuntimeError) as exc:
        report.update(status="unavailable" if not report["results"] else "failed", error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
