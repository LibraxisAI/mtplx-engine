"""Bounded, CPU-only inspection of Flash-Next packs. Never constructs a model.

Headers establish every stored precision. Checksums cover every byte. Numerical
checks sample the first/middle/last row of the first/last tensor in each class,
plus both sides of every n-gram shard boundary. They do not certify generation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from mtplx.commands.forge_qwen4_exp import (
    NGRAM_FILE, MTP_FILE, MTP_NORM_SHIFT_SUFFIXES, QUALITY_RECIPE, Qwen4ForgeError,
    read_safetensors_header, recipe_params,
)

ITEM_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I64": 8, "I32": 4}


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_pack_checksums(pack: Path, expected: dict) -> None:
    """Promotion must verify the same bytes that received the streaming audit."""
    paths = {p.name: p for p in pack.glob("*.safetensors")}
    if not expected or paths.keys() != expected.keys():
        raise Qwen4ForgeError("Checksum manifest and pack shard inventory disagree")
    for name, path in paths.items():
        if path.stat().st_size != expected[name]["bytes"] or checksum(path) != expected[name]["sha256"]:
            raise Qwen4ForgeError(f"Checksum mismatch before full-load verification: {name}")


def inventory(root: Path) -> tuple[dict, dict]:
    """Validate byte extents, the complete index and uniqueness across shards."""
    tensors, files = {}, {}
    for path in sorted(root.glob("*.safetensors")):
        header, start = read_safetensors_header(path)
        end = 0
        entries = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                         key=lambda item: item[1]["data_offsets"][0])
        for key, info in entries:
            shape, dtype = info["shape"], info["dtype"]
            if dtype not in ITEM_BYTES or any(not isinstance(n, int) or n < 0 for n in shape):
                raise Qwen4ForgeError(f"Unsupported tensor header: {path.name}:{key}")
            lo, hi = info["data_offsets"]
            if lo != end or hi - lo != math.prod(shape) * ITEM_BYTES[dtype]:
                raise Qwen4ForgeError(f"Invalid tensor extent: {path.name}:{key}")
            if key in tensors:
                raise Qwen4ForgeError(f"Duplicate tensor: {key}")
            tensors[key] = {**info, "file": path.name, "start": start}
            end = hi
        if start + end != path.stat().st_size:
            raise Qwen4ForgeError(f"Truncated or trailing payload: {path.name}")
        files[path.name] = {"bytes": start + end, "tensors": len(entries),
                            "metadata": header.get("__metadata__", {})}
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    for key, filename in index.items():
        if key not in tensors or tensors[key]["file"] != filename:
            raise Qwen4ForgeError(f"Index does not resolve tensor: {key}")
    for key, info in tensors.items():
        if info["file"] not in (NGRAM_FILE, MTP_FILE) and index.get(key) != info["file"]:
            raise Qwen4ForgeError(f"Unindexed tensor: {key}")
    if not tensors:
        raise Qwen4ForgeError("No safetensors payload found")
    return tensors, files


def validate_bf16_source(source: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    if config.get("quantization") or config.get("quantization_config"):
        raise Qwen4ForgeError("Quality requires an original BF16 checkpoint, not FP8 or a quantized pack")
    tensors, _ = inventory(source)
    for key, info in tensors.items():
        if info["dtype"] not in ("BF16", "I64", "I32"):
            raise Qwen4ForgeError(f"Quality source must be BF16 (integer buffers preserved): {key} is {info['dtype']}")
    if not any(k.startswith("mtp.") for k in tensors):
        raise Qwen4ForgeError("Quality source has no MTP head")
    return tensors


def tensor_class(key: str) -> str:
    """Fail closed on new, unclassified tensors instead of guessing precision."""
    if key.startswith("vision_tower."):
        return "vision_tower"
    if key.startswith("ngram."):
        return "ngram_table"
    prefix = "mtp_" if key.startswith("mtp.") else ""
    if "norm" in key:
        return prefix + "norms"
    if "hyper_connection" in key:
        return prefix + "hyper_connections"
    if ".ple." in key:
        if key.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")):
            return "ple_integer_buffers"
        return "ple_projections_and_conv"
    if key.startswith(("mtp.fc_embedding.", "mtp.fc_hidden.")):
        return "mtp_fc"
    if ".switch_mlp." in key:
        return prefix + "routed_experts"
    if ".shared_expert." in key:
        return prefix + "shared_expert"
    if ".mlp.gate." in key or ".shared_expert_gate." in key:
        return prefix + "router"
    if ".indexer.index_qk_proj." in key:
        return prefix + "qsa_indexer"
    if ".self_attn." in key:
        return prefix + "attention"
    if ".linear_attn." in key:
        if ".conv1d." in key:
            return "gdn_conv"
        if key.endswith(".A_log"):
            return "gdn_A_log"
        if key.endswith(".dt_bias"):
            return "gdn_dt_bias"
        return "gdn_projections"
    if ".embed_tokens." in key:
        return "embeddings"
    if ".lm_head." in key:
        return "lm_head"
    raise Qwen4ForgeError(f"Unclassified Flash-Next tensor: {key}")


BF16_CLASSES = {
    "vision_tower", "norms", "mtp_norms", "hyper_connections", "mtp_hyper_connections",
    "ple_projections_and_conv", "mtp_fc", "gdn_conv", "gdn_A_log", "gdn_dt_bias",
}


def source_layout(tensors: dict, hidden: int) -> dict:
    """Map HF names/layouts to stored names using only header geometry."""
    refs = {}
    for raw, info in tensors.items():
        if ".ngram_embedding.shard_" in raw:
            continue
        key = raw
        if key.startswith("model.visual."):
            key = "vision_tower." + key[len("model.visual."):]
        elif key.startswith("model.language_model."):
            key = "language_model.model." + key[len("model.language_model."):]
        elif key == "lm_head.weight":
            key = "language_model.lm_head.weight"
        elif not key.startswith(("mtp.", "language_model.", "vision_tower.")):
            key = "language_model." + key
        shape = list(info["shape"])
        if key.endswith(".mlp.experts.gate_up_proj"):
            prefix = key.split(".mlp.experts.")[0]
            bmm = shape[1] == hidden
            logical = [shape[0], shape[2] // 2, shape[1]] if bmm else [shape[0], shape[1] // 2, shape[2]]
            for half, proj in enumerate(("gate", "up")):
                refs[f"{prefix}.mlp.switch_mlp.{proj}_proj.weight"] = {
                    "raw": raw, "shape": logical, "op": "packed", "half": half, "bmm": bmm}
        elif key.endswith(".mlp.experts.down_proj"):
            bmm = shape[2] == hidden
            refs[key.replace("experts.down_proj", "switch_mlp.down_proj.weight")] = {
                "raw": raw, "shape": [shape[0], shape[2], shape[1]] if bmm else shape,
                "op": "transpose" if bmm else "identity"}
        elif ".mlp.experts." in key:
            prefix, rest = key.split(".mlp.experts.")
            expert, proj = rest.split(".", 1)
            dest = f"{prefix}.mlp.switch_mlp.{proj}"
            ref = refs.setdefault(dest, {"parts": {}, "shape": [0, *shape], "op": "stack"})
            ref["parts"][int(expert)] = raw
            ref["shape"][0] += 1
        else:
            op = "identity"
            if key.endswith("ple.conv1d.weight") or (key.endswith("linear_attn.conv1d.weight") and shape[-1] != 1):
                key = key.replace("ple.conv1d.weight", "ple.conv_weight")
                shape = [shape[0], shape[2], shape[1]]
                op = "conv"
            refs[key] = {"raw": raw, "shape": shape, "op": op}
    return refs


def _mmap(root: Path, info: dict):
    import numpy as np

    dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "U32": "<u4", "I64": "<i8", "I32": "<i4"}[info["dtype"]]
    return np.memmap(root / info["file"], mode="r", dtype=dtype,
                     offset=info["start"] + info["data_offsets"][0], shape=tuple(info["shape"]))


def _float(values, dtype: str):
    import numpy as np

    if dtype == "BF16":
        return (np.asarray(values, dtype=np.uint32) << 16).view(np.float32)
    return np.asarray(values)


def _source_row(source: Path, tensors: dict, ref: dict, index: tuple):
    raw = ref.get("raw")
    if ref["op"] == "stack":
        raw = ref["parts"][index[0]]
        index = index[1:]
    info = tensors[raw]
    value = _mmap(source, info)
    if ref["op"] == "packed":
        if ref["bmm"]:
            value = value.swapaxes(1, 2)
        width = ref["shape"][1]
        value = value[:, ref["half"] * width:(ref["half"] + 1) * width, :]
    elif ref["op"] == "transpose":
        value = value.swapaxes(1, 2)
    elif ref["op"] == "conv":
        value = value.swapaxes(1, 2)
    return _float(value[index], info["dtype"])


def _rows(shape: list[int]) -> list[tuple]:
    import numpy as np

    count = math.prod(shape[:-1])
    return [tuple(int(x) for x in np.unravel_index(i, tuple(shape[:-1])))
            for i in sorted({0, count // 2, count - 1})]


def _check_rows(pack: Path, tensors: dict, key: str, ref_rows: list, *, bits: int = 0, group: int = 0) -> dict:
    import numpy as np

    error = 0.0
    for index, reference in ref_rows:
        reference = np.asarray(reference)
        weight = _mmap(pack, tensors[key])[index]
        if bits:
            base = key.removesuffix(".weight")
            scale = _float(_mmap(pack, tensors[base + ".scales"])[index], "BF16")
            bias = _float(_mmap(pack, tensors[base + ".biases"])[index], "BF16")
            codes = ((weight[..., None] >> np.arange(0, 32, bits, dtype=np.uint32)) & ((1 << bits) - 1)).reshape(-1)
            got = codes.astype(np.float32) * np.repeat(scale, group) + np.repeat(bias, group)
            groups = reference.reshape(-1, group)
            span = np.ptp(groups, axis=-1)
            magnitude = np.max(np.abs(groups), axis=-1)
            # One quantization step plus BF16 scale/bias rounding. This is
            # a conversion error bound, not a model-quality acceptance test.
            tolerance = np.repeat(span / ((1 << bits) - 1) + (2 * magnitude + span) / 128, group) + 1e-7
        else:
            got = _float(weight, tensors[key]["dtype"])
            tolerance = 0
        delta = np.abs(got.astype(np.float64) - reference.astype(np.float64))
        if not np.all(np.isfinite(delta)) or not np.all(delta <= tolerance):
            raise Qwen4ForgeError(f"Source parity failed: {key} row {index}; max error {float(np.max(delta))}")
        error = max(error, float(np.max(delta)))
    return {"tensor": key, "rows": [list(i) for i, _ in ref_rows], "max_abs_error": error}


def audit_pack(source: Path, pack: Path, recipe: dict, *, expected_checksums: dict | None = None) -> dict:
    """Audit actual bytes and deterministic row samples with bounded host memory."""
    import numpy as np

    params = recipe_params(recipe)
    source_tensors = validate_bf16_source(source)
    tensors, files = inventory(pack)
    cfg = json.loads((source / "config.json").read_text())
    hidden = int(cfg.get("text_config", cfg)["hidden_size"])
    refs = source_layout(source_tensors, hidden)
    table = sorted((k for k in source_tensors if ".ngram_embedding.shard_" in k),
                   key=lambda k: int(k.rsplit("shard_", 1)[1].split(".")[0]))
    if not table:
        raise Qwen4ForgeError("Source has no n-gram table")
    refs["ngram.weight"] = {"shape": [sum(source_tensors[k]["shape"][0] for k in table), source_tensors[table[0]]["shape"][1]]}
    table_meta = files.get(NGRAM_FILE, {}).get("metadata", {})
    if (int(table_meta.get("ngram_bits", 4)), int(table_meta.get("ngram_group_size", 32))) != (4, 32):
        raise Qwen4ForgeError("N-gram header must declare the fixed-M4 4-bit/g32 layout")
    primary = {k for k in tensors if not k.endswith((".scales", ".biases"))}
    if primary != set(refs):
        raise Qwen4ForgeError(f"Tensor accounting failed: missing={sorted(set(refs) - primary)[:8]}, extra={sorted(primary - set(refs))[:8]}")
    classes, rules = {}, {}
    for key in sorted(primary):
        info, shape = tensors[key], refs[key]["shape"]
        cls = tensor_class(key)
        base = key.removesuffix(".weight")
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if info["dtype"] == "U32":
            scales, biases = tensors.get(base + ".scales"), tensors.get(base + ".biases")
            if not scales or not biases or scales["dtype"] != "BF16" or biases["dtype"] != "BF16":
                raise Qwen4ForgeError(f"Missing BF16 affine parameters: {key}")
            bits, rem = divmod(info["shape"][-1] * 32, shape[-1])
            group, group_rem = divmod(shape[-1], scales["shape"][-1])
            if rem or group_rem or info["shape"][:-1] != shape[:-1] or scales["shape"] != [*shape[:-1], shape[-1] // group] or scales["shape"] != biases["shape"]:
                raise Qwen4ForgeError(f"Invalid affine geometry: {key}")
            precision = f"Q{bits}/g{group} affine; BF16 scales and biases"
            rules[key] = (bits, group)
            size += sum(t["data_offsets"][1] - t["data_offsets"][0] for t in (scales, biases))
        else:
            if info["shape"] != shape:
                raise Qwen4ForgeError(f"Source/output shape mismatch: {key}")
            precision = info["dtype"]
            rules[key] = (0, 0)
        if recipe.get("name") == QUALITY_RECIPE:
            expected = "BF16" if cls in BF16_CLASSES else "I64" if cls == "ple_integer_buffers" else (
                "Q4/g32 affine; BF16 scales and biases" if cls == "ngram_table" else "Q8/g64 affine; BF16 scales and biases")
            if precision != expected:
                raise Qwen4ForgeError(f"Quality precision mismatch: {key}: stored {precision}, required {expected}")
        entry = classes.setdefault(cls, {"stored_precision": precision, "tensors": [], "bytes": 0})
        if entry["stored_precision"] != precision:
            raise Qwen4ForgeError(f"Mixed precision inside tensor class {cls}")
        entry["tensors"].append(key)
        entry["bytes"] += size
    # No orphan scales/biases can escape the primary-tensor accounting.
    for key in tensors:
        if key.endswith((".scales", ".biases")) and key.rsplit(".", 1)[0] + ".weight" not in primary:
            raise Qwen4ForgeError(f"Orphan affine tensor: {key}")
    samples = {}
    # Match the source's zero-centred norm convention without loading the model.
    raw_hf = any(k.startswith("model.language_model.") or
                 (k.endswith("conv1d.weight") and v["shape"][-1] != 1)
                 for k, v in source_tensors.items())
    for cls, entry in classes.items():
        samples[cls] = []
        if cls == "ngram_table":
            offset, ref_rows = 0, []
            for raw in table:
                info = source_tensors[raw]
                for row in sorted({0, info["shape"][0] - 1}):
                    ref_rows.append(((offset + row,), _float(_mmap(source, info)[row], "BF16")))
                offset += info["shape"][0]
            samples[cls].append(_check_rows(pack, tensors, "ngram.weight", ref_rows, bits=params["ngram_bits"], group=params["ngram_group"]))
            continue
        for key in sorted({entry["tensors"][0], entry["tensors"][-1]}):
            ref_rows = []
            for index in _rows(refs[key]["shape"]):
                value = _source_row(source, source_tensors, refs[key], index)
                if (raw_hf or key.startswith("mtp.")) and len(refs[key]["shape"]) == 1 and key.endswith(MTP_NORM_SHIFT_SUFFIXES):
                    shifted = value.astype(np.float32) + 1
                    # Round-to-nearest-even BF16, as the converter does.
                    words = shifted.view(np.uint32)
                    value = ((words + 0x7fff + ((words >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
                ref_rows.append((index, value))
            bits, group = rules[key]
            samples[cls].append(_check_rows(pack, tensors, key, ref_rows, bits=bits, group=group))
    for name, info in files.items():
        info["sha256"] = checksum(pack / name)
        if expected_checksums is not None and expected_checksums.get(name) != info["sha256"]:
            raise Qwen4ForgeError(f"Checksum mismatch: {name}")
    if expected_checksums is not None and set(expected_checksums) != set(files):
        raise Qwen4ForgeError("Checksum manifest and pack shard inventory disagree")
    return {"passed": True, "method": "headers-sha256-source-samples", "files": files,
            "tensor_count": len(tensors), "tensor_payload_bytes": sum(v["bytes"] for v in classes.values()),
            "stored_precision": classes, "samples": samples,
            "sample_policy": "first/last tensor per class; first/middle/last row; every n-gram shard edge",
            "quantized_error_bound": "one quantization step + (2*group_absmax + group_range)/128 for BF16 rounding"}


def source_identity(source: Path) -> dict[str, Any]:
    """Use a real Hub revision when available, else a labelled content revision."""
    from mtplx.hf_loader import read_source_marker

    marker = read_source_marker(source) or {}
    files = {p.name: checksum(p) for p in sorted(source.glob("*.safetensors"))}
    for name in ("config.json", "model.safetensors.index.json"):
        files[name] = checksum(source / name)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    revision = marker.get("resolved_sha") or (source.name if source.parent.name == "snapshots" else None)
    return {"repo": marker.get("repo_id"), "revision": revision or f"sha256:{digest}",
            "revision_kind": "hub-commit" if revision else "local-content-sha256",
            "content_sha256": digest, "files": files}


def quality_metadata(source: dict, audit: dict) -> dict:
    from mtplx.commands.forge_qwen4_exp import (
        QUALITY_NAME, QUALITY_REPO, QUALITY_SERVED_ID, QUALITY_ENGINE_FLOOR, named_recipe,
    )

    return {
        "name": QUALITY_NAME, "repo": QUALITY_REPO, "served_id": QUALITY_SERVED_ID,
        "min_engine_version": QUALITY_ENGINE_FLOOR, "recipe": named_recipe(QUALITY_RECIPE),
        "source": source, "stored_precision": audit["stored_precision"],
        "tensor_payload_bytes": audit["tensor_payload_bytes"], "files": audit["files"],
        "ram_guidance": {
            "256 GB": "Fits at 128K with the n-gram table streamed from SSD (planning estimate; measure the final pack).",
            "192 GB": "Fits with the n-gram table streamed from SSD. Validate the context workload on this machine.",
            "128 GB": "Cannot load: body, MTP and vision alone are approximately 128.46 GiB.",
        },
        "performance": "Q8 weights transfer more bytes per parameter than the Speed pack, so bandwidth-limited decode is expected to be slower. Speed is unmeasured until the first run; no speed claim.",
        "license": {"id": "other", "name": "qwen-community-1.0", "file": "LICENSE"},
        "credits": {"base_model": "Qwen/Qwen3.8-Flash-Next", "engine": "MTPLX",
                    "carried_from": "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
                    "upstream_card": "README-upstream-qwen.md"},
    }
