# Copyright © 2026 MTPLX.
#
# Loader for Prism ML "prism_hadamard_qwen35" packs (Ternary Bonsai 2 27B).
#
# The pack is Qwen3.8-27B with the architecture unchanged. Its language model
# stores 402 matrices (every attention, GatedDeltaNet and MLP projection, the
# embedding and the LM head) as ternary weights in MLX's affine 2-bit
# container (group 128, scale s, bias -s, fp16), in a ROTATED basis: each row
# of the original matrix was transformed by a blockwise normalized
# Walsh-Hadamard matrix with a fixed sign vector before the ternary
# assignment. The runtime therefore has to apply the matching transform to the
# activation that enters each packed projection, and the inverse transform to
# each embedding row after lookup. A loader that skips either step returns
# wrong output, not an error, so this module refuses every pack whose rotation
# metadata, packed tensors or vision tower it cannot account for.
#
# The loading contract (the module list in config.json, the transform
# T(x) = H(s * x) with its inverse s * H(y), the inverse transform on embedding
# rows, the float32 transform precision, the affine 2-bit group-128 container)
# is adapted from the Apache-2.0 reference runtime that Prism ML ships inside
# the pack (runtime/runtime.py, runtime/artifact.py, runtime/vision_artifact.py,
# Copyright 2026 Prism ML, Inc.). See NOTICE.
#
# What MTPLX adds on top of the reference contract:
#   * the packed weights are never expanded: every projection is one
#     mx.quantized_matmul on the uint32 words;
#   * sibling projections that read the same activation (q/k/v, gate/up,
#     in_proj_qkv/in_proj_z) share one transform instead of repeating it;
#   * activations run in float16, the dtype the pack declares for every module
#     and the dtype of its scales, on every chip (no bfloat16 anywhere, which
#     is also what M1 and M2 need);
#   * the vision tower is mandatory: a pack of this type without its vision
#     tensors, vision_config or preprocessor file is refused;
#   * the residual stream stays in the standard basis (the rotation lives
#     inside each projection), so a Qwen3.8-27B MTP head attaches unchanged.

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5 as _qwen3_5
from mlx_lm.models.base import BaseModelArgs

from ..kernels.hadamard_rotate import rotate as fused_hadamard_rotate
from ..kernels.ternary_qmv import (
    plan as ternary_plan,
    run as ternary_run,
    ternary_layout,
    worthwhile as ternary_worthwhile,
)

MODEL_TYPE = "prism_hadamard_qwen35"
SUPPORTED_SCHEMA_VERSIONS = (2,)
SUPPORTED_BLOCKS = (512, 1024, 2048, 4096)
SUPPORTED_TENSOR_NAMESPACE = "mlx-vlm-qwen3_5"
SUPPORTED_BASE_MODEL_TYPE = "qwen3_5"
SUPPORTED_GDN_LAYOUT = "grouped"
SUPPORTED_TRANSFORM = "normalized-sylvester-walsh-hadamard"
SUPPORTED_TRANSFORM_AXIS = "input-last-dimension"
PACKED_BITS = 2
PACKED_GROUP_SIZE = 128
PACKED_MODE = "affine"
LANGUAGE_PREFIX = "language_model."
VISION_PREFIXES = ("vision_tower.", "model.visual.")
PREPROCESSOR_FILE = "preprocessor_config.json"

# Auxiliary float tensors (norm weights, GatedDeltaNet conv/gate parameters)
# ship as float32. The default casts them to the activation dtype the pack
# declares (float16), so the residual stream, the KV cache and every
# quantized matmul run in float16. "float32" keeps them as stored, which
# promotes the residual stream to float32 at the first RMSNorm exactly like
# Prism's bundled Python runtime does; it exists for parity measurements only.
AUX_DTYPE_ENV = "MTPLX_PRISM_AUX_DTYPE"
# "0" makes every packed projection run its own transform (the reference
# graph). The shared transform is the same operation on the same input, so the
# two settings are bit-identical; the switch exists to prove that on a GPU.
SHARE_ROTATION_ENV = "MTPLX_PRISM_SHARE_ROTATION"

_ACTIVATION_DTYPES = {"float16": mx.float16}
_AUX_DTYPES = {"float16": mx.float16, "float32": mx.float32}

# Projections that read the same activation inside one parent block.
_SIBLING_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
)


class PrismHadamardContractError(ValueError):
    """The pack does not satisfy the prism_hadamard_qwen35 loading contract."""


def _refuse(message: str) -> "PrismHadamardContractError":
    return PrismHadamardContractError(f"prism_hadamard_qwen35: {message}")


def hadamard_rotate(
    x: mx.array, signs: mx.array, block: int, *, inverse: bool = False
) -> mx.array:
    """Blockwise normalized Walsh-Hadamard transform with fixed signs.

    Forward T(x) = H(s * x); inverse T^-1(y) = s * H(y). H is symmetric and
    orthonormal, so T^-1(T(x)) = x and <T(a), T(b)> = <a, b>. The transform
    runs in float32 and returns the input dtype: the unscaled butterfly sums
    1,024 values, which overflows float16 long before the input does.
    """

    width = int(x.shape[-1])
    if width % block:
        raise _refuse(
            f"Hadamard block {block} does not divide activation width {width}"
        )
    fused = fused_hadamard_rotate(x, signs, block, inverse=inverse)
    if fused is not None:
        # Same bits as the chain below, in one dispatch.
        return fused
    return mlx_chain_rotate(x, signs, block, inverse=inverse)


def mlx_chain_rotate(
    x: mx.array, signs: mx.array, block: int, *, inverse: bool = False
) -> mx.array:
    """The stock four-op rotation: float32 cast, sign multiply, mx.hadamard_transform, cast.

    The path every rotation takes when the fused kernel declines, and the
    reference the load-time self-check holds that kernel to, bit for bit.
    """

    width = int(x.shape[-1])
    dtype = x.dtype
    y = x.astype(mx.float32)
    if not inverse:
        y = y * signs
    y = mx.hadamard_transform(
        mx.unflatten(y, -1, (width // block, block)),
        scale=1.0 / math.sqrt(block),
    )
    y = mx.flatten(y, -2, -1)
    if inverse:
        y = y * signs
    return y.astype(dtype)


class _SharedRotation:
    """One transform per distinct input for sibling projections.

    A hit requires the very same array object. The entry holds a strong
    reference to that input, so its identity cannot be recycled while the
    entry lives, and the entry is dropped as soon as every sibling has read
    it. A miss only costs one more transform, so any interleaving stays
    correct. The entry is one tuple so a reader never sees half an update.
    """

    __slots__ = ("fanout", "_entry")

    def __init__(self, fanout: int) -> None:
        self.fanout = int(fanout)
        self._entry: tuple[Any, Any, int] | None = None

    def __call__(self, x: mx.array, signs: mx.array, block: int) -> mx.array:
        entry = self._entry
        if entry is not None and entry[0] is x:
            left = entry[2] - 1
            self._entry = (entry[0], entry[1], left) if left > 0 else None
            return entry[1]
        rotated = hadamard_rotate(x, signs, block)
        self._entry = (x, rotated, self.fanout - 1) if self.fanout > 1 else None
        return rotated


def _check_packed_dims(width: int, block: int, group_size: int, bits: int) -> None:
    if width <= 0 or width % group_size or (width * bits) % 32:
        raise _refuse(
            f"packed width {width} is not a multiple of group size {group_size}"
        )
    if block and (block not in SUPPORTED_BLOCKS or width % block):
        raise _refuse(f"unsupported Hadamard block {block} for width {width}")


class HadamardQuantizedLinear(nn.Module):
    """y = quantized_matmul(T(x), W') on packed 2-bit weights, never expanded.

    W' holds the rows of the original matrix after T, so W' T(x) = W x.
    Deliberately not an nn.QuantizedLinear subclass: every MTPLX fast path
    that reads .weight/.scales/.biases directly gates on that class (and on
    4-bit or 8-bit), so it declines here and the call below, the only place
    that knows about the rotation, stays the single way into the weights.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        block: int,
        group_size: int = PACKED_GROUP_SIZE,
        bits: int = PACKED_BITS,
    ) -> None:
        super().__init__()
        _check_packed_dims(input_dims, block, group_size, bits)
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)
        self.block = int(block)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = PACKED_MODE
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size), dtype=mx.float16)
        self.biases = mx.zeros((output_dims, input_dims // group_size), dtype=mx.float16)
        if self.block:
            self.signs = mx.ones((input_dims,), dtype=mx.float32)
        self._rotation: _SharedRotation | None = None
        # Set by Model.post_weight_load once the loaded biases are proven to
        # be exactly -scales (Prism's ternary layout), with the kernel's
        # launch plan for this matrix.
        self._ternary = False
        self._ternary_plan = None
        self.freeze()

    def rotate(self, x: mx.array) -> mx.array:
        if not self.block:
            return x
        rotation = self._rotation
        if rotation is not None:
            return rotation(x, self["signs"], self.block)
        return hadamard_rotate(x, self["signs"], self.block)

    def __call__(self, x: mx.array) -> mx.array:
        rotated = self.rotate(x)
        planned = self._ternary_plan
        if planned is not None:
            # Verify and decode rows (M <= 4) on the ternary kernel; it
            # declines every other shape and the stock matmul below runs.
            out = ternary_run(planned, rotated, self["weight"], self["scales"])
            if out is not None:
                return out
        return mx.quantized_matmul(
            rotated,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"block={self.block}, group_size={self.group_size}, bits={self.bits}"
        )


class HadamardQuantizedEmbedding(nn.Module):
    """Row lookup on packed 2-bit weights, then the inverse transform.

    The stored rows are T(e), so the lookup returns T^-1 of the dequantized
    row: a standard-basis embedding. Anything spliced into the embedding
    stream afterwards (vision tower rows) is already in the standard basis
    and must not be transformed.
    """

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        *,
        block: int,
        group_size: int = PACKED_GROUP_SIZE,
        bits: int = PACKED_BITS,
    ) -> None:
        super().__init__()
        _check_packed_dims(dims, block, group_size, bits)
        self.num_embeddings = int(num_embeddings)
        self.dims = int(dims)
        self.block = int(block)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.mode = PACKED_MODE
        self.weight = mx.zeros((num_embeddings, dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((num_embeddings, dims // group_size), dtype=mx.float16)
        self.biases = mx.zeros((num_embeddings, dims // group_size), dtype=mx.float16)
        if self.block:
            self.signs = mx.ones((dims,), dtype=mx.float32)
        self.freeze()

    def __call__(self, ids: mx.array) -> mx.array:
        rows = mx.dequantize(
            self["weight"][ids],
            scales=self["scales"][ids],
            biases=self["biases"][ids],
            group_size=self.group_size,
            bits=self.bits,
        )
        if not self.block:
            return rows
        return hadamard_rotate(rows, self["signs"], self.block, inverse=True)

    def as_linear(self, x: mx.array) -> mx.array:
        if self.block:
            x = hadamard_rotate(x, self["signs"], self.block)
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def _extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.dims}, block={self.block}, "
            f"group_size={self.group_size}, bits={self.bits}"
        )


@dataclass(frozen=True)
class PackedModuleRecord:
    path: str
    block: int
    embedding: bool
    dtype: str


def _parse_module_records(raw: Any) -> tuple[PackedModuleRecord, ...]:
    if not isinstance(raw, list) or not raw:
        raise _refuse(
            "config.json carries no 'modules' list. The list names every "
            "rotated projection; without it the pack cannot be run correctly."
        )
    records: list[PackedModuleRecord] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise _refuse(f"malformed module record: {entry!r}")
        path = entry["path"]
        if path in seen:
            raise _refuse(f"duplicate packed module {path!r}")
        seen.add(path)
        if "block" not in entry or "embedding" not in entry:
            raise _refuse(f"module record {path!r} misses 'block' or 'embedding'")
        try:
            block = int(entry["block"])
        except (TypeError, ValueError):
            raise _refuse(f"module record {path!r} has a non-integer block") from None
        if block and block not in SUPPORTED_BLOCKS:
            raise _refuse(f"module record {path!r} has unsupported block {block}")
        dtype = str(entry.get("dtype") or "")
        if dtype not in _ACTIVATION_DTYPES:
            raise _refuse(
                f"module record {path!r} declares activation dtype {dtype!r}; "
                "only float16 packs are supported"
            )
        records.append(
            PackedModuleRecord(
                path=path,
                block=block,
                embedding=bool(entry["embedding"]),
                dtype=dtype,
            )
        )
    return tuple(records)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    modules: list = field(default_factory=list)
    quantization: dict = field(default_factory=dict)
    schema_version: int = 0
    base_model_type: str = ""
    tensor_namespace: str = ""
    gdn_activation_layout: str = ""
    hadamard_config: str = ""
    components: dict = field(default_factory=dict)
    vision_config: Optional[dict] = None


def _validate_args(args: ModelArgs) -> tuple[PackedModuleRecord, ...]:
    if args.model_type != MODEL_TYPE:
        raise _refuse(f"unexpected model_type {args.model_type!r}")
    if args.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise _refuse(
            f"unsupported schema_version {args.schema_version!r} "
            f"(supported: {list(SUPPORTED_SCHEMA_VERSIONS)})"
        )
    if args.base_model_type != SUPPORTED_BASE_MODEL_TYPE:
        raise _refuse(f"unsupported base_model_type {args.base_model_type!r}")
    if args.tensor_namespace != SUPPORTED_TENSOR_NAMESPACE:
        raise _refuse(f"unsupported tensor_namespace {args.tensor_namespace!r}")
    if args.gdn_activation_layout != SUPPORTED_GDN_LAYOUT:
        raise _refuse(
            f"unsupported gdn_activation_layout {args.gdn_activation_layout!r}: "
            "the GatedDeltaNet value heads must already be grouped"
        )
    quantization = args.quantization if isinstance(args.quantization, dict) else {}
    declared = (
        quantization.get("bits"),
        quantization.get("group_size"),
        str(quantization.get("mode") or PACKED_MODE),
    )
    if declared != (PACKED_BITS, PACKED_GROUP_SIZE, PACKED_MODE):
        raise _refuse(
            f"unsupported quantization {quantization!r}; the contract is affine "
            f"{PACKED_BITS}-bit with group size {PACKED_GROUP_SIZE}"
        )
    if not isinstance(args.text_config, dict) or not args.text_config:
        raise _refuse("config.json carries no text_config")
    components = args.components if isinstance(args.components, dict) else {}
    if not components.get("text"):
        raise _refuse("components.text is not true")
    # Bonsai is a vision-language model. A pack of this type that does not
    # declare and carry its vision tower has had a capability removed.
    if not components.get("vision") or not isinstance(args.vision_config, dict):
        raise _refuse(
            "the pack does not declare its vision tower (components.vision and "
            "vision_config are required). MTPLX never loads this model type "
            "as text only."
        )
    return _parse_module_records(args.modules)


def _resolve_parent(root: nn.Module, path: str) -> tuple[Any, str]:
    parts = path.split(".")
    parent: Any = root
    try:
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        getattr(parent, parts[-1])
    except (AttributeError, IndexError, KeyError, TypeError):
        raise _refuse(f"packed module {path!r} does not exist in the model") from None
    return parent, parts[-1]


def _aux_dtype() -> mx.Dtype:
    name = (os.environ.get(AUX_DTYPE_ENV) or "float16").strip().lower()
    if name not in _AUX_DTYPES:
        raise _refuse(f"{AUX_DTYPE_ENV} must be float16 or float32, not {name!r}")
    return _AUX_DTYPES[name]


def _share_rotation_enabled() -> bool:
    return (os.environ.get(SHARE_ROTATION_ENV, "1").strip().lower()) not in {
        "0",
        "false",
        "no",
        "off",
    }


class Model(_qwen3_5.Model):
    """Qwen3.8-27B text model with Prism's rotated ternary projections."""

    def __init__(self, args: ModelArgs) -> None:
        records = _validate_args(args)
        super().__init__(args)
        # Plain attributes, kept out of the nn.Module parameter dictionary.
        object.__setattr__(self, "_prism_records", records)
        object.__setattr__(self, "_prism_aux_dtype", _aux_dtype())
        object.__setattr__(self, "_prism_post_load_report", None)
        text = self.language_model
        for record in records:
            parent, leaf = _resolve_parent(text, record.path)
            original = getattr(parent, leaf)
            if record.embedding:
                if not isinstance(original, nn.Embedding):
                    raise _refuse(f"{record.path!r} is not an embedding")
                rows, width = original.weight.shape
                packed: nn.Module = HadamardQuantizedEmbedding(
                    rows, width, block=record.block
                )
            else:
                if not isinstance(original, nn.Linear) or "bias" in original:
                    raise _refuse(f"{record.path!r} is not a bias-free linear layer")
                rows, width = original.weight.shape
                packed = HadamardQuantizedLinear(width, rows, block=record.block)
            setattr(parent, leaf, packed)

    # -- weights -----------------------------------------------------------

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        if not any(key.startswith(VISION_PREFIXES) for key in weights):
            raise _refuse(
                "the weight files carry no vision tower tensors "
                "(vision_tower.*). MTPLX never loads this model type as text "
                "only; restore the vision tower from the original pack."
            )
        packed_keys: dict[str, PackedModuleRecord] = {}
        for record in self._prism_records:
            packed_keys[LANGUAGE_PREFIX + record.path] = record

        sanitized: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.startswith(VISION_PREFIXES):
                # Loaded from the same files by mtplx.vision, as stored.
                continue
            if not key.startswith(LANGUAGE_PREFIX):
                raise _refuse(f"unexpected tensor {key!r} outside {LANGUAGE_PREFIX}*")
            if key.startswith(LANGUAGE_PREFIX + "mtp."):
                # A draft head never lives inside the trunk shards of this
                # pack type; mtp.safetensors carries it. Dropped without the
                # +1.0 norm shift mlx-lm keys on these names: every norm in a
                # Prism pack is already absolute.
                continue
            sanitized[key] = value

        self._validate_packed_tensors(sanitized, packed_keys)

        aux_dtype = self._prism_aux_dtype
        packed_prefixes = tuple(prefix + "." for prefix in packed_keys)
        for key, value in list(sanitized.items()):
            if key.startswith(packed_prefixes):
                continue
            if mx.issubdtype(value.dtype, mx.floating) and value.dtype != aux_dtype:
                sanitized[key] = value.astype(aux_dtype)
        return sanitized

    def _validate_packed_tensors(
        self,
        weights: dict[str, mx.array],
        packed_keys: dict[str, PackedModuleRecord],
    ) -> None:
        for prefix, record in packed_keys.items():
            parent, leaf = _resolve_parent(self.language_model, record.path)
            module = getattr(parent, leaf)
            expected = {
                "weight": (module.weight.shape, (mx.uint32,)),
                "scales": (module.scales.shape, (mx.float16,)),
                "biases": (module.biases.shape, (mx.float16,)),
            }
            if record.block:
                expected["signs"] = (module.signs.shape, (mx.float32,))
            for suffix, (shape, dtypes) in expected.items():
                tensor = weights.get(f"{prefix}.{suffix}")
                if tensor is None:
                    raise _refuse(
                        f"missing tensor {prefix}.{suffix}. A rotated module "
                        "without its sign vector or packed weights cannot be "
                        "run correctly."
                    )
                if tuple(tensor.shape) != tuple(shape) or tensor.dtype not in dtypes:
                    raise _refuse(
                        f"{prefix}.{suffix} is {tensor.dtype} {tuple(tensor.shape)}, "
                        f"expected {dtypes[0]} {tuple(shape)}"
                    )
            if not record.block and f"{prefix}.signs" in weights:
                raise _refuse(f"{prefix} declares no rotation but carries signs")
        # Anything that looks packed but is not in the module list would be
        # run without its rotation by a stock quantized layer.
        for key in weights:
            if key.endswith(".signs") and key[: -len(".signs")] not in packed_keys:
                raise _refuse(
                    f"{key} belongs to a module that config.json does not list"
                )

    # -- after load --------------------------------------------------------

    def post_weight_load(self, model_path: Path | str) -> dict[str, Any]:
        """Value checks and wiring that need the loaded tensors and the folder.

        Called by mtplx.runtime right after the weights load. Refuses packs
        whose rotation metadata file, sign values or vision sidecar are
        missing or inconsistent, then arms the shared sibling transforms.
        """

        path = Path(model_path)
        report: dict[str, Any] = {"model_type": MODEL_TYPE}
        if not (path / PREPROCESSOR_FILE).is_file():
            raise _refuse(
                f"{PREPROCESSOR_FILE} is missing. The vision tower cannot run "
                "without it and MTPLX never loads this model type as text only."
            )
        report["hadamard_config"] = self._check_rotation_metadata(path)
        report["shared_rotation_groups"] = self._arm_shared_rotations()
        report["ternary_layout_modules"] = self._arm_ternary_kernels()
        report["float_dtypes"] = self._check_float_dtypes()
        report["packed_modules"] = len(self._prism_records)
        object.__setattr__(self, "_prism_post_load_report", report)
        return report

    def _packed_modules(self) -> list[tuple[PackedModuleRecord, Any]]:
        out = []
        for record in self._prism_records:
            parent, leaf = _resolve_parent(self.language_model, record.path)
            out.append((record, getattr(parent, leaf)))
        return out

    def _check_rotation_metadata(self, path: Path) -> str:
        packed = self._packed_modules()
        declared = str(self.args.hadamard_config or "").strip()
        if not declared:
            # No side file declared: the sign tensors are the only metadata,
            # so their values are checked directly.
            ok = mx.array(True)
            for record, module in packed:
                if record.block:
                    signs = module["signs"]
                    ok = ok & mx.all((signs == 1) | (signs == -1))
            if not bool(ok.item()):
                raise _refuse("a sign vector holds values other than +1 and -1")
            return "not declared; sign tensors checked"

        meta_path = path / declared
        if not meta_path.is_file():
            raise _refuse(
                f"config.json names {declared!r} as the rotation metadata but "
                "the file is missing"
            )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _refuse(f"{declared} is unreadable: {exc}") from None
        if not isinstance(meta, dict):
            raise _refuse(f"{declared} is not a JSON object")

        def field_of(name: str) -> Any:
            return meta.get("prism.hadamard." + name)

        if field_of("version") != 1:
            raise _refuse(f"{declared}: unsupported version {field_of('version')!r}")
        if field_of("transform") != SUPPORTED_TRANSFORM:
            raise _refuse(f"{declared}: unsupported transform {field_of('transform')!r}")
        if field_of("axis") != SUPPORTED_TRANSFORM_AXIS:
            raise _refuse(f"{declared}: unsupported axis {field_of('axis')!r}")
        if field_of("sign_mode") != "explicit":
            raise _refuse(f"{declared}: explicit signs are required")
        if field_of("gdn_v_grouped") is not True:
            raise _refuse(f"{declared}: GatedDeltaNet value heads are not grouped")
        block = field_of("block_size")
        widths = field_of("sign_widths")
        values = field_of("sign_values")
        if not isinstance(widths, list) or not isinstance(values, list):
            raise _refuse(f"{declared}: sign widths or values are missing")
        if sum(int(w) for w in widths) != len(values):
            raise _refuse(f"{declared}: sign values do not match the sign widths")
        if any(v not in (1, -1, 1.0, -1.0) for v in values):
            raise _refuse(f"{declared}: a sign value is not +1 or -1")
        by_width: dict[int, mx.array] = {}
        offset = 0
        for width in widths:
            width = int(width)
            by_width[width] = mx.array(values[offset : offset + width], dtype=mx.float32)
            offset += width

        forward = {
            LANGUAGE_PREFIX + r.path + ".weight" for r, _ in packed if not r.embedding
        }
        inverse = {
            LANGUAGE_PREFIX + r.path + ".weight" for r, _ in packed if r.embedding
        }
        if set(field_of("weight_names") or ()) != forward:
            raise _refuse(f"{declared}: weight_names differ from config.json modules")
        if set(field_of("inverse_weight_names") or ()) != inverse:
            raise _refuse(
                f"{declared}: inverse_weight_names differ from config.json modules"
            )

        ok = mx.array(True)
        for record, module in packed:
            if not record.block:
                raise _refuse(f"{declared} lists {record.path!r} but it has no block")
            if record.block != block:
                raise _refuse(
                    f"{record.path!r} uses block {record.block}, {declared} says {block}"
                )
            signs = module["signs"]
            reference = by_width.get(int(signs.shape[0]))
            if reference is None:
                raise _refuse(f"{declared} has no sign vector of width {signs.shape[0]}")
            ok = ok & mx.array_equal(signs, reference)
        if not bool(ok.item()):
            raise _refuse(f"a sign tensor differs from {declared}")
        return declared

    def _arm_shared_rotations(self) -> int:
        groups = 0
        by_parent: dict[str, dict[str, Any]] = {}
        for record, module in self._packed_modules():
            if isinstance(module, HadamardQuantizedLinear):
                module._rotation = None
                parent, _, leaf = record.path.rpartition(".")
                by_parent.setdefault(parent, {})[leaf] = module
        if not _share_rotation_enabled():
            return 0
        for members in by_parent.values():
            for names in _SIBLING_GROUPS:
                siblings = [members[name] for name in names if name in members]
                if len(siblings) < 2 or not all(m.block for m in siblings):
                    continue
                first = siblings[0]
                same = mx.array(True)
                for other in siblings[1:]:
                    if other.block != first.block or other.input_dims != first.input_dims:
                        same = mx.array(False)
                        break
                    same = same & mx.array_equal(other["signs"], first["signs"])
                if not bool(same.item()):
                    continue
                shared = _SharedRotation(len(siblings))
                for module in siblings:
                    module._rotation = shared
                groups += 1
        return groups

    def _arm_ternary_kernels(self) -> int:
        """Mark every packed projection whose biases are exactly -scales.

        Only those matrices may take the ternary kernel: it folds the bias
        into the code (a weight is scale * (code - 1)), which is exact for
        that layout and wrong for any other affine 2-bit matrix. Matrices
        too small to beat stock (the key and value projections) keep stock.
        """

        armed = 0
        for _record, module in self._packed_modules():
            if not isinstance(module, HadamardQuantizedLinear):
                continue
            planned = None
            if ternary_worthwhile(int(module["weight"].shape[0])) and ternary_layout(
                module["scales"], module["biases"]
            ):
                planned = ternary_plan(module["weight"], module["scales"])
            module._ternary_plan = planned
            module._ternary = planned is not None
            armed += int(module._ternary)
        return armed

    def _check_float_dtypes(self) -> list[str]:
        """No bfloat16 anywhere: the pack is float16 scaled on every chip."""

        found: set[str] = set()
        for name, value in tree_flatten(self.language_model.parameters()):
            if not mx.issubdtype(value.dtype, mx.floating):
                continue
            if value.dtype == mx.bfloat16:
                raise _refuse(
                    f"{name} is bfloat16. This pack runs in float16 on every "
                    "chip; a bfloat16 tensor would push every quantized matmul "
                    "to float32 and is emulated on M1 and M2."
                )
            found.add(str(value.dtype).rsplit(".", 1)[-1])
        return sorted(found)
