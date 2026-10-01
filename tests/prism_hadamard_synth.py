"""Synthetic prism_hadamard_qwen35 packs for CPU tests.

Builds a two-layer Qwen3.5-family pack (one GatedDeltaNet layer, one full
attention layer) in exactly the container Prism ML publishes: ternary codes in
MLX's affine 2-bit words (group 128, scale s, bias -s, fp16), rows stored after
the blockwise Walsh-Hadamard transform with a fixed sign vector per input
width, float32 auxiliary tensors, a float16 vision tower in the same file, the
``modules`` list in config.json and a ``hadamard.json`` side file.

Alongside the pack it returns the dense float32 weights the pack encodes (the
rows transformed back to the standard basis in float64 numpy), so a test can
compare the rotated quantized model against a plain dense reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BLOCK = 1024
GROUP = 128
HIDDEN = 1024
INTERMEDIATE = 2048
VOCAB = 384
IMAGE_TOKEN_ID = 380
VIDEO_TOKEN_ID = 381
VISION_START_TOKEN_ID = 382
VISION_END_TOKEN_ID = 383

TEXT_CONFIG: dict[str, Any] = {
    "model_type": "qwen3_5_text",
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "vocab_size": VOCAB,
    "max_position_embeddings": 4096,
    "linear_num_value_heads": 8,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mtp_num_hidden_layers": 0,
    "mtp_use_dedicated_embeddings": False,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    },
}

VISION_CONFIG: dict[str, Any] = {
    "model_type": "qwen3_5",
    "depth": 1,
    "hidden_size": 32,
    "intermediate_size": 64,
    "out_hidden_size": HIDDEN,
    "num_heads": 2,
    "patch_size": 4,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 16,
    "deepstack_visual_indexes": [],
}

PREPROCESSOR_CONFIG: dict[str, Any] = {
    "size": {"longest_edge": 4096, "shortest_edge": 64},
    "patch_size": 4,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


def fwht_blocks(values: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Normalized blockwise Walsh-Hadamard transform on the last axis (float64)."""

    out = np.asarray(values, dtype=np.float64).copy()
    shape = out.shape
    out = out.reshape(-1, block)
    h = 1
    while h < block:
        view = out.reshape(out.shape[0], -1, 2, h)
        a = view[:, :, 0, :].copy()
        b = view[:, :, 1, :].copy()
        view[:, :, 0, :] = a + b
        view[:, :, 1, :] = a - b
        h *= 2
    return (out / np.sqrt(block)).reshape(shape)


def rotate(values: np.ndarray, signs: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """T(x) = H(s * x); inverse T^-1(y) = s * H(y)."""

    if inverse:
        return fwht_blocks(values) * signs
    return fwht_blocks(np.asarray(values, dtype=np.float64) * signs)


def pack_ternary(rotated_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ternary codes {0,1,2} in 2-bit little-endian uint32 words, fp16 s and -s."""

    rows, width = rotated_rows.shape
    groups = rotated_rows.reshape(rows, width // GROUP, GROUP)
    scales = (np.abs(groups).mean(axis=-1) * 1.25 + 1e-4).astype(np.float16)
    codes = np.clip(np.rint(groups / scales.astype(np.float64)[..., None]), -1, 1) + 1
    codes = codes.astype(np.uint32).reshape(rows, width // 16, 16)
    words = np.bitwise_or.reduce(
        codes << (2 * np.arange(16, dtype=np.uint32)), axis=-1
    ).astype(np.uint32)
    return words, scales, (-scales).astype(np.float16)


def dequantize_ternary(words: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows, n_words = words.shape
    codes = np.empty((rows, n_words * 16), dtype=np.float64)
    for lane in range(16):
        codes[:, lane::16] = (words >> np.uint32(2 * lane)) & np.uint32(3)
    groups = codes.reshape(rows, -1, GROUP) - 1.0
    return (groups * scales.astype(np.float64)[..., None]).reshape(rows, -1)


@dataclass
class SyntheticPack:
    path: Path
    config: dict[str, Any]
    dense: dict[str, np.ndarray]
    signs: dict[int, np.ndarray]
    module_paths: list[str]


def _packed_module_shapes() -> dict[str, tuple[int, int]]:
    cfg = TEXT_CONFIG
    key_dim = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    value_dim = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    q_out = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_out = cfg["num_key_value_heads"] * cfg["head_dim"]
    shapes: dict[str, tuple[int, int]] = {
        "lm_head": (VOCAB, HIDDEN),
        "model.embed_tokens": (VOCAB, HIDDEN),
        # layer 0: GatedDeltaNet
        "model.layers.0.linear_attn.in_proj_qkv": (2 * key_dim + value_dim, HIDDEN),
        "model.layers.0.linear_attn.in_proj_z": (value_dim, HIDDEN),
        "model.layers.0.linear_attn.out_proj": (HIDDEN, value_dim),
        # layer 1: gated full attention (q_proj carries the output gate)
        "model.layers.1.self_attn.q_proj": (2 * q_out, HIDDEN),
        "model.layers.1.self_attn.k_proj": (kv_out, HIDDEN),
        "model.layers.1.self_attn.v_proj": (kv_out, HIDDEN),
        "model.layers.1.self_attn.o_proj": (HIDDEN, q_out),
    }
    for layer in (0, 1):
        shapes[f"model.layers.{layer}.mlp.gate_proj"] = (INTERMEDIATE, HIDDEN)
        shapes[f"model.layers.{layer}.mlp.up_proj"] = (INTERMEDIATE, HIDDEN)
        shapes[f"model.layers.{layer}.mlp.down_proj"] = (HIDDEN, INTERMEDIATE)
    return shapes


def _aux_tensors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    cfg = TEXT_CONFIG
    key_dim = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    value_dim = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    conv_dim = 2 * key_dim + value_dim
    heads = cfg["linear_num_value_heads"]

    def norm(width: int) -> np.ndarray:
        return (1.0 + 0.05 * rng.standard_normal(width)).astype(np.float32)

    aux: dict[str, np.ndarray] = {"model.norm.weight": norm(HIDDEN)}
    for layer in (0, 1):
        aux[f"model.layers.{layer}.input_layernorm.weight"] = norm(HIDDEN)
        aux[f"model.layers.{layer}.post_attention_layernorm.weight"] = norm(HIDDEN)
    prefix = "model.layers.0.linear_attn."
    aux[prefix + "conv1d.weight"] = (
        0.2 * rng.standard_normal((conv_dim, cfg["linear_conv_kernel_dim"], 1))
    ).astype(np.float32)
    aux[prefix + "in_proj_a.weight"] = (
        0.02 * rng.standard_normal((heads, HIDDEN))
    ).astype(np.float32)
    aux[prefix + "in_proj_b.weight"] = (
        0.02 * rng.standard_normal((heads, HIDDEN))
    ).astype(np.float32)
    aux[prefix + "A_log"] = np.log(rng.uniform(0.5, 8.0, size=heads)).astype(np.float32)
    aux[prefix + "dt_bias"] = rng.uniform(-1.0, 1.0, size=heads).astype(np.float32)
    aux[prefix + "norm.weight"] = norm(cfg["linear_value_head_dim"])
    aux["model.layers.1.self_attn.q_norm.weight"] = norm(cfg["head_dim"])
    aux["model.layers.1.self_attn.k_norm.weight"] = norm(cfg["head_dim"])
    return aux


def _vision_tensors(seed: int) -> dict[str, np.ndarray]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionConfig, Qwen3VLVisionTower

    mx.random.seed(seed)
    tower = Qwen3VLVisionTower(Qwen3VLVisionConfig.from_dict(VISION_CONFIG))
    tensors: dict[str, np.ndarray] = {}
    for name, value in tree_flatten(tower.parameters()):
        tensors["vision_tower." + name] = np.asarray(value.astype(mx.float32)).astype(
            np.float16
        )
    return tensors


def build_synthetic_pack(
    directory: Path | str,
    *,
    seed: int = 7,
    with_vision: bool = True,
    with_preprocessor: bool = True,
    with_hadamard_json: bool = True,
) -> SyntheticPack:
    import mlx.core as mx

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    shapes = _packed_module_shapes()
    widths = sorted({width for _rows, width in shapes.values()})
    signs = {
        width: rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
        for width in widths
    }

    tensors: dict[str, np.ndarray] = {}
    dense: dict[str, np.ndarray] = {}
    for path, (rows, width) in shapes.items():
        original = rng.standard_normal((rows, width)) / np.sqrt(width)
        words, scales, biases = pack_ternary(rotate(original, signs[width]))
        key = "language_model." + path
        tensors[key + ".weight"] = words
        tensors[key + ".scales"] = scales
        tensors[key + ".biases"] = biases
        tensors[key + ".signs"] = signs[width]
        # The dense matrix the pack encodes: dequantize, then undo the rotation.
        dense[path + ".weight"] = rotate(
            dequantize_ternary(words, scales), signs[width], inverse=True
        ).astype(np.float32)
    for name, value in _aux_tensors(rng).items():
        tensors["language_model." + name] = value
        dense[name] = value
    if with_vision:
        tensors.update(_vision_tensors(seed))

    mx.save_safetensors(
        str(directory / "model.safetensors"),
        {name: mx.array(value) for name, value in tensors.items()},
        metadata={"format": "mlx"},
    )

    modules = [
        {
            "path": path,
            "block": BLOCK,
            "embedding": path == "model.embed_tokens",
            "dtype": "float16",
        }
        for path in shapes
    ]
    config: dict[str, Any] = {
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "text_config": dict(TEXT_CONFIG),
        "modules": modules,
        "quantization": {"bits": 2, "group_size": GROUP, "mode": "affine"},
        "requires_runtime": "runtime/artifact.py",
        "hadamard_config": "hadamard.json",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "gdn_activation_layout": "grouped",
        "components": {"text": True, "vision": True, "mtp": False},
        "base_model_type": "qwen3_5",
        "vision_config": dict(VISION_CONFIG),
        "image_token_id": IMAGE_TOKEN_ID,
        "video_token_id": VIDEO_TOKEN_ID,
        "vision_start_token_id": VISION_START_TOKEN_ID,
        "vision_end_token_id": VISION_END_TOKEN_ID,
        "tie_word_embeddings": False,
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2))

    if with_hadamard_json:
        hadamard = {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": BLOCK,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": sorted(
                f"language_model.{path}.weight"
                for path in shapes
                if path != "model.embed_tokens"
            ),
            "prism.hadamard.inverse_weight_names": [
                "language_model.model.embed_tokens.weight"
            ],
            "prism.hadamard.sign_widths": widths,
            "prism.hadamard.sign_values": [
                float(v) for width in widths for v in signs[width]
            ],
            "prism.hadamard.gdn_v_grouped": True,
        }
        (directory / "hadamard.json").write_text(json.dumps(hadamard))
    if with_preprocessor:
        (directory / "preprocessor_config.json").write_text(
            json.dumps(PREPROCESSOR_CONFIG, indent=2)
        )
    _write_tiny_tokenizer(directory)
    return SyntheticPack(
        path=directory,
        config=config,
        dense=dense,
        signs=signs,
        module_paths=list(shapes),
    )


def _write_tiny_tokenizer(directory: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    from tokenizers import AddedToken

    vocab = {f"t{index}": index for index in range(VOCAB)}
    specials = {
        IMAGE_TOKEN_ID: "<|image_pad|>",
        VIDEO_TOKEN_ID: "<|video_pad|>",
        VISION_START_TOKEN_ID: "<|vision_start|>",
        VISION_END_TOKEN_ID: "<|vision_end|>",
    }
    for index, text in specials.items():
        vocab.pop(f"t{index}")
        vocab[text] = index
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(
        [AddedToken(text, special=True) for _index, text in sorted(specials.items())]
    )
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "chat_template": (
                    "{% for message in messages %}{{ message['content'] }} {% endfor %}"
                ),
                "eos_token": "t2",
                "pad_token": "t3",
                "unk_token": "t0",
            }
        )
    )


def build_dense_reference(pack: SyntheticPack, *, dtype: Any = None) -> Any:
    """The plain mlx-lm Qwen3.5 model holding the dense weights the pack encodes."""

    import mlx.core as mx
    from mlx_lm.models import qwen3_5

    dtype = dtype or mx.float32
    model = qwen3_5.Model(
        qwen3_5.ModelArgs(model_type="qwen3_5", text_config=dict(TEXT_CONFIG))
    )
    weights = [
        ("language_model." + name, mx.array(value).astype(dtype))
        for name, value in pack.dense.items()
    ]
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def write_synthetic_mtp_sidecar(pack: SyntheticPack, *, seed: int = 11) -> Path:
    """A dense float16 one-layer MTP head in the published mtp.* namespace."""

    import mlx.core as mx

    rng = np.random.default_rng(seed)
    cfg = TEXT_CONFIG
    q_out = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_out = cfg["num_key_value_heads"] * cfg["head_dim"]

    def matrix(rows: int, width: int) -> np.ndarray:
        return (rng.standard_normal((rows, width)) / np.sqrt(width)).astype(np.float16)

    def norm(width: int) -> np.ndarray:
        return (1.0 + 0.05 * rng.standard_normal(width)).astype(np.float16)

    layer = "mtp.layers.0."
    tensors = {
        "mtp.fc.weight": matrix(HIDDEN, 2 * HIDDEN),
        "mtp.norm.weight": norm(HIDDEN),
        "mtp.pre_fc_norm_embedding.weight": norm(HIDDEN),
        "mtp.pre_fc_norm_hidden.weight": norm(HIDDEN),
        layer + "input_layernorm.weight": norm(HIDDEN),
        layer + "post_attention_layernorm.weight": norm(HIDDEN),
        layer + "self_attn.q_proj.weight": matrix(2 * q_out, HIDDEN),
        layer + "self_attn.k_proj.weight": matrix(kv_out, HIDDEN),
        layer + "self_attn.v_proj.weight": matrix(kv_out, HIDDEN),
        layer + "self_attn.o_proj.weight": matrix(HIDDEN, q_out),
        layer + "self_attn.q_norm.weight": norm(cfg["head_dim"]),
        layer + "self_attn.k_norm.weight": norm(cfg["head_dim"]),
        layer + "mlp.gate_proj.weight": matrix(INTERMEDIATE, HIDDEN),
        layer + "mlp.up_proj.weight": matrix(INTERMEDIATE, HIDDEN),
        layer + "mlp.down_proj.weight": matrix(HIDDEN, INTERMEDIATE),
    }
    target = pack.path / "mtp.safetensors"
    mx.save_safetensors(
        str(target),
        {name: mx.array(value) for name, value in tensors.items()},
        metadata={"format": "mlx"},
    )
    config = json.loads((pack.path / "config.json").read_text())
    config["text_config"]["mtp_num_hidden_layers"] = 1
    config["components"]["mtp"] = True
    (pack.path / "config.json").write_text(json.dumps(config, indent=2))
    return target
