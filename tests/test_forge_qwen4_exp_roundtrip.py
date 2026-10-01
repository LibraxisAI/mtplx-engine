"""The Flash-Next Forge lane writes files the runtime's own loaders read back.

PR #508's unit tests cover the arithmetic of the lane. These cover the two
places where a converter and a runtime can disagree without anyone noticing:

* the n-gram table. The lane streams it shard by shard into a hand-written
  safetensors file; the runtime gathers rows from that file with its own
  reader. A layout mismatch reads garbage rows, not an error.
* the draft head. A raw checkpoint stores its norms zero-centred ((1 + w)
  convention) and its experts packed; the runtime expects absolute norms and
  split experts, and shifts two of the norms itself. A head that is shifted
  twice, or not at all, still loads, still runs, and proposes nonsense:
  acceptance near zero and decode slower than plain decoding, with no
  warning (the defect PR #511 fixed for another family).

Both are checked end to end on a tiny synthetic source, on the CPU device.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mtplx.commands import forge_qwen4_exp as lane
from mtplx.models import qwen4_exp as runtime
from mtplx.models.qwen4_exp import Model, Qwen4ExpMTP, TextArgs


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


# --------------------------------------------------------------------- n-gram
def _ngram_source(root: Path, shards: list[mx.array]) -> Path:
    source = root / "source"
    source.mkdir()
    weight_map = {}
    for index, table in enumerate(shards):
        key = lane.NGRAM_KEY.format(layer=1, index=index)
        name = f"model-{index:05d}.safetensors"
        # A second tensor in front of the table moves its data offset off zero.
        mx.save_safetensors(
            str(source / name),
            {"a.pad": mx.zeros((3,), dtype=mx.float16), key: table},
        )
        weight_map[key] = name
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": {"ple_layer_ids": [2]}})
    )
    return source


def test_the_streamed_ngram_table_reads_back_through_the_runtime_gather(tmp_path):
    mx.random.seed(3)
    shards = [mx.random.normal((rows, 160)).astype(mx.bfloat16) for rows in (7, 5)]
    source = _ngram_source(tmp_path, shards)

    report = lane.write_ngram_sidecar(source, tmp_path / "pack", bits=4, group=32)

    assert report["rows"] == 12 and report["dim"] == 160
    written = tmp_path / "pack" / lane.NGRAM_FILE
    assert not written.with_suffix(".partial").exists()

    # Shard by shard equals the whole table at once: quantization is per row.
    q, s, b = mx.quantize(mx.concatenate(shards, axis=0), group_size=32, bits=4)
    loaded = mx.load(str(written))
    assert bool(mx.all(loaded["ngram.weight"] == q))
    assert bool(mx.all(loaded["ngram.scales"].view(mx.uint16) == s.view(mx.uint16)))
    assert bool(mx.all(loaded["ngram.biases"].view(mx.uint16) == b.view(mx.uint16)))

    # The runtime's reader, built the way NGramTable.attach_sidecar builds it.
    header, data_start = runtime._read_safetensors_header(written)
    meta = header["__metadata__"]
    entries = {
        name: (header[f"ngram.{name}"], data_start)
        for name in ("weight", "scales", "biases")
    }
    gather = runtime._SidecarGather(
        written,
        entries,
        bits=int(meta["ngram_bits"]),
        group_size=int(meta["ngram_group_size"]),
    )
    ids = np.array([0, 6, 7, 11, 3], dtype=np.int64)  # both shards, both edges
    want = mx.dequantize(q, s, b, group_size=32, bits=4)[mx.array(ids)]
    got = gather.gather_np(ids)
    assert got.shape == (5, 160)
    assert bool(mx.all(got.astype(mx.float32) == want.astype(mx.float32)))


def test_the_production_layout_is_the_one_the_runtime_defaults_to():
    # attach_sidecar falls back to 4-bit / 32 when the metadata is absent.
    assert lane.NGRAM_PRODUCTION_LAYOUT == (4, 32)


# ----------------------------------------------------------------- draft head
def _tiny_args() -> TextArgs:
    # 2 * moe_intermediate != hidden and moe_intermediate != hidden, so the
    # packed-expert layout is decidable, as it is on the real model
    # (hidden 2560, expert width 640).
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=96,
        shared_expert_intermediate_size=128,
    )


_NORM_SUFFIXES = (
    *Model._HF_NORM_SHIFT_SUFFIXES,
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def _raw_head(args: TextArgs, *, layout: str) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """A raw checkpoint's ``mtp.`` tensors and the head the runtime must end
    up with. Norms are stored zero-centred in the raw form."""

    mx.random.seed(11)
    expected: dict[str, mx.array] = {}
    raw: dict[str, mx.array] = {}
    experts: dict[str, mx.array] = {}
    for name, value in tree_flatten(Qwen4ExpMTP(args).parameters()):
        tensor = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16)
        if value.ndim == 1 and any(name.endswith(s) for s in _NORM_SUFFIXES):
            raw["mtp." + name] = tensor
            expected[name] = (tensor.astype(mx.float32) + 1.0).astype(mx.bfloat16)
            continue
        expected[name] = tensor
        if ".mlp.switch_mlp." in name:
            experts[name] = tensor
            continue
        raw["mtp." + name] = tensor
    prefix = "layers.0.mlp"
    gate = experts[f"{prefix}.switch_mlp.gate_proj.weight"]  # [E, inter, hidden]
    up = experts[f"{prefix}.switch_mlp.up_proj.weight"]
    down = experts[f"{prefix}.switch_mlp.down_proj.weight"]  # [E, hidden, inter]
    if layout == "hub":  # Linear [out, in] halves
        raw[f"mtp.{prefix}.experts.gate_up_proj"] = mx.concatenate([gate, up], axis=1)
        raw[f"mtp.{prefix}.experts.down_proj"] = down
    else:  # transformers save_pretrained: the bmm orientation
        raw[f"mtp.{prefix}.experts.gate_up_proj"] = mx.concatenate(
            [gate.swapaxes(1, 2), up.swapaxes(1, 2)], axis=-1
        )
        raw[f"mtp.{prefix}.experts.down_proj"] = down.swapaxes(1, 2)
    return raw, expected


def _head_source(root: Path, raw: dict[str, mx.array], args: TextArgs) -> Path:
    source = root / "source"
    source.mkdir()
    mx.save_safetensors(str(source / "model-00001.safetensors"), raw)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model-00001.safetensors" for key in raw}})
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": {"hidden_size": args.hidden_size}})
    )
    return source


def _attach(pack: Path, args: TextArgs) -> Qwen4ExpMTP:
    holder = SimpleNamespace(language_model=SimpleNamespace(args=args))
    assert Model.attach_mtp(holder, pack) is True  # strict load: every key, no extras
    return holder.language_model.mtp


@pytest.mark.parametrize("layout", ["hub", "transformers"])
def test_a_bf16_head_loads_with_absolute_norms_and_split_experts(tmp_path, layout):
    args = _tiny_args()
    raw, expected = _raw_head(args, layout=layout)
    source = _head_source(tmp_path, raw, args)

    report = lane.write_mtp_sidecar(source, tmp_path / "pack", mtp_bits=0, mtp_group=32)

    assert report["written"] is True
    head = _attach(tmp_path / "pack", args)
    loaded = dict(tree_flatten(head.parameters()))
    # "bf16" is the experts and the attention projections. The router gate,
    # the shared expert and the indexer projection are 8-bit in every recipe,
    # as they are in the trunk.
    eight_bit = {name[: -len(".scales")] for name in loaded if name.endswith(".scales")}
    assert eight_bit == {
        "layers.0.mlp.gate",
        "layers.0.mlp.shared_expert_gate",
        "layers.0.mlp.shared_expert.gate_proj",
        "layers.0.mlp.shared_expert.up_proj",
        "layers.0.mlp.shared_expert.down_proj",
        "layers.0.self_attn.indexer.index_qk_proj",
    }
    for name, want in expected.items():
        module = name[: -len(".weight")]
        if module in eight_bit:
            decoded = mx.dequantize(
                loaded[name],
                loaded[f"{module}.scales"],
                loaded[f"{module}.biases"],
                group_size=64,
                bits=8,
            )
            error = mx.max(mx.abs(decoded.astype(mx.float32) - want.astype(mx.float32)))
            # 8-bit steps over a +-0.15 range, with the scale and bias held
            # in bf16 (8 mantissa bits): about 0.0025 at worst.
            assert float(error) < 0.005, name
            continue
        got = loaded[name]
        assert got.shape == want.shape, name
        assert bool(mx.all(got.astype(mx.float32) == want.astype(mx.float32))), name
    assert {n for n in loaded if n.endswith(".weight")} == set(expected)
    # The two the runtime shifts itself were left alone by the lane, the rest
    # were shifted exactly once: every norm sits near 1, none near 0 or 2.
    for name, got in loaded.items():
        if got.ndim == 1 and any(name.endswith(s) for s in _NORM_SUFFIXES):
            assert 0.5 < float(mx.mean(got.astype(mx.float32))) < 1.5, name


@pytest.mark.parametrize("bits,group", [(4, 32), (8, 64)])
def test_a_quantized_head_loads_with_the_trunk_recipe(tmp_path, bits, group):
    args = _tiny_args()
    if bits == 8:
        args.moe_intermediate_size = 128
    raw, expected = _raw_head(args, layout="hub")
    source = _head_source(tmp_path, raw, args)

    lane.write_mtp_sidecar(source, tmp_path / "pack", mtp_bits=bits, mtp_group=group, qsa_8bit=True)

    head = _attach(tmp_path / "pack", args)
    layer = head.layers[0]
    assert (layer.mlp.switch_mlp.gate_proj.bits, layer.mlp.switch_mlp.gate_proj.group_size) == (bits, group)
    assert (layer.mlp.switch_mlp.down_proj.bits, layer.mlp.switch_mlp.down_proj.group_size) == (bits, group)
    assert (layer.mlp.gate.bits, layer.mlp.gate.group_size) == (8, 64)
    assert (layer.mlp.shared_expert.down_proj.bits, layer.mlp.shared_expert.down_proj.group_size) == (8, 64)
    assert (layer.self_attn.q_proj.bits, layer.self_attn.q_proj.group_size) == (8, 64)
    assert not hasattr(head.fc_hidden, "bits")  # structural pieces stay bf16
    # Quantized experts decode back to the source within 4-bit error.
    gate = layer.mlp.switch_mlp.gate_proj
    decoded = mx.dequantize(gate.weight, gate.scales, gate.biases, group_size=group, bits=bits)
    want = expected["layers.0.mlp.switch_mlp.gate_proj.weight"].astype(mx.float32)
    assert float(mx.max(mx.abs(decoded.astype(mx.float32) - want))) < (0.005 if bits == 8 else 0.02)


def test_the_lanes_norm_list_is_the_models_own():
    """A copy that drifts from the model's list would write a head whose norms
    are wrong by exactly 1.0, which loads and runs."""
    assert tuple(lane.MTP_NORM_SHIFT_SUFFIXES) == tuple(Model._HF_NORM_SHIFT_SUFFIXES)


def test_the_lanes_eight_bit_set_is_inside_the_models_recipe():
    predicate = Model.quant_predicate.fget(None)
    module = SimpleNamespace(to_quantized=lambda **_: None)
    for suffix in lane.MTP_EIGHT_BIT_SUFFIXES:
        assert predicate(f"language_model.model.layers.3.{suffix}", module) == {
            "bits": 8,
            "group_size": 64,
        }, suffix


def _quality_source(root: Path, layout: str):
    from dataclasses import asdict
    from mtplx.models.qwen4_exp import ModelArgs
    from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionConfig, Qwen3VLVisionTower

    args = _tiny_args()
    args.moe_intermediate_size = 128  # every Q8 matrix has group-64 input width
    args.vocab_size = 128
    args.linear_num_value_heads, args.linear_num_key_heads = 4, 2
    args.linear_key_head_dim = args.linear_value_head_dim = 16
    args.hc_count, args.hc_lowrank = 2, 16
    args.ple_layer_ids, args.ple_embed_dim = [2], 640
    args.heads_per_ngram, args.ngram_vocab_size_base = 2, 7
    args.make_ngram_vocab_size_divisible_by, args.ngram_sidecar = 4, True
    config = {"model_type": "qwen4_exp", "torch_dtype": "bfloat16", "text_config": asdict(args)}
    model = Model(ModelArgs.from_dict(config))
    raw, experts = {}, {}
    mx.random.seed(2114)
    for key, value in tree_flatten(model.parameters()):
        if mx.issubdtype(value.dtype, mx.floating):
            value = (mx.random.normal(value.shape) * 0.03).astype(mx.bfloat16)
        if ".switch_mlp." in key:
            experts[key] = value
            continue
        raw_key = key.replace("language_model.model.", "model.language_model.", 1)
        if key == "language_model.lm_head.weight":
            raw_key = "lm_head.weight"
        if key.endswith("ple.conv_weight"):
            raw_key = raw_key.replace("ple.conv_weight", "ple.conv1d.weight")
            value = value.swapaxes(1, 2)
        elif key.endswith("linear_attn.conv1d.weight"):
            value = value.swapaxes(1, 2)
        raw[raw_key] = value
    for i in range(args.num_hidden_layers):
        prefix = f"language_model.model.layers.{i}.mlp"
        gate, up, down = [experts[f"{prefix}.switch_mlp.{proj}_proj.weight"] for proj in ("gate", "up", "down")]
        raw_prefix = f"model.language_model.layers.{i}.mlp.experts"
        raw[raw_prefix + ".gate_up_proj"] = (mx.concatenate([gate, up], axis=1) if layout == "hub"
            else mx.concatenate([gate.swapaxes(1, 2), up.swapaxes(1, 2)], axis=-1))
        raw[raw_prefix + ".down_proj"] = down if layout == "hub" else down.swapaxes(1, 2)
    table = model.layers[1].ple.ple_embedding.ngram_embedding
    for i, count in enumerate((table.rows // 2, table.rows - table.rows // 2)):
        raw[lane.NGRAM_KEY.format(layer=1, index=i)] = mx.random.normal((count, table.dim)).astype(mx.bfloat16)
    head, _ = _raw_head(args, layout=layout)
    raw.update(head)
    vision = {"model_type": "qwen3_5", "depth": 1, "hidden_size": 8, "intermediate_size": 16,
              "num_heads": 2, "out_hidden_size": 64, "patch_size": 2, "spatial_merge_size": 2,
              "temporal_patch_size": 1, "in_channels": 3, "num_position_embeddings": 4,
              "deepstack_visual_indexes": []}
    tower = Qwen3VLVisionTower(Qwen3VLVisionConfig.from_dict(vision))
    raw.update({"model.visual." + k: v.astype(mx.bfloat16) for k, v in tree_flatten(tower.parameters())})
    config["vision_config"] = vision
    source = root / "source"
    source.mkdir()
    index = {}
    for i in range(3):
        chunk = {k: v for j, (k, v) in enumerate(raw.items()) if j % 3 == i}
        filename = f"model-{i:05d}.safetensors"
        mx.save_safetensors(str(source / filename), chunk)
        index.update({k: filename for k in chunk})
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (source / "preprocessor_config.json").write_text(json.dumps({"patch_size": 2, "merge_size": 2, "temporal_patch_size": 1}))
    return source, args


@pytest.mark.parametrize("layout", ["hub", "transformers"])
def test_quality_pack_loaded_roundtrip_checks_every_stored_tensor_class(tmp_path, monkeypatch, layout):
    from functools import partial
    from mlx_lm.utils import load_model
    from mtplx.models.qwen4_exp import ModelArgs
    from mtplx.commands.forge_qwen4_audit import audit_pack, inventory, BF16_CLASSES
    from mtplx.vision_graft import graft_vision_tower

    # This is the actual converter and actual runtime loader, all on CPU.
    for name in ("MTPLX_FUSED_GATE_UP", "MTPLX_FUSED_GDN_INPROJ", "MTPLX_FUSED_QSA_QKV"):
        monkeypatch.setenv(name, "0")
    source, args = _quality_source(tmp_path, layout)
    pack = tmp_path / "quality"
    monkeypatch.setattr(lane, "convert_body", partial(lane.convert_body, shard_bytes=16_384))
    report = lane.run_lane(source, pack, recipe=lane.named_recipe(lane.QUALITY_RECIPE))
    assert report["body"]["shards"] > 1
    graft_vision_tower(source, pack, verify_load=True)
    audit = audit_pack(source, pack, lane.named_recipe(lane.QUALITY_RECIPE))
    assert len(audit["stored_precision"]) == 25
    assert all(audit["samples"][cls] for cls in audit["stored_precision"])
    for cls, info in audit["stored_precision"].items():
        expected = "BF16" if cls in BF16_CLASSES else "I64" if cls == "ple_integer_buffers" else (
            "Q4/g32 affine; BF16 scales and biases" if cls == "ngram_table" else "Q8/g64 affine; BF16 scales and biases")
        assert info["stored_precision"] == expected, cls
    # mlx_lm calls the hook by keyword: get_model_classes(config=config).
    loaded, _ = load_model(pack, lazy=True, get_model_classes=lambda config: (Model, ModelArgs))
    assert loaded.attach_mtp(pack)
    headers, _ = inventory(pack)
    actual = dict(tree_flatten(loaded.parameters()))
    for key, value in actual.items():
        stored_key = key.replace("language_model.mtp.", "mtp.", 1)
        assert list(value.shape) == headers[stored_key]["shape"], stored_key
        assert str(value.dtype).split(".")[-1] == {"U32": "uint32", "BF16": "bfloat16", "I64": "int64"}[headers[stored_key]["dtype"]]
    table = loaded.layers[1].ple.ple_embedding.ngram_embedding
    table.attach_sidecar(pack / lane.NGRAM_FILE)
    ids = np.array([0, table.rows // 2 - 1, table.rows // 2, table.rows - 1, 0], dtype=np.int64)
    packed = mx.load(str(pack / lane.NGRAM_FILE))
    expected = mx.dequantize(packed["ngram.weight"], packed["ngram.scales"], packed["ngram.biases"], bits=4, group_size=32)[mx.array(ids)]
    assert bool(mx.all(table._sidecar.gather_np(ids).astype(mx.float32) == expected.astype(mx.float32)))
