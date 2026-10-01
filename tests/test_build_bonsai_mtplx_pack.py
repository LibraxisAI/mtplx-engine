"""The Bonsai MTPLX pack builder: byte-for-byte trunk, vision kept, head contract."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest

from tests import prism_hadamard_synth as synth

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_bonsai_mtplx_pack.py"
_spec = importlib.util.spec_from_file_location("build_bonsai_mtplx_pack", _SCRIPT)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _source(tmp_path: Path, **kwargs) -> Path:
    pack = synth.build_synthetic_pack(tmp_path / "prism", **kwargs)
    files = {
        "model.safetensors": {
            "sha256": builder.sha256_file(pack.path / "model.safetensors"),
            "size": (pack.path / "model.safetensors").stat().st_size,
        }
    }
    (pack.path / "files.json").write_text(json.dumps(files))
    (pack.path / "LICENSE").write_text("Apache License 2.0 (test stand-in)\n")
    (pack.path / "NOTICE.txt").write_text("Created using Bonsai by Prism ML.\n")
    (pack.path / "chat_template.jinja").write_text("{{ messages }}")
    return pack.path


def _head_pack(tmp_path: Path, *, dtype=mx.float16, quantized: bool = False) -> Path:
    """A Qwen-style MTPLX pack folder that only carries what the builder reads."""

    scratch = synth.build_synthetic_pack(tmp_path / "head-scratch")
    synth.write_synthetic_mtp_sidecar(scratch)
    tensors = dict(mx.load(str(scratch.path / "mtp.safetensors")))
    out: dict[str, mx.array] = {}
    for name, value in tensors.items():
        if quantized and value.ndim == 2:
            weight, scales, biases = mx.quantize(
                value.astype(mx.float32), group_size=64, bits=4
            )
            stem = name.removesuffix(".weight")
            out[stem + ".weight"] = weight
            out[stem + ".scales"] = scales.astype(dtype)
            out[stem + ".biases"] = biases.astype(dtype)
        else:
            out[name] = value.astype(dtype)
    mx.eval(out)
    head_dir = tmp_path / "qwen-head-pack"
    head_dir.mkdir()
    mx.save_safetensors(str(head_dir / "mtp.safetensors"), out, metadata={"format": "mlx"})
    (head_dir / "processor_config.json").write_text(json.dumps({"processor_class": "Qwen3VLProcessor"}))
    return head_dir


def test_builds_a_complete_vision_pack_with_the_trunk_byte_for_byte(tmp_path):
    source = _source(tmp_path)
    head = _head_pack(tmp_path, quantized=True)
    output = tmp_path / "out" / builder.PACK_NAME
    output.parent.mkdir()
    manifest = builder.build_pack(
        source, output, mtp_source=builder.resolve_mtp_source(str(head))
    )

    # Same inode: the trunk and the vision tower are Prism's bytes, not a copy.
    assert os.stat(output / "model.safetensors").st_ino == os.stat(
        source / "model.safetensors"
    ).st_ino
    published = json.loads((source / "files.json").read_text())["model.safetensors"]["sha256"]
    assert manifest["files"]["model.safetensors"]["sha256"] == published
    assert manifest["provenance"]["source_model_sha256"] == published
    assert manifest["provenance"]["requantized"] is False
    assert manifest["provenance"]["vision_tensors"] > 0

    source_config = json.loads((source / "config.json").read_text())
    config = json.loads((output / "config.json").read_text())
    for key in ("model_type", "modules", "quantization", "hadamard_config", "vision_config",
                "image_token_id", "tensor_namespace", "gdn_activation_layout", "schema_version"):
        assert config[key] == source_config[key], key
    assert config["components"] == {"text": True, "vision": True, "mtp": True}
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    # The head's own packing, never the trunk's 2-bit group 128.
    assert config["mtplx_mtp_quantization"]["bits"] == 4
    assert config["mtplx_mtp_quantization"]["group_size"] == 64
    assert config["mtplx_mtp_quantization"]["prequantized"] is True

    runtime = json.loads((output / "mtplx_runtime.json").read_text())
    assert runtime["arch_id"] == "qwen3-next-mtp"
    assert runtime["sampler"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert runtime["capabilities"] == {"vision": True, "mtp": True}
    assert runtime["mtp_contract"]["mtp_quant_bits"] == 4
    assert runtime["base_trunk"] == "Qwen/Qwen3.8-27B"

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert any(key.startswith("vision_tower.") for key in index["weight_map"])
    assert set(index["weight_map"].values()) == {"model.safetensors"}
    for name in ("LICENSE", "NOTICE.txt", "hadamard.json", "preprocessor_config.json",
                 "tokenizer.json", "README.md", "MTPLX_PACK_MANIFEST.json",
                 "processor_config.json"):
        assert (output / name).is_file(), name
    assert (output / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    card = (output / "README.md").read_text()
    assert "Prism ML" in card and "vision-language model" in card
    assert "library_name: mtplx" in card
    assert "—" not in card  # no em dashes in public text


@pytest.mark.parametrize("generation_mode", ["mtp", "ar"])
def test_built_pack_inspects_as_mtp_and_vision_and_loads_with_the_head(tmp_path, generation_mode):
    from mtplx import runtime
    from mtplx.artifacts import inspect_model
    from mtplx.vision import vision_spec_for_model_dir

    source = _source(tmp_path)
    head = _head_pack(tmp_path, quantized=True)
    output = tmp_path / builder.PACK_NAME
    builder.build_pack(source, output, mtp_source=builder.resolve_mtp_source(str(head)),
                       recommended_generation_mode=generation_mode)

    inspection = inspect_model(output).to_dict()
    assert inspection["vision"]["capable"] is True
    assert inspection["mtp"]["exists"] is True
    assert inspection["mtp_num_hidden_layers"] == 1
    assert inspection["compatibility"]["arch_id"] == "qwen3-next-mtp"
    assert inspection["compatibility"]["mtp_supported"] != "no"
    assert inspection["compatibility"]["runtime_contract"]["recommended_generation_mode"] == generation_mode
    assert vision_spec_for_model_dir(output) is not None

    rt = runtime.load(output, mtp=True)
    assert rt.mtp_enabled
    ids = mx.array([[1, 5, 9, 200, 17, 33]])
    logits, hidden = rt.model(ids, return_hidden=True)
    draft = rt.model.mtp_forward(
        hidden[:, :-1, :], ids[:, 1:], mtp_cache=rt.model.make_mtp_cache()
    )
    mx.eval(logits, draft)
    assert draft.dtype == mx.float16 and bool(mx.all(mx.isfinite(draft)))
    # The head is 4-bit group 64 even though the trunk is 2-bit group 128.
    fc = rt.model.language_model.mtp.fc
    assert (int(fc.bits), int(fc.group_size)) == (4, 64)


def test_a_bfloat16_head_is_cast_to_float16_once(tmp_path):
    source = _source(tmp_path)
    head = _head_pack(tmp_path, dtype=mx.bfloat16)
    output = tmp_path / builder.PACK_NAME
    manifest = builder.build_pack(
        source, output, mtp_source=builder.resolve_mtp_source(str(head))
    )
    assert manifest["provenance"]["mtp_head"]["action"] == "cast_to_float16"
    assert builder.head_float_dtypes(output / "mtp.safetensors") == {"F16"}
    assert "mtplx_mtp_quantization" not in json.loads((output / "config.json").read_text())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"with_vision": False}, "vision"),
        ({"with_preprocessor": False}, "preprocessor_config.json"),
    ],
)
def test_a_source_without_its_vision_pieces_is_refused(tmp_path, kwargs, message):
    source = _source(tmp_path, **kwargs)
    head = _head_pack(tmp_path)
    with pytest.raises(builder.PackBuildError, match=message):
        builder.build_pack(
            source, tmp_path / "out", mtp_source=builder.resolve_mtp_source(str(head))
        )
    assert not (tmp_path / "out").exists()


def test_a_source_without_vision_config_is_refused(tmp_path):
    source = _source(tmp_path)
    config = json.loads((source / "config.json").read_text())
    config.pop("vision_config")
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(builder.PackBuildError, match="vision"):
        builder.build_pack(
            source,
            tmp_path / "out",
            mtp_source=builder.resolve_mtp_source(str(_head_pack(tmp_path))),
        )


def test_a_trunk_that_differs_from_the_published_hash_is_refused(tmp_path):
    source = _source(tmp_path)
    files = json.loads((source / "files.json").read_text())
    files["model.safetensors"]["sha256"] = "0" * 64
    (source / "files.json").write_text(json.dumps(files))
    with pytest.raises(builder.PackBuildError, match="sha256"):
        builder.build_pack(
            source,
            tmp_path / "out",
            mtp_source=builder.resolve_mtp_source(str(_head_pack(tmp_path))),
        )


def test_an_existing_output_is_never_deleted(tmp_path):
    source = _source(tmp_path)
    head = builder.resolve_mtp_source(str(_head_pack(tmp_path)))
    output = tmp_path / "models" / builder.PACK_NAME
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("mine")
    with pytest.raises(builder.PackBuildError, match="already exists"):
        builder.build_pack(source, output, mtp_source=head)
    builder.build_pack(source, output, mtp_source=head, move_existing_aside=True)
    moved = list((output.parent / "_aside").iterdir())
    assert len(moved) == 1 and (moved[0] / "keep.txt").read_text() == "mine"
    assert (output / "model.safetensors").is_file()


def test_head_packing_is_read_from_the_head_not_the_trunk(tmp_path):
    head = _head_pack(tmp_path, quantized=True)
    spec = builder.mtp_quantization_from_header(head / "mtp.safetensors")
    assert spec == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "policy": "all",
        "prequantized": True,
    }
    dense = _head_pack(tmp_path / "dense")
    assert builder.mtp_quantization_from_header(dense / "mtp.safetensors") is None


def test_stamping_measured_results_lifts_the_pending_status(tmp_path):
    from mtplx.artifacts import inspect_model

    source = _source(tmp_path)
    head = builder.resolve_mtp_source(str(_head_pack(tmp_path, quantized=True)))
    output = tmp_path / "models" / builder.PACK_NAME
    output.parent.mkdir()
    builder.build_pack(source, output, mtp_source=head)
    before = inspect_model(output).to_dict()["compatibility"]
    assert before["tier"] != "verified" and "pending_measurement" in before["message"]

    contract = builder.stamp_pack(
        output,
        exactness={"top1_agreement": 1.0, "kl_mean": 3e-6},
        exactness_status="passed",
        mtp_depth_default=3,
        speed_evidence={"status": "measured", "rows": [{
            "hardware": "synthetic test fixture", "context_tokens": 4096,
            "ar_tokens_per_second": 40.0, "mtp_tokens_per_second": 60.0,
            "accepted_tokens_per_step": 2.0,
        }]},
        verified_on={"timestamp": "2026-09-18T05:00:00-0700"},
    )
    assert contract["mtp_depth_default"] == 3
    assert "mtp_depth_default_status" not in contract
    after = inspect_model(output).to_dict()["compatibility"]
    assert after["tier"] == "verified"
    kept = list((output.parent / "_aside").iterdir())
    assert len(kept) == 1 and "pending_measurement" in kept[0].read_text()
    with pytest.raises(builder.PackBuildError):
        builder.stamp_pack(output, mtp_depth_default=9)
