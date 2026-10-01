"""Header-only builder tests, runnable even when importing MLX is unavailable."""

import hashlib
import json
from pathlib import Path
import struct

import pytest

from scripts import build_bonsai_mtplx_pack as builder
from scripts import bonsai_memory_table as memory


def write_tensor_file(path: Path, names: list[str]) -> None:
    # Valid tiny float16 safetensors, without importing a tensor runtime.
    header = {name: {"dtype": "F16", "shape": [1], "data_offsets": [i * 2, (i + 1) * 2]}
              for i, name in enumerate(names)}
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00\x00" * len(names))


@pytest.fixture
def pack(tmp_path):
    source, head, output = tmp_path / "prism", tmp_path / "head", tmp_path / "old-name"
    source.mkdir()
    head.mkdir()
    config = {
        "model_type": builder.MODEL_TYPE,
        "modules": [{"path": "model.embed_tokens", "block": 1024, "embedding": True, "dtype": "float16"}],
        "components": {"text": True, "vision": True, "mtp": False},
        "vision_config": {"hidden_size": 1},
        "text_config": {"num_hidden_layers": 64, "full_attention_interval": 4,
                        "num_key_value_heads": 4, "head_dim": 256, "max_position_embeddings": 262144},
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "custom_prism_metadata": {"must_survive": True},
    }
    (source / "config.json").write_text(json.dumps(config))
    for name in ("hadamard.json", "preprocessor_config.json", "tokenizer.json"):
        (source / name).write_text("{}")
    (source / "LICENSE").write_bytes(b"Apache License, Version 2.0\r\nTest license bytes\n")
    (source / "NOTICE.txt").write_bytes(b"Created using Bonsai by Prism ML.\r\nTest notice bytes\n")
    write_tensor_file(source / "model.safetensors", ["language_model.model.norm.weight", "vision_tower.norm.weight"])
    write_tensor_file(head / "mtp.safetensors", ["mtp.norm.weight"])
    builder.build_pack(source, output, mtp_source=head / "mtp.safetensors")
    return output


def digests(pack):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pack.iterdir() if p.is_file()}


def speed_evidence():
    return {"status": "measured", "measured_at": "2026-09-21T17:09-07:00", "rows": [{
        "hardware": "unit test fixture", "context_tokens": 4096,
        "ar_tokens_per_second": 30.0, "mtp_tokens_per_second": 31.0,
        "accepted_tokens_per_step": 1.2,
    }]}


def test_stamps_use_a_floor_independent_of_builder_version(pack):
    stamps = builder.identity_stamps()
    assert stamps["pack_name"] == "Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
    assert stamps["hf_repo"] == "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"
    assert stamps["public_model_id"] == "mtplx-bonsai-2-27b-optimized-speed"
    assert stamps["model_family"] == "qwen3_8"
    assert stamps["min_engine_version"] == "2.12.0"
    runtime = builder.build_runtime_contract(mtplx_version="2.11.3", provenance={}, head=None)
    assert runtime["mtplx_version"] == "2.11.3"
    assert runtime["min_engine_version"] == "2.12.0"
    assert runtime["quantization"]["format"] == "prism_hadamard_qwen35"
    for name in ("mtplx_runtime.json", "MTPLX_PACK_MANIFEST.json"):
        written = json.loads((pack / name).read_text())
        assert all(written[k] == v for k, v in stamps.items())
    config = json.loads((pack / "config.json").read_text())
    assert config["model_type"] == "prism_hadamard_qwen35"  # loader dispatch must not change
    assert config["model_family"] == "qwen3_8"
    assert config["quantization"] == {"bits": 2, "group_size": 128, "mode": "affine"}


@pytest.mark.parametrize("mode", ["hardlink", "copy"])
def test_restamp_preserves_all_source_bytes_and_parity_metadata(pack, tmp_path, mode):
    runtime = json.loads((pack / "mtplx_runtime.json").read_text())
    runtime.update(public_model_id="mtplx-bonsai-38-27b-optimized-speed", mtplx_version="2.11.3")
    runtime["exactness_baseline"] = {"status": "passed", "measured": {"kl": 3e-6}}
    runtime["custom_metadata"] = {"keep": True}
    (pack / "mtplx_runtime.json").write_text(json.dumps(runtime))
    before = digests(pack)
    output = tmp_path / f"renamed-{mode}"
    manifest = builder.restamp_pack(pack, output, link_mode=mode)
    assert digests(pack) == before
    for name in ("model.safetensors", "mtp.safetensors"):
        assert (output / name).read_bytes() == (pack / name).read_bytes()
        assert ((output / name).stat().st_ino == (pack / name).stat().st_ino) is (mode == "hardlink")
    for name in ("config.json", "mtplx_runtime.json", "LICENSE", "NOTICE.txt"):
        assert (output / name).stat().st_ino != (pack / name).stat().st_ino
    for name in ("LICENSE", "NOTICE.txt", "hadamard.json", "tokenizer.json"):
        assert (output / name).read_bytes() == (pack / name).read_bytes()
    new_runtime = json.loads((output / "mtplx_runtime.json").read_text())
    assert new_runtime["mtplx_version"] == "2.11.3"
    assert new_runtime["min_engine_version"] == "2.12.0"
    assert new_runtime["exactness_baseline"] == runtime["exactness_baseline"]
    assert new_runtime["custom_metadata"] == runtime["custom_metadata"]
    old_config = json.loads((pack / "config.json").read_text())
    assert json.loads((output / "config.json").read_text()) == old_config
    for name, entry in manifest["files"].items():
        assert builder.sha256_file(output / name) == entry["sha256"]
        assert (output / name).stat().st_size == entry["size"]


def test_restamp_refuses_in_place_nested_or_existing_destinations(pack, tmp_path):
    before = digests(pack)
    alias = tmp_path / "alias"
    alias.symlink_to(pack, target_is_directory=True)
    for destination in (pack, alias, pack / "nested", pack.parent, tmp_path / "head"):
        with pytest.raises(builder.PackBuildError):
            builder.restamp_pack(pack, destination)
    assert digests(pack) == before


def test_default_card_has_credit_historical_parity_and_no_unmeasured_ram_or_speed_claim(pack):
    card = (pack / "README.md").read_text()
    assert "Full credit for the model goes to [Prism ML]" in card
    assert "verbatim" in card and "[LICENSE](LICENSE)" in card and "[NOTICE.txt](NOTICE.txt)" in card
    assert "follows the model's own distribution" in card and "2.6e-6" not in card
    assert "No speed measurements supplied" in card
    assert "RAM recommendations are pending" in card
    assert "32K tokens or lower" not in card and "16 to 24 GB" not in card
    assert "2.12.0 or newer" in card
    assert "—" not in card and "–" not in card
    runtime = json.loads((pack / "mtplx_runtime.json").read_text())
    assert runtime["exactness_baseline"]["status"] == "pending_measurement"


def test_measurement_table_and_explicit_speed_json_fill_new_card(pack, tmp_path):
    report = memory.make_report(memory.pack_metadata(pack), [16, 18, 24], [4096])
    report["rows"][0] = memory.summarize_case(report["rows"][0], {
        "status": "completed", "completed": True, "request_peak_memory_bytes": 10 * memory.GIB,
    }, 9 * memory.GIB)
    output = tmp_path / "with-evidence"
    builder.restamp_pack(pack, output, memory_evidence=report, speed_evidence=speed_evidence())
    card = (output / "README.md").read_text()
    assert "| 16 GB |" in card and "| 10.0 GiB |" in card
    assert "unit test fixture | 4096 | 30.0 | 31.0 | 1.2" in card
    assert "No speed measurements supplied" not in card
    assert "No speed measurements supplied" in (pack / "README.md").read_text()


@pytest.mark.parametrize("evidence", [{}, {"status": "measured", "rows": []}, {"tok_s": [60]}])
def test_unmeasured_or_unstructured_speed_is_refused(pack, tmp_path, evidence):
    with pytest.raises(builder.PackBuildError, match="speed JSON"):
        builder.restamp_pack(pack, tmp_path / "invalid", speed_evidence=evidence)
    assert not (tmp_path / "invalid").exists()


def test_memory_evidence_for_different_weights_is_refused(pack, tmp_path):
    report = memory.make_report(memory.pack_metadata(pack), [16], [4096])
    report["pack"]["weight_files_bytes"]["mtp.safetensors"] += 1
    with pytest.raises(builder.PackBuildError, match="do not match"):
        builder.restamp_pack(pack, tmp_path / "wrong-memory", memory_evidence=report)


def test_reading_evidence_accepts_pretty_json_and_last_json_log_line(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(speed_evidence(), indent=2))
    assert builder._read_json_arg(str(path)) == speed_evidence()
    path.write_text("measurement log\n" + json.dumps(speed_evidence()) + "\n")
    assert builder._read_json_arg(str(path)) == speed_evidence()
    path.write_text("")
    with pytest.raises(builder.PackBuildError, match="empty"):
        builder._read_json_arg(str(path))


def test_stamp_refreshes_the_card_and_manifest_without_changing_weights(pack):
    before = digests(pack)
    contract = builder.stamp_pack(pack, speed_evidence=speed_evidence())
    assert contract["exactness_baseline"]["status"] == "pending_measurement"
    after = digests(pack)
    assert after["model.safetensors"] == before["model.safetensors"]
    assert after["mtp.safetensors"] == before["mtp.safetensors"]
    assert "unit test fixture" in (pack / "README.md").read_text()
    manifest = json.loads((pack / "MTPLX_PACK_MANIFEST.json").read_text())
    for name in ("mtplx_runtime.json", "README.md"):
        assert manifest["files"][name]["sha256"] == after[name]
    saved = list((pack.parent / "_aside").iterdir())
    assert len(saved) == 1
    assert builder.sha256_file(saved[0]) == before["mtplx_runtime.json"]


def test_restamp_cli_requires_out_and_never_resolves_a_new_mtp_source(pack, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("restamp must not read or rebuild an external MTP source")
    monkeypatch.setattr(builder, "resolve_mtp_source", unexpected)
    with pytest.raises(SystemExit):
        builder.main(["--restamp", str(pack)])
    output = tmp_path / builder.PACK_NAME
    assert builder.main(["--restamp", str(pack), "--out", str(output)]) == 0
    assert (output / "model.safetensors").stat().st_ino == (pack / "model.safetensors").stat().st_ino


@pytest.mark.parametrize("operation", ["build", "stamp", "restamp"])
@pytest.mark.parametrize("mode", ["ar", "mtp"])
def test_generation_recommendation_cli_writes_contract_card_and_manifest(pack, tmp_path, operation, mode):
    evidence = speed_evidence()
    evidence_path = tmp_path / "speed.json"
    evidence_path.write_text(json.dumps(evidence))
    runtime_path = pack / "mtplx_runtime.json"
    original = json.loads(runtime_path.read_text())
    original["mtp_depth_default"] = 1
    runtime_path.write_text(json.dumps(original))
    before = digests(pack)
    output = pack if operation == "stamp" else tmp_path / "recommended-pack"
    if operation == "build":
        argv = ["--source", str(tmp_path / "prism"), "--out", str(output),
                "--mtp-source", str(tmp_path / "head")]
    elif operation == "stamp":
        argv = ["--stamp", str(pack), "--mtp-depth-default", "1"]
    else:
        argv = ["--restamp", str(pack), "--out", str(output)]
    reason = "The measured native-sampler results determine this default."
    assert builder.main([
        *argv, "--recommended-generation-mode", mode,
        "--recommended-generation-mode-reason", reason,
        "--speed-evidence-json", str(evidence_path),
    ]) == 0
    contract = json.loads((output / "mtplx_runtime.json").read_text())
    assert contract["recommended_generation_mode"] == mode
    assert contract["recommended_generation_mode_reason"] == reason
    assert contract["recommended_generation_mode_evidence"] == {
        "measured_at": evidence["measured_at"], "rows": evidence["rows"],
    }
    assert contract["mtp_depth_default"] == (2 if operation == "build" else 1)
    assert contract["capabilities"]["mtp"] is True
    after = digests(output)
    assert after["model.safetensors"] == before["model.safetensors"]
    assert after["mtp.safetensors"] == before["mtp.safetensors"]
    card = (output / "README.md").read_text()
    if mode == "ar":
        assert "off by default" in card and reason in card
        assert "--generation-mode mtp" in card
    else:
        assert "on by default" in card and "off by default" not in card
    assert "unit test fixture | 4096 | 30.0 | 31.0 | 1.2" in card
    manifest = json.loads((output / "MTPLX_PACK_MANIFEST.json").read_text())
    for name in ("README.md", "mtplx_runtime.json"):
        assert manifest["files"][name]["sha256"] == after[name]
    if operation != "stamp":
        assert digests(pack) == before


@pytest.mark.parametrize("operation", ["stamp", "restamp"])
def test_generation_only_update_preserves_recorded_speed_memory_and_depth(pack, tmp_path, operation):
    evidence = speed_evidence()
    report = memory.make_report(memory.pack_metadata(pack), [16], [4096])
    report["rows"][0] = memory.summarize_case(report["rows"][0], {
        "status": "completed", "completed": True, "request_peak_memory_bytes": 10 * memory.GIB,
    }, 9 * memory.GIB)
    path = pack / "mtplx_runtime.json"
    contract = json.loads(path.read_text())
    contract.update(mtp_depth_default=1, speed_evidence=evidence, memory_evidence=report)
    path.write_text(json.dumps(contract))
    reason = "No consistent speed-up measured."
    options = {"recommended_generation_mode": "ar", "recommended_generation_mode_reason": reason}
    if operation == "stamp":
        builder.stamp_pack(pack, **options)
        output = pack
    else:
        output = tmp_path / "generation-only"
        builder.restamp_pack(pack, output, **options)
    updated = json.loads((output / "mtplx_runtime.json").read_text())
    assert updated["mtp_depth_default"] == 1
    assert updated["memory_evidence"] == report
    assert updated["speed_evidence"] == evidence
    assert updated["recommended_generation_mode_evidence"]["rows"] == evidence["rows"]
    card = (output / "README.md").read_text()
    assert reason in card and "unit test fixture" in card and "| 10.0 GiB |" in card
    # Restamping without new options retains both the recommendation and its receipts.
    preserved = tmp_path / "preserved"
    builder.restamp_pack(output, preserved)
    assert (preserved / "README.md").read_text() == card
    final = json.loads((preserved / "mtplx_runtime.json").read_text())
    for key in ("recommended_generation_mode", "recommended_generation_mode_reason",
                "recommended_generation_mode_evidence", "mtp_depth_default"):
        assert final[key] == updated[key]


def test_generation_recommendation_without_measurements_does_not_invent_evidence(pack):
    contract = builder.stamp_pack(pack, recommended_generation_mode="ar", mtp_depth_default=1)
    assert "recommended_generation_mode_evidence" not in contract
    card = (pack / "README.md").read_text()
    assert "off by default" in card
    assert "No measured reason was supplied" in card
    assert "No speed measurements supplied" in card


def test_switching_recommendation_drops_the_previous_modes_reason(pack, tmp_path):
    builder.stamp_pack(pack, recommended_generation_mode="ar",
                       recommended_generation_mode_reason="Old AR rationale.")
    output = tmp_path / "mtp-default"
    builder.restamp_pack(pack, output, recommended_generation_mode="mtp")
    contract = json.loads((output / "mtplx_runtime.json").read_text())
    assert "recommended_generation_mode_reason" not in contract
    assert "Old AR rationale" not in (output / "README.md").read_text()


def test_invalid_generation_mode_refused_before_metadata_changes(pack, tmp_path):
    before = digests(pack)
    with pytest.raises(builder.PackBuildError, match="generation mode"):
        builder.stamp_pack(pack, recommended_generation_mode="auto")
    with pytest.raises(builder.PackBuildError, match="generation mode"):
        builder.restamp_pack(pack, tmp_path / "invalid-mode", recommended_generation_mode="auto")
    assert digests(pack) == before
    assert not (tmp_path / "invalid-mode").exists()


def test_memory_guidance_reads_the_measured_rows_and_never_extrapolates():
    from scripts.build_bonsai_mtplx_pack import MEMORY_GUIDANCE_PENDING, render_memory_guidance

    gib = 1024 ** 3
    assert render_memory_guidance(None) == MEMORY_GUIDANCE_PENDING
    assert render_memory_guidance({"rows": []}) == MEMORY_GUIDANCE_PENDING

    def row(ram, prompt, decode, quant, peak_gib, verdict="admit", fit=8192, tight=True, status="completed"):
        return {
            "ram_gib": ram, "prompt_tokens": prompt, "decode_tokens": decode, "kv_quantization": quant,
            "engine_budget_bytes": 12 * gib, "planner_verdict": verdict,
            "planner": {"context_window_fit": fit, "tight_machine": tight},
            "status": status, "peak_memory_bytes": int(peak_gib * gib) if status == "completed" else None,
            "allocation_failure": False,
        }

    rows = [
        row(16, 4096, 0, "off", 11.55), row(16, 4096, 1024, "off", 11.55),
        row(16, 8192, 0, "off", 11.78), row(16, 8192, 1024, "q8", 11.80),
        row(16, 16384, 0, "off", 12.11), row(16, 16384, 1024, "q8", 12.11),
        row(16, 4096, 0, "q8", 11.55, status="not_run"),
    ]
    text = render_memory_guidance({"rows": rows})
    # The 16K runs peaked over the 12 GiB budget, so they are not the class peak.
    assert "| 16 GB | 8,192 tokens | 11.8 GiB |" in text and "12.1" not in text
    assert "OpenCode" not in text
    refused = [row(16, 4096, 0, "off", 11.55, verdict="refuse", fit=4096, tight=False)]
    assert "| 16 GB | Does not fit |" in render_memory_guidance({"rows": refused})
    larger = rows + [row(18, 4096, 0, "off", 11.62, fit=20480), row(18, 4096, 0, "q8", 11.62, fit=36864)]
    text = render_memory_guidance({"rows": larger})
    assert "| 18 GB | 20,480 tokens (36,864 with 8-bit KV cache) | 11.6 GiB |" in text
    assert "On a 16 GB Mac the window is too small" in text and "use 18 GB or more for those." in text


def test_card_carries_the_measured_reason_for_an_mtp_default_too():
    from scripts.build_bonsai_mtplx_pack import render_card

    card = render_card(source_sha="abc", head_note="head", recommended_generation_mode="mtp",
                       recommended_generation_mode_reason="Depth 1 measured 1.3x through the daemon.")
    assert "The draft head is on by default (MTP speculative decoding)." in card
    assert "Depth 1 measured 1.3x through the daemon." in card
