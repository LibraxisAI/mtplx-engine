"""Release orchestration checks without a download, GPU or live server."""
import json
import shutil

import pytest

from scripts import build_flash_next_quality_pack as build
from mtplx.commands.forge_qwen4_audit import audit_pack, quality_metadata
from mtplx.commands.forge_qwen4_exp import named_recipe, QUALITY_NAME, QUALITY_RECIPE
from test_forge_qwen4_exp_lane import _audit_fixture


def test_card_uses_artifact_precision_and_verification(tmp_path):
    source, pack = _audit_fixture(tmp_path)
    audit = audit_pack(source, pack, named_recipe(QUALITY_RECIPE))
    metadata = quality_metadata({"revision": "test-revision", "revision_kind": "hub-commit"}, audit)
    (pack / "mtplx_runtime.json").write_text(json.dumps({"quality_pack": metadata,
        "verification": {"mode": "streaming", "status": "streaming-audited"}}))
    card = build.generate_card(pack)
    assert "test-revision" in card and "streaming-audited" not in card
    assert "Q4/g32" in card and "Q8/g64" in card and "BF16" in card
    assert "qwen-community-1.0" in card and "Speed on 256 GB and 512 GB Macs is not measured yet." in card
    assert "128 GB" in card and "Cannot load" in card


def test_dry_run_has_no_side_effects_or_preflight_imports(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run performed work")
    monkeypatch.setattr(build, "preflight", forbidden)
    monkeypatch.setattr(build, "run_step", forbidden)
    assert build.main(["Qwen/Qwen3.8-Flash-Next", str(tmp_path / "quality"), "--upload", "--dry-run"]) == 0
    assert not list(tmp_path.iterdir())
    plan = json.loads(capsys.readouterr().out)
    assert plan["disk"]["additional_bytes"] == 570_000_000_000
    assert len(plan["disk"]["filesystems"]) == 1


@pytest.mark.parametrize("upload,fail_smoke", [(False, False), (True, False), (True, True)])
def test_pipeline_only_uploads_after_full_verify_and_smoke(tmp_path, monkeypatch, upload, fail_smoke):
    source, fixture = _audit_fixture(tmp_path)
    (source / "LICENSE").write_text("fixture upstream license")
    (source / "README.md").write_text("fixture upstream card")
    audit = audit_pack(source, fixture, named_recipe(QUALITY_RECIPE))
    meta = quality_metadata({"revision": "test-sha", "revision_kind": "hub-commit"}, audit)
    output = tmp_path / QUALITY_NAME
    steps = []
    monkeypatch.setattr(build, "preflight", lambda plan: {"test": True})
    def step(run, label, command, **kwargs):
        steps.append(label)
        if label == "forge-build":
            assert command[command.index("--verification") + 1] == "full-load"
            assert command[command.index("--recipe") + 1] == QUALITY_RECIPE
            shutil.copytree(fixture, output)
            build.write_json(output / "mtplx_runtime.json", {"quality_pack": meta,
                "verification": {"mode": "full-load", "status": "verified", "full_load_verified": True}})
    def smoke(*args, **kwargs):
        steps.append("smoke")
        if fail_smoke:
            raise RuntimeError("deliberate smoke failure")
        return {"passed": True, "peak_listener_rss_bytes": 1000}
    monkeypatch.setattr(build, "run_step", step)
    monkeypatch.setattr(build, "serve_smoke", smoke)
    argv = [str(source), str(output)] + (["--upload"] if upload else [])
    if fail_smoke:
        with pytest.raises(RuntimeError, match="deliberate smoke failure"):
            build.main(argv)
        assert steps == ["forge-build", "smoke"]
        assert not (output / "size-checksums.json").exists()
    else:
        assert build.main(argv) == 0
        assert steps == ["forge-build", "smoke"] + (["login-check", "upload"] if upload else [])
        manifest = json.loads((output / "size-checksums.json").read_text())
        assert manifest["files"]["README.md"]["sha256"] == build.checksum(output / "README.md")
        assert (output / "LICENSE").read_text() == "fixture upstream license"
