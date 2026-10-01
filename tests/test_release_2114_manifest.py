from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mtplx.model_catalog import OFFICIAL_CATALOG
from mtplx import model_updates
from scripts import audit_catalog_sizes as audit
from scripts import gen_models_manifest as manifest
from scripts.model_release_policy import NOT_YET_PUBLISHED_REPOS


class Hub:
    def __init__(self, failing=()):
        self.failing = set(failing)

    def model_info(self, repo_id, **kwargs):
        if repo_id in self.failing:
            raise RuntimeError("repository unavailable")
        size = next(m.size_bytes for m in OFFICIAL_CATALOG if m.hf_model_id == repo_id)
        return SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(size=size)])


def run_script(module, monkeypatch, hub, args):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: hub)
    monkeypatch.setattr("sys.argv", [module.__file__, *args])
    module.main()


def test_manifest_requires_2120_for_new_packs(tmp_path, monkeypatch):
    out = tmp_path / "models.json"
    run_script(manifest, monkeypatch, Hub(), ["--out", str(out)])
    data = json.loads(out.read_text())
    assert len(data["models"]) == len(manifest.BLESSED)
    for repo in NOT_YET_PUBLISHED_REPOS:
        entry = data["models"][repo]
        assert entry["min_engine_version"] == "2.12.0"
        monkeypatch.setattr(model_updates, "ENGINE_VERSION", "2.11.4")
        assert not model_updates.engine_satisfies(entry["min_engine_version"])
        monkeypatch.setattr(model_updates, "ENGINE_VERSION", "2.12.0")
        assert model_updates.engine_satisfies(entry["min_engine_version"])
        assert entry["revision"] == "a" * 40
        assert entry["note"] == next(m.detail for m in OFFICIAL_CATALOG if m.hf_model_id == repo)


@pytest.mark.parametrize("repo", [sorted(NOT_YET_PUBLISHED_REPOS)[0], "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed"])
def test_unresolved_repo_is_an_error_and_manifest_is_not_overwritten(repo, tmp_path, monkeypatch):
    out = tmp_path / "models.json"
    out.write_text("previous complete manifest")
    with pytest.raises(SystemExit):
        run_script(manifest, monkeypatch, Hub([repo]), ["--out", str(out)])
    assert out.read_text() == "previous complete manifest"
    with pytest.raises(SystemExit):
        run_script(audit, monkeypatch, Hub([repo]), [])


def test_explicit_not_yet_published_markers_are_recorded(tmp_path, monkeypatch, capsys):
    out = tmp_path / "models.json"
    flags = [arg for repo in sorted(NOT_YET_PUBLISHED_REPOS) for arg in ["--not-yet-published", repo]]
    hub = Hub(NOT_YET_PUBLISHED_REPOS)
    run_script(manifest, monkeypatch, hub, ["--out", str(out), *flags])
    data = json.loads(out.read_text())
    assert data["not_yet_published"] == sorted(NOT_YET_PUBLISHED_REPOS)
    assert len(data["models"]) == len(manifest.BLESSED) - 2
    assert not (set(data["models"]) & NOT_YET_PUBLISHED_REPOS)
    run_script(audit, monkeypatch, hub, flags)
    assert "2 explicitly not yet published" in capsys.readouterr().out
    # A flag must never suppress a resolved pack or a bad size pin.
    run_script(manifest, monkeypatch, Hub(), ["--out", str(out), *flags])
    assert not json.loads(out.read_text())["not_yet_published"]


@pytest.mark.parametrize("module", [manifest, audit])
def test_published_repositories_cannot_use_the_marker(module, tmp_path, monkeypatch):
    flags = ["--out", str(tmp_path / "models.json")] if module is manifest else []
    with pytest.raises(SystemExit) as exc:
        run_script(module, monkeypatch, Hub(), [*flags, "--not-yet-published", "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed"])
    assert exc.value.code == 2


def test_audit_rejects_incomplete_metadata_and_wrong_totals(monkeypatch):
    for size in (None, 1):
        class BadHub:
            def model_info(self, **kwargs):
                return SimpleNamespace(siblings=[SimpleNamespace(size=size)])
        with pytest.raises(SystemExit):
            run_script(audit, monkeypatch, BadHub(), [])
