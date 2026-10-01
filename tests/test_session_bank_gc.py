"""#493: manual GC for the SessionBank SSD cold tier.

394,155 orphaned blobs (44.1 GB) accumulated against 17 live manifest
entries over three weeks of normal use, with no way to reclaim them short
of deleting the whole session bank by hand -- the automatic reconciliation
in ``cache_bank/cold_tier.py`` only runs as a side effect of a write that
finds the store near its configured size cap, which a Mac with generous
free disk rarely reaches.

These tests build a session-bank directory by hand (manifest.sqlite +
entries/ + blobs/, the same on-disk shape ``cache_bank/cold_tier.py``
produces) rather than constructing a real ``SessionBankColdTier`` -- the
whole point of ``cache_bank/reconcile.py`` is that it does not import that
class or its MLX-touching codec dependency. The tier's own use of the same
walk (startup pass, cap accounting, in-flight protection) is covered in
``test_cold_tier_reconcile.py``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from mtplx.cache_bank.reconcile import collect_garbage

SCHEMA = """
CREATE TABLE entries (
    entry_id TEXT PRIMARY KEY,
    entry_dir TEXT NOT NULL
);
"""


def _make_bank(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(root / "manifest.sqlite"))
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def _add_entry(root: Path, entry_id: str, prefix: str, blob_hashes: list[str]) -> None:
    entry_dir_rel = f"entries/{prefix}/{entry_id}"
    entry_dir = root / entry_dir_rel
    entry_dir.mkdir(parents=True, exist_ok=True)
    tensor_blobs = {
        f"tensor-{i}": {"sha256": digest, "nbytes": 1}
        for i, digest in enumerate(blob_hashes)
    }
    (entry_dir / "payload.json").write_text(
        json.dumps({"tensor_blobs": tensor_blobs}), encoding="utf-8"
    )
    conn = sqlite3.connect(str(root / "manifest.sqlite"))
    conn.execute(
        "INSERT INTO entries (entry_id, entry_dir) VALUES (?, ?)",
        (entry_id, entry_dir_rel),
    )
    conn.commit()
    conn.close()


def _write_blob(root: Path, digest: str, data: bytes = b"x") -> Path:
    path = root / "blobs" / digest[:2] / f"{digest}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestDryRun:
    def test_referenced_entry_and_blob_are_not_flagged(self, tmp_path):
        root = tmp_path / "session-bank"
        _make_bank(root)
        _add_entry(root, "live1", "ab", ["deadbeef"])
        _write_blob(root, "deadbeef")

        report = collect_garbage(root, dry_run=True)
        assert report["orphan_entry_dirs"] == []
        assert report["orphan_blob_files"] == 0
        assert report["deleted"] is False

    def test_orphan_entry_dir_is_reported(self, tmp_path):
        root = tmp_path / "session-bank"
        _make_bank(root)
        orphan_dir = root / "entries" / "cd" / "gone"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "payload.json").write_text("{}", encoding="utf-8")

        report = collect_garbage(root, dry_run=True)
        assert report["orphan_entry_dirs"] == ["entries/cd/gone"]
        assert orphan_dir.exists(), "dry run must not delete anything"

    def test_orphan_blob_is_reported_with_size(self, tmp_path):
        root = tmp_path / "session-bank"
        _make_bank(root)
        path = _write_blob(root, "abcabcabc", data=b"0123456789")

        report = collect_garbage(root, dry_run=True)
        assert report["orphan_blob_files"] == 1
        assert report["orphan_blob_bytes"] == 10
        assert path.exists()

    def test_evicted_entries_reported_but_not_deleted(self, tmp_path):
        root = tmp_path / "session-bank"
        _make_bank(root)
        stale = root / "evicted_entries" / "ab" / "stale-entry"
        stale.mkdir(parents=True)
        (stale / "payload.json").write_text("x" * 100, encoding="utf-8")

        report = collect_garbage(root, dry_run=True)
        assert report["evicted_entries_bytes"] == 100
        assert stale.exists()

    def test_missing_bank_directory_is_inert(self, tmp_path):
        report = collect_garbage(tmp_path / "does-not-exist", dry_run=True)
        assert report["orphan_entry_dirs"] == []
        assert report["orphan_blob_files"] == 0

    def test_missing_manifest_treats_every_entry_dir_as_orphan(self, tmp_path):
        root = tmp_path / "session-bank"
        root.mkdir()
        stray = root / "entries" / "ab" / "stray"
        stray.mkdir(parents=True)

        report = collect_garbage(root, dry_run=True)
        assert report["orphan_entry_dirs"] == ["entries/ab/stray"]


class TestApply:
    def test_apply_deletes_orphans_and_keeps_referenced(self, tmp_path):
        root = tmp_path / "session-bank"
        _make_bank(root)
        _add_entry(root, "live1", "ab", ["deadbeef"])
        live_blob = _write_blob(root, "deadbeef")
        orphan_blob = _write_blob(root, "0000000000")
        orphan_entry = root / "entries" / "cd" / "gone"
        orphan_entry.mkdir(parents=True)
        (orphan_entry / "payload.json").write_text("{}", encoding="utf-8")
        evicted = root / "evicted_entries" / "ab" / "stale"
        evicted.mkdir(parents=True)
        (evicted / "payload.json").write_text("x", encoding="utf-8")

        report = collect_garbage(root, dry_run=False)

        assert report["deleted"] is True
        assert live_blob.exists(), "a referenced blob must survive --apply"
        assert (root / "entries" / "ab" / "live1" / "payload.json").exists()
        assert not orphan_blob.exists()
        assert not orphan_entry.exists()
        assert not evicted.exists()

    def test_apply_on_a_clean_bank_deletes_nothing(self, tmp_path):
        root = tmp_path / "session-bank"
        _make_bank(root)
        _add_entry(root, "live1", "ab", ["deadbeef"])
        live_blob = _write_blob(root, "deadbeef")

        report = collect_garbage(root, dry_run=False)

        assert report["orphan_entry_dirs"] == []
        assert report["orphan_blob_files"] == 0
        assert live_blob.exists()


class TestCliHandler:
    def _args(self, tmp_path, **overrides):
        base = {
            "dir": str(tmp_path / "session-bank"),
            "apply": False,
            "force": False,
            "host": "127.0.0.1",
            "json": True,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_dry_run_reports_without_deleting(self, tmp_path, capsys):
        from mtplx.commands.public import cmd_gc_public

        root = tmp_path / "session-bank"
        _make_bank(root)
        orphan = root / "entries" / "ab" / "gone"
        orphan.mkdir(parents=True)
        (orphan / "payload.json").write_text("{}", encoding="utf-8")

        rc = cmd_gc_public(self._args(tmp_path))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["orphan_entry_dirs"] == ["entries/ab/gone"]
        assert payload["deleted"] is False
        assert orphan.exists()

    def test_apply_blocked_when_server_running_without_force(
        self, tmp_path, capsys, monkeypatch
    ):
        from mtplx import daemon_client
        from mtplx.commands.public import cmd_gc_public

        root = tmp_path / "session-bank"
        _make_bank(root)

        monkeypatch.setattr(
            daemon_client,
            "probe_running_daemons",
            lambda **kwargs: [SimpleNamespace(port=7999)],
        )

        rc = cmd_gc_public(self._args(tmp_path, apply=True))
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["reason"] == "server_running"

    def test_apply_with_force_ignores_running_server(
        self, tmp_path, capsys, monkeypatch
    ):
        from mtplx import daemon_client
        from mtplx.commands.public import cmd_gc_public

        root = tmp_path / "session-bank"
        _make_bank(root)
        orphan = root / "entries" / "ab" / "gone"
        orphan.mkdir(parents=True)
        (orphan / "payload.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            daemon_client,
            "probe_running_daemons",
            lambda **kwargs: [SimpleNamespace(port=7999)],
        )

        rc = cmd_gc_public(self._args(tmp_path, apply=True, force=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["deleted"] is True
        assert not orphan.exists()
