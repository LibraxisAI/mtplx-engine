"""Manifest-vs-disk reconciliation for the SessionBank SSD cold tier (#493).

The store is content-addressed: ``entries/<xx>/<entry_id>/payload.json``
names the ``blobs/<xx>/<sha256>.bin`` files that make up one snapshot, and
``manifest.sqlite`` names the entries that restores may use. Anything on
disk that the manifest does not reach is garbage: an entry directory
without a manifest row (a crash between the directory rename and the row
insert, or a row the tier evicted after the directory rename raced it), a
blob no live entry names (a snapshot whose spill was interrupted before its
row landed, a crash mid-write, a blob whose last referencing entry was
evicted while the shared-blob sweep missed it), everything under
``evicted_entries/`` (the archive the tier keeps for an entry directory it
found untracked at write time), and a dot-prefixed temp file or directory
that outlived the write that created it.

This module is the ONE implementation of that walk. ``SessionBankColdTier``
runs it in dry-run mode as its disk-usage census, in delete mode as its
orphan cleanup (at daemon start, and whenever the store prices over the SSD
cap), and ``mtplx gc`` runs it from the command line. It is pure SQLite +
filesystem so the CLI keeps working on a machine whose MLX install is
missing or broken (``cold_tier`` imports the tensor codec, which imports
``mlx.core`` at module scope); ``tests/test_no_mlx_imports.py`` pins that.

Concurrency contract for an in-process caller (the tier): the walk itself
takes no lock, so a 400k-file bank never stalls the writer or the owner
thread; deletions happen per directory batch under ``lock``, and before a
batch is deleted its candidates are re-checked against a fresh manifest
read (when ``manifest_generation`` says the store changed) and against
``protected`` (the entry directory and blob digests of writes in flight,
whose blobs are on disk before their manifest row is). A file this pass
never saw cannot be deleted by it. Out of process (``mtplx gc``) there is
no such coordination, which is why ``--apply`` refuses to run beside a
live server unless forced.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_COLD_TIER_DIR = Path("~/.mtplx/session-bank").expanduser()
MANIFEST_FILENAME = "manifest.sqlite"
# The walk yields to the caller's foreground hook every this many files
# (same cadence as the tier's blob writes).
YIELD_EVERY_FILES = 4096
# A dot-prefixed temp file or directory (``.<digest>.bin.tmp-<ns>``,
# ``.<entry_id>.tmp-<rand>``) belongs to a write in progress; one older than
# this is a crash leftover. A blob write is seconds and the writer pauses
# between blobs, never inside one, so an hour is far outside any live write.
TEMP_GRACE_S = 3600.0

ProtectedSets = tuple[set[str], set[str]]


@dataclass
class ReconcileReport:
    """One pass over the store: what is there, what is garbage, what went.

    Census totals (``file_bytes`` ... ``dir_count``) describe the store as
    the pass left it: after its deletions in delete mode, untouched in a dry
    run. Orphan figures are what the pass found reclaimable, at their
    pre-deletion sizes, so a dry run and the delete run that follows it
    report the same numbers. ``protected_skipped`` counts candidates kept
    because a write in flight owned them.
    """

    base_dir: str
    dry_run: bool
    manifest_found: bool = False
    entries: int = 0
    file_bytes: int = 0
    disk_bytes: int = 0
    database_file_bytes: int = 0
    database_disk_bytes: int = 0
    file_count: int = 0
    dir_count: int = 0
    orphan_entry_dirs: list[str] = field(default_factory=list)
    orphan_blob_files: int = 0
    orphan_blob_bytes: int = 0
    orphan_file_bytes: int = 0
    orphan_disk_bytes: int = 0
    orphan_file_count: int = 0
    evicted_entries_bytes: int = 0
    stale_temp_files: int = 0
    files_deleted: int = 0
    dirs_deleted: int = 0
    file_bytes_deleted: int = 0
    disk_bytes_deleted: int = 0
    protected_skipped: int = 0
    elapsed_s: float = 0.0
    interrupted: bool = False

    @property
    def deleted(self) -> bool:
        return not self.dry_run

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["deleted"] = self.deleted
        return payload


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def entry_blob_hashes(entry_dir: Path) -> set[str]:
    """The blob digests one entry's ``payload.json`` names (empty if unreadable)."""
    payload_path = entry_dir / "payload.json"
    if not payload_path.exists():
        return set()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    tensor_blobs = payload.get("tensor_blobs") or {}
    hashes: set[str] = set()
    if isinstance(tensor_blobs, dict):
        for blob in tensor_blobs.values():
            if isinstance(blob, dict) and blob.get("sha256"):
                hashes.add(str(blob["sha256"]))
    return hashes


def manifest_entry_dirs(base_dir: Path) -> set[str]:
    """Entry directories (relative to ``base_dir``) the manifest names."""
    db_path = base_dir / MANIFEST_FILENAME
    if not db_path.exists():
        return set()
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT entry_dir FROM entries").fetchall()
    return {str(row["entry_dir"]) for row in rows}


def manifest_blob_hashes(
    base_dir: Path, *, on_yield: Callable[[], None] | None = None
) -> set[str]:
    """Blob digests reachable from every manifest entry's ``payload.json``."""
    hashes: set[str] = set()
    for rel in manifest_entry_dirs(base_dir):
        if on_yield is not None:
            on_yield()
        hashes.update(entry_blob_hashes(base_dir / rel))
    return hashes


def manifest_sets(base_dir: Path) -> tuple[set[str], set[str]]:
    """``(entry_dirs, blob_hashes)`` in one manifest read."""
    entry_dirs = manifest_entry_dirs(base_dir)
    hashes: set[str] = set()
    for rel in entry_dirs:
        hashes.update(entry_blob_hashes(base_dir / rel))
    return entry_dirs, hashes


def allocated_bytes(stat: os.stat_result) -> int:
    blocks = int(getattr(stat, "st_blocks", 0) or 0)
    return blocks * 512 if blocks > 0 else int(stat.st_size)


def prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    """Remove ``path`` and its now-empty parents up to (excluding) ``stop_at``."""
    current = path
    try:
        stop = stop_at.resolve()
    except OSError:
        return
    while True:
        try:
            if current.resolve() == stop:
                return
            current.rmdir()
        except (FileNotFoundError, OSError):
            return
        current = current.parent


@dataclass
class _Candidate:
    path: Path
    kind: str  # "entry_dir" | "blob" | "temp" | "evicted"
    key: str  # entry dir rel path, blob digest, or ""
    file_bytes: int
    disk_bytes: int
    file_count: int
    dir_count: int  # directories strictly below ``path``
    is_dir: bool = False


class _Walk:
    """State for one pass; the module-level ``reconcile_store`` drives it."""

    def __init__(
        self,
        base_dir: Path,
        *,
        dry_run: bool,
        lock: AbstractContextManager[Any] | None,
        protected: Callable[[], ProtectedSets] | None,
        manifest_generation: Callable[[], int] | None,
        on_yield: Callable[[], None] | None,
        should_stop: Callable[[], bool] | None,
        yield_every: int,
        now: float,
    ) -> None:
        self.base_dir = base_dir
        self.report = ReconcileReport(base_dir=str(base_dir), dry_run=bool(dry_run))
        self.lock: AbstractContextManager[Any] = lock if lock is not None else nullcontext()
        self.protected = protected
        self.manifest_generation = manifest_generation
        self.on_yield = on_yield
        self.should_stop = should_stop
        self.yield_every = max(1, int(yield_every))
        self.now = now
        self.visited = 0
        self.generation_seen: int | None = None
        self.entry_dirs: set[str] = set()
        self.blob_hashes: set[str] = set()

    # -- pacing -------------------------------------------------------------

    def tick(self, count: int = 1) -> bool:
        """Count visited files; yield at the cadence; True when told to stop."""
        before = self.visited
        self.visited += count
        if self.visited // self.yield_every != before // self.yield_every:
            if self.on_yield is not None:
                self.on_yield()
            if self.should_stop is not None and self.should_stop():
                self.report.interrupted = True
                return True
        return False

    # -- manifest -----------------------------------------------------------

    def read_manifest(self) -> None:
        self.entry_dirs, self.blob_hashes = manifest_sets(self.base_dir)
        self.report.entries = len(self.entry_dirs)
        if self.manifest_generation is not None:
            self.generation_seen = int(self.manifest_generation())

    def refresh_manifest_if_changed(self) -> None:
        if self.manifest_generation is None:
            return
        generation = int(self.manifest_generation())
        if generation != self.generation_seen:
            self.read_manifest()

    # -- accounting -----------------------------------------------------------

    def keep(self, stat: os.stat_result, *, database: bool = False) -> None:
        r = self.report
        size = int(stat.st_size)
        alloc = allocated_bytes(stat)
        r.file_bytes += size
        r.disk_bytes += alloc
        r.file_count += 1
        if database:
            r.database_file_bytes += size
            r.database_disk_bytes += alloc

    def keep_candidate(self, cand: _Candidate) -> None:
        r = self.report
        r.file_bytes += cand.file_bytes
        r.disk_bytes += cand.disk_bytes
        r.file_count += cand.file_count
        r.dir_count += cand.dir_count + (1 if cand.is_dir else 0)

    def note_orphan(self, cand: _Candidate) -> None:
        r = self.report
        r.orphan_file_bytes += cand.file_bytes
        r.orphan_disk_bytes += cand.disk_bytes
        r.orphan_file_count += cand.file_count
        if cand.kind == "entry_dir":
            r.orphan_entry_dirs.append(cand.key)
        elif cand.kind == "blob":
            r.orphan_blob_files += 1
            r.orphan_blob_bytes += cand.file_bytes
        elif cand.kind == "evicted":
            r.evicted_entries_bytes += cand.file_bytes
        elif cand.kind == "temp":
            r.stale_temp_files += cand.file_count

    def measure_tree(self, root: Path) -> tuple[int, int, int, int]:
        """(file_bytes, disk_bytes, file_count, dir_count) under ``root``."""
        file_bytes = disk_bytes = files = dirs = 0
        for current, subdirs, filenames in os.walk(root):
            dirs += len(subdirs)
            for name in filenames:
                try:
                    stat = (Path(current) / name).stat()
                except OSError:
                    continue
                file_bytes += int(stat.st_size)
                disk_bytes += allocated_bytes(stat)
                files += 1
                if self.tick():
                    return file_bytes, disk_bytes, files, dirs
        return file_bytes, disk_bytes, files, dirs

    def is_stale_temp(self, stat: os.stat_result) -> bool:
        return (self.now - float(stat.st_mtime)) >= TEMP_GRACE_S

    # -- deletion -------------------------------------------------------------

    def settle(self, candidates: list[_Candidate], *, stop_at: Path | None) -> None:
        """Account for a batch; in delete mode remove what is still garbage.

        Runs under ``lock``: the manifest is re-read if the store changed
        since the last read, in-flight writes are consulted, and only a
        candidate that nothing claims is removed.
        """
        if not candidates:
            return
        if self.report.dry_run:
            for cand in candidates:
                self.note_orphan(cand)
                self.keep_candidate(cand)
            return
        with self.lock:
            self.refresh_manifest_if_changed()
            protected_dirs: set[str] = set()
            protected_hashes: set[str] = set()
            if self.protected is not None:
                protected_dirs, protected_hashes = self.protected()
            for cand in candidates:
                claimed = False
                if cand.kind == "entry_dir":
                    claimed = cand.key in self.entry_dirs or cand.key in protected_dirs
                elif cand.kind == "blob":
                    claimed = cand.key in self.blob_hashes or cand.key in protected_hashes
                if claimed:
                    self.report.protected_skipped += 1
                    self.keep_candidate(cand)
                    continue
                self.note_orphan(cand)
                self.remove(cand)
            if stop_at is not None:
                for parent in {cand.path.parent for cand in candidates}:
                    prune_empty_parents(parent, stop_at=stop_at)

    def remove(self, cand: _Candidate) -> None:
        r = self.report
        try:
            if cand.is_dir:
                shutil.rmtree(cand.path)
            else:
                cand.path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            # Left in place; the next pass sees it again. Counted as kept so
            # the census still describes what is on disk.
            self.keep_candidate(cand)
            return
        if cand.is_dir:
            r.dirs_deleted += cand.dir_count + 1
        r.files_deleted += cand.file_count
        r.file_bytes_deleted += cand.file_bytes
        r.disk_bytes_deleted += cand.disk_bytes

    # -- the walk ---------------------------------------------------------------

    def run(self) -> ReconcileReport:
        base = self.base_dir
        self.read_manifest()
        self.report.manifest_found = (base / MANIFEST_FILENAME).exists()

        # Top level: the manifest (and its -wal/-shm/journal siblings) is the
        # database; any other stray file is kept and counted.
        try:
            top = sorted(os.scandir(base), key=lambda e: e.name)
        except OSError:
            return self.report
        for entry in top:
            if entry.is_dir(follow_symlinks=False):
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            self.keep(stat, database=entry.name.startswith(MANIFEST_FILENAME))
            if self.tick():
                return self.report

        # evicted_entries/: the whole tree is an archive of untracked entry
        # directories; nothing restores from it.
        evicted_root = base / "evicted_entries"
        if evicted_root.is_dir():
            fb, db, fc, dc = self.measure_tree(evicted_root)
            if self.report.interrupted:
                return self.report
            self.settle(
                [_Candidate(evicted_root, "evicted", "", fb, db, fc, dc, is_dir=True)],
                stop_at=None,
            )

        if self.walk_entries(base / "entries"):
            return self.report
        if self.walk_blobs(base / "blobs"):
            return self.report
        return self.report

    def walk_entries(self, entries_root: Path) -> bool:
        if not entries_root.is_dir():
            return False
        self.report.dir_count += 1
        for prefix_dir in self.sorted_dirs(entries_root):
            self.report.dir_count += 1
            batch: list[_Candidate] = []
            try:
                children = sorted(os.scandir(prefix_dir), key=lambda e: e.name)
            except OSError:
                continue
            for child in children:
                try:
                    stat = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not child.is_dir(follow_symlinks=False):
                    self.keep(stat)
                    if self.tick():
                        return True
                    continue
                path = Path(child.path)
                rel = str(path.relative_to(self.base_dir))
                fb, db, fc, dc = self.measure_tree(path)
                if self.report.interrupted:
                    return True
                cand = _Candidate(path, "entry_dir", rel, fb, db, fc, dc, is_dir=True)
                if child.name.startswith("."):
                    cand.kind = "temp"
                    cand.key = ""
                    if self.is_stale_temp(stat):
                        batch.append(cand)
                    else:
                        self.keep_candidate(cand)
                    continue
                if rel in self.entry_dirs:
                    self.keep_candidate(cand)
                    continue
                batch.append(cand)
            self.settle(batch, stop_at=entries_root)
        return False

    def walk_blobs(self, blobs_root: Path) -> bool:
        if not blobs_root.is_dir():
            return False
        self.report.dir_count += 1
        for prefix_dir in self.sorted_dirs(blobs_root):
            self.report.dir_count += 1
            batch: list[_Candidate] = []
            try:
                children = sorted(os.scandir(prefix_dir), key=lambda e: e.name)
            except OSError:
                continue
            for child in children:
                try:
                    stat = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                if child.is_dir(follow_symlinks=False):
                    fb, db, fc, dc = self.measure_tree(Path(child.path))
                    if self.report.interrupted:
                        return True
                    self.keep_candidate(
                        _Candidate(Path(child.path), "temp", "", fb, db, fc, dc, is_dir=True)
                    )
                    continue
                path = Path(child.path)
                size = int(stat.st_size)
                alloc = allocated_bytes(stat)
                if child.name.startswith("."):
                    cand = _Candidate(path, "temp", "", size, alloc, 1, 0)
                    if self.is_stale_temp(stat):
                        batch.append(cand)
                    else:
                        self.keep(stat)
                elif child.name.endswith(".bin"):
                    digest = child.name[: -len(".bin")]
                    if digest in self.blob_hashes:
                        self.keep(stat)
                    else:
                        batch.append(_Candidate(path, "blob", digest, size, alloc, 1, 0))
                else:
                    self.keep(stat)
                if self.tick():
                    return True
            self.settle(batch, stop_at=blobs_root)
        return False

    @staticmethod
    def sorted_dirs(root: Path) -> list[Path]:
        try:
            return sorted(
                (Path(e.path) for e in os.scandir(root) if e.is_dir(follow_symlinks=False)),
                key=lambda p: p.name,
            )
        except OSError:
            return []


def reconcile_store(
    base_dir: Path | str = DEFAULT_COLD_TIER_DIR,
    *,
    dry_run: bool = True,
    lock: AbstractContextManager[Any] | None = None,
    protected: Callable[[], ProtectedSets] | None = None,
    manifest_generation: Callable[[], int] | None = None,
    on_yield: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    yield_every: int = YIELD_EVERY_FILES,
) -> ReconcileReport:
    """Walk ``base_dir``, census it, and (unless ``dry_run``) delete its garbage.

    ``lock`` wraps each deletion batch; ``protected`` returns the entry
    directories and blob digests of writes in flight (called under
    ``lock``); ``manifest_generation`` lets the pass notice a store that
    changed under it and re-read the manifest before deleting;
    ``on_yield`` runs every ``yield_every`` files and ``should_stop`` is
    polled at the same cadence (a stopped pass returns a partial report
    with ``interrupted=True``). A missing ``base_dir`` yields an empty
    report; a missing manifest treats every entry directory and blob as
    unreachable, which is what they are.
    """
    base_dir = Path(base_dir).expanduser()
    started = time.perf_counter()
    walk = _Walk(
        base_dir,
        dry_run=dry_run,
        lock=lock,
        protected=protected,
        manifest_generation=manifest_generation,
        on_yield=on_yield,
        should_stop=should_stop,
        yield_every=yield_every,
        now=time.time(),
    )
    if base_dir.is_dir():
        walk.run()
    walk.report.elapsed_s = time.perf_counter() - started
    return walk.report


def collect_garbage(
    base_dir: Path | str = DEFAULT_COLD_TIER_DIR, *, dry_run: bool = True
) -> dict[str, Any]:
    """``mtplx gc``'s view of :func:`reconcile_store`: a plain dict report."""
    return reconcile_store(base_dir, dry_run=dry_run).as_dict()
