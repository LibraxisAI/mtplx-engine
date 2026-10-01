"""Eviction must yield without locking out restores or losing shared blobs."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.cache_bank.codec import ColdEncodeInterrupted
from mtplx.cache_bank.cold_tier import SessionBankColdTier


def seed_entry(tier, name, *, blobs=130, shared=()):
    tensors = {digest: raw for digest, raw in shared}
    for i in range(blobs):
        raw = f"{name}-{i}".encode() * 64
        tensors[hashlib.sha256(raw).hexdigest()] = raw
    entry = SimpleNamespace(
        token_ids=(1, 2, 3),
        nbytes=sum(map(len, tensors.values())),
        model_path="test",
        mtp_enabled=False,
        snapshot_epoch=0,
        session_id=name,
    )
    metadata = tier._metadata_for_entry(
        entry, capabilities=(), payload_nbytes=entry.nbytes
    )
    metadata["entry_dir"] = f"entries/{name}/{metadata['entry_id']}"
    directory = tier.base_dir / metadata["entry_dir"]
    directory.mkdir(parents=True)
    for digest, raw in tensors.items():
        tier._write_blob(digest, raw)
    payload = {"tensor_blobs": {key: {"sha256": key} for key in tensors}}
    (directory / "payload.json").write_text(json.dumps(payload))
    tier._insert_manifest(metadata)
    tier._invalidate_disk_usage_cache()
    return metadata, tensors


def test_eviction_pauses_on_arrival_without_holding_store_lock(tmp_path, monkeypatch):
    tier = SessionBankColdTier(base_dir=tmp_path, mode="on")
    busy = threading.Event()
    paused = threading.Event()
    tier.foreground_busy = busy.is_set
    metadata, tensors = seed_entry(tier, "victim")
    tier._managed_disk_usage(force=True)
    original_unlink = Path.unlink
    original_pause = tier._pause_for_foreground
    deleted = []

    def unlink(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if path.suffix == ".bin" and path.is_relative_to(tmp_path):
            deleted.append(path)
            if len(deleted) == 1:
                busy.set()  # Arrival during an already-running deletion batch.
        return result

    def pause():
        if busy.is_set():
            paused.set()
        original_pause()

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(tier, "_pause_for_foreground", pause)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            eviction = pool.submit(tier._evict_until_room, 1, cap_bytes=1)
            try:
                assert paused.wait(2), "eviction ignored an arriving foreground request"

                def restore_lock():
                    with tier._base_lock:
                        return True

                assert pool.submit(restore_lock).result(timeout=1)
                assert 1 <= len(deleted) <= 64
                assert not eviction.done()
            finally:
                busy.clear()
            eviction.result(timeout=5)
        assert tier._stats["entries_evicted"] == 1
        assert all(not tier._blob_path(d).exists() for d in tensors)
        assert not (tmp_path / metadata["entry_dir"]).exists()
    finally:
        busy.clear()
        tier.close()


def test_reclaim_rechecks_new_references_and_protects_inflight_blobs(tmp_path):
    tier = SessionBankColdTier(base_dir=tmp_path, mode="on")
    _, tensors = seed_entry(tier, "victim", blobs=140)
    with tier._connect() as conn:
        row = conn.execute("SELECT * FROM entries").fetchone()
    with tier._base_lock:
        candidates = tier._retire_entry_row(row)
    ordered = sorted(candidates)
    shared, inflight = ordered[-2:]
    tier._claim_inflight(digests={inflight})
    changed = False

    def arrival():
        nonlocal changed
        # After the first bounded delete batch, another writer commits a
        # reference to a not-yet-deleted blob. The old reference set is stale.
        if not changed and not tier._blob_path(ordered[0]).exists():
            seed_entry(tier, "survivor", blobs=0, shared=[(shared, tensors[shared])])
            changed = True

    try:
        tier._delete_unreferenced_blobs(candidates, on_yield=arrival)
        assert changed
        assert tier._blob_path(shared).read_bytes() == tensors[shared]
        assert tier._blob_path(inflight).read_bytes() == tensors[inflight]
        assert sum(tier._blob_path(d).exists() for d in candidates) == 2
    finally:
        tier._release_inflight(digests={inflight})
        tier.close()


def test_owner_spill_yields_before_eviction_instead_of_waiting(tmp_path):
    tier = SessionBankColdTier(base_dir=tmp_path, mode="on", min_prefix_tokens=1)
    seed_entry(tier, "victim", blobs=8)
    tier._managed_disk_usage(force=True)
    tier.max_bytes = 1024
    tier.foreground_busy = lambda: True
    entry = SimpleNamespace(
        token_ids=(4, 5, 6),
        nbytes=128,
        model_path="test",
        mtp_enabled=False,
        snapshot_epoch=0,
        session_id="incoming",
    )
    try:
        with pytest.raises(ColdEncodeInterrupted):
            tier.spill_entry(entry, raise_on_yield=True)
        assert tier._stats["entries_evicted"] == 0
        assert tier._stats["encode_yields_foreground"] == 1
        assert tier.spill_entry(entry) is False
    finally:
        tier.foreground_busy = None
        tier.close()
