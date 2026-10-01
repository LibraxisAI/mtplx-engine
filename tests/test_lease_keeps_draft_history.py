"""A lease must keep the draft head's history it was handed (issue #499).

The reported turn: 149,649 prompt tokens, ``cached_tokens 0``, time to first
token 570 s, ``cache_miss_reason: ssd_missing_mtp_history``, on a 48 GB Mac
where a 142K-token snapshot (11 GB) is over the 7.4 GB per-session budget.

What happened, end to end. A snapshot over the per-session budget becomes a
live-reference lease. At generation-final the server hands the bank the draft
head's committed history as a SNAPSHOT (``_generation_final_bank_metadata``),
never as a live reference, and the lease threw that snapshot away while
keeping its epoch. So with MTP on:

* the next turn's restore took the lease's trunk reference, then found no
  draft history to go with it and failed with ``no_snapshot_coverage``. The
  failed attempt had already consumed the lease;
* the idle-lane spill wrote the SSD copy from the same lease, so without the
  history, and ``_restore_cold`` refuses such a record under a committed
  history policy (``ssd_missing_mtp_history``);
* the turn prefilled everything cold. Every turn, for the rest of the session.

The draft history is small next to the trunk (one attention layer: about 4 KB
per token on the 27B, against 64 KB per token of trunk KV), so keeping it is
cheap, and it is the same trunk-reference-plus-history-snapshot pairing every
durable generation-final entry already restores with.

Exactness: the draft head only proposes tokens; acceptance uses the target's
probabilities with residual correction, so the target distribution cannot
depend on this. The restored history is the one a durable entry would restore.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank

IDENTITY = {"template_hash": "template-a", "policy_fingerprint": "policy-a"}


class KV:
    """Real-contract cache leaf: state, meta_state, trimmable."""

    def __init__(self, rows: int = 16, width: int = 4, fill: float = 1.0) -> None:
        self.state = mx.full((1, 1, rows, width), fill, dtype=mx.float16)
        self.meta_state = ("kv", str(rows))
        self.offset = rows

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        self.offset -= int(n)
        return int(n)


class Runtime:
    model_path = Path("models/example")
    mtp_enabled = True

    def make_cache(self):
        return [KV(fill=0.0)]

    def make_mtp_cache(self):
        return [KV(rows=4, fill=0.0)]


def _history_snapshot(fill: float = 7.0) -> CacheSnapshot:
    return CacheSnapshot(
        states=(mx.full((1, 1, 4, 4), fill, dtype=mx.float16),),
        meta_states=(("kv", "4"),),
    )


def _generation_final_lease(bank: SessionBank, tokens, cache, **kwargs):
    """The put the server makes at generation-final, over the per-session cap."""

    entry = bank.put(
        runtime=Runtime(),
        token_ids=list(tokens),
        cache=cache,
        logits=mx.array([[0.5, 1.5]], dtype=mx.float16),
        hidden=mx.array([[[2.0, 3.0]]], dtype=mx.float16),
        hidden_variant="post_norm",
        keep_live_ref=True,
        session_id="s1",
        mtp_history_policy="committed",
        mtp_history_snapshot=_history_snapshot(),
        snapshot_epoch=len(list(tokens)),
        mtp_snapshot_epoch=len(list(tokens)),
        **IDENTITY,
        **kwargs,
    )
    assert entry is not None and entry.live_ref_only is True
    return entry


def test_the_next_turn_restores_from_the_lease_with_its_draft_history():
    bank = SessionBank(per_session_max_bytes=1)
    cache = [KV()]
    _generation_final_lease(bank, range(10), cache)

    restored = bank.restore(
        Runtime(),
        [*range(10), 99, 98],
        mode="reference",
        session_id="s1",
        hidden_variant="post_norm",
        mtp_history_policy="committed",
        **IDENTITY,
    )

    # Before: None, miss reason no_snapshot_coverage, and the lease consumed.
    assert restored is not None, bank.last_miss_reason
    assert restored.restore_mode == "reference_lease"
    assert restored.cache is cache
    assert restored.mtp_history_cache is not None
    assert float(restored.mtp_history_cache[0].state[0, 0, 0, 0]) == 7.0


def test_the_ssd_copy_of_a_lease_carries_the_draft_history(tmp_path):
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2
    )
    try:
        jobs = []
        bank = SessionBank(per_session_max_bytes=1, cold_tier=cold)
        bank.cold_enqueue_dispatch = jobs.append
        _generation_final_lease(bank, range(10), [KV()])
        assert len(jobs) == 1 and jobs[0]() is True

        # A restart: nothing in RAM, the same conversation comes back.
        fresh = SessionBank(per_session_max_bytes=1, cold_tier=cold)
        restored = fresh.restore(
            Runtime(),
            [*range(10), 99, 98],
            session_id="s1",
            hidden_variant="post_norm",
            mtp_history_policy="committed",
            **IDENTITY,
        )

        # Before: None with last_miss_reason == "ssd_missing_mtp_history",
        # which is the line in the reporter's trace.
        assert restored is not None, fresh.last_miss_reason
        assert restored.cache_source == "ssd"
        assert restored.ssd_cached_tokens == 10
        assert restored.mtp_history_cache is not None
        assert float(restored.mtp_history_cache[0].state[0, 0, 0, 0]) == 7.0
    finally:
        cold.close()


def test_a_lease_that_has_a_live_history_reference_does_not_hold_a_copy_as_well():
    bank = SessionBank(per_session_max_bytes=1)
    history = [KV(rows=4)]
    entry = _generation_final_lease(
        bank, range(10), [KV()], mtp_history_cache_ref=history
    )

    assert entry.mtp_history_cache_ref is history
    assert entry.mtp_history_snapshot is None


def test_the_kept_history_is_counted_in_what_the_lease_holds():
    bank = SessionBank(per_session_max_bytes=1)
    entry = _generation_final_lease(bank, range(10), [KV()])

    history_nbytes = 4 * 4 * 2  # 1x1x4x4 float16
    assert entry.mtp_history_snapshot is not None
    assert entry.lease_aux_nbytes >= history_nbytes
    # Still held after a restore took the trunk reference.
    entry.cache_ref = None
    assert entry.held_nbytes == entry.lease_aux_nbytes
