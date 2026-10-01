"""A partial restore of a recurrent (hybrid GDN) entry lands on a stored
recurrent boundary for EVERY gap, tiny ones included.

Until 2026-09-08 gaps of 1..8 tokens kept the attention-only "tokenizer drift"
tolerance on hybrid entries: the KV was trimmed to matched-1 but the GDN/conv
state stayed at the entry's end, and the caller then re-forwarded token
matched-1 on it -- state from up to eight tokens the new prompt does not
contain plus one token consumed twice (three audits reproduced it with this
probe). Attention-only entries keep the tolerance: there the trim is exact.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cache_state import snapshot_cache, snapshot_untrimmable_cache
from mtplx.session_bank import SessionBank


class _FakeKV:
    """Trimmable attention KV: offset carried by the keys' seq axis."""

    def __init__(self, offset: int = 0) -> None:
        self.keys = mx.arange(offset, dtype=mx.float32).reshape(1, 1, offset, 1)
        self.offset = int(offset)

    @property
    def state(self):
        return (self.keys,)

    @state.setter
    def state(self, value) -> None:
        (keys,) = value
        self.keys = keys
        self.offset = int(keys.shape[2])

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, value) -> None:
        self.offset = int(value[0])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(int(self.offset), int(n))
        self.offset -= n
        return n


class _FakeRecurrent:
    """Non-trimmable GDN-style entry: state = [conv, gdn] stamped with the token count."""

    def __init__(self, consumed: int = 0) -> None:
        self.cache = [mx.array([float(consumed)]), mx.array([float(consumed)])]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value) -> None:
        self.cache = [v for v in value]

    def replace_state(self, value) -> None:
        self.cache = [v for v in value]

    @property
    def meta_state(self):
        return ("owned_recurrent_state", "persistent_eval")

    @meta_state.setter
    def meta_state(self, value) -> None:
        pass

    def is_trimmable(self) -> bool:
        return False


def _runtime(hybrid: bool):
    return SimpleNamespace(
        model_path=Path("hybrid-tiny" if hybrid else "attn-tiny"),
        mtp_enabled=True,
        make_cache=(lambda: [_FakeKV(), _FakeRecurrent()]) if hybrid else (lambda: [_FakeKV()]),
        make_mtp_cache=lambda: [_FakeKV()],
    )


def _entry(bank, rt, *, hybrid: bool, prefix_len: int, boundary_at: int | None):
    live = [_FakeKV(prefix_len), _FakeRecurrent(prefix_len)] if hybrid else [_FakeKV(prefix_len)]
    boundaries = None
    if boundary_at is not None:
        at_boundary = [_FakeKV(boundary_at), _FakeRecurrent(boundary_at)]
        boundaries = [(boundary_at, snapshot_untrimmable_cache(at_boundary), None)]
    return bank.put(
        runtime=rt,
        token_ids=list(range(prefix_len)),
        cache=live,
        logits=mx.zeros((1, 4)),
        hidden=mx.zeros((1, 1, 2)),
        hidden_variant="post_norm",
        mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache([_FakeKV(prefix_len - 1)]),
        snapshot_epoch=prefix_len,
        mtp_snapshot_epoch=prefix_len,
        gdn_boundaries=boundaries,
    )


P = 100


@pytest.mark.parametrize("gap", [1, 4, 8, 9, 40])
def test_hybrid_partial_restore_lands_on_the_boundary_for_every_gap(gap):
    rt = _runtime(True)
    bank = SessionBank()
    entry = _entry(bank, rt, hybrid=True, prefix_len=P + gap, boundary_at=P - 2)
    assert entry is not None and entry.has_recurrent
    result = bank.restore_entry_prefix_cache(rt, entry, P, mode="clone")
    assert result is not None
    cache, mtp_cache, _mode, restore_point, _hidden = result
    kv, rec = cache
    assert restore_point == P - 2
    assert kv.offset == P - 2
    # the recurrent state is the boundary's, never the entry's end
    assert float(rec.state[1][0]) == float(P - 2)
    assert mtp_cache[0].offset == P - 3


@pytest.mark.parametrize("gap", [1, 8])
def test_hybrid_partial_restore_without_a_boundary_declines(gap):
    rt = _runtime(True)
    bank = SessionBank()
    entry = _entry(bank, rt, hybrid=True, prefix_len=P + gap, boundary_at=None)
    assert entry is not None and entry.has_recurrent
    assert bank.restore_entry_prefix_cache(rt, entry, P, mode="clone") is None
    assert bank.last_miss_reason == "no_snapshot_coverage"


@pytest.mark.parametrize("gap", [1, 8])
def test_attention_only_partial_restore_keeps_the_exact_trim(gap):
    rt = _runtime(False)
    bank = SessionBank()
    entry = _entry(bank, rt, hybrid=False, prefix_len=P + gap, boundary_at=None)
    assert entry is not None and not entry.has_recurrent
    result = bank.restore_entry_prefix_cache(rt, entry, P, mode="clone")
    assert result is not None
    cache, _mtp, _mode, restore_point, _hidden = result
    assert restore_point == P
    assert cache[0].offset == P - 1  # seed-forward slot: the caller re-runs token P-1
