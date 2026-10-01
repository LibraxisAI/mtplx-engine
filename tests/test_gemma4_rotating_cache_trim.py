"""Exactness of ``Gemma4RollbackRotatingKVCache.trim`` outside the last-update
rollback: the session bank restoring a prompt-boundary buffer to an earlier
prefix (PR #283's near-prefix restore for rewritten Gemma 4 turns).

A bank entry is installed into a fresh cache through ``state``/``meta_state``
(what ``restore_cache`` does), so it carries no last update and every trim
takes the fall-through. The sliding-window cache is a window, not a history:
that trim is exact only when every row the next forward attends to is still
physically present, in temporal order. The old fall-through moved
``offset``/``_idx`` back and left the trimmed rows in place, where the next
in-place step's front trim and the next concat's temporal rotation both
counted them as history: silently wrong sliding-window KV on a one-token
suffix, and on any restore deeper than the last warm turn's suffix. The trim
now discards the stale rows and refuses (returns 0, so the bank's trim
helpers fail closed to a cold prefill) whenever a full window cannot be
served. Real cache class, CPU stream, no model.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.backends.gemma4_assistant import Gemma4RollbackRotatingKVCache

W = 8


def _rows(positions):
    # Every key/value row carries its own token position, so a buffer's
    # contents say exactly which positions it holds.
    pos = mx.array([float(p) for p in positions], dtype=mx.float32).reshape(
        1, 1, -1, 1
    )
    keys = mx.broadcast_to(pos, (1, 2, len(positions), 4))
    return keys, keys * 10.0


def _feed(cache, positions):
    keys, values = _rows(positions)
    cache.update_and_fetch(keys, values)


def _restored(cache):
    """A bank restore of ``cache``: state and meta into a fresh cache."""
    fresh = Gemma4RollbackRotatingKVCache(max_size=cache.max_size)
    fresh.state = cache.state
    fresh.meta_state = cache.meta_state
    assert fresh._last_update is None
    return fresh


def _attended(cache):
    """(positions the next forward would attend over, oldest first; offset)."""
    if cache.keys is None:
        return [], int(cache.offset)
    ordered = cache._temporal_order(cache.keys)
    span = min(int(cache.offset), int(cache.max_size))
    tail = ordered[..., -span:, :] if span else ordered[..., :0, :]
    mx.eval(tail)
    return [int(v) for v in tail[0, 0, :, 0].tolist()], int(cache.offset)


def _cold(prefix_len, follow):
    cache = Gemma4RollbackRotatingKVCache(max_size=W)
    _feed(cache, range(prefix_len))
    if follow:
        _feed(cache, follow)
    return _attended(cache)


def _warm_turn_buffer():
    # A cold prefill of 20 followed by a warm turn's 5-token suffix: the
    # buffer holds max_size-1 window rows plus the suffix (positions 13..24).
    cache = Gemma4RollbackRotatingKVCache(max_size=W)
    _feed(cache, range(20))
    _feed(cache, range(20, 25))
    assert int(cache.keys.shape[2]) == W - 1 + 5
    assert cache.offset == 25
    return cache


@pytest.mark.parametrize("restore_point", [1, 2, 5, 7, 8, 9, 13, 19])
@pytest.mark.parametrize("follow_len", [1, 3])
def test_restored_cold_prefill_trims_exactly_at_any_depth(restore_point, follow_len):
    # A cold prefill keeps the whole prompt in temporal order, so a restore
    # to any earlier position must equal a cold prefill of that position,
    # for the one-token (in-place) and the multi-token (concat) follow-up.
    with mx.stream(mx.cpu):
        total = 20
        follow = list(range(restore_point, restore_point + follow_len))
        live = Gemma4RollbackRotatingKVCache(max_size=W)
        _feed(live, range(total))
        warm = _restored(live)
        assert warm.trim(total - restore_point) == total - restore_point
        assert warm.offset == restore_point
        _feed(warm, follow)
        assert _attended(warm) == _cold(restore_point, follow)


@pytest.mark.parametrize("trim", [1, 4])
@pytest.mark.parametrize("follow_len", [1, 3])
def test_restored_warm_turn_trims_exactly_within_a_full_window(trim, follow_len):
    # Trimming at most suffix-1 rows leaves a full window before the restore
    # point (rows 13..20 for trim 4) and must equal the cold prefill.
    with mx.stream(mx.cpu):
        restore_point = 25 - trim
        follow = list(range(restore_point, restore_point + follow_len))
        warm = _restored(_warm_turn_buffer())
        assert warm.trim(trim) == trim
        assert warm.offset == restore_point
        _feed(warm, follow)
        assert _attended(warm) == _cold(restore_point, follow)


@pytest.mark.parametrize("trim", [5, 6, 11, 12, 25])
def test_restored_warm_turn_refuses_a_trim_beyond_a_full_window(trim):
    # Trimming 5 or more would leave fewer than max_size rows before the
    # restore point (position 12 is gone). The trim refuses and leaves the
    # cache untouched, which is what lets the bank fail closed to a cold
    # prefill instead of decoding on stale rows.
    with mx.stream(mx.cpu):
        warm = _restored(_warm_turn_buffer())
        before = _attended(warm)
        assert warm.trim(trim) == 0
        assert _attended(warm) == before
        assert warm.offset == 25


def test_restored_circular_buffer_refuses_any_trim():
    # In-place decode has wrapped the buffer: the newest rows are not the
    # physical tail, so a restored copy cannot trim honestly at all.
    with mx.stream(mx.cpu):
        live = Gemma4RollbackRotatingKVCache(max_size=W)
        _feed(live, range(20))
        for position in (20, 21, 22):
            _feed(live, [position])
        restored = _restored(live)
        assert restored.trim(2) == 0
        assert restored.trim(1) == 0
        assert restored.offset == 23
        # The live cache's own last-update rollback (one in-place step) is
        # the exact path and stays exact.
        assert live.trim(1) == 1
        assert _attended(live) == _cold(22, [])


def test_live_last_update_block_rollback_is_unchanged():
    with mx.stream(mx.cpu):
        cache = Gemma4RollbackRotatingKVCache(max_size=W)
        _feed(cache, range(20))
        _feed(cache, range(20, 23))
        assert cache.trim(2) == 2
        assert _attended(cache) == _cold(21, [])
        # The whole block, too.
        cache = Gemma4RollbackRotatingKVCache(max_size=W)
        _feed(cache, range(20))
        _feed(cache, range(20, 23))
        assert cache.trim(3) == 3
        assert _attended(cache) == _cold(20, [])


def test_restored_trim_to_empty_resets_the_buffer():
    with mx.stream(mx.cpu):
        live = Gemma4RollbackRotatingKVCache(max_size=W)
        _feed(live, range(5))
        cache = _restored(live)
        assert cache.trim(5) == 5
        assert cache.empty() and cache.offset == 0
        _feed(cache, range(3))
        assert _attended(cache) == _cold(3, [])
