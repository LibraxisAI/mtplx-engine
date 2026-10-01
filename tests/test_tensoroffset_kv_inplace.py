"""CPU test: the fixed-M4 verify KV update writes in place, not a full copy.

The 261,120-token decode OOM (after the score-plane head-chunk fix) is the
eager verify's KV update: ``TensorOffsetKVCache.update_and_fetch`` used the
functional ``mx.slice_update``, which reallocates the whole
``[1, 2, capacity, 256]`` buffer per key and value tensor (267 MB each at 262K,
6.4 GB across the 12 full-attention layers). The stock ``KVCache`` writes in
place; an S=1 AR decode (MTP off) therefore fits at 94 GB while the S>1 verify
overflows. The fix writes the S new rows in place on the eager (concrete-offset)
path and keeps the compile-visible ``slice_update`` only when the offset is a
tracer. These tests run on CPU with a realistic 262K-capacity buffer.
"""

import mlx.core as mx
import pytest

from mtplx.graphbank import TensorOffsetKVCache, ensure_eager_window_capacity


CAP = 262144          # fixed-bank capacity at the pack's 262K context
D = 256               # head_dim
H_KV = 2              # key/value heads
S = 5                 # verify rows (depth + 2)
OFF = 261000          # a mid-buffer concrete offset (like the served verify)
ONE_BUFFER = H_KV * CAP * D * 2          # bytes of one [1,2,CAP,256] bf16 tensor
ROW_THRESHOLD = CAP * D * 2              # T x 256 x 2 (the coordinator's bar)


@pytest.fixture(autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def _make_cache(offset: int) -> TensorOffsetKVCache:
    keys = mx.zeros((1, H_KV, CAP, D), mx.bfloat16)
    values = mx.zeros((1, H_KV, CAP, D), mx.bfloat16)
    mx.eval(keys, values)
    return TensorOffsetKVCache(keys, values, offset)


def _am() -> int:
    return int(mx.get_active_memory())


def test_eager_update_is_in_place_no_full_buffer_alloc(monkeypatch):
    # Spy: the eager path must never call the full-buffer functional op.
    real_slice_update = mx.slice_update
    calls = []

    def _spy(a, *args, **kwargs):
        calls.append(tuple(a.shape))
        return real_slice_update(a, *args, **kwargs)

    monkeypatch.setattr(mx, "slice_update", _spy)

    cache = _make_cache(OFF)
    keys = mx.ones((1, H_KV, S, D), mx.bfloat16)
    values = (mx.ones((1, H_KV, S, D), mx.bfloat16) * 2).astype(mx.bfloat16)
    mx.eval(keys, values)

    before = _am()
    k_out, v_out = cache.update_and_fetch(keys, values)
    mx.eval(k_out, v_out, cache.cache[2])
    delta = _am() - before

    # No full [1,2,CAP,256] reallocation: the delta is the tiny rollback
    # snapshot (2 x [1,2,S,256]) plus scalars, far under one row-plane.
    assert delta < ROW_THRESHOLD, f"delta={delta} >= T*256*2={ROW_THRESHOLD}"
    # And the functional full-buffer op was not used on the eager path.
    assert calls == [], f"slice_update called on eager path: {calls}"


def test_eager_update_writes_correct_rows_and_offset():
    cache = _make_cache(OFF)
    keys = mx.arange(H_KV * S * D, dtype=mx.float32).reshape(1, H_KV, S, D).astype(mx.bfloat16)
    values = (keys + mx.array(1000, mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(keys, values)

    k_out, v_out = cache.update_and_fetch(keys, values)
    mx.eval(k_out, v_out)

    # offset advanced by S
    assert int(cache.cache[2].item()) == OFF + S
    # written rows equal the input
    assert bool(mx.all(k_out[:, :, OFF:OFF + S, :] == keys).item())
    assert bool(mx.all(v_out[:, :, OFF:OFF + S, :] == values).item())
    # rows before the offset untouched (still zero)
    assert float(mx.max(mx.abs(k_out[:, :, :OFF, :])).item()) == 0.0
    # returned buffer is the same object mutated in place (no reallocation)
    assert k_out is cache.cache[0]
    assert v_out is cache.cache[1]


def test_rollback_snapshot_independent_and_trim_restores():
    # Pre-fill the S rows with a known "old" value so we can prove the
    # snapshot is an independent copy and that trim restores it.
    cache = _make_cache(OFF)
    old = (mx.ones((1, H_KV, S, D), mx.bfloat16) * 7).astype(mx.bfloat16)
    cache.cache[0][:, :, OFF:OFF + S, :] = old
    cache.cache[1][:, :, OFF:OFF + S, :] = old
    mx.eval(cache.cache[0], cache.cache[1])

    new = (mx.ones((1, H_KV, S, D), mx.bfloat16) * 3).astype(mx.bfloat16)
    cache.update_and_fetch(new, new)
    mx.eval(cache.cache[0], cache.cache[1])

    # snapshot captured the OLD rows and is not aliased to the mutated buffer
    assert float(mx.max(cache.rollback_state[1]).item()) == 7.0
    assert float(mx.min(cache.rollback_state[1]).item()) == 7.0

    # trim (rejected draft) restores the old rows in place and rewinds offset
    cache.trim(S)
    mx.eval(cache.cache[0], cache.cache[1])
    assert int(cache.cache[2].item()) == OFF
    assert bool(mx.all(cache.cache[0][:, :, OFF:OFF + S, :] == old).item())
    assert bool(mx.all(cache.cache[1][:, :, OFF:OFF + S, :] == old).item())


def test_eager_matches_functional_slice_update_numerics():
    # The in-place write must be byte-identical to the functional path it
    # replaces (same bytes land at the same positions).
    off = 120000
    keys = mx.random.normal((1, H_KV, S, D)).astype(mx.bfloat16)
    values = mx.random.normal((1, H_KV, S, D)).astype(mx.bfloat16)
    mx.eval(keys, values)

    a = _make_cache(off)
    ka, va = a.update_and_fetch(keys, values)

    # reference: functional slice_update at the same offset
    ref_k = mx.zeros((1, H_KV, CAP, D), mx.bfloat16)
    ref_v = mx.zeros((1, H_KV, CAP, D), mx.bfloat16)
    ref_k = mx.slice_update(ref_k, keys, mx.array(off, mx.int32), axes=(2,))
    ref_v = mx.slice_update(ref_v, values, mx.array(off, mx.int32), axes=(2,))
    mx.eval(ka, va, ref_k, ref_v)

    assert bool(mx.all(ka == ref_k).item())
    assert bool(mx.all(va == ref_v).item())


def test_concrete_offset_detects_tracer_vs_value():
    cache = _make_cache(OFF)
    assert cache._concrete_offset() == OFF  # concrete scalar

    # Inside an mx.compile trace the offset is a tracer; _concrete_offset must
    # return None so the compile-visible functional path is taken.
    seen = {}

    @mx.compile
    def _probe(off_arr):
        c = TensorOffsetKVCache(
            mx.zeros((1, H_KV, 64, D), mx.bfloat16),
            mx.zeros((1, H_KV, 64, D), mx.bfloat16),
            off_arr,
        )
        seen["concrete"] = c._concrete_offset()
        return off_arr + 1

    mx.eval(_probe(mx.array(3, mx.int32)))
    assert seen["concrete"] is None


def test_eager_write_past_capacity_grows_instead_of_dropping_rows():
    # The copy-block route writes 1 + block rows in one eager update. When
    # that window straddles the granted capacity, the functional path
    # clamped silently (MLX drops the rows that do not fit and the offset
    # still advances past them) and the in-place write refused the shape.
    # The buffer must grow first so every row lands and the offset is right.
    keys = mx.zeros((1, H_KV, 64, D), mx.bfloat16)
    values = mx.zeros((1, H_KV, 64, D), mx.bfloat16)
    mx.eval(keys, values)
    cache = TensorOffsetKVCache(keys, values, 52, step=16)
    cache._granted = True
    new_k = mx.full((1, H_KV, 25, D), 1.0, mx.bfloat16)
    new_v = mx.full((1, H_KV, 25, D), 2.0, mx.bfloat16)
    k, v = cache.update_and_fetch(new_k, new_v)
    mx.eval(k, v)
    assert int(k.shape[2]) == 80 and int(v.shape[2]) == 80  # 77 rounded up to the 16-row step
    assert cache.size() == 77
    assert mx.array_equal(k[:, :, 52:77, :], new_k).item()
    assert mx.array_equal(v[:, :, 52:77, :], new_v).item()
    assert cache.growth_after_grant is True
    # The rollback snapshot still covers the whole write: trim restores the
    # pre-write rows (zeros) and the pre-write offset.
    assert cache.trim(25) == 25
    assert cache.size() == 52
    assert not mx.any(cache.keys[:, :, 52:77, :] != 0).item()
    assert not mx.any(cache.values[:, :, 52:77, :] != 0).item()


def test_eager_window_preflight_grows_only_entries_at_the_edge():
    # The copy-block route calls this before its forward so the mask the
    # forward builds from the first full-attention layer's capacity matches
    # every layer's buffer. Entries with room are left alone; other cache
    # types and empty slots are skipped.
    def _entry(cap, off):
        k = mx.zeros((1, H_KV, cap, D), mx.bfloat16)
        v = mx.zeros((1, H_KV, cap, D), mx.bfloat16)
        mx.eval(k, v)
        c = TensorOffsetKVCache(k, v, off, step=16)
        c._granted = True
        return c
    tight = _entry(64, 52)
    roomy = _entry(128, 52)
    cache = [None, "linear-layer-state", tight, roomy]
    assert ensure_eager_window_capacity(cache, 25) == 1
    assert int(tight.keys.shape[2]) == 80 and tight.growth_after_grant is True
    assert int(roomy.keys.shape[2]) == 128 and roomy.growth_after_grant is False
    assert ensure_eager_window_capacity(cache, 25) == 0
    assert ensure_eager_window_capacity([], 25) == 0
