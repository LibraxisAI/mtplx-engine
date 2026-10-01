"""Issue #505: a request must never wait behind the SSD cold-tier encode for
longer than one bounded unit of work.

The encode runs on the single model-owner thread. Its abort check used to sit
BEFORE a whole-tensor ``mx.eval`` and a whole-tensor host copy, and the
boundary hydration that runs before it had no check at all, so a request that
arrived mid-encode waited for everything that was already in flight. On a
64 GB Mac at ~104K context that was 500+ s and the stream stall watchdog
killed the request.

These tests drive the codec with a fake slow eval (every ``mx.eval`` costs a
fixed wall time) and measure the wait itself: the time from the moment the
foreground signal turns on to the moment ColdEncodeInterrupted is raised.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from mtplx.cache_bank import codec as codec_module
from mtplx.cache_bank.codec import (
    ColdEncodeInterrupted,
    TreeCodec,
    decode_gdn_boundaries,
    decode_tree,
    encode_payload,
)
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBankEntry

# The plan's bound for the foreground wait. One slow unit here is 50 ms, so a
# wait under this bound with a 10 s total encode proves the wait follows the
# unit and not the tensor or the payload.
FOREGROUND_WAIT_BOUND_S = 2.0
SLOW_EVAL_S = 0.05


class _SlowEval:
    """Stand-in for a machine where every eval is slow (memory pressure,
    swap): each ``mx.eval`` call costs a fixed wall time, whatever its size."""

    def __init__(self, monkeypatch, seconds: float = SLOW_EVAL_S) -> None:
        self.calls = 0
        self.seconds = seconds
        real_eval = mx.eval

        def slow_eval(*arrays):
            self.calls += 1
            time.sleep(self.seconds)
            return real_eval(*arrays)

        monkeypatch.setattr(codec_module.mx, "eval", slow_eval)


class _Foreground:
    """A request that arrives ``after_s`` into the encode. Records when the
    signal first read true, which is when the waiting starts."""

    def __init__(self, after_s: float) -> None:
        self.started = time.perf_counter()
        self.after_s = after_s
        self.arrived_at: float | None = None

    def __call__(self) -> bool:
        now = time.perf_counter()
        if now - self.started < self.after_s:
            return False
        if self.arrived_at is None:
            self.arrived_at = now
        return True


def _interrupted_wait(run, foreground: _Foreground) -> float:
    with pytest.raises(ColdEncodeInterrupted):
        run()
    assert foreground.arrived_at is not None
    return time.perf_counter() - foreground.arrived_at


def test_foreground_wait_is_one_unit_inside_a_long_blocked_tensor(monkeypatch):
    # 200 blocks x 50 ms = 10 s of encode for this one tensor.
    kv = mx.zeros((1, 2, 200 * 4, 4), dtype=mx.float16)
    slow = _SlowEval(monkeypatch)
    foreground = _Foreground(after_s=0.12)
    codec = TreeCodec(block_size=4, should_abort=foreground)

    waited = _interrupted_wait(lambda: codec.encode(kv), foreground)

    assert waited < FOREGROUND_WAIT_BOUND_S
    assert waited < 4 * SLOW_EVAL_S
    assert slow.calls < 20, "the encode stopped early instead of finishing the tensor"


def test_blocked_tensor_is_never_evaluated_whole(monkeypatch):
    """The unbounded unit of #505 was a whole-tensor eval ahead of the block
    loop. Every eval the codec issues is now a slice of bounded size."""
    kv = mx.zeros((1, 2, 64, 4), dtype=mx.float16)
    sizes: list[int] = []
    real_eval = mx.eval

    def recording_eval(*arrays):
        sizes.extend(int(a.nbytes) for a in arrays)
        return real_eval(*arrays)

    monkeypatch.setattr(codec_module.mx, "eval", recording_eval)

    TreeCodec(block_size=8).encode(kv)

    block_nbytes = 1 * 2 * 8 * 4 * 2
    assert sizes == [block_nbytes] * 8
    assert int(kv.nbytes) not in sizes


def test_foreground_wait_is_one_unit_inside_a_large_unblocked_tensor(monkeypatch):
    # No token axis to block on (axis 2 is short), 100 units of 50 ms.
    table = mx.zeros((1, 100 * 16, 4), dtype=mx.float16)
    slow = _SlowEval(monkeypatch)
    foreground = _Foreground(after_s=0.12)
    codec = TreeCodec(should_abort=foreground, unit_bytes=16 * 4 * 2)

    waited = _interrupted_wait(lambda: codec.encode(table), foreground)

    assert waited < FOREGROUND_WAIT_BOUND_S
    assert waited < 4 * SLOW_EVAL_S
    assert slow.calls < 20


def test_a_request_that_arrives_during_the_last_unit_is_seen_after_it(monkeypatch):
    """The check after the unit is the one that matters at a tensor's end:
    what follows is hashing, a blob write or the next tensor's graph."""
    _SlowEval(monkeypatch, seconds=0.2)
    foreground = _Foreground(after_s=0.05)
    codec = TreeCodec(should_abort=foreground)

    with pytest.raises(ColdEncodeInterrupted):
        codec.encode(mx.zeros((1, 8), dtype=mx.float16))
    assert not codec.tensors, "an interrupted unit must not be kept"


def test_sliced_capture_is_byte_identical_and_keeps_the_format():
    rng = np.random.default_rng(505)
    source = rng.standard_normal((1, 1, 37, 5)).astype(np.float32)
    value = mx.array(source)

    whole = TreeCodec()
    whole_spec = whole.encode(value)
    sliced = TreeCodec(unit_bytes=3 * 5 * 4)
    sliced_spec = sliced.encode(value)

    assert sliced_spec == whole_spec
    assert sliced_spec["kind"] == "tensor"
    assert sliced.tensors == whole.tensors
    assert sliced.tensors[sliced_spec["name"]] == source.tobytes()
    decoded = decode_tree(sliced_spec, lambda name: sliced.tensors[name])
    assert np.array_equal(np.array(decoded), source)


def test_sliced_capture_of_a_strided_view_is_row_major():
    base = mx.arange(2 * 6 * 3, dtype=mx.float32).reshape(6, 2, 3)
    view = base.transpose(1, 0, 2)  # not contiguous
    codec = TreeCodec(unit_bytes=2 * 3 * 4)
    spec = codec.encode(view)
    assert codec.tensors[spec["name"]] == np.array(view).tobytes()


def test_blocked_bytes_are_unchanged_so_stored_blobs_still_dedupe():
    rng = np.random.default_rng(7)
    source = rng.standard_normal((1, 2, 40, 4)).astype(np.float16)
    codec = TreeCodec(block_size=8)
    spec = codec.encode(mx.array(source))

    assert spec["kind"] == "tensor_blocks"
    assert [(b["start"], b["end"]) for b in spec["blocks"]] == [
        (s, s + 8) for s in range(0, 40, 8)
    ]
    for block in spec["blocks"]:
        expected = source[:, :, block["start"] : block["end"], :].tobytes()
        assert codec.tensors[block["name"]] == expected


def test_units_are_reported_with_kind_time_and_bytes():
    seen: list[tuple[str, float, int]] = []
    encode_payload(
        cache_snapshot=CacheSnapshot(
            states=(mx.zeros((1, 2, 16, 4), dtype=mx.float16),), meta_states=(None,)
        ),
        logits=mx.zeros((1, 8), dtype=mx.float16),
        hidden=None,
        mtp_history_snapshot=None,
        block_size=8,
        on_unit=lambda kind, seconds, nbytes: seen.append((kind, seconds, nbytes)),
    )
    assert [(kind, nbytes) for kind, _, nbytes in seen] == [
        ("block", 128),
        ("block", 128),
        ("tensor", 16),
    ]
    assert all(seconds >= 0.0 for _, seconds, _ in seen)


# ---- boundary hydration (runs before the encode, on the same thread) -------


def _boundary_spec(records: int) -> tuple[dict, dict[str, bytes]]:
    codec = TreeCodec()
    spec = {
        "gdn_boundaries": [
            {
                "tokens": 256 * (i + 1),
                "states": codec.encode((mx.full((1, 4), float(i), dtype=mx.float32),)),
                "meta_states": codec.encode((None,)),
                "hidden_last": codec.encode(None),
            }
            for i in range(records)
        ]
    }
    return spec, dict(codec.tensors)


def test_boundary_hydration_wait_is_one_record(monkeypatch):
    spec, tensors = _boundary_spec(records=40)  # 40 x 50 ms = 2 s in full
    slow = _SlowEval(monkeypatch)
    foreground = _Foreground(after_s=0.12)

    waited = _interrupted_wait(
        lambda: decode_gdn_boundaries(
            spec, lambda name: tensors[name], should_abort=foreground
        ),
        foreground,
    )

    assert waited < 4 * SLOW_EVAL_S
    assert slow.calls < 10


def test_boundary_hydration_without_a_signal_is_never_interrupted():
    spec, tensors = _boundary_spec(records=3)
    boundaries = decode_gdn_boundaries(spec, lambda name: tensors[name])
    assert [record[0] for record in boundaries] == [256, 512, 768]
    assert [float(record[1].states[0][0, 0]) for record in boundaries] == [0.0, 1.0, 2.0]


def _entry_with_loader(loader) -> SessionBankEntry:
    snapshot = CacheSnapshot(states=(), meta_states=())
    return SessionBankEntry(
        token_ids=(1, 2, 3),
        token_hash="hash",
        model_path="/models/test",
        mtp_enabled=False,
        hidden_variant=None,
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        session_id="sess-505",
        gdn_boundary_loader=loader,
    )


def test_an_interrupted_hydration_keeps_the_loader():
    """Losing the loader would persist the entry without its boundaries and
    downgrade the whole lineage on the next restart."""
    calls = {"n": 0}

    def loader(should_abort=None):
        calls["n"] += 1
        if should_abort is not None and should_abort():
            raise ColdEncodeInterrupted()
        return [(256, CacheSnapshot(states=(), meta_states=()), None)]

    entry = _entry_with_loader(loader)

    with pytest.raises(ColdEncodeInterrupted):
        entry._ensure_boundaries_loaded(should_abort=lambda: True)
    assert entry.gdn_boundary_loader is loader
    assert entry.gdn_boundaries == []

    entry._ensure_boundaries_loaded(should_abort=lambda: False)
    assert entry.gdn_boundary_loader is None
    assert [record[0] for record in entry.gdn_boundaries] == [256]
    assert calls["n"] == 2


def test_a_loader_without_the_signal_argument_still_hydrates():
    entry = _entry_with_loader(
        lambda: [(128, CacheSnapshot(states=(), meta_states=()), None)]
    )
    entry._ensure_boundaries_loaded(should_abort=lambda: True)
    assert [record[0] for record in entry.gdn_boundaries] == [128]


# ---- the tier reports its units --------------------------------------------


def test_tier_reports_the_longest_unit_and_counts_slow_ones(tmp_path, monkeypatch):
    import mtplx.cache_bank.cold_tier as cold_tier_module

    monkeypatch.setattr(cold_tier_module, "ENCODE_SLOW_UNIT_S", 0.03)
    _SlowEval(monkeypatch, seconds=0.04)
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:

        class Entry:
            token_ids = tuple(range(600))
            nbytes = 4096
            cache_snapshot = CacheSnapshot(
                states=(mx.zeros((1, 2, 8, 4), dtype=mx.float16),), meta_states=(None,)
            )
            logits = mx.zeros((1, 8), dtype=mx.float16)
            hidden = None
            mtp_history_snapshot = None
            gdn_boundaries = ()
            has_recurrent = False
            session_id = "sess-505-units"
            token_hash = "beef" * 4
            prefix_len = 600

        assert tier.put_entry(Entry(), capabilities=["ar_insert"]) is True
        stats = tier.stats()
        assert stats["encode_units"] == 2
        assert stats["encode_slow_units"] == 2
        assert stats["encode_longest_unit_s"] >= 0.04
        assert stats["encode_longest_unit"].startswith(("tensor:", "block:"))
    finally:
        tier.close() if hasattr(tier, "close") else None


def test_tier_shares_one_foreground_signal_with_hydration(tmp_path):
    tier = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1)
    try:
        assert tier.encode_should_abort() is None  # unwired: legacy behaviour
        signal = lambda: True  # noqa: E731
        tier.foreground_busy = signal
        assert tier.encode_should_abort() is signal
        tier._encode_yield_enabled = False
        assert tier.encode_should_abort() is None
        before = tier.stats()["encode_yields_foreground"]
        tier.note_encode_yield()
        assert tier.stats()["encode_yields_foreground"] == before + 1
    finally:
        tier.close() if hasattr(tier, "close") else None
