"""Long-cycle repetition stop: exact loops above the short-block window.

The #311 stop tries every block size up to 96 tokens on every commit. On
2026-09-22 a 4-bit MiMo 9B build looped a 1,323-token cycle (100,912 tokens,
no answer) and a 129-token cycle (16,916 tokens) straight past it.
``_LongCycleStop`` does constant work per committed token and fires at three
whole copies of one exact cycle spanning at least 512 tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mtplx.generation import (
    RepetitionStopConfig,
    _detect_repeated_token_suffix,
    _long_cycle_stop,
    _LongCycleStop,
    _repetition_stop_config,
    _trim_repeated_suffix,
)

FIXTURE = Path(__file__).parent / "fixtures" / "repetition_long_cycle_loops.json"
PREFIX = 1000  # distinct tokens before every synthetic loop


@pytest.fixture(autouse=True)
def _default_repetition_env(monkeypatch):
    # The product defaults are under test; a developer's shell must not
    # change them.
    for name in (
        "MTPLX_REPETITION_STOP_MIN_TOKENS",
        "MTPLX_REPETITION_STOP_MIN_REPEATED_TOKENS",
        "MTPLX_REPETITION_STOP_MIN_REPEATS",
        "MTPLX_REPETITION_STOP_MIN_BLOCK_TOKENS",
        "MTPLX_REPETITION_STOP_MAX_BLOCK_TOKENS",
        "MTPLX_REPETITION_STOP_MAX_CYCLE_TOKENS",
        "MTPLX_REPETITION_STOP_MIN_CYCLE_COPIES",
        "MTPLX_REPETITION_STOP_MIN_CYCLE_SPAN_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def _need(period: int, config: RepetitionStopConfig) -> int:
    """Run length (tokens equal to the one a period earlier) that fires."""
    return max(
        (config.min_cycle_copies - 1) * period,
        config.min_cycle_span_tokens - period,
    )


def _loop(period: int, copies: int, *, prefix: int = PREFIX) -> list[int]:
    # Prefix ids and cycle ids never overlap, so the only repetition is the
    # cycle itself.
    head = list(range(1_000_000, 1_000_000 + prefix))
    cycle = list(range(1, period + 1))
    return head + cycle * copies


def _feed(tokens: list[int], config: RepetitionStopConfig, chunk: int = 1):
    """Replay tokens as a decode loop commits them; (fire length, result)."""
    detector = _long_cycle_stop(config)
    assert detector is not None
    committed: list[int] = []
    for start in range(0, len(tokens), chunk):
        committed.extend(tokens[start : start + chunk])
        result = detector.observe(committed)
        if result is not None:
            return len(committed), result
    return None, None


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


def test_product_defaults_cover_periods_above_the_short_block_window():
    config = _repetition_stop_config(True)
    assert config.max_block_tokens == 96
    assert config.max_cycle_tokens == 8192
    assert config.min_cycle_copies == 3
    assert config.min_cycle_span_tokens == 512
    assert isinstance(_long_cycle_stop(config), _LongCycleStop)


def test_disarmed_or_empty_range_builds_no_detector(monkeypatch):
    assert _long_cycle_stop(_repetition_stop_config(False)) is None
    assert (
        _long_cycle_stop(
            RepetitionStopConfig(enabled=True, max_block_tokens=96, max_cycle_tokens=96)
        )
        is None
    )
    # The cycle cap alone turns the long-cycle stop off, the short one stays.
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MAX_CYCLE_TOKENS", "0")
    config = _repetition_stop_config(True)
    assert config.enabled
    assert _long_cycle_stop(config) is None


def test_env_knobs_reach_the_config(monkeypatch):
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MAX_CYCLE_TOKENS", "2048")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_CYCLE_COPIES", "1")
    monkeypatch.setenv("MTPLX_REPETITION_STOP_MIN_CYCLE_SPAN_TOKENS", "300")
    config = _repetition_stop_config(True)
    assert config.max_cycle_tokens == 2048
    assert config.min_cycle_copies == 2  # floor: one copy is not a loop
    assert config.min_cycle_span_tokens == 300


# ---------------------------------------------------------------------------
# Synthetic sequences.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("period", [97, 129, 170, 171, 500, 1323, 4096, 8192])
def test_exact_cycle_fires_at_three_copies_spanning_512_tokens(period):
    config = _repetition_stop_config(True)
    need = _need(period, config)
    fire_len, result = _feed(_loop(period, copies=6), config)
    # The run starts with the second copy; it fires the token it reaches need.
    assert fire_len == PREFIX + period + need
    assert result.block_tokens == period
    assert result.repeats == (need + period) // period
    assert result.repeats >= 3
    assert result.reason == "long_cycle"
    # Nothing is trimmed: the copies are already on the wire.
    assert result.repeated_tokens == 0
    assert result.trim_start == fire_len


def test_cycle_with_internal_repeats_fires_at_its_true_period():
    # Each copy holds the same 150-token block twice, so windows inside the
    # second block recur 151 tokens back before they recur a period back.
    config = _repetition_stop_config(True)
    block = list(range(10, 160))
    cycle = block + [7] + block + [8]
    tokens = list(range(1_000_000, 1_000_000 + PREFIX)) + cycle * 5
    period = len(cycle)
    fire_len, result = _feed(tokens, config)
    assert result.block_tokens == period == 302
    assert fire_len == PREFIX + period + _need(period, config)


@pytest.mark.parametrize("period", [150, 1323])
def test_near_cycle_with_one_differing_token_per_copy_never_fires(period):
    config = _repetition_stop_config(True)
    cycle = list(range(1, period + 1))
    tokens = list(range(1_000_000, 1_000_000 + PREFIX))
    for copy in range(12):
        near = list(cycle)
        near[(copy * 37) % period] = 500_000 + copy  # a counter-like change
        tokens.extend(near)
    assert _feed(tokens, config) == (None, None)


def test_third_copy_off_by_its_last_token_does_not_fire():
    config = _repetition_stop_config(True)
    period = 400
    cycle = list(range(1, period + 1))
    tokens = (
        list(range(1_000_000, 1_000_000 + PREFIX))
        + cycle * 2
        + cycle[:-1]
        + [999_999]
        + list(range(2_000_000, 2_000_600))
    )
    assert _feed(tokens, config) == (None, None)


def test_period_beyond_the_cap_does_not_fire():
    config = _repetition_stop_config(True)
    period = config.max_cycle_tokens + 1
    detector = _long_cycle_stop(config)
    assert detector.observe(_loop(period, copies=4)) is None


def test_short_periods_stay_with_the_short_block_stop():
    config = _repetition_stop_config(True)
    tokens = _loop(50, copies=40)
    assert _long_cycle_stop(config).observe(tokens) is None
    # The #311 rule owns this loop and still trims every copy.
    trimmed = list(tokens)
    result = _trim_repeated_suffix(trimmed, config, _long_cycle_stop(config))
    assert result.reason == "exact_repeated_token_suffix"
    assert result.block_tokens == 50
    assert result.repeated_tokens > 0
    assert len(trimmed) == result.trim_start


def test_never_fires_before_min_tokens():
    config = _repetition_stop_config(True)
    period = 200
    # Periodic from the first token: three copies are complete at 600 tokens,
    # but the stop arms at min_tokens (768).
    fire_len, result = _feed(list(range(1, period + 1)) * 6, config)
    assert fire_len == config.min_tokens
    assert result.block_tokens == period


@pytest.mark.parametrize("chunk", [3, 7, 24])
def test_commit_batching_keeps_the_verdict(chunk):
    config = _repetition_stop_config(True)
    tokens = _loop(333, copies=6)
    serial_len, serial = _feed(tokens, config)
    batched_len, batched = _feed(tokens, config, chunk=chunk)
    assert (batched.block_tokens, batched.repeats) == (
        serial.block_tokens,
        serial.repeats,
    )
    # A speculative round reports within the commit that crossed the point.
    assert serial_len <= batched_len < serial_len + chunk


def test_state_is_bounded_by_the_period_cap():
    config = RepetitionStopConfig(enabled=True, max_cycle_tokens=1024)
    detector = _long_cycle_stop(config)
    # 40k tokens with no exact repeat of any 32-token window.
    tokens = [(i * 7919) % 104_729 for i in range(40_000)]
    assert detector.observe(tokens) is None
    assert len(detector._ring) == 1025
    assert len(detector._last) <= 1025


def test_numpy_token_ids_hash_like_python_ints():
    # Some commit paths can hand numpy scalars to the token list; the rolling
    # hash must not overflow on them.
    config = _repetition_stop_config(True)
    tokens = _loop(250, copies=5)
    as_numpy = [np.int64(token) for token in tokens]
    assert _feed(as_numpy, config) == _feed(tokens, config)
    assert _feed(tokens, config)[1].block_tokens == 250


def test_rewound_token_list_rebuilds_the_rolling_state():
    config = _repetition_stop_config(True)
    tokens = _loop(300, copies=6)
    detector = _long_cycle_stop(config)
    assert detector.observe(tokens[:1500]) is None
    # A shorter list than the detector has seen: state restarts from it.
    assert detector.observe(tokens[:200]) is None
    result = detector.observe(tokens)
    assert result is not None
    assert result.block_tokens == 300


# ---------------------------------------------------------------------------
# The two real loops, replayed from compact token-id fixtures.
# ---------------------------------------------------------------------------


def _real_loops():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["loops"]


@pytest.mark.parametrize(
    ("name", "limit"),
    [
        # Fire within these many tokens of the first exact copy.
        ("mimo_1323", 4500),
        ("mimo_129", 600),
    ],
)
def test_real_mimo_loops_fire_at_three_copies(name, limit):
    loop = _real_loops()[name]
    tokens = loop["tokens"]
    period, onset = loop["period"], loop["onset"]
    config = _repetition_stop_config(True)
    detector = _long_cycle_stop(config)
    committed: list[int] = []
    result = None
    for token in tokens:
        committed.append(token)
        result = _trim_repeated_suffix(committed, config, detector)
        if result is not None:
            break
    assert result is not None
    assert result.reason == "long_cycle"
    assert result.block_tokens == period
    assert result.repeats == 3
    assert result.repeated_tokens == 0
    assert len(committed) == onset + period + _need(period, config)
    assert len(committed) - onset <= limit
    # The short-block stop alone never sees these loops (the 09-22 miss).
    assert _detect_repeated_token_suffix(committed, config) is None
    assert _detect_repeated_token_suffix(list(tokens), config) is None


@pytest.mark.parametrize("name", ["mimo_1323", "mimo_129"])
def test_real_mimo_loops_fire_under_speculative_commits(name):
    loop = _real_loops()[name]
    config = _repetition_stop_config(True)
    serial_len, serial = _feed(loop["tokens"], config)
    batched_len, batched = _feed(loop["tokens"], config, chunk=3)
    assert batched.block_tokens == serial.block_tokens == loop["period"]
    assert batched.repeats == serial.repeats == 3
    assert serial_len <= batched_len < serial_len + 3
