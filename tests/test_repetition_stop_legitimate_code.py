"""Why the repetition stops ship off (2.12.0).

Exact repetition is not proof of a loop: legitimate code repeats exactly.
Armed, the short-block stop deletes a Tetris board literal and a patch that
renames a type at 15 identical call sites (it removes every copy and ends
the turn), and the long-cycle stop ends a platformer map with 8 identical
empty rows. So the server arms neither unless MTPLX_REPETITION_STOP=1 (see
``_repetition_stop_enabled`` in mtplx/server/openai.py). If a detector
change stops the armed cases below from firing, this fixture is the
evidence for revisiting the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.generation import (
    _detect_repeated_token_suffix,
    _long_cycle_stop,
    _repetition_stop_config,
    _trim_repeated_suffix,
)
from mtplx.server.openai import _repetition_stop_enabled

FIXTURE = Path(__file__).parent / "fixtures" / "repetition_legitimate_code.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.fixture(autouse=True)
def _default_repetition_env(monkeypatch):
    for name in (
        "MTPLX_REPETITION_STOP",
        "MTPLX_UNCAPPED_REPETITION_STOP",
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


def _first_fire(ids: list[int], enabled: bool):
    """Replay a decode loop's per-commit check; (kind, length, result)."""
    config = _repetition_stop_config(enabled)
    long_cycle = _long_cycle_stop(config)
    committed: list[int] = []
    for token in ids:
        committed.append(token)
        short = _detect_repeated_token_suffix(committed, config)
        if short is not None:
            return "short_block", len(committed), short
        if long_cycle is not None:
            result = long_cycle.observe(committed)
            if result is not None:
                return "long_cycle", len(committed), result
    return None, None, None


def test_fixture_is_the_qwen_tokenization_of_its_text():
    assert set(CASES) == {"tetris_board", "platformer_map", "rename_patch"}
    board = CASES["tetris_board"]["text"]
    assert board.count("  [0,0,0,0,0,0,0,0,0,0],\n") == 20
    patch = CASES["rename_patch"]["text"]
    assert patch.count("TileCache::new(TILE_CACHE_CAPACITY)") == 16
    assert patch.rstrip().endswith("</tool_call>")


@pytest.mark.parametrize("name", sorted(CASES))
def test_server_default_leaves_legitimate_repeated_code_untouched(name):
    ids = list(CASES[name]["ids"])
    assert _repetition_stop_enabled({}) is False
    config = _repetition_stop_config(_repetition_stop_enabled({}))
    long_cycle = _long_cycle_stop(config)
    assert long_cycle is None
    committed: list[int] = []
    for token in ids:
        committed.append(token)
        assert _trim_repeated_suffix(committed, config, long_cycle) is None
    assert committed == ids


@pytest.mark.parametrize(
    ("name", "kind", "fire_at", "block", "copies"),
    [
        # The board's 23-token row, 9 copies in: the stop would delete all 9
        # rows and end the answer inside the literal.
        ("tetris_board", "short_block", 2430, 23, 9),
        # A 123-token empty map row, 4 copies in.
        ("platformer_map", "long_cycle", 2735, 123, 4),
        # The repeated 34-token hunk, 6 copies in: the stop would delete all
        # 6 hunks and end the turn, so the tool call never closes.
        ("rename_patch", "short_block", 2180, 34, 6),
    ],
)
def test_armed_stops_cut_legitimate_repeated_code(name, kind, fire_at, block, copies):
    fired_kind, length, result = _first_fire(list(CASES[name]["ids"]), True)
    assert (fired_kind, length) == (kind, fire_at)
    assert result.block_tokens == block
    assert result.repeats == copies
