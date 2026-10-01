"""The decode round overlaps host work with the GPU's draft steps without changing a byte.

CPU-only: the sidecar half runs the SHIPPED methods against real memmaps (the
lookahead tests' stand-in), the aux half runs the shipped row arithmetic, and
the loop half pins the wiring in the generation source.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from mtplx import qwen4_draft_device_chain as chain
from tests.test_qwen4_ple_prefill_lookahead import (
    FakeSidecar,
    _bind_method,
    _bind_top_level,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATION_TEXT = (ROOT / "mtplx" / "generation.py").read_text("utf-8")
FIXED_VERIFY_TEXT = (ROOT / "mtplx" / "qwen4_fixed_verify.py").read_text("utf-8")
GRAPHBANK_TEXT = (ROOT / "mtplx" / "graphbank.py").read_text("utf-8")

NAMES = ("weight", "scales", "biases")


class HotSidecar(FakeSidecar):
    """The stand-in plus the hot-row cache state `_rows_matrices` uses."""

    def __init__(self, tmp_path, rows=50_000, hot_cap_rows=1000):
        super().__init__(tmp_path, rows, hot_cap_rows=hot_cap_rows)
        self._hot = OrderedDict()
        self.hot_hits = 0
        self.hot_misses = 0

    _rows_matrices = _bind_method("_SidecarGather", "_rows_matrices")
    _warm = _bind_method("_SidecarGather", "_warm")
    _submit_warm = _bind_method("_SidecarGather", "_submit_warm")
    prefetch_rows_np = _bind_method("_SidecarGather", "prefetch_rows_np")


@pytest.fixture
def sidecar(tmp_path):
    sc = HotSidecar(tmp_path)
    try:
        yield sc
    finally:
        sc.close()


def test_a_prefetched_gather_returns_the_same_bytes_and_reads_nothing(sidecar, tmp_path):
    rng = np.random.default_rng(7)
    flat = rng.integers(0, 50_000, size=64, dtype=np.int64)
    cold = HotSidecar(tmp_path / "cold") if (tmp_path / "cold").mkdir() is None else None
    try:
        want = cold._rows_matrices(flat, NAMES)
    finally:
        cold.close()

    fetched = sidecar.prefetch_rows_np(flat[:16])
    assert fetched == len(np.unique(flat[:16]))
    fetched += sidecar.prefetch_rows_np(flat[:32])
    fetched += sidecar.prefetch_rows_np(flat)
    assert fetched == len(np.unique(flat))
    # The ledger stays a statement about gathers, not about prefetches.
    assert (sidecar.hot_hits, sidecar.hot_misses) == (0, 0)
    assert sidecar.prefetched_rows == fetched

    got = sidecar._rows_matrices(flat, NAMES)
    assert sidecar.hot_misses == 0 and sidecar.hot_hits == len(np.unique(flat))
    for name in NAMES:
        assert got[name].dtype == want[name].dtype
        assert np.array_equal(got[name], want[name])


def test_prefetch_is_inert_without_a_hot_cache_or_for_large_requests(tmp_path):
    off = HotSidecar(tmp_path, hot_cap_rows=0)
    try:
        assert off.prefetch_rows_np(np.arange(8)) == 0
        assert not off._hot
        assert off.prefetch_rows_np(np.arange(0)) == 0
    finally:
        off.close()
    (tmp_path / "big").mkdir()
    big = HotSidecar(tmp_path / "big")
    try:
        assert big.prefetch_rows_np(np.arange(big._HOT_PATH_MAX_ROWS + 1)) == 0
        assert not big._hot
    finally:
        big.close()


def _aux(prefetch):
    """The shipped `_FixedM4SidecarAux` over the shipped row arithmetic."""

    import ast

    namespace: dict = {"np": np, "mx": None}
    tree = ast.parse(FIXED_VERIFY_TEXT)
    wanted = {"_FixedM4SidecarAux", "_fixed_m4_previous_tokens"}
    body = [n for n in tree.body if getattr(n, "name", None) in wanted]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<qwen4_fixed_verify>", "exec"), namespace)
    rows = partial(
        _bind_top_level("_ngram_rows_np"),
        mult=np.asarray([1_000_003, 998_244_353, 1_000_000_007], dtype=np.int64),
        sizes=np.arange(10_007, 10_007 + 16, dtype=np.int64),
        offs=np.arange(16, dtype=np.int64) * 20_000,
        eos=248_044,
        ngram_size=3,
        heads_per_ngram=8,
    )
    return (
        namespace["_FixedM4SidecarAux"](
            prompt_tail=(11, 12),
            rows=rows,
            gather=None,
            output_dim=2560,
            prefetch=prefetch,
        ),
        rows,
    )


def test_a_window_prefix_prefetches_the_first_rows_of_the_whole_window():
    seen: list[np.ndarray] = []

    def prefetch(flat):
        seen.append(np.asarray(flat).copy())
        return len(flat)

    aux, rows = _aux(prefetch)
    completion = [21, 22, 23, 24, 25]
    committed = len(completion) - 1
    window = [25, 31, 32, 33]
    full, _ = rows(
        np.asarray((window,), dtype=np.int64),
        np.asarray(((completion[committed - 2], completion[committed - 1]),), dtype=np.int64),
    )
    full = full.reshape(4, 16)
    for take in (1, 2, 3, 4):
        assert aux.prefetch(window[:take], completion, committed) == 16 * take
        assert np.array_equal(seen[-1].reshape(take, 16), full[:take])
    # Nothing to do, nothing done.
    assert aux.prefetch([], completion, committed) == 0
    inert, _ = _aux(None)
    assert inert.prefetch(window, completion, committed) == 0


def test_the_switch_defaults_on_and_turns_off():
    assert chain.pipeline_enabled({}) is True
    for value in ("0", "off", "false", "no"):
        assert chain.pipeline_enabled({"MTPLX_QWEN4_DRAFT_CHAIN_PIPELINE": value}) is False
    assert chain.pipeline_enabled({"MTPLX_QWEN4_DRAFT_CHAIN_PIPELINE": "1"}) is True


def test_the_loop_hands_each_depth_over_and_reads_depth_by_depth():
    start = GENERATION_TEXT.index("_sc_pipeline = _qwen4_sampled_chain.pipeline_enabled()")
    body = GENERATION_TEXT[start : GENERATION_TEXT.index("_sampled_chain_rounds += 1", start)]
    # Each depth is scheduled inside the build loop, never after it.
    assert "mx.async_eval(*_sc_support, _sc_token, _sc_hidden)" in body
    # The single evaluation survives as the switch-off path.
    assert body.count("*_sc_predicted,") == 1
    # Depth d is awaited alone, so depth d+1 keeps running under the host read.
    assert "_eval(*_sc_supports[_sc_depth], _sc_predicted[_sc_depth])" in body
    # The window prefix is warmed with the verify call's own ledger arguments.
    assert body.count("completion_tokens=tokens,") == 2
    assert body.count("committed_count=len(tokens) - 1,") == 2
    verify = GENERATION_TEXT[GENERATION_TEXT.index("compiled_verify_bank.forward_fixed_m4(") :][:400]
    assert "completion_tokens=tokens," in verify
    assert "committed_count=len(tokens) - 1," in verify


def test_the_bank_never_lets_a_prefetch_failure_reach_the_decode_loop():
    start = GRAPHBANK_TEXT.index("def prefetch_fixed_m4_aux(")
    body = GRAPHBANK_TEXT[start : GRAPHBANK_TEXT.index("def reserve_fixed_m4_window(", start)]
    assert "except Exception as error:" in body
    assert 'self.stats["fixed_m4_aux_prefetch_error"]' in body
    assert body.index("except Exception") < body.index("return 0", body.index("except Exception"))


# --------------------------------------------------------------------------
# The warm pass reads only the maps that are not already in core
# --------------------------------------------------------------------------


def test_the_warm_pass_reads_only_the_named_maps(sidecar, monkeypatch):
    import os as _os

    reads: list[tuple[int, int]] = []
    real_pread = _os.pread

    def counting_pread(fd, length, offset):
        reads.append((length, offset))
        return real_pread(fd, length, offset)

    monkeypatch.setattr(_os, "pread", counting_pread)
    rows = np.arange(0, 640, 5, dtype=np.int64)
    sidecar._warm(rows)
    every_map = len(reads)
    assert every_map == 3 * len(rows)
    reads.clear()
    sidecar._warm(rows, only=["weight"])
    assert len(reads) == len(rows)
    assert {length for length, _ in reads} == {sidecar._row_meta[0][1]}
    reads.clear()
    sidecar._warm(rows, only=["scales", "biases"])
    assert len(reads) == 2 * len(rows)


def test_cold_map_names_reports_per_map_and_declines_when_unknown(monkeypatch):
    from mtplx import ple_row_gather as row_gather

    answers = {"weight": 0.31, "scales": 1.0, "biases": 0.995}
    monkeypatch.setattr(
        row_gather,
        "resident_fraction",
        lambda memmap, rows, sample=256: answers[memmap],
    )
    maps = {name: name for name in answers}
    assert row_gather.cold_map_names(maps, [1, 2, 3]) == ["weight"]
    answers["weight"] = 1.0
    assert row_gather.cold_map_names(maps, [1, 2, 3]) == []
    answers["scales"] = None
    assert row_gather.cold_map_names(maps, [1, 2, 3]) is None
