"""Host-side regressions without importing MLX or initializing Metal.

Execute the production functions extracted from their AST, with NumPy doing
the comparator's array arithmetic. These tests prove the admission and report
logic, not compiled execution or the MLX/Metal numerical lane.
"""

from __future__ import annotations

import ast
import contextlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), path, "exec"), namespace)
    return {name: namespace[name] for name in names}


@pytest.mark.parametrize("rebase,refusal", [("64", "vision_state_rebase"), (None, None), ("0", None)])
def test_sequential_image_admission_checks_rebase_on_cpu(monkeypatch, rebase, refusal):
    splice_module = ModuleType("mtplx.vision.splice")
    splice_module.__dict__.update(_functions(
        "mtplx/vision/splice.py", ["mrope_rope_state"], {},
    ))
    monkeypatch.setitem(sys.modules, "mtplx.vision.splice", splice_module)
    monkeypatch.delenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_STATE_REBASE_EVERY", raising=False)
    if rebase is not None:
        monkeypatch.setenv("MTPLX_STATE_REBASE_EVERY", rebase)
    ns = {"os": os, "env_bool": lambda name, default: os.environ.get(name, "1") != "0"}
    functions = _functions("mtplx/generation.py", [
        "_qwen4_vision_compiled_verify_admission", "_vision_image_count",
        "_dense_mrope_state_of", "_env_int",
    ], ns)
    splice = SimpleNamespace(image_pad_token_id=900, pad_counts=(1,),
                             mrope_table=None, mrope_delta=0, dense_mrope=None)
    verdict = functions["_qwen4_vision_compiled_verify_admission"](None, splice, [1, 900])
    assert verdict == {"positions": "vision_sequential", "rope_delta": None,
                       "images": 1, "refusal": refusal}


class _ArrayType(type):
    def __instancecheck__(cls, value):
        return isinstance(value, np.ndarray)

    def __call__(cls, *args, **kwargs):
        return np.array(*args, **kwargs)


class _Array(metaclass=_ArrayType):
    pass


@pytest.fixture
def bank():
    mx = SimpleNamespace(
        array=_Array, float32=np.float32, abs=np.abs, max=np.max, sum=np.sum,
        exp=np.exp, logsumexp=np.logaddexp.reduce, eval=lambda *_args: None,
        int64=np.int64, integer=np.integer, issubdtype=np.issubdtype,
    )
    ns = {"mx": mx, "attention_phase": lambda _: contextlib.nullcontext(),
          "VERIFY_SPEC_KIND_QSA": "qsa"}
    names = ["_fixed_m4_parity2_record", "_fixed_m4_parity2_compare",
             "_artifact_kind", "_decode_length"]
    methods = _functions("mtplx/graphbank.py", names, ns)
    instance = type("Comparator", (), {name: methods[name] for name in names[:2]})()
    instance.stats = {"calls": 1, "parity2_calls": 0, "parity2_divergent_calls": 0}
    instance._spec = [(0, "qsa", 5), (1, "gdn", 10)]
    instance._named_eager_captures = lambda value: value
    return instance


def _state(*, offset=12, capacity=12):
    return [np.zeros((1, 2, capacity, 3), dtype=np.float32),
            np.zeros((1, 2, capacity, 3), dtype=np.float32),
            np.array([offset], dtype=np.int32),
            np.zeros((1, capacity, 3), dtype=np.float32),
            np.zeros((1, capacity // 4, 3), dtype=np.float32),
            *[np.zeros((2, 3), dtype=np.float32) for _ in range(10)]]


def _compare(bank, reference, candidate, *, kind="compiled", width=4):
    clone = [SimpleNamespace(
        kv=SimpleNamespace(keys=reference[0], values=reference[1], offset=int(reference[2].item())),
        raw_keys=reference[3], pooled=reference[4], ratio=4,
    ), SimpleNamespace(cache=reference[5:])]
    logits = np.zeros((1, width, 8), dtype=np.float32)
    hidden = np.zeros((1, width, 4), dtype=np.float32)
    captures = {"capture[1].conv": np.zeros((1, width, 2), dtype=np.float32)}
    bank._runtime_forward = lambda *args, **kwargs: (logits, hidden, captures)
    bank._fixed_m4_parity2_compare(
        {"rope_delta": -12, "hidden_variant": "post_norm", "base_offset": 8},
        clone, np.zeros((1, width), dtype=np.int32), compiled_aux=None,
        candidate_logits=logits.copy(), candidate_hidden=hidden.copy(),
        candidate_captures={key: value.copy() for key, value in captures.items()},
        candidate_state=candidate, committed_count=0, dispatch_kind=kind,
    )
    return bank.stats["fixed_m4_parity2"]


@pytest.mark.parametrize("side", ["reference", "candidate", "both"])
@pytest.mark.parametrize("leaf,axis", [(0, 2), (1, 2), (3, 1), (4, 1)])
def test_storage_must_cover_each_sides_live_rows(bank, side, leaf, axis):
    ref, cand = _state(), _state()
    for state, label in ((ref, "reference"), (cand, "candidate")):
        if side in (label, "both"):
            index = [slice(None)] * state[leaf].ndim
            index[axis] = slice(None, -1)
            state[leaf] = state[leaf][tuple(index)]
    record = _compare(bank, ref, cand)
    name = f"state[0:qsa].{leaf}"
    assert record["divergent_rounds"] == 1
    assert record["first_divergence"]["leaf"] == name
    assert record["first_divergence"]["reason"] == "live_storage_shortfall"
    metric = record["round_diagnostics"][0]["leaves"][name]
    assert metric["reason"] == "live_storage_shortfall"
    for label in ("reference", "candidate"):
        assert metric[f"{label}_live_rows"] == (3 if leaf == 4 else 12)
    assert metric["max_abs_diff"] is None and metric["differing_elements"] is None


@pytest.mark.parametrize("offset", [3, 12, 15])
def test_unused_capacity_and_pooled_remainders_do_not_diverge(bank, offset):
    ref, cand = _state(offset=offset, capacity=16), _state(offset=offset, capacity=24)
    for leaf, axis in ((0, 2), (1, 2), (3, 1), (4, 1)):
        index = [slice(None)] * cand[leaf].ndim
        index[axis] = slice(offset // 4 if leaf == 4 else offset, None)
        cand[leaf][tuple(index)] = 99  # unused rows need not match
    record = _compare(bank, ref, cand)
    assert record["divergent_rounds"] == 0
    assert record["compiled_exact_rounds"] == 1
    assert all(m["max_abs_diff"] == 0 for m in record["round_diagnostics"][0]["leaves"].values())


@pytest.mark.parametrize("leaf,axis", [(0, 2), (1, 2), (3, 1), (4, 1)])
def test_the_last_live_row_is_compared(bank, leaf, axis):
    ref, cand = _state(), _state(capacity=20)
    index = [0] * cand[leaf].ndim
    index[axis] = (12 // 4 if leaf == 4 else 12) - 1
    cand[leaf][tuple(index)] = 2
    record = _compare(bank, ref, cand)
    metric = record["round_diagnostics"][0]["leaves"][f"state[0:qsa].{leaf}"]
    assert record["divergent_rounds"] == 1
    assert metric["max_abs_diff"] == 2 and metric["differing_elements"] == 1


def test_storage_is_checked_against_the_candidates_own_offset(bank):
    record = _compare(bank, _state(), _state(offset=13, capacity=12))
    leaves = record["round_diagnostics"][0]["leaves"]
    assert leaves["state[0:qsa].0"]["reason"] == "live_storage_shortfall"
    assert leaves["state[0:qsa].2"] == {
        "max_abs_diff": 1, "differing_elements": 1, "reason": "offset_mismatch",
    }


def test_all_leaves_of_all_rounds_have_diagnostics_and_kl_is_logits_only(bank):
    ref, cand = _state(), _state()
    first = _compare(bank, ref, cand)
    assert first["compiled_exact_rounds"] == 1
    for leaf in cand[5:]:
        leaf[:] = 1  # ten state mismatches, beyond the previous eight-leaf cap
    _compare(bank, ref, cand, kind="eager")  # even width 4 can be eager
    cand[-1][:] = 3
    record = _compare(bank, ref, cand)
    assert record["rounds"] == 3 and record["divergent_rounds"] == 2
    assert record["compiled_rounds"] == 2 and record["eager_rounds"] == 1
    assert record["compiled_exact_rounds"] == 1 and record["eager_exact_rounds"] == 0
    assert len(record["first_divergence"]["report"]) == 10
    assert record["kl_scope"] == "logits_only"
    assert record["state_metrics"] == ["max_abs_diff", "differing_elements"]
    assert len(record["round_diagnostics"]) == 3
    for round_number, diagnostic in enumerate(record["round_diagnostics"], 1):
        assert diagnostic["round"] == round_number
        leaves = diagnostic["leaves"]
        assert len(leaves) == record["compared_leaves_per_round"] == 18
        assert {name for name, metric in leaves.items() if "max_kl" in metric} == {"logits"}
        assert leaves["state[1:gdn].9"]["max_abs_diff"] == (0, 1, 3)[round_number - 1]
        assert leaves["state[1:gdn].9"]["differing_elements"] == (0, 6, 6)[round_number - 1]


def test_integer_state_differences_are_measured_without_float32_rounding(bank):
    ref, cand = _state(), _state()
    ref[5] = np.array([2**24, -(2**31)], dtype=np.int32)
    cand[5] = np.array([2**24 + 1, 2**31 - 1], dtype=np.int32)
    record = _compare(bank, ref, cand)
    metric = record["round_diagnostics"][0]["leaves"]["state[1:gdn].0"]
    assert metric == {"max_abs_diff": 2**32 - 1, "differing_elements": 2,
                      "reason": "value_mismatch"}


def test_dispatch_sites_label_actual_execution_not_window_width():
    tree = ast.parse((ROOT / "mtplx/graphbank.py").read_text())
    sites = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        if function.name not in ("_forward_installed_fixed_m4", "forward_ar_capture"):
            continue
        for call in ast.walk(function):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                if call.func.attr in ("_fixed_m4_parity2_round", "_fixed_m4_parity2_compare"):
                    (keyword,) = [k for k in call.keywords if k.arg == "dispatch_kind"]
                    sites.append((function.name, ast.literal_eval(keyword.value)))
    assert sites == [("_forward_installed_fixed_m4", "compiled"), ("forward_ar_capture", "eager")]
