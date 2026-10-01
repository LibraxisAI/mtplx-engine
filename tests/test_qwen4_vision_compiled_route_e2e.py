"""An image request through the whole generation loop on the compiled verifier.

The tiny random pack of ``scripts/qwen4exp_mtp_tiny_smoke.py`` (trunk and MTP
head, no artifact), given the two things the compiled fixed-M4 lane needs: one
PLE layer and an M-RoPE contract. bfloat16 on the GPU, the served dtype on the
served device, where the compiled verify step and the eager forward agree bit
for bit, so every comparison here is exact. Sampled at the family's native
settings with a fixed seed: greedy-only evidence is refused in this project.

Both arms use the production position scopes: prefill owns its scope and
_decode_trunk_scope covers each decode trunk forward only. No fixture supplies
an ambient scope that could hide a missing production scope or shift drafts.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
from functools import wraps
from pathlib import Path

import mlx.core as mx
import mlx.utils
import pytest

import mtplx.generation as generation
import mtplx.graphbank as graphbank
from mtplx import demotions
from mtplx.attention_context import current_attention_phase, vision_rope_state
from mtplx.sampling import SamplerConfig
from mtplx.vision.mrope import build_mrope_positions
from mtplx.vision.splice import VisionSplice

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="bfloat16 expert gathers need the GPU; the CPU has no exact full-model lane",
)

_SMOKE = Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_mtp_tiny_smoke.py"
PAD = 120
IMAGE_TOKENS = 16
GRID = (1, 8, 8)
NATIVE = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
MAX_TOKENS = 40
SEED = 1234


def _smoke():
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_tiny_smoke", _SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pack():
    """The tiny pack in bf16 plus a synthetic image (rows the tower would emit)."""

    import mlx_lm.models.cache as cache_module

    import mtplx.models.qwen4_exp as qwen4_exp
    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support

    smoke = _smoke()
    prev = mx.default_device()
    mx.set_default_device(mx.gpu)
    # A runtime load earlier in the session swaps mlx-lm's ArraysCache for the
    # leak-free class (mtplx/arrays_cache_patch.py); the verify bank looks the
    # class up when it is called, so the model must build the same one.
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.random.seed(0)
    args = dataclasses.replace(
        smoke._tiny_text_args(),
        head_dim=32,
        indexer_head_dim=32,
        indexer_compress_ratio=4,
        ple_layer_ids=[1],
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=dataclasses.asdict(args)))
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    model.update(
        mlx.utils.tree_map(
            lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
            model.parameters(),
        )
    )
    model.eval()
    mx.eval(model.parameters())
    assert validate_mtp_support(model)
    mx.random.seed(99)
    rows = mx.random.normal((IMAGE_TOKENS, args.hidden_size)).astype(mx.bfloat16)
    mx.eval(rows)
    yield smoke, model, rows
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    # One request's route must not depend on what the shell exported.
    import os

    for name in tuple(os.environ):
        if name.startswith("MTPLX_QWEN4_") or name in {
            "MTPLX_COMPILED_VERIFY",
            "MTPLX_STATE_REBASE_EVERY",
            "MTPLX_FAMILY_CAPTURE_COMMIT",
        }:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    demotions.reset()
    yield
    demotions.reset()


def _image_prompt(tail: int):
    ids = [3, 5, 7, 9, 11, 13] + [PAD] * IMAGE_TOKENS + list(range(20, 20 + tail))
    table, delta = build_mrope_positions(
        ids, image_token_id=PAD, image_grids=[GRID], spatial_merge_size=2
    )
    return ids, mx.array(table), int(delta)


def _generate(pack, monkeypatch, *, mode, ids, table=None, delta=None):
    """One request. ``mode`` is MTPLX_COMPILED_VERIFY: 0 (eager), 1 or parity2."""

    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    smoke, model, rows = pack
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", mode)
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    splice = (
        VisionSplice(
            image_pad_token_id=PAD,
            embeddings=rows,
            image_digests=(1,),
            pad_counts=(IMAGE_TOKENS,),
            image_grids=(GRID,),
            mrope_table=table,
            mrope_delta=delta,
        )
        if table is not None
        else None
    )
    return generation.generate_mtpk(
        rt,
        list(ids),
        max_tokens=MAX_TOKENS,
        sampler=NATIVE,
        draft_sampler=NATIVE,
        speculative_depth=3,
        seed=SEED,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
        vision_splice=splice,
    )


def _bank_report(result):
    return (result.stats.graphbank or {}).get("compiled_verify") or {}


@pytest.mark.parametrize("mode", ["0", "1", "parity2"])
def test_production_decode_scopes_cover_trunk_forwards_only(pack, monkeypatch, mode):
    ids, table, delta = _image_prompt(18)
    real_scope = generation._decode_trunk_scope
    real_draft = pack[1].mtp_forward
    trunk_scopes, draft_calls = [], []

    @contextlib.contextmanager
    def observed_scope(splice):
        assert vision_rope_state() is None  # no fixture or leaked outer scope
        with real_scope(splice):
            active_table, active_delta = vision_rope_state()
            assert active_table is table and active_delta == delta
            assert current_attention_phase() == "decode_verify"
            trunk_scopes.append(True)
            yield
        assert vision_rope_state() is None

    @wraps(real_draft)
    def observed_draft(*args, **kwargs):
        if current_attention_phase() != "prefill":
            assert vision_rope_state() is None
            assert current_attention_phase() != "decode_verify"
            draft_calls.append(True)
        return real_draft(*args, **kwargs)

    monkeypatch.setattr(generation, "_decode_trunk_scope", observed_scope)
    monkeypatch.setattr(pack[1], "mtp_forward", observed_draft)
    _generate(pack, monkeypatch, mode=mode, ids=ids, table=table, delta=delta)
    assert trunk_scopes and draft_calls
    assert vision_rope_state() is None


@pytest.mark.parametrize("tail", [18, 17])  # prompt lengths 40 and 39: n % 4 of 0 and 3
def test_an_image_request_decodes_the_same_tokens_on_the_compiled_route(
    pack, monkeypatch, tail
):
    ids, table, delta = _image_prompt(tail)
    eager = _generate(pack, monkeypatch, mode="0", ids=ids, table=table, delta=delta)
    compiled = _generate(pack, monkeypatch, mode="1", ids=ids, table=table, delta=delta)
    assert len(eager.tokens) == MAX_TOKENS
    assert compiled.tokens == eager.tokens

    # It really took the compiled route, and the request record says how.
    admission = compiled.stats.fixed_m4_admission
    assert admission["positions"] == "vision_delta"
    assert admission["rope_delta"] == delta == 4 - IMAGE_TOKENS
    assert admission["images"] == 1
    assert admission["engaged"] is True and admission["reason"] == "admitted"
    bank = _bank_report(compiled)
    assert bank["fixed_m4"]["rope_delta_input"] is True
    assert bank["fixed_m4"]["rope_delta"] == delta
    (key,) = bank["compiled_keys"]  # one trace, and it takes the delta
    assert key.startswith("m4:") and key.endswith(":rope_delta")
    assert bank["compiled_calls"] >= 8 and bank["fallback_calls"] == 0
    assert "vision_request_eager_verify" not in compiled.stats.demotions
    # The eager arm says why it was eager, with the same position fields.
    assert eager.stats.fixed_m4_admission["reason"] == "compiled_verify_off"
    assert eager.stats.fixed_m4_admission["positions"] == "vision_delta"


def test_a_text_request_is_untouched(pack, monkeypatch):
    ids = [3, 5, 7, 9, 11, 13] + list(range(20, 54))
    eager = _generate(pack, monkeypatch, mode="0", ids=ids)
    compiled = _generate(pack, monkeypatch, mode="1", ids=ids)
    assert compiled.tokens == eager.tokens
    assert compiled.stats.fixed_m4_admission["positions"] == "text"
    assert compiled.stats.fixed_m4_admission["rope_delta"] is None
    bank = _bank_report(compiled)
    assert bank["fixed_m4"]["rope_delta_input"] is False
    (key,) = bank["compiled_keys"]  # the text trace: no delta input
    assert key.startswith("m4:") and "rope_delta" not in key


def test_parity2_checks_every_round_of_an_image_request_and_finds_nothing(
    pack, monkeypatch
):
    ids, table, delta = _image_prompt(18)
    compiled = _generate(pack, monkeypatch, mode="1", ids=ids, table=table, delta=delta)
    checked = _generate(
        pack, monkeypatch, mode="parity2", ids=ids, table=table, delta=delta
    )
    # The compiled lane stays authoritative under the instrument.
    assert checked.tokens == compiled.tokens
    bank = _bank_report(checked)
    record = bank["fixed_m4_parity2"]
    assert record["positions"] == "vision_delta" and record["rope_delta"] == delta
    assert record["rounds"] == bank["calls"] >= 8
    assert record["rounds_by_width"].get("4", 0) >= 8
    assert record["reference_scope_missing_rounds"] == 0
    assert record["reference_scope_missing_by_width"] == {}
    assert record["divergent_rounds"] == 0 and record["first_divergence"] is None
    # Not "close": equal. bf16 on the GPU is the exact lane.
    for name in (
        "logits_max_abs_diff",
        "logits_max_kl",
        "hidden_max_abs_diff",
        "state_max_abs_diff",
        "capture_max_abs_diff",
    ):
        assert record[name] == 0.0, name
    assert bank["parity2_calls"] == record["rounds"]
    assert bank["parity2_divergent_calls"] == 0
    assert record["compiled_rounds"] == bank["compiled_calls"] > 0
    assert record["compiled_exact_rounds"] == record["compiled_rounds"]
    assert record["compiled_rounds"] + record["eager_rounds"] == record["rounds"]
    assert len(record["round_diagnostics"]) == record["rounds"]


def test_parity2_on_a_text_request(pack, monkeypatch):
    ids = [3, 5, 7, 9, 11, 13] + list(range(20, 54))
    record = _bank_report(_generate(pack, monkeypatch, mode="parity2", ids=ids))[
        "fixed_m4_parity2"
    ]
    assert record["positions"] == "text" and record["rope_delta"] is None
    assert record["rounds"] >= 8 and record["divergent_rounds"] == 0
    assert record["logits_max_abs_diff"] == 0.0 and record["logits_max_kl"] == 0.0


def test_parity2_catches_a_delta_that_is_off_by_one(pack, monkeypatch):
    """An instrument that cannot fail proves nothing.

    The reference leg takes its positions from the request scope the
    generation loop opens from the splice, not from the delta the bank was
    handed, so a wrong delta at the admission cannot hide behind itself.
    """

    real = generation._qwen4_vision_compiled_verify_admission

    def off_by_one(rt, vision_splice, prompt_ids):
        verdict = real(rt, vision_splice, prompt_ids)
        if verdict["rope_delta"] is not None:
            verdict = dict(verdict, rope_delta=verdict["rope_delta"] + 1)
        return verdict

    monkeypatch.setattr(
        generation, "_qwen4_vision_compiled_verify_admission", off_by_one
    )
    ids, table, delta = _image_prompt(18)
    bank = _bank_report(
        _generate(pack, monkeypatch, mode="parity2", ids=ids, table=table, delta=delta)
    )
    record = bank["fixed_m4_parity2"]
    assert record["rope_delta"] == delta + 1
    assert record["divergent_rounds"] == record["rounds"] > 0
    assert record["logits_max_abs_diff"] > 1e-3 and record["logits_max_kl"] > 0.0
    first = record["first_divergence"]
    assert first["round"] == 1 and first["width"] == 4
    assert any(line.startswith("state[") for line in first["report"])
    assert bank["parity2_first_divergence"]["leaf"] == first["leaf"]


def test_parity2_counts_a_round_it_has_no_reference_for(pack, monkeypatch):
    """A caller with no request scope open leaves nothing to compare with.

    The reference leg reads the positions from the caller's scope. Without one
    it would rotate at text positions and the round would read as divergent
    whatever the lane did (before the eager scope fix, a copy round routed
    through the bank by MTPLX_CCOPY_BANK_ROUTE=1 is such a caller). The round
    is counted, not compared, and the maxima stay the maxima of real
    comparisons. The lane itself does not notice: it never reads the scope.
    """

    import mtplx.attention_context as attention_context

    ids, table, delta = _image_prompt(18)
    plain = _generate(pack, monkeypatch, mode="1", ids=ids, table=table, delta=delta)

    real = graphbank.CompiledVerifyBank.forward_fixed_m4
    calls = []

    def third_call_without_a_scope(self, *args, **kwargs):
        calls.append(True)
        if len(calls) != 3:
            return real(self, *args, **kwargs)
        token = attention_context._VISION_ROPE.set(None)
        try:
            return real(self, *args, **kwargs)
        finally:
            attention_context._VISION_ROPE.reset(token)

    monkeypatch.setattr(
        graphbank.CompiledVerifyBank, "forward_fixed_m4", third_call_without_a_scope
    )
    checked = _generate(
        pack, monkeypatch, mode="parity2", ids=ids, table=table, delta=delta
    )
    assert checked.tokens == plain.tokens
    bank = _bank_report(checked)
    record = bank["fixed_m4_parity2"]
    assert record["reference_scope_missing_rounds"] == 1
    assert record["reference_scope_missing_by_width"] == {"4": 1}
    assert record["rounds"] == bank["calls"] - 1 >= 7
    assert record["divergent_rounds"] == 0 and record["first_divergence"] is None
    assert record["logits_max_abs_diff"] == 0.0 and record["state_max_abs_diff"] == 0.0


def test_an_image_request_outgrows_its_first_grant_and_stays_exact(pack, monkeypatch):
    """A capacity transition rebuilds the shadow twins and picks the trace again.

    The delta lives on the real bank entries and the key carries the delta
    flag, so the grown bank must replay the image trace at the same origin.
    A grant of eight tokens makes a 40-token request cross several of them
    (the product's first grant is 1,024 tokens).
    """

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_GROWTH_RESERVE", "8")
    ids, table, delta = _image_prompt(18)
    eager = _generate(pack, monkeypatch, mode="0", ids=ids, table=table, delta=delta)
    compiled = _generate(pack, monkeypatch, mode="1", ids=ids, table=table, delta=delta)
    assert compiled.tokens == eager.tokens
    bank = _bank_report(compiled)
    assert bank["fixed_m4_capacity_transitions"] >= 2
    assert bank["fixed_m4"]["rope_delta_input"] is True
    assert bank["fixed_m4"]["rope_delta"] == delta
    assert bank["fallback_calls"] == 0
    assert all(key.endswith(":rope_delta") for key in bank["compiled_keys"])

    checked = _generate(
        pack, monkeypatch, mode="parity2", ids=ids, table=table, delta=delta
    )
    assert checked.tokens == compiled.tokens
    checked_bank = _bank_report(checked)
    record = checked_bank["fixed_m4_parity2"]
    assert checked_bank["fixed_m4_capacity_transitions"] >= 2
    assert record["rounds"] == checked_bank["calls"] >= 8
    assert record["divergent_rounds"] == 0 and record["state_max_abs_diff"] == 0.0


def _parity_script():
    path = _SMOKE.with_name("qwen4_vision_compiled_parity.py")
    spec = importlib.util.spec_from_file_location("qwen4_vision_compiled_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_request_entry(kind, result):
    """One request as the real-pack script reads it from the request log."""

    import hashlib

    entry = {
        "kind": kind,
        "output_sha256": hashlib.sha256(repr(result.tokens).encode()).hexdigest(),
        "fixed_m4_admission": result.stats.fixed_m4_admission,
    }
    if _bank_report(result):
        entry["compiled_verify"] = _bank_report(result)
    return entry


def test_the_real_pack_script_requires_a_follow_up_after_exact_single_turns(pack, monkeypatch):
    """The verdict of scripts/qwen4_vision_compiled_parity.py over real records.

    Its three arms, run here on the tiny pack through the generation loop:
    the instrument, the product mode, and the kill switch. These single-turn
    records prove per-round parity, but cannot prove restoration of generated
    state. The script must require that additional evidence.
    """

    script = _parity_script()
    ids, table, delta = _image_prompt(18)
    text_ids = [3, 5, 7, 9, 11, 13] + list(range(20, 54))

    def arm(mode, *, kill=False):
        if kill:
            monkeypatch.setenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "0")
        else:
            monkeypatch.delenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", raising=False)
        image = _generate(pack, monkeypatch, mode=mode, ids=ids, table=table, delta=delta)
        text = _generate(pack, monkeypatch, mode=mode, ids=text_ids)
        return {
            "requests": {
                "image_turn1": _as_request_entry("image", image),
                "text_control": _as_request_entry("text", text),
            }
        }

    arms = {"instrument": arm("parity2"), "product": arm("1"), "eager": arm("1", kill=True)}
    outcome = script.evaluate(arms, eager_scope_fix_present=True)
    assert outcome["verdict"] == "generated_state_not_restored", outcome
    assert outcome["failed_checks"] == [
        "image_turns_completed", "follow_up_restored_generated_tokens",
    ]
    assert outcome["exit_code"] == 1
    assert outcome["instrument"]["image_turn1"]["rounds"] >= 8
    assert outcome["instrument"]["text_control"]["state"] == "exact"
    assert {c["name"]: c["ok"] for c in outcome["checks"]} == {
        "image_turns_completed": False,
        "instrument_image_rounds_exact": True,
        "instrument_every_image_round_had_a_reference": True,
        "instrument_text_control_exact": True,
        "product_image_requests_take_the_compiled_route": True,
        "instrument_did_not_change_the_output": True,
        "eager_arm_ran_on_the_eager_verifier": True,
        "compiled_output_equals_eager_output": True,
        "compiled_arms_executed_compiled_verification": True,
        "follow_up_restored_generated_tokens": False,
    }


def test_the_real_pack_script_reads_a_wrong_delta_as_the_image_routes_fault(
    pack, monkeypatch
):
    script = _parity_script()
    real = generation._qwen4_vision_compiled_verify_admission

    def off_by_one(rt, vision_splice, prompt_ids):
        verdict = real(rt, vision_splice, prompt_ids)
        if verdict["rope_delta"] is not None:
            verdict = dict(verdict, rope_delta=verdict["rope_delta"] + 1)
        return verdict

    ids, table, delta = _image_prompt(18)
    text_ids = [3, 5, 7, 9, 11, 13] + list(range(20, 54))
    text = _generate(pack, monkeypatch, mode="parity2", ids=text_ids)
    monkeypatch.setattr(
        generation, "_qwen4_vision_compiled_verify_admission", off_by_one
    )
    image = _generate(pack, monkeypatch, mode="parity2", ids=ids, table=table, delta=delta)
    arms = {
        "instrument": {
            "requests": {
                "image_turn1": _as_request_entry("image", image),
                "text_control": _as_request_entry("text", text),
            }
        }
    }
    outcome = script.evaluate(arms, eager_scope_fix_present=True)
    assert outcome["verdict"] == "image_route_diverges" and outcome["exit_code"] == 1
    assert outcome["instrument"]["image_turn1"]["first_divergence"]["round"] == 1
    assert outcome["instrument"]["text_control"]["state"] == "exact"


def test_the_refused_shape_stays_eager_and_says_why(pack, monkeypatch):
    ids, table, delta = _image_prompt(1)  # 23 tokens: the last block starts at 20
    assert max(i for i, token in enumerate(ids) if token == PAD) == 21
    built = []
    real_bank = generation.CompiledVerifyBank

    def counting_bank(*args, **kwargs):
        built.append(True)
        return real_bank(*args, **kwargs)

    monkeypatch.setattr(generation, "CompiledVerifyBank", counting_bank)
    eager = _generate(pack, monkeypatch, mode="0", ids=ids, table=table, delta=delta)
    refused = _generate(pack, monkeypatch, mode="1", ids=ids, table=table, delta=delta)
    assert built == []  # no bank, no promotion: byte for byte the eager request
    assert refused.tokens == eager.tokens
    admission = refused.stats.fixed_m4_admission
    assert admission["engaged"] is False
    assert admission["reason"] == "vision_tail_block_in_image"
    assert admission["positions"] == "vision_delta" and admission["rope_delta"] == delta
    assert refused.stats.demotions == {"vision_request_eager_verify": 1}


def test_the_kill_switch_restores_the_eager_route(pack, monkeypatch):
    monkeypatch.setenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "0")
    ids, table, delta = _image_prompt(18)
    eager = _generate(pack, monkeypatch, mode="0", ids=ids, table=table, delta=delta)
    switched = _generate(pack, monkeypatch, mode="1", ids=ids, table=table, delta=delta)
    assert switched.tokens == eager.tokens
    assert switched.stats.fixed_m4_admission["reason"] == "vision_kill_switch"
    assert _bank_report(switched) == {}
