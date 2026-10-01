"""Dense-path image positions through the real generation loop.

A four-layer synthetic ``qwen3_5`` model with a one-layer draft head runs
``generate_mtpk`` end to end (chunked prefill with the committed draft
history, draft, verify, repair, commit, the session bank). Random weights, no
pack. The reference for every generated token is ONE teacher-forced forward
without a cache, which positions the whole sequence through a different code
path (per-axis positions for the prompt, index + delta for the tail) than the
cached decode does (the stock rope at cache offset + delta).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
from mtplx import demotions
from mtplx.attention_context import vision_rope_state
from mtplx.dense_mrope import (
    DenseMRopeState,
    configure_dense_mrope,
    dense_mrope_scope,
    dense_mrope_state,
)
from mtplx.generation import generate_mtpk, restore_or_prefill_prompt_state
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig
from mtplx.session_bank import SessionBank
from mtplx.vision.splice import VisionSplice, spliced_chunk_embeddings
from tests import dense_mrope_synth as synth

PAD, GRID, PROMPT, DELTA = synth.PAD, synth.GRID, synth.PROMPT, synth.DELTA
FIRST_IMAGE = PROMPT.index(PAD)
# A generated token must be the teacher-forced argmax up to the float noise
# between a chunked cached forward and one uncached forward on this model
# (a few 1e-3 from the GatedDeltaNet layers); a row at a wrong position moves
# logits by 1e-1.
LOGIT_NOISE = 2e-2


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return " ".join(str(int(token)) for token in tokens)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (
        "MTPLX_DENSE_MROPE",
        "MTPLX_DENSE_MROPE_STRICT",
        "MTPLX_COMPILED_VERIFY",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD",
        "MTPLX_MTP_POSITION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    demotions.reset()
    yield
    demotions.reset()


@pytest.fixture()
def rig(tmp_path):
    """(runtime, model, image rows, spies) on the synthetic model."""

    # Its own LM head: greedy decoding gives varied tokens, not an echo.
    model = synth.model_with_draft_head(tmp_path, seed=8, tie=False)
    # The product's attention hook (runtime.load installs it on every model):
    # a cache that owns a rotary origin rotates at it, everything else is the
    # stock forward. Without it a compiled route would rope at the plain
    # tensor offset through the unhooked stock forward.
    from mtplx.attention_split import configure_split_full_attention

    configure_split_full_attention(model)
    assert configure_dense_mrope(model, synth.pack_config()).installed
    spies = SimpleNamespace(
        trunk=[synth.spy_on(attn) for attn in synth.full_attention(model)],
        mtp=synth.spy_on(model.mtp.layers[0].self_attn),
    )
    rt = MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=tmp_path,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    mx.random.seed(3)
    rows = mx.random.normal((6, 64)) * 0.05
    mx.eval(rows)
    return SimpleNamespace(rt=rt, model=model, rows=rows, spies=spies)


def _state_for(prompt) -> DenseMRopeState:
    return synth.position_state(prompt)


def _splice(rig, prompt, *, armed: bool = True, digest: int = 1) -> VisionSplice:
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=rig.rows,
        image_digests=(digest,),
        pad_counts=(6,),
        image_grids=(GRID,),
        dense_mrope=_state_for(prompt) if armed else None,
    )


def _generate(rig, prompt=PROMPT, *, splice=None, depth=3, max_tokens=12, **kwargs):
    kwargs.setdefault("verify_strategy", "capture_commit")
    kwargs.setdefault("mtp_history_policy", "committed")
    return generate_mtpk(
        rig.rt,
        list(prompt),
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
        speculative_depth=depth,
        stop_token_ids=set(),
        vision_splice=_splice(rig, prompt) if splice is None else splice,
        **kwargs,
    )


def _teacher_forced_gaps(rig, prompt, tokens, *, armed: bool = True) -> np.ndarray:
    """max logit minus the generated token's logit, per generated token."""
    full = list(prompt) + [int(t) for t in tokens]
    ids = mx.array([full])
    embedded = spliced_chunk_embeddings(
        rig.model.model.embed_tokens, ids, _splice(rig, prompt, armed=False)
    )
    scope = dense_mrope_scope(_state_for(prompt)) if armed else contextlib.nullcontext()
    with scope:
        logits = rig.model(ids, input_embeddings=embedded)
    rows = np.asarray(logits[0, len(prompt) - 1 : -1])
    return rows.max(axis=-1) - rows[np.arange(len(tokens)), np.asarray(tokens)]


def test_every_forward_of_an_image_request_ropes_at_grid_positions(rig):
    n = len(PROMPT)
    for depth in (1, 3):
        for spy in (*rig.spies.trunk, rig.spies.mtp):
            spy.calls.clear()
        out = _generate(rig, depth=depth)
        assert len(out.tokens) == 12
        # Draft, verify, repair and commit all positioned the rows the way
        # one uncached forward over the final sequence does.
        assert _teacher_forced_gaps(rig, PROMPT, out.tokens).max() < LOGIT_NOISE
        for spy in rig.spies.trunk:
            assert all(isinstance(offset, int) for _rows, offset in spy.calls)
            # Prompt text runs and everything after the prompt are stock calls
            # at the shifted offset: the last row lands at n + tokens + delta.
            assert max(rows + offset for rows, offset in spy.calls) <= n + 12 + DELTA
            assert any(offset == n + DELTA for _rows, offset in spy.calls)
        # The draft head follows the same positions, one row behind.
        assert all(isinstance(offset, int) for _rows, offset in rig.spies.mtp.calls)
        assert (1, n - 1 + DELTA) in rig.spies.mtp.calls
        assert max(offset for _rows, offset in rig.spies.mtp.calls) < n + 12 + DELTA
    assert demotions.snapshot()["total"] == 0
    assert dense_mrope_state() is None and vision_rope_state() is None


def _oracle_draft_head(rig, full_sequence):
    """Make every draft correct, so rounds commit several tokens.

    The real draft head still runs (its cache, its rope calls); only the
    logits it returns are replaced by a one-hot on the token the target will
    produce. Draft row r pairs hidden r with token r + 1 and predicts r + 2.
    """
    real = rig.model.mtp_forward

    def oracle(hidden_states, next_token_ids, **kwargs):
        cache = kwargs.get("mtp_cache")
        row = int(cache[0].offset) if cache else 0
        out = real(hidden_states, next_token_ids, **kwargs)
        logits, hidden = out if kwargs.get("return_hidden") else (out, None)
        targets = [
            full_sequence[min(row + i + 2, len(full_sequence) - 1)]
            for i in range(int(logits.shape[1]))
        ]
        peaked = mx.zeros_like(logits)
        peaked[0, mx.arange(len(targets)), mx.array(targets)] = 50.0
        return (peaked, hidden) if kwargs.get("return_hidden") else peaked

    rig.model.mtp_forward = oracle


def test_accepted_drafts_and_a_live_history_reset_keep_grid_positions(rig, monkeypatch):
    n = len(PROMPT)
    reference = _generate(rig, depth=1).tokens
    _oracle_draft_head(rig, list(PROMPT) + reference)

    for spy in (*rig.spies.trunk, rig.spies.mtp):
        spy.calls.clear()
    out = _generate(rig, depth=3)
    assert out.tokens == reference  # speculative decoding is exact
    assert out.stats.verify_calls < len(reference) // 2  # drafts were accepted
    assert _teacher_forced_gaps(rig, PROMPT, out.tokens).max() < LOGIT_NOISE
    for spy in (*rig.spies.trunk, rig.spies.mtp):
        assert all(isinstance(offset, int) for _rows, offset in spy.calls)
        assert max(rows + offset for rows, offset in spy.calls) <= n + 12 + DELTA
    # Committed rows are appended to the draft history several at a time.
    assert any(rows > 1 for rows, _offset in rig.spies.mtp.calls)
    assert demotions.snapshot()["total"] == 0

    # A live history reset restarts the draft cache at row 0 with rows past
    # the prompt only, where the stock rope is exact: the draft head is
    # released, nothing is counted, the output does not change.
    monkeypatch.setenv("MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD", "4")
    splice = _splice(rig, PROMPT)
    rig.spies.mtp.calls.clear()
    out = _generate(rig, depth=3, splice=splice)
    assert out.stats.mtp_history_live_resets >= 1
    assert splice.dense_mrope.mtp_aligned is False
    assert out.tokens == reference
    assert min(offset for _rows, offset in rig.spies.mtp.calls) == 0  # fresh cache
    assert demotions.snapshot()["total"] == 0


def test_final_state_keeps_the_draft_history_row_aligned(rig):
    """The rule the next turn's restore rests on.

    A restored draft history is taken as row-aligned with the prompt (row j =
    prompt index j, so the position table can be indexed by its offset) exactly
    when it holds one row per prefix token but the last. The state a generation
    hands to the bank must satisfy that, with and without accepted drafts.
    """
    reference = _generate(rig, depth=1).tokens
    for accept in (False, True):
        if accept:
            _oracle_draft_head(rig, list(PROMPT) + reference)
        for max_tokens in (8, 9, 10):
            out = _generate(rig, depth=3, max_tokens=max_tokens, capture_final_state=True)
            state = out.final_state
            assert state.safe_to_commit
            total = len(PROMPT) + len(out.tokens)
            assert generation._mtp_cache_offset(state.final_committed_mtp_cache) == total - 1
            assert max(
                int(entry.offset)
                for entry in state.final_trunk_cache
                if isinstance(getattr(entry, "offset", None), int)
            ) == total
    assert demotions.snapshot()["total"] == 0


def test_sequential_image_request_is_what_it_was(rig):
    """dense_mrope None (kill switch, fallback): no shifted call anywhere."""
    out = _generate(rig, splice=_splice(rig, PROMPT, armed=False))
    n = len(PROMPT)
    assert _teacher_forced_gaps(rig, PROMPT, out.tokens, armed=False).max() < LOGIT_NOISE
    for spy in rig.spies.trunk:
        assert min(offset for _rows, offset in spy.calls) == 0
        assert max(rows + offset for rows, offset in spy.calls) > n + 12 + DELTA
    assert demotions.snapshot()["total"] == 0


def test_the_armed_request_takes_the_compiled_verifier_with_its_delta(
    rig, monkeypatch
):
    """The bank owns the rotary origin, so the armed request keeps the route a
    text request has (tests/test_dense_mrope_compiled_route.py proves the
    route bit for bit; this pins the wiring and the record)."""
    eager = _generate(rig)
    monkeypatch.setenv("MTPLX_DENSE_VISION_COMPILED_VERIFY", "1")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    built: list[dict] = []
    real_bank = generation.CompiledVerifyBank

    def counting_bank(*args, **kwargs):
        built.append(dict(kwargs))
        return real_bank(*args, **kwargs)

    monkeypatch.setattr(generation, "CompiledVerifyBank", counting_bank)

    out = _generate(rig)
    assert len(built) == 1 and built[0]["rope_delta"] == DELTA
    bank = out.stats.graphbank["compiled_verify"]
    assert bank["compiled_calls"] >= 1 and bank["fallback_calls"] == 0
    assert bank["rope_delta_input"] is True and bank["rope_delta"] == DELTA
    assert bank["compiled_keys"] and all(k.endswith(":rope_delta") for k in bank["compiled_keys"])
    assert out.tokens == eager.tokens
    assert out.stats.compiled_verify_admission == {
        "family": "dense",
        "positions": "vision_delta",
        "rope_delta": DELTA,
        "images": 1,
        "engaged": True,
        "reason": "admitted",
    }
    assert demotions.snapshot()["total"] == 0 and out.stats.demotions == {}

    # A sequentially positioned image request keeps the route it had: the
    # text trace, said so in the record.
    out = _generate(rig, splice=_splice(rig, PROMPT, armed=False))
    assert len(built) == 2 and built[1]["rope_delta"] is None
    bank = out.stats.graphbank["compiled_verify"]
    assert bank["rope_delta_input"] is False and bank["rope_delta"] is None
    assert not any(k.endswith(":rope_delta") for k in bank["compiled_keys"])
    assert out.stats.compiled_verify_admission["positions"] == "vision_sequential"
    assert out.stats.compiled_verify_admission["reason"] == "admitted"
    assert demotions.snapshot()["total"] == 0


def test_the_graph_bank_and_the_compiled_draft_core_carry_the_delta(rig, monkeypatch):
    """Both promote caches; both stamp the request's delta on them."""
    monkeypatch.setenv("MTPLX_DENSE_VISION_COMPILED_VERIFY", "1")
    graph_banks: list[dict] = []
    draft_cores: list[dict] = []
    real_graph_bank = generation.SpecDecodeGraphBank
    real_draft_core = generation._make_device_draft_core

    def counting_graph_bank(*args, **kwargs):
        graph_banks.append(dict(kwargs))
        return real_graph_bank(*args, **kwargs)

    def counting_draft_core(*args, **kwargs):
        draft_cores.append(dict(kwargs))
        return real_draft_core(*args, **kwargs)

    monkeypatch.setattr(generation, "SpecDecodeGraphBank", counting_graph_bank)
    monkeypatch.setattr(generation, "_make_device_draft_core", counting_draft_core)
    out = _generate(
        rig,
        verify_strategy="graphbank_capture_commit",
        draft_core="device",
        max_tokens=10,
    )
    assert _teacher_forced_gaps(rig, PROMPT, out.tokens).max() < LOGIT_NOISE
    assert [g["rope_delta"] for g in graph_banks] == [DELTA]
    assert out.stats.graphbank["rope_delta"] == DELTA
    assert out.stats.graphbank["compiled_calls"] >= 1
    assert draft_cores and all(c["rope_delta"] == DELTA for c in draft_cores)
    assert out.stats.compiled_verify_admission["engaged"] is True
    assert "draft_core_forced" not in out.stats.compiled_verify_admission
    assert demotions.snapshot()["total"] == 0

    # A refused request keeps today's routes end to end: the kill switch
    # keeps the graph bank away and the draft head on the stock route, each
    # counted once, and the record says so.
    monkeypatch.setenv("MTPLX_QWEN4_VISION_COMPILED_VERIFY", "0")
    graph_banks.clear()
    draft_cores.clear()
    out = _generate(
        rig, verify_strategy="graphbank_capture_commit", draft_core="device", max_tokens=10
    )
    assert _teacher_forced_gaps(rig, PROMPT, out.tokens).max() < LOGIT_NOISE
    assert graph_banks == [] and draft_cores == []
    snap = demotions.snapshot()
    assert snap["counts"]["vision_request_eager_verify"] == 1
    assert snap["counts"]["vision_request_eager_draft"] == 1
    assert "kept off the compiled verify route" in snap["reasons"]["vision_request_eager_draft"]
    record = out.stats.compiled_verify_admission
    assert record["engaged"] is False and record["reason"] == "vision_kill_switch"
    assert "kept off the compiled verify route" in record["draft_core_forced"]


def test_draft_head_release_rules(rig):
    # cycle: per-cycle draft caches hold rows past the prompt only; the stock
    # rope is exact there, nothing is counted.
    splice = _splice(rig, PROMPT)
    out = _generate(rig, splice=splice, mtp_history_policy="cycle")
    assert splice.dense_mrope.mtp_aligned is False
    assert _teacher_forced_gaps(rig, PROMPT, out.tokens).max() < LOGIT_NOISE
    assert demotions.snapshot()["total"] == 0

    # last_window: the draft history is a window; the draft head keeps its
    # sequential positions (counted once), the trunk stays exact.
    splice = _splice(rig, PROMPT)
    out = _generate(rig, splice=splice, mtp_history_policy="last_window")
    assert splice.dense_mrope.mtp_aligned is False
    assert _teacher_forced_gaps(rig, PROMPT, out.tokens).max() < LOGIT_NOISE
    assert demotions.snapshot()["counts"] == {
        **dict.fromkeys(demotions.KINDS, 0),
        "vision_draft_head_sequential_positions": 1,
    }



def test_draft_head_release_helper_and_the_live_reset_site():
    import inspect

    # The live history reset releases the draft head without a count (run for
    # real in the oracle-draft test above); the site is pinned by source too.
    source = inspect.getsource(generation.generate_mtpk)
    reset = source.index("mtp_history_live_resets += 1")
    release = source.index("_dense_mrope_release_draft_head(_dense_mrope_request, None)")
    assert 0 < release - reset < 500

    state = _state_for(PROMPT)
    generation._dense_mrope_release_draft_head(None, "text request")
    generation._dense_mrope_release_draft_head(state, None)
    assert state.mtp_aligned is False and demotions.snapshot()["total"] == 0
    generation._dense_mrope_release_draft_head(state, "already released")
    assert demotions.snapshot()["total"] == 0

    state = _state_for(PROMPT)
    generation._dense_mrope_release_draft_head(state, "windowed")
    generation._dense_mrope_release_draft_head(state, "windowed")
    snap = demotions.snapshot()
    assert state.mtp_aligned is False
    assert snap["counts"]["vision_draft_head_sequential_positions"] == 1
    assert snap["reasons"]["vision_draft_head_sequential_positions"] == "windowed"


def test_session_bank_never_crosses_position_schemes(rig):
    bank = SessionBank()

    def turn(prompt, *, armed):
        return _generate(
            rig,
            prompt,
            splice=_splice(rig, prompt, armed=armed),
            depth=2,
            max_tokens=8,
            session_bank=bank,
            session_id="s",
            session_template_hash="t",
            session_draft_head_identity="d",
            session_policy_fingerprint="p",
            commit_prompt_state_to_bank=True,
        )

    first = turn(PROMPT, armed=True)
    assert first.stats.cached_tokens == 0
    longer = list(PROMPT) + [20, 21, 22, 23]

    warm = turn(longer, armed=True)
    assert warm.stats.cached_tokens == len(PROMPT)  # through the image, warm
    assert _teacher_forced_gaps(rig, longer, warm.tokens).max() < LOGIT_NOISE
    cold = _generate(rig, longer, depth=2, max_tokens=8)
    assert warm.tokens == cold.tokens
    assert warm.stats.demotions == {}  # restored draft history is row-aligned

    # The same pixels and text under sequential positions: nothing banked by
    # the grid-positioned turns may be restored at or past the image.
    sequential = turn(longer, armed=False)
    assert sequential.stats.cached_tokens <= FIRST_IMAGE
    assert (
        _teacher_forced_gaps(rig, longer, sequential.tokens, armed=False).max()
        < LOGIT_NOISE
    )
    # ...and the other way round.
    again = turn(longer + [24, 25], armed=True)
    assert again.stats.cached_tokens == len(longer)
    assert _teacher_forced_gaps(rig, longer + [24, 25], again.tokens).max() < LOGIT_NOISE
    sequential_again = turn(longer + [24, 25], armed=False)
    assert sequential_again.stats.cached_tokens == len(longer)


def test_prompt_changed_after_the_table_was_built_is_refused(rig):
    splice = _splice(rig, PROMPT)
    with pytest.raises(ValueError, match="does not match the prompt"):
        restore_or_prefill_prompt_state(
            rig.rt, [4] + list(PROMPT), vision_splice=splice, mtp_history_policy="committed"
        )
    assert dense_mrope_state() is None


def test_scope_helpers_and_wiring():
    assert hasattr(generation.generate_mtpk, "__wrapped__")
    assert isinstance(generation._vision_rope_scope_for(None), contextlib.nullcontext)

    state = _state_for(PROMPT)
    dense = SimpleNamespace(dense_mrope=state, mrope_table=None, mrope_delta=0)
    with generation._vision_rope_scope_for(dense):
        assert dense_mrope_state() is state
        assert vision_rope_state() is None  # the Flash-Next state stays unarmed
    assert dense_mrope_state() is None

    # A Flash-Next splice keeps its own scope and never arms the dense state.
    flash = SimpleNamespace(dense_mrope=None, mrope_table=None, mrope_delta=-3)
    with generation._vision_rope_scope_for(flash):
        assert vision_rope_state() == (None, -3)
        assert dense_mrope_state() is None
    assert generation._dense_mrope_state_of(None) is None
    assert generation._dense_mrope_state_of(SimpleNamespace(dense_mrope="x")) is None


def test_serve_layer_builds_the_state_for_dense_packs_only(tmp_path, monkeypatch):
    import mtplx.server.openai as openai
    from mtplx.vision import VisionSpec

    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    rows = mx.zeros((6, 64))
    monkeypatch.setattr(
        openai, "_vision_rows_for_image", lambda *a, **k: (rows, 6, GRID)
    )
    model = synth.text_model()
    assert configure_dense_mrope(model, synth.pack_config()).installed

    def materialize(model_type: str):
        spec = VisionSpec(
            model_dir=str(tmp_path),
            image_token_id=PAD,
            video_token_id=synth.VIDEO_PAD,
            vision_start_token_id=1,
            vision_end_token_id=2,
            spatial_merge_size=2,
            patch_size=16,
            temporal_patch_size=2,
            out_hidden_size=64,
            model_type=model_type,
            mrope_section=(3, 3, 2),
            mrope_interleaved=True,
        )
        state = SimpleNamespace(
            _vision_spec_cache=spec,
            args=SimpleNamespace(model=str(tmp_path)),
            runtime=SimpleNamespace(model=model),
        )
        one_placeholder = [5, 6, 7, PAD, 8, 9, 10, 11]
        return openai._materialize_vision_splice(state, [b"pixels"], one_placeholder)

    ids, splice = materialize("qwen3_5")
    assert ids == PROMPT
    assert isinstance(splice.dense_mrope, DenseMRopeState)
    assert splice.dense_mrope.delta == DELTA
    assert splice.mrope_table is None and splice.mrope_delta == 0
    np.testing.assert_array_equal(
        splice.dense_mrope.pad_positions, np.flatnonzero(np.asarray(PROMPT) == PAD)
    )

    _ids, flash = materialize("qwen4_exp")
    assert flash.dense_mrope is None
    assert flash.mrope_table is not None and flash.mrope_delta == DELTA

    monkeypatch.setenv("MTPLX_DENSE_MROPE", "0")
    _ids, off = materialize("qwen3_5")
    assert off.dense_mrope is None and off.mrope_table is None
    assert demotions.snapshot()["total"] == 0


def test_text_request_keeps_the_compiled_verifier_and_its_tokens(tmp_path, monkeypatch):
    """Adapters installed, nothing armed: a text request is what it was."""
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    text = [(7 * i + 3) % 97 for i in range(40)]

    def run(folder, *, install: bool):
        folder.mkdir()
        model = synth.model_with_draft_head(folder, seed=8, tie=False)
        if install:
            assert configure_dense_mrope(model, synth.pack_config()).installed
        rt = MTPLXRuntime(
            model=model,
            tokenizer=_Tokenizer(),
            model_path=folder,
            mtp_enabled=True,
            contract=MTPContract(),
        )
        return generate_mtpk(
            rt,
            text,
            max_tokens=24,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=20),
            speculative_depth=3,
            mtp_history_policy="committed",
            verify_strategy="capture_commit",
            stop_token_ids=set(),
        )

    stock = run(tmp_path / "stock", install=False)
    adapted = run(tmp_path / "adapted", install=True)
    assert adapted.tokens == stock.tokens
    bank = adapted.stats.graphbank["compiled_verify"]
    assert bank["compiled_calls"] == stock.stats.graphbank["compiled_verify"]["compiled_calls"]
    assert bank["compiled_calls"] >= 1 and bank["fallback_calls"] == 0
    assert demotions.snapshot()["total"] == 0
