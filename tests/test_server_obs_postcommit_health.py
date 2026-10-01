"""Issue #487: a generation-final commit that normally takes 0.2 s took 20-25 s
on a 110-150k vision agent session, the app's /health probe went unanswered
and the daemon was reaped mid-commit. Three contracts guard that here:

* the commit names its slow phase in the ``pc`` flight event (history
  render, committed-stream decode, tokenizer, vision tower, bank put);
* /health answers while a commit holds the model lock and stalls inside
  the bank -- the route shares no lock with the commit;
* a prompt whose screenshots exceed the vision embed cache's row budget is
  not evicted by its own pass, so the commit's second walk over the same
  images hits the cache instead of re-running the tower for every one.

Engine always faked -- no model loads, no GPU.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai as oa
from mtplx.server.flight_recorder import FlightRecorder
from mtplx.server.openai import (
    ChatMessage,
    _encode_messages,
    _store_generation_final_history_snapshot,
    create_app,
)

from tests.test_openai_bridge import _final_state, _ids, _postcommit_state
from tests.test_server_openai import _fake_state

_PHASES = (
    "flatten_s",
    "canonicalize_s",
    "committed_decode_s",
    "render_encode_s",
    "history_s",
    "compat_s",
    "lock_wait_s",
    "mtp_snapshot_s",
    "put_s",
)


def _stored_commit(state, *, session_id: str = "session-1") -> dict:
    messages = [ChatMessage(role="user", content="hi")]
    prompt_ids = _encode_messages(
        state.runtime.tokenizer,
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    generated_tokens = _ids("ok")
    generated = {
        "tokens": generated_tokens,
        "_final_state": _final_state(generated_tokens),
    }
    return _store_generation_final_history_snapshot(
        state,
        session_id=session_id,
        prompt_ids=prompt_ids,
        generated=generated,
        messages=messages,
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy",
    )


# --- phase receipts -----------------------------------------------------------


def test_generation_final_commit_names_every_phase_and_the_flight_event_carries_it():
    state = _postcommit_state()
    events: list[dict] = []
    state.flight = SimpleNamespace(pc=lambda _session_id, payload: events.append(payload))

    result = _stored_commit(state)

    assert result["stored"] is True
    timing = result["timing"]
    for key in _PHASES:
        assert key in timing, f"missing phase {key}"
        assert timing[key] >= 0.0
    assert timing["vision_images"] == 0
    assert timing["compat_s"] >= timing["history_s"]
    assert result["elapsed_s"] >= timing["compat_s"]
    assert events and events[-1]["timing"] is timing


def test_flight_recorder_pc_event_keeps_the_timing_block(tmp_path):
    recorder = FlightRecorder(str(tmp_path / "flight.jsonl"))
    captured: list[dict] = []
    recorder._emit = captured.append  # type: ignore[method-assign]

    timing = {"compat_s": 0.1, "put_s": 20.2, "put": {"trunk_snapshot_s": 0.01}}
    recorder.pc(
        "session-1",
        {"action": "generation_final", "stored": True, "elapsed_s": 20.3, "timing": timing},
    )

    assert captured[-1]["ev"] == "pc"
    assert captured[-1]["timing"] == timing


def test_bank_put_receives_a_timing_dict_only_when_it_accepts_one():
    class TimedBank:
        def put(self, *, timing_out=None, **kwargs):
            assert timing_out is not None
            timing_out["trunk_snapshot_s"] = 0.25
            timing_out["cold_enqueue"] = {"enabled": False, "skip_reason": "no_cold_tier"}
            return SimpleNamespace(prefix_len=len(kwargs["token_ids"]), nbytes=1, token_hash="h")

    class LegacyBank:
        """Explicit keyword list, no **kwargs: the pre-#487 put contract."""

        def __init__(self) -> None:
            self.calls = 0

        def put(
            self,
            *,
            runtime,
            token_ids,
            cache,
            logits,
            hidden,
            hidden_variant,
            keep_live_ref,
            session_id,
            template_hash,
            mtp_history_policy,
            draft_head_identity,
            policy_fingerprint,
            mtp_history_snapshot,
            snapshot_epoch,
            mtp_snapshot_epoch,
            extra_state,
        ):
            self.calls += 1
            return SimpleNamespace(prefix_len=len(token_ids), nbytes=1, token_hash="h")

    timed = _postcommit_state()
    timed.sessions = SimpleNamespace(bank=TimedBank())
    result = _stored_commit(timed)
    assert result["stored"] is True
    assert result["timing"]["put"]["trunk_snapshot_s"] == 0.25
    assert result["timing"]["put"]["cold_enqueue"]["skip_reason"] == "no_cold_tier"

    legacy = _postcommit_state()
    legacy_bank = LegacyBank()
    legacy.sessions = SimpleNamespace(bank=legacy_bank)
    result = _stored_commit(legacy)
    assert result["stored"] is True and legacy_bank.calls == 1
    assert "put" not in result["timing"]


# --- /health while the commit holds the model lock --------------------------


def test_health_answers_while_a_generation_final_commit_holds_the_model_lock(monkeypatch):
    state = _fake_state()
    entered = threading.Event()
    release = threading.Event()

    class StallingBank:
        def put(self, **kwargs):
            entered.set()
            assert release.wait(10.0), "test harness never released the stalled put"
            return SimpleNamespace(
                prefix_len=len(kwargs["token_ids"]), nbytes=1, token_hash="h"
            )

        def to_dict(self):
            return {"entries": 0}

    state.sessions.bank = StallingBank()
    state.sessions.peek = lambda _session_id: None
    monkeypatch.setattr(
        oa,
        "_generation_final_postcommit_compatibility",
        lambda *_args, **_kwargs: {
            "safe": True,
            "mode": "generation_final_exact",
            "reason": "token_identical",
            "token_ids": [1, 2, 3],
            "history_suffix_tokens": 0,
        },
    )
    client = TestClient(create_app(state))
    outcome: dict = {}

    def commit() -> None:
        outcome.update(
            _store_generation_final_history_snapshot(
                state,
                session_id="session-1",
                prompt_ids=[1, 2],
                generated={"tokens": [3], "_final_state": _final_state([3])},
                messages=[ChatMessage(role="user", content="hi")],
                assistant_content="ok",
                thinking_enabled=False,
                policy_fingerprint="policy",
            )
        )

    worker = threading.Thread(target=commit, name="commit", daemon=True)
    worker.start()
    try:
        assert entered.wait(5.0), "the commit never reached the bank put"
        assert state.lock.locked(), "the commit holds the model lock while the bank stores"
        started = time.perf_counter()
        response = client.get("/health")
        elapsed = time.perf_counter() - started
    finally:
        release.set()
        worker.join(10.0)
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert elapsed < 1.0, f"/health waited {elapsed:.2f}s behind the commit"
    assert outcome["stored"] is True
    assert not state.lock.locked()


# --- vision embed cache: a prompt's own images survive its pass ---------------


def _fake_tower(monkeypatch, tower_calls: list) -> None:
    import mtplx.vision as vision_pkg
    import mtplx.vision.processing as processing_pkg

    # Rows are plain Python values: mx.eval ignores non-array leaves, so no
    # Metal device is ever touched here.
    monkeypatch.setattr(
        vision_pkg,
        "load_vision_tower",
        lambda _path: (lambda pixel_values, _grids: (("rows", pixel_values), {})),
    )
    monkeypatch.setattr(processing_pkg, "decode_image", lambda raw: raw)
    monkeypatch.setattr(processing_pkg, "image_pad_token_count", lambda _grid: 4)

    def preprocess(images, _config):
        tower_calls.append(images[0])
        return images[0], [(1, 4, 4)]

    monkeypatch.setattr(processing_pkg, "preprocess_images", preprocess)


def test_a_prompts_own_images_survive_the_embed_cache_pass(monkeypatch):
    tower_calls: list = []
    _fake_tower(monkeypatch, tower_calls)
    # Budget for two images; the prompt carries three (the #487 shape:
    # more screenshot rows in one history than the LRU holds).
    monkeypatch.setattr(oa, "_VISION_EMBED_CACHE_MAX_ROWS", 8)
    oa._VISION_EMBED_CACHE.clear()
    try:
        images = [b"shot-1", b"shot-2", b"shot-3"]
        digests = [111, 222, 333]
        pinned = frozenset(("/m", digest) for digest in digests)
        timing: dict = {}
        for raw, digest in zip(images, digests):  # the request's pass
            oa._vision_rows_for_image(None, "/m", {}, raw, digest, pinned=pinned, timing=timing)
        assert len(tower_calls) == 3 and timing["vision_tower_misses"] == 3
        assert all(("/m", digest) in oa._VISION_EMBED_CACHE for digest in digests), (
            "a prompt must never evict its own images mid-pass"
        )

        for raw, digest in zip(images, digests):  # the generation-final commit's pass
            oa._vision_rows_for_image(None, "/m", {}, raw, digest, pinned=pinned, timing=timing)
        assert len(tower_calls) == 3, (
            "the commit re-ran the tower for images the request had just embedded"
        )
        assert timing["vision_cache_hits"] == 3
        assert timing["vision_tower_s"] >= 0.0

        # The budget still binds across prompts: a later prompt's insert evicts
        # the least recently used keys it does not pin.
        oa._vision_rows_for_image(
            None, "/m", {}, b"shot-4", 444, pinned=frozenset({("/m", 444)})
        )
        assert ("/m", 111) not in oa._VISION_EMBED_CACHE
        assert ("/m", 222) not in oa._VISION_EMBED_CACHE
        assert ("/m", 333) in oa._VISION_EMBED_CACHE
        assert ("/m", 444) in oa._VISION_EMBED_CACHE
    finally:
        oa._VISION_EMBED_CACHE.clear()


def test_an_unpinned_insert_keeps_the_plain_lru_contract(monkeypatch):
    tower_calls: list = []
    _fake_tower(monkeypatch, tower_calls)
    monkeypatch.setattr(oa, "_VISION_EMBED_CACHE_MAX_ROWS", 4)
    oa._VISION_EMBED_CACHE.clear()
    try:
        oa._vision_rows_for_image(None, "/m", {}, b"a", 1)
        oa._vision_rows_for_image(None, "/m", {}, b"b", 2)
        assert ("/m", 1) not in oa._VISION_EMBED_CACHE
        assert ("/m", 2) in oa._VISION_EMBED_CACHE
    finally:
        oa._VISION_EMBED_CACHE.clear()
