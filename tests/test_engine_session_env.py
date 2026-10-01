"""Unit tests for engine_session bank-cap env-var overrides."""

import pytest


def _engine_session():
    """Import — never reload — the live module.

    Every function under test reads its env vars at call time
    (``os.environ.get`` inside the function bodies), so the old
    ``importlib.reload`` here was pure ritual. It was also a process-global
    leak: reload re-executes the module body in place, replacing the
    ``EngineSession*``/``EngineSessionBusy`` class objects, so any file
    that bound them at collection time stopped matching what product code
    raises afterwards (test_engine_session_concurrency's
    ``pytest.raises(EngineSessionBusy)`` no longer caught the busy error
    when this file ran first). A plain import keeps this file hermetic in
    any run order — and pins call-time env semantics: these tests now fail
    honestly if env parsing ever regresses to import-time caching.
    """
    import mtplx.engine_session

    return mtplx.engine_session


def test_bank_bytes_from_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_BANK_BYTES", raising=False)
    es = _engine_session()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 1234) == 1234


def test_bank_bytes_from_env_plain_integer(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "987654321")
    es = _engine_session()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 0) == 987654321


@pytest.mark.parametrize("raw,expected", [
    ("16G", 16 * 1024**3),
    ("16g", 16 * 1024**3),
    ("32G", 32 * 1024**3),
    ("8G",   8 * 1024**3),
    ("512M", 512 * 1024**2),
    ("4K",   4 * 1024),
    ("1T",   1 * 1024**4),
    ("0.5G", int(0.5 * 1024**3)),
])
def test_bank_bytes_from_env_with_suffix(monkeypatch, raw, expected):
    monkeypatch.setenv("TEST_BANK_BYTES", raw)
    es = _engine_session()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 0) == expected


def test_bank_bytes_from_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "not-a-number")
    es = _engine_session()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 5555) == 5555


def test_bank_bytes_from_env_empty_string_uses_default(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "")
    es = _engine_session()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 7777) == 7777


@pytest.mark.parametrize("raw", ["0", "-1", "0G", "-2G"])
def test_bank_bytes_from_env_nonpositive_uses_default(monkeypatch, raw):
    monkeypatch.setenv("TEST_BANK_BYTES", raw)
    es = _engine_session()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 8888) == 8888


def test_short_no_history_api_request_is_foreground_by_default():
    es = _engine_session()
    messages = [
        {"role": "system", "content": "Return only the final answer."},
        {"role": "user", "content": "Compute 17 + 29 + 101."},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={},
            metadata={},
            main_system_hash=None,
        )
        is False
    )


def test_openwebui_task_header_still_marks_background():
    es = _engine_session()
    messages = [
        {"role": "system", "content": "Return a short title."},
        {"role": "user", "content": "Conversation text"},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={"x-openwebui-task": "title"},
            metadata={},
            main_system_hash=None,
        )
        is True
    )


def test_system_prompt_mismatch_still_marks_background():
    es = _engine_session()
    main_hash = es.hash_text("main chat system")
    messages = [
        {"role": "system", "content": "Return a short title."},
        {"role": "user", "content": "Conversation text"},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={},
            metadata={},
            main_system_hash=main_hash,
        )
        is True
    )


def test_short_answer_turn_with_history_stays_foreground_despite_mismatch():
    """Issue #454: a second client's conversation with 30-token answers is
    not a title job once it carries assistant history; it must keep its
    session and bank entry instead of re-prefilling every turn."""
    es = _engine_session()
    main_hash = es.hash_text("main chat system")
    messages = [
        {"role": "system", "content": "Assistant S2. filler filler filler"},
        {"role": "user", "content": "Turn 1: reply with just the number 1."},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "Turn 2: reply with just the number 2."},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=30,
            headers={},
            metadata={},
            main_system_hash=main_hash,
        )
        is False
    )
    # The explicit task markers still win regardless of shape.
    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=30,
            headers={},
            metadata={"task": "title_generation"},
            main_system_hash=main_hash,
        )
        is True
    )


# --- model-aware auto budget (v2, founder memory ruling 2026-07-05) ----------

GIB = 1024**3


def _es_with_ram(monkeypatch, total_ram_bytes):
    es = _engine_session()
    monkeypatch.setattr(
        es, "_detect_total_ram_bytes_for_session_bank", lambda: total_ram_bytes
    )
    return es


def test_auto_budget_is_half_the_post_model_surplus(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 64 * GIB)
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (int(45 * GIB * 0.5), True)


def test_auto_budget_capped_on_big_machines(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "auto")
    es = _es_with_ram(monkeypatch, 128 * GIB)
    # 0.5 * (128 - 19) = 54.5G -> capped at 48G
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (48 * GIB, True)


def test_auto_budget_floors_when_model_fills_ram(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 16 * GIB)
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (1 * GIB, True)


def test_auto_budget_small_machine_gets_small_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 36 * GIB)
    # 0.5 * (36 - 19) = 8.5G — NOT the old flat 24G default.
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (int(17 * GIB * 0.5), True)


def test_auto_budget_unknown_model_falls_back_to_legacy_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 128 * GIB)
    from mtplx.session_bank import DEFAULT_MAX_BYTES

    assert es.resolve_session_bank_max_bytes(None) == (DEFAULT_MAX_BYTES, False)


def test_auto_budget_unknown_ram_falls_back_to_legacy_default(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, None)
    from mtplx.session_bank import DEFAULT_MAX_BYTES

    assert es.resolve_session_bank_max_bytes(19 * GIB) == (DEFAULT_MAX_BYTES, False)


def test_explicit_max_bytes_env_overrides_auto(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "16G")
    es = _es_with_ram(monkeypatch, 128 * GIB)
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (16 * GIB, False)


def test_per_session_auto_is_two_thirds_of_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 128 * GIB)
    # 2/3 of 30G = 20G, inside the >=96G tier ceiling of 24G.
    assert es.resolve_session_bank_per_session_bytes(30 * GIB) == 20 * GIB


def test_per_session_auto_clamped_by_ram_tier_small_box(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 64 * GIB)
    # 2/3 of a 22.5G budget = 15G, but the <96G tier ceiling is 8G (#150:
    # the auto rule alone RAISED the admission gate vs the v1.0.4 flat gate,
    # letting a 64GB box admit snapshots whose restore transients blow RAM).
    assert es.resolve_session_bank_per_session_bytes(int(22.5 * GIB)) == 8 * GIB


def test_per_session_auto_clamped_by_ram_tier_big_box(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 128 * GIB)
    # 2/3 of 48G = 32G, tier ceiling >=96G is 24G.
    assert es.resolve_session_bank_per_session_bytes(48 * GIB) == 24 * GIB


def _plan(usable_gib: float, weights_gib: float):
    from types import SimpleNamespace

    return SimpleNamespace(
        available=True,
        usable_bytes=int(usable_gib * GIB),
        model_weights_bytes=int(weights_gib * GIB),
    )


@pytest.mark.parametrize(
    ("ram_gib", "usable_gib", "weights_gib", "budget_gib", "expected_gib", "seat"),
    [
        # PR #496 (Dizzler7) asked for a flat 32 GiB so a 12.2 GiB Q8-KV
        # 100k-token 27B snapshot persists. The plan sizes that ask by the
        # machine: (usable - weights - 3 GiB transients) / 2, under 2/3 of
        # the bank budget.
        (64, 48, 18.6, 26.4, 13.2, "64 GB + 27B: the 12.2 GiB snapshot fits"),
        (128, 96, 18.6, 48, 32, "128 GB + 27B: play allows 37.2, budget rule holds 32"),
        (128, 96, 72, 21, 10.5, "128 GB + Flash-Next: 10.5, below the old 24 flat"),
        (48, 36, 18.6, 14.4, 7.2, "48 GB + 27B: 7.2, below the old 8 flat"),
    ],
)
def test_per_session_auto_is_the_plan_play_ceiling(
    monkeypatch, ram_gib, usable_gib, weights_gib, budget_gib, expected_gib, seat
):
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, ram_gib * GIB)
    plan = _plan(usable_gib, weights_gib)
    resolved = es.resolve_session_bank_per_session_bytes(
        int(budget_gib * GIB), memory_plan=plan
    )
    assert resolved == pytest.approx(expected_gib * GIB, abs=GIB // 1024), seat


def test_per_session_play_ceiling_needs_a_usable_plan(monkeypatch):
    from types import SimpleNamespace

    es = _es_with_ram(monkeypatch, 64 * GIB)
    assert es.per_session_play_ceiling_bytes(None) is None
    assert es.per_session_play_ceiling_bytes(SimpleNamespace(available=False)) is None
    assert (
        es.per_session_play_ceiling_bytes(
            SimpleNamespace(available=True, usable_bytes=0, model_weights_bytes=1)
        )
        is None
    )
    # A plan with no play left still floors at 1 GiB: the live-ref lease
    # covers the rest, the gate never goes below the floor.
    assert es.per_session_play_ceiling_bytes(_plan(20, 19)) == GIB


def test_per_session_explicit_env_wins_over_the_plan_ceiling(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "20G")
    es = _es_with_ram(monkeypatch, 64 * GIB)
    # Explicit values keep their semantics, clamped to the auto budget.
    assert (
        es.resolve_session_bank_per_session_bytes(
            int(26.4 * GIB), memory_plan=_plan(48, 18.6)
        )
        == 20 * GIB
    )


def test_memory_budget_env_tightens_auto_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.setenv("MTPLX_MEMORY_BUDGET", "32G")
    es = _es_with_ram(monkeypatch, 128 * GIB)
    # The declared envelope substitutes for machine RAM in the surplus
    # rule: 0.5 * (32 - 19) = 6.5G.
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (int(13 * GIB * 0.5), True)


def test_memory_budget_env_ignored_when_looser_than_ram(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.setenv("MTPLX_MEMORY_BUDGET", "256G")
    es = _es_with_ram(monkeypatch, 64 * GIB)
    assert es.resolve_session_bank_max_bytes(19 * GIB) == (int(45 * GIB * 0.5), True)


def test_per_session_explicit_env_clamped_to_budget(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "24G")
    es = _engine_session()
    assert es.resolve_session_bank_per_session_bytes(8 * GIB) == 8 * GIB


def test_model_weights_bytes_sums_safetensors(tmp_path):
    es = _engine_session()
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1024)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"y" * 2048)
    (tmp_path / "mtp.safetensors").write_bytes(b"z" * 512)
    (tmp_path / "config.json").write_text("{}")
    assert es.model_weights_bytes(tmp_path) == 1024 + 2048 + 512
    assert es.model_weights_bytes(tmp_path / "missing") is None


def test_manager_uses_auto_budget_for_bank(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _es_with_ram(monkeypatch, 64 * GIB)
    manager = es.EngineSessionManager(model_weights_bytes=19 * GIB)
    expected = int(45 * GIB * 0.5)
    assert manager.bank.max_bytes == expected
    # 2/3 of the 22.5G budget = 15G, clamped to the <96G tier ceiling (#150).
    assert manager.bank.per_session_max_bytes == 8 * GIB


def test_idle_ttl_default_when_unset(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_IDLE_TTL_S", raising=False)
    es = _engine_session()
    assert es.session_bank_idle_ttl_s() == float(es.DEFAULT_IDLE_TTL_S)


def test_idle_ttl_reads_seconds(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_IDLE_TTL_S", "7200")
    es = _engine_session()
    assert es.session_bank_idle_ttl_s() == 7200.0


@pytest.mark.parametrize("raw", ["0", "-5", "0.0"])
def test_idle_ttl_zero_means_never(monkeypatch, raw):
    monkeypatch.setenv("MTPLX_SESSION_BANK_IDLE_TTL_S", raw)
    es = _engine_session()
    assert es.session_bank_idle_ttl_s() == float("inf")


@pytest.mark.parametrize("raw", ["", "   ", "soon", "nan"])
def test_idle_ttl_garbage_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv("MTPLX_SESSION_BANK_IDLE_TTL_S", raw)
    es = _engine_session()
    assert es.session_bank_idle_ttl_s() == float(es.DEFAULT_IDLE_TTL_S)


def test_manager_never_expires_sessions_when_ttl_is_off(monkeypatch):
    """Issue #481: the only idle clock in the daemon is this one; with the
    knob at 0 a day-old session is still warm."""
    monkeypatch.setenv("MTPLX_SESSION_BANK_IDLE_TTL_S", "0")
    es = _engine_session()
    session = es.EngineSession("s-481", idle_ttl_s=es.session_bank_idle_ttl_s())
    assert session.idle_ttl_s == float("inf")
    assert not session.is_stale(now_s=session.last_access_s + 86_400.0)
    monkeypatch.setenv("MTPLX_SESSION_BANK_IDLE_TTL_S", "60")
    session = es.EngineSession("s-481b", idle_ttl_s=es.session_bank_idle_ttl_s())
    assert session.is_stale(now_s=session.last_access_s + 61.0)
    assert not session.is_stale(now_s=session.last_access_s + 59.0)
