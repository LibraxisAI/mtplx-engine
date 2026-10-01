"""Pack defaults through the public serve handoff, without loading a model."""

import json
import shlex

import pytest

from mtplx.backends.registry import RuntimeContract, load_runtime_contract
from mtplx.cli import build_parser
from mtplx.commands import public


REASON = "The draft head measured no consistent speed-up at the native sampler."
EVIDENCE = {
    "measured_at": "2026-09-21T17:09-07:00",
    "rows": [{"hardware": "test fixture", "ar_tokens_per_second": 32.2,
              "mtp_tokens_per_second": 30.3}],
}
NOTICE = (
    "Serving plain decoding as recommended by this pack (2026-09-21). "
    f"{REASON} Pass --generation-mode mtp to use the head."
)


def _contract(mode):
    data = {
        "mtplx_version": "2.11.4", "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 3, "mtp_depth_default": 1,
        "recommended_profile": "turbo", "exactness_baseline": {}, "verified_on": {},
    }
    if mode is not None:
        data.update(recommended_generation_mode=mode,
                    recommended_generation_mode_reason=REASON,
                    recommended_generation_mode_evidence=EVIDENCE)
    return data


@pytest.mark.parametrize("mode", [None, "mtp", "ar"])
def test_runtime_contract_round_trip_preserves_generation_recommendation(tmp_path, mode):
    raw = _contract(mode)
    (tmp_path / "mtplx_runtime.json").write_text(json.dumps(raw))
    contract, error = load_runtime_contract(tmp_path)
    assert error is None
    assert contract.recommended_generation_mode == mode
    serialized = contract.to_dict()
    assert RuntimeContract.from_dict(serialized).to_dict() == serialized
    for key in ("recommended_generation_mode", "recommended_generation_mode_reason",
                "recommended_generation_mode_evidence"):
        if mode is None:
            assert key not in serialized
        else:
            assert serialized[key] == raw[key]


@pytest.mark.parametrize("key,value", [
    ("recommended_generation_mode", "auto"),
    ("recommended_generation_mode", "typo"),
    ("recommended_generation_mode_reason", 42),
    ("recommended_generation_mode_evidence", []),
])
def test_runtime_contract_rejects_malformed_generation_recommendation(key, value):
    data = _contract("ar")
    data[key] = value
    with pytest.raises(ValueError, match=key):
        RuntimeContract.from_dict(data)


MODE_FLAGS = [
    pytest.param([], None, True, id="no-flag"),
    pytest.param(["--generation-mode", "auto"], None, True, id="auto"),
    pytest.param(["--generation-mode", "mtp"], "mtp", True, id="explicit-mtp"),
    pytest.param(["--generation-mode", "ar"], "ar", True, id="explicit-ar"),
    pytest.param(["--no-mtp"], "ar", True, id="no-mtp"),
    pytest.param(["--stock-ar"], "ar", False, id="stock-ar"),
]


@pytest.mark.parametrize("mode", [None, "mtp", "ar"], ids=["absent", "mtp", "ar"])
@pytest.mark.parametrize("flags,explicit_mode,loads_head", MODE_FLAGS)
@pytest.mark.parametrize("metadata_fallback", [False, True], ids=["typed", "metadata"])
def test_serve_pack_generation_mode_matrix(
    tmp_path, monkeypatch, capsys, mode, flags, explicit_mode, loads_head, metadata_fallback
):
    raw = _contract(mode)
    (tmp_path / "mtplx_runtime.json").write_text(json.dumps(raw))
    typed = RuntimeContract.from_dict(raw).to_dict()
    if metadata_fallback:
        typed = {key: value for key, value in typed.items()
                 if not key.startswith("recommended_generation_mode")}
    inspection = {"model_dir": str(tmp_path), "compatibility": {
        "tier": "verified", "can_run": True, "runtime_contract": typed,
    }}
    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda *a, **kw: (str(tmp_path), None))
    monkeypatch.setattr(public, "_model_gate", lambda *a, **kw: (inspection, None))
    args = build_parser().parse_args([
        "serve", "--model", str(tmp_path), "--yes",
        # Deliberate profile mismatch: mode and depth belong to the pack.
        "--profile", "sustained", *flags,
    ])
    # The start wrapper uses these controls; serve itself has no dry-run flags.
    args.dry_run = True
    args.json = True
    assert public.cmd_serve_public(args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    expected = explicit_mode or mode or "mtp"
    assert payload["generation_mode"] == expected
    cmd = shlex.split(payload["server_command"])
    assert cmd[cmd.index("--generation-mode") + 1] == expected
    assert ("--stock-ar" not in cmd and "--no-load-mtp" not in cmd) is loads_head
    assert args.load_mtp is True  # stock-ar is resolved by the child's parser.
    assert cmd[cmd.index("--depth") + 1] == "1"
    notice_expected = mode == "ar" and explicit_mode is None
    assert captured.err.splitlines() == ([NOTICE] if notice_expected else [])
    # Resolution may be re-entered by callers; do not announce twice.
    public._apply_model_contract_generation_mode_default(args, inspection)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("flags,expected", [
    (["--mtp"], "mtp"),
    (["--load-mtp"], "mtp"),
    (["--no-load-mtp"], "ar"),
    (["--generation-mode", "auto", "--load-mtp"], "ar"),
    (["--generation-mode", "auto", "--no-load-mtp"], "ar"),
    (["--generation-mode", "auto", "--no-mtp"], "ar"),
])
def test_explicit_load_controls_keep_their_semantics(flags, expected, capsys):
    args = build_parser().parse_args(["serve", *flags])
    load_mtp = args.load_mtp
    public._apply_model_contract_generation_mode_default(args, {"runtime_contract": _contract("ar")})
    assert public._generation_mode_from_args(args) == expected
    assert args.load_mtp is load_mtp


def test_typed_recommendation_precedes_top_level_metadata(tmp_path):
    (tmp_path / "mtplx_runtime.json").write_text(json.dumps(_contract("ar")))
    args = build_parser().parse_args(["serve"])
    public._apply_model_contract_generation_mode_default(args, {
        "model_dir": str(tmp_path), "runtime_contract": _contract("mtp"),
    }, printer=lambda line: pytest.fail(line))
    assert public._generation_mode_from_args(args) == "mtp"


def test_recommendation_without_evidence_does_not_claim_a_measurement(capsys):
    args = build_parser().parse_args(["serve"])
    public._apply_model_contract_generation_mode_default(args, {
        "runtime_contract": {"recommended_generation_mode": "ar"},
    })
    assert capsys.readouterr().out.splitlines() == [
        "Serving plain decoding as recommended by this pack. "
        "Pass --generation-mode mtp to use the head."
    ]


def test_recommended_ar_reaches_daemon_with_head_loaded(tmp_path, monkeypatch, capsys):
    inspection = {"runtime_contract": _contract("ar")}
    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda *a, **kw: (str(tmp_path), None))
    monkeypatch.setattr(public, "_model_gate", lambda *a, **kw: (inspection, None))
    monkeypatch.setattr(public, "_port_is_busy", lambda *a: False)
    monkeypatch.delenv("MTPLX_APP_PARENT_PID", raising=False)
    calls = []

    def capture_exec(_executable, cmd, _env):
        calls.append(cmd)
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", capture_exec)
    args = build_parser().parse_args([
        "serve", "--model", str(tmp_path), "--yes", "--fan-mode", "default",
    ])
    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)
    assert exc.value.code == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[cmd.index("--generation-mode") + 1] == "ar"
    assert "--no-load-mtp" not in cmd and "--stock-ar" not in cmd
    assert args.load_mtp is True
    assert capsys.readouterr().out.splitlines().count(NOTICE) == 1
