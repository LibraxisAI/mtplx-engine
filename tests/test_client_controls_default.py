"""MTPLX_CLIENT_CONTROLS_DEFAULT: anonymous body params get applied by default.

Default is 'honor' since 2.5.3 (OpenAI-API semantics for anonymous clients;
issue #241 receipts). Managed surfaces stay server-owned in BOTH modes, and
MTPLX_CLIENT_CONTROLS_DEFAULT=hints restores the pre-2.5.3 policy.
"""

from __future__ import annotations

from mtplx.server.openai import (
    _client_controls_allowed,
    _client_thinking_controls_allowed,
)


def test_default_honors_anonymous_controls(monkeypatch):
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    assert _client_controls_allowed({}, {}) is True


def test_header_opt_in_works_under_hints_mode(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "hints")
    assert _client_controls_allowed({"x-mtplx-allow-client-controls": "1"}, {}) is True


def test_hints_mode_ignores_anonymous_controls(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "hints")
    assert _client_controls_allowed({}, {}) is False


def test_default_keeps_managed_surfaces_server_owned(monkeypatch):
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    assert _client_controls_allowed({"x-mtplx-client": "opencode"}, {}) is False
    assert _client_controls_allowed({"x-mtplx-client": "chat"}, {}) is False


def test_honor_mode_keeps_managed_surfaces_server_owned(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "honor")
    assert _client_controls_allowed({"x-mtplx-client": "opencode"}, {}) is False
    assert _client_controls_allowed({"x-mtplx-client": "chat"}, {}) is False


def test_unknown_value_falls_back_to_honor(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "yolo")
    assert _client_controls_allowed({}, {}) is True


def test_managed_surfaces_keep_thinking_controls(monkeypatch):
    """Effort pickers in managed clients (OpenCode/Pi/app) govern the request
    even though their sampler params stay server-owned."""
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    for hint in ("opencode", "pi", "chat"):
        headers = {"x-mtplx-client": hint}
        assert _client_controls_allowed(headers, {}) is False
        assert _client_thinking_controls_allowed(headers, {}) is True


def test_anonymous_thinking_controls_follow_the_general_contract(monkeypatch):
    monkeypatch.delenv("MTPLX_CLIENT_CONTROLS_DEFAULT", raising=False)
    assert _client_thinking_controls_allowed({}, {}) is True
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "hints")
    assert _client_thinking_controls_allowed({}, {}) is False
    assert (
        _client_thinking_controls_allowed(
            {"x-mtplx-allow-client-controls": "1"}, {}
        )
        is True
    )


def test_explicit_app_policy_is_scoped_and_reversible(monkeypatch):
    from types import SimpleNamespace

    # App launch policy must not leak into normal API traffic or native chat.
    monkeypatch.setenv("MTPLX_MANAGED_CLIENT_CONTROLS", "app")
    state = SimpleNamespace(args=SimpleNamespace())
    for hint in ("pi", "opencode", "hermes", "openwebui"):
        headers = {"x-mtplx-client": hint}
        assert not _client_controls_allowed(headers, {}, state=state)
        assert not _client_thinking_controls_allowed(headers, {}, state=state)
    assert _client_controls_allowed({}, {}, state=state)
    assert _client_thinking_controls_allowed({}, {}, state=state)
    assert _client_thinking_controls_allowed({"x-mtplx-client": "chat"}, {}, state=state)

    # A live update takes precedence over launch environment, in both directions.
    state.args.managed_client_controls = "client"
    for hint in ("pi", "opencode", "hermes", "openwebui"):
        headers = {"x-mtplx-client": hint}
        assert _client_controls_allowed(headers, {}, state=state)
        assert _client_thinking_controls_allowed(headers, {}, state=state)
    state.args.managed_client_controls = "app"
    assert not _client_thinking_controls_allowed({"x-mtplx-client": "pi"}, {}, state=state)


def test_pi_mirror_tracks_app_and_restores_client_choice(tmp_path):
    import json
    import shutil
    import subprocess
    import pytest
    from mtplx.pi import build_pi_settings_extension_source

    if not shutil.which("node"):
        pytest.skip("node required to execute the real Pi extension")
    module = tmp_path / "mirror.mjs"
    module.write_text(build_pi_settings_extension_source().replace(": any", ""))
    script = tmp_path / "test.mjs"
    script.write_text('''
import register from MODULE;
import assert from "node:assert/strict";
const handlers = {};
let thinking = "medium", status, settings;
register({on: (name, fn) => handlers[name] = fn,
  getThinkingLevel: () => thinking, setThinkingLevel: value => thinking = value});
const ctx = {model: {provider: "mtplx", id: "mtplx-test", baseUrl: "http://localhost:8000/v1"},
  ui: {setStatus: (_name, value) => status = value},
  sessionManager: {getSessionId: () => "session"},
  modelRegistry: {getApiKeyAndHeaders: async () => ({ok: true})}, isIdle: () => true};
globalThis.fetch = async (url, options) => {
  assert.equal(options.headers.Connection, "close");
  assert.equal(url, "http://localhost:8000/v1/mtplx/settings");
  return {ok: true, json: async () => settings};
};
settings = {managed_client_controls: "app", enable_thinking: true, reasoning_effort: "xhigh"};
await handlers.session_start({}, ctx);
assert.equal(thinking, "xhigh");
assert.match(status, /MTPLX controls reasoning: xhigh/);
settings.enable_thinking = false;
await handlers.before_agent_start({}, ctx);
assert.equal(thinking, "off");
settings.managed_client_controls = "client";
await handlers.before_agent_start({}, ctx);
assert.equal(thinking, "medium");
assert.equal(status, undefined);
thinking = "low";
await handlers.before_agent_start({}, ctx);
assert.equal(thinking, "low");
settings.managed_client_controls = "app"; settings.enable_thinking = true;
await handlers.before_agent_start({}, ctx);
assert.equal(thinking, "xhigh");
settings.managed_client_controls = "client";
await handlers.before_agent_start({}, ctx);
assert.equal(thinking, "low");
globalThis.fetch = async () => ({ok: false, status: 503});
await handlers.before_agent_start({}, ctx);
assert.match(status, /sync failed.*503/);
assert.equal(thinking, "low");
ctx.model = {...ctx.model, provider: "another-provider"};
await handlers.before_agent_start({}, ctx);
assert.equal(thinking, "low");
assert.equal(status, undefined);
handlers.session_shutdown();
'''.replace('MODULE', json.dumps(str(module))))
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_settings_mirror_installs_without_replacing_custom_request_bridge(tmp_path):
    from mtplx.pi import write_pi_models_config

    custom = tmp_path / "extensions" / "mtplx-request-policy.ts"
    custom.parent.mkdir()
    custom.write_text("export default function myCustomBridge(pi) {}\n")
    result = write_pi_models_config(
        base_url="http://localhost:8000/v1", model_id="mtplx-test", path=tmp_path / "models.json"
    )
    assert custom.read_text() == "export default function myCustomBridge(pi) {}\n"
    from pathlib import Path
    mirror = Path(result["settings_extension_path"])
    assert mirror != custom
    assert 'pi.setThinkingLevel(effort)' in mirror.read_text()
