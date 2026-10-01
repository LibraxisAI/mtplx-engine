"""Model downloads behind a TLS-inspecting proxy (issue #495).

A company proxy (Zscaler in the report) terminates HTTPS and signs with its
own root. macOS trusts that root through the keychain, so curl and Safari
work. Python verifies against the ``certifi`` bundle and never reads the
keychain, so the download died at setup step 6 of 7 with

    [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get
    local issuer certificate (_ssl.c:1081)

which reads like a network block and says nothing about what to do.

Two things are pinned here. The error is recognised anywhere in the exception
chain (httpx and huggingface_hub wrap it) and turned into steps a person can
follow. And when the ``truststore`` package is present, verification goes
through the operating system, which is how pip has worked since 24.2. Building
a CA bundle by dumping the keychain is deliberately NOT done: that would trust
certificates an administrator marked as untrusted.
"""

from __future__ import annotations

import ssl
import sys
import types

import pytest

from mtplx import hf_loader


def _verify_error() -> ssl.SSLCertVerificationError:
    return ssl.SSLCertVerificationError(
        1,
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "unable to get local issuer certificate (_ssl.c:1081)",
    )


def _wrapped(depth: int) -> BaseException:
    exc: BaseException = _verify_error()
    for level in range(depth):
        try:
            try:
                raise exc
            except BaseException as inner:
                raise RuntimeError(f"transport layer {level}") from inner
        except RuntimeError as outer:
            exc = outer
    return exc


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_a_certificate_failure_is_explained_wherever_it_sits_in_the_chain(depth):
    message = hf_loader._classify_pull_error(_wrapped(depth), "Youssofal/Example")

    assert "SSL_CERT_FILE" in message
    assert "proxy" in message.lower()
    assert "truststore" in message
    # The raw text stays in, so a bug report still carries it.
    assert "CERTIFICATE_VERIFY_FAILED" in message


def test_a_certificate_failure_that_only_survives_as_text_is_still_explained():
    exc = RuntimeError(
        "Cannot connect: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
    )
    assert "SSL_CERT_FILE" in hf_loader._classify_pull_error(exc, "Youssofal/Example")


def test_other_failures_keep_their_own_message():
    assert hf_loader._classify_pull_error(RuntimeError("disk on fire"), "r/x") == "disk on fire"


@pytest.fixture
def fresh_trust_state(monkeypatch):
    monkeypatch.setattr(
        hf_loader,
        "_SYSTEM_TRUST_STATE",
        {"attempted": False, "active": False, "reason": None},
    )
    monkeypatch.delenv("MTPLX_SYSTEM_TRUST", raising=False)


def _fake_truststore(monkeypatch) -> list[str]:
    calls: list[str] = []
    module = types.ModuleType("truststore")
    module.inject_into_ssl = lambda: calls.append("inject")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "truststore", module)
    return calls


def test_the_operating_systems_trust_store_is_used_when_truststore_is_installed(
    monkeypatch, fresh_trust_state
):
    calls = _fake_truststore(monkeypatch)

    first = hf_loader.use_system_trust_store()
    second = hf_loader.use_system_trust_store()

    assert first["active"] is True and second["active"] is True
    assert calls == ["inject"], "injecting twice would stack SSLContext patches"


def test_without_the_package_nothing_changes(monkeypatch, fresh_trust_state):
    monkeypatch.setitem(sys.modules, "truststore", None)  # import raises ImportError

    state = hf_loader.use_system_trust_store()

    assert state["active"] is False
    assert state["reason"] == "truststore_not_installed"


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_it_can_be_switched_off(monkeypatch, fresh_trust_state, value):
    calls = _fake_truststore(monkeypatch)
    monkeypatch.setenv("MTPLX_SYSTEM_TRUST", value)

    state = hf_loader.use_system_trust_store()

    assert state["active"] is False and state["reason"] == "disabled_by_env"
    assert calls == []
