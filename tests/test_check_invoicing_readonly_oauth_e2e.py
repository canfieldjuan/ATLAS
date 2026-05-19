from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_invoicing_readonly_oauth_e2e.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_invoicing_readonly_oauth_e2e",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "issuer_url": "https://atlas.example.com/invoicing-readonly",
        "resource_url": "https://atlas.example.com/invoicing-readonly/mcp",
        "approval_token": "approval-token-with-enough-entropy",
        "redirect_uri": "https://chat.openai.com/aip/callback",
        "scope": "invoices.read",
        "timeout": 10.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_config_from_args_requires_values() -> None:
    module = _load_script_module()

    try:
        module._config_from_args(
            _args(
                issuer_url="",
                resource_url="",
                approval_token="",
            )
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")

    assert "--issuer-url" in message
    assert "--resource-url" in message
    assert "--approval-token" in message


def test_pkce_challenge_uses_s256_urlsafe_without_padding() -> None:
    module = _load_script_module()

    expected = base64.urlsafe_b64encode(hashlib.sha256(b"verifier-1").digest()).decode().rstrip("=")

    assert module._pkce_challenge("verifier-1") == expected
    assert not module._pkce_challenge("verifier-1").endswith("=")


def test_tool_surface_errors_accept_exact_readonly_tools() -> None:
    module = _load_script_module()

    assert module._tool_surface_errors(set(module.EXPECTED_READONLY_TOOLS)) == []


def test_tool_surface_errors_reject_extra_and_mutating_tools() -> None:
    module = _load_script_module()
    tools = set(module.EXPECTED_READONLY_TOOLS)
    tools.update({"approve_and_send", "surprise_tool"})

    errors = module._tool_surface_errors(tools)

    assert "unexpected tools exposed: approve_and_send, surprise_tool" in errors
    assert "mutating tools exposed: approve_and_send" in errors


def test_register_client_requires_client_secret(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())

    monkeypatch.setattr(module, "_post_json", lambda *_args, **_kwargs: (201, {}, {"client_id": "client-1"}))

    try:
        module._register_client(config)
    except RuntimeError as exc:
        assert "missing client_id or client_secret" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_start_authorization_requires_approval_redirect(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())
    client = module.RegisteredClient(client_id="client-1", client_secret="secret-1")

    monkeypatch.setattr(module, "_get_no_redirect", lambda *_args, **_kwargs: (302, {"Location": "https://bad.example.com"}, ""))

    try:
        module._start_authorization(config, client)
    except RuntimeError as exc:
        assert "did not redirect to the approval page" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_approve_authorization_extracts_code_without_following_chatgpt_redirect(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())
    auth = module.AuthorizationRequest(
        approval_url="https://atlas.example.com/invoicing-readonly/oauth/approve?request_id=req-1",
        request_id="req-1",
        verifier="verifier-1",
    )

    def fake_post_form(url, payload, timeout, *, follow_redirects=False):
        assert url == "https://atlas.example.com/invoicing-readonly/oauth/approve"
        assert payload["approval_token"] == "approval-token-with-enough-entropy"
        assert follow_redirects is False
        return 302, {"Location": "https://chat.openai.com/aip/callback?code=code-1&state=state-1"}, ""

    monkeypatch.setattr(module, "_post_form", fake_post_form)

    assert module._approve_authorization(config, auth) == "code-1"


def test_exchange_token_requires_bearer_access_token(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())
    client = module.RegisteredClient(client_id="client-1", client_secret="secret-1")
    auth = module.AuthorizationRequest(approval_url="url", request_id="req-1", verifier="verifier-1")

    monkeypatch.setattr(
        module,
        "_post_form",
        lambda *_args, **_kwargs: (200, {}, '{"token_type":"Bearer","scope":"invoices.read"}'),
    )

    try:
        module._exchange_token(config, client, auth, "code-1")
    except RuntimeError as exc:
        assert "missing access_token" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_main_requires_config_before_network(monkeypatch, capsys) -> None:
    module = _load_script_module()

    async def fail_run(*_args, **_kwargs):
        raise AssertionError("network touched")

    monkeypatch.delenv("ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL", raising=False)
    monkeypatch.delenv("ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL", raising=False)
    monkeypatch.delenv("ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN", raising=False)
    monkeypatch.setattr(module, "_run_smoke", fail_run)

    result = module._main([])

    captured = capsys.readouterr()
    assert result == 2
    assert "--issuer-url" in captured.err


def test_main_reports_success_without_printing_secrets(monkeypatch, capsys) -> None:
    module = _load_script_module()

    async def fake_run(_config):
        return set(module.EXPECTED_READONLY_TOOLS)

    monkeypatch.setattr(module, "_run_smoke", fake_run)

    result = module._main(
        [
            "--issuer-url",
            "https://atlas.example.com/invoicing-readonly",
            "--resource-url",
            "https://atlas.example.com/invoicing-readonly/mcp",
            "--approval-token",
            "secret-approval-token-value",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "OAuth e2e smoke completed" in captured.out
    assert "secret-approval-token-value" not in captured.out
    assert "get_invoice" in captured.out


def test_main_reports_tool_surface_errors(monkeypatch, capsys) -> None:
    module = _load_script_module()

    async def fake_run(_config):
        return {"get_invoice", "approve_and_send"}

    monkeypatch.setattr(module, "_run_smoke", fake_run)

    result = module._main(
        [
            "--issuer-url",
            "https://atlas.example.com/invoicing-readonly",
            "--resource-url",
            "https://atlas.example.com/invoicing-readonly/mcp",
            "--approval-token",
            "secret-approval-token-value",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "missing read-only tools:" in captured.err
    assert "mutating tools exposed: approve_and_send" in captured.err


def test_run_smoke_sequence_uses_token_for_tool_list(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())
    calls: list[str] = []

    def fake_register(_config):
        calls.append("register")
        return module.RegisteredClient(client_id="client-1", client_secret="secret-1")

    def fake_start(_config, client):
        calls.append(f"authorize:{client.client_id}")
        return module.AuthorizationRequest(approval_url="url", request_id="req-1", verifier="verifier-1")

    def fake_approve(_config, auth):
        calls.append(f"approve:{auth.request_id}")
        return "code-1"

    def fake_exchange(_config, client, auth, code):
        calls.append(f"token:{client.client_id}:{auth.request_id}:{code}")
        return module.TokenResult(access_token="access-token-1", scope="invoices.read")

    async def fake_list(_config, token):
        calls.append(f"list:{token}")
        return set(module.EXPECTED_READONLY_TOOLS)

    monkeypatch.setattr(module, "_register_client", fake_register)
    monkeypatch.setattr(module, "_start_authorization", fake_start)
    monkeypatch.setattr(module, "_approve_authorization", fake_approve)
    monkeypatch.setattr(module, "_exchange_token", fake_exchange)
    monkeypatch.setattr(module, "_list_mcp_tools", fake_list)

    result = asyncio.run(module._run_smoke(config))

    assert result == set(module.EXPECTED_READONLY_TOOLS)
    assert calls == [
        "register",
        "authorize:client-1",
        "approve:req-1",
        "token:client-1:req-1:code-1",
        "list:access-token-1",
    ]
