from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_content_ops_marketer_verify_oauth_e2e.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_content_ops_marketer_verify_oauth_e2e",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "issuer_url": "https://atlas.example.com/content-ops-marketer",
        "resource_url": "https://atlas.example.com/content-ops-marketer/mcp",
        "approval_token": "approval-token-with-enough-entropy",
        "approval_token_file": "",
        "redirect_uri": "https://chat.openai.com/aip/callback",
        "scope": "content_ops.review.verify",
        "timeout": 10.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_content_ops_e2e_defaults_to_verify_scope() -> None:
    module = _load_script_module()

    assert module.DEFAULT_SCOPE == "content_ops.review.verify"
    assert module.EXPECTED_TOOLS == {"verify_draft"}


def test_config_from_args_requires_content_ops_values() -> None:
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
    assert module.ISSUER_ENV in message
    assert "--resource-url" in message
    assert module.RESOURCE_ENV in message
    assert "--approval-token" in message
    assert module.APPROVAL_ENV in message


def test_config_from_args_reads_approval_token_file(tmp_path) -> None:
    module = _load_script_module()
    token_file = tmp_path / "content-ops-token"
    token_file.write_text("approval-token-from-local-secret-file\n")

    config = module._config_from_args(
        _args(
            approval_token="",
            approval_token_file=str(token_file),
        )
    )

    assert config.approval_token == "approval-token-from-local-secret-file"


def test_config_from_args_rejects_empty_approval_token_file(tmp_path) -> None:
    module = _load_script_module()
    token_file = tmp_path / "content-ops-token"
    token_file.write_text("\n")

    try:
        module._config_from_args(_args(approval_token_file=str(token_file)))
    except ValueError as exc:
        assert "is empty" in str(exc)
        assert "content-ops-token" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")


def test_register_client_uses_content_ops_client_name(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())
    captured = {}

    def fake_post_json(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return 201, {}, {"client_id": "client-1", "client_secret": "secret-1"}

    monkeypatch.setattr(module, "_post_json", fake_post_json)

    client = module._register_client(config)

    assert client.client_id == "client-1"
    assert captured["url"] == "https://atlas.example.com/content-ops-marketer/register"
    assert captured["payload"]["client_name"] == (
        "Atlas Content Ops marketer verify OAuth e2e smoke"
    )
    assert captured["payload"]["scope"] == "content_ops.review.verify"


def test_register_client_requires_client_secret(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())

    monkeypatch.setattr(
        module,
        "_post_json",
        lambda *_args, **_kwargs: (201, {}, {"client_id": "client-1"}),
    )

    try:
        module._register_client(config)
    except RuntimeError as exc:
        assert "missing client_id or client_secret" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_exchange_token_requires_bearer_access_token(monkeypatch) -> None:
    module = _load_script_module()
    config = module._config_from_args(_args())
    client = module.RegisteredClient(client_id="client-1", client_secret="secret-1")
    auth = module.AuthorizationRequest(approval_url="url", request_id="req-1", verifier="verifier-1")

    monkeypatch.setattr(
        module._draft_e2e,
        "_post_form",
        lambda *_args, **_kwargs: (
            200,
            {},
            '{"token_type":"Bearer","scope":"content_ops.review.verify"}',
        ),
    )

    try:
        module._exchange_token(config, client, auth, "code-1")
    except RuntimeError as exc:
        assert "missing access_token" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_tool_surface_errors_accept_exact_verify_tool() -> None:
    module = _load_script_module()

    assert module._tool_surface_errors({"verify_draft"}) == []


def test_tool_surface_errors_reject_extra_and_denied_tools() -> None:
    module = _load_script_module()

    errors = module._tool_surface_errors({"verify_draft", "start_brief", "surprise_tool"})

    assert "unexpected tools exposed: start_brief, surprise_tool" in errors
    assert "denied tools exposed: start_brief" in errors


def test_main_requires_config_before_network(monkeypatch, capsys) -> None:
    module = _load_script_module()

    async def fail_run(*_args, **_kwargs):
        raise AssertionError("network touched")

    monkeypatch.delenv(module.ISSUER_ENV, raising=False)
    monkeypatch.delenv(module.RESOURCE_ENV, raising=False)
    monkeypatch.delenv(module.APPROVAL_ENV, raising=False)
    monkeypatch.setattr(module, "_run_smoke", fail_run)

    result = module._main([])

    captured = capsys.readouterr()
    assert result == 2
    assert "--issuer-url" in captured.err


def test_main_reports_success_without_printing_secrets(monkeypatch, capsys) -> None:
    module = _load_script_module()

    async def fake_run(_config):
        return {"verify_draft"}

    monkeypatch.setattr(module, "_run_smoke", fake_run)

    result = module._main(
        [
            "--issuer-url",
            "https://atlas.example.com/content-ops-marketer",
            "--resource-url",
            "https://atlas.example.com/content-ops-marketer/mcp",
            "--approval-token",
            "secret-approval-token-value",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Content Ops marketer verify OAuth e2e smoke completed" in captured.out
    assert "secret-approval-token-value" not in captured.out
    assert "verify_draft" in captured.out


def test_main_reports_tool_surface_errors(monkeypatch, capsys) -> None:
    module = _load_script_module()

    async def fake_run(_config):
        return {"verify_draft", "start_brief"}

    monkeypatch.setattr(module, "_run_smoke", fake_run)

    result = module._main(
        [
            "--issuer-url",
            "https://atlas.example.com/content-ops-marketer",
            "--resource-url",
            "https://atlas.example.com/content-ops-marketer/mcp",
            "--approval-token",
            "secret-approval-token-value",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "unexpected tools exposed: start_brief" in captured.err
    assert "denied tools exposed: start_brief" in captured.err


def test_run_smoke_sequence_lists_tools_without_calling_verify_draft(monkeypatch) -> None:
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
        return module.TokenResult(access_token="access-token-1", scope="content_ops.review.verify")

    async def fake_list(_config, token):
        calls.append(f"list:{token}")
        return {"verify_draft"}

    monkeypatch.setattr(module, "_register_client", fake_register)
    monkeypatch.setattr(module, "_start_authorization", fake_start)
    monkeypatch.setattr(module, "_approve_authorization", fake_approve)
    monkeypatch.setattr(module, "_exchange_token", fake_exchange)
    monkeypatch.setattr(module, "_list_mcp_tools", fake_list)

    result = asyncio.run(module._run_smoke(config))

    assert result == {"verify_draft"}
    assert calls == [
        "register",
        "authorize:client-1",
        "approve:req-1",
        "token:client-1:req-1:code-1",
        "list:access-token-1",
    ]
