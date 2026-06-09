from __future__ import annotations

import importlib.util
import io
import shlex
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_ID_ENV = "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID"
CHECKER_ENV_KEYS = (
    "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL",
    "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL",
    "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN",
)


def _load_script_module(script_name: str):
    path = ROOT / "scripts" / script_name
    module_name = script_name.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_launcher_env(launcher) -> dict[str, str]:
    return {
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE": "oauth",
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL": launcher.DEFAULT_ISSUER_URL,
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL": launcher.DEFAULT_RESOURCE_URL,
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN": (
            "approval-token-with-enough-entropy"
        ),
        ACCOUNT_ID_ENV: "acct-content-ops-demo",
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT": launcher.DEFAULT_PORT,
    }


def _operator_guidance(launcher) -> str:
    config = launcher.LaunchConfig(
        env=_valid_launcher_env(launcher),
        python="/venv/bin/python",
        host="0.0.0.0",
        port="9000",
        dry_run=True,
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        launcher._print_operator_guidance(config)
    return buffer.getvalue()


def _command_for_script(output: str, script_name: str) -> list[str]:
    lines = output.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if script_name in line)
    except StopIteration as exc:  # pragma: no cover - assertion clarity
        raise AssertionError(f"operator guidance missing {script_name}") from exc

    command_parts: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.endswith("\\"):
            command_parts.append(stripped[:-1].strip())
            continue
        command_parts.append(stripped)
        break
    return shlex.split(" ".join(command_parts))


def _clear_checker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in CHECKER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_launcher_discovery_command_satisfies_discovery_checker_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_checker_env(monkeypatch)
    launcher = _load_script_module("start_content_ops_marketer_verify_oauth_server.py")
    discovery = _load_script_module("check_content_ops_marketer_verify_oauth_discovery.py")

    command = _command_for_script(
        _operator_guidance(launcher),
        "scripts/check_content_ops_marketer_verify_oauth_discovery.py",
    )
    args = discovery._build_parser().parse_args(command[2:])

    assert command[:2] == [
        ".venv/bin/python",
        "scripts/check_content_ops_marketer_verify_oauth_discovery.py",
    ]
    assert args.issuer_url == launcher.DEFAULT_ISSUER_URL
    assert args.resource_url == launcher.DEFAULT_RESOURCE_URL


def test_launcher_e2e_command_satisfies_e2e_checker_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_checker_env(monkeypatch)
    launcher = _load_script_module("start_content_ops_marketer_verify_oauth_server.py")
    e2e = _load_script_module("check_content_ops_marketer_verify_oauth_e2e.py")

    command = _command_for_script(
        _operator_guidance(launcher),
        "scripts/check_content_ops_marketer_verify_oauth_e2e.py",
    )
    args = e2e._build_parser().parse_args(command[2:])

    assert command[:2] == [
        ".venv/bin/python",
        "scripts/check_content_ops_marketer_verify_oauth_e2e.py",
    ]
    assert args.issuer_url == launcher.DEFAULT_ISSUER_URL
    assert args.resource_url == launcher.DEFAULT_RESOURCE_URL
    assert args.client_profile == "claude-rich"
    assert args.approval_token_file == "/path/to/local-approval-token"
    assert args.approval_token == ""
    assert "start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py" in (
        _operator_guidance(launcher)
    )


def test_launcher_required_env_covers_server_oauth_account_binding(monkeypatch) -> None:
    launcher = _load_script_module("start_content_ops_marketer_verify_oauth_server.py")
    from atlas_brain.config import settings
    from atlas_brain.mcp import content_ops_marketer_verify_server as verify

    monkeypatch.setattr(verify, "_oauth_provider", None)
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "oauth")
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_issuer_url",
        "https://atlas.example.com/content-ops-marketer",
    )
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_resource_url",
        "https://atlas.example.com/content-ops-marketer/mcp",
    )
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_approval_token",
        "approval-token-with-enough-entropy",
    )
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_account_id", "")

    env = _valid_launcher_env(launcher)
    env[ACCOUNT_ID_ENV] = " "

    assert ACCOUNT_ID_ENV in launcher.REQUIRED_KEYS
    assert f"{ACCOUNT_ID_ENV}=acct-content-ops-demo" in launcher._masked_env_report(
        _valid_launcher_env(launcher)
    )
    assert launcher._validate_env(env) == [
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID is required in oauth mode"
    ]
    with pytest.raises(RuntimeError, match="ACCOUNT_ID"):
        verify._configure_oauth_auth()


def test_chatgpt_adapter_launcher_discovery_command_satisfies_checker_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_checker_env(monkeypatch)
    launcher = _load_script_module(
        "start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py"
    )
    discovery = _load_script_module("check_content_ops_marketer_verify_oauth_discovery.py")

    command = _command_for_script(
        _operator_guidance(launcher),
        "scripts/check_content_ops_marketer_verify_oauth_discovery.py",
    )
    args = discovery._build_parser().parse_args(command[2:])

    assert command[:2] == [
        ".venv/bin/python",
        "scripts/check_content_ops_marketer_verify_oauth_discovery.py",
    ]
    assert args.issuer_url == launcher.DEFAULT_ISSUER_URL
    assert args.resource_url == launcher.DEFAULT_RESOURCE_URL


def test_chatgpt_adapter_launcher_e2e_command_satisfies_checker_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_checker_env(monkeypatch)
    launcher = _load_script_module(
        "start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py"
    )
    e2e = _load_script_module("check_content_ops_marketer_verify_oauth_e2e.py")

    command = _command_for_script(
        _operator_guidance(launcher),
        "scripts/check_content_ops_marketer_verify_oauth_e2e.py",
    )
    args = e2e._build_parser().parse_args(command[2:])

    assert command[:2] == [
        ".venv/bin/python",
        "scripts/check_content_ops_marketer_verify_oauth_e2e.py",
    ]
    assert args.issuer_url == launcher.DEFAULT_ISSUER_URL
    assert args.resource_url == launcher.DEFAULT_RESOURCE_URL
    assert args.client_profile == "chatgpt-search-fetch"
    assert args.approval_token_file == "/path/to/local-approval-token"
    assert args.approval_token == ""


def test_chatgpt_adapter_launcher_dual_client_command_satisfies_checker_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_checker_env(monkeypatch)
    launcher = _load_script_module(
        "start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py"
    )
    dual = _load_script_module("check_content_ops_marketer_verify_dual_client_rollout.py")

    command = _command_for_script(
        _operator_guidance(launcher),
        "scripts/check_content_ops_marketer_verify_dual_client_rollout.py",
    )
    args = dual._build_parser().parse_args(command[2:])

    assert command[:2] == [
        ".venv/bin/python",
        "scripts/check_content_ops_marketer_verify_dual_client_rollout.py",
    ]
    assert args.rich_issuer_url == launcher.rich_launcher.DEFAULT_ISSUER_URL
    assert args.rich_resource_url == launcher.rich_launcher.DEFAULT_RESOURCE_URL
    assert args.chatgpt_adapter_issuer_url == launcher.DEFAULT_ISSUER_URL
    assert args.chatgpt_adapter_resource_url == launcher.DEFAULT_RESOURCE_URL
    assert args.approval_token_file == "/path/to/local-approval-token"
    assert args.approval_token == ""
