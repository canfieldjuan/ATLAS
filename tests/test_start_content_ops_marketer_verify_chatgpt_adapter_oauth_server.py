from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/start_content_ops_marketer_verify_chatgpt_adapter_oauth_server.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "start_content_ops_marketer_verify_chatgpt_adapter_oauth_server",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "env_file": [],
        "python": "/venv/bin/python",
        "host": None,
        "port": None,
        "issuer_url": None,
        "resource_url": None,
        "approval_token_file": None,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _valid_env(module):
    return {
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE": "oauth",
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL": module.DEFAULT_ISSUER_URL,
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL": module.DEFAULT_RESOURCE_URL,
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN": (
            "approval-token-with-enough-entropy"
        ),
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID": "acct-content-ops-demo",
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT": module.DEFAULT_PORT,
    }


def test_adapter_launcher_defaults_to_chatgpt_path_and_port() -> None:
    module = _load_script_module()

    config = module._build_launch_config(_args())

    assert config.port == "8069"
    assert config.env[module.ADAPTER_PORT_ENV] == "8069"
    assert config.env[module.RUNTIME_PORT_ENV] == "8069"
    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"] == (
        "https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt"
    )
    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"] == (
        "https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer-chatgpt/mcp"
    )


def test_adapter_launcher_ignores_rich_port_from_shared_dotenv(tmp_path) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text("ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT=8068\n")

    config = module._build_launch_config(_args(env_file=[str(env_file)]))

    assert config.port == "8069"
    assert config.env[module.ADAPTER_PORT_ENV] == "8069"
    assert config.env[module.RUNTIME_PORT_ENV] == "8069"


def test_adapter_launcher_accepts_dedicated_adapter_port_env(tmp_path) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT=8068",
                f"{module.ADAPTER_PORT_ENV}=8070",
            ]
        )
        + "\n"
    )

    config = module._build_launch_config(_args(env_file=[str(env_file)]))

    assert config.port == "8070"
    assert config.env[module.ADAPTER_PORT_ENV] == "8070"
    assert config.env[module.RUNTIME_PORT_ENV] == "8070"


def test_adapter_launcher_cli_port_overrides_dedicated_env(tmp_path) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(f"{module.ADAPTER_PORT_ENV}=8070\n")

    config = module._build_launch_config(_args(env_file=[str(env_file)], port="8071"))

    assert config.port == "8071"
    assert config.env[module.ADAPTER_PORT_ENV] == "8071"
    assert config.env[module.RUNTIME_PORT_ENV] == "8071"


def test_adapter_launcher_reuses_hardened_env_validation() -> None:
    module = _load_script_module()
    env = _valid_env(module)
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID"] = " "

    assert module._validate_env(env) == [
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID is required in oauth mode"
    ]


def test_adapter_server_command_uses_chatgpt_adapter_module() -> None:
    module = _load_script_module()
    config = module.LaunchConfig(
        env=_valid_env(module),
        python="/custom/python",
        host="0.0.0.0",
        port="8069",
        dry_run=False,
    )

    assert module._server_command(config) == [
        "/custom/python",
        "-m",
        "atlas_brain.mcp.content_ops_marketer_verify_chatgpt_adapter_server",
        "--sse",
    ]


def test_adapter_guidance_prints_chatgpt_profile_and_masks_secrets(capsys) -> None:
    module = _load_script_module()
    env = _valid_env(module)
    config = module.LaunchConfig(
        env=env,
        python="/venv/bin/python",
        host="0.0.0.0",
        port="8069",
        dry_run=True,
    )

    module._print_operator_guidance(config)

    captured = capsys.readouterr()
    assert "ChatGPT adapter OAuth launch configuration" in captured.out
    assert "--set-path /content-ops-marketer-chatgpt" in captured.out
    assert f"{module.ADAPTER_PORT_ENV}=8069" in captured.out
    assert f"{module.RUNTIME_PORT_ENV}=8069" in captured.out
    assert "chatgpt-search-fetch" in captured.out
    assert "claude-rich" not in captured.out
    assert "approval-token-with-enough-entropy" not in captured.out
    assert "SET len=34" in captured.out


def test_adapter_main_dry_run_does_not_start_subprocess(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module()
    token_file = tmp_path / "content-ops-token"
    token_file.write_text("approval-token-with-enough-entropy\n")
    monkeypatch.setenv(
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID",
        "acct-content-ops-demo",
    )

    def fail_call(*_args, **_kwargs):
        raise AssertionError("subprocess touched")

    monkeypatch.setattr(module.subprocess, "call", fail_call)

    result = module._main(
        [
            "--approval-token-file",
            str(token_file),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "dry-run: server not started" in captured.out
    assert "content_ops_marketer_verify_chatgpt_adapter_server" in captured.out
    assert "approval-token-with-enough-entropy" not in captured.out
