from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/start_content_ops_marketer_verify_oauth_server.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "start_content_ops_marketer_verify_oauth_server",
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


def _valid_env():
    return {
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE": "oauth",
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL": (
            "https://atlas.example.com/content-ops-marketer"
        ),
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL": (
            "https://atlas.example.com/content-ops-marketer/mcp"
        ),
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN": (
            "approval-token-with-enough-entropy"
        ),
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT": "8068",
    }


def test_validate_env_accepts_content_ops_oauth_config() -> None:
    module = _load_script_module()

    assert module._validate_env(_valid_env()) == []


def test_validate_env_rejects_short_token_and_bad_port() -> None:
    module = _load_script_module()
    env = _valid_env()
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN"] = "short"
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT"] = "70000"

    errors = module._validate_env(env)

    assert any("APPROVAL_TOKEN" in error for error in errors)
    assert any("between 1 and 65535" in error for error in errors)


def test_validate_env_rejects_non_oauth_mode_and_non_public_urls() -> None:
    module = _load_script_module()
    env = _valid_env()
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE"] = "bearer"
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"] = "http://example.com"
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"] = "ftp://example.com/mcp"

    errors = module._validate_env(env)

    assert any("AUTH_MODE must be oauth" in error for error in errors)
    assert any("ISSUER_URL must be HTTPS" in error for error in errors)
    assert any("RESOURCE_URL must be HTTPS" in error for error in errors)


def test_validate_env_rejects_localhost_prefix_attack_urls() -> None:
    module = _load_script_module()
    env = _valid_env()
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"] = (
        "http://localhost.evil.example.com/content-ops-marketer"
    )
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"] = (
        "http://127.0.0.1.evil.example.com/content-ops-marketer/mcp"
    )

    errors = module._validate_env(env)

    assert any("ISSUER_URL must be HTTPS" in error for error in errors)
    assert any("RESOURCE_URL must be HTTPS" in error for error in errors)


def test_validate_env_accepts_exact_localhost_http_urls() -> None:
    module = _load_script_module()
    env = _valid_env()
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"] = (
        "http://localhost:8068/content-ops-marketer"
    )
    env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL"] = (
        "http://127.0.0.1:8068/content-ops-marketer/mcp"
    )

    assert module._validate_env(env) == []


def test_build_launch_config_loads_env_files_and_forces_oauth(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE=bearer",
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN=approval-token-with-enough-entropy",
            ]
        )
        + "\n"
    )
    monkeypatch.delenv(
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN",
        raising=False,
    )

    config = module._build_launch_config(_args(env_file=[str(env_file)], port="9000"))

    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE"] == "oauth"
    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_PORT"] == "9000"
    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"] == (
        module.DEFAULT_ISSUER_URL
    )


def test_build_launch_config_process_env_overrides_env_file(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN=approval-token-from-file",
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL=https://file.example.com/content-ops-marketer",
            ]
        )
        + "\n"
    )
    monkeypatch.setenv(
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN",
        "approval-token-from-process-env",
    )

    config = module._build_launch_config(_args(env_file=[str(env_file)]))

    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN"] == (
        "approval-token-from-process-env"
    )
    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL"] == (
        "https://file.example.com/content-ops-marketer"
    )


def test_build_launch_config_reads_approval_token_file(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    token_file = tmp_path / "content-ops-token"
    token_file.write_text("approval-token-from-local-secret-file\n")
    monkeypatch.delenv(
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN",
        raising=False,
    )

    config = module._build_launch_config(
        _args(env_file=[], approval_token_file=str(token_file))
    )

    assert config.env["ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN"] == (
        "approval-token-from-local-secret-file"
    )


def test_build_launch_config_rejects_empty_approval_token_file(tmp_path) -> None:
    module = _load_script_module()
    token_file = tmp_path / "content-ops-token"
    token_file.write_text("\n")

    try:
        module._build_launch_config(_args(env_file=[], approval_token_file=str(token_file)))
    except ValueError as exc:
        assert "is empty" in str(exc)
        assert "content-ops-token" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")


def test_server_command_uses_content_ops_marketer_verify_module() -> None:
    module = _load_script_module()
    config = module.LaunchConfig(
        env=_valid_env(),
        python="/custom/python",
        host="0.0.0.0",
        port="8068",
        dry_run=False,
    )

    assert module._server_command(config) == [
        "/custom/python",
        "-m",
        "atlas_brain.mcp.content_ops_marketer_verify_server",
        "--sse",
    ]


def test_funnel_paths_derive_parent_path_for_mcp_resource() -> None:
    module = _load_script_module()

    assert module._funnel_app_path("https://atlas.example.com/content-ops-marketer/mcp") == (
        "/content-ops-marketer"
    )
    assert module._funnel_app_path("http://127.0.0.1:8068/mcp") == "/"
    assert module._funnel_metadata_path("https://atlas.example.com/content-ops-marketer/mcp") == (
        "/.well-known/oauth-protected-resource/content-ops-marketer"
    )
    assert module._funnel_metadata_path("http://127.0.0.1:8068/mcp") == (
        "/.well-known/oauth-protected-resource"
    )


def test_guidance_masks_secrets_and_prints_content_ops_smokes(capsys) -> None:
    module = _load_script_module()
    env = _valid_env()
    env["ATLAS_MCP_AUTH_TOKEN"] = "bearer-token-value-that-must-not-print"
    config = module.LaunchConfig(
        env=env,
        python="/venv/bin/python",
        host="0.0.0.0",
        port="9000",
        dry_run=True,
    )

    module._print_operator_guidance(config)

    captured = capsys.readouterr()
    assert "SET len=34" in captured.out
    assert "approval-token-with-enough-entropy" not in captured.out
    assert "bearer-token-value-that-must-not-print" not in captured.out
    assert "--set-path /content-ops-marketer" in captured.out
    assert "http://127.0.0.1:9000" in captured.out
    assert "--set-path /.well-known/oauth-protected-resource/content-ops-marketer" in captured.out
    assert "check_content_ops_marketer_verify_oauth_discovery.py" in captured.out
    assert "check_content_ops_marketer_verify_oauth_e2e.py" in captured.out
    assert "--approval-token-file /path/to/local-approval-token" in captured.out
    assert "pass --approval-token" in captured.out


def test_guidance_prints_e2e_approval_token_file_path(capsys) -> None:
    module = _load_script_module()
    config = module.LaunchConfig(
        env=_valid_env(),
        python="/venv/bin/python",
        host="0.0.0.0",
        port="9000",
        dry_run=True,
        approval_token_file="/tmp/content ops token",
    )

    module._print_operator_guidance(config)

    captured = capsys.readouterr()
    assert "--approval-token-file '/tmp/content ops token'" in captured.out
    assert "approval-token-with-enough-entropy" not in captured.out
    assert "pass --approval-token" not in captured.out


def test_main_dry_run_does_not_start_subprocess(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN=approval-token-with-enough-entropy",
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL=https://atlas.example.com/content-ops-marketer",
                "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL=https://atlas.example.com/content-ops-marketer/mcp",
            ]
        )
        + "\n"
    )

    def fail_call(*_args, **_kwargs):
        raise AssertionError("subprocess touched")

    monkeypatch.setattr(module.subprocess, "call", fail_call)

    result = module._main(["--env-file", str(env_file), "--dry-run"])

    captured = capsys.readouterr()
    assert result == 0
    assert "dry-run: server not started" in captured.out
    assert "approval-token-with-enough-entropy" not in captured.out


def test_main_returns_validation_errors_before_subprocess(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN=short\n"
    )

    def fail_call(*_args, **_kwargs):
        raise AssertionError("subprocess touched")

    monkeypatch.setattr(module.subprocess, "call", fail_call)

    result = module._main(["--env-file", str(env_file)])

    captured = capsys.readouterr()
    assert result == 2
    assert "APPROVAL_TOKEN" in captured.err
    assert "short" not in captured.err
