from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/start_invoicing_draft_writer_oauth_server.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "start_invoicing_draft_writer_oauth_server",
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
        "ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE": "oauth",
        "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL": "https://atlas.example.com/invoicing-draft-writer",
        "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL": "https://atlas.example.com/invoicing-draft-writer/mcp",
        "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN": "approval-token-with-enough-entropy",
        "ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT": "8065",
    }


def test_parse_dotenv_line_handles_quotes_comments_and_empty_lines() -> None:
    module = _load_script_module()

    assert module._parse_dotenv_line("KEY=value") == ("KEY", "value")
    assert module._parse_dotenv_line("KEY='quoted value'") == ("KEY", "quoted value")
    assert module._parse_dotenv_line('KEY="quoted value"') == ("KEY", "quoted value")
    assert module._parse_dotenv_line("# comment") is None
    assert module._parse_dotenv_line("") is None


def test_load_dotenv_files_applies_later_file_override(tmp_path) -> None:
    module = _load_script_module()
    first = tmp_path / ".env"
    second = tmp_path / ".env.local"
    first.write_text("KEY=base\nOTHER=value\n")
    second.write_text("KEY=local\n")

    values = module._load_dotenv_files([first, second])

    assert values == {"KEY": "local", "OTHER": "value"}


def test_validate_env_accepts_oauth_config() -> None:
    module = _load_script_module()

    assert module._validate_env(_valid_env()) == []


def test_validate_env_rejects_short_token_and_bad_port() -> None:
    module = _load_script_module()
    env = _valid_env()
    env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN"] = "short"
    env["ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT"] = "70000"

    errors = module._validate_env(env)

    assert any("APPROVAL_TOKEN" in error for error in errors)
    assert any("between 1 and 65535" in error for error in errors)


def test_build_launch_config_loads_env_files_and_forces_oauth(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE=bearer",
                "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN=approval-token-with-enough-entropy",
            ]
        )
        + "\n"
    )
    monkeypatch.delenv("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN", raising=False)

    config = module._build_launch_config(_args(env_file=[str(env_file)], port="9000"))

    assert config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_AUTH_MODE"] == "oauth"
    assert config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_PORT"] == "9000"
    assert config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL"] == module.DEFAULT_ISSUER_URL
    assert config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN"] == (
        "approval-token-with-enough-entropy"
    )


def test_build_launch_config_process_env_overrides_env_file(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN=approval-token-from-file",
                "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL=https://file.example.com/invoicing-draft-writer",
            ]
        )
        + "\n"
    )
    monkeypatch.setenv(
        "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN",
        "approval-token-from-process-env",
    )

    config = module._build_launch_config(_args(env_file=[str(env_file)]))

    assert config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN"] == (
        "approval-token-from-process-env"
    )
    assert config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL"] == (
        "https://file.example.com/invoicing-draft-writer"
    )


def test_build_launch_config_reads_approval_token_file(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    token_file = tmp_path / "draft-writer-token"
    token_file.write_text("approval-token-from-local-secret-file\n")
    monkeypatch.delenv("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN", raising=False)

    config = module._build_launch_config(
        _args(env_file=[], approval_token_file=str(token_file))
    )

    assert config.env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN"] == (
        "approval-token-from-local-secret-file"
    )


def test_build_launch_config_rejects_empty_approval_token_file(tmp_path) -> None:
    module = _load_script_module()
    token_file = tmp_path / "draft-writer-token"
    token_file.write_text("\n")

    try:
        module._build_launch_config(_args(env_file=[], approval_token_file=str(token_file)))
    except ValueError as exc:
        assert "is empty" in str(exc)
        assert "draft-writer-token" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")


def test_server_command_uses_current_config_python() -> None:
    module = _load_script_module()
    config = module.LaunchConfig(
        env=_valid_env(),
        python="/custom/python",
        host="0.0.0.0",
        port="8065",
        dry_run=False,
    )

    assert module._server_command(config) == [
        "/custom/python",
        "-m",
        "atlas_brain.mcp.invoicing_draft_writer_server",
        "--sse",
    ]


def test_funnel_app_path_derives_parent_path_for_mcp_resource() -> None:
    module = _load_script_module()

    assert module._funnel_app_path("https://atlas.example.com/invoicing-draft-writer/mcp") == (
        "/invoicing-draft-writer"
    )
    assert module._funnel_app_path("http://127.0.0.1:8066/mcp") == "/"
    assert module._funnel_metadata_path("https://atlas.example.com/invoicing-draft-writer/mcp") == (
        "/.well-known/oauth-protected-resource/invoicing-draft-writer"
    )
    assert module._funnel_metadata_path("http://127.0.0.1:8066/mcp") == (
        "/.well-known/oauth-protected-resource"
    )


def test_guidance_masks_secrets_and_uses_configured_port(capsys) -> None:
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
    assert "--set-path /invoicing-draft-writer" in captured.out
    assert "http://127.0.0.1:9000" in captured.out
    assert "--set-path /.well-known/oauth-protected-resource/invoicing-draft-writer" in captured.out
    assert (
        "http://127.0.0.1:9000/.well-known/oauth-protected-resource/invoicing-draft-writer"
        in captured.out
    )
    assert "http://127.0.0.1:8065/.well-known/oauth-protected-resource" not in captured.out
    assert "check_invoicing_draft_writer_oauth_e2e.py" in captured.out


def test_guidance_prints_root_funnel_route_for_root_resource(capsys) -> None:
    module = _load_script_module()
    env = _valid_env()
    env["ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL"] = "http://127.0.0.1:8066/mcp"
    config = module.LaunchConfig(
        env=env,
        python="/venv/bin/python",
        host="127.0.0.1",
        port="8066",
        dry_run=True,
    )

    module._print_operator_guidance(config)

    captured = capsys.readouterr()
    assert "Required Funnel route for the draft-writer connector path:" in captured.out
    assert "  http://127.0.0.1:8066\n" in captured.out
    assert "--set-path /\n" not in captured.out


def test_main_dry_run_does_not_start_subprocess(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN=approval-token-with-enough-entropy",
                "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL=https://atlas.example.com/invoicing-draft-writer",
                "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL=https://atlas.example.com/invoicing-draft-writer/mcp",
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
    env_file.write_text("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN=short\n")

    def fail_call(*_args, **_kwargs):
        raise AssertionError("subprocess touched")

    monkeypatch.setattr(module.subprocess, "call", fail_call)

    result = module._main(["--env-file", str(env_file)])

    captured = capsys.readouterr()
    assert result == 2
    assert "APPROVAL_TOKEN" in captured.err
    assert "short" not in captured.err
