from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_invoicing_draft_writer_funnel_routes.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_invoicing_draft_writer_funnel_routes",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _status(*, app_proxy: str = "http://127.0.0.1:8066", metadata_proxy: str = ""):
    if not metadata_proxy:
        metadata_proxy = (
            "http://127.0.0.1:8066/.well-known/oauth-protected-resource/"
            "invoicing-draft-writer"
        )
    return {
        "Web": {
            "atlas-brain.tailc7bd29.ts.net:443": {
                "Handlers": {
                    "/invoicing-draft-writer": {"Proxy": app_proxy},
                    "/.well-known/oauth-protected-resource/invoicing-draft-writer": {
                        "Proxy": metadata_proxy
                    },
                }
            }
        }
    }


def test_expectation_derives_public_host_and_routes() -> None:
    module = _load_script_module()

    expected = module._expectation(
        "https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp",
        "8066",
    )

    assert expected.host_key == "atlas-brain.tailc7bd29.ts.net:443"
    assert expected.app_path == "/invoicing-draft-writer"
    assert expected.app_proxy == "http://127.0.0.1:8066"
    assert expected.metadata_path == "/.well-known/oauth-protected-resource/invoicing-draft-writer"
    assert expected.metadata_proxy == (
        "http://127.0.0.1:8066/.well-known/oauth-protected-resource/invoicing-draft-writer"
    )


def test_expectation_preserves_explicit_public_port_and_root_path() -> None:
    module = _load_script_module()

    expected = module._expectation("https://atlas.example.com:8443/mcp", "9000")

    assert expected.host_key == "atlas.example.com:8443"
    assert expected.app_path == "/"
    assert expected.app_proxy == "http://127.0.0.1:9000"


def test_expectation_rejects_non_https_public_resource() -> None:
    module = _load_script_module()

    try:
        module._expectation("http://atlas.example.com/invoicing-draft-writer/mcp", "8066")
    except ValueError as exc:
        assert "must use https" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")


def test_route_errors_accept_expected_status() -> None:
    module = _load_script_module()
    expected = module._expectation(
        "https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp",
        "8066",
    )

    assert module._route_errors(_status(), expected) == []


def test_route_errors_report_missing_and_wrong_routes() -> None:
    module = _load_script_module()
    expected = module._expectation(
        "https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp",
        "8066",
    )
    status = {
        "Web": {
            "atlas-brain.tailc7bd29.ts.net:443": {
                "Handlers": {
                    "/.well-known/oauth-protected-resource/invoicing-draft-writer": {
                        "Proxy": (
                            "http://127.0.0.1:8065/.well-known/oauth-protected-resource/"
                            "invoicing-draft-writer"
                        )
                    }
                }
            }
        }
    }

    errors = module._route_errors(status, expected)

    assert errors == [
        (
            "atlas-brain.tailc7bd29.ts.net:443 handler /invoicing-draft-writer "
            "proxy is missing, expected http://127.0.0.1:8066"
        ),
        (
            "atlas-brain.tailc7bd29.ts.net:443 handler "
            "/.well-known/oauth-protected-resource/invoicing-draft-writer "
            "proxy is http://127.0.0.1:8065/.well-known/oauth-protected-resource/"
            "invoicing-draft-writer, expected http://127.0.0.1:8066/"
            ".well-known/oauth-protected-resource/invoicing-draft-writer"
        ),
    ]


def test_repair_commands_include_both_required_routes() -> None:
    module = _load_script_module()
    expected = module._expectation(
        "https://atlas-brain.tailc7bd29.ts.net/invoicing-draft-writer/mcp",
        "8066",
    )

    commands = module._repair_commands(expected)

    assert commands == [
        "tailscale funnel --bg --yes \\\n"
        "  --set-path /invoicing-draft-writer \\\n"
        "  http://127.0.0.1:8066",
        "tailscale funnel --bg --yes \\\n"
        "  --set-path /.well-known/oauth-protected-resource/invoicing-draft-writer \\\n"
        "  http://127.0.0.1:8066/.well-known/oauth-protected-resource/invoicing-draft-writer",
    ]


def test_load_funnel_status_rejects_invalid_json(monkeypatch) -> None:
    module = _load_script_module()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["tailscale", "funnel", "status", "--json"],
            returncode=0,
            stdout="not json",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    try:
        module._load_funnel_status("tailscale")
    except RuntimeError as exc:
        assert "did not return JSON" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_main_reports_success_for_expected_routes(monkeypatch, capsys) -> None:
    module = _load_script_module()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["tailscale", "funnel", "status", "--json"],
            returncode=0,
            stdout=json.dumps(_status()),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module._main([])

    captured = capsys.readouterr()
    assert result == 0
    assert "Funnel routes are configured" in captured.out
    assert "/invoicing-draft-writer -> http://127.0.0.1:8066" in captured.out


def test_main_prints_repair_commands_for_route_drift(monkeypatch, capsys) -> None:
    module = _load_script_module()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["tailscale", "funnel", "status", "--json"],
            returncode=0,
            stdout=json.dumps({"Web": {}}),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module._main([])

    captured = capsys.readouterr()
    assert result == 1
    assert "handler /invoicing-draft-writer proxy is missing" in captured.err
    assert "Run the required route commands:" in captured.err
    assert "--set-path /invoicing-draft-writer" in captured.err
