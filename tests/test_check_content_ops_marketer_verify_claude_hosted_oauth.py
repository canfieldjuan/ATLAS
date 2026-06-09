from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_content_ops_marketer_verify_claude_hosted_oauth.py"
ISSUER = "https://atlas.example.com/content-ops-marketer"
RESOURCE = "https://atlas.example.com/content-ops-marketer/mcp"
APPROVAL = "https://atlas.example.com/content-ops-marketer/oauth/approve?request_id=req-1"
BASE_ARGS = ["--issuer-url", ISSUER, "--resource-url", RESOURCE]


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_content_ops_marketer_verify_claude_hosted_oauth",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_redirect_errors_accepts_content_ops_approval_redirect() -> None:
    module = _load_script_module()

    assert module._redirect_errors(
        issuer_url=ISSUER,
        status=302,
        headers={"Location": APPROVAL},
    ) == []


def test_redirect_errors_cover_failure_branches() -> None:
    module = _load_script_module()

    assert module._redirect_errors(
        issuer_url=ISSUER,
        status=502,
        headers={},
    ) == ["root authorize returned HTTP 502, expected 302 or 303"]
    assert module._redirect_errors(issuer_url=ISSUER, status=303, headers={}) == ["root authorize redirect missing Location header"]
    errors = module._redirect_errors(
        issuer_url=ISSUER,
        status=302,
        headers={"Location": "https://evil.example.com/oauth/approve"},
    )
    assert "root authorize redirect host does not match issuer host" in errors
    assert "root authorize redirect path != /content-ops-marketer/oauth/approve" in errors
    assert "root authorize redirect missing request_id" in errors


def test_main_reports_success_from_mocked_redirect(monkeypatch, capsys) -> None:
    module = _load_script_module()

    def fake_open(url: str, timeout: float):
        assert url.startswith("https://atlas.example.com/authorize?")
        assert "resource=https%3A%2F%2Fatlas.example.com%2Fcontent-ops-marketer%2Fmcp" in url
        assert timeout == 3.0
        return (
            302,
            {"Location": APPROVAL},
        )

    monkeypatch.setattr(module, "_open_no_redirect", fake_open)

    result = module._main(
        [
            *BASE_ARGS,
            "--client-id",
            "client-1",
            "--timeout",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "OK: Claude hosted OAuth root authorize route redirects" in captured.out


def test_main_fails_on_missing_client_id_and_bad_redirect(monkeypatch, capsys) -> None:
    module = _load_script_module()

    result = module._main(
        [
            *BASE_ARGS,
        ]
    )

    assert result == 2
    assert "--client-id" in capsys.readouterr().err

    monkeypatch.setattr(module, "_open_no_redirect", lambda *_args, **_kwargs: (200, {}))
    result = module._main(
        [
            *BASE_ARGS,
            "--client-id",
            "client-1",
        ]
    )

    assert result == 1
    assert "root authorize returned HTTP 200" in capsys.readouterr().err
