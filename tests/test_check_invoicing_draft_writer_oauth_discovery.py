from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_invoicing_draft_writer_oauth_discovery.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "check_invoicing_draft_writer_oauth_discovery",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authorization_metadata_url_preserves_path_prefixed_issuer() -> None:
    module = _load_script_module()

    assert module._authorization_metadata_url("https://atlas.example.com/invoicing-draft-writer/") == (
        "https://atlas.example.com/invoicing-draft-writer/.well-known/oauth-authorization-server"
    )


def test_protected_resource_metadata_url_uses_resource_path() -> None:
    module = _load_script_module()

    assert module._protected_resource_metadata_url("https://atlas.example.com/invoicing-draft-writer/mcp") == (
        "https://atlas.example.com/.well-known/oauth-protected-resource/invoicing-draft-writer/mcp"
    )


def test_metadata_errors_accept_expected_discovery_documents() -> None:
    module = _load_script_module()
    issuer = "https://atlas.example.com/invoicing-draft-writer"
    resource = "https://atlas.example.com/invoicing-draft-writer/mcp"

    errors = module._metadata_errors(
        issuer_url=issuer,
        resource_url=resource,
        scope="invoices.draft.write",
        auth_metadata={
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "registration_endpoint": f"{issuer}/register",
            "scopes_supported": ["invoices.draft.write"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
        },
        resource_metadata={
            "resource": resource,
            "authorization_servers": [issuer],
            "scopes_supported": ["invoices.draft.write"],
        },
        mcp_status=401,
        mcp_headers={
            "www-authenticate": (
                'Bearer resource_metadata="https://atlas.example.com/.well-known/'
                'oauth-protected-resource/invoicing-draft-writer/mcp"'
            )
        },
    )

    assert errors == []


def test_metadata_errors_reject_missing_resource_challenge() -> None:
    module = _load_script_module()
    issuer = "https://atlas.example.com/invoicing-draft-writer"
    resource = "https://atlas.example.com/invoicing-draft-writer/mcp"

    errors = module._metadata_errors(
        issuer_url=issuer,
        resource_url=resource,
        scope="invoices.draft.write",
        auth_metadata={
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/authorize",
            "token_endpoint": f"{issuer}/token",
            "registration_endpoint": f"{issuer}/register",
            "scopes_supported": ["invoices.draft.write"],
            "grant_types_supported": ["authorization_code"],
        },
        resource_metadata={
            "resource": resource,
            "authorization_servers": [issuer],
            "scopes_supported": ["invoices.draft.write"],
        },
        mcp_status=401,
        mcp_headers={"www-authenticate": "Bearer"},
    )

    assert errors == [
        "WWW-Authenticate header missing resource_metadata="
        "https://atlas.example.com/.well-known/oauth-protected-resource/invoicing-draft-writer/mcp"
    ]


def test_main_requires_urls_before_network(monkeypatch, capsys) -> None:
    module = _load_script_module()

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("network touched")

    monkeypatch.delenv("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL", raising=False)
    monkeypatch.delenv("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL", raising=False)
    monkeypatch.setattr(module, "_fetch_json", fail_fetch)

    result = module._main([])

    captured = capsys.readouterr()
    assert result == 2
    assert "--issuer-url" in captured.err


def test_main_reports_success_for_valid_public_discovery(monkeypatch, capsys) -> None:
    module = _load_script_module()
    issuer = "https://atlas.example.com/invoicing-draft-writer"
    resource = "https://atlas.example.com/invoicing-draft-writer/mcp"

    def fake_fetch(url: str, _timeout: float):
        if url.endswith("/.well-known/oauth-authorization-server"):
            return {
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
                "registration_endpoint": f"{issuer}/register",
                "scopes_supported": ["invoices.draft.write"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
            }
        return {
            "resource": resource,
            "authorization_servers": [issuer],
            "scopes_supported": ["invoices.draft.write"],
        }

    monkeypatch.setattr(module, "_fetch_json", fake_fetch)
    monkeypatch.setattr(
        module,
        "_probe_mcp_unauthenticated",
        lambda *_args: (
            401,
            {
                "www-authenticate": (
                    'Bearer resource_metadata="https://atlas.example.com/.well-known/'
                    'oauth-protected-resource/invoicing-draft-writer/mcp"'
                )
            },
        ),
    )

    result = module._main(["--issuer-url", issuer, "--resource-url", resource])

    captured = capsys.readouterr()
    assert result == 0
    assert "OAuth discovery is routable" in captured.out
