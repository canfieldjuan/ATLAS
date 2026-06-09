from __future__ import annotations

import sys
import types
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class _MockToolManager:
    def __init__(self) -> None:
        self._tools = {}


class _MockFastMCP:
    def __init__(self, *args, **kwargs) -> None:
        self.settings = MagicMock()
        self._tool_manager = _MockToolManager()
        self.lifespan = kwargs.get("lifespan")

    def tool(self, *args, **kwargs):
        def _register(fn):
            self._tool_manager._tools[fn.__name__] = fn
            return fn

        return _register

    def custom_route(self, *args, **kwargs):
        def _register(fn):
            return fn

        return _register

    def streamable_http_app(self):
        return object()

    def run(self, **kwargs) -> None:
        return None


class _OAuthBase:
    def __class_getitem__(cls, item):
        return cls


class _OAuthRecord:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def model_dump(self, mode="json"):
        return dict(self.__dict__)


class _OAuthException(Exception):
    pass


class _ProviderTokenVerifier:
    def __init__(self, provider) -> None:
        self.provider = provider


class _ClientRegistrationOptions:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _AuthSettings:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _TransportSecuritySettings:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


sys.modules.setdefault("mcp", MagicMock())
sys.modules.setdefault("mcp.server", MagicMock())
_fastmcp_mod = MagicMock()
_fastmcp_mod.FastMCP = _MockFastMCP
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)
_auth_provider_mod = MagicMock()
_auth_provider_mod.AccessToken = _OAuthRecord
_auth_provider_mod.AuthorizationCode = _OAuthRecord
_auth_provider_mod.AuthorizationParams = _OAuthRecord
_auth_provider_mod.AuthorizeError = _OAuthException
_auth_provider_mod.OAuthAuthorizationServerProvider = _OAuthBase
_auth_provider_mod.ProviderTokenVerifier = _ProviderTokenVerifier
_auth_provider_mod.RefreshToken = _OAuthRecord
_auth_provider_mod.TokenError = _OAuthException
_auth_provider_mod.construct_redirect_uri = lambda redirect_uri, **params: redirect_uri
sys.modules.setdefault("mcp.server.auth.provider", _auth_provider_mod)
_auth_settings_mod = MagicMock()
_auth_settings_mod.AuthSettings = _AuthSettings
_auth_settings_mod.ClientRegistrationOptions = _ClientRegistrationOptions
sys.modules.setdefault("mcp.server.auth.settings", _auth_settings_mod)
_transport_security_mod = MagicMock()
_transport_security_mod.TransportSecuritySettings = _TransportSecuritySettings
sys.modules.setdefault("mcp.server.transport_security", _transport_security_mod)
_shared_auth_mod = MagicMock()
_shared_auth_mod.OAuthClientInformationFull = _OAuthRecord
_shared_auth_mod.OAuthToken = _OAuthRecord
sys.modules.setdefault("mcp.shared.auth", _shared_auth_mod)

from atlas_brain.config import settings
from atlas_brain.mcp import content_ops_marketer_verify_chatgpt_adapter_server as adapter
from atlas_brain.mcp import content_ops_marketer_verify_server as verify
from atlas_brain.mcp.auth import BearerAuthMiddleware
from atlas_brain.mcp.content_ops_marketer_verify_oauth import (
    ContentOpsMarketerVerifyOAuthProvider,
    DEFAULT_CONTENT_OPS_VERIFY_SCOPE,
    validate_oauth_settings,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.claims_map import RegistryClaim
from extracted_content_pipeline.review_contract import RiskTier


VERIFY_TOOLS = {"verify_draft"}
CHATGPT_ADAPTER_TOOLS = {"search", "fetch"}
DENIED_TOOLS = {
    "add_registry_claim",
    "approve",
    "fetch",
    "generate",
    "publish",
    "search",
    "unlock",
}


@dataclass
class _RegistryReader:
    scopes: list[TenantScope]

    async def list_registry_claims(self, *, scope: TenantScope):
        self.scopes.append(scope)
        return {
            "feature.sso": RegistryClaim(
                id="feature.sso",
                approved_wording="SSO is included on every plan",
                risk_tier=RiskTier.MEDIUM,
                expiration=date(2026, 12, 31),
            )
        }


@pytest.fixture(autouse=True)
def _reset_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "_registry_reader_override", None)
    monkeypatch.setattr(verify, "_account_resolver_override", None)
    monkeypatch.setattr(verify, "_oauth_provider", None)
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_account_id", "")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "bearer")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_oauth_issuer_url", "")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_oauth_resource_url", "")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_oauth_approval_token", "")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_oauth_state_file", "")
    verify.mcp.settings.auth = None
    verify.mcp.settings.transport_security = None
    verify.mcp._auth_server_provider = None
    verify.mcp._token_verifier = None
    adapter.mcp.settings.auth = None
    adapter.mcp.settings.transport_security = None
    adapter.mcp._auth_server_provider = None
    adapter.mcp._token_verifier = None
    adapter._verdict_cache.clear()


def _tool_names() -> set[str]:
    return set(verify.mcp._tool_manager._tools)


def _valid_payload() -> dict[str, object]:
    return {
        "asset_id": "asset-1",
        "rule_packet": {
            "brief": "brief-v1",
            "brand_voice": "voice-v1",
            "claim_registry": "claims-v1",
            "compliance": "compliance-v1",
            "channel_schema": "channel-v1",
        },
        "coverage": [
            {
                "rule_id": "VOICE-01",
                "requirement": "Rule must be evidenced",
                "status": "pass",
                "evidence": "quoted draft span",
            }
        ],
        "extracted_claims": [
            {
                "text": "SSO is included on every plan",
                "location": "hero",
                "registry_id": "feature.sso",
            }
        ],
        "as_of": "2026-06-08",
    }


def test_content_ops_marketer_verify_exposes_exact_tool_surface() -> None:
    assert _tool_names() == VERIFY_TOOLS
    assert _tool_names().isdisjoint(DENIED_TOOLS)


def test_chatgpt_adapter_exposes_exact_search_fetch_surface() -> None:
    tool_names = set(adapter.mcp._tool_manager._tools)

    assert tool_names == CHATGPT_ADAPTER_TOOLS
    assert tool_names.isdisjoint(VERIFY_TOOLS)
    assert tool_names.isdisjoint(DENIED_TOOLS - {"fetch", "search"})


def test_chatgpt_adapter_reuses_verifier_database_lifespan() -> None:
    assert adapter.mcp.lifespan is verify._lifespan


@pytest.mark.asyncio
async def test_chatgpt_adapter_non_json_search_returns_contract_document() -> None:
    result = await adapter.search(query="how do I verify a draft?", limit=10)
    fetched = await adapter.fetch(adapter.CONTRACT_ID)

    assert result["metadata"] == {
        "ok": True,
        "mode": "contract",
        "query": "how do I verify a draft?",
        "count": 1,
    }
    assert result["results"] == [
        {
            "id": adapter.CONTRACT_ID,
            "title": "Content Ops verify draft JSON contract",
            "url": f"atlas://content-ops/marketer-verify/{adapter.CONTRACT_ID}",
        }
    ]
    assert fetched["metadata"]["ok"] is True
    assert fetched["metadata"]["type"] == "adapter_contract"
    assert "asset_id" in fetched["metadata"]["accepted_fields"]


@pytest.mark.asyncio
async def test_chatgpt_adapter_search_delegates_to_bound_tenant_and_fetches_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver(" acct-1 "),
    )

    search_result = await adapter.search(query=json.dumps(_valid_payload()), limit=5)
    verdict_id = search_result["results"][0]["id"]
    fetched = await adapter.fetch(verdict_id)

    assert search_result["metadata"]["ok"] is True
    assert search_result["metadata"]["mode"] == "verification"
    assert search_result["metadata"]["decision"] == "approved"
    assert verdict_id.startswith("content-ops-verdict:")
    assert fetched["metadata"]["ok"] is True
    assert fetched["metadata"]["found"] is True
    assert fetched["metadata"]["verdict"]["decision"] == "approved"
    assert "Decision: approved" in fetched["text"]
    assert reader.scopes == [TenantScope(account_id="acct-1")]


@pytest.mark.asyncio
async def test_chatgpt_adapter_blocks_json_search_without_account_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver(" "),
    )

    result = await adapter.search(query=json.dumps(_valid_payload()))

    assert result["results"] == []
    assert result["metadata"]["ok"] is False
    assert result["metadata"]["error"] == "account_binding_required"
    assert reader.scopes == []


@pytest.mark.asyncio
async def test_chatgpt_adapter_fetch_fails_closed_for_other_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver("acct-1"),
    )
    search_result = await adapter.search(query=json.dumps(_valid_payload()))
    verdict_id = search_result["results"][0]["id"]
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver("acct-2"),
    )

    fetched = await adapter.fetch(verdict_id)

    assert fetched["metadata"]["ok"] is False
    assert fetched["metadata"]["error"] == "verdict_not_found"
    assert "verdict" not in fetched["metadata"]


@pytest.mark.asyncio
async def test_chatgpt_adapter_fetch_requires_bound_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver(" "),
    )

    fetched = await adapter.fetch("content-ops-verdict:missing")

    assert fetched["metadata"]["ok"] is False
    assert fetched["metadata"]["error"] == "account_binding_required"


@pytest.mark.asyncio
async def test_verify_draft_fails_closed_without_account_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver(" "),
    )

    payload = await verify.verify_draft(**_valid_payload())

    assert payload["ok"] is False
    assert payload["decision"] == "blocked"
    assert payload["reasons"] == ["tenant scope required"]
    assert reader.scopes == []


@pytest.mark.asyncio
async def test_verify_draft_delegates_to_bound_tenant_review_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver(" acct-1 "),
    )

    payload = await verify.verify_draft(**_valid_payload())

    assert payload["ok"] is True
    assert payload["decision"] == "approved"
    assert payload["reasons"] == []
    assert payload["mapped_claims"][0]["status"] == "match"
    assert reader.scopes == [TenantScope(account_id="acct-1")]


@pytest.mark.asyncio
async def test_verify_draft_malformed_decoded_rows_block_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver("acct-1"),
    )

    payload = await verify.verify_draft(
        asset_id=123,
        rule_packet={"brief": None},
        coverage=[{"rule_id": "VOICE-01", "status": "wat"}],
        extracted_claims=[{"text": None, "registry_id": 42}],
        comments=[{"category": "nit", "blocking": True}],
        as_of="not-a-date",
    )

    assert payload["ok"] is False
    assert payload["decision"] == "blocked"
    assert "rule packet not pinned" in payload["reasons"][0]
    assert "unresolved required coverage: VOICE-01" in payload["reasons"]
    assert "1 blocking comment(s)" in payload["reasons"]
    assert reader.scopes == [TenantScope(account_id="acct-1")]


@pytest.mark.asyncio
async def test_verify_draft_preserves_malformed_coverage_rows_as_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(
        verify,
        "_account_resolver_override",
        verify.StaticContentOpsMarketerAccountResolver("acct-1"),
    )
    payload = _valid_payload()
    payload["coverage"] = [
        {
            "rule_id": "VOICE-01",
            "requirement": "Rule must be evidenced",
            "status": "pass",
            "evidence": "quoted draft span",
        },
        {"status": "pass", "evidence": "missing rule id should not disappear"},
        {"rule_id": 42, "status": "pass", "evidence": "non-string rule id"},
        "not-a-row",
    ]

    result = await verify.verify_draft(**payload)

    coverage = result["content_pr"]["coverage"]
    assert result["ok"] is False
    assert result["decision"] == "blocked"
    assert (
        "unresolved required coverage: "
        "MALFORMED-COVERAGE-2, MALFORMED-COVERAGE-3, MALFORMED-COVERAGE-4"
    ) in result["reasons"]
    assert [row["rule_id"] for row in coverage] == [
        "VOICE-01",
        "MALFORMED-COVERAGE-2",
        "MALFORMED-COVERAGE-3",
        "MALFORMED-COVERAGE-4",
    ]
    assert reader.scopes == [TenantScope(account_id="acct-1")]


def test_content_ops_marketer_verify_http_requires_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", "")

    with pytest.raises(RuntimeError, match="ATLAS_MCP_AUTH_TOKEN is required"):
        verify._streamable_http_app()


def test_content_ops_marketer_verify_http_wraps_with_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", "test-token-with-enough-entropy")

    assert isinstance(verify._streamable_http_app(), BearerAuthMiddleware)


@pytest.mark.parametrize("token", ["<token>", "token", "test-token"])
def test_content_ops_marketer_verify_http_rejects_bad_tokens(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", token)

    with pytest.raises(RuntimeError):
        verify._streamable_http_app()


def test_content_ops_marketer_verify_rejects_unknown_http_auth_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "wat")

    with pytest.raises(RuntimeError, match="AUTH_MODE"):
        verify._streamable_http_app()


def test_content_ops_marketer_verify_oauth_settings_require_approval_token() -> None:
    with pytest.raises(RuntimeError, match="OAUTH_APPROVAL_TOKEN"):
        validate_oauth_settings(
            issuer_url="https://atlas.example.com/content-ops-marketer",
            resource_server_url="https://atlas.example.com/content-ops-marketer/mcp",
            approval_token="short",
        )


@pytest.mark.asyncio
async def test_content_ops_marketer_oauth_provider_binds_access_and_refresh_tokens() -> None:
    provider = ContentOpsMarketerVerifyOAuthProvider(
        issuer_url="https://atlas.example.com/content-ops-marketer",
        approval_token="approval-token-with-enough-entropy",
        account_id=" acct-oauth ",
    )
    client = _OAuthRecord(client_id="client-1")
    params = _OAuthRecord(
        scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
        code_challenge="challenge-1",
        redirect_uri="https://chat.openai.com/aip/callback",
        redirect_uri_provided_explicitly=True,
        resource="https://atlas.example.com/content-ops-marketer/mcp",
        state="state-1",
    )

    await provider.authorize(client, params)
    request_id = next(iter(provider._pending))
    provider.approve_pending_authorization(
        request_id=request_id,
        approval_token="approval-token-with-enough-entropy",
    )
    code = next(iter(provider._authorization_codes))
    authorization_code = await provider.load_authorization_code(client, code)
    token = await provider.exchange_authorization_code(client, authorization_code)

    refresh = await provider.load_refresh_token(client, token.refresh_token)
    refreshed = await provider.exchange_refresh_token(client, refresh, [])

    assert provider.account_id == "acct-oauth"
    assert provider.account_id_for_access_token(token.access_token) == "acct-oauth"
    assert provider.account_id_for_access_token(refreshed.access_token) == "acct-oauth"


@pytest.mark.asyncio
async def test_content_ops_marketer_oauth_provider_rejects_unbound_tokens() -> None:
    provider = ContentOpsMarketerVerifyOAuthProvider(
        issuer_url="https://atlas.example.com/content-ops-marketer",
        approval_token="approval-token-with-enough-entropy",
        account_id="",
    )
    client = _OAuthRecord(client_id="client-1")
    params = _OAuthRecord(
        scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
        code_challenge="challenge-1",
        redirect_uri="https://chat.openai.com/aip/callback",
        redirect_uri_provided_explicitly=True,
        resource="https://atlas.example.com/content-ops-marketer/mcp",
        state="state-1",
    )

    await provider.authorize(client, params)
    request_id = next(iter(provider._pending))
    provider.approve_pending_authorization(
        request_id=request_id,
        approval_token="approval-token-with-enough-entropy",
    )
    code = next(iter(provider._authorization_codes))
    authorization_code = await provider.load_authorization_code(client, code)

    with pytest.raises(_OAuthException, match="tenant-bound"):
        await provider.exchange_authorization_code(client, authorization_code)


@pytest.mark.asyncio
async def test_verify_draft_oauth_resolves_scope_from_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    provider = ContentOpsMarketerVerifyOAuthProvider(
        issuer_url="https://atlas.example.com/content-ops-marketer",
        approval_token="approval-token-with-enough-entropy",
        account_id="acct-token",
    )
    provider._record_token_binding(
        access_token="access-token-1",
        refresh_token="refresh-token-1",
        account_id="acct-token",
    )
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(verify, "_oauth_provider", provider)
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "oauth")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_account_id", "acct-config")
    monkeypatch.setitem(
        sys.modules,
        "mcp.server.auth.middleware.auth_context",
        types.SimpleNamespace(get_access_token=lambda: _OAuthRecord(token="access-token-1")),
    )

    payload = await verify.verify_draft(**_valid_payload())

    assert payload["ok"] is True
    assert reader.scopes == [TenantScope(account_id="acct-token")]


@pytest.mark.asyncio
async def test_verify_draft_oauth_blocks_without_bound_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    provider = ContentOpsMarketerVerifyOAuthProvider(
        issuer_url="https://atlas.example.com/content-ops-marketer",
        approval_token="approval-token-with-enough-entropy",
        account_id="acct-token",
    )
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(verify, "_oauth_provider", provider)
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "oauth")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_account_id", "acct-config")
    monkeypatch.setitem(
        sys.modules,
        "mcp.server.auth.middleware.auth_context",
        types.SimpleNamespace(get_access_token=lambda: _OAuthRecord(token="unknown-token")),
    )

    payload = await verify.verify_draft(**_valid_payload())

    assert payload["ok"] is False
    assert payload["decision"] == "blocked"
    assert payload["reasons"] == ["tenant scope required"]
    assert reader.scopes == []


@pytest.mark.asyncio
async def test_bearer_mode_keeps_configured_account_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _RegistryReader(scopes=[])
    monkeypatch.setattr(verify, "_registry_reader_override", reader)
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "bearer")
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_account_id", "acct-config")

    payload = await verify.verify_draft(**_valid_payload())

    assert payload["ok"] is True
    assert reader.scopes == [TenantScope(account_id="acct-config")]


def test_content_ops_marketer_verify_oauth_mode_configures_fastmcp_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "content-ops-marketer-oauth-state.json"
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
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_account_id", "acct-oauth")
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_state_file",
        str(state_file),
    )

    app = verify._streamable_http_app()
    provider = verify._oauth_provider

    assert not isinstance(app, BearerAuthMiddleware)
    assert provider is not None
    assert provider.account_id == "acct-oauth"
    assert provider.scopes == [DEFAULT_CONTENT_OPS_VERIFY_SCOPE]
    assert provider.state_file == state_file
    assert verify.mcp.settings.auth.required_scopes == [DEFAULT_CONTENT_OPS_VERIFY_SCOPE]
    assert verify.mcp.settings.auth.client_registration_options.enabled is True
    assert verify.mcp.settings.auth.client_registration_options.valid_scopes == [
        DEFAULT_CONTENT_OPS_VERIFY_SCOPE
    ]
    assert verify.mcp._auth_server_provider is provider
    assert verify.mcp._token_verifier.provider is provider


def test_content_ops_oauth_metadata_advertises_public_and_confidential_clients() -> None:
    metadata = verify._content_ops_oauth_metadata(
        issuer_url="https://atlas.example.com/content-ops-marketer/",
        scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
    )

    assert metadata["issuer"] == "https://atlas.example.com/content-ops-marketer"
    assert metadata["authorization_endpoint"] == "https://atlas.example.com/content-ops-marketer/authorize"
    assert metadata["token_endpoint"] == "https://atlas.example.com/content-ops-marketer/token"
    assert metadata["registration_endpoint"] == "https://atlas.example.com/content-ops-marketer/register"
    assert metadata["token_endpoint_auth_methods_supported"] == [
        "none",
        "client_secret_post",
        "client_secret_basic",
    ]
    assert metadata["code_challenge_methods_supported"] == ["S256"]


@pytest.mark.asyncio
async def test_content_ops_public_client_metadata_replaces_only_auth_metadata_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_metadata = types.SimpleNamespace(path="/.well-known/oauth-authorization-server")
    token_route = types.SimpleNamespace(path="/token")
    app = types.SimpleNamespace(router=types.SimpleNamespace(routes=[old_metadata, token_route]))
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_issuer_url",
        "https://atlas.example.com/content-ops-marketer",
    )

    patched = verify._apply_content_ops_public_client_metadata(app)
    metadata_route, remaining_route = patched.router.routes
    response = await metadata_route.endpoint(types.SimpleNamespace(method="GET"))
    options = await metadata_route.endpoint(types.SimpleNamespace(method="OPTIONS"))

    assert patched is app
    assert metadata_route.path == "/.well-known/oauth-authorization-server"
    assert remaining_route is token_route
    assert json.loads(response.body)["token_endpoint_auth_methods_supported"][0] == "none"
    assert response.headers["access-control-allow-origin"] == "*"
    assert options.status_code == 204


def test_chatgpt_adapter_oauth_mode_configures_adapter_fastmcp_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "content-ops-marketer-chatgpt-oauth-state.json"
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "oauth")
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_issuer_url",
        "https://atlas.example.com/content-ops-marketer-chatgpt",
    )
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_resource_url",
        "https://atlas.example.com/content-ops-marketer-chatgpt/mcp",
    )
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_approval_token",
        "approval-token-with-enough-entropy",
    )
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_account_id", "acct-oauth")
    monkeypatch.setattr(
        settings.mcp,
        "content_ops_marketer_verify_oauth_state_file",
        str(state_file),
    )

    app = adapter._streamable_http_app()
    provider = verify._oauth_provider

    assert not isinstance(app, BearerAuthMiddleware)
    assert provider is not None
    assert provider.account_id == "acct-oauth"
    assert provider.scopes == [DEFAULT_CONTENT_OPS_VERIFY_SCOPE]
    assert provider.state_file == state_file
    assert adapter.mcp.settings.auth.required_scopes == [DEFAULT_CONTENT_OPS_VERIFY_SCOPE]
    assert adapter.mcp._auth_server_provider is provider
    assert adapter.mcp._token_verifier.provider is provider


def test_chatgpt_adapter_http_wraps_with_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "content_ops_marketer_verify_auth_mode", "bearer")
    monkeypatch.setattr(
        settings.mcp,
        "auth_token",
        "content-ops-bearer-token-with-enough-entropy",
    )

    app = adapter._streamable_http_app()

    assert isinstance(app, BearerAuthMiddleware)


def test_content_ops_marketer_verify_oauth_mode_requires_account_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    with pytest.raises(RuntimeError, match="ACCOUNT_ID"):
        verify._streamable_http_app()


def test_content_ops_marketer_verify_oauth_transport_security_allows_configured_hosts() -> None:
    transport = verify._oauth_transport_security_settings(
        issuer_url="https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer",
        resource_url="https://atlas-brain.tailc7bd29.ts.net/content-ops-marketer/mcp",
    )

    assert transport.enable_dns_rebinding_protection is True
    assert "atlas-brain.tailc7bd29.ts.net" in transport.allowed_hosts
    assert "atlas-brain.tailc7bd29.ts.net:443" in transport.allowed_hosts
    assert "localhost:*" in transport.allowed_hosts
    assert "127.0.0.1:*" in transport.allowed_hosts
    assert "evil.example.com" not in transport.allowed_hosts


def test_content_ops_marketer_verify_oauth_transport_security_allows_explicit_default_port() -> None:
    transport = verify._oauth_transport_security_settings(
        issuer_url="https://atlas-brain.tailc7bd29.ts.net:443/content-ops-marketer",
        resource_url="https://atlas-brain.tailc7bd29.ts.net:443/content-ops-marketer/mcp",
    )

    assert "atlas-brain.tailc7bd29.ts.net" in transport.allowed_hosts
    assert "atlas-brain.tailc7bd29.ts.net:443" in transport.allowed_hosts
