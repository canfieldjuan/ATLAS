from __future__ import annotations

import base64
import hashlib
import stat
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from atlas_brain.mcp import invoicing_readonly_server as readonly
from atlas_brain.mcp.invoicing_readonly_oauth import (
    DEFAULT_READONLY_SCOPE,
    InvoicingReadonlyOAuthProvider,
    PendingAuthorization,
    validate_oauth_settings,
)
from mcp.server.auth.provider import AuthorizationParams, TokenError
from mcp.shared.auth import OAuthClientInformationFull


def _client() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="client-1",
        client_secret="secret-1",
        redirect_uris=["https://chat.openai.com/aip/callback"],
        token_endpoint_auth_method="client_secret_post",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=DEFAULT_READONLY_SCOPE,
    )


def _other_client() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="client-2",
        client_secret="secret-2",
        redirect_uris=["https://chat.openai.com/aip/callback"],
        token_endpoint_auth_method="client_secret_post",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=DEFAULT_READONLY_SCOPE,
    )


def _params() -> AuthorizationParams:
    return AuthorizationParams(
        state="state-1",
        scopes=None,
        code_challenge=_challenge("verifier-1"),
        redirect_uri="https://chat.openai.com/aip/callback",
        redirect_uri_provided_explicitly=True,
        resource="https://atlas.example.com/invoicing-readonly/mcp",
    )


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def _reset_readonly_auth_state() -> None:
    readonly._oauth_provider = None
    readonly.mcp.settings.auth = None
    readonly.mcp._auth_server_provider = None
    readonly.mcp._token_verifier = None
    readonly.mcp._session_manager = None


@pytest.fixture(autouse=True)
def reset_readonly_auth_state():
    _reset_readonly_auth_state()
    yield
    _reset_readonly_auth_state()


@pytest.mark.asyncio
async def test_oauth_provider_requires_operator_approval_before_token_exchange():
    provider = InvoicingReadonlyOAuthProvider(
        issuer_url="https://atlas.example.com/invoicing-readonly",
        approval_token="approval-token-with-enough-entropy",
    )
    client = _client()
    await provider.register_client(client)

    approval_url = await provider.authorize(client, _params())
    request_id = parse_qs(urlparse(approval_url).query)["request_id"][0]

    with pytest.raises(PermissionError):
        provider.approve_pending_authorization(
            request_id=request_id,
            approval_token="wrong-token",
        )

    redirect_uri = provider.approve_pending_authorization(
        request_id=request_id,
        approval_token="approval-token-with-enough-entropy",
    )
    code = parse_qs(urlparse(redirect_uri).query)["code"][0]

    auth_code = await provider.load_authorization_code(client, code)
    assert auth_code is not None
    token = await provider.exchange_authorization_code(client, auth_code)

    assert token.token_type == "Bearer"
    assert token.scope == DEFAULT_READONLY_SCOPE
    access = await provider.load_access_token(token.access_token)
    assert access is not None
    assert access.scopes == [DEFAULT_READONLY_SCOPE]
    assert access.resource == "https://atlas.example.com/invoicing-readonly/mcp"
    assert await provider.load_authorization_code(client, code) is None


@pytest.mark.asyncio
async def test_oauth_provider_binds_authorization_codes_to_client():
    provider = InvoicingReadonlyOAuthProvider(
        issuer_url="https://atlas.example.com/invoicing-readonly",
        approval_token="approval-token-with-enough-entropy",
    )
    client = _client()
    other_client = _other_client()
    await provider.register_client(client)
    await provider.register_client(other_client)
    approval_url = await provider.authorize(client, _params())
    request_id = parse_qs(urlparse(approval_url).query)["request_id"][0]
    redirect_uri = provider.approve_pending_authorization(
        request_id=request_id,
        approval_token="approval-token-with-enough-entropy",
    )
    code = parse_qs(urlparse(redirect_uri).query)["code"][0]

    assert await provider.load_authorization_code(other_client, code) is None
    assert await provider.load_authorization_code(client, code) is not None


@pytest.mark.asyncio
async def test_oauth_provider_refresh_token_issues_new_access_token():
    provider = InvoicingReadonlyOAuthProvider(
        issuer_url="https://atlas.example.com/invoicing-readonly",
        approval_token="approval-token-with-enough-entropy",
    )
    client = _client()
    await provider.register_client(client)
    approval_url = await provider.authorize(client, _params())
    request_id = parse_qs(urlparse(approval_url).query)["request_id"][0]
    redirect_uri = provider.approve_pending_authorization(
        request_id=request_id,
        approval_token="approval-token-with-enough-entropy",
    )
    code = parse_qs(urlparse(redirect_uri).query)["code"][0]
    auth_code = await provider.load_authorization_code(client, code)
    assert auth_code is not None
    first_token = await provider.exchange_authorization_code(client, auth_code)
    assert first_token.refresh_token is not None

    refresh = await provider.load_refresh_token(client, first_token.refresh_token)
    assert refresh is not None
    second_token = await provider.exchange_refresh_token(client, refresh, [DEFAULT_READONLY_SCOPE])

    assert second_token.access_token != first_token.access_token
    assert second_token.refresh_token == first_token.refresh_token


@pytest.mark.asyncio
async def test_oauth_provider_state_file_survives_restart(tmp_path):
    state_file = tmp_path / "readonly-oauth-state.json"
    provider = InvoicingReadonlyOAuthProvider(
        issuer_url="https://atlas.example.com/invoicing-readonly",
        approval_token="approval-token-with-enough-entropy",
        state_file=state_file,
    )
    client = _client()
    await provider.register_client(client)
    approval_url = await provider.authorize(client, _params())
    request_id = parse_qs(urlparse(approval_url).query)["request_id"][0]
    redirect_uri = provider.approve_pending_authorization(
        request_id=request_id,
        approval_token="approval-token-with-enough-entropy",
    )
    code = parse_qs(urlparse(redirect_uri).query)["code"][0]
    auth_code = await provider.load_authorization_code(client, code)
    assert auth_code is not None
    first_token = await provider.exchange_authorization_code(client, auth_code)
    assert first_token.refresh_token is not None

    restarted = InvoicingReadonlyOAuthProvider(
        issuer_url="https://atlas.example.com/invoicing-readonly",
        approval_token="approval-token-with-enough-entropy",
        state_file=state_file,
    )
    loaded_client = await restarted.get_client(client.client_id)
    assert loaded_client is not None
    assert loaded_client.client_secret == "secret-1"
    refresh = await restarted.load_refresh_token(loaded_client, first_token.refresh_token)
    assert refresh is not None

    second_token = await restarted.exchange_refresh_token(
        loaded_client,
        refresh,
        [DEFAULT_READONLY_SCOPE],
    )

    assert second_token.refresh_token == first_token.refresh_token
    assert second_token.access_token != first_token.access_token
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_oauth_provider_state_file_caps_persisted_client_registrations(tmp_path):
    provider = InvoicingReadonlyOAuthProvider(
        issuer_url="https://atlas.example.com/invoicing-readonly",
        approval_token="approval-token-with-enough-entropy",
        state_file=tmp_path / "readonly-oauth-state.json",
        max_persisted_clients=1,
    )
    await provider.register_client(_client())

    with pytest.raises(RuntimeError, match="registration limit"):
        await provider.register_client(_other_client())

    assert await provider.get_client("client-1") is not None
    assert await provider.get_client("client-2") is None


@pytest.mark.asyncio
async def test_oauth_provider_binds_refresh_tokens_to_client():
    provider = InvoicingReadonlyOAuthProvider(
        issuer_url="https://atlas.example.com/invoicing-readonly",
        approval_token="approval-token-with-enough-entropy",
    )
    client = _client()
    other_client = _other_client()
    await provider.register_client(client)
    await provider.register_client(other_client)
    approval_url = await provider.authorize(client, _params())
    request_id = parse_qs(urlparse(approval_url).query)["request_id"][0]
    redirect_uri = provider.approve_pending_authorization(
        request_id=request_id,
        approval_token="approval-token-with-enough-entropy",
    )
    code = parse_qs(urlparse(redirect_uri).query)["code"][0]
    auth_code = await provider.load_authorization_code(client, code)
    assert auth_code is not None
    token = await provider.exchange_authorization_code(client, auth_code)
    assert token.refresh_token is not None

    assert await provider.load_refresh_token(other_client, token.refresh_token) is None
    refresh = await provider.load_refresh_token(client, token.refresh_token)
    assert refresh is not None

    with pytest.raises(TokenError, match="refresh token does not belong to client"):
        await provider.exchange_refresh_token(other_client, refresh, [DEFAULT_READONLY_SCOPE])


def test_validate_oauth_settings_requires_approval_token():
    with pytest.raises(RuntimeError, match="OAUTH_APPROVAL_TOKEN"):
        validate_oauth_settings(
            issuer_url="https://atlas.example.com/invoicing-readonly",
            resource_server_url="https://atlas.example.com/invoicing-readonly/mcp",
            approval_token="short",
        )


def test_streamable_http_app_in_oauth_mode_exposes_metadata_and_requires_auth(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ATLAS_MCP_INVOICING_READONLY_AUTH_MODE", "oauth")
    monkeypatch.setenv(
        "ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL",
        "https://atlas.example.com/invoicing-readonly",
    )
    monkeypatch.setenv(
        "ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL",
        "https://atlas.example.com/invoicing-readonly/mcp",
    )
    monkeypatch.setenv(
        "ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN",
        "approval-token-with-enough-entropy",
    )
    state_file = tmp_path / "readonly-state.json"
    monkeypatch.setenv("ATLAS_MCP_INVOICING_READONLY_OAUTH_STATE_FILE", str(state_file))

    app = readonly._streamable_http_app()
    assert readonly._oauth_provider is not None
    assert readonly._oauth_provider.state_file == state_file

    with TestClient(app) as client:
        auth_metadata = client.get("/.well-known/oauth-authorization-server")
        assert auth_metadata.status_code == 200
        assert auth_metadata.json()["registration_endpoint"] == (
            "https://atlas.example.com/invoicing-readonly/register"
        )

        resource_metadata = client.get(
            "/.well-known/oauth-protected-resource/invoicing-readonly/mcp"
        )
        assert resource_metadata.status_code == 200
        assert resource_metadata.json()["resource"] == (
            "https://atlas.example.com/invoicing-readonly/mcp"
        )

        response = client.get("/mcp")
        assert response.status_code == 401
        assert "resource_metadata=" in response.headers["www-authenticate"]


def test_approval_page_omits_absolute_form_action_for_prefixed_mount(monkeypatch):
    monkeypatch.setenv("ATLAS_MCP_INVOICING_READONLY_AUTH_MODE", "oauth")
    monkeypatch.setenv(
        "ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL",
        "https://atlas.example.com/invoicing-readonly",
    )
    monkeypatch.setenv(
        "ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL",
        "https://atlas.example.com/invoicing-readonly/mcp",
    )
    monkeypatch.setenv(
        "ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN",
        "approval-token-with-enough-entropy",
    )

    app = readonly._streamable_http_app()
    provider = readonly._oauth_provider
    assert provider is not None
    provider._pending["request-1"] = PendingAuthorization(
        request_id="request-1",
        client_id="client-1",
        params=_params(),
        scopes=[DEFAULT_READONLY_SCOPE],
        expires_at=9999999999,
    )

    with TestClient(app, root_path="/invoicing-readonly") as client:
        response = client.get("/oauth/approve?request_id=request-1")

    assert response.status_code == 200
    assert '<form method="post">' in response.text
    assert 'action="/oauth/approve"' not in response.text
    assert 'action="/invoicing-readonly/oauth/approve"' not in response.text


def test_streamable_http_app_rejects_invalid_auth_mode(monkeypatch):
    monkeypatch.setenv("ATLAS_MCP_INVOICING_READONLY_AUTH_MODE", "open")

    with pytest.raises(RuntimeError, match="AUTH_MODE"):
        readonly._streamable_http_app()
