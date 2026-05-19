"""OAuth support for the read-only invoicing MCP server.

The provider is intentionally small and process-local. It exists to let
ChatGPT-style remote MCP clients complete a standard OAuth authorization-code
flow without exposing mutating invoice tools or accepting anonymous access.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from html import escape
from typing import Any

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response


DEFAULT_READONLY_SCOPE = "invoices.read"


@dataclass(frozen=True)
class PendingAuthorization:
    request_id: str
    client_id: str
    params: AuthorizationParams
    scopes: list[str]
    expires_at: float


class InvoicingReadonlyOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """Single-operator in-memory OAuth provider for read-only invoice access."""

    def __init__(
        self,
        *,
        issuer_url: str,
        approval_token: str,
        scopes: list[str] | None = None,
        authorization_ttl_seconds: int = 300,
        access_token_ttl_seconds: int = 3600,
    ) -> None:
        self.issuer_url = issuer_url.rstrip("/")
        self.approval_token = approval_token
        self.scopes = scopes or [DEFAULT_READONLY_SCOPE]
        self.authorization_ttl_seconds = authorization_ttl_seconds
        self.access_token_ttl_seconds = access_token_ttl_seconds

        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._pending: dict[str, PendingAuthorization] = {}
        self._authorization_codes: dict[str, AuthorizationCode] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}
        self._access_tokens: dict[str, AccessToken] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:  # pragma: no cover - registration handler sets this
            client_info.client_id = secrets.token_urlsafe(24)
        self._clients[client_info.client_id] = client_info

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if not client.client_id:
            raise AuthorizeError("invalid_request", "client_id is required")

        scopes = params.scopes or list(self.scopes)
        for scope in scopes:
            if scope not in self.scopes:
                raise AuthorizeError("invalid_scope", f"Unsupported scope: {scope}")

        request_id = secrets.token_urlsafe(24)
        self._pending[request_id] = PendingAuthorization(
            request_id=request_id,
            client_id=client.client_id,
            params=params,
            scopes=scopes,
            expires_at=time.time() + self.authorization_ttl_seconds,
        )
        return f"{self.issuer_url}/oauth/approve?request_id={request_id}"

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        code = self._authorization_codes.get(authorization_code)
        if code is None or code.client_id != client.client_id:
            return None
        return code

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        if not client.client_id:
            raise TokenError("invalid_client", "client_id is required")

        self._authorization_codes.pop(authorization_code.code, None)
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + self.access_token_ttl_seconds

        self._access_tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            expires_at=expires_at,
            resource=authorization_code.resource,
        )
        self._refresh_tokens[refresh_token] = RefreshToken(
            token=refresh_token,
            client_id=client.client_id,
            scopes=authorization_code.scopes,
        )
        return OAuthToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_ttl_seconds,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        token = self._refresh_tokens.get(refresh_token)
        if token is None or token.client_id != client.client_id:
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        if not client.client_id:
            raise TokenError("invalid_client", "client_id is required")
        if refresh_token.client_id != client.client_id:
            raise TokenError("invalid_grant", "refresh token does not belong to client")
        if not scopes:
            scopes = refresh_token.scopes
        if any(scope not in refresh_token.scopes for scope in scopes):
            raise TokenError("invalid_scope", "Requested scope was not granted")

        access_token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + self.access_token_ttl_seconds
        self._access_tokens[access_token] = AccessToken(
            token=access_token,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=expires_at,
        )
        return OAuthToken(
            access_token=access_token,
            refresh_token=refresh_token.token,
            expires_in=self.access_token_ttl_seconds,
            scope=" ".join(scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        return self._access_tokens.get(token)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self._access_tokens.pop(token.token, None)
        self._refresh_tokens.pop(token.token, None)

    def approve_pending_authorization(self, *, request_id: str, approval_token: str) -> str:
        """Approve a pending authorization request and return the redirect URI."""
        if not secrets.compare_digest(approval_token, self.approval_token):
            raise PermissionError("Invalid approval token")

        pending = self._pending.pop(request_id, None)
        if pending is None:
            raise KeyError("Authorization request not found")
        if pending.expires_at < time.time():
            raise TimeoutError("Authorization request expired")

        code = secrets.token_urlsafe(32)
        self._authorization_codes[code] = AuthorizationCode(
            code=code,
            scopes=pending.scopes,
            expires_at=time.time() + self.authorization_ttl_seconds,
            client_id=pending.client_id,
            code_challenge=pending.params.code_challenge,
            redirect_uri=pending.params.redirect_uri,
            redirect_uri_provided_explicitly=pending.params.redirect_uri_provided_explicitly,
            resource=pending.params.resource,
        )
        return construct_redirect_uri(str(pending.params.redirect_uri), code=code, state=pending.params.state)

    def pending_authorization(self, request_id: str) -> PendingAuthorization | None:
        return self._pending.get(request_id)


def approval_page(
    provider: InvoicingReadonlyOAuthProvider,
    request_id: str,
    *,
    form_action: str | None = None,
) -> Response:
    pending = provider.pending_authorization(request_id)
    if pending is None:
        return HTMLResponse("<h1>Authorization request not found</h1>", status_code=404)
    if pending.expires_at < time.time():
        return HTMLResponse("<h1>Authorization request expired</h1>", status_code=410)

    client = escape(pending.client_id)
    scopes = escape(", ".join(pending.scopes))
    request_value = escape(request_id)
    action_attr = ""
    if form_action is not None:
        action_attr = f' action="{escape(form_action, quote=True)}"'
    return HTMLResponse(
        f"""
<!doctype html>
<html>
  <head><title>Approve Atlas Invoicing Connector</title></head>
  <body>
    <h1>Approve Atlas read-only invoicing access</h1>
    <p>Client: <code>{client}</code></p>
    <p>Scopes: <code>{scopes}</code></p>
    <p>This grants read-only invoice review tools. It does not grant invoice creation, sending, payment recording, voiding, or service mutation tools.</p>
    <form method="post"{action_attr}>
      <input type="hidden" name="request_id" value="{request_value}" />
      <label>Approval token <input name="approval_token" type="password" /></label>
      <button type="submit">Approve connector</button>
    </form>
  </body>
</html>
""".strip()
    )


async def handle_approval_request(provider: InvoicingReadonlyOAuthProvider, request: Request) -> Response:
    """Handle GET/POST requests for the operator approval page."""
    if request.method == "GET":
        request_id = request.query_params.get("request_id", "")
        return approval_page(provider, request_id)

    form: FormData = await request.form()
    request_id = str(form.get("request_id") or "")
    approval_token = str(form.get("approval_token") or "")
    try:
        redirect_uri = provider.approve_pending_authorization(
            request_id=request_id,
            approval_token=approval_token,
        )
    except PermissionError:
        return HTMLResponse("<h1>Invalid approval token</h1>", status_code=403)
    except KeyError:
        return HTMLResponse("<h1>Authorization request not found</h1>", status_code=404)
    except TimeoutError:
        return HTMLResponse("<h1>Authorization request expired</h1>", status_code=410)
    return RedirectResponse(redirect_uri, status_code=302)


def validate_oauth_settings(*, issuer_url: str, resource_server_url: str, approval_token: str) -> None:
    """Fail fast on missing OAuth settings before exposing financial data."""
    if not issuer_url.strip():
        raise RuntimeError("ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL is required in oauth mode")
    if not resource_server_url.strip():
        raise RuntimeError("ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL is required in oauth mode")
    if len(approval_token.strip()) < 24:
        raise RuntimeError(
            "ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN must be at least 24 characters in oauth mode"
        )


def as_any_http_url(value: str) -> AnyHttpUrl:
    """Validate a string URL for MCP auth settings."""
    return AnyHttpUrl(value)
