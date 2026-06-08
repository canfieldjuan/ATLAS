"""OAuth support for the Content Ops marketer verify MCP server."""

from __future__ import annotations

import json
import os
import time
from html import escape
from pathlib import Path

from mcp.server.auth.provider import TokenError
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from .invoicing_readonly_oauth import (
    InvoicingReadonlyOAuthProvider,
    PendingAuthorization,
    as_any_http_url,
    handle_approval_request as _handle_approval_request,
)


DEFAULT_CONTENT_OPS_VERIFY_SCOPE = "content_ops.review.verify"


class ContentOpsMarketerVerifyOAuthProvider(InvoicingReadonlyOAuthProvider):
    """Single-tenant OAuth provider for marketer content verification."""

    def __init__(
        self,
        *,
        issuer_url: str,
        approval_token: str,
        scopes: list[str] | None = None,
        account_id: str = "",
        authorization_ttl_seconds: int = 300,
        access_token_ttl_seconds: int = 3600,
        state_file: str | Path | None = None,
    ) -> None:
        self.account_id = _clean(account_id)
        self._access_token_account_ids: dict[str, str] = {}
        self._refresh_token_account_ids: dict[str, str] = {}
        super().__init__(
            issuer_url=issuer_url,
            approval_token=approval_token,
            scopes=scopes or [DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
            authorization_ttl_seconds=authorization_ttl_seconds,
            access_token_ttl_seconds=access_token_ttl_seconds,
            state_file=state_file,
        )

    async def exchange_authorization_code(self, client, authorization_code):
        account_id = self.account_id
        if not account_id:
            raise TokenError("invalid_grant", "authorization code is not tenant-bound")
        token = await super().exchange_authorization_code(client, authorization_code)
        self._record_token_binding(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            account_id=account_id,
        )
        self._save_state()
        return token

    async def exchange_refresh_token(self, client, refresh_token, scopes):
        account_id = _clean(self._refresh_token_account_ids.get(refresh_token.token))
        if not account_id:
            raise TokenError("invalid_grant", "refresh token is not tenant-bound")
        token = await super().exchange_refresh_token(client, refresh_token, scopes)
        self._record_token_binding(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            account_id=account_id,
        )
        self._save_state()
        return token

    async def revoke_token(self, token) -> None:
        await super().revoke_token(token)
        token_value = _clean(getattr(token, "token", ""))
        if token_value:
            self._access_token_account_ids.pop(token_value, None)
            self._refresh_token_account_ids.pop(token_value, None)
            self._save_state()

    def account_id_for_access_token(self, access_token: str) -> str | None:
        return _clean(self._access_token_account_ids.get(access_token)) or None

    def _record_token_binding(
        self,
        *,
        access_token: str,
        refresh_token: str | None,
        account_id: str,
    ) -> None:
        account_id = _clean(account_id)
        if not account_id:
            return
        self._access_token_account_ids[access_token] = account_id
        if refresh_token:
            self._refresh_token_account_ids[refresh_token] = account_id

    def _load_state(self) -> None:
        super()._load_state()
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not load OAuth state file {self.state_file}") from exc
        bindings = data.get("content_ops_refresh_token_account_ids", {})
        if not isinstance(bindings, dict):
            raise RuntimeError("OAuth state content_ops_refresh_token_account_ids must be a JSON object")
        for token, account_id in bindings.items():
            if not isinstance(token, str) or not isinstance(account_id, str):
                raise RuntimeError(
                    "OAuth state content_ops_refresh_token_account_ids must map tokens to account ids"
                )
            if token in self._refresh_tokens and _clean(account_id):
                self._refresh_token_account_ids[token] = _clean(account_id)

    def _save_state(self) -> None:
        super()._save_state()
        if self.state_file is None:
            return
        try:
            data = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not update OAuth state file {self.state_file}") from exc
        data["content_ops_refresh_token_account_ids"] = {
            token: account_id
            for token, account_id in sorted(self._refresh_token_account_ids.items())
            if token in self._refresh_tokens and _clean(account_id)
        }
        tmp_path = self.state_file.with_name(f"{self.state_file.name}.tmp")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        os.replace(tmp_path, self.state_file)
        os.chmod(self.state_file, 0o600)


def approval_page(
    provider: ContentOpsMarketerVerifyOAuthProvider,
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
  <head><title>Approve Atlas Content Verification Connector</title></head>
  <body>
    <h1>Approve Atlas content verification access</h1>
    <p>Client: <code>{client}</code></p>
    <p>Scopes: <code>{scopes}</code></p>
    <p>This grants verify-only content review tools for one bound tenant. It does not grant content generation, publishing, approval, registry mutation, search, or fetch tools.</p>
    <form method="post"{action_attr}>
      <input type="hidden" name="request_id" value="{request_value}" />
      <label>Approval token <input name="approval_token" type="password" /></label>
      <button type="submit">Approve content verification connector</button>
    </form>
  </body>
</html>
""".strip()
    )


async def handle_approval_request(
    provider: ContentOpsMarketerVerifyOAuthProvider,
    request: Request,
) -> Response:
    """Handle GET/POST requests for the marketer verify approval page."""
    if request.method == "GET":
        request_id = request.query_params.get("request_id", "")
        return approval_page(provider, request_id)
    return await _handle_approval_request(provider, request)


def validate_oauth_settings(*, issuer_url: str, resource_server_url: str, approval_token: str) -> None:
    """Fail fast on missing OAuth settings before exposing content verdicts."""
    if not issuer_url.strip():
        raise RuntimeError("ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_ISSUER_URL is required in oauth mode")
    if not resource_server_url.strip():
        raise RuntimeError("ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_RESOURCE_URL is required in oauth mode")
    if len(approval_token.strip()) < 24:
        raise RuntimeError(
            "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_OAUTH_APPROVAL_TOKEN must be at least "
            "24 characters in oauth mode"
        )


__all__ = [
    "DEFAULT_CONTENT_OPS_VERIFY_SCOPE",
    "ContentOpsMarketerVerifyOAuthProvider",
    "PendingAuthorization",
    "approval_page",
    "as_any_http_url",
    "handle_approval_request",
    "validate_oauth_settings",
]


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
