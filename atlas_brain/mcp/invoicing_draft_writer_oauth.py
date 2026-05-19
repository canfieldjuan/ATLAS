"""OAuth support for the draft-writer invoicing MCP server."""

from __future__ import annotations

import time
from html import escape

from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from .invoicing_readonly_oauth import (
    InvoicingReadonlyOAuthProvider,
    PendingAuthorization,
    as_any_http_url,
    handle_approval_request as _handle_approval_request,
)


DEFAULT_DRAFT_WRITE_SCOPE = "invoices.draft.write"


class InvoicingDraftWriterOAuthProvider(InvoicingReadonlyOAuthProvider):
    """Single-operator OAuth provider for draft invoice write access."""

    def __init__(
        self,
        *,
        issuer_url: str,
        approval_token: str,
        scopes: list[str] | None = None,
        authorization_ttl_seconds: int = 300,
        access_token_ttl_seconds: int = 3600,
    ) -> None:
        super().__init__(
            issuer_url=issuer_url,
            approval_token=approval_token,
            scopes=scopes or [DEFAULT_DRAFT_WRITE_SCOPE],
            authorization_ttl_seconds=authorization_ttl_seconds,
            access_token_ttl_seconds=access_token_ttl_seconds,
        )


def approval_page(
    provider: InvoicingDraftWriterOAuthProvider,
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
  <head><title>Approve Atlas Draft Invoice Connector</title></head>
  <body>
    <h1>Approve Atlas draft invoice write access</h1>
    <p>Client: <code>{client}</code></p>
    <p>Scopes: <code>{scopes}</code></p>
    <p>This grants draft invoice creation/update tools for operator review. It does not grant sending, payment recording, voiding, PDF export, or service mutation tools.</p>
    <form method="post"{action_attr}>
      <input type="hidden" name="request_id" value="{request_value}" />
      <label>Approval token <input name="approval_token" type="password" /></label>
      <button type="submit">Approve draft writer connector</button>
    </form>
  </body>
</html>
""".strip()
    )


async def handle_approval_request(provider: InvoicingDraftWriterOAuthProvider, request: Request) -> Response:
    """Handle GET/POST requests for the draft-writer operator approval page."""
    if request.method == "GET":
        request_id = request.query_params.get("request_id", "")
        return approval_page(provider, request_id)
    return await _handle_approval_request(provider, request)


def validate_oauth_settings(*, issuer_url: str, resource_server_url: str, approval_token: str) -> None:
    """Fail fast on missing OAuth settings before exposing draft writes."""
    if not issuer_url.strip():
        raise RuntimeError("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL is required in oauth mode")
    if not resource_server_url.strip():
        raise RuntimeError("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL is required in oauth mode")
    if len(approval_token.strip()) < 24:
        raise RuntimeError(
            "ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN must be at least "
            "24 characters in oauth mode"
        )


__all__ = [
    "DEFAULT_DRAFT_WRITE_SCOPE",
    "InvoicingDraftWriterOAuthProvider",
    "PendingAuthorization",
    "approval_page",
    "as_any_http_url",
    "handle_approval_request",
    "validate_oauth_settings",
]
