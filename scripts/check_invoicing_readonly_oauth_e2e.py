#!/usr/bin/env python3
"""Run a no-invoice OAuth e2e smoke for the read-only invoicing MCP connector."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


DEFAULT_SCOPE = "invoices.read"
DEFAULT_REDIRECT_URI = "https://chat.openai.com/aip/callback"
EXPECTED_READONLY_TOOLS = {
    "customer_balance",
    "get_invoice",
    "get_service",
    "list_invoices",
    "list_pending_drafts",
    "list_services",
    "payment_history",
    "search_invoices",
}
KNOWN_MUTATING_TOOLS = {
    "approve_and_send",
    "create_invoice",
    "create_service",
    "export_invoice_pdf",
    "mark_void",
    "record_payment",
    "send_invoice",
    "set_service_status",
    "update_invoice",
    "update_service",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


_NO_REDIRECT = urllib.request.build_opener(_NoRedirect)


@dataclass(frozen=True)
class OAuthE2EConfig:
    issuer_url: str
    resource_url: str
    approval_token: str
    redirect_uri: str
    scope: str
    timeout: float


@dataclass(frozen=True)
class RegisteredClient:
    client_id: str
    client_secret: str


@dataclass(frozen=True)
class AuthorizationRequest:
    approval_url: str
    request_id: str
    verifier: str


@dataclass(frozen=True)
class TokenResult:
    access_token: str
    scope: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a public OAuth e2e smoke for the read-only invoicing MCP connector. "
            "This lists tools only and does not read invoice data."
        )
    )
    parser.add_argument(
        "--issuer-url",
        default=os.environ.get("ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL", ""),
        help="Public OAuth issuer URL. Defaults to ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL.",
    )
    parser.add_argument(
        "--resource-url",
        default=os.environ.get("ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL", ""),
        help=(
            "Public Streamable HTTP MCP resource URL. Defaults to "
            "ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL."
        ),
    )
    parser.add_argument(
        "--approval-token",
        default=os.environ.get("ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN", ""),
        help="Operator approval token. Defaults to ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN.",
    )
    parser.add_argument(
        "--redirect-uri",
        default=DEFAULT_REDIRECT_URI,
        help=f"OAuth redirect URI to register. Default: {DEFAULT_REDIRECT_URI}.",
    )
    parser.add_argument(
        "--scope",
        default=DEFAULT_SCOPE,
        help=f"Required OAuth scope. Default: {DEFAULT_SCOPE}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds. Default: 10.",
    )
    return parser


def _strip_trailing_slash(url: str) -> str:
    return url.strip().rstrip("/")


def _config_from_args(args: argparse.Namespace) -> OAuthE2EConfig:
    issuer_url = _strip_trailing_slash(args.issuer_url)
    resource_url = _strip_trailing_slash(args.resource_url)
    approval_token = args.approval_token.strip()
    redirect_uri = args.redirect_uri.strip()
    scope = args.scope.strip()
    missing: list[str] = []
    if not issuer_url:
        missing.append("--issuer-url or ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL")
    if not resource_url:
        missing.append("--resource-url or ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL")
    if not approval_token:
        missing.append("--approval-token or ATLAS_MCP_INVOICING_READONLY_OAUTH_APPROVAL_TOKEN")
    if not redirect_uri:
        missing.append("--redirect-uri")
    if not scope:
        missing.append("--scope")
    if missing:
        raise ValueError("missing required values: " + ", ".join(missing))
    return OAuthE2EConfig(
        issuer_url=issuer_url,
        resource_url=resource_url,
        approval_token=approval_token,
        redirect_uri=redirect_uri,
        scope=scope,
        timeout=args.timeout,
    )


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _post_json(url: str, payload: Mapping[str, Any], timeout: float) -> tuple[int, Mapping[str, str], dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with _NO_REDIRECT.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body) if body else {}
            if not isinstance(data, dict):
                raise RuntimeError(f"{url} returned non-object JSON")
            return response.status, response.headers, data
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{url} did not return JSON") from exc


def _post_form(
    url: str,
    payload: Mapping[str, str],
    timeout: float,
    *,
    follow_redirects: bool = False,
) -> tuple[int, Mapping[str, str], str]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    opener = urllib.request.urlopen if follow_redirects else _NO_REDIRECT.open
    try:
        with opener(request, timeout=timeout) as response:
            return response.status, response.headers, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in {302, 303, 307, 308}:
            return exc.code, exc.headers, exc.read().decode("utf-8")
        raise RuntimeError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} request failed: {exc.reason}") from exc


def _get_no_redirect(url: str, timeout: float) -> tuple[int, Mapping[str, str], str]:
    request = urllib.request.Request(url, method="GET")
    try:
        with _NO_REDIRECT.open(request, timeout=timeout) as response:
            return response.status, response.headers, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in {302, 303, 307, 308}:
            return exc.code, exc.headers, exc.read().decode("utf-8")
        raise RuntimeError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} request failed: {exc.reason}") from exc


def _register_client(config: OAuthE2EConfig) -> RegisteredClient:
    status, _headers, body = _post_json(
        f"{config.issuer_url}/register",
        {
            "client_name": "Atlas read-only invoicing OAuth e2e smoke",
            "redirect_uris": [config.redirect_uri],
            "token_endpoint_auth_method": "client_secret_post",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": config.scope,
        },
        config.timeout,
    )
    if status not in {200, 201}:
        raise RuntimeError(f"client registration returned HTTP {status}")
    client_id = str(body.get("client_id") or "")
    client_secret = str(body.get("client_secret") or "")
    if not client_id or not client_secret:
        raise RuntimeError("client registration response missing client_id or client_secret")
    return RegisteredClient(client_id=client_id, client_secret=client_secret)


def _start_authorization(config: OAuthE2EConfig, client: RegisteredClient) -> AuthorizationRequest:
    verifier = secrets.token_urlsafe(48)
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": config.scope,
            "state": secrets.token_urlsafe(16),
            "code_challenge": _pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "resource": config.resource_url,
        }
    )
    status, headers, _body = _get_no_redirect(f"{config.issuer_url}/authorize?{query}", config.timeout)
    if status not in {302, 303}:
        raise RuntimeError(f"authorization endpoint returned HTTP {status}")
    approval_url = str(headers.get("Location") or headers.get("location") or "")
    if not approval_url.startswith(f"{config.issuer_url}/oauth/approve?"):
        raise RuntimeError("authorization endpoint did not redirect to the approval page")
    request_id = urllib.parse.parse_qs(urllib.parse.urlparse(approval_url).query).get("request_id", [""])[0]
    if not request_id:
        raise RuntimeError("approval redirect missing request_id")
    return AuthorizationRequest(
        approval_url=approval_url,
        request_id=request_id,
        verifier=verifier,
    )


def _approve_authorization(config: OAuthE2EConfig, auth: AuthorizationRequest) -> str:
    status, headers, _body = _post_form(
        f"{config.issuer_url}/oauth/approve",
        {
            "request_id": auth.request_id,
            "approval_token": config.approval_token,
        },
        config.timeout,
    )
    if status not in {302, 303}:
        raise RuntimeError(f"approval endpoint returned HTTP {status}")
    redirect_location = str(headers.get("Location") or headers.get("location") or "")
    code = urllib.parse.parse_qs(urllib.parse.urlparse(redirect_location).query).get("code", [""])[0]
    if not code:
        raise RuntimeError("approval redirect missing authorization code")
    return code


def _exchange_token(
    config: OAuthE2EConfig,
    client: RegisteredClient,
    auth: AuthorizationRequest,
    code: str,
) -> TokenResult:
    status, _headers, body = _post_form(
        f"{config.issuer_url}/token",
        {
            "grant_type": "authorization_code",
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "code": code,
            "redirect_uri": config.redirect_uri,
            "code_verifier": auth.verifier,
            "resource": config.resource_url,
        },
        config.timeout,
        follow_redirects=True,
    )
    if status != 200:
        raise RuntimeError(f"token endpoint returned HTTP {status}")
    try:
        token = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("token endpoint did not return JSON") from exc
    if not isinstance(token, dict):
        raise RuntimeError("token endpoint returned non-object JSON")
    if token.get("token_type") != "Bearer":
        raise RuntimeError("token endpoint did not return a Bearer token")
    access_token = str(token.get("access_token") or "")
    scope = str(token.get("scope") or "")
    if not access_token:
        raise RuntimeError("token endpoint response missing access_token")
    if config.scope not in scope.split():
        raise RuntimeError(f"token endpoint response missing scope {config.scope}")
    return TokenResult(access_token=access_token, scope=scope)


async def _list_mcp_tools(config: OAuthE2EConfig, access_token: str) -> set[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        config.resource_url,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=config.timeout,
        sse_read_timeout=config.timeout,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            return {tool.name for tool in result.tools}


def _tool_surface_errors(tool_names: set[str]) -> list[str]:
    errors: list[str] = []
    missing = sorted(EXPECTED_READONLY_TOOLS - tool_names)
    extra = sorted(tool_names - EXPECTED_READONLY_TOOLS)
    mutating = sorted(tool_names & KNOWN_MUTATING_TOOLS)
    if missing:
        errors.append("missing read-only tools: " + ", ".join(missing))
    if extra:
        errors.append("unexpected tools exposed: " + ", ".join(extra))
    if mutating:
        errors.append("mutating tools exposed: " + ", ".join(mutating))
    return errors


async def _run_smoke(config: OAuthE2EConfig) -> set[str]:
    client = _register_client(config)
    auth = _start_authorization(config, client)
    code = _approve_authorization(config, auth)
    token = _exchange_token(config, client, auth, code)
    return await _list_mcp_tools(config, token.access_token)


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        tool_names = asyncio.run(_run_smoke(config))
    except Exception as exc:
        print(f"OAuth e2e smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _tool_surface_errors(tool_names)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("OK: read-only invoicing OAuth e2e smoke completed")
    print("- dynamic client registration succeeded")
    print("- operator approval and token exchange succeeded")
    print(f"- OAuth-authenticated MCP session exposes {len(tool_names)} read-only tools")
    for name in sorted(tool_names):
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
