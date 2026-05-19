#!/usr/bin/env python3
"""Verify public OAuth discovery for the read-only invoicing MCP connector.

This check is intentionally read-only. It validates public routing and metadata
only; it does not complete OAuth approval, exchange tokens, or call invoice
tools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any


DEFAULT_SCOPE = "invoices.read"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-check OAuth discovery for the read-only invoicing MCP connector."
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


def _authorization_metadata_url(issuer_url: str) -> str:
    return f"{_strip_trailing_slash(issuer_url)}/.well-known/oauth-authorization-server"


def _protected_resource_metadata_url(resource_url: str) -> str:
    parsed = urllib.parse.urlparse(_strip_trailing_slash(resource_url))
    resource_path = parsed.path if parsed.path != "/" else ""
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            f"/.well-known/oauth-protected-resource{resource_path}",
            "",
            "",
            "",
        )
    )


def _fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} request failed: {exc.reason}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{url} did not return JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return data


def _probe_mcp_unauthenticated(url: str, timeout: float) -> tuple[int, Mapping[str, str]]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} request failed: {exc.reason}") from exc


def _list_value(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if isinstance(value, list):
        return value
    return []


def _metadata_errors(
    *,
    issuer_url: str,
    resource_url: str,
    scope: str,
    auth_metadata: Mapping[str, Any],
    resource_metadata: Mapping[str, Any],
    mcp_status: int,
    mcp_headers: Mapping[str, str],
) -> list[str]:
    issuer = _strip_trailing_slash(issuer_url)
    resource = _strip_trailing_slash(resource_url)
    protected_url = _protected_resource_metadata_url(resource)
    errors: list[str] = []

    expected_auth = {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token",
        "registration_endpoint": f"{issuer}/register",
    }
    for key, expected in expected_auth.items():
        if str(auth_metadata.get(key) or "").rstrip("/") != expected:
            errors.append(f"authorization metadata {key} != {expected}")

    if scope not in _list_value(auth_metadata, "scopes_supported"):
        errors.append(f"authorization metadata missing scope {scope}")
    if "authorization_code" not in _list_value(auth_metadata, "grant_types_supported"):
        errors.append("authorization metadata missing authorization_code grant")

    if str(resource_metadata.get("resource") or "").rstrip("/") != resource:
        errors.append(f"resource metadata resource != {resource}")
    if issuer not in [str(item).rstrip("/") for item in _list_value(resource_metadata, "authorization_servers")]:
        errors.append(f"resource metadata missing authorization server {issuer}")
    if scope not in _list_value(resource_metadata, "scopes_supported"):
        errors.append(f"resource metadata missing scope {scope}")

    if mcp_status != 401:
        errors.append(f"unauthenticated MCP probe returned {mcp_status}, expected 401")

    authenticate = mcp_headers.get("www-authenticate") or mcp_headers.get("WWW-Authenticate") or ""
    if protected_url not in authenticate:
        errors.append(f"WWW-Authenticate header missing resource_metadata={protected_url}")

    return errors


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    issuer_url = _strip_trailing_slash(args.issuer_url)
    resource_url = _strip_trailing_slash(args.resource_url)
    if not issuer_url:
        print("--issuer-url or ATLAS_MCP_INVOICING_READONLY_OAUTH_ISSUER_URL is required", file=sys.stderr)
        return 2
    if not resource_url:
        print(
            "--resource-url or ATLAS_MCP_INVOICING_READONLY_OAUTH_RESOURCE_URL is required",
            file=sys.stderr,
        )
        return 2

    auth_url = _authorization_metadata_url(issuer_url)
    protected_url = _protected_resource_metadata_url(resource_url)
    try:
        auth_metadata = _fetch_json(auth_url, args.timeout)
        resource_metadata = _fetch_json(protected_url, args.timeout)
        mcp_status, mcp_headers = _probe_mcp_unauthenticated(resource_url, args.timeout)
    except Exception as exc:
        print(f"OAuth discovery smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _metadata_errors(
        issuer_url=issuer_url,
        resource_url=resource_url,
        scope=args.scope,
        auth_metadata=auth_metadata,
        resource_metadata=resource_metadata,
        mcp_status=mcp_status,
        mcp_headers=mcp_headers,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("OK: read-only invoicing OAuth discovery is routable")
    print(f"- authorization metadata: {auth_url}")
    print(f"- protected resource metadata: {protected_url}")
    print(f"- MCP resource: {resource_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
