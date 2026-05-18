#!/usr/bin/env python3
"""Verify the read-only invoicing MCP connector boundary.

This check is intentionally read-only. It verifies auth behavior and the tool
surface only; it does not call invoice, service, balance, or payment tools.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8065/mcp"

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-check the authenticated read-only invoicing MCP connector."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ATLAS_INVOICING_READONLY_MCP_URL", DEFAULT_URL),
        help=(
            "Full Streamable HTTP MCP URL. Defaults to "
            "ATLAS_INVOICING_READONLY_MCP_URL or http://127.0.0.1:8065/mcp."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("ATLAS_MCP_AUTH_TOKEN", ""),
        help="Bearer token. Defaults to ATLAS_MCP_AUTH_TOKEN.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds. Default: 10.",
    )
    return parser


def _unauth_status_code(url: str, timeout: float) -> int:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


async def _list_tools(url: str, token: str, timeout: float) -> set[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(
        url,
        headers=headers,
        timeout=timeout,
        sse_read_timeout=timeout,
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


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    token = args.token.strip()
    if not token:
        print("ATLAS_MCP_AUTH_TOKEN or --token is required", file=sys.stderr)
        return 2

    try:
        unauth_status = _unauth_status_code(args.url, args.timeout)
    except Exception as exc:
        print(f"unauthenticated probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if unauth_status != 401:
        print(
            f"expected unauthenticated request to return 401, got {unauth_status}",
            file=sys.stderr,
        )
        return 1

    try:
        tool_names = asyncio.run(_list_tools(args.url, token, args.timeout))
    except Exception as exc:
        print(f"authenticated MCP tool-list failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _tool_surface_errors(tool_names)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"OK: {args.url} exposes {len(tool_names)} read-only invoicing tools")
    for name in sorted(tool_names):
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
