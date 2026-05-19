#!/usr/bin/env python3
"""Verify the draft-writer invoicing MCP connector boundary."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


DEFAULT_URL = "http://127.0.0.1:8066/mcp"
EXPECTED_DRAFT_WRITER_TOOLS = {
    "create_draft_invoice",
    "update_draft_invoice",
    "get_invoice",
    "list_pending_drafts",
}
DENIED_TOOLS = {
    "approve_and_send",
    "create_invoice",
    "create_service",
    "customer_balance",
    "export_invoice_pdf",
    "list_invoices",
    "list_services",
    "mark_void",
    "payment_history",
    "record_payment",
    "search_invoices",
    "send_invoice",
    "set_service_status",
    "update_invoice",
    "update_service",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-check the authenticated draft-writer invoicing MCP connector."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ATLAS_INVOICING_DRAFT_WRITER_MCP_URL", DEFAULT_URL),
        help=(
            "Full Streamable HTTP MCP URL. Defaults to "
            "ATLAS_INVOICING_DRAFT_WRITER_MCP_URL or http://127.0.0.1:8066/mcp."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("ATLAS_MCP_AUTH_TOKEN", ""),
        help="Bearer token. Defaults to ATLAS_MCP_AUTH_TOKEN.",
    )
    return parser


def _tool_surface_errors(tool_names: set[str]) -> list[str]:
    errors: list[str] = []
    missing = EXPECTED_DRAFT_WRITER_TOOLS - tool_names
    unexpected = tool_names - EXPECTED_DRAFT_WRITER_TOOLS
    denied = DENIED_TOOLS & tool_names
    if missing:
        errors.append(f"missing draft-writer tools: {', '.join(sorted(missing))}")
    if unexpected:
        errors.append(f"unexpected tools exposed: {', '.join(sorted(unexpected))}")
    if denied:
        errors.append(f"denied tools exposed: {', '.join(sorted(denied))}")
    return errors


def _unauth_status_code(url: str) -> int:
    try:
        response = httpx.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            timeout=10.0,
        )
    except httpx.RequestError as exc:
        raise RuntimeError(f"unauthenticated probe failed: {exc}") from exc
    return response.status_code


async def _list_tools(url: str, token: str) -> set[str]:
    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(url, headers=headers) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            return {tool.name for tool in tools.tools}


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    token = args.token.strip()
    if not token:
        print("ATLAS_MCP_AUTH_TOKEN or --token is required", file=sys.stderr)
        return 2

    try:
        status_code = _unauth_status_code(args.url)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if status_code != 401:
        print(
            f"expected unauthenticated request to return 401, got {status_code}",
            file=sys.stderr,
        )
        return 1

    try:
        tool_names = asyncio.run(_list_tools(args.url, token))
    except Exception as exc:
        print(f"authenticated MCP tool-list failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _tool_surface_errors(tool_names)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"authenticated MCP connector exposes {len(tool_names)} draft-writer tools:")
    for tool_name in sorted(tool_names):
        print(f"- {tool_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
