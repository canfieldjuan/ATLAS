#!/usr/bin/env python3
"""Run a live write smoke for the draft-writer invoicing OAuth connector."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_invoicing_draft_writer_oauth_e2e import (  # noqa: E402
    DEFAULT_REDIRECT_URI,
    DEFAULT_SCOPE,
    OAuthE2EConfig,
    _approve_authorization,
    _config_from_args,
    _exchange_token,
    _register_client,
    _start_authorization,
)


DEFAULT_CUSTOMER_NAME = "ATLAS TEST - DO NOT SEND - Draft Writer Connector"
DEFAULT_BUSINESS_CONTEXT_ID = "atlas-mcp-live-smoke"
DEFAULT_LINE_DESCRIPTION = "Live connector smoke - blocked draft, do not send"
EXPECTED_METADATA = {
    "mcp_connector": "invoicing_draft_writer",
    "operator_review_required": True,
    "created_by_remote_connector": True,
}


@dataclass(frozen=True)
class LiveWriteConfig:
    idempotency_key: str
    customer_name: str
    business_context_id: str


@dataclass(frozen=True)
class LiveWriteResult:
    created: bool
    invoice_number: str
    invoice_id: str | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


def _default_idempotency_key(today: date | None = None) -> str:
    day = today or date.today()
    return f"atlas-draft-writer-live-smoke-{day.isoformat()}-v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or reuse one intentionally blocked draft invoice through "
            "the public draft-writer OAuth MCP connector, then verify readback."
        )
    )
    parser.add_argument(
        "--issuer-url",
        default=os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_ISSUER_URL", ""),
        help="Public OAuth issuer URL.",
    )
    parser.add_argument(
        "--resource-url",
        default=os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_RESOURCE_URL", ""),
        help="Public Streamable HTTP MCP resource URL.",
    )
    parser.add_argument(
        "--approval-token",
        default=os.environ.get("ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN", ""),
        help="Operator approval token.",
    )
    parser.add_argument(
        "--approval-token-file",
        default="",
        help="Read the operator approval token from this local file.",
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
    parser.add_argument(
        "--idempotency-key",
        default="",
        help="Idempotency key for the smoke draft. Defaults to one key per day.",
    )
    parser.add_argument(
        "--customer-name",
        default=DEFAULT_CUSTOMER_NAME,
        help=f"Smoke draft customer name. Default: {DEFAULT_CUSTOMER_NAME}.",
    )
    parser.add_argument(
        "--business-context-id",
        default=DEFAULT_BUSINESS_CONTEXT_ID,
        help=f"Smoke draft business context. Default: {DEFAULT_BUSINESS_CONTEXT_ID}.",
    )
    parser.add_argument(
        "--create-blocked-draft",
        action="store_true",
        help=(
            "Required safety acknowledgement. The smoke creates or reuses a "
            "draft invoice with no email and zero subtotal so it cannot be sent."
        ),
    )
    return parser


def _live_config_from_args(args: argparse.Namespace) -> LiveWriteConfig:
    key = (args.idempotency_key or _default_idempotency_key()).strip()
    customer = args.customer_name.strip()
    context = args.business_context_id.strip()
    missing: list[str] = []
    if not key:
        missing.append("--idempotency-key")
    if not customer:
        missing.append("--customer-name")
    if not context:
        missing.append("--business-context-id")
    if missing:
        raise ValueError("missing required values: " + ", ".join(missing))
    return LiveWriteConfig(
        idempotency_key=key,
        customer_name=customer,
        business_context_id=context,
    )


def _create_arguments(config: LiveWriteConfig) -> dict[str, Any]:
    return {
        "customer_name": config.customer_name,
        "line_items": json.dumps(
            [
                {
                    "description": DEFAULT_LINE_DESCRIPTION,
                    "quantity": 1,
                    "unit_price": 0,
                }
            ]
        ),
        "idempotency_key": config.idempotency_key,
        "due_days": 30,
        "invoice_for": "Draft-writer connector live smoke",
        "business_context_id": config.business_context_id,
        "notes": "Blocked live smoke draft. Do not send.",
    }


def _pending_drafts_arguments(config: LiveWriteConfig) -> dict[str, Any]:
    return {
        "business_context_id": config.business_context_id,
        "only_blocked": True,
        "limit": 200,
    }


def _tool_text(result: Any) -> str:
    return "".join(getattr(item, "text", "") for item in getattr(result, "content", []))


async def _call_json_tool(
    session: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    result = await session.call_tool(tool_name, arguments)
    text = _tool_text(result)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{tool_name} returned non-JSON output") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{tool_name} returned non-object JSON")
    return payload


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_pending_draft(
    pending_payload: dict[str, Any],
    invoice_number: str,
) -> dict[str, Any] | None:
    drafts = pending_payload.get("drafts")
    if not isinstance(drafts, list):
        return None
    for draft in drafts:
        if isinstance(draft, dict) and draft.get("invoice_number") == invoice_number:
            return draft
    return None


def _validate_smoke_result(
    create_payload: dict[str, Any],
    get_payload: dict[str, Any],
    pending_payload: dict[str, Any],
    config: LiveWriteConfig,
) -> list[str]:
    errors: list[str] = []
    if create_payload.get("success") is not True:
        errors.append("create_draft_invoice did not return success=true")

    invoice = create_payload.get("invoice")
    if not isinstance(invoice, dict):
        return errors + ["create_draft_invoice response missing invoice object"]

    invoice_number = str(invoice.get("invoice_number") or "")
    if not invoice_number:
        errors.append("created invoice missing invoice_number")
    if invoice.get("status") != "draft":
        errors.append("created invoice is not draft")
    if invoice.get("customer_name") != config.customer_name:
        errors.append("created invoice customer_name does not match smoke config")
    if invoice.get("business_context_id") != config.business_context_id:
        errors.append("created invoice business_context_id does not match smoke config")
    if invoice.get("customer_email") not in {None, ""}:
        errors.append("smoke invoice unexpectedly has customer_email")
    if _as_float(invoice.get("total_amount")) != 0.0:
        errors.append("smoke invoice total_amount is not zero")

    metadata = invoice.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("created invoice metadata is missing")
    else:
        for key, expected in EXPECTED_METADATA.items():
            if metadata.get(key) != expected:
                errors.append(f"created invoice metadata.{key} mismatch")
        if metadata.get("idempotency_key") != config.idempotency_key:
            errors.append("created invoice metadata.idempotency_key mismatch")

    if get_payload.get("found") is not True:
        errors.append("get_invoice did not find created invoice by invoice number")
    read_invoice = get_payload.get("invoice")
    if not isinstance(read_invoice, dict):
        errors.append("get_invoice response missing invoice object")
    elif invoice_number and read_invoice.get("invoice_number") != invoice_number:
        errors.append("get_invoice returned a different invoice_number")

    pending = _find_pending_draft(pending_payload, invoice_number)
    if pending is None:
        errors.append("list_pending_drafts did not include the smoke invoice")
    else:
        blockers = pending.get("blockers")
        warnings = pending.get("warnings")
        if not isinstance(blockers, list) or "no_email" not in blockers:
            errors.append("pending smoke invoice is missing no_email blocker")
        if pending.get("send_safe") is not False:
            errors.append("pending smoke invoice is not blocked from sending")
        if not isinstance(warnings, list) or "subtotal_zero" not in warnings:
            errors.append("pending smoke invoice is missing subtotal_zero warning")

    return errors


async def _run_live_write_smoke(
    oauth_config: OAuthE2EConfig,
    live_config: LiveWriteConfig,
) -> LiveWriteResult:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    client = _register_client(oauth_config)
    auth = _start_authorization(oauth_config, client)
    code = _approve_authorization(oauth_config, auth)
    token = _exchange_token(oauth_config, client, auth, code)

    async with streamablehttp_client(
        oauth_config.resource_url,
        headers={"Authorization": f"Bearer {token.access_token}"},
        timeout=oauth_config.timeout,
        sse_read_timeout=oauth_config.timeout,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            create_payload = await _call_json_tool(
                session,
                "create_draft_invoice",
                _create_arguments(live_config),
            )
            invoice = create_payload.get("invoice") if isinstance(create_payload, dict) else None
            if not isinstance(invoice, dict):
                raise RuntimeError("create_draft_invoice response missing invoice object")
            invoice_number = str(invoice.get("invoice_number") or "")
            if not invoice_number:
                raise RuntimeError("create_draft_invoice response missing invoice_number")

            get_payload = await _call_json_tool(
                session,
                "get_invoice",
                {"invoice_id": invoice_number},
            )
            pending_payload = await _call_json_tool(
                session,
                "list_pending_drafts",
                _pending_drafts_arguments(live_config),
            )

    errors = _validate_smoke_result(create_payload, get_payload, pending_payload, live_config)
    if errors:
        raise RuntimeError("; ".join(errors))

    pending = _find_pending_draft(pending_payload, invoice_number) or {}
    return LiveWriteResult(
        created=create_payload.get("created") is True,
        invoice_number=invoice_number,
        invoice_id=str(invoice.get("id")) if invoice.get("id") else None,
        blockers=tuple(pending.get("blockers") or ()),
        warnings=tuple(pending.get("warnings") or ()),
    )


def _main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.create_blocked_draft:
        print(
            "refusing to create a live draft without --create-blocked-draft",
            file=sys.stderr,
        )
        return 2
    try:
        oauth_config = _config_from_args(args)
        live_config = _live_config_from_args(args)
        result = asyncio.run(_run_live_write_smoke(oauth_config, live_config))
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    action = "created" if result.created else "reused"
    print("Draft-writer live write smoke completed")
    print(f"- action: {action}")
    print(f"- invoice_number: {result.invoice_number}")
    if result.invoice_id:
        print(f"- invoice_id: {result.invoice_id}")
    print(f"- blockers: {', '.join(result.blockers) or 'none'}")
    print(f"- warnings: {', '.join(result.warnings) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
