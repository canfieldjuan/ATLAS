"""
Atlas Email MCP Server.

Provider-agnostic MCP server for email operations backed by Gmail + Resend.

Sending:  Gmail preferred; Resend as fallback.
Reading:  Gmail only (inbox, threads, search).
History:  Atlas sent_emails DB table (all outbound email from Atlas).

Tools:
    send_email          — send a plain email
    send_estimate       — send a cleaning estimate confirmation (templated)
    send_proposal       — send a cleaning proposal (templated, auto-PDF)
    list_inbox          — list Gmail inbox messages
    get_message         — fetch a full Gmail message with body
    search_inbox        — search Gmail with arbitrary query syntax
    get_thread          — fetch a Gmail thread
    list_sent_history   — query Atlas sent_emails history from the DB

Run:
    python -m atlas_brain.mcp.email_server          # stdio (Claude Desktop / Cursor)
    python -m atlas_brain.mcp.email_server --sse    # SSE HTTP transport
"""

import json
import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.email")

mcp = FastMCP(
    "atlas-email",
    instructions=(
        "Email MCP server for Atlas. "
        "Handles sending (Gmail preferred / Resend fallback) and reading (Gmail). "
        "For business emails (estimates, proposals) use the specialized tools. "
        "For generic messages use send_email. "
        "Always summarise the email content and confirm with the user before "
        "sending unless explicitly told to auto-send."
    ),
)


def _provider():
    from ..services.email_provider import get_email_provider

    return get_email_provider()


def _to_list(value: Optional[str | list[str]]) -> list[str]:
    """Normalise a single address or comma-separated string to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [v.strip() for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Tool: send_email
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_email(
    to: str,
    subject: str,
    body: str,
    from_email: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> str:
    """
    Send a plain email.

    to / cc / bcc: single address or comma-separated list.
    Attempts Gmail first; falls back to Resend if Gmail is unavailable.
    """
    try:
        result = await _provider().send(
            to=_to_list(to),
            subject=subject,
            body=body,
            from_email=from_email,
            cc=_to_list(cc) or None,
            bcc=_to_list(bcc) or None,
            reply_to=reply_to,
        )
        return json.dumps({"success": True, "result": result}, default=str)
    except Exception as exc:
        logger.exception("send_email error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: send_estimate
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_estimate(
    to: str,
    client_name: str,
    address: str,
    service_date: str,
    service_time: str,
    price: str,
    client_type: str,
) -> str:
    """
    Send a cleaning estimate confirmation email using a professional template.

    client_type: 'business' or 'residential'
    price: numeric string without the dollar sign (e.g. '150.00')
    service_date: human-readable date (e.g. 'January 20, 2026')
    service_time: human-readable time (e.g. '9:00 AM')
    """
    try:
        from ..tools.email import estimate_email_tool

        result = await estimate_email_tool.execute({
            "to": to,
            "client_name": client_name,
            "address": address,
            "service_date": service_date,
            "service_time": service_time,
            "price": price,
            "client_type": client_type,
        })
        return json.dumps(
            {"success": result.success, "message": result.message, "data": result.data},
            default=str,
        )
    except Exception as exc:
        logger.exception("send_estimate error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: send_proposal
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_proposal(
    to: str,
    client_name: str,
    contact_name: str,
    address: str,
    areas_to_clean: str,
    cleaning_description: str,
    price: str,
    client_type: str,
    frequency: str = "As needed",
    contact_phone: Optional[str] = None,
) -> str:
    """
    Send a cleaning proposal email with optional auto-attached PDF.

    client_type: 'business' or 'residential'
    contact_phone: required for business proposals
    areas_to_clean: e.g. 'Offices, Bathrooms, Break Room'
    cleaning_description: e.g. 'Dust surfaces, vacuum floors, empty trash'
    frequency: e.g. 'Weekly', 'Bi-weekly', 'Monthly', 'As needed'

    If a PDF exists at the configured proposals_dir matching client_name,
    it is automatically attached.
    """
    try:
        from ..tools.email import proposal_email_tool

        result = await proposal_email_tool.execute({
            "to": to,
            "client_name": client_name,
            "contact_name": contact_name,
            "address": address,
            "areas_to_clean": areas_to_clean,
            "cleaning_description": cleaning_description,
            "price": price,
            "client_type": client_type,
            "frequency": frequency,
            "contact_phone": contact_phone,
        })
        return json.dumps(
            {"success": result.success, "message": result.message, "data": result.data},
            default=str,
        )
    except Exception as exc:
        logger.exception("send_proposal error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_inbox
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_inbox(
    query: str = "is:unread",
    max_results: int = 20,
) -> str:
    """
    List Gmail inbox messages matching a search query.

    query: Gmail search syntax (default 'is:unread'). Examples:
        'is:unread newer_than:1d'
        'from:john@example.com'
        'subject:invoice has:attachment'
        'label:important'
    max_results: capped at 100
    """
    try:
        messages = await _provider().list_messages(
            query=query, max_results=min(max_results, 100)
        )
        return json.dumps({"messages": messages, "count": len(messages)}, default=str)
    except Exception as exc:
        logger.exception("list_inbox error")
        return json.dumps({"error": str(exc), "messages": []})


# ---------------------------------------------------------------------------
# Tool: get_message
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_message(message_id: str) -> str:
    """
    Fetch a Gmail message with its full body content.

    Returns from, subject, date, snippet, body_text, label_ids, thread_id.
    """
    try:
        msg = await _provider().get_message(message_id)
        return json.dumps({"message": msg}, default=str)
    except Exception as exc:
        logger.exception("get_message error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: search_inbox
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_inbox(query: str, max_results: int = 20) -> str:
    """
    Search Gmail inbox and return messages with metadata.

    Uses full Gmail search syntax:
        from:alice@example.com
        subject:"invoice" newer_than:7d
        is:unread has:attachment

    Returns up to max_results message stubs; fetches metadata for the first 10.
    """
    try:
        provider = _provider()
        message_stubs = await provider.list_messages(
            query=query, max_results=min(max_results, 100)
        )
        # Enrich up to 10 results with metadata for useful context
        enriched: list[dict] = []
        for stub in message_stubs[:10]:
            try:
                meta = await provider.get_message_metadata(stub["id"])
                enriched.append(meta)
            except Exception:
                enriched.append(stub)

        return json.dumps(
            {"results": enriched, "total_matched": len(message_stubs)}, default=str
        )
    except Exception as exc:
        logger.exception("search_inbox error")
        return json.dumps({"error": str(exc), "results": []})


# ---------------------------------------------------------------------------
# Tool: get_thread
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_thread(thread_id: str) -> str:
    """
    Fetch a Gmail thread with all of its messages (metadata format).

    Useful for reading an entire conversation chain in one call.
    """
    try:
        thread = await _provider().get_thread(thread_id)
        return json.dumps({"thread": thread}, default=str)
    except Exception as exc:
        logger.exception("get_thread error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_sent_history
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_sent_history(
    hours: int = 24,
    template_type: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Query the history of emails sent through Atlas (from the DB).

    hours: how far back to look (default 24)
    template_type: 'estimate' | 'proposal' | 'generic' — or omit for all
    limit: max results (default 20)

    This covers all outbound mail sent via Atlas tools, not Gmail generally.
    """
    try:
        from ..tools.email import query_email_history_tool

        result = await query_email_history_tool.execute({
            "hours": hours,
            "template_type": template_type,
            "limit": limit,
        })
        return json.dumps(
            {"success": result.success, "data": result.data, "message": result.message},
            default=str,
        )
    except Exception as exc:
        logger.exception("list_sent_history error")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.email_port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
