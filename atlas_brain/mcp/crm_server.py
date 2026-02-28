"""
Atlas CRM MCP Server.

Provider-agnostic MCP server that exposes the Atlas CRM (Directus / direct-DB)
to any MCP-compatible client (Claude Desktop, Cursor, custom agents, etc.).

The `contacts` table is the single source of truth for customer data.
Contacts are enriched over time via interaction logs, linked appointments,
and email history — replacing the previous approach of scraping appointment
rows and relying solely on GraphRAG accumulation.

Tools:
    search_contacts         — find contacts by name / phone / email
    get_contact             — fetch a contact by UUID
    create_contact          — create a new contact record
    update_contact          — update contact fields
    delete_contact          — archive (soft-delete) a contact
    list_contacts           — paginated list with filters
    log_interaction         — record a customer touch-point
    get_interactions        — retrieve interaction history
    get_contact_appointments — fetch appointments linked to a contact
    get_customer_context     — unified view: contact + interactions + calls + emails

Run:
    python -m atlas_brain.mcp.crm_server          # stdio (Claude Desktop / Cursor)
    python -m atlas_brain.mcp.crm_server --sse    # SSE HTTP transport
"""

import json
import logging
import sys
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.crm")


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import init_database, close_database
    await init_database()
    logger.info("CRM MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-crm",
    instructions=(
        "CRM server for Atlas. "
        "Contacts are the SINGLE SOURCE OF TRUTH for all customer data. "
        "Always search here first before looking at appointments. "
        "Log every customer interaction (calls, emails, appointments) via "
        "log_interaction to build a complete customer history over time."
    ),
    lifespan=_lifespan,
)


def _provider():
    from ..services.crm_provider import get_crm_provider

    return get_crm_provider()


def _is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Tool: search_contacts
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_contacts(
    query: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    business_context_id: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Search for contacts by name, phone, or email.

    This is the primary customer lookup.  At least one of query / phone /
    email is required.  Searches the CRM contacts table first; if nothing
    is found, falls back to appointment records so legacy customers that
    have not been migrated to the CRM are still discoverable.

    query: partial name match (case-insensitive)
    phone: any format accepted (digits extracted automatically)
    limit: max results (default 20)
    """
    if not any([query, phone, email, business_context_id]):
        return json.dumps({"error": "At least one of query, phone, email, or business_context_id is required"})
    try:
        results = await _provider().search_contacts(
            query=query,
            phone=phone,
            email=email,
            business_context_id=business_context_id,
            limit=limit,
        )
        if results:
            return json.dumps(
                {"found": True, "contacts": results, "count": len(results),
                 "source": "crm"},
                default=str,
            )
    except Exception as exc:
        logger.warning("CRM search failed, trying appointment fallback: %s", exc)

    # ------------------------------------------------------------------
    # Fallback: scrape customer data from appointment rows for contacts
    # not yet in the CRM table.
    # ------------------------------------------------------------------
    try:
        from ..storage.repositories.appointment import get_appointment_repo

        repo = get_appointment_repo()
        appointments = []

        if phone:
            appointments = await repo.get_by_phone(
                phone, status=None, upcoming_only=False, limit=limit,
            )
        if not appointments and query:
            appointments = await repo.search_by_name(
                query, include_history=True, limit=limit,
            )

        if not appointments:
            return json.dumps({"found": False, "contacts": [], "count": 0})

        # Deduplicate by (name, phone) and build contact-shaped dicts
        seen = set()
        contacts = []
        for appt in appointments:
            key = (appt.get("customer_name", ""), appt.get("customer_phone", ""))
            if key in seen:
                continue
            seen.add(key)
            contacts.append({
                "full_name": appt.get("customer_name"),
                "phone": appt.get("customer_phone"),
                "email": appt.get("customer_email"),
                "address": appt.get("customer_address"),
                "source": "appointments",
            })

        return json.dumps(
            {"found": True, "contacts": contacts, "count": len(contacts),
             "source": "appointments"},
            default=str,
        )
    except Exception as fallback_exc:
        logger.exception("search_contacts fallback error")
        return json.dumps(
            {"error": str(fallback_exc), "found": False, "contacts": [],
             "count": 0}
        )


# ---------------------------------------------------------------------------
# Tool: get_contact
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_contact(contact_id: str) -> str:
    """
    Fetch a contact by UUID or name.

    If contact_id is not a valid UUID, treats it as a name search
    and returns the first matching contact.
    """
    try:
        # If it doesn't look like a UUID, search by name instead
        if not _is_uuid(contact_id):
            results = await _provider().search_contacts(query=contact_id, limit=1)
            if results:
                return json.dumps({"found": True, "contact": results[0]}, default=str)
            return json.dumps({"found": False, "contact": None})

        contact = await _provider().get_contact(contact_id)
        if contact is None:
            return json.dumps({"found": False, "contact": None})
        return json.dumps({"found": True, "contact": contact}, default=str)
    except Exception as exc:
        logger.exception("get_contact error")
        return json.dumps({"error": str(exc), "found": False})


# ---------------------------------------------------------------------------
# Tool: create_contact
# ---------------------------------------------------------------------------

@mcp.tool()
async def create_contact(
    full_name: str,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    zip_code: Optional[str] = None,
    business_context_id: Optional[str] = None,
    contact_type: str = "customer",
    notes: Optional[str] = None,
    source: str = "manual",
    tags: Optional[list[str]] = None,
) -> str:
    """
    Create a new contact in the CRM.

    contact_type: customer | lead | prospect | vendor  (default: customer)
    source: manual | phone_call | email | appointment_import | web
    tags: optional list of string tags (e.g. ["vip", "repeat"])
    """
    try:
        parts = full_name.strip().split(" ", 1)
        data: dict = {
            "full_name": full_name,
            "first_name": parts[0] if parts else None,
            "last_name": parts[1] if len(parts) > 1 else None,
            "phone": phone,
            "email": email,
            "address": address,
            "city": city,
            "state": state,
            "zip": zip_code,
            "business_context_id": business_context_id,
            "contact_type": contact_type,
            "notes": notes,
            "source": source,
            "tags": tags or [],
        }
        contact = await _provider().create_contact(data)
        return json.dumps({"success": True, "contact": contact}, default=str)
    except Exception as exc:
        logger.exception("create_contact error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: update_contact
# ---------------------------------------------------------------------------

@mcp.tool()
async def update_contact(
    contact_id: str,
    full_name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    zip_code: Optional[str] = None,
    notes: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> str:
    """
    Update a contact's information.

    Only supply fields you want to change.
    status: active | inactive | archived
    """
    try:
        data = {
            k: v for k, v in {
                "full_name": full_name,
                "phone": phone,
                "email": email,
                "address": address,
                "city": city,
                "state": state,
                "zip": zip_code,
                "notes": notes,
                "status": status,
                "tags": tags,
            }.items() if v is not None
        }
        if not data:
            return json.dumps({"success": False, "error": "No fields provided to update"})

        updated = await _provider().update_contact(contact_id, data)
        if updated is None:
            return json.dumps({"success": False, "error": "Contact not found"})
        return json.dumps({"success": True, "contact": updated}, default=str)
    except Exception as exc:
        logger.exception("update_contact error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: delete_contact
# ---------------------------------------------------------------------------

@mcp.tool()
async def delete_contact(contact_id: str) -> str:
    """
    Archive (soft-delete) a contact.

    The record is marked status=archived rather than permanently removed so
    interaction history and appointment links are preserved.
    """
    try:
        success = await _provider().delete_contact(contact_id)
        return json.dumps({"success": success})
    except Exception as exc:
        logger.exception("delete_contact error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_contacts
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_contacts(
    business_context_id: Optional[str] = None,
    status: str = "active",
    contact_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """
    List contacts with optional filters.

    status:       active (default) | inactive | archived
    contact_type: customer | lead | prospect | vendor
    limit / offset: for pagination
    """
    try:
        contacts = await _provider().list_contacts(
            business_context_id=business_context_id,
            status=status,
            contact_type=contact_type,
            limit=limit,
            offset=offset,
        )
        return json.dumps({"contacts": contacts, "count": len(contacts)}, default=str)
    except Exception as exc:
        logger.exception("list_contacts error")
        return json.dumps({"error": str(exc), "contacts": []})


# ---------------------------------------------------------------------------
# Tool: log_interaction
# ---------------------------------------------------------------------------

@mcp.tool()
async def log_interaction(
    contact_id: str,
    interaction_type: str,
    summary: str,
    occurred_at: Optional[str] = None,
) -> str:
    """
    Log a customer interaction.

    interaction_type: call | email | appointment | sms | note | meeting
    occurred_at: ISO 8601 datetime string (defaults to now)

    Call this after every meaningful customer touch-point to build a
    longitudinal history that enriches GraphRAG and surfaces patterns.
    """
    try:
        interaction = await _provider().log_interaction(
            contact_id=contact_id,
            interaction_type=interaction_type,
            summary=summary,
            occurred_at=occurred_at,
        )
        return json.dumps({"success": True, "interaction": interaction}, default=str)
    except Exception as exc:
        logger.exception("log_interaction error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: get_interactions
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_interactions(contact_id: str, limit: int = 20) -> str:
    """
    Retrieve interaction history for a contact.

    Returns calls, emails, appointments, and notes — most recent first.
    This is the longitudinal view of the customer relationship.
    """
    try:
        interactions = await _provider().get_interactions(contact_id, limit=limit)
        return json.dumps(
            {"interactions": interactions, "count": len(interactions)}, default=str
        )
    except Exception as exc:
        logger.exception("get_interactions error")
        return json.dumps({"error": str(exc), "interactions": []})


# ---------------------------------------------------------------------------
# Tool: get_contact_appointments
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_contact_appointments(contact_id: str) -> str:
    """
    Fetch all appointments linked to a contact.

    Returns appointments that have the contact_id FK set.
    Legacy appointments (booked before the CRM existed) will not appear here
    until they are linked via the contact_id column.
    """
    try:
        appointments = await _provider().get_contact_appointments(contact_id)
        return json.dumps(
            {"appointments": appointments, "count": len(appointments)}, default=str
        )
    except Exception as exc:
        logger.exception("get_contact_appointments error")
        return json.dumps({"error": str(exc), "appointments": []})


# ---------------------------------------------------------------------------
# Tool: get_customer_context
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_customer_context(
    contact_id: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
    max_interactions: int = 10,
    max_calls: int = 10,
    max_appointments: int = 10,
    max_emails: int = 10,
) -> str:
    """
    Get the full unified customer context — everything Atlas knows about a customer.

    Provide at least one of contact_id, phone, email, or name.  The service resolves
    the contact record first, then fetches all linked data in parallel.

    contact_id: UUID of the contact (use search_contacts first if you only have a name)
    name: customer name — will search contacts and use the first match
    phone: phone number in any format
    email: email address
    """
    if not any([contact_id, phone, email, name]):
        return json.dumps(
            {"error": "Provide at least one of: contact_id, phone, email, or name", "found": False}
        )

    try:
        from ..services.customer_context import get_customer_context_service

        svc = get_customer_context_service()
        kwargs = {
            "max_interactions": max_interactions,
            "max_calls": max_calls,
            "max_appointments": max_appointments,
            "max_emails": max_emails,
        }

        # If contact_id doesn't look like a UUID, treat it as a name
        if contact_id and not _is_uuid(contact_id):
            name = contact_id
            contact_id = None

        # Name-based lookup: search contacts first, then get context by ID
        if name and not contact_id:
            results = await _provider().search_contacts(query=name, limit=1)
            if results:
                contact_id = results[0].get("id")
            else:
                return json.dumps({"found": False, "context": None,
                                   "message": f"No contact found matching '{name}'"})

        if contact_id:
            ctx = await svc.get_context(contact_id, **kwargs)
        elif phone:
            ctx = await svc.get_context_by_phone(phone, **kwargs)
        else:
            ctx = await svc.get_context_by_email(email, **kwargs)

        if ctx.is_empty:
            return json.dumps({"found": False, "context": None})

        result: dict = {
            "found": True,
            "contact": ctx.contact,
            "interactions": ctx.interactions,
            "appointments": ctx.appointments,
            "call_transcripts": ctx.call_transcripts,
            "sent_emails": ctx.sent_emails,
            "inbox_emails": ctx.inbox_emails,
        }

        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("get_customer_context error")
        return json.dumps({"error": str(exc), "found": False})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings
        from .auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.crm_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.crm_port)
    else:
        mcp.run(transport="stdio")
