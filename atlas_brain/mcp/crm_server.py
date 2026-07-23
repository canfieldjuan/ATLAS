"""
Atlas CRM MCP Server.

Provider-agnostic MCP server that exposes the Atlas CRM (Directus / direct-DB)
to any MCP-compatible client (Claude Desktop, Cursor, custom agents, etc.).

The `contacts` table is the single source of truth for customer data.
Contacts are enriched over time via interaction logs, linked appointments,
and email history --replacing the previous approach of scraping appointment
rows and relying solely on GraphRAG accumulation.

Tools:
    search_contacts         --find contacts by name / phone / email
    get_contact             --fetch a contact by UUID
    create_contact          --create a new contact record
    update_contact          --update contact fields
    delete_contact          --archive (soft-delete) a contact
    list_contacts           --paginated list with filters
    log_interaction         --record a customer touch-point
    get_interactions        --retrieve interaction history
    get_contact_appointments --fetch appointments linked to a contact
    get_customer_context     --unified view: contact + interactions + calls + emails

Run:
    python -m atlas_brain.mcp.crm_server          # stdio (Claude Desktop / Cursor)
    python -m atlas_brain.mcp.crm_server --sse    # SSE HTTP transport
"""

import json
import logging
import sys
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

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


_provider_override: "Callable[[], Any] | None" = None


def set_provider_override(factory: "Callable[[], Any] | None") -> None:
    """Install (or clear with ``None``) a CRM provider factory override.

    Injection seam for unit tests, mirroring the FastAPI
    ``dependency_overrides`` pattern used by the HTTP surface: tests supply a
    fake provider through this public setter rather than patching module
    internals.
    """
    global _provider_override
    _provider_override = factory


def _provider():
    if _provider_override is not None:
        return _provider_override()
    from ..services.crm_provider import get_crm_provider

    return get_crm_provider()


def _default_context() -> "str | None":
    """Deployment-default tenant for read scoping (issue #2151).

    When configured (ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT), read tools that
    receive no explicit business_context_id operate on that tenant's rows
    plus NULL-context legacy rows (the claimable population established in
    PR #2153). An explicit argument always wins; unset preserves the legacy
    unscoped behavior.
    """
    from ..config import settings

    return settings.mcp.crm_default_business_context or None


def _visible_under_default(contact: "dict | None") -> bool:
    """Tenant guard for id-addressed tools: with a default configured, a
    contact belonging to a DIFFERENT tenant is reported as not found
    (fail-closed, no cross-tenant existence leak)."""
    if contact is None:
        return False
    default = _default_context()
    if not default:
        return True
    return contact.get("business_context_id") in (None, default)


async def _guarded_contact(
    contact_id: str, business_context_id: "str | None" = None
) -> "tuple[bool, dict | None]":
    """Resolve + tenant-check an id-addressed operation.

    Returns ``(allowed, contact)`` so callers that need the resolved row
    (e.g. claim-on-update) avoid a second lookup. An explicit
    ``business_context_id`` addresses exactly that tenant's page (mirroring
    explicit search: no NULL-context fallback). Otherwise the deployment
    default applies: the default tenant plus NULL-context legacy rows are
    visible. Foreign-tenant rows read as nonexistent (fail-closed).
    """
    if business_context_id:
        contact = await _provider().get_contact(contact_id)
        if contact is None:
            return False, None
        return contact.get("business_context_id") == business_context_id, contact
    if not _default_context():
        return True, None
    contact = await _provider().get_contact(contact_id)
    return _visible_under_default(contact), contact


async def _guard_contact_id(
    contact_id: str, business_context_id: "str | None" = None
) -> bool:
    """Boolean form of :func:`_guarded_contact` for tools that only gate."""
    allowed, _ = await _guarded_contact(contact_id, business_context_id)
    return allowed


async def _scoped_search(provider, **kwargs):
    """search_contacts honoring the default scope: tenant page first, then
    the NULL-context legacy page (mirrors crm_provider's stamped dedupe)."""
    explicit = kwargs.pop("business_context_id", None)
    if explicit:
        return await provider.search_contacts(business_context_id=explicit, **kwargs)
    default = _default_context()
    if not default:
        return await provider.search_contacts(**kwargs)
    results = await provider.search_contacts(business_context_id=default, **kwargs)
    if results:
        return results
    return await provider.search_contacts(business_context_id_is_null=True, **kwargs)


def _appointments_in_scope(
    appointments: "list[dict]", business_context_id: "str | None"
) -> "list[dict]":
    """Filter appointment rows to the effective tenant scope.

    ``appointments.business_context_id`` is NOT NULL by schema, so under an
    effective scope (explicit argument or deployment default) only rows
    stamped with that tenant qualify; with no scope every row passes (legacy
    unscoped behavior).
    """
    effective = business_context_id or _default_context()
    if not effective:
        return appointments
    return [a for a in appointments if a.get("business_context_id") == effective]


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
    if not any([query, phone, email, business_context_id, _default_context()]):
        return json.dumps({"error": "At least one of query, phone, email, or business_context_id is required",
                           "found": False, "contacts": [], "count": 0})
    try:
        results = await _scoped_search(
            _provider(),
            query=query,
            phone=phone,
            email=email,
            business_context_id=business_context_id,
            limit=min(limit, 100),
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
        appointments = _appointments_in_scope(appointments, business_context_id)

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
            {"error": "Internal error", "found": False, "contacts": [],
             "count": 0}
        )


# ---------------------------------------------------------------------------
# Tool: get_contact
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_contact(contact_id: str, business_context_id: Optional[str] = None) -> str:
    """
    Fetch a contact by UUID or name.

    If contact_id is not a valid UUID, treats it as a name search
    and returns the first matching contact.
    business_context_id: address a specific tenant's page explicitly;
    omitted, the deployment default (plus NULL-context legacy rows) applies.
    """
    try:
        # If it doesn't look like a UUID, search by name instead
        if not _is_uuid(contact_id):
            results = await _scoped_search(
                _provider(), query=contact_id, limit=1,
                business_context_id=business_context_id,
            )
            if results:
                return json.dumps({"found": True, "contact": results[0]}, default=str)
            return json.dumps({"found": False, "contact": None})

        if not await _guard_contact_id(contact_id, business_context_id):
            return json.dumps({"found": False, "contact": None})
        contact = await _provider().get_contact(contact_id)
        if contact is None:
            return json.dumps({"found": False, "contact": None})
        return json.dumps({"found": True, "contact": contact}, default=str)
    except Exception as exc:
        logger.exception("get_contact error")
        return json.dumps({"error": "Internal error", "found": False, "contact": None})


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
            "business_context_id": business_context_id or _default_context(),
            "contact_type": contact_type,
            "notes": notes,
            "source": source,
            "tags": tags or [],
        }
        contact = await _provider().create_contact(data)
        return json.dumps({"success": True, "contact": contact}, default=str)
    except Exception as exc:
        logger.exception("create_contact error")
        return json.dumps({"success": False, "error": "Internal error"})


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
    business_context_id: Optional[str] = None,
) -> str:
    """
    Update a contact's information.

    Only supply fields you want to change.
    status: active | inactive | archived
    business_context_id: addresses which tenant page the contact is looked
    up on (explicit override of the deployment default); it is NOT an
    updatable field.
    """
    if not _is_uuid(contact_id):
        return json.dumps({"success": False, "error": "Invalid contact_id (must be UUID)"})

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

        allowed, existing = await _guarded_contact(contact_id, business_context_id)
        if not allowed:
            return json.dumps({"success": False, "error": "Contact not found"})
        default = _default_context()
        if (
            default
            and not business_context_id
            and existing is not None
            and existing.get("business_context_id") is None
        ):
            # Claim-on-write: an update from a scoped session takes ownership
            # of the NULL-context legacy row, so corrected data stops being
            # visible to every tenant as unclaimed legacy.
            data["business_context_id"] = default
        updated = await _provider().update_contact(contact_id, data)
        if updated is None:
            return json.dumps({"success": False, "error": "Contact not found"})
        return json.dumps({"success": True, "contact": updated}, default=str)
    except Exception as exc:
        logger.exception("update_contact error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: delete_contact
# ---------------------------------------------------------------------------

@mcp.tool()
async def delete_contact(contact_id: str, business_context_id: Optional[str] = None) -> str:
    """
    Archive (soft-delete) a contact.

    The record is marked status=archived rather than permanently removed so
    interaction history and appointment links are preserved.
    business_context_id: explicit tenant-page override of the deployment
    default for the lookup.
    """
    if not _is_uuid(contact_id):
        return json.dumps({"success": False, "error": "Invalid contact_id (must be UUID)"})

    try:
        if not await _guard_contact_id(contact_id, business_context_id):
            return json.dumps({"success": False, "error": "Contact not found"})
        success = await _provider().delete_contact(contact_id)
        return json.dumps({"success": success})
    except Exception as exc:
        logger.exception("delete_contact error")
        return json.dumps({"success": False, "error": "Internal error"})


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

    With a deployment default tenant configured and no explicit
    business_context_id, results merge the default tenant's page with the
    NULL-context legacy page (offset applies to each page; the merged
    result is truncated to limit).
    """
    try:
        provider = _provider()
        default = _default_context()
        capped = min(limit, 200)
        if business_context_id or not default:
            contacts = await provider.list_contacts(
                business_context_id=business_context_id,
                status=status,
                contact_type=contact_type,
                limit=capped,
                offset=offset,
            )
        else:
            tenant_rows = await provider.list_contacts(
                business_context_id=default,
                status=status,
                contact_type=contact_type,
                limit=capped,
                offset=offset,
            )
            legacy_rows = await provider.list_contacts(
                business_context_id_is_null=True,
                status=status,
                contact_type=contact_type,
                limit=capped,
                offset=offset,
            )
            contacts = (tenant_rows + legacy_rows)[:capped]
        return json.dumps({"contacts": contacts, "count": len(contacts)}, default=str)
    except Exception as exc:
        logger.exception("list_contacts error")
        return json.dumps({"error": "Internal error", "contacts": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: log_interaction
# ---------------------------------------------------------------------------

@mcp.tool()
async def log_interaction(
    contact_id: str,
    interaction_type: str,
    summary: str,
    occurred_at: Optional[str] = None,
    business_context_id: Optional[str] = None,
) -> str:
    """
    Log a customer interaction.

    interaction_type: call | email | appointment | sms | note | meeting
    occurred_at: ISO 8601 datetime string (defaults to now)

    Call this after every meaningful customer touch-point to build a
    longitudinal history that enriches GraphRAG and surfaces patterns.
    """
    if not _is_uuid(contact_id):
        return json.dumps({"success": False, "error": "Invalid contact_id (must be UUID)"})
    if not summary or not summary.strip():
        return json.dumps({"success": False, "error": "summary is required"})
    if not interaction_type or not interaction_type.strip():
        return json.dumps({"success": False, "error": "interaction_type is required"})

    try:
        if not await _guard_contact_id(contact_id, business_context_id):
            return json.dumps({"success": False, "error": "Contact not found"})
        interaction = await _provider().log_interaction(
            contact_id=contact_id,
            interaction_type=interaction_type,
            summary=summary,
            occurred_at=occurred_at,
        )
        return json.dumps({"success": True, "interaction": interaction}, default=str)
    except Exception as exc:
        logger.exception("log_interaction error")
        return json.dumps({"success": False, "error": "Internal error"})


# ---------------------------------------------------------------------------
# Tool: get_interactions
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_interactions(
    contact_id: str, limit: int = 20, business_context_id: Optional[str] = None
) -> str:
    """
    Retrieve interaction history for a contact.

    Returns calls, emails, appointments, and notes --most recent first.
    This is the longitudinal view of the customer relationship.
    """
    if not _is_uuid(contact_id):
        return json.dumps({"error": "Invalid contact_id (must be UUID)", "interactions": [], "count": 0})

    try:
        if not await _guard_contact_id(contact_id, business_context_id):
            return json.dumps({"error": "Contact not found", "interactions": [], "count": 0})
        interactions = await _provider().get_interactions(contact_id, limit=min(limit, 100))
        return json.dumps(
            {"interactions": interactions, "count": len(interactions)}, default=str
        )
    except Exception as exc:
        logger.exception("get_interactions error")
        return json.dumps({"error": "Internal error", "interactions": [], "count": 0})


# ---------------------------------------------------------------------------
# Tool: get_contact_appointments
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_contact_appointments(
    contact_id: str, business_context_id: Optional[str] = None
) -> str:
    """
    Fetch all appointments linked to a contact.

    Returns appointments that have the contact_id FK set.
    Legacy appointments (booked before the CRM existed) will not appear here
    until they are linked via the contact_id column.
    """
    if not _is_uuid(contact_id):
        return json.dumps({"error": "Invalid contact_id (must be UUID)", "appointments": [], "count": 0})

    try:
        if not await _guard_contact_id(contact_id, business_context_id):
            return json.dumps({"error": "Contact not found", "appointments": [], "count": 0})
        appointments = await _provider().get_contact_appointments(contact_id)
        appointments = _appointments_in_scope(appointments, business_context_id)
        return json.dumps(
            {"appointments": appointments, "count": len(appointments)}, default=str
        )
    except Exception as exc:
        logger.exception("get_contact_appointments error")
        return json.dumps({"error": "Internal error", "appointments": [], "count": 0})


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
    business_context_id: Optional[str] = None,
) -> str:
    """
    Get the full unified customer context --everything Atlas knows about a customer.

    Provide at least one of contact_id, phone, email, or name.  The service resolves
    the contact record first, then fetches all linked data in parallel.

    contact_id: UUID of the contact (use search_contacts first if you only have a name)
    name: customer name --will search contacts and use the first match
    phone: phone number in any format
    email: email address
    business_context_id: explicit tenant-page override of the deployment
    default; resolution and the tenant guard follow the same scoping rules
    as search_contacts / get_contact.
    """
    if not any([contact_id, phone, email, name]):
        return json.dumps(
            {"error": "Provide at least one of: contact_id, phone, email, or name", "found": False}
        )

    try:
        from ..services.customer_context import get_customer_context_service

        kwargs = {
            "max_interactions": min(max_interactions, 50),
            "max_calls": min(max_calls, 50),
            "max_appointments": min(max_appointments, 50),
            "max_emails": min(max_emails, 50),
        }

        # If contact_id doesn't look like a UUID, treat it as a name
        if contact_id and not _is_uuid(contact_id):
            name = contact_id
            contact_id = None

        resolved_in_scope = False

        # Name-based lookup: search contacts first, then get context by ID
        if name and not contact_id:
            results = await _scoped_search(
                _provider(), query=name, limit=1,
                business_context_id=business_context_id,
            )
            if results:
                contact_id = results[0].get("id")
                resolved_in_scope = True
            else:
                return json.dumps({"found": False, "context": None,
                                   "message": f"No contact found matching '{name}'"})

        # Under a tenant scope, phone/email must resolve through the scoped
        # contact search as well; the context service's own lookups are
        # unscoped and would return foreign-tenant rows.
        if (
            not contact_id
            and (phone or email)
            and (business_context_id or _default_context())
        ):
            results = await _scoped_search(
                _provider(), phone=phone, email=email, limit=1,
                business_context_id=business_context_id,
            )
            if not results:
                return json.dumps({"found": False, "context": None})
            contact_id = results[0].get("id")
            resolved_in_scope = True

        if contact_id and not resolved_in_scope:
            if not await _guard_contact_id(contact_id, business_context_id):
                return json.dumps({"found": False, "context": None})

        svc = get_customer_context_service()
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
            "b2b_churn_signals": ctx.b2b_churn_signals,
        }

        return json.dumps(result, default=str)
    except Exception as exc:
        logger.exception("get_customer_context error")
        return json.dumps({"error": "Internal error", "found": False})


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
