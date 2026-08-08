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
    open_customer_service_ticket --open a tenant-scoped complaint ticket
    list_customer_service_tickets --list the tenant's complaint queue
    update_customer_service_ticket --update an open complaint ticket
    close_customer_service_ticket --close a complaint with a resolution
    log_interaction         --record a customer touch-point
    get_interactions        --retrieve interaction history
    update_contact_appointment_operations --set recurrence, cleaner, and price
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
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.crm")

# The one migration this server needs applied to serve scoped Gmail bindings.
SCOPED_MAILBOX_MIGRATIONS = ("350_scoped_mailbox_credentials",)


@asynccontextmanager
async def _database_lifespan(
    *,
    init_database_fn,
    get_db_pool_fn,
    run_migrations_fn,
    close_database_fn,
):
    """Initialize the DB pool and apply this server's prerequisite migration.

    Parameterized like the invoicing MCP's lifespan so tests drive it with
    edge fakes instead of monkeypatching storage internals.

    Only ``350_scoped_mailbox_credentials`` is applied, not the whole chain.
    Two reasons, and the first is a hard blocker:

    * **The full chain is not fresh-applicable.** Migrations from 076 on
      reference an out-of-band ``product_metadata`` table that no migration
      creates, so a fresh database dies partway through. The main app survives
      that because it logs and continues (``main.py``); a standalone server
      that ran the chain unguarded would abort its entire lifespan on an
      unrelated pending migration. Migration 350 is self-contained and
      idempotent, so applying exactly it is safe on any database state.
    * This server is a documented standalone deployment and scoped Gmail
      bindings depend on that one table.

    A failure here degrades the Gmail binding to its documented fail-closed
    state (the row is absent, so scoped inbox reads are omitted). It must not
    take contacts, tickets and appointments down with it.
    """
    try:
        await init_database_fn()
        pool = get_db_pool_fn()
        if pool.is_initialized:
            try:
                await run_migrations_fn(pool, only=SCOPED_MAILBOX_MIGRATIONS)
                logger.info("CRM MCP: DB pool initialized and migrated")
            except Exception as exc:
                logger.warning(
                    "CRM MCP: prerequisite migration failed (%s); scoped Gmail "
                    "bindings will be unavailable until it is applied. Other "
                    "CRM tools are unaffected.",
                    type(exc).__name__,
                )
        else:
            logger.warning(
                "CRM MCP: DB pool not initialized (persistence disabled?); "
                "migrations skipped -- scoped Gmail bindings will be unavailable"
            )
        yield
    finally:
        await close_database_fn()


@asynccontextmanager
async def _lifespan(server):
    from ..storage.database import init_database, close_database, get_db_pool
    from ..storage.migrations import run_migrations

    async with _database_lifespan(
        init_database_fn=init_database,
        get_db_pool_fn=get_db_pool,
        run_migrations_fn=run_migrations,
        close_database_fn=close_database,
    ):
        yield


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


def _pipeline_text(value: str, field: str, max_length: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty; use its clear flag")
    if len(normalized) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    return normalized


def _pipeline_timestamp(value: str, field: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty; use clear_next_follow_up")
    if len(normalized) > 64:
        raise ValueError(f"{field} must be at most 64 characters")
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _ticket_text(value: str, field: str, max_length: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    if len(normalized) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    return normalized


def _ticket_optional_text(
    value: "str | None", field: str, max_length: int
) -> "str | None":
    if value is None:
        return None
    return _ticket_text(value, field, max_length)


def _appointment_price(value: str) -> Decimal:
    """Parse an exact appointment price without binary-float rounding."""
    try:
        normalized = value.strip()
        if not normalized or len(normalized) > 64:
            raise InvalidOperation
        parsed = Decimal(normalized)
    except (AttributeError, InvalidOperation) as exc:
        raise ValueError("per_visit_price must be numeric") from exc
    if not parsed.is_finite():
        raise ValueError("per_visit_price must be finite")
    if parsed < 0:
        raise ValueError("per_visit_price must be non-negative")
    if parsed > Decimal("9999999999.99"):
        raise ValueError("per_visit_price exceeds the database limit")
    try:
        cents = parsed.quantize(Decimal("0.01"))
    except InvalidOperation as exc:
        raise ValueError(
            "per_visit_price must have at most 2 decimal places"
        ) from exc
    if cents != parsed:
        raise ValueError("per_visit_price must have at most 2 decimal places")
    return cents


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


def _row_visible(
    contact: "dict | None", business_context_id: "str | None"
) -> bool:
    """Visibility of a FETCHED contact row under the effective addressing.

    Explicit tenant = exact page; default = tenant plus NULL-context legacy;
    no scope = everything. Callers must apply this to the same row object
    they return or act on -- checking one fetch and returning another
    reopens the claim race (#2157 post-merge review).
    """
    if contact is None:
        return False
    if business_context_id:
        return contact.get("business_context_id") == business_context_id
    return _visible_under_default(contact)


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
    if not business_context_id and not _default_context():
        return True, None
    contact = await _provider().get_contact(contact_id)
    return _row_visible(contact, business_context_id), contact


async def _guard_contact_id(
    contact_id: str, business_context_id: "str | None" = None
) -> bool:
    """Boolean form of :func:`_guarded_contact` for tools that only gate."""
    allowed, _ = await _guarded_contact(contact_id, business_context_id)
    return allowed


async def _scoped_search(provider, **kwargs):
    """search_contacts honoring the default scope.

    The visible population is the default tenant's page PLUS the
    NULL-context legacy page: both are queried and merged (tenant rows
    first, truncated to the caller's limit). A tenant hit must not hide
    claimable legacy rows (#2157 post-merge review) -- the earlier
    first-page-wins shape did exactly that. An explicit argument addresses
    exactly one page; no default preserves legacy unscoped behavior.
    """
    explicit = kwargs.pop("business_context_id", None)
    if explicit:
        return await provider.search_contacts(business_context_id=explicit, **kwargs)
    default = _default_context()
    if not default:
        return await provider.search_contacts(**kwargs)
    tenant_rows = await provider.search_contacts(business_context_id=default, **kwargs)
    legacy_rows = await provider.search_contacts(business_context_id_is_null=True, **kwargs)
    merged = list(tenant_rows) + list(legacy_rows)
    limit = kwargs.get("limit")
    return merged[:limit] if limit else merged


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


async def _claim_if_legacy(
    contact_id: str, existing: "dict | None", business_context_id: "str | None"
) -> bool:
    """Claim-on-write of a NULL-context legacy row under the default.

    Compare-and-set in SQL (``claim_contact``): returns False when a
    concurrent claim moved the row to another tenant between the guard read
    and this write, so callers fail closed instead of overwriting the other
    tenant's claim.
    """
    default = _default_context()
    if not default or business_context_id or existing is None:
        return True
    if existing.get("business_context_id") is not None:
        return True
    claimed = await _provider().claim_contact(contact_id, default)
    return claimed is not None


def _calls_in_scope(
    rows: "list[dict]", business_context_id: "str | None"
) -> "list[dict]":
    """Call-transcript rows may carry a NULL business_context_id (legacy),
    so under an effective scope the visible set is the tenant's rows plus
    NULL-context ones -- the same claimable population as contacts."""
    effective = business_context_id or _default_context()
    if not effective:
        return rows
    return [r for r in rows if r.get("business_context_id") in (None, effective)]


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

        effective_scope = business_context_id or _default_context()
        if phone:
            appointments = await repo.get_by_phone(
                phone, status=None, upcoming_only=False, limit=limit,
                business_context_id=effective_scope,
            )
        if not appointments and query:
            appointments = await repo.search_by_name(
                query, include_history=True, limit=limit,
                business_context_id=effective_scope,
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

        # Single fetch: the row that is validated IS the row that is
        # returned (a guard-then-refetch pair loses to a concurrent claim).
        contact = await _provider().get_contact(contact_id)
        if not _row_visible(contact, business_context_id):
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
        effective_business_context_id = business_context_id or _default_context()
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
            "business_context_id": effective_business_context_id,
            "contact_type": contact_type,
            "notes": notes,
            "source": source,
            "tags": tags or [],
        }
        # Tenant is required (website #124, D1). An agent-callable create must
        # not mint a NULL-context contact under weaker rules than the CRM UI,
        # which always writes a tenant. `_default_context()` returns None when
        # no deployment default is configured (the live runtime sets none), so
        # without this a create with no explicit business_context_id silently
        # produces an unclassified row. A missing tenant is a typed refusal, not
        # a silent default. This runs BEFORE the EOM guard below because a NULL
        # tenant is never the EOM tenant and would otherwise slip past it.
        #
        # Admission closure (the axis this guard owns): tenant PRESENCE.
        #   reject  iff  str(x or "").strip() == ""      -> "required" refusal
        #   reject  iff  the resolved id == EOM tenant    -> EOM-ingress refusal
        #   admit   otherwise (reaches the provider)
        # This guard owns tenant PRESENCE. Tenant EXISTENCE (is it a REAL tenant?)
        # is enforced separately below and, durably, by the FK
        # contacts.business_context_id -> business_contexts.id added in migration
        # 365 (#2318). D1 could not validate existence because business_contexts
        # was empty and there was no FK -- validating then would have rejected
        # every real tenant. The existence net below is fail-safe: it enforces only
        # when the registry is populated, otherwise it degrades to this
        # presence-only behavior. The presence closure's property proof is
        # test_create_contact_tenant_admission_closure.
        if not str(effective_business_context_id or "").strip():
            return json.dumps({
                "success": False,
                "error": (
                    "business_context_id is required to create a contact; no "
                    "deployment default is configured. Specify the tenant "
                    "explicitly."
                ),
            })

        from ..services.eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

        if str(effective_business_context_id or "").strip() == EOM_BUSINESS_CONTEXT_ID:
            return json.dumps({
                "success": False,
                "error": (
                    "New EOM contacts must be created through the EOM ingress "
                    "or funnel transition service"
                ),
            })

        # Tenant EXISTENCE net (#2318). By here the tenant is non-blank (presence
        # guard) and not the EOM tenant (EOM guard) -- a concrete non-EOM tenant
        # that must be a REAL one. Migration 365 makes `business_contexts` the
        # enforced registry (seeds it + adds the FK), so the FK is the durable
        # enforcement; this is the clean typed refusal that fires before the INSERT.
        resolved_tenant = str(effective_business_context_id or "").strip()
        from ..storage.repositories.business_context import BusinessContextRepository
        from ..storage.exceptions import DatabaseUnavailableError

        try:
            # (enforced, known): `enforced` is gated on migration 365 having run
            # (the FK exists) -- NOT on business_contexts being non-empty, since that
            # voice-config table can hold an unrelated row before 365 seeds the real
            # tenants. `known` is a COMPLETE-registry membership check (not
            # list_enabled(), which filters on `enabled` and LIMIT 100).
            registry_enforced, tenant_known = (
                await BusinessContextRepository().admission_check(resolved_tenant)
            )
        except DatabaseUnavailableError:
            # Expected pre-seed / persistence-disabled state -> fail-safe admit.
            registry_enforced, tenant_known = False, False
        except Exception:
            # Unexpected registry failure (permissions, outage, query regression):
            # keep it OBSERVABLE, then admit -- the FK still enforces existence, so a
            # genuinely unknown tenant is rejected at the INSERT regardless.
            logger.warning(
                "create_contact: tenant registry admission check failed; admitting "
                "(the FK still enforces tenant existence)",
                exc_info=True,
            )
            registry_enforced, tenant_known = False, False

        # Fail-safe: enforce ONLY once migration 365 has run (the FK exists).
        # Before then this degrades to the D1 presence-only behavior, so it is safe
        # to deploy in ANY order relative to 365 and never rejects a real tenant
        # before the seed lands -- and the runtime refusal exactly mirrors the FK.
        if registry_enforced and not tenant_known:
            return json.dumps({
                "success": False,
                "error": (
                    f"business_context_id '{resolved_tenant}' is not a known "
                    "tenant; seed it in business_contexts (the tenant registry) "
                    "before creating contacts under it"
                ),
            })

        # Stamp the NORMALIZED tenant so the persisted value matches the registry and
        # the FK: an admitted "  churnsignals  " must persist as "churnsignals", not
        # be rejected by the FK (which compares the raw value) as an opaque error.
        data["business_context_id"] = resolved_tenant

        contact = await _provider().create_contact(data)
        return json.dumps({"success": True, "contact": contact}, default=str)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
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
    lead_stage: Optional[str] = None,
    lead_owner: Optional[str] = None,
    next_follow_up_at: Optional[str] = None,
    clear_lead_owner: bool = False,
    clear_next_follow_up: bool = False,
    business_context_id: Optional[str] = None,
) -> str:
    """
    Update a contact's information.

    Only supply fields you want to change.
    status: active | inactive | archived
    lead_stage / lead_owner: pipeline values for lead contacts only.
    next_follow_up_at: ISO 8601 timestamp with a UTC offset, for leads only.
    clear_lead_owner / clear_next_follow_up: explicitly clear those fields.
    business_context_id: addresses which tenant page the contact is looked
    up on (explicit override of the deployment default); it is NOT an
    updatable field.
    """
    if not _is_uuid(contact_id):
        return json.dumps({"success": False, "error": "Invalid contact_id (must be UUID)"})

    try:
        if clear_lead_owner and lead_owner is not None:
            return json.dumps({
                "success": False,
                "error": "lead_owner and clear_lead_owner are mutually exclusive",
            })
        if clear_next_follow_up and next_follow_up_at is not None:
            return json.dumps({
                "success": False,
                "error": (
                    "next_follow_up_at and clear_next_follow_up are mutually exclusive"
                ),
            })

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
        pipeline_requested = any((
            lead_stage is not None,
            lead_owner is not None,
            next_follow_up_at is not None,
            clear_lead_owner,
            clear_next_follow_up,
        ))
        eom_stage_change_requested = lead_stage is not None
        if lead_stage is not None:
            data["lead_stage"] = _pipeline_text(
                lead_stage, "lead_stage", 64
            )
        if lead_owner is not None:
            data["lead_owner"] = _pipeline_text(
                lead_owner, "lead_owner", 128
            )
        elif clear_lead_owner:
            data["lead_owner"] = None
        if next_follow_up_at is not None:
            data["next_follow_up_at"] = _pipeline_timestamp(
                next_follow_up_at, "next_follow_up_at"
            )
        elif clear_next_follow_up:
            data["next_follow_up_at"] = None
        if not data:
            return json.dumps({"success": False, "error": "No fields provided to update"})

        allowed, existing = await _guarded_contact(contact_id, business_context_id)
        if not allowed:
            return json.dumps({"success": False, "error": "Contact not found"})
        if pipeline_requested and existing is None:
            existing = await _provider().get_contact(contact_id)
            if existing is None:
                return json.dumps({"success": False, "error": "Contact not found"})
        if pipeline_requested and existing.get("contact_type") != "lead":
            return json.dumps({
                "success": False,
                "error": "Lead pipeline fields require a lead contact",
            })
        if eom_stage_change_requested:
            from ..services.eom_lead_ingress import EOM_BUSINESS_CONTEXT_ID

            effective_business_context_id = (
                business_context_id or _default_context()
            )
            if (
                existing.get("business_context_id") == EOM_BUSINESS_CONTEXT_ID
                or effective_business_context_id == EOM_BUSINESS_CONTEXT_ID
            ):
                return json.dumps({
                    "success": False,
                    "error": (
                        "EOM lead stages can only change through the funnel "
                        "transition service"
                    ),
                })
        # Claim-on-write: an update from a scoped session takes ownership of
        # the NULL-context legacy row, so corrected data stops being visible
        # to every tenant as unclaimed legacy.
        if not await _claim_if_legacy(contact_id, existing, business_context_id):
            return json.dumps({"success": False, "error": "Contact not found"})
        if pipeline_requested:
            updated = await _provider().update_contact(
                contact_id,
                data,
                require_contact_type="lead",
            )
        else:
            updated = await _provider().update_contact(contact_id, data)
        if updated is None:
            return json.dumps({"success": False, "error": "Contact not found"})
        return json.dumps({"success": True, "contact": updated}, default=str)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
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
        allowed, existing = await _guarded_contact(contact_id, business_context_id)
        if not allowed:
            return json.dumps({"success": False, "error": "Contact not found"})
        # Archiving is a legacy mutation too: claim the NULL-context row so
        # one tenant cannot soft-delete shared legacy data out of the other
        # tenant's default view.
        if not await _claim_if_legacy(contact_id, existing, business_context_id):
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
    lead_stage: Optional[str] = None,
    lead_owner: Optional[str] = None,
    next_follow_up_before: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """
    List contacts with optional filters.

    status:       active (default) | inactive | archived
    contact_type: customer | lead | prospect | vendor
    lead_stage / lead_owner: exact-match lead pipeline filters.
    next_follow_up_before: include leads due at or before this ISO 8601
        timestamp; a UTC offset is required.
    limit / offset: for pagination

    With a deployment default tenant and no explicit business_context_id,
    pipeline-filtered calls query the tenant-plus-legacy population in one SQL
    statement so due ordering and pagination are global. Other list calls keep
    the established two-page merge behavior.
    """
    try:
        provider = _provider()
        default = _default_context()
        capped = min(limit, 200)
        pipeline_filtered = any((
            lead_stage is not None,
            lead_owner is not None,
            next_follow_up_before is not None,
        ))
        if pipeline_filtered and contact_type not in (None, "lead"):
            return json.dumps({
                "error": "Lead pipeline filters require contact_type='lead'",
                "contacts": [],
                "count": 0,
            })
        effective_contact_type = "lead" if pipeline_filtered else contact_type
        normalized_stage = (
            _pipeline_text(lead_stage, "lead_stage", 64)
            if lead_stage is not None
            else None
        )
        normalized_owner = (
            _pipeline_text(lead_owner, "lead_owner", 128)
            if lead_owner is not None
            else None
        )
        due_before = (
            _pipeline_timestamp(
                next_follow_up_before, "next_follow_up_before"
            )
            if next_follow_up_before is not None
            else None
        )
        filters = {
            "status": status,
            "contact_type": effective_contact_type,
            "lead_stage": normalized_stage,
            "lead_owner": normalized_owner,
            "next_follow_up_before": due_before,
            "limit": capped,
            "offset": offset,
        }
        if business_context_id or not default:
            contacts = await provider.list_contacts(
                business_context_id=business_context_id,
                **filters,
            )
        elif pipeline_filtered:
            contacts = await provider.list_contacts(
                business_context_id=default,
                include_unclaimed_legacy=True,
                **filters,
            )
        else:
            tenant_rows = await provider.list_contacts(
                business_context_id=default,
                **filters,
            )
            legacy_rows = await provider.list_contacts(
                business_context_id_is_null=True,
                **filters,
            )
            contacts = (tenant_rows + legacy_rows)[:capped]
        return json.dumps({"contacts": contacts, "count": len(contacts)}, default=str)
    except ValueError as exc:
        return json.dumps({"error": str(exc), "contacts": [], "count": 0})
    except Exception as exc:
        logger.exception("list_contacts error")
        return json.dumps({"error": "Internal error", "contacts": [], "count": 0})


# ---------------------------------------------------------------------------
# Tools: customer-service complaint tickets
# ---------------------------------------------------------------------------

@mcp.tool()
async def open_customer_service_ticket(
    contact_id: str,
    summary: str,
    details: Optional[str] = None,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
    business_context_id: Optional[str] = None,
) -> str:
    """
    Open a customer-service complaint ticket linked to a CRM contact.

    The ticket is stamped with the explicit business_context_id or the
    deployment default. A visible NULL-context legacy contact is atomically
    claimed for that tenant. Archived, missing, and foreign contacts are
    reported as not found.
    """
    if not _is_uuid(contact_id):
        return json.dumps({
            "success": False,
            "error": "Invalid contact_id (must be UUID)",
        })
    effective = (
        business_context_id
        if business_context_id is not None
        else _default_context()
    )
    if effective is None or not effective.strip():
        return json.dumps({
            "success": False,
            "error": "business_context_id is required",
        })

    try:
        ticket = await _provider().open_customer_service_ticket(
            contact_id=contact_id,
            business_context_id=_ticket_text(
                effective, "business_context_id", 64
            ),
            summary=_ticket_text(summary, "summary", 500),
            details=_ticket_optional_text(details, "details", 10000),
            priority=_ticket_optional_text(priority, "priority", 64),
            assignee=_ticket_optional_text(assignee, "assignee", 128),
        )
        if ticket is None:
            return json.dumps({
                "success": False,
                "error": "Contact not found",
            })
        return json.dumps({"success": True, "ticket": ticket}, default=str)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception:
        logger.exception("open_customer_service_ticket error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_customer_service_tickets(
    status: Optional[str] = "open",
    contact_id: Optional[str] = None,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
    business_context_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """
    List customer-service complaint tickets for one tenant.

    status defaults to open; pass closed for resolved complaints or null for
    both states. Optional contact_id, priority, and assignee filters are applied
    before newest-first pagination.
    """
    effective = (
        business_context_id
        if business_context_id is not None
        else _default_context()
    )
    if effective is None or not effective.strip():
        return json.dumps({
            "error": "business_context_id is required",
            "tickets": [],
            "count": 0,
        })
    if status not in (None, "open", "closed"):
        return json.dumps({
            "error": "status must be open, closed, or null",
            "tickets": [],
            "count": 0,
        })
    if contact_id is not None and not _is_uuid(contact_id):
        return json.dumps({
            "error": "Invalid contact_id (must be UUID)",
            "tickets": [],
            "count": 0,
        })

    try:
        tickets = await _provider().list_customer_service_tickets(
            business_context_id=_ticket_text(
                effective, "business_context_id", 64
            ),
            status=status,
            contact_id=contact_id,
            priority=_ticket_optional_text(priority, "priority", 64),
            assignee=_ticket_optional_text(assignee, "assignee", 128),
            limit=min(max(limit, 1), 200),
            offset=max(offset, 0),
        )
        return json.dumps(
            {"tickets": tickets, "count": len(tickets)},
            default=str,
        )
    except ValueError as exc:
        return json.dumps({
            "error": str(exc),
            "tickets": [],
            "count": 0,
        })
    except Exception:
        logger.exception("list_customer_service_tickets error")
        return json.dumps({
            "error": "Internal error",
            "tickets": [],
            "count": 0,
        })


@mcp.tool()
async def update_customer_service_ticket(
    ticket_id: str,
    summary: Optional[str] = None,
    details: Optional[str] = None,
    priority: Optional[str] = None,
    assignee: Optional[str] = None,
    business_context_id: Optional[str] = None,
) -> str:
    """
    Update mutable fields on an open customer-service complaint ticket.

    Closed and foreign-tenant tickets are reported as not found or not open.
    Closing is a separate operation so resolution is always recorded.
    """
    if not _is_uuid(ticket_id):
        return json.dumps({
            "success": False,
            "error": "Invalid ticket_id (must be UUID)",
        })
    effective = (
        business_context_id
        if business_context_id is not None
        else _default_context()
    )
    if effective is None or not effective.strip():
        return json.dumps({
            "success": False,
            "error": "business_context_id is required",
        })

    try:
        data = {
            key: _ticket_optional_text(value, key, max_length)
            for key, value, max_length in (
                ("summary", summary, 500),
                ("details", details, 10000),
                ("priority", priority, 64),
                ("assignee", assignee, 128),
            )
            if value is not None
        }
        if not data:
            return json.dumps({
                "success": False,
                "error": "No fields provided to update",
            })
        ticket = await _provider().update_customer_service_ticket(
            ticket_id=ticket_id,
            business_context_id=_ticket_text(
                effective, "business_context_id", 64
            ),
            data=data,
        )
        if ticket is None:
            return json.dumps({
                "success": False,
                "error": "Ticket not found or not open",
            })
        return json.dumps({"success": True, "ticket": ticket}, default=str)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception:
        logger.exception("update_customer_service_ticket error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def close_customer_service_ticket(
    ticket_id: str,
    resolution: str,
    business_context_id: Optional[str] = None,
) -> str:
    """
    Close a customer-service complaint ticket with its resolution.

    The first close records the resolution and timestamp. Repeated calls return
    the already-closed ticket without replacing either value.
    """
    if not _is_uuid(ticket_id):
        return json.dumps({
            "success": False,
            "error": "Invalid ticket_id (must be UUID)",
        })
    effective = (
        business_context_id
        if business_context_id is not None
        else _default_context()
    )
    if effective is None or not effective.strip():
        return json.dumps({
            "success": False,
            "error": "business_context_id is required",
        })

    try:
        ticket = await _provider().close_customer_service_ticket(
            ticket_id=ticket_id,
            business_context_id=_ticket_text(
                effective, "business_context_id", 64
            ),
            resolution=_ticket_text(resolution, "resolution", 10000),
        )
        if ticket is None:
            return json.dumps({
                "success": False,
                "error": "Ticket not found",
            })
        already_closed = bool(ticket.pop("already_closed", False))
        return json.dumps({
            "success": True,
            "ticket": ticket,
            "already_closed": already_closed,
        }, default=str)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception:
        logger.exception("close_customer_service_ticket error")
        return json.dumps({"success": False, "error": "Internal error"})


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
        allowed, existing = await _guarded_contact(contact_id, business_context_id)
        if not allowed:
            return json.dumps({"success": False, "error": "Contact not found"})
        # Claim-on-write: logging an interaction from a scoped session takes
        # ownership of the NULL-context legacy row first, so the new note is
        # not readable through another tenant's default.
        if not await _claim_if_legacy(contact_id, existing, business_context_id):
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
        interactions = await _provider().get_interactions(
            contact_id, limit=min(limit, 100),
            business_context_id=business_context_id or _default_context(),
        )
        return json.dumps(
            {"interactions": interactions, "count": len(interactions)}, default=str
        )
    except Exception as exc:
        logger.exception("get_interactions error")
        return json.dumps({"error": "Internal error", "interactions": [], "count": 0})


# ---------------------------------------------------------------------------
# Tools: linked appointment operations
# ---------------------------------------------------------------------------

@mcp.tool()
async def update_contact_appointment_operations(
    contact_id: str,
    appointment_id: str,
    recurrence_interval: Optional[int] = None,
    recurrence_unit: Optional[str] = None,
    assigned_cleaner: Optional[str] = None,
    per_visit_price: Optional[str] = None,
    business_context_id: Optional[str] = None,
) -> str:
    """
    Update operating facts on one appointment linked to a CRM contact.

    Recurrence is expressed as every recurrence_interval recurrence_unit,
    where unit is day, week, or month. Omitted fields remain unchanged.
    Price is a decimal string so the persisted visit price is exact.
    """
    if not _is_uuid(contact_id):
        return json.dumps({
            "success": False,
            "error": "Invalid contact_id (must be UUID)",
        })
    if not _is_uuid(appointment_id):
        return json.dumps({
            "success": False,
            "error": "Invalid appointment_id (must be UUID)",
        })

    effective = (
        business_context_id
        if business_context_id is not None
        else _default_context()
    )
    if effective is None or not effective.strip():
        return json.dumps({
            "success": False,
            "error": "business_context_id is required",
        })

    try:
        if (recurrence_interval is None) != (recurrence_unit is None):
            raise ValueError(
                "recurrence_interval and recurrence_unit must be provided together"
            )

        data: dict[str, Any] = {}
        if recurrence_interval is not None:
            if (
                not isinstance(recurrence_interval, int)
                or isinstance(recurrence_interval, bool)
                or recurrence_interval < 1
                or recurrence_interval > 365
            ):
                raise ValueError(
                    "recurrence_interval must be between 1 and 365"
                )
            normalized_unit = (
                recurrence_unit.strip().lower()
                if isinstance(recurrence_unit, str)
                else ""
            )
            if normalized_unit not in {"day", "week", "month"}:
                raise ValueError(
                    "recurrence_unit must be day, week, or month"
                )
            data["recurrence_interval"] = recurrence_interval
            data["recurrence_unit"] = normalized_unit
        if assigned_cleaner is not None:
            data["assigned_cleaner"] = _ticket_text(
                assigned_cleaner,
                "assigned_cleaner",
                128,
            )
        if per_visit_price is not None:
            data["per_visit_price"] = _appointment_price(per_visit_price)
        if not data:
            raise ValueError("No appointment operating fields provided")

        appointment = (
            await _provider().update_contact_appointment_operations(
                contact_id=contact_id,
                appointment_id=appointment_id,
                business_context_id=_ticket_text(
                    effective,
                    "business_context_id",
                    64,
                ),
                data=data,
            )
        )
        if appointment is None:
            return json.dumps({
                "success": False,
                "error": "Appointment not found",
            })
        return json.dumps({
            "success": True,
            "appointment": appointment,
        }, default=str)
    except ValueError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception:
        logger.exception("update_contact_appointment_operations error")
        return json.dumps({"success": False, "error": "Internal error"})


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
        appointments = await _provider().get_contact_appointments(
            contact_id,
            business_context_id=business_context_id or _default_context(),
        )
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
        from ..services.customer_context import (
            CustomerContextService,
            get_customer_context_service,
        )

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
            if phone:
                results = await _scoped_search(
                    _provider(), phone=phone, limit=1,
                    business_context_id=business_context_id,
                )
            else:
                results = await _scoped_search(
                    _provider(), email=email, limit=1,
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
            ctx = await svc.get_context(
                contact_id,
                business_context_id=business_context_id or _default_context(),
                **kwargs,
            )
        elif phone:
            ctx = await svc.get_context_by_phone(phone, **kwargs)
        else:
            ctx = await svc.get_context_by_email(email, **kwargs)

        if ctx.is_empty:
            return json.dumps({"found": False, "context": None})

        effective = business_context_id or _default_context()
        # Re-fetch ownership after every child-source await. A reassignment
        # during the gather must fail closed before any child data serializes.
        # Pass the explicit argument to _row_visible so deployment-default
        # scope retains tenant-plus-NULL legacy visibility.
        if effective:
            latest_contact = await _provider().get_contact(
                str(ctx.contact["id"])
            )
            if not _row_visible(latest_contact, business_context_id):
                return json.dumps({"found": False, "context": None})
            queried_inbox_address = getattr(
                ctx,
                "inbox_email_query_address",
                None,
            )
            latest_inbox_address = CustomerContextService._normalize_ascii_mailbox(
                latest_contact.get("email")
            )
            if ctx.inbox_emails and (
                not isinstance(queried_inbox_address, str)
                or latest_inbox_address is None
                or not CustomerContextService._same_ascii_mailbox(
                    queried_inbox_address,
                    latest_inbox_address,
                )
            ):
                logger.warning(
                    "CustomerContext inbox results discarded after contact "
                    "email changed during aggregation for %s",
                    ctx.contact["id"],
                )
                ctx.inbox_emails = []
            ctx.contact = latest_contact

        inbox_email_source_omitted = bool(
            effective
            and getattr(ctx, "inbox_email_source_omitted", True)
        )
        result: dict = {
            "found": True,
            "contact": ctx.contact,
            "interactions": ctx.interactions,
            "appointments": _appointments_in_scope(
                ctx.appointments, business_context_id),
            "call_transcripts": _calls_in_scope(
                ctx.call_transcripts, business_context_id),
            "sent_emails": ctx.sent_emails,
            "inbox_emails": (
                [] if inbox_email_source_omitted else ctx.inbox_emails
            ),
            # B2B churn enrichment is keyed by email domain against a global
            # table with no tenant column -- omitted under a scope like the
            # email history (the B2B MCP server is the scoped surface for it).
            "b2b_churn_signals": [] if effective else ctx.b2b_churn_signals,
        }
        if effective:
            omitted_email_sources = (
                ["inbox_emails"] if inbox_email_source_omitted else []
            )
            result["emails_omitted_under_scope"] = bool(omitted_email_sources)
            result["email_sources_omitted_under_scope"] = omitted_email_sources
            result["b2b_enrichment_omitted_under_scope"] = True

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
