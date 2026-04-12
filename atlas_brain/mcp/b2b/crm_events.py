"""B2B Churn MCP -- crm_events tools."""

import json
from typing import Optional


def _clean_optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

from ._shared import _is_uuid, get_pool, logger
from .server import mcp


@mcp.tool()
async def list_crm_pushes(
    subscription_id: Optional[str] = None,
    vendor_name: Optional[str] = None,
    limit: int = 50,
) -> str:
    """
    List CRM push log entries showing what intelligence data was pushed to external CRMs.

    subscription_id: Optional UUID to filter by specific webhook subscription
    vendor_name: Optional vendor name filter (case-insensitive)
    limit: Max results (default 50, max 200)
    """
    try:
        pool = get_pool()
        conditions = []
        params: list = []
        idx = 1

        if subscription_id:
            if not _is_uuid(subscription_id):
                return json.dumps({"error": "subscription_id must be a valid UUID"})
            conditions.append(f"pl.subscription_id = ${idx}::uuid")
            params.append(subscription_id)
            idx += 1

        if vendor_name:
            conditions.append(f"pl.vendor_name ILIKE ${idx}")
            params.append(f"%{vendor_name}%")
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit = max(1, min(limit, 200))

        rows = await pool.fetch(
            f"""
            SELECT pl.id, pl.subscription_id, pl.signal_type, pl.review_id, pl.vendor_name,
                   pl.company_name, pl.crm_record_id, pl.crm_record_type,
                   pl.status, pl.error, pl.pushed_at,
                   COALESCE(ws.channel, 'generic') AS channel,
                   ws.url AS webhook_url
            FROM b2b_crm_push_log pl
            JOIN b2b_webhook_subscriptions ws ON ws.id = pl.subscription_id
            {where}
            ORDER BY pl.pushed_at DESC
            LIMIT {limit}
            """,
            *params,
        )

        pushes = []
        for r in rows:
            pushes.append({
                "id": str(r["id"]),
                "subscription_id": str(r["subscription_id"]),
                "channel": r["channel"],
                "webhook_url": r["webhook_url"],
                "signal_type": r["signal_type"],
                "review_id": str(r["review_id"]) if r.get("review_id") else None,
                "vendor_name": r["vendor_name"],
                "company_name": r["company_name"],
                "crm_record_id": r["crm_record_id"],
                "crm_record_type": r["crm_record_type"],
                "status": r["status"],
                "error": r["error"],
                "pushed_at": r["pushed_at"].isoformat(),
            })

        return json.dumps({"pushes": pushes, "count": len(pushes)}, default=str)
    except Exception:
        logger.exception("list_crm_pushes error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_crm_events(
    status: Optional[str] = None,
    crm_provider: Optional[str] = None,
    company_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
) -> str:
    """List ingested CRM events with optional filters.

    Args:
        status: Filter by processing status (pending, matched, unmatched, skipped).
        crm_provider: Filter by CRM provider (hubspot, salesforce, pipedrive, generic).
        company_name: Filter by company name (partial, case-insensitive).
        start_date: Filter events received on or after (ISO 8601, e.g. 2026-01-01).
        end_date: Filter events received before (ISO 8601).
        limit: Max events to return (default 50, max 200).
    """
    try:
        from datetime import datetime as _dt

        status = (_clean_optional_text(status) or "").lower() or None
        crm_provider = (_clean_optional_text(crm_provider) or "").lower() or None
        company_name = _clean_optional_text(company_name)
        start_date = _clean_optional_text(start_date)
        end_date = _clean_optional_text(end_date)

        _valid_statuses = {"pending", "matched", "unmatched", "skipped"}
        _valid_providers = {"hubspot", "salesforce", "pipedrive", "generic"}
        if status and status not in _valid_statuses:
            return json.dumps({"error": f"Invalid status. Must be one of: {sorted(_valid_statuses)}"})
        if crm_provider and crm_provider not in _valid_providers:
            return json.dumps({"error": f"Invalid crm_provider. Must be one of: {sorted(_valid_providers)}"})

        sd = None
        if start_date:
            try:
                sd = _dt.fromisoformat(start_date.replace("Z", "+00:00"))
            except ValueError:
                return json.dumps({"error": "Invalid start_date (ISO 8601 expected)"})

        ed = None
        if end_date:
            try:
                ed = _dt.fromisoformat(end_date.replace("Z", "+00:00"))
            except ValueError:
                return json.dumps({"error": "Invalid end_date (ISO 8601 expected)"})

        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        limit = max(1, min(limit, 200))
        conditions = ["1=1"]
        params: list = []
        idx = 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        if crm_provider:
            conditions.append(f"crm_provider = ${idx}")
            params.append(crm_provider)
            idx += 1
        if company_name:
            conditions.append(f"LOWER(company_name) LIKE '%' || LOWER(${idx}) || '%'")
            params.append(company_name)
            idx += 1
        if sd is not None:
            conditions.append(f"received_at >= ${idx}")
            params.append(sd)
            idx += 1
        if ed is not None:
            conditions.append(f"received_at < ${idx}")
            params.append(ed)
            idx += 1

        where = " AND ".join(conditions)
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, crm_provider, event_type, company_name, contact_email,
                   deal_stage, deal_amount, status, matched_sequence_id,
                   outcome_recorded, processing_notes,
                   event_timestamp, received_at, processed_at
            FROM b2b_crm_events
            WHERE {where}
            ORDER BY received_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        events = []
        for r in rows:
            events.append({
                "id": str(r["id"]),
                "crm_provider": r["crm_provider"],
                "event_type": r["event_type"],
                "company_name": r["company_name"],
                "contact_email": r["contact_email"],
                "deal_stage": r["deal_stage"],
                "deal_amount": float(r["deal_amount"]) if r["deal_amount"] else None,
                "status": r["status"],
                "matched_sequence_id": str(r["matched_sequence_id"]) if r["matched_sequence_id"] else None,
                "outcome_recorded": r["outcome_recorded"],
                "received_at": str(r["received_at"]),
            })

        return json.dumps({"events": events, "count": len(events)}, default=str)
    except Exception:
        logger.exception("list_crm_events error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def ingest_crm_event(
    crm_provider: str,
    event_type: str,
    company_name: Optional[str] = None,
    contact_email: Optional[str] = None,
    deal_stage: Optional[str] = None,
    deal_amount: Optional[float] = None,
    deal_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Manually ingest a CRM event for processing.

    The event will be stored and processed by the crm_event_processing task
    which matches it to campaign sequences and auto-records outcomes.

    Args:
        crm_provider: CRM source (hubspot, salesforce, pipedrive, generic).
        event_type: Event type (deal_won, deal_lost, meeting_booked, deal_stage_change, etc.).
        company_name: Company name for matching to campaign sequences.
        contact_email: Contact email for matching to campaign sequences.
        deal_stage: Current deal stage in the CRM.
        deal_amount: Deal value if available.
        deal_id: CRM deal/opportunity ID.
        notes: Optional notes about this event.
    """
    valid_providers = {"hubspot", "salesforce", "pipedrive", "generic"}
    valid_types = {
        "deal_stage_change", "deal_won", "deal_lost",
        "meeting_booked", "activity_logged", "contact_updated",
    }
    crm_provider = (_clean_optional_text(crm_provider) or "").lower()
    event_type = (_clean_optional_text(event_type) or "").lower()
    company_name = _clean_optional_text(company_name)
    contact_email = _clean_optional_text(contact_email)
    deal_stage = _clean_optional_text(deal_stage)
    deal_id = _clean_optional_text(deal_id)
    notes = _clean_optional_text(notes)

    if crm_provider not in valid_providers:
        return json.dumps({"error": f"crm_provider must be one of {sorted(valid_providers)}"})
    if event_type not in valid_types:
        return json.dumps({"error": f"event_type must be one of {sorted(valid_types)}"})
    if not company_name and not contact_email:
        return json.dumps({"error": "At least one of company_name or contact_email is required"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        event_data = {}
        if notes:
            event_data["notes"] = notes

        event_id = await pool.fetchval(
            """
            INSERT INTO b2b_crm_events (
                crm_provider, event_type, company_name, contact_email,
                deal_id, deal_stage, deal_amount, event_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING id
            """,
            crm_provider, event_type, company_name, contact_email,
            deal_id, deal_stage, deal_amount,
            json.dumps(event_data, default=str),
        )

        return json.dumps({
            "success": True,
            "event_id": str(event_id),
            "status": "pending",
            "message": "Event ingested. Will be processed by crm_event_processing task.",
        })
    except Exception:
        logger.exception("ingest_crm_event error")
        return json.dumps({"error": "Failed to ingest CRM event"})


@mcp.tool()
async def get_crm_enrichment_stats() -> str:
    """
    Show enrichment coverage and effectiveness stats for CRM events.

    Returns total events, match rates, field coverage (company_name, contact_email),
    and enrichment counts (cross-event lookups, vendor normalization).
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_events,
                COUNT(*) FILTER (WHERE status = 'matched') AS matched,
                COUNT(*) FILTER (WHERE status = 'unmatched') AS unmatched,
                COUNT(*) FILTER (WHERE status = 'skipped') AS skipped,
                COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                COUNT(*) FILTER (WHERE status = 'error') AS errored,
                COUNT(*) FILTER (WHERE company_name IS NOT NULL) AS has_company,
                COUNT(*) FILTER (WHERE contact_email IS NOT NULL) AS has_email,
                COUNT(*) FILTER (WHERE company_name IS NULL AND contact_email IS NULL) AS missing_both,
                COUNT(*) FILTER (WHERE processing_notes LIKE '%[enriched]%') AS enriched_count,
                COUNT(*) FILTER (WHERE processing_notes LIKE '%[enriched]%' AND status = 'matched') AS enriched_matched
            FROM b2b_crm_events
            """
        )

        total = row["total_events"] or 0
        return json.dumps({
            "total_events": total,
            "matched": row["matched"],
            "unmatched": row["unmatched"],
            "skipped": row["skipped"],
            "pending": row["pending"],
            "errored": row["errored"],
            "field_coverage": {
                "has_company_name": row["has_company"],
                "has_contact_email": row["has_email"],
                "missing_both": row["missing_both"],
            },
            "enrichment": {
                "events_enriched": row["enriched_count"],
                "enriched_then_matched": row["enriched_matched"],
            },
            "match_rate": round(row["matched"] / total * 100, 1) if total else 0,
        })
    except Exception:
        logger.exception("get_crm_enrichment_stats error")
        return json.dumps({"error": "Internal error"})
