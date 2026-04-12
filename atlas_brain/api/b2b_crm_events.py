"""
Inbound CRM event webhook receiver and query endpoints.

Accepts deal/activity events from external CRM systems (HubSpot, Salesforce,
Pipedrive, or generic webhooks) and stores them for asynchronous processing.
The ``crm_event_processing`` autonomous task matches events to campaign
sequences and auto-records outcomes.
"""

import json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, optional_auth
from ..config import settings
from ..services.tracing import (
    build_business_trace_context,
    build_reasoning_trace_context,
    tracer,
)
from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.b2b_crm_events")

router = APIRouter(prefix="/b2b/crm", tags=["b2b-crm-events"])

VALID_CRM_PROVIDERS = {"hubspot", "salesforce", "pipedrive", "generic"}
VALID_EVENT_TYPES = {
    "deal_stage_change", "deal_won", "deal_lost",
    "meeting_booked", "activity_logged", "contact_updated",
}
VALID_EVENT_STATUSES = {"pending", "matched", "unmatched", "skipped"}


class CRMEventBody(BaseModel):
    crm_provider: str = Field(..., min_length=1, max_length=50)
    event_type: str = Field(..., min_length=1, max_length=50)
    crm_event_id: str | None = None
    company_name: str | None = None
    contact_email: str | None = None
    contact_name: str | None = None
    deal_id: str | None = None
    deal_name: str | None = None
    deal_stage: str | None = None
    deal_amount: float | None = None
    event_data: dict | None = None
    event_timestamp: str | None = None


def _pool_or_503():
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(status_code=503, detail="Database not ready")
    return pool


def _clean_optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _start_crm_trace(
    *,
    provider: str,
    event_type: str | None = None,
    account_id: str | None = None,
    batch_size: int | None = None,
) -> object:
    metadata = {
        "business": build_business_trace_context(
            account_id=account_id,
            workflow="crm_event_ingest",
            crm_provider=provider,
            event_type=event_type,
        ),
    }
    if batch_size is not None:
        metadata["business"]["batch_size"] = batch_size
    return tracer.start_span(
        span_name="b2b.crm_event.ingest",
        operation_type="business_operation",
        session_id=account_id,
        metadata=metadata,
    )


def _end_crm_trace(
    span,
    *,
    status: str,
    output_data: dict[str, object] | None = None,
    event_type: str | None = None,
    ingested: int | None = None,
    errors: int | None = None,
    error_message: str | None = None,
    error_type: str | None = None,
) -> None:
    tracer.end_span(
        span,
        status=status,
        output_data=output_data,
        error_message=error_message,
        error_type=error_type,
        metadata={
            "reasoning": build_reasoning_trace_context(
                decision={"event_type": event_type, "status": status},
                evidence={"ingested": ingested, "errors": errors},
            ),
        },
    )


# ---------------------------------------------------------------------------
# POST /events -- ingest a single CRM event
# ---------------------------------------------------------------------------


@router.post("/events")
async def ingest_crm_event(
    body: CRMEventBody,
    user: AuthUser | None = Depends(optional_auth),
):
    """Ingest a CRM event for asynchronous processing.

    Returns the event ID. Processing happens asynchronously via the
    ``crm_event_processing`` autonomous task.
    """
    cfg = settings.crm_event
    if not cfg.enabled:
        raise HTTPException(status_code=503, detail="CRM event ingestion is disabled")

    crm_provider = (_clean_optional_text(body.crm_provider) or "").lower()
    event_type = (_clean_optional_text(body.event_type) or "").lower()
    crm_event_id = _clean_optional_text(body.crm_event_id)
    company_name = _clean_optional_text(body.company_name)
    contact_email = _clean_optional_text(body.contact_email)
    contact_name = _clean_optional_text(body.contact_name)
    deal_id = _clean_optional_text(body.deal_id)
    deal_name = _clean_optional_text(body.deal_name)
    deal_stage = _clean_optional_text(body.deal_stage)
    event_timestamp = _clean_optional_text(body.event_timestamp)

    if crm_provider not in VALID_CRM_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"crm_provider must be one of {sorted(VALID_CRM_PROVIDERS)}",
        )
    if event_type not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"event_type must be one of {sorted(VALID_EVENT_TYPES)}",
        )

    if not company_name and not contact_email:
        raise HTTPException(
            status_code=400,
            detail="At least one of company_name or contact_email is required for matching",
        )

    pool = _pool_or_503()
    span = _start_crm_trace(
        provider=crm_provider,
        event_type=event_type,
        account_id=str(user.account_id) if user else None,
    )

    event_ts = None
    if event_timestamp:
        try:
            event_ts = datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

    account_id = user.account_id if user else None

    try:
        event_id = await pool.fetchval(
            """
            INSERT INTO b2b_crm_events (
                crm_provider, crm_event_id, event_type,
                company_name, contact_email, contact_name,
                deal_id, deal_name, deal_stage, deal_amount,
                event_data, event_timestamp, account_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, $13::uuid)
            ON CONFLICT (crm_provider, crm_event_id)
                WHERE crm_event_id IS NOT NULL
            DO UPDATE SET
                event_type = EXCLUDED.event_type,
                deal_stage = EXCLUDED.deal_stage,
                deal_amount = EXCLUDED.deal_amount,
                event_data = EXCLUDED.event_data,
                event_timestamp = EXCLUDED.event_timestamp,
                status = 'pending',
                processed_at = NULL
            RETURNING id
            """,
            crm_provider,
            crm_event_id,
            event_type,
            company_name,
            contact_email,
            contact_name,
            deal_id,
            deal_name,
            deal_stage,
            body.deal_amount,
            json.dumps(body.event_data or {}, default=str),
            event_ts,
            str(account_id) if account_id else None,
        )
    except Exception as exc:
        _end_crm_trace(
            span,
            status="failed",
            event_type=event_type,
            error_message=str(exc),
            error_type=type(exc).__name__,
        )
        logger.exception("Failed to ingest CRM event")
        raise HTTPException(status_code=500, detail="Failed to store CRM event") from exc

    response = {
        "id": str(event_id),
        "status": "pending",
        "crm_provider": crm_provider,
        "event_type": event_type,
    }
    _end_crm_trace(
        span,
        status="completed",
        output_data=response,
        event_type=event_type,
        ingested=1,
        errors=0,
    )
    return response


# ---------------------------------------------------------------------------
# POST /events/batch -- ingest multiple CRM events
# ---------------------------------------------------------------------------


@router.post("/events/batch")
async def ingest_crm_events_batch(
    request: Request,
    user: AuthUser | None = Depends(optional_auth),
):
    """Ingest a batch of CRM events. Body: {"events": [...]}."""
    cfg = settings.crm_event
    if not cfg.enabled:
        raise HTTPException(status_code=503, detail="CRM event ingestion is disabled")

    try:
        raw = await request.json()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc

    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="Body must be an object containing an 'events' array")

    events = raw.get("events", [])
    if not isinstance(events, list) or not events:
        raise HTTPException(status_code=400, detail="Body must contain a non-empty 'events' array")
    if len(events) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 events per batch")

    pool = _pool_or_503()
    account_id = str(user.account_id) if user else None
    span = _start_crm_trace(
        provider="batch",
        account_id=account_id,
        batch_size=len(events),
    )

    ingested = 0
    errors = []
    created_ids: list[str] = []

    for i, evt in enumerate(events):
        if not isinstance(evt, dict):
            errors.append({"index": i, "error": "Event must be a dict"})
            continue

        provider = (_clean_optional_text(evt.get("crm_provider")) or "").lower()
        event_type = (_clean_optional_text(evt.get("event_type")) or "").lower()
        if provider not in VALID_CRM_PROVIDERS or event_type not in VALID_EVENT_TYPES:
            errors.append({"index": i, "error": "Invalid crm_provider or event_type"})
            continue

        company = _clean_optional_text(evt.get("company_name"))
        email = _clean_optional_text(evt.get("contact_email"))
        if not company and not email:
            errors.append({"index": i, "error": "company_name or contact_email required"})
            continue

        event_ts = None
        ts_raw = _clean_optional_text(evt.get("event_timestamp"))
        if ts_raw:
            try:
                event_ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        try:
            row_id = await pool.fetchval(
                """
                INSERT INTO b2b_crm_events (
                    crm_provider, crm_event_id, event_type,
                    company_name, contact_email, contact_name,
                    deal_id, deal_name, deal_stage, deal_amount,
                    event_data, event_timestamp, account_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, $13::uuid)
                ON CONFLICT (crm_provider, crm_event_id)
                    WHERE crm_event_id IS NOT NULL
                DO UPDATE SET
                    event_type = EXCLUDED.event_type,
                    deal_stage = EXCLUDED.deal_stage,
                    deal_amount = EXCLUDED.deal_amount,
                    event_data = EXCLUDED.event_data,
                    event_timestamp = EXCLUDED.event_timestamp,
                    status = 'pending',
                    processed_at = NULL
                RETURNING id
                """,
                provider,
                _clean_optional_text(evt.get("crm_event_id")),
                event_type,
                company,
                email,
                _clean_optional_text(evt.get("contact_name")),
                _clean_optional_text(evt.get("deal_id")),
                _clean_optional_text(evt.get("deal_name")),
                _clean_optional_text(evt.get("deal_stage")),
                evt.get("deal_amount"),
                json.dumps(evt.get("event_data", {}), default=str),
                event_ts,
                account_id,
            )
            ingested += 1
            if row_id:
                created_ids.append(str(row_id))
        except Exception:
            logger.debug("Batch event %d failed", i, exc_info=True)
            errors.append({"index": i, "error": "Insert failed"})

    response = {"ingested": ingested, "errors": errors, "total": len(events), "created_ids": created_ids}
    _end_crm_trace(
        span,
        status="completed",
        output_data={"ingested": ingested, "total": len(events)},
        ingested=ingested,
        errors=len(errors),
    )
    return response


# ---------------------------------------------------------------------------
# POST /events/hubspot -- HubSpot-native webhook format
# ---------------------------------------------------------------------------


@router.post("/events/hubspot")
async def ingest_hubspot_webhook(
    request: Request,
    user: AuthUser | None = Depends(optional_auth),
):
    """Accept HubSpot webhook payloads and normalize to internal format.

    Requires authentication (Bearer token or API key). HubSpot sends an
    array of events. Each event has:
    - subscriptionType: "deal.propertyChange", "deal.creation", etc.
    - objectId: the deal/contact ID
    - propertyName, propertyValue: what changed
    - occurredAt: timestamp in ms
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required for webhook ingestion")
    cfg = settings.crm_event
    if not cfg.enabled:
        raise HTTPException(status_code=503, detail="CRM event ingestion is disabled")

    pool = _pool_or_503()
    span = _start_crm_trace(
        provider="hubspot",
        account_id=str(user.account_id),
        batch_size=1,
    )

    try:
        payload = await request.json()
    except Exception:
        _end_crm_trace(
            span,
            status="failed",
            error_message="invalid hubspot webhook json",
            error_type="ValueError",
        )
        raise HTTPException(status_code=400, detail="Invalid JSON")

    events = payload if isinstance(payload, list) else [payload]
    ingested = 0

    for evt in events:
        if not isinstance(evt, dict):
            continue

        sub_type = evt.get("subscriptionType", "")
        obj_id = str(evt.get("objectId", ""))
        prop_name = evt.get("propertyName", "")
        prop_value = evt.get("propertyValue", "")
        occurred_at = evt.get("occurredAt")

        # Map HubSpot subscription types to our event types
        if "dealstage" in prop_name.lower() or sub_type == "deal.propertyChange":
            event_type = _hubspot_stage_to_event_type(prop_value)
        elif sub_type == "deal.creation":
            event_type = "deal_stage_change"
        elif "meeting" in sub_type.lower():
            event_type = "meeting_booked"
        else:
            event_type = "activity_logged"

        event_ts = None
        if occurred_at:
            try:
                event_ts = datetime.fromtimestamp(int(occurred_at) / 1000, tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                pass

        # Extract company/contact from event properties if present
        company_name = None
        contact_email = None
        deal_name = None
        deal_amount = None
        properties = evt.get("properties") or {}
        if isinstance(properties, dict):
            company_name = properties.get("company") or properties.get("dealname")
            contact_email = properties.get("email")
            deal_name = properties.get("dealname")
            try:
                deal_amount = float(properties.get("amount", 0)) or None
            except (ValueError, TypeError):
                deal_amount = None
        # Also check top-level keys (some HubSpot webhook formats)
        if not company_name:
            company_name = evt.get("associatedCompanyName") or evt.get("companyName")
        if not contact_email:
            contact_email = evt.get("email")

        notes = None
        if not company_name and not contact_email:
            notes = "HubSpot raw event -- needs company/email enrichment via HubSpot API"

        try:
            await pool.execute(
                """
                INSERT INTO b2b_crm_events (
                    crm_provider, crm_event_id, event_type,
                    company_name, contact_email, deal_id, deal_name,
                    deal_stage, deal_amount, event_data, event_timestamp,
                    processing_notes
                ) VALUES ('hubspot', $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11)
                ON CONFLICT (crm_provider, crm_event_id)
                    WHERE crm_event_id IS NOT NULL
                DO UPDATE SET
                    event_type = EXCLUDED.event_type,
                    deal_stage = EXCLUDED.deal_stage,
                    event_data = EXCLUDED.event_data,
                    status = 'pending',
                    processed_at = NULL
                """,
                f"hs-{sub_type}-{obj_id}-{occurred_at}",
                event_type,
                company_name,
                contact_email,
                obj_id,
                deal_name,
                prop_value or None,
                deal_amount,
                json.dumps(evt, default=str),
                event_ts,
                notes,
            )
            ingested += 1
        except Exception:
            logger.debug("HubSpot event ingest failed", exc_info=True)

    response = {"ingested": ingested, "total": len(events)}
    _end_crm_trace(
        span,
        status="completed",
        output_data=response,
        ingested=ingested,
        errors=max(len(events) - ingested, 0),
    )
    return response


def _hubspot_stage_to_event_type(stage_value: str) -> str:
    """Map a HubSpot deal stage value to an internal event type."""
    s = (stage_value or "").lower().replace(" ", "_")
    if "won" in s or "closed_won" in s:
        return "deal_won"
    if "lost" in s or "closed_lost" in s:
        return "deal_lost"
    if "meeting" in s or "demo" in s:
        return "meeting_booked"
    return "deal_stage_change"


# ---------------------------------------------------------------------------
# POST /events/salesforce -- Salesforce-native webhook format
# ---------------------------------------------------------------------------


@router.post("/events/salesforce")
async def ingest_salesforce_webhook(
    request: Request,
    user: AuthUser | None = Depends(optional_auth),
):
    """Accept Salesforce Outbound Message or Platform Event payloads.

    Salesforce sends events as JSON with:
    - sobject: the record type (Opportunity, Task, Event, etc.)
    - record: dict of field values
    - old: dict of previous field values (for update triggers)
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required for webhook ingestion")
    cfg = settings.crm_event
    if not cfg.enabled:
        raise HTTPException(status_code=503, detail="CRM event ingestion is disabled")

    pool = _pool_or_503()
    span = _start_crm_trace(
        provider="salesforce",
        account_id=str(user.account_id),
        batch_size=1,
    )

    try:
        payload = await request.json()
    except Exception:
        _end_crm_trace(
            span,
            status="failed",
            error_message="invalid salesforce webhook json",
            error_type="ValueError",
        )
        raise HTTPException(status_code=400, detail="Invalid JSON")

    events = payload if isinstance(payload, list) else [payload]
    ingested = 0

    for evt in events:
        if not isinstance(evt, dict):
            continue

        sobject = evt.get("sobject", evt.get("type", ""))
        record = evt.get("record", evt)
        old_record = evt.get("old", {})

        # Extract fields from Salesforce record
        sf_id = str(record.get("Id", record.get("id", "")))
        stage_name = record.get("StageName", record.get("stageName", ""))
        old_stage = (old_record.get("StageName") or old_record.get("stageName")) if old_record else None
        amount = record.get("Amount", record.get("amount"))
        company_name = (
            record.get("Account", {}).get("Name")
            or record.get("AccountName")
            or record.get("account_name")
            or record.get("Name", record.get("name"))
        )
        contact_email = (
            record.get("Contact", {}).get("Email")
            or record.get("ContactEmail")
            or record.get("email")
        )
        deal_name = record.get("Name", record.get("name", ""))

        # Determine event type
        event_type = _salesforce_stage_to_event_type(stage_name, sobject)

        event_ts = None
        ts_field = record.get("LastModifiedDate") or record.get("CreatedDate")
        if ts_field:
            try:
                event_ts = datetime.fromisoformat(str(ts_field).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        deal_amount = None
        if amount:
            try:
                deal_amount = float(amount)
            except (ValueError, TypeError):
                pass

        try:
            await pool.execute(
                """
                INSERT INTO b2b_crm_events (
                    crm_provider, crm_event_id, event_type,
                    company_name, contact_email, deal_id, deal_name,
                    deal_stage, deal_amount, event_data, event_timestamp,
                    account_id
                ) VALUES ('salesforce', $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11::uuid)
                ON CONFLICT (crm_provider, crm_event_id)
                    WHERE crm_event_id IS NOT NULL
                DO UPDATE SET
                    event_type = EXCLUDED.event_type,
                    deal_stage = EXCLUDED.deal_stage,
                    deal_amount = EXCLUDED.deal_amount,
                    event_data = EXCLUDED.event_data,
                    status = 'pending',
                    processed_at = NULL
                """,
                f"sf-{sobject}-{sf_id}",
                event_type,
                company_name,
                contact_email,
                sf_id,
                deal_name,
                stage_name or None,
                deal_amount,
                json.dumps(evt, default=str),
                event_ts,
                str(user.account_id) if user else None,
            )
            ingested += 1
        except Exception:
            logger.debug("Salesforce event ingest failed", exc_info=True)

    response = {"ingested": ingested, "total": len(events)}
    _end_crm_trace(
        span,
        status="completed",
        output_data=response,
        ingested=ingested,
        errors=max(len(events) - ingested, 0),
    )
    return response


def _salesforce_stage_to_event_type(stage: str, sobject: str) -> str:
    """Map a Salesforce stage/object to an internal event type."""
    s = (stage or "").lower()
    obj = (sobject or "").lower()
    if "closed won" in s or "closed_won" in s:
        return "deal_won"
    if "closed lost" in s or "closed_lost" in s:
        return "deal_lost"
    if obj in ("event", "task") and ("meeting" in s or "demo" in s):
        return "meeting_booked"
    if obj == "event" or "meeting" in obj:
        return "meeting_booked"
    if obj == "opportunity" or "opportunity" in obj:
        return "deal_stage_change"
    return "activity_logged"


# ---------------------------------------------------------------------------
# POST /events/pipedrive -- Pipedrive-native webhook format
# ---------------------------------------------------------------------------


@router.post("/events/pipedrive")
async def ingest_pipedrive_webhook(
    request: Request,
    user: AuthUser | None = Depends(optional_auth),
):
    """Accept Pipedrive webhook payloads.

    Pipedrive sends:
    - event: action type (e.g., "updated.deal", "added.activity")
    - current: current state of the object
    - previous: previous state (for updates)
    - meta: webhook metadata
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required for webhook ingestion")
    cfg = settings.crm_event
    if not cfg.enabled:
        raise HTTPException(status_code=503, detail="CRM event ingestion is disabled")

    pool = _pool_or_503()
    span = _start_crm_trace(
        provider="pipedrive",
        account_id=str(user.account_id),
        batch_size=1,
    )

    try:
        payload = await request.json()
    except Exception:
        _end_crm_trace(
            span,
            status="failed",
            error_message="invalid pipedrive webhook json",
            error_type="ValueError",
        )
        raise HTTPException(status_code=400, detail="Invalid JSON")

    events = payload if isinstance(payload, list) else [payload]
    ingested = 0

    for evt in events:
        if not isinstance(evt, dict):
            continue

        pd_event = evt.get("event", "")
        current = evt.get("current", evt)
        previous = evt.get("previous", {})
        meta = evt.get("meta", {})

        # Extract Pipedrive fields
        pd_id = str(current.get("id", ""))
        stage_name = current.get("stage_id") or current.get("pipeline_id")
        status_field = (current.get("status") or "").lower()
        deal_name = current.get("title", "")
        company_name = (
            current.get("org_name")
            or (current.get("org_id", {}).get("name") if isinstance(current.get("org_id"), dict) else None)
        )
        contact_email = (
            current.get("person_id", {}).get("email", [{}])[0].get("value")
            if isinstance(current.get("person_id"), dict) else None
        )
        if not contact_email:
            contact_email = current.get("email")

        amount = current.get("value") or current.get("weighted_value")
        deal_amount = None
        if amount:
            try:
                deal_amount = float(amount)
            except (ValueError, TypeError):
                pass

        # Determine event type
        event_type = _pipedrive_to_event_type(pd_event, status_field)

        event_ts = None
        ts_field = current.get("update_time") or current.get("add_time")
        if ts_field:
            try:
                event_ts = datetime.fromisoformat(str(ts_field).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        try:
            await pool.execute(
                """
                INSERT INTO b2b_crm_events (
                    crm_provider, crm_event_id, event_type,
                    company_name, contact_email, deal_id, deal_name,
                    deal_stage, deal_amount, event_data, event_timestamp,
                    account_id
                ) VALUES ('pipedrive', $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11::uuid)
                ON CONFLICT (crm_provider, crm_event_id)
                    WHERE crm_event_id IS NOT NULL
                DO UPDATE SET
                    event_type = EXCLUDED.event_type,
                    deal_stage = EXCLUDED.deal_stage,
                    deal_amount = EXCLUDED.deal_amount,
                    event_data = EXCLUDED.event_data,
                    status = 'pending',
                    processed_at = NULL
                """,
                f"pd-{pd_event}-{pd_id}",
                event_type,
                company_name,
                contact_email,
                pd_id,
                deal_name,
                str(stage_name) if stage_name else None,
                deal_amount,
                json.dumps(evt, default=str),
                event_ts,
                str(user.account_id) if user else None,
            )
            ingested += 1
        except Exception:
            logger.debug("Pipedrive event ingest failed", exc_info=True)

    response = {"ingested": ingested, "total": len(events)}
    _end_crm_trace(
        span,
        status="completed",
        output_data=response,
        ingested=ingested,
        errors=max(len(events) - ingested, 0),
    )
    return response


def _pipedrive_to_event_type(event: str, status: str) -> str:
    """Map a Pipedrive event/status to an internal event type."""
    e = (event or "").lower()
    s = (status or "").lower()
    if s == "won" or "won" in e:
        return "deal_won"
    if s == "lost" or "lost" in e:
        return "deal_lost"
    if "activity" in e and ("meeting" in s or "call" in s):
        return "meeting_booked"
    if "deal" in e:
        return "deal_stage_change"
    if "activity" in e:
        return "activity_logged"
    return "deal_stage_change"


# ---------------------------------------------------------------------------
# GET /events -- list ingested CRM events
# ---------------------------------------------------------------------------


@router.get("/events")
async def list_crm_events(
    status: Optional[str] = Query(None, description="Filter: pending, matched, unmatched, skipped"),
    crm_provider: Optional[str] = Query(None),
    company_name: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None, description="Filter events received on or after (ISO 8601)"),
    end_date: Optional[str] = Query(None, description="Filter events received before (ISO 8601)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: AuthUser | None = Depends(optional_auth),
):
    """List ingested CRM events with optional filters."""
    status = (_clean_optional_text(status) or "").lower() or None
    crm_provider = (_clean_optional_text(crm_provider) or "").lower() or None
    company_name = _clean_optional_text(company_name)
    start_date = _clean_optional_text(start_date)
    end_date = _clean_optional_text(end_date)

    if status and status not in VALID_EVENT_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {sorted(VALID_EVENT_STATUSES)}",
        )
    if crm_provider and crm_provider not in VALID_CRM_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crm_provider. Must be one of: {sorted(VALID_CRM_PROVIDERS)}",
        )

    sd = None
    if start_date:
        try:
            sd = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid start_date (ISO 8601 expected)") from exc

    ed = None
    if end_date:
        try:
            ed = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid end_date (ISO 8601 expected)") from exc

    pool = _pool_or_503()
    conditions = ["1=1"]
    params: list = []
    idx = 1

    if user:
        conditions.append(f"account_id = ${idx}::uuid")
        params.append(str(user.account_id))
        idx += 1

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
    params.extend([limit, offset])

    rows = await pool.fetch(
        f"""
        SELECT id, crm_provider, crm_event_id, event_type,
               company_name, contact_email, contact_name,
               deal_id, deal_name, deal_stage, deal_amount,
               status, matched_sequence_id, outcome_recorded,
               processing_notes, event_timestamp, received_at, processed_at
        FROM b2b_crm_events
        WHERE {where}
        ORDER BY received_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
    )

    return {
        "events": [
            {
                "id": str(r["id"]),
                "crm_provider": r["crm_provider"],
                "crm_event_id": r["crm_event_id"],
                "event_type": r["event_type"],
                "company_name": r["company_name"],
                "contact_email": r["contact_email"],
                "deal_stage": r["deal_stage"],
                "deal_amount": float(r["deal_amount"]) if r["deal_amount"] else None,
                "status": r["status"],
                "matched_sequence_id": str(r["matched_sequence_id"]) if r["matched_sequence_id"] else None,
                "outcome_recorded": r["outcome_recorded"],
                "event_timestamp": str(r["event_timestamp"]) if r["event_timestamp"] else None,
                "received_at": str(r["received_at"]),
                "processed_at": str(r["processed_at"]) if r["processed_at"] else None,
            }
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# GET /events/enrichment-stats  -- enrichment effectiveness report
# ---------------------------------------------------------------------------


@router.get("/events/enrichment-stats")
async def get_enrichment_stats(
    user: AuthUser | None = Depends(optional_auth),
):
    """Show enrichment coverage and effectiveness stats for CRM events."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not ready")

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
    return {
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
    }
