"""Process pending CRM events and auto-record campaign outcomes.

Matches ingested CRM events (from b2b_crm_events) to campaign sequences
by company name or contact email. When a match is found, auto-records the
appropriate campaign outcome (meeting_booked, deal_won, etc.) and logs an
audit event.

Runs on a 5-minute interval.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.tasks.crm_event_processing")


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Process pending CRM events and match to campaign sequences."""
    cfg = settings.crm_event
    if not cfg.enabled:
        return {"_skip_synthesis": "CRM event ingestion disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Fetch pending events
    rows = await pool.fetch(
        """
        SELECT id, crm_provider, event_type, company_name, contact_email,
               deal_id, deal_name, deal_stage, deal_amount, event_data,
               event_timestamp, account_id
        FROM b2b_crm_events
        WHERE status = 'pending'
        ORDER BY received_at ASC
        LIMIT $1
        """,
        cfg.batch_size,
    )

    if not rows:
        return {"_skip_synthesis": "No pending CRM events"}

    matched = 0
    unmatched = 0
    outcomes_recorded = 0
    skipped = 0
    errors = 0

    for row in rows:
        event_id = row["id"]
        try:
            result = await _process_single_event(pool, row, cfg)
            if result == "matched":
                matched += 1
                outcomes_recorded += 1
            elif result == "unmatched":
                unmatched += 1
            elif result == "skipped":
                skipped += 1
        except Exception:
            logger.exception("Error processing CRM event %s", event_id)
            errors += 1
            await pool.execute(
                """
                UPDATE b2b_crm_events
                SET status = 'error', processing_notes = 'Processing exception',
                    processed_at = NOW()
                WHERE id = $1
                """,
                event_id,
            )

    return {
        "_skip_synthesis": "CRM event processing complete",
        "processed": len(rows),
        "matched": matched,
        "unmatched": unmatched,
        "outcomes_recorded": outcomes_recorded,
        "skipped": skipped,
        "errors": errors,
    }


async def _process_single_event(pool, row: dict, cfg) -> str:
    """Process a single CRM event. Returns 'matched', 'unmatched', or 'skipped'."""
    event_id = row["id"]
    event_type = row["event_type"]
    company_name = row["company_name"]
    contact_email = row["contact_email"]
    deal_stage = row["deal_stage"]

    # Determine the outcome to record
    outcome = _resolve_outcome(event_type, deal_stage, cfg.stage_outcome_map)
    if not outcome:
        await pool.execute(
            """
            UPDATE b2b_crm_events
            SET status = 'skipped', processing_notes = 'No outcome mapping for event',
                processed_at = NOW()
            WHERE id = $1
            """,
            event_id,
        )
        return "skipped"

    # Enrich missing fields before matching
    company_name, contact_email = await _enrich_event_fields(
        pool, event_id, company_name, contact_email, row.get("deal_id"),
    )

    # Try to find a matching campaign sequence
    sequence = await _find_matching_sequence(pool, company_name, contact_email)
    if not sequence:
        await pool.execute(
            """
            UPDATE b2b_crm_events
            SET status = 'unmatched', processing_notes = 'No matching campaign sequence',
                processed_at = NOW()
            WHERE id = $1
            """,
            event_id,
        )
        return "unmatched"

    seq_id = sequence["id"]
    current_outcome = sequence["outcome"]

    # Don't downgrade outcomes (e.g. deal_won -> meeting_booked)
    if _outcome_rank(current_outcome) >= _outcome_rank(outcome):
        await pool.execute(
            """
            UPDATE b2b_crm_events
            SET status = 'skipped', matched_sequence_id = $2,
                processing_notes = $3, processed_at = NOW()
            WHERE id = $1
            """,
            event_id, seq_id,
            f"Current outcome '{current_outcome}' already >= '{outcome}'",
        )
        return "skipped"

    # Record the outcome on the campaign sequence
    now = datetime.now(timezone.utc)
    history_entry = {
        "stage": outcome,
        "recorded_at": now.isoformat(),
        "previous": current_outcome,
        "notes": f"Auto-recorded from {row['crm_provider']} event",
        "recorded_by": f"crm:{row['crm_provider']}",
    }
    if row["deal_name"]:
        history_entry["deal_name"] = row["deal_name"]
    if row["deal_amount"]:
        history_entry["deal_amount"] = float(row["deal_amount"])

    # Fetch current history
    current_history = await pool.fetchval(
        "SELECT outcome_history FROM campaign_sequences WHERE id = $1",
        seq_id,
    )
    history_list = current_history if isinstance(current_history, list) else []
    history_list.append(history_entry)

    await pool.execute(
        """
        UPDATE campaign_sequences
        SET outcome = $1,
            outcome_recorded_at = $2,
            outcome_recorded_by = $3,
            outcome_notes = $4,
            outcome_revenue = COALESCE($5, outcome_revenue),
            outcome_history = $6::jsonb,
            updated_at = NOW()
        WHERE id = $7
        """,
        outcome,
        now,
        f"crm:{row['crm_provider']}",
        f"Auto-recorded from {row['crm_provider']}: {row['event_type']}",
        float(row["deal_amount"]) if row["deal_amount"] else None,
        json.dumps(history_list, default=str),
        seq_id,
    )

    # Log audit event
    try:
        from .campaign_audit import log_campaign_event
        await log_campaign_event(
            pool,
            event_type=f"outcome_{outcome}",
            source=f"crm:{row['crm_provider']}",
            sequence_id=seq_id,
            metadata={
                "outcome": outcome,
                "previous_outcome": current_outcome,
                "crm_event_id": str(event_id),
                "crm_provider": row["crm_provider"],
                "deal_id": row["deal_id"],
                "deal_amount": float(row["deal_amount"]) if row["deal_amount"] else None,
            },
        )
    except Exception:
        logger.debug("Audit log skipped for CRM event %s", event_id)

    # Mark event as processed
    await pool.execute(
        """
        UPDATE b2b_crm_events
        SET status = 'matched', matched_sequence_id = $2,
            outcome_recorded = $3, processed_at = NOW(),
            processing_notes = $4
        WHERE id = $1
        """,
        event_id, seq_id, outcome,
        f"Matched sequence {seq_id}, recorded outcome '{outcome}'",
    )

    return "matched"


async def _enrich_event_fields(
    pool, event_id, company_name: str | None, contact_email: str | None, deal_id: str | None,
) -> tuple[str | None, str | None]:
    """Enrich missing company_name or contact_email from cross-event data.

    1. Cross-event: look up other events with the same deal_id that have the field.
    2. Vendor normalization: normalize company_name via resolve_vendor_name().
    Returns (company_name, contact_email) -- possibly enriched.
    """
    enriched = False

    # Cross-event enrichment via deal_id
    if deal_id and (not company_name or not contact_email):
        try:
            sibling = await pool.fetchrow(
                """
                SELECT company_name, contact_email
                FROM b2b_crm_events
                WHERE deal_id = $1 AND id != $2
                  AND (company_name IS NOT NULL OR contact_email IS NOT NULL)
                ORDER BY received_at DESC
                LIMIT 1
                """,
                deal_id, event_id,
            )
            if sibling:
                if not company_name and sibling["company_name"]:
                    company_name = sibling["company_name"]
                    enriched = True
                if not contact_email and sibling["contact_email"]:
                    contact_email = sibling["contact_email"]
                    enriched = True
        except Exception:
            logger.debug("Cross-event enrichment failed for event %s", event_id)

    # Normalize company name via vendor registry
    if company_name:
        try:
            from ...services.vendor_registry import resolve_vendor_name
            normalized = await resolve_vendor_name(company_name)
            if normalized and normalized != company_name:
                company_name = normalized
                enriched = True
        except Exception:
            pass  # Fail-open

    # Persist enriched fields back to the event row
    if enriched:
        try:
            await pool.execute(
                """
                UPDATE b2b_crm_events
                SET company_name = COALESCE($2, company_name),
                    contact_email = COALESCE($3, contact_email),
                    processing_notes = COALESCE(processing_notes, '') || ' [enriched]'
                WHERE id = $1
                """,
                event_id, company_name, contact_email,
            )
        except Exception:
            logger.debug("Failed to persist enriched fields for event %s", event_id)

    return company_name, contact_email


async def _find_matching_sequence(pool, company_name: str | None, contact_email: str | None):
    """Find the best matching campaign sequence by company name or email.

    Priority:
    1. Exact match on recipient_email
    2. Case-insensitive match on company_name
    Returns the most recent active sequence.
    """
    # Priority 1: match by email
    if contact_email:
        row = await pool.fetchrow(
            """
            SELECT id, outcome FROM campaign_sequences
            WHERE LOWER(recipient_email) = LOWER($1)
              AND status NOT IN ('bounced', 'unsubscribed')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            contact_email,
        )
        if row:
            return row

    # Priority 2: match by company name (exact, case-insensitive)
    if company_name:
        row = await pool.fetchrow(
            """
            SELECT id, outcome FROM campaign_sequences
            WHERE LOWER(company_name) = LOWER($1)
              AND status NOT IN ('bounced', 'unsubscribed')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            company_name,
        )
        if row:
            return row

    # Priority 3: fuzzy company match (trigram similarity >= 0.6)
    if company_name and len(company_name) >= 4:
        try:
            row = await pool.fetchrow(
                """
                SELECT id, outcome FROM campaign_sequences
                WHERE similarity(LOWER(company_name), LOWER($1)) >= 0.6
                  AND status NOT IN ('bounced', 'unsubscribed')
                ORDER BY similarity(LOWER(company_name), LOWER($1)) DESC, created_at DESC
                LIMIT 1
                """,
                company_name,
            )
            if row:
                return row
        except Exception:
            pass  # pg_trgm may not be available; fail-open

    return None


_OUTCOME_RANKS = {
    "pending": 0,
    "no_opportunity": 1,
    "disqualified": 1,
    "meeting_booked": 2,
    "deal_opened": 3,
    "deal_lost": 3,     # Terminal negative -- same rank as deal_opened so deal_won can override
    "deal_won": 5,
}


def _outcome_rank(outcome: str) -> int:
    """Return a numeric rank for outcome progression."""
    return _OUTCOME_RANKS.get(outcome, 0)


def _resolve_outcome(event_type: str, deal_stage: str | None, stage_map: dict) -> str | None:
    """Resolve a CRM event to a campaign outcome value.

    Checks direct event_type mapping first, then deal_stage mapping.
    """
    # Direct event type mapping
    direct_map = {
        "deal_won": "deal_won",
        "deal_lost": "deal_lost",
        "meeting_booked": "meeting_booked",
    }
    if event_type in direct_map:
        return direct_map[event_type]

    # Deal stage mapping via config
    if deal_stage:
        normalized = deal_stage.lower().replace(" ", "_")
        outcome = stage_map.get(normalized)
        if outcome:
            return outcome

    # For generic stage changes, try to infer
    if event_type == "deal_stage_change" and deal_stage:
        normalized = deal_stage.lower()
        if "won" in normalized:
            return "deal_won"
        if "lost" in normalized:
            return "deal_lost"
        if "meeting" in normalized or "demo" in normalized:
            return "meeting_booked"
        # Only map to deal_opened if stage name suggests active pipeline
        if any(k in normalized for k in ("proposal", "negotiat", "qualif", "pipeline", "active")):
            return "deal_opened"

    return None
