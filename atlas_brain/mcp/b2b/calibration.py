"""B2B Churn MCP -- calibration tools."""

import json
import uuid as _uuid
from typing import Optional

from ._shared import _is_uuid, get_pool, logger
from .server import mcp

_VALID_OUTCOMES = {
    "pending", "meeting_booked", "deal_opened",
    "deal_won", "deal_lost", "no_opportunity", "disqualified",
}

_GROUP_BY_EXPRESSIONS = {
    "buying_stage": "bc.buying_stage",
    "role_type": "bc.role_type",
    "target_mode": "bc.target_mode",
    "opportunity_score_bucket": """CASE
        WHEN bc.opportunity_score >= 90 THEN '90-100'
        WHEN bc.opportunity_score >= 70 THEN '70-89'
        WHEN bc.opportunity_score >= 50 THEN '50-69'
        ELSE 'below_50'
    END""",
    "urgency_bucket": """CASE
        WHEN bc.urgency_score >= 8 THEN 'high_8+'
        WHEN bc.urgency_score >= 5 THEN 'medium_5-7'
        ELSE 'low_0-4'
    END""",
    "pain_category": "bc.pain_categories->0->>'category'",
}


@mcp.tool()
async def record_campaign_outcome(
    sequence_id: str,
    outcome: str,
    notes: Optional[str] = None,
    revenue: Optional[float] = None,
    recorded_by: str = "mcp",
) -> str:
    """
    Record a business outcome for a campaign sequence.

    sequence_id: UUID of the campaign sequence
    outcome: One of: pending, meeting_booked, deal_opened, deal_won, deal_lost, no_opportunity, disqualified
    notes: Optional notes about the outcome
    revenue: Optional revenue amount (for deal_won)
    recorded_by: Who recorded this (default "mcp")
    """
    if not _is_uuid(sequence_id):
        return json.dumps({"success": False, "error": "Invalid sequence_id (must be UUID)"})

    if outcome not in _VALID_OUTCOMES:
        return json.dumps({"success": False, "error": f"Invalid outcome. Must be one of: {sorted(_VALID_OUTCOMES)}"})

    try:
        from datetime import datetime, timezone

        from atlas_brain.autonomous.tasks.campaign_audit import log_campaign_event

        pool = get_pool()

        # suppression: intentionally excluded -- writes to outcome measurement table;
        # suppressing campaign sequences would distort calibration math.
        row = await pool.fetchrow(
            "SELECT id, outcome, outcome_history FROM campaign_sequences WHERE id = $1",
            _uuid.UUID(sequence_id),
        )
        if not row:
            return json.dumps({"success": False, "error": "Sequence not found"})

        previous = row["outcome"]
        now = datetime.now(timezone.utc)

        history_entry = {
            "stage": outcome,
            "recorded_at": now.isoformat(),
            "previous": previous,
            "notes": notes,
            "recorded_by": recorded_by,
        }
        existing_history = row["outcome_history"] if isinstance(row["outcome_history"], list) else []
        updated_history = existing_history + [history_entry]

        await pool.execute(
            """
            UPDATE campaign_sequences
            SET outcome = $1,
                outcome_recorded_at = $2,
                outcome_recorded_by = $3,
                outcome_notes = $4,
                outcome_revenue = $5,
                outcome_history = $6::jsonb,
                updated_at = NOW()
            WHERE id = $7
            """,
            outcome,
            now,
            recorded_by,
            notes,
            revenue,
            json.dumps(updated_history),
            _uuid.UUID(sequence_id),
        )

        await log_campaign_event(
            pool,
            event_type=f"outcome_{outcome}",
            source=recorded_by,
            sequence_id=_uuid.UUID(sequence_id),
            metadata={
                "outcome": outcome,
                "previous": previous,
                "revenue": revenue,
                "notes": notes,
            },
        )

        return json.dumps({
            "success": True,
            "sequence_id": sequence_id,
            "outcome": outcome,
            "previous_outcome": previous,
            "recorded_at": now.isoformat(),
        })
    except Exception:
        logger.exception("record_campaign_outcome error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_signal_effectiveness(
    vendor_name: Optional[str] = None,
    min_sequences: int = 5,
    group_by: str = "buying_stage",
) -> str:
    """
    Analyze which signal dimensions produce the best campaign outcomes.

    Groups completed campaign sequences by a signal dimension from the
    originating b2b_campaigns row (step 1 only) and computes outcome rates.

    vendor_name: Optional vendor filter (partial match, case-insensitive)
    min_sequences: Minimum sequences per group to include (default 5, 1-100)
    group_by: Signal dimension to group by. Options:
        buying_stage (default), role_type, target_mode,
        opportunity_score_bucket, urgency_bucket, pain_category
    """
    # suppression: intentionally excluded -- operates on campaign_sequences/b2b_campaigns
    # (outcome measurement tables); suppressing these would distort calibration math.
    min_sequences = max(1, min(min_sequences, 100))

    if group_by not in _GROUP_BY_EXPRESSIONS:
        return json.dumps({
            "success": False,
            "error": f"Invalid group_by. Must be one of: {sorted(_GROUP_BY_EXPRESSIONS.keys())}",
        })

    group_expr = _GROUP_BY_EXPRESSIONS[group_by]

    try:
        pool = get_pool()

        vendor_filter = ""
        params: list = [min_sequences]
        idx = 2

        if vendor_name:
            vendor_filter = f"AND bc.vendor_name ILIKE '%' || ${idx} || '%'"
            params.append(vendor_name)
            idx += 1

        sql = f"""
        WITH seq_signals AS (
            SELECT DISTINCT ON (cs.id)
                cs.id, cs.outcome, cs.outcome_revenue,
                ({group_expr}) AS signal_group
            FROM campaign_sequences cs
            JOIN b2b_campaigns bc ON bc.sequence_id = cs.id
            WHERE bc.sequence_id IS NOT NULL
              AND cs.outcome != 'pending'
              {vendor_filter}
            ORDER BY cs.id, bc.step_number ASC
        )
        SELECT
            signal_group,
            COUNT(*) AS total_sequences,
            COUNT(*) FILTER (WHERE outcome = 'meeting_booked') AS meetings,
            COUNT(*) FILTER (WHERE outcome = 'deal_opened') AS deals_opened,
            COUNT(*) FILTER (WHERE outcome = 'deal_won') AS deals_won,
            COUNT(*) FILTER (WHERE outcome = 'deal_lost') AS deals_lost,
            COUNT(*) FILTER (WHERE outcome = 'no_opportunity') AS no_opportunity,
            COUNT(*) FILTER (WHERE outcome = 'disqualified') AS disqualified,
            ROUND(
                COUNT(*) FILTER (WHERE outcome IN ('meeting_booked', 'deal_opened', 'deal_won'))::numeric
                / NULLIF(COUNT(*), 0), 3
            ) AS positive_outcome_rate,
            COALESCE(SUM(outcome_revenue) FILTER (WHERE outcome = 'deal_won'), 0) AS total_revenue
        FROM seq_signals
        WHERE signal_group IS NOT NULL
        GROUP BY signal_group
        HAVING COUNT(*) >= $1
        ORDER BY positive_outcome_rate DESC
        """

        rows = await pool.fetch(sql, *params)

        groups = []
        for r in rows:
            groups.append({
                "signal_group": r["signal_group"],
                "total_sequences": r["total_sequences"],
                "meetings": r["meetings"],
                "deals_opened": r["deals_opened"],
                "deals_won": r["deals_won"],
                "deals_lost": r["deals_lost"],
                "no_opportunity": r["no_opportunity"],
                "disqualified": r["disqualified"],
                "positive_outcome_rate": float(r["positive_outcome_rate"]) if r["positive_outcome_rate"] is not None else 0.0,
                "total_revenue": float(r["total_revenue"]),
            })

        return json.dumps({
            "success": True,
            "group_by": group_by,
            "vendor_filter": vendor_name,
            "min_sequences": min_sequences,
            "groups": groups,
            "total_groups": len(groups),
        }, default=str)
    except Exception:
        logger.exception("get_signal_effectiveness error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def get_outcome_distribution(
    vendor_name: Optional[str] = None,
) -> str:
    """System-wide campaign outcome distribution (funnel view).

    Returns count and percentage for each outcome state, plus total revenue.

    vendor_name: Optional vendor filter (partial match, case-insensitive).
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        # suppression: intentionally excluded -- outcome measurement table;
        # suppressing records here would distort calibration math.
        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"bc.vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT cs.outcome,
                   COUNT(*) AS count,
                   COALESCE(SUM(cs.outcome_revenue), 0) AS total_revenue,
                   MIN(cs.outcome_recorded_at) AS first_recorded,
                   MAX(cs.outcome_recorded_at) AS last_recorded
            FROM campaign_sequences cs
            LEFT JOIN b2b_campaigns bc ON bc.sequence_id = cs.id AND bc.step_number = 1
            {where}
            GROUP BY cs.outcome
            ORDER BY count DESC
            """,
            *params,
        )

        total = sum(r["count"] for r in rows)
        buckets = []
        for r in rows:
            buckets.append({
                "outcome": r["outcome"],
                "count": r["count"],
                "pct": round(r["count"] / total * 100, 1) if total else 0,
                "total_revenue": float(r["total_revenue"]) if r["total_revenue"] else 0.0,
                "first_recorded": str(r["first_recorded"]) if r["first_recorded"] else None,
                "last_recorded": str(r["last_recorded"]) if r["last_recorded"] else None,
            })

        return json.dumps({
            "success": True,
            "total_sequences": total,
            "vendor_filter": vendor_name,
            "buckets": buckets,
        }, default=str)
    except Exception:
        logger.exception("get_outcome_distribution error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def trigger_score_calibration(
    window_days: int = 180,
) -> str:
    """Trigger score calibration from campaign outcomes on-demand.

    Recalculates calibration weights based on observed outcome rates per
    signal dimension. Results are stored in score_calibration_weights.

    window_days: How many days of outcome data to consider (default 180, min 30, max 730).
    """
    try:
        from atlas_brain.autonomous.tasks.b2b_score_calibration import calibrate

        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        window_days = max(30, min(window_days, 730))
        result = await calibrate(pool, window_days=window_days)
        return json.dumps({
            "success": True,
            "triggered_by": "mcp",
            "window_days": window_days,
            **result,
        }, default=str)
    except Exception:
        logger.exception("trigger_score_calibration error")
        return json.dumps({"success": False, "error": "Calibration failed"})


@mcp.tool()
async def get_calibration_weights(
    dimension: Optional[str] = None,
    model_version: Optional[int] = None,
) -> str:
    """
    Get current score calibration weights derived from campaign outcomes.

    Calibration weights adjust the static opportunity scoring formula based
    on observed conversion rates per signal dimension.

    dimension: Filter by dimension (role_type, buying_stage, urgency_bucket, seat_bucket, context_keyword)
    model_version: Specific version (default: latest)
    """
    valid_dims = {"role_type", "buying_stage", "urgency_bucket", "seat_bucket", "context_keyword"}
    if dimension and dimension not in valid_dims:
        return json.dumps({"success": False, "error": f"Invalid dimension. Must be one of: {sorted(valid_dims)}"})

    try:
        pool = get_pool()

        if model_version is None:
            model_version = await pool.fetchval(
                "SELECT MAX(model_version) FROM score_calibration_weights"
            )
            if model_version is None:
                return json.dumps({
                    "success": True,
                    "weights": [],
                    "count": 0,
                    "message": "No calibration data yet. Run calibration after recording campaign outcomes.",
                })

        conditions = ["model_version = $1"]
        params: list = [model_version]
        idx = 2

        if dimension:
            conditions.append(f"dimension = ${idx}")
            params.append(dimension)
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT dimension, dimension_value, total_sequences, positive_outcomes,
                   deals_won, total_revenue, positive_rate, baseline_rate, lift,
                   weight_adjustment, static_default, calibrated_at,
                   sample_window_days, model_version
            FROM score_calibration_weights
            WHERE {where}
            ORDER BY dimension, lift DESC
            """,
            *params,
        )

        weights = []
        for r in rows:
            weights.append({
                "dimension": r["dimension"],
                "dimension_value": r["dimension_value"],
                "total_sequences": r["total_sequences"],
                "positive_outcomes": r["positive_outcomes"],
                "deals_won": r["deals_won"],
                "total_revenue": float(r["total_revenue"]),
                "positive_rate": float(r["positive_rate"]),
                "baseline_rate": float(r["baseline_rate"]),
                "lift": float(r["lift"]),
                "weight_adjustment": float(r["weight_adjustment"]),
                "static_default": float(r["static_default"]),
                "calibrated_at": r["calibrated_at"],
                "sample_window_days": r["sample_window_days"],
                "model_version": r["model_version"],
            })

        return json.dumps({
            "success": True,
            "model_version": model_version,
            "weights": weights,
            "count": len(weights),
        }, default=str)
    except Exception:
        logger.exception("get_calibration_weights error")
        return json.dumps({"success": False, "error": "Internal error"})
