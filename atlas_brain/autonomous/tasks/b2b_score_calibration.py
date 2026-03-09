"""
B2B score calibration task.

Computes calibration weights from campaign outcome data. For each scoring
dimension (role_type, buying_stage, urgency_bucket, seat_bucket, context_keyword),
calculates positive outcome rates and derives weight adjustments relative to
the overall baseline.

Runs weekly. Requires at least MIN_SEQUENCES_FOR_CALIBRATION completed sequences
with non-pending outcomes before producing any weights.
"""

import json
import logging
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_score_calibration")

MIN_SEQUENCES_FOR_CALIBRATION = 20
MIN_PER_BUCKET = 3
SAMPLE_WINDOW_DAYS = 90

# Maps scoring dimensions to SQL expressions that extract the dimension
# value from the step-1 b2b_campaigns row joined to its sequence outcome.
_DIMENSION_SQL = {
    "role_type": "bc.role_type",
    "buying_stage": "bc.buying_stage",
    "urgency_bucket": """CASE
        WHEN bc.urgency_score >= 8 THEN 'high'
        WHEN bc.urgency_score >= 5 THEN 'medium'
        ELSE 'low'
    END""",
    "seat_bucket": """CASE
        WHEN (bc.enrichment->'budget_signals'->>'seat_count')::int >= 500 THEN '500+'
        WHEN (bc.enrichment->'budget_signals'->>'seat_count')::int >= 100 THEN '100-499'
        WHEN (bc.enrichment->'budget_signals'->>'seat_count')::int >= 20 THEN '20-99'
        ELSE 'small'
    END""",
    "context_keyword": """CASE
        WHEN bc.enrichment->'competitors_mentioned'->0->>'context' ILIKE '%considering%' THEN 'considering'
        WHEN bc.enrichment->'competitors_mentioned'->0->>'context' ILIKE '%switched_to%' THEN 'switched_to'
        WHEN bc.enrichment->'competitors_mentioned'->0->>'context' ILIKE '%compared%' THEN 'compared'
        WHEN bc.enrichment->'competitors_mentioned'->0->>'context' ILIKE '%switched_from%' THEN 'switched_from'
        ELSE 'none'
    END""",
}

# Static defaults from _compute_score in b2b_campaign_generation.py
_STATIC_DEFAULTS = {
    "role_type": {
        "decision_maker": 20.0,
        "economic_buyer": 15.0,
        "champion": 15.0,
        "evaluator": 10.0,
    },
    "buying_stage": {
        "active_purchase": 25.0,
        "evaluation": 20.0,
        "renewal_decision": 15.0,
        "post_purchase": 5.0,
    },
    "urgency_bucket": {
        "high": 30.0,
        "medium": 15.0,
        "low": 0.0,
    },
    "seat_bucket": {
        "500+": 15.0,
        "100-499": 10.0,
        "20-99": 5.0,
        "small": 0.0,
    },
    "context_keyword": {
        "considering": 10.0,
        "switched_to": 8.0,
        "compared": 6.0,
        "switched_from": 2.0,
        "none": 0.0,
    },
}


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: calibrate opportunity scores from outcomes."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database unavailable"}

    try:
        result = await calibrate(pool)
        return {"_skip_synthesis": True, **result}
    except Exception as exc:
        logger.exception("Score calibration failed")
        return {"_skip_synthesis": True, "error": str(exc)}


async def calibrate(
    pool,
    window_days: int = SAMPLE_WINDOW_DAYS,
) -> dict[str, Any]:
    """Core calibration logic. Also callable from API/MCP trigger."""

    # 1. Check we have enough outcome data
    total_with_outcomes = await pool.fetchval(
        """
        SELECT COUNT(*) FROM campaign_sequences
        WHERE outcome != 'pending'
          AND outcome_recorded_at > NOW() - make_interval(days => $1)
        """,
        window_days,
    )

    if total_with_outcomes < MIN_SEQUENCES_FOR_CALIBRATION:
        logger.info(
            "Insufficient outcome data for calibration: %d/%d required",
            total_with_outcomes, MIN_SEQUENCES_FOR_CALIBRATION,
        )
        return {
            "calibrated": False,
            "reason": f"Need {MIN_SEQUENCES_FOR_CALIBRATION} sequences with outcomes, have {total_with_outcomes}",
            "total_with_outcomes": total_with_outcomes,
        }

    # 2. Compute overall baseline positive rate
    baseline_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE outcome IN ('meeting_booked', 'deal_opened', 'deal_won')) AS positive
        FROM campaign_sequences
        WHERE outcome != 'pending'
          AND outcome_recorded_at > NOW() - make_interval(days => $1)
        """,
        window_days,
    )
    baseline_total = baseline_row["total"]
    baseline_positive = baseline_row["positive"]
    baseline_rate = baseline_positive / max(baseline_total, 1)

    # 3. For each dimension, compute per-value stats
    # Get current model version
    current_version = await pool.fetchval(
        "SELECT COALESCE(MAX(model_version), 0) FROM score_calibration_weights"
    )
    new_version = current_version + 1

    total_weights = 0
    dimensions_calibrated = []

    for dimension, sql_expr in _DIMENSION_SQL.items():
        rows = await pool.fetch(
            f"""
            WITH seq_signals AS (
                SELECT DISTINCT ON (cs.id)
                    cs.id, cs.outcome, cs.outcome_revenue,
                    ({sql_expr}) AS dim_value
                FROM campaign_sequences cs
                JOIN b2b_campaigns bc ON bc.sequence_id = cs.id
                WHERE bc.sequence_id IS NOT NULL
                  AND cs.outcome != 'pending'
                  AND cs.outcome_recorded_at > NOW() - make_interval(days => $1)
                ORDER BY cs.id, bc.step_number ASC
            )
            SELECT
                dim_value,
                COUNT(*) AS total_sequences,
                COUNT(*) FILTER (WHERE outcome IN ('meeting_booked', 'deal_opened', 'deal_won')) AS positive_outcomes,
                COUNT(*) FILTER (WHERE outcome = 'deal_won') AS deals_won,
                COALESCE(SUM(outcome_revenue) FILTER (WHERE outcome = 'deal_won'), 0) AS total_revenue
            FROM seq_signals
            WHERE dim_value IS NOT NULL
            GROUP BY dim_value
            HAVING COUNT(*) >= $2
            """,
            window_days, MIN_PER_BUCKET,
        )

        if not rows:
            continue

        dim_defaults = _STATIC_DEFAULTS.get(dimension, {})
        dim_weight_count = 0

        for r in rows:
            dim_value = r["dim_value"]
            total_seq = r["total_sequences"]
            positive = r["positive_outcomes"]
            pos_rate = positive / max(total_seq, 1)
            lift = pos_rate / max(baseline_rate, 0.001)

            static_default = dim_defaults.get(dim_value, 0.0)

            # Weight adjustment: scale static default by lift
            # lift > 1 = this dimension value converts better than average -> increase weight
            # lift < 1 = converts worse -> decrease weight
            # Capped at +/- 50% of the static default to prevent wild swings
            max_adjustment = max(abs(static_default) * 0.5, 5.0)
            raw_adjustment = static_default * (lift - 1.0)
            weight_adjustment = max(-max_adjustment, min(max_adjustment, raw_adjustment))

            await pool.execute(
                """
                INSERT INTO score_calibration_weights (
                    dimension, dimension_value, total_sequences, positive_outcomes,
                    deals_won, total_revenue, positive_rate, baseline_rate, lift,
                    weight_adjustment, static_default, calibrated_at,
                    sample_window_days, model_version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), $12, $13)
                ON CONFLICT (dimension, dimension_value, model_version) DO UPDATE SET
                    total_sequences = EXCLUDED.total_sequences,
                    positive_outcomes = EXCLUDED.positive_outcomes,
                    deals_won = EXCLUDED.deals_won,
                    total_revenue = EXCLUDED.total_revenue,
                    positive_rate = EXCLUDED.positive_rate,
                    baseline_rate = EXCLUDED.baseline_rate,
                    lift = EXCLUDED.lift,
                    weight_adjustment = EXCLUDED.weight_adjustment,
                    static_default = EXCLUDED.static_default,
                    calibrated_at = NOW(),
                    sample_window_days = EXCLUDED.sample_window_days
                """,
                dimension,
                dim_value,
                total_seq,
                positive,
                r["deals_won"],
                float(r["total_revenue"]),
                round(pos_rate, 4),
                round(baseline_rate, 4),
                round(lift, 4),
                round(weight_adjustment, 2),
                static_default,
                window_days,
                new_version,
            )
            dim_weight_count += 1

        total_weights += dim_weight_count
        dimensions_calibrated.append(dimension)

    logger.info(
        "Score calibration complete: version=%d, weights=%d, dimensions=%s",
        new_version, total_weights, dimensions_calibrated,
    )

    return {
        "calibrated": True,
        "model_version": new_version,
        "total_weights": total_weights,
        "dimensions_calibrated": dimensions_calibrated,
        "baseline_rate": round(baseline_rate, 4),
        "total_with_outcomes": total_with_outcomes,
        "sample_window_days": window_days,
    }
