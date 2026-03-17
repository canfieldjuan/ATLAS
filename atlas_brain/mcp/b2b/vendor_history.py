"""B2B Churn MCP -- vendor history and change event tools."""
import json
from typing import Optional

from ._shared import _safe_json, logger, get_pool
from .server import mcp


@mcp.tool()
async def get_vendor_history(
    vendor_name: str,
    days: int = 90,
    limit: int = 90,
) -> str:
    """
    Get daily health snapshots for a vendor over time.

    vendor_name: Vendor name (case-insensitive partial match)
    days: How many days back to look (default 90)
    limit: Max snapshots to return (default 90, max 365)
    """
    limit = min(max(limit, 1), 365)
    days = min(max(days, 1), 365)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        # suppression: N/A -- b2b_vendor_snapshots is a materialized derived table;
        # rows have no independent entity UUID in data_corrections.
        rows = await pool.fetch(
            """
            SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
                   churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                   top_pain, top_competitor, pain_count, competitor_count,
                   displacement_edge_count, high_intent_company_count,
                   pressure_score, dm_churn_rate, price_complaint_rate,
                   archetype, archetype_confidence
            FROM b2b_vendor_snapshots
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND snapshot_date >= CURRENT_DATE - $2::int
            ORDER BY snapshot_date DESC
            LIMIT $3
            """,
            vendor_name, days, limit,
        )

        resolved = rows[0]["vendor_name"] if rows else vendor_name
        snapshots = []
        for r in rows:
            snapshots.append({
                "snapshot_date": str(r["snapshot_date"]),
                "total_reviews": r["total_reviews"],
                "churn_intent": r["churn_intent"],
                "churn_density": float(r["churn_density"]),
                "avg_urgency": float(r["avg_urgency"]),
                "positive_review_pct": float(r["positive_review_pct"]) if r["positive_review_pct"] is not None else None,
                "recommend_ratio": float(r["recommend_ratio"]) if r["recommend_ratio"] is not None else None,
                "top_pain": r["top_pain"],
                "top_competitor": r["top_competitor"],
                "pain_count": r["pain_count"],
                "competitor_count": r["competitor_count"],
                "displacement_edge_count": r["displacement_edge_count"],
                "high_intent_company_count": r["high_intent_company_count"],
                "pressure_score": float(r["pressure_score"]) if r["pressure_score"] is not None else None,
                "dm_churn_rate": float(r["dm_churn_rate"]) if r["dm_churn_rate"] is not None else None,
                "price_complaint_rate": float(r["price_complaint_rate"]) if r["price_complaint_rate"] is not None else None,
                "archetype": r["archetype"],
                "archetype_confidence": float(r["archetype_confidence"]) if r["archetype_confidence"] is not None else None,
            })

        return json.dumps({"vendor_name": resolved, "snapshots": snapshots, "count": len(snapshots)}, default=str)
    except Exception:
        logger.exception("get_vendor_history error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_change_events(
    vendor_name: Optional[str] = None,
    event_type: Optional[str] = None,
    days: int = 30,
    limit: int = 50,
) -> str:
    """
    List structural change events detected across vendors.

    vendor_name: Filter by vendor (case-insensitive partial match)
    event_type: Filter by event type (urgency_spike, urgency_drop, churn_density_spike, nps_shift, new_pain_category, new_competitor, review_volume_spike)
    days: How many days back to look (default 30)
    limit: Max events to return (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    days = min(max(days, 1), 365)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        # suppression: intentionally excluded -- b2b_change_events are structural signals,
        # not primary data entities; no change_event entity_type in data_corrections.
        conditions: list[str] = ["event_date >= CURRENT_DATE - $1::int"]
        params: list = [days]
        idx = 2

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if event_type:
            conditions.append(f"event_type = ${idx}")
            params.append(event_type)
            idx += 1

        where = "WHERE " + " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT vendor_name, event_date, event_type, description,
                   old_value, new_value, delta, metadata,
                   z_score, severity
            FROM b2b_change_events
            {where}
            ORDER BY event_date DESC, created_at DESC
            LIMIT ${idx}
            """,
            *params, limit,
        )

        events = []
        for r in rows:
            entry = {
                "vendor_name": r["vendor_name"],
                "event_date": str(r["event_date"]),
                "event_type": r["event_type"],
                "description": r["description"],
                "old_value": float(r["old_value"]) if r["old_value"] is not None else None,
                "new_value": float(r["new_value"]) if r["new_value"] is not None else None,
                "delta": float(r["delta"]) if r["delta"] is not None else None,
                "metadata": _safe_json(r["metadata"]),
            }
            if r["z_score"] is not None:
                entry["z_score"] = round(float(r["z_score"]), 2)
            if r["severity"] is not None:
                entry["severity"] = r["severity"]
            events.append(entry)

        return json.dumps({"events": events, "count": len(events)}, default=str)
    except Exception:
        logger.exception("list_change_events error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def compare_vendor_periods(
    vendor_name: str,
    period_a_days_ago: int = 30,
    period_b_days_ago: int = 0,
) -> str:
    """
    Compare a vendor's health between two dates.

    vendor_name: Vendor name (case-insensitive partial match)
    period_a_days_ago: The older snapshot (days ago, default 30)
    period_b_days_ago: The newer snapshot (days ago, default 0 = today)
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        # suppression: N/A -- b2b_vendor_snapshots is a materialized derived table;
        # rows have no independent entity UUID in data_corrections.
        async def _nearest_snapshot(target_days_ago: int):
            return await pool.fetchrow(
                """
                SELECT vendor_name, snapshot_date, total_reviews, churn_intent,
                       churn_density, avg_urgency, positive_review_pct, recommend_ratio,
                       top_pain, top_competitor, pain_count, competitor_count,
                       displacement_edge_count, high_intent_company_count,
                       pressure_score, dm_churn_rate, price_complaint_rate,
                       archetype, archetype_confidence
                FROM b2b_vendor_snapshots
                WHERE vendor_name ILIKE '%' || $1 || '%'
                  AND snapshot_date <= CURRENT_DATE - $2::int
                ORDER BY snapshot_date DESC
                LIMIT 1
                """,
                vendor_name, target_days_ago,
            )

        snap_a = await _nearest_snapshot(period_a_days_ago)
        snap_b = await _nearest_snapshot(period_b_days_ago)

        if not snap_a and not snap_b:
            return json.dumps({"error": f"No snapshots found for vendor matching '{vendor_name}'"})

        def _format(snap):
            if not snap:
                return None
            return {
                "snapshot_date": str(snap["snapshot_date"]),
                "total_reviews": snap["total_reviews"],
                "churn_intent": snap["churn_intent"],
                "churn_density": float(snap["churn_density"]),
                "avg_urgency": float(snap["avg_urgency"]),
                "positive_review_pct": float(snap["positive_review_pct"]) if snap["positive_review_pct"] is not None else None,
                "recommend_ratio": float(snap["recommend_ratio"]) if snap["recommend_ratio"] is not None else None,
                "top_pain": snap["top_pain"],
                "top_competitor": snap["top_competitor"],
                "pain_count": snap["pain_count"],
                "competitor_count": snap["competitor_count"],
                "displacement_edge_count": snap["displacement_edge_count"],
                "high_intent_company_count": snap["high_intent_company_count"],
                "pressure_score": float(snap["pressure_score"]) if snap["pressure_score"] is not None else None,
                "dm_churn_rate": float(snap["dm_churn_rate"]) if snap["dm_churn_rate"] is not None else None,
                "price_complaint_rate": float(snap["price_complaint_rate"]) if snap["price_complaint_rate"] is not None else None,
                "archetype": snap["archetype"],
                "archetype_confidence": float(snap["archetype_confidence"]) if snap["archetype_confidence"] is not None else None,
            }

        a_fmt = _format(snap_a)
        b_fmt = _format(snap_b)

        deltas = {}
        if a_fmt and b_fmt:
            for key in ("churn_density", "avg_urgency", "recommend_ratio", "total_reviews",
                        "churn_intent", "pain_count", "competitor_count",
                        "displacement_edge_count", "high_intent_company_count"):
                a_val = a_fmt.get(key)
                b_val = b_fmt.get(key)
                if a_val is not None and b_val is not None:
                    deltas[key] = round(b_val - a_val, 2)

        resolved = (snap_a or snap_b)["vendor_name"]
        return json.dumps({
            "vendor_name": resolved,
            "period_a": a_fmt,
            "period_b": b_fmt,
            "deltas": deltas,
        }, default=str)
    except Exception:
        logger.exception("compare_vendor_periods error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_concurrent_events(
    days: int = 30,
    event_type: Optional[str] = None,
    min_vendors: int = 2,
    limit: int = 50,
) -> str:
    """
    Find dates where multiple vendors had the same change event type.

    Surfaces cross-vendor correlations like 'urgency spiked at 4 vendors on
    the same day' -- may indicate market-level trends vs vendor-specific issues.

    days: Lookback period (default 30)
    event_type: Optional filter (urgency_spike, churn_density_spike, nps_shift, etc.)
    min_vendors: Minimum vendor count to qualify as concurrent (default 2)
    limit: Max results (default 50)
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        type_filter = ""
        params: list = [days, min_vendors, limit]
        if event_type:
            type_filter = "AND event_type = $4"
            params.append(event_type)

        rows = await pool.fetch(
            f"""
            SELECT event_date, event_type,
                   COUNT(DISTINCT vendor_name) AS vendor_count,
                   ARRAY_AGG(DISTINCT vendor_name ORDER BY vendor_name) AS vendors,
                   AVG(delta) AS avg_delta,
                   MIN(delta) AS min_delta,
                   MAX(delta) AS max_delta
            FROM b2b_change_events
            WHERE event_date >= CURRENT_DATE - $1::int
              {type_filter}
            GROUP BY event_date, event_type
            HAVING COUNT(DISTINCT vendor_name) >= $2
            ORDER BY vendor_count DESC, event_date DESC
            LIMIT $3
            """,
            *params,
        )

        results = [
            {
                "event_date": str(r["event_date"]),
                "event_type": r["event_type"],
                "vendor_count": r["vendor_count"],
                "vendors": r["vendors"],
                "avg_delta": round(float(r["avg_delta"]), 2) if r["avg_delta"] is not None else None,
                "min_delta": round(float(r["min_delta"]), 2) if r["min_delta"] is not None else None,
                "max_delta": round(float(r["max_delta"]), 2) if r["max_delta"] is not None else None,
            }
            for r in rows
        ]
        return json.dumps({"concurrent_events": results, "total": len(results)}, default=str)
    except Exception:
        logger.exception("list_concurrent_events error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def get_vendor_correlation(
    vendor_a: str,
    vendor_b: str,
    days: int = 90,
    metric: str = "churn_density",
) -> str:
    """
    Compare two vendors' metric trends and compute correlation coefficient.

    Returns aligned time-series and Pearson r. Negative correlation (r < -0.5)
    suggests one vendor gains when the other loses -- potential displacement.

    vendor_a: First vendor name (partial match)
    vendor_b: Second vendor name (partial match)
    days: Lookback period (default 90)
    metric: Metric to correlate (churn_density, avg_urgency, recommend_ratio,
            total_reviews, displacement_edge_count, high_intent_company_count)
    """
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        valid_metrics = {
            "churn_density", "avg_urgency", "recommend_ratio", "total_reviews",
            "displacement_edge_count", "high_intent_company_count", "pain_count",
            "competitor_count",
        }
        if metric not in valid_metrics:
            return json.dumps({"error": f"metric must be one of: {sorted(valid_metrics)}"})

        rows = await pool.fetch(
            f"""
            SELECT a.snapshot_date,
                   a.{metric} AS value_a,
                   b.{metric} AS value_b
            FROM b2b_vendor_snapshots a
            JOIN b2b_vendor_snapshots b
              ON a.snapshot_date = b.snapshot_date
            WHERE a.vendor_name ILIKE '%' || $1 || '%'
              AND b.vendor_name ILIKE '%' || $2 || '%'
              AND a.snapshot_date >= CURRENT_DATE - $3::int
            ORDER BY a.snapshot_date ASC
            """,
            vendor_a, vendor_b, days,
        )

        if not rows:
            return json.dumps({"error": "No overlapping snapshots found for these vendors"})

        vals_a = [float(r["value_a"] or 0) for r in rows]
        vals_b = [float(r["value_b"] or 0) for r in rows]

        # Pearson correlation
        n = len(vals_a)
        correlation = None
        if n >= 3:
            mean_a = sum(vals_a) / n
            mean_b = sum(vals_b) / n
            dx = [v - mean_a for v in vals_a]
            dy = [v - mean_b for v in vals_b]
            num = sum(a * b for a, b in zip(dx, dy))
            den_a = sum(a * a for a in dx) ** 0.5
            den_b = sum(b * b for b in dy) ** 0.5
            if den_a > 0 and den_b > 0:
                correlation = round(num / (den_a * den_b), 4)

        series = [
            {
                "date": str(r["snapshot_date"]),
                "value_a": float(r["value_a"] or 0),
                "value_b": float(r["value_b"] or 0),
            }
            for r in rows
        ]

        # Recent displacement edges between the pair
        edge_rows = await pool.fetch(
            """
            SELECT from_vendor, to_vendor, mention_count, signal_strength, primary_driver
            FROM b2b_displacement_edges
            WHERE (from_vendor ILIKE '%' || $1 || '%' AND to_vendor ILIKE '%' || $2 || '%')
               OR (from_vendor ILIKE '%' || $2 || '%' AND to_vendor ILIKE '%' || $1 || '%')
            ORDER BY computed_date DESC
            LIMIT 5
            """,
            vendor_a, vendor_b,
        )
        displacement = [
            {
                "from_vendor": r["from_vendor"],
                "to_vendor": r["to_vendor"],
                "mention_count": r["mention_count"],
                "signal_strength": r["signal_strength"],
                "primary_driver": r["primary_driver"],
            }
            for r in edge_rows
        ]

        return json.dumps({
            "vendor_a": vendor_a,
            "vendor_b": vendor_b,
            "metric": metric,
            "data_points": len(series),
            "correlation": correlation,
            "series": series,
            "displacement_edges": displacement,
        }, default=str)
    except Exception:
        logger.exception("get_vendor_correlation error")
        return json.dumps({"error": "Internal error"})
