"""B2B Churn MCP -- displacement graph tools."""
import json
from typing import Optional

from ._shared import _safe_json, _suppress_predicate, logger, get_pool
from .server import mcp


@mcp.tool()
async def list_displacement_edges(
    from_vendor: Optional[str] = None,
    to_vendor: Optional[str] = None,
    min_strength: Optional[str] = None,
    min_confidence: Optional[float] = None,
    window_days: int = 90,
    limit: int = 50,
) -> str:
    """
    Query persisted competitive displacement edges (vendor A -> vendor B flows).

    from_vendor: Filter by source vendor losing customers (case-insensitive partial match)
    to_vendor: Filter by destination vendor gaining customers (case-insensitive partial match)
    min_strength: Minimum signal strength: 'strong', 'moderate', or 'emerging'
    min_confidence: Minimum confidence score (0.0-1.0)
    window_days: Only edges computed within this many days (default 90)
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    strength_order = {"strong": 3, "moderate": 2, "emerging": 1}
    if min_strength and min_strength not in strength_order:
        return json.dumps({"error": f"Invalid min_strength: {min_strength}. Use strong/moderate/emerging"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions = ["computed_date > NOW() - make_interval(days => $1)"]
        params: list = [window_days]
        idx = 2

        if from_vendor:
            conditions.append(f"from_vendor ILIKE '%' || ${idx} || '%'")
            params.append(from_vendor)
            idx += 1

        if to_vendor:
            conditions.append(f"to_vendor ILIKE '%' || ${idx} || '%'")
            params.append(to_vendor)
            idx += 1

        if min_strength:
            min_val = strength_order[min_strength]
            allowed = [k for k, v in strength_order.items() if v >= min_val]
            conditions.append(f"signal_strength = ANY(${idx}::text[])")
            params.append(allowed)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        conditions.append(_suppress_predicate('displacement_edge'))
        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT id, from_vendor, to_vendor, mention_count,
                   primary_driver, signal_strength, key_quote,
                   source_distribution, sample_review_ids,
                   confidence_score, computed_date, report_id, created_at
            FROM b2b_displacement_edges
            WHERE {where}
            ORDER BY confidence_score DESC, mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        edges = []
        for r in rows:
            edges.append({
                "id": str(r["id"]),
                "from_vendor": r["from_vendor"],
                "to_vendor": r["to_vendor"],
                "mention_count": r["mention_count"],
                "primary_driver": r["primary_driver"],
                "signal_strength": r["signal_strength"],
                "key_quote": r["key_quote"],
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "computed_date": str(r["computed_date"]),
                "report_id": str(r["report_id"]) if r["report_id"] else None,
                "created_at": str(r["created_at"]),
            })

        return json.dumps({"edges": edges, "count": len(edges)}, default=str)
    except Exception:
        logger.exception("list_displacement_edges error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def get_displacement_history(
    from_vendor: str,
    to_vendor: str,
    window_days: int = 365,
) -> str:
    """
    Time-series of displacement edge strength for a specific vendor pair.

    from_vendor: Source vendor (exact match, case-insensitive)
    to_vendor: Destination vendor (exact match, case-insensitive)
    window_days: How far back to look (default 365)
    """
    if not from_vendor or not to_vendor:
        return json.dumps({"error": "Both from_vendor and to_vendor are required"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        rows = await pool.fetch(
            f"""
            SELECT computed_date, mention_count, signal_strength,
                   confidence_score, primary_driver, key_quote
            FROM b2b_displacement_edges
            WHERE LOWER(from_vendor) = LOWER($1)
              AND LOWER(to_vendor) = LOWER($2)
              AND computed_date > NOW() - make_interval(days => $3)
              AND {_suppress_predicate('displacement_edge')}
            ORDER BY computed_date ASC
            """,
            from_vendor,
            to_vendor,
            window_days,
        )

        history = []
        for r in rows:
            history.append({
                "computed_date": str(r["computed_date"]),
                "mention_count": r["mention_count"],
                "signal_strength": r["signal_strength"],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "primary_driver": r["primary_driver"],
                "key_quote": r["key_quote"],
            })

        return json.dumps({
            "from_vendor": from_vendor,
            "to_vendor": to_vendor,
            "window_days": window_days,
            "history": history,
            "data_points": len(history),
        }, default=str)
    except Exception:
        logger.exception("get_displacement_history error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_pain_points(
    vendor_name: Optional[str] = None,
    pain_category: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_mentions: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query vendor pain points aggregated from review intelligence.

    vendor_name: Filter by vendor (case-insensitive partial match)
    pain_category: Exact pain category (pricing, support, features, ux, reliability, performance, integration, security, onboarding, other)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_mentions: Minimum mention count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if pain_category:
            conditions.append(f"pain_category = ${idx}")
            params.append(pain_category)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_mentions is not None:
            conditions.append(f"mention_count >= ${idx}")
            params.append(min_mentions)
            idx += 1

        conditions.append(_suppress_predicate('pain_point'))
        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, pain_category, mention_count,
                   primary_count, secondary_count, minor_count,
                   avg_urgency, avg_rating,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_pain_points
            {where}
            ORDER BY mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        items = []
        for r in rows:
            items.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "pain_category": r["pain_category"],
                "mention_count": r["mention_count"],
                "primary_count": r["primary_count"],
                "secondary_count": r["secondary_count"],
                "minor_count": r["minor_count"],
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
                "avg_rating": float(r["avg_rating"]) if r["avg_rating"] else 0,
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"pain_points": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_vendor_pain_points error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_use_cases(
    vendor_name: Optional[str] = None,
    use_case_name: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_mentions: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query vendor use cases (modules/features mentioned in reviews).

    vendor_name: Filter by vendor (case-insensitive partial match)
    use_case_name: Filter by use case name (case-insensitive partial match)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_mentions: Minimum mention count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if use_case_name:
            conditions.append(f"use_case_name ILIKE '%' || ${idx} || '%'")
            params.append(use_case_name)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_mentions is not None:
            conditions.append(f"mention_count >= ${idx}")
            params.append(min_mentions)
            idx += 1

        conditions.append(_suppress_predicate('use_case'))
        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, use_case_name, mention_count,
                   avg_urgency, lock_in_distribution,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_use_cases
            {where}
            ORDER BY mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        items = []
        for r in rows:
            items.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "use_case_name": r["use_case_name"],
                "mention_count": r["mention_count"],
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
                "lock_in_distribution": _safe_json(r["lock_in_distribution"]),
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"use_cases": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_vendor_use_cases error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_integrations(
    vendor_name: Optional[str] = None,
    integration_name: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_mentions: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query vendor integrations (tools/services mentioned in reviews).

    vendor_name: Filter by vendor (case-insensitive partial match)
    integration_name: Filter by integration name (case-insensitive partial match)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_mentions: Minimum mention count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if integration_name:
            conditions.append(f"integration_name ILIKE '%' || ${idx} || '%'")
            params.append(integration_name)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_mentions is not None:
            conditions.append(f"mention_count >= ${idx}")
            params.append(min_mentions)
            idx += 1

        conditions.append(_suppress_predicate('integration'))
        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, integration_name, mention_count,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_integrations
            {where}
            ORDER BY mention_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        items = []
        for r in rows:
            items.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "integration_name": r["integration_name"],
                "mention_count": r["mention_count"],
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"integrations": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_vendor_integrations error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def list_vendor_buyer_profiles(
    vendor_name: Optional[str] = None,
    role_type: Optional[str] = None,
    buying_stage: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_reviews: Optional[int] = None,
    limit: int = 50,
) -> str:
    """
    Query aggregated buyer authority profiles per vendor.

    vendor_name: Filter by vendor (case-insensitive partial match)
    role_type: Filter by role type (economic_buyer, champion, evaluator, end_user, unknown)
    buying_stage: Filter by buying stage (active_purchase, evaluation, renewal_decision, post_purchase, unknown)
    min_confidence: Minimum confidence score (0.0-1.0)
    min_reviews: Minimum review count
    limit: Max results (default 50, max 200)
    """
    limit = min(max(limit, 1), 200)
    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name)
            idx += 1

        if role_type:
            conditions.append(f"role_type = ${idx}")
            params.append(role_type)
            idx += 1

        if buying_stage:
            conditions.append(f"buying_stage = ${idx}")
            params.append(buying_stage)
            idx += 1

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${idx}")
            params.append(min_confidence)
            idx += 1

        if min_reviews is not None:
            conditions.append(f"review_count >= ${idx}")
            params.append(min_reviews)
            idx += 1

        conditions.append(_suppress_predicate('buyer_profile'))
        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = await pool.fetch(
            f"""
            SELECT id, vendor_name, role_type, buying_stage,
                   review_count, dm_count, avg_urgency,
                   source_distribution, sample_review_ids,
                   confidence_score, first_seen_at, last_seen_at
            FROM b2b_vendor_buyer_profiles
            {where}
            ORDER BY review_count DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )

        profiles = []
        for r in rows:
            profiles.append({
                "id": str(r["id"]),
                "vendor_name": r["vendor_name"],
                "role_type": r["role_type"],
                "buying_stage": r["buying_stage"],
                "review_count": r["review_count"],
                "dm_count": r["dm_count"],
                "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] is not None else None,
                "source_distribution": _safe_json(r["source_distribution"]),
                "sample_review_ids": [str(rid) for rid in (r["sample_review_ids"] or [])],
                "confidence_score": float(r["confidence_score"]) if r["confidence_score"] else 0,
                "first_seen_at": str(r["first_seen_at"]) if r["first_seen_at"] else None,
                "last_seen_at": str(r["last_seen_at"]) if r["last_seen_at"] else None,
            })

        return json.dumps({"profiles": profiles, "count": len(profiles)}, default=str)
    except Exception:
        logger.exception("list_vendor_buyer_profiles error")
        return json.dumps({"error": "Internal error"})
