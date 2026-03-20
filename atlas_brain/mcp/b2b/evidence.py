"""B2B Churn MCP -- evidence vault, segment & temporal intelligence pool tools."""

import json
from datetime import date
from typing import Optional

from ._shared import _safe_json, get_pool, logger
from .server import mcp


# -------------------------------------------------------------------
# Evidence Vault
# -------------------------------------------------------------------


@mcp.tool()
async def get_evidence_vault(
    vendor_name: str,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
) -> str:
    """Get the latest evidence vault for a vendor.

    Returns canonical weakness/strength evidence, company signals,
    metric snapshot, and provenance for downstream report building.

    vendor_name: Vendor to look up (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})

    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT vendor_name, as_of_date, analysis_window_days,
                   schema_version, vault, created_at
            FROM b2b_evidence_vault
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            vendor_name.strip(),
            target_date,
            window_days,
        )

        if not row:
            return json.dumps({"success": False, "error": "No evidence vault found for that vendor"})

        vault = _safe_json(row["vault"])
        if not isinstance(vault, dict):
            vault = {}

        return json.dumps({
            "success": True,
            "vendor_name": row["vendor_name"],
            "as_of_date": str(row["as_of_date"]),
            "analysis_window_days": row["analysis_window_days"],
            "schema_version": row["schema_version"],
            "created_at": str(row["created_at"]),
            "weakness_evidence": vault.get("weakness_evidence", []),
            "strength_evidence": vault.get("strength_evidence", []),
            "company_signals": vault.get("company_signals", []),
            "metric_snapshot": vault.get("metric_snapshot", {}),
            "provenance": vault.get("provenance", {}),
        }, default=str)
    except Exception:
        logger.exception("get_evidence_vault error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_evidence_vaults(
    vendor_name: Optional[str] = None,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """List evidence vault summaries across vendors (latest per vendor).

    Returns metric snapshot, provenance sources, and evidence counts
    without the full weakness/strength/company arrays.

    vendor_name: Filter by vendor name (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        conditions = [
            "as_of_date <= $1",
            "analysis_window_days = $2",
        ]
        params: list = [target_date, window_days]
        idx = 3

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name.strip())
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT DISTINCT ON (vendor_name)
                   vendor_name, as_of_date, schema_version, vault, created_at
            FROM b2b_evidence_vault
            WHERE {where}
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
            """,
            *params,
        )

        rows = rows[:limit]

        vaults = []
        for r in rows:
            vault = _safe_json(r["vault"])
            if not isinstance(vault, dict):
                vault = {}
            ms = vault.get("metric_snapshot") or {}
            prov = vault.get("provenance") or {}
            vaults.append({
                "vendor_name": r["vendor_name"],
                "as_of_date": str(r["as_of_date"]),
                "schema_version": r["schema_version"],
                "created_at": str(r["created_at"]),
                "weakness_count": len(vault.get("weakness_evidence") or []),
                "strength_count": len(vault.get("strength_evidence") or []),
                "company_signal_count": len(vault.get("company_signals") or []),
                "total_reviews": ms.get("total_reviews"),
                "churn_density": ms.get("churn_density"),
                "avg_urgency": ms.get("avg_urgency"),
                "recommend_ratio": ms.get("recommend_ratio"),
                "displacement_mention_count": ms.get("displacement_mention_count"),
                "sources": prov.get("sources", []),
            })

        vaults.sort(key=lambda v: v.get("avg_urgency") or 0, reverse=True)

        return json.dumps({"vaults": vaults, "count": len(vaults)}, default=str)
    except Exception:
        logger.exception("list_evidence_vaults error")
        return json.dumps({"error": "Internal error", "vaults": [], "count": 0})


# -------------------------------------------------------------------
# Segment Intelligence
# -------------------------------------------------------------------


@mcp.tool()
async def get_segment_intelligence(
    vendor_name: str,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
) -> str:
    """Get the latest buyer segment intelligence for a vendor.

    Returns affected roles, departments, company sizes, budget pressure,
    contract segments, usage duration, use cases under pressure, and
    buying stage distribution.

    vendor_name: Vendor to look up (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})

    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT vendor_name, as_of_date, analysis_window_days,
                   schema_version, segments, created_at
            FROM b2b_segment_intelligence
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            vendor_name.strip(),
            target_date,
            window_days,
        )

        if not row:
            return json.dumps({"success": False, "error": "No segment intelligence found for that vendor"})

        seg = _safe_json(row["segments"])
        if not isinstance(seg, dict):
            seg = {}

        return json.dumps({
            "success": True,
            "vendor_name": row["vendor_name"],
            "as_of_date": str(row["as_of_date"]),
            "analysis_window_days": row["analysis_window_days"],
            "schema_version": row["schema_version"],
            "created_at": str(row["created_at"]),
            "affected_roles": seg.get("affected_roles", []),
            "affected_departments": seg.get("affected_departments", []),
            "affected_company_sizes": seg.get("affected_company_sizes", {}),
            "budget_pressure": seg.get("budget_pressure", {}),
            "contract_segments": seg.get("contract_segments", []),
            "usage_duration_segments": seg.get("usage_duration_segments", []),
            "top_use_cases_under_pressure": seg.get("top_use_cases_under_pressure", []),
            "buying_stage_distribution": seg.get("buying_stage_distribution", []),
            "best_fit_challenger_segments": seg.get("best_fit_challenger_segments"),
        }, default=str)
    except Exception:
        logger.exception("get_segment_intelligence error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_segment_intelligence(
    vendor_name: Optional[str] = None,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """List buyer segment intelligence summaries across vendors (latest per vendor).

    Returns role/department counts and budget pressure headline without
    the full segment arrays.

    vendor_name: Filter by vendor name (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        conditions = [
            "as_of_date <= $1",
            "analysis_window_days = $2",
        ]
        params: list = [target_date, window_days]
        idx = 3

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name.strip())
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT DISTINCT ON (vendor_name)
                   vendor_name, as_of_date, schema_version, segments, created_at
            FROM b2b_segment_intelligence
            WHERE {where}
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
            """,
            *params,
        )

        rows = rows[:limit]

        segments = []
        for r in rows:
            seg = _safe_json(r["segments"])
            if not isinstance(seg, dict):
                seg = {}
            bp = seg.get("budget_pressure") or {}
            roles = seg.get("affected_roles") or []
            depts = seg.get("affected_departments") or []
            stages = seg.get("buying_stage_distribution") or []
            segments.append({
                "vendor_name": r["vendor_name"],
                "as_of_date": str(r["as_of_date"]),
                "schema_version": r["schema_version"],
                "created_at": str(r["created_at"]),
                "role_count": len(roles),
                "department_count": len(depts),
                "top_role": roles[0]["role_type"] if roles else None,
                "top_department": depts[0]["department"] if depts else None,
                "top_buying_stage": stages[0]["stage"] if stages else None,
                "price_increase_rate": bp.get("price_increase_rate"),
                "dm_churn_rate": bp.get("dm_churn_rate"),
                "contract_segment_count": len(seg.get("contract_segments") or []),
                "use_case_count": len(seg.get("top_use_cases_under_pressure") or []),
            })

        return json.dumps({"segments": segments, "count": len(segments)}, default=str)
    except Exception:
        logger.exception("list_segment_intelligence error")
        return json.dumps({"error": "Internal error", "segments": [], "count": 0})


# -------------------------------------------------------------------
# Temporal Intelligence
# -------------------------------------------------------------------


@mcp.tool()
async def get_temporal_intelligence(
    vendor_name: str,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
) -> str:
    """Get the latest temporal intelligence for a vendor.

    Returns immediate triggers (deadline-driven accounts), evaluation
    deadlines, keyword spikes, sentiment trajectory, and timeline
    signal summary.

    vendor_name: Vendor to look up (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})

    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT vendor_name, as_of_date, analysis_window_days,
                   schema_version, temporal, created_at
            FROM b2b_temporal_intelligence
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            vendor_name.strip(),
            target_date,
            window_days,
        )

        if not row:
            return json.dumps({"success": False, "error": "No temporal intelligence found for that vendor"})

        temp = _safe_json(row["temporal"])
        if not isinstance(temp, dict):
            temp = {}

        return json.dumps({
            "success": True,
            "vendor_name": row["vendor_name"],
            "as_of_date": str(row["as_of_date"]),
            "analysis_window_days": row["analysis_window_days"],
            "schema_version": row["schema_version"],
            "created_at": str(row["created_at"]),
            "immediate_triggers": temp.get("immediate_triggers", []),
            "evaluation_deadlines": temp.get("evaluation_deadlines", []),
            "keyword_spikes": temp.get("keyword_spikes", {}),
            "sentiment_trajectory": temp.get("sentiment_trajectory", {}),
            "timeline_signal_summary": temp.get("timeline_signal_summary", {}),
            "trend_per_weakness": temp.get("trend_per_weakness"),
            "acceleration_metrics": temp.get("acceleration_metrics"),
        }, default=str)
    except Exception:
        logger.exception("get_temporal_intelligence error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_temporal_intelligence(
    vendor_name: Optional[str] = None,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """List temporal intelligence summaries across vendors (latest per vendor).

    Returns trigger/deadline counts, spike count, and sentiment
    trajectory headline without the full arrays.

    vendor_name: Filter by vendor name (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        conditions = [
            "as_of_date <= $1",
            "analysis_window_days = $2",
        ]
        params: list = [target_date, window_days]
        idx = 3

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name.strip())
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT DISTINCT ON (vendor_name)
                   vendor_name, as_of_date, schema_version, temporal, created_at
            FROM b2b_temporal_intelligence
            WHERE {where}
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
            """,
            *params,
        )

        rows = rows[:limit]

        items = []
        for r in rows:
            temp = _safe_json(r["temporal"])
            if not isinstance(temp, dict):
                temp = {}
            st = temp.get("sentiment_trajectory") or {}
            ks = temp.get("keyword_spikes") or {}
            tss = temp.get("timeline_signal_summary") or {}
            items.append({
                "vendor_name": r["vendor_name"],
                "as_of_date": str(r["as_of_date"]),
                "schema_version": r["schema_version"],
                "created_at": str(r["created_at"]),
                "trigger_count": len(temp.get("immediate_triggers") or []),
                "deadline_count": len(temp.get("evaluation_deadlines") or []),
                "spike_count": ks.get("spike_count", 0),
                "spike_keywords": ks.get("spike_keywords", []),
                "sentiment_declining_pct": st.get("declining_pct"),
                "sentiment_improving_pct": st.get("improving_pct"),
                "total_timeline_signals": tss.get("total_signals", 0),
            })

        # Sort by trigger count desc -- vendors with most active deadlines first
        items.sort(key=lambda v: v.get("trigger_count") or 0, reverse=True)

        return json.dumps({"temporal": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_temporal_intelligence error")
        return json.dumps({"error": "Internal error", "temporal": [], "count": 0})


# -------------------------------------------------------------------
# Displacement Dynamics
# -------------------------------------------------------------------


@mcp.tool()
async def get_displacement_dynamics(
    from_vendor: str,
    to_vendor: str,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
) -> str:
    """Get the latest displacement dynamics for a vendor pair.

    Returns edge metrics (mention count, velocity, signal strength),
    evidence breakdown by type, switch reasons, battle summary, and
    flow summary for a specific from->to competitive displacement.

    from_vendor: Incumbent vendor losing customers (partial match)
    to_vendor: Challenger vendor gaining customers (partial match)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    """
    if not from_vendor or not from_vendor.strip():
        return json.dumps({"success": False, "error": "from_vendor is required"})
    if not to_vendor or not to_vendor.strip():
        return json.dumps({"success": False, "error": "to_vendor is required"})

    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT from_vendor, to_vendor, as_of_date, analysis_window_days,
                   schema_version, dynamics, created_at
            FROM b2b_displacement_dynamics
            WHERE from_vendor ILIKE '%' || $1 || '%'
              AND to_vendor ILIKE '%' || $2 || '%'
              AND as_of_date <= $3
              AND analysis_window_days = $4
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            from_vendor.strip(),
            to_vendor.strip(),
            target_date,
            window_days,
        )

        if not row:
            return json.dumps({"success": False, "error": "No displacement dynamics found for that pair"})

        dyn = _safe_json(row["dynamics"])
        if not isinstance(dyn, dict):
            dyn = {}

        return json.dumps({
            "success": True,
            "from_vendor": row["from_vendor"],
            "to_vendor": row["to_vendor"],
            "as_of_date": str(row["as_of_date"]),
            "analysis_window_days": row["analysis_window_days"],
            "schema_version": row["schema_version"],
            "created_at": str(row["created_at"]),
            "edge_metrics": dyn.get("edge_metrics", {}),
            "evidence_breakdown": dyn.get("evidence_breakdown", []),
            "switch_reasons": dyn.get("switch_reasons", []),
            "battle_summary": dyn.get("battle_summary"),
            "flow_summary": dyn.get("flow_summary", {}),
            "segment_displacement": dyn.get("segment_displacement"),
            "trend_acceleration": dyn.get("trend_acceleration"),
        }, default=str)
    except Exception:
        logger.exception("get_displacement_dynamics error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_displacement_dynamics(
    from_vendor: Optional[str] = None,
    to_vendor: Optional[str] = None,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
    min_mentions: int = 0,
    limit: int = 20,
) -> str:
    """List displacement dynamics summaries across vendor pairs (latest per pair).

    Returns edge metrics headline and flow summary without the full
    evidence/reason arrays. Useful for scanning all competitive flows
    involving a vendor.

    from_vendor: Filter incumbent vendor (partial match, case-insensitive)
    to_vendor: Filter challenger vendor (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    min_mentions: Minimum mention count to include (default 0)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    min_mentions = max(0, min_mentions)
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        conditions = [
            "as_of_date <= $1",
            "analysis_window_days = $2",
        ]
        params: list = [target_date, window_days]
        idx = 3

        if from_vendor:
            conditions.append(f"from_vendor ILIKE '%' || ${idx} || '%'")
            params.append(from_vendor.strip())
            idx += 1

        if to_vendor:
            conditions.append(f"to_vendor ILIKE '%' || ${idx} || '%'")
            params.append(to_vendor.strip())
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT DISTINCT ON (from_vendor, to_vendor)
                   from_vendor, to_vendor, as_of_date, schema_version,
                   dynamics, created_at
            FROM b2b_displacement_dynamics
            WHERE {where}
            ORDER BY from_vendor, to_vendor, as_of_date DESC, created_at DESC
            """,
            *params,
        )

        items = []
        for r in rows:
            dyn = _safe_json(r["dynamics"])
            if not isinstance(dyn, dict):
                dyn = {}
            em = dyn.get("edge_metrics") or {}
            fs = dyn.get("flow_summary") or {}

            mention_count = em.get("mention_count") or 0
            if mention_count < min_mentions:
                continue

            items.append({
                "from_vendor": r["from_vendor"],
                "to_vendor": r["to_vendor"],
                "as_of_date": str(r["as_of_date"]),
                "schema_version": r["schema_version"],
                "created_at": str(r["created_at"]),
                "mention_count": mention_count,
                "signal_strength": em.get("signal_strength"),
                "primary_driver": em.get("primary_driver"),
                "confidence_score": em.get("confidence_score"),
                "velocity_7d": em.get("velocity_7d"),
                "velocity_30d": em.get("velocity_30d"),
                "explicit_switch_count": fs.get("explicit_switch_count", 0),
                "active_evaluation_count": fs.get("active_evaluation_count", 0),
                "reason_count": len(dyn.get("switch_reasons") or []),
                "has_battle_summary": dyn.get("battle_summary") is not None,
            })

        items.sort(key=lambda v: v.get("mention_count") or 0, reverse=True)
        items = items[:limit]

        return json.dumps({"dynamics": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_displacement_dynamics error")
        return json.dumps({"error": "Internal error", "dynamics": [], "count": 0})


# -------------------------------------------------------------------
# Category Dynamics
# -------------------------------------------------------------------


@mcp.tool()
async def get_category_dynamics(
    category: str,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
) -> str:
    """Get the latest category dynamics for a product category.

    Returns market regime (type, churn velocity, price pressure,
    outlier vendors), council summary (winner/loser, insights,
    durability), vendor count, and displacement flow count.

    category: Product category to look up (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    """
    if not category or not category.strip():
        return json.dumps({"success": False, "error": "category is required"})

    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT category, as_of_date, analysis_window_days,
                   schema_version, dynamics, created_at
            FROM b2b_category_dynamics
            WHERE category ILIKE '%' || $1 || '%'
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            category.strip(),
            target_date,
            window_days,
        )

        if not row:
            return json.dumps({"success": False, "error": "No category dynamics found for that category"})

        dyn = _safe_json(row["dynamics"])
        if not isinstance(dyn, dict):
            dyn = {}

        return json.dumps({
            "success": True,
            "category": row["category"],
            "as_of_date": str(row["as_of_date"]),
            "analysis_window_days": row["analysis_window_days"],
            "schema_version": row["schema_version"],
            "created_at": str(row["created_at"]),
            "market_regime": dyn.get("market_regime"),
            "council_summary": dyn.get("council_summary"),
            "vendor_count": dyn.get("vendor_count", 0),
            "displacement_flow_count": dyn.get("displacement_flow_count", 0),
            "cross_category_comparison": dyn.get("cross_category_comparison"),
        }, default=str)
    except Exception:
        logger.exception("get_category_dynamics error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_category_dynamics(
    category: Optional[str] = None,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """List category dynamics summaries across categories (latest per category).

    Returns market regime headline, vendor/flow counts, and council
    winner/loser without the full insight arrays.

    category: Filter by category name (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        conditions = [
            "as_of_date <= $1",
            "analysis_window_days = $2",
        ]
        params: list = [target_date, window_days]
        idx = 3

        if category:
            conditions.append(f"category ILIKE '%' || ${idx} || '%'")
            params.append(category.strip())
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT DISTINCT ON (category)
                   category, as_of_date, schema_version, dynamics, created_at
            FROM b2b_category_dynamics
            WHERE {where}
            ORDER BY category, as_of_date DESC, created_at DESC
            """,
            *params,
        )

        rows = rows[:limit]

        items = []
        for r in rows:
            dyn = _safe_json(r["dynamics"])
            if not isinstance(dyn, dict):
                dyn = {}
            mr = dyn.get("market_regime") or {}
            cs = dyn.get("council_summary") or {}
            items.append({
                "category": r["category"],
                "as_of_date": str(r["as_of_date"]),
                "schema_version": r["schema_version"],
                "created_at": str(r["created_at"]),
                "regime_type": mr.get("regime_type"),
                "avg_churn_velocity": mr.get("avg_churn_velocity"),
                "avg_price_pressure": mr.get("avg_price_pressure"),
                "outlier_vendor_count": len(mr.get("outlier_vendors") or []),
                "council_winner": cs.get("winner"),
                "council_loser": cs.get("loser"),
                "council_confidence": cs.get("confidence"),
                "vendor_count": dyn.get("vendor_count", 0),
                "displacement_flow_count": dyn.get("displacement_flow_count", 0),
            })

        items.sort(key=lambda v: v.get("vendor_count") or 0, reverse=True)

        return json.dumps({"categories": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_category_dynamics error")
        return json.dumps({"error": "Internal error", "categories": [], "count": 0})


# -------------------------------------------------------------------
# Account Intelligence
# -------------------------------------------------------------------


@mcp.tool()
async def get_account_intelligence(
    vendor_name: str,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
) -> str:
    """Get the latest account intelligence for a vendor.

    Returns per-company signals (urgency, pain, buyer role, decision
    maker status, seat count, contract end, buying stage, source) and
    summary statistics. Accounts sorted by urgency score descending.

    vendor_name: Vendor to look up (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name is required"})

    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT vendor_name, as_of_date, analysis_window_days,
                   schema_version, accounts, created_at
            FROM b2b_account_intelligence
            WHERE vendor_name ILIKE '%' || $1 || '%'
              AND as_of_date <= $2
              AND analysis_window_days = $3
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 1
            """,
            vendor_name.strip(),
            target_date,
            window_days,
        )

        if not row:
            return json.dumps({"success": False, "error": "No account intelligence found for that vendor"})

        acct = _safe_json(row["accounts"])
        if not isinstance(acct, dict):
            acct = {}

        return json.dumps({
            "success": True,
            "vendor_name": row["vendor_name"],
            "as_of_date": str(row["as_of_date"]),
            "analysis_window_days": row["analysis_window_days"],
            "schema_version": row["schema_version"],
            "created_at": str(row["created_at"]),
            "accounts": acct.get("accounts", []),
            "summary": acct.get("summary", {}),
        }, default=str)
    except Exception:
        logger.exception("get_account_intelligence error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def list_account_intelligence(
    vendor_name: Optional[str] = None,
    as_of_date: Optional[str] = None,
    window_days: int = 30,
    limit: int = 20,
) -> str:
    """List account intelligence summaries across vendors (latest per vendor).

    Returns summary stats (total accounts, decision maker count,
    contract end count, seat count coverage) without the full
    account arrays.

    vendor_name: Filter by vendor name (partial match, case-insensitive)
    as_of_date: Latest date to consider (ISO format, default today)
    window_days: Analysis window in days (default 30)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    window_days = max(1, min(window_days, 3650))
    target_date = date.today()
    if as_of_date:
        try:
            target_date = date.fromisoformat(as_of_date)
        except ValueError:
            return json.dumps({"success": False, "error": "as_of_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        conditions = [
            "as_of_date <= $1",
            "analysis_window_days = $2",
        ]
        params: list = [target_date, window_days]
        idx = 3

        if vendor_name:
            conditions.append(f"vendor_name ILIKE '%' || ${idx} || '%'")
            params.append(vendor_name.strip())
            idx += 1

        where = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT DISTINCT ON (vendor_name)
                   vendor_name, as_of_date, schema_version, accounts, created_at
            FROM b2b_account_intelligence
            WHERE {where}
            ORDER BY vendor_name, as_of_date DESC, created_at DESC
            """,
            *params,
        )

        rows = rows[:limit]

        items = []
        for r in rows:
            acct = _safe_json(r["accounts"])
            if not isinstance(acct, dict):
                acct = {}
            summary = acct.get("summary") or {}
            acct_list = acct.get("accounts") or []
            top_urgency = None
            if acct_list:
                top_urgency = acct_list[0].get("urgency_score")
            items.append({
                "vendor_name": r["vendor_name"],
                "as_of_date": str(r["as_of_date"]),
                "schema_version": r["schema_version"],
                "created_at": str(r["created_at"]),
                "total_accounts": summary.get("total_accounts", 0),
                "decision_maker_count": summary.get("decision_maker_count", 0),
                "with_contract_end": summary.get("with_contract_end", 0),
                "with_seat_count": summary.get("with_seat_count", 0),
                "top_urgency": top_urgency,
            })

        items.sort(key=lambda v: v.get("total_accounts") or 0, reverse=True)

        return json.dumps({"accounts": items, "count": len(items)}, default=str)
    except Exception:
        logger.exception("list_account_intelligence error")
        return json.dumps({"error": "Internal error", "accounts": [], "count": 0})
