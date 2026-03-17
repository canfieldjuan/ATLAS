"""B2B Churn MCP -- cross-vendor reasoning tools."""
import json
import uuid as _uuid
from typing import Optional

from ._shared import _is_uuid, _safe_json, get_pool, logger
from .server import mcp


@mcp.tool()
async def list_cross_vendor_conclusions(
    analysis_type: Optional[str] = None,
    vendor: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List cross-vendor reasoning conclusions (battles, category councils, asymmetry analyses).

    analysis_type: Filter by type: pairwise_battle, category_council, resource_asymmetry
    vendor: Filter by vendor name (matches any vendor in the analysis, partial match)
    limit: Maximum results (default 20, cap 100)
    """
    limit = max(1, min(limit, 100))
    valid_types = ("pairwise_battle", "category_council", "resource_asymmetry")
    if analysis_type and analysis_type not in valid_types:
        return json.dumps({
            "error": f"analysis_type must be one of {valid_types}",
            "conclusions": [],
            "count": 0,
        })

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        conditions: list[str] = []
        params: list = []
        idx = 1

        if analysis_type:
            conditions.append(f"analysis_type = ${idx}")
            params.append(analysis_type)
            idx += 1

        if vendor:
            conditions.append(f"EXISTS (SELECT 1 FROM unnest(vendors) v WHERE v ILIKE '%' || ${idx} || '%')")
            params.append(vendor)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await pool.fetch(
            f"""
            SELECT id, analysis_type, vendors, category, conclusion,
                   confidence, evidence_hash, tokens_used, cached,
                   computed_date, created_at
            FROM b2b_cross_vendor_conclusions
            {where}
            ORDER BY computed_date DESC, created_at DESC
            LIMIT ${idx}
            """,
            *params,
        )

        conclusions = []
        for r in rows:
            conclusions.append({
                "id": str(r["id"]),
                "analysis_type": r["analysis_type"],
                "vendors": list(r["vendors"]) if r["vendors"] else [],
                "category": r["category"],
                "conclusion": _safe_json(r["conclusion"]),
                "confidence": float(r["confidence"]) if r["confidence"] is not None else 0,
                "evidence_hash": r["evidence_hash"],
                "tokens_used": r["tokens_used"],
                "cached": r["cached"],
                "computed_date": r["computed_date"],
                "created_at": r["created_at"],
            })

        return json.dumps({"conclusions": conclusions, "count": len(conclusions)}, default=str)
    except Exception:
        logger.exception("list_cross_vendor_conclusions error")
        return json.dumps({"error": "Internal error", "conclusions": [], "count": 0})


@mcp.tool()
async def get_cross_vendor_conclusion(
    conclusion_id: str,
) -> str:
    """
    Get a single cross-vendor reasoning conclusion by ID.

    conclusion_id: UUID of the conclusion
    """
    if not _is_uuid(conclusion_id):
        return json.dumps({"error": "Invalid UUID"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            SELECT id, analysis_type, vendors, category, conclusion,
                   confidence, evidence_hash, tokens_used, cached,
                   computed_date, created_at
            FROM b2b_cross_vendor_conclusions
            WHERE id = $1
            """,
            _uuid.UUID(conclusion_id),
        )

        if not row:
            return json.dumps({"error": "Conclusion not found"})

        return json.dumps({
            "id": str(row["id"]),
            "analysis_type": row["analysis_type"],
            "vendors": list(row["vendors"]) if row["vendors"] else [],
            "category": row["category"],
            "conclusion": _safe_json(row["conclusion"]),
            "confidence": float(row["confidence"]) if row["confidence"] is not None else 0,
            "evidence_hash": row["evidence_hash"],
            "tokens_used": row["tokens_used"],
            "cached": row["cached"],
            "computed_date": row["computed_date"],
            "created_at": row["created_at"],
        }, default=str)
    except Exception:
        logger.exception("get_cross_vendor_conclusion error")
        return json.dumps({"error": "Internal error"})
