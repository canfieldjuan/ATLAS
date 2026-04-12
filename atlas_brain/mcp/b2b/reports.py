"""B2B Churn MCP -- report tools."""
import json
import uuid as _uuid
from typing import Optional

from ...api.b2b_dashboard import _extract_report_account_preview_fields
from ._shared import _is_uuid, _safe_json, get_pool, logger, VALID_REPORT_TYPES
from .server import mcp


@mcp.tool()
async def list_reports(
    report_type: Optional[str] = None,
    vendor_filter: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    List B2B intelligence reports.

    report_type: Filter by type (weekly_churn_feed, vendor_scorecard,
                 displacement_report, category_overview, vendor_retention,
                 challenger_intel)
    vendor_filter: Filter by vendor name in report (partial match)
    limit: Maximum results (default 10, cap 50)
    """
    limit = max(1, min(limit, 50))
    if report_type and report_type not in VALID_REPORT_TYPES:
        return json.dumps({"error": f"report_type must be one of {VALID_REPORT_TYPES}", "reports": [], "count": 0})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        conditions = []
        params = []
        idx = 1

        if report_type:
            conditions.append(f"report_type = ${idx}")
            params.append(report_type)
            idx += 1

        if vendor_filter:
            conditions.append(f"vendor_filter ILIKE '%' || ${idx} || '%'")
            params.append(vendor_filter)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        capped = min(limit, 50)
        params.append(capped)

        rows = await pool.fetch(
            f"""
            SELECT id, report_date, report_type, executive_summary,
                   vendor_filter, status, created_at, intelligence_data
            FROM b2b_intelligence
            {where}
            ORDER BY report_date DESC
            LIMIT ${idx}
            """,
            *params,
        )

        reports = [
            {
                "id": str(r["id"]),
                "report_date": r["report_date"],
                "report_type": r["report_type"],
                "executive_summary": r["executive_summary"],
                "vendor_filter": r["vendor_filter"],
                "status": r["status"],
                "created_at": r["created_at"],
                **_extract_report_account_preview_fields(r.get("intelligence_data")),
            }
            for r in rows
        ]

        return json.dumps({"reports": reports, "count": len(reports)}, default=str)
    except Exception as exc:
        logger.exception("list_reports error")
        return json.dumps({"error": "Internal error", "reports": [], "count": 0})


@mcp.tool()
async def get_report(report_id: str) -> str:
    """
    Fetch a full B2B intelligence report by UUID.

    report_id: UUID of the report to retrieve
    """
    if not _is_uuid(report_id):
        return json.dumps({"success": False, "error": "Invalid report_id (must be UUID)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})
        row = await pool.fetchrow(
            "SELECT * FROM b2b_intelligence WHERE id = $1",
            _uuid.UUID(report_id),
        )

        if not row:
            return json.dumps({"success": False, "error": "Report not found"})

        report = {
            "id": str(row["id"]),
            "report_date": row["report_date"],
            "report_type": row["report_type"],
            "vendor_filter": row["vendor_filter"],
            "category_filter": row["category_filter"],
            "executive_summary": row["executive_summary"],
            "intelligence_data": _safe_json(row["intelligence_data"]),
            "data_density": _safe_json(row["data_density"]),
            "status": row["status"],
            "llm_model": row["llm_model"],
            "created_at": row["created_at"],
        }

        return json.dumps({"success": True, "report": report}, default=str)
    except Exception as exc:
        logger.exception("get_report error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def export_report_pdf(report_id: str) -> str:
    """Export a B2B intelligence report as a PDF download link.

    Returns base64-encoded PDF bytes so MCP clients can save the file locally.

    Args:
        report_id: UUID of the report from b2b_intelligence table.
    """
    if not _is_uuid(report_id):
        return json.dumps({"error": "report_id must be a valid UUID"})
    try:
        import base64
        import uuid as _u

        from atlas_brain.services.b2b.pdf_renderer import render_report_pdf

        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"error": "Database not ready"})

        row = await pool.fetchrow(
            "SELECT * FROM b2b_intelligence WHERE id = $1",
            _u.UUID(report_id),
        )
        if not row:
            return json.dumps({"error": "Report not found"})

        intel_data = row["intelligence_data"]
        if isinstance(intel_data, str):
            try:
                intel_data = json.loads(intel_data)
            except (json.JSONDecodeError, TypeError):
                intel_data = {}

        density = row["data_density"]
        if isinstance(density, str):
            try:
                density = json.loads(density)
            except (json.JSONDecodeError, TypeError):
                density = {}

        pdf_bytes = render_report_pdf(
            report_type=row["report_type"],
            vendor_filter=row["vendor_filter"],
            category_filter=row["category_filter"],
            report_date=row["report_date"],
            executive_summary=row["executive_summary"],
            intelligence_data=intel_data,
            data_density=density,
        )

        import re as _re

        vendor = row["vendor_filter"] or row["report_type"]
        filename = f"atlas-report-{vendor}-{row['report_date'] or 'latest'}.pdf"
        filename = _re.sub(r"[^a-z0-9._-]", "-", filename.lower())

        return json.dumps({
            "filename": filename,
            "size_bytes": len(pdf_bytes),
            "content_base64": base64.b64encode(pdf_bytes).decode(),
        })
    except Exception:
        logger.exception("export_report_pdf error")
        return json.dumps({"error": "Failed to generate PDF"})
