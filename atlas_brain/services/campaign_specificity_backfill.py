"""Helpers for backfilling campaign specificity audits on legacy rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .campaign_quality import campaign_quality_revalidation, coerce_json_dict

DEFAULT_BACKFILL_STATUSES = (
    "draft",
    "approved",
    "queued",
    "sent",
    "cancelled",
    "expired",
)
def derive_campaign_specificity_patch(
    row: Mapping[str, Any],
    *,
    min_anchor_hits: int,
    require_anchor_support: bool,
    require_timing_or_numeric_when_available: bool,
) -> dict[str, Any]:
    metadata = coerce_json_dict(row.get("metadata"))
    if isinstance(metadata.get("latest_specificity_audit"), dict):
        return {}

    body = str(row.get("body") or "").strip()
    if not body:
        return {}

    revalidation = campaign_quality_revalidation(
        campaign={
            "subject": str(row.get("subject") or ""),
            "body": body,
            "cta": str(row.get("cta") or ""),
            "channel": str(row.get("channel") or ""),
            "target_mode": str(row.get("target_mode") or ""),
            "tier": str(row.get("tier") or ""),
            "vendor_name": str(row.get("vendor_name") or ""),
            "company_name": str(row.get("company_name") or ""),
            "metadata": metadata,
        },
        boundary="backfill",
        company_context=row.get("company_context"),
        metadata=metadata,
        min_anchor_hits=int(min_anchor_hits),
        require_anchor_support=bool(require_anchor_support),
        require_timing_or_numeric_when_available=bool(
            require_timing_or_numeric_when_available
        ),
    )
    audit = revalidation.get("audit")
    patched_metadata = revalidation.get("metadata")
    if not isinstance(audit, dict) or not isinstance(patched_metadata, dict):
        return {}
    return {"metadata": patched_metadata}


async def plan_campaign_specificity_backfill(
    pool,
    *,
    limit: int | None = None,
    statuses: Sequence[str] = DEFAULT_BACKFILL_STATUSES,
    min_anchor_hits: int,
    require_anchor_support: bool,
    require_timing_or_numeric_when_available: bool,
) -> dict[str, Any]:
    rows = await pool.fetch(
        """
        SELECT
            bc.id::text AS id,
            bc.company_name,
            bc.vendor_name,
            bc.status,
            bc.channel,
            bc.body,
            bc.metadata,
            cs.company_context
        FROM b2b_campaigns bc
        LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
        WHERE bc.status = ANY($1::text[])
          AND COALESCE(jsonb_typeof(bc.metadata -> 'latest_specificity_audit'), 'null') <> 'object'
        ORDER BY bc.created_at DESC
        LIMIT $2
        """,
        list(statuses),
        int(limit or 1000000),
    )

    items: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        patch = derive_campaign_specificity_patch(
            row,
            min_anchor_hits=min_anchor_hits,
            require_anchor_support=require_anchor_support,
            require_timing_or_numeric_when_available=require_timing_or_numeric_when_available,
        )
        items.append(
            {
                "id": str(row["id"]),
                "company_name": str(row.get("company_name") or ""),
                "vendor_name": str(row.get("vendor_name") or ""),
                "status": str(row.get("status") or ""),
                "patch": patch,
            }
        )

    return {
        "scanned": len(items),
        "changed": sum(1 for item in items if item["patch"]),
        "items": items,
    }


async def apply_campaign_specificity_backfill(
    pool,
    *,
    limit: int | None = None,
    statuses: Sequence[str] = DEFAULT_BACKFILL_STATUSES,
    min_anchor_hits: int,
    require_anchor_support: bool,
    require_timing_or_numeric_when_available: bool,
) -> dict[str, Any]:
    plan = await plan_campaign_specificity_backfill(
        pool,
        limit=limit,
        statuses=statuses,
        min_anchor_hits=min_anchor_hits,
        require_anchor_support=require_anchor_support,
        require_timing_or_numeric_when_available=require_timing_or_numeric_when_available,
    )

    applied = 0
    for item in plan["items"]:
        patch = item["patch"]
        if not patch:
            continue
        await pool.execute(
            """
            UPDATE b2b_campaigns
            SET metadata = $2::jsonb
            WHERE id = $1::uuid
            """,
            item["id"],
            json.dumps(patch["metadata"], default=str),
        )
        applied += 1

    return {
        "scanned": plan["scanned"],
        "changed": plan["changed"],
        "applied": applied,
        "items": plan["items"],
    }
