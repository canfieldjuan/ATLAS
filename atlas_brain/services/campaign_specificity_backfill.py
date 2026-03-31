"""Helpers for backfilling campaign specificity audits on legacy rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..autonomous.tasks._b2b_specificity import (
    merge_specificity_contexts,
    specificity_audit_snapshot,
    surface_specificity_context,
)

DEFAULT_BACKFILL_STATUSES = (
    "draft",
    "approved",
    "queued",
    "sent",
    "cancelled",
    "expired",
)


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _campaign_specificity_context(
    metadata: Any,
    company_context: Any,
) -> dict[str, Any]:
    metadata_context = surface_specificity_context(
        _coerce_json_dict(metadata),
        surface="campaign",
        nested_keys=("briefing_context",),
    )
    company_specificity = surface_specificity_context(
        _coerce_json_dict(company_context),
        surface="campaign",
        nested_keys=("briefing_context",),
    )
    return merge_specificity_contexts(metadata_context, company_specificity)


def derive_campaign_specificity_patch(
    row: Mapping[str, Any],
    *,
    min_anchor_hits: int,
    require_anchor_support: bool,
    require_timing_or_numeric_when_available: bool,
) -> dict[str, Any]:
    metadata = _coerce_json_dict(row.get("metadata"))
    if isinstance(metadata.get("latest_specificity_audit"), dict):
        return {}

    body = str(row.get("body") or "").strip()
    if not body:
        return {}

    context = _campaign_specificity_context(metadata, row.get("company_context"))
    if not context:
        return {}

    audit = specificity_audit_snapshot(
        body,
        anchor_examples=context.get("anchor_examples"),
        witness_highlights=context.get("witness_highlights"),
        reference_ids=context.get("reference_ids"),
        allow_company_names=False,
        min_anchor_hits=int(min_anchor_hits),
        require_anchor_support=bool(require_anchor_support),
        require_timing_or_numeric_when_available=bool(
            require_timing_or_numeric_when_available
        ),
        include_competitor_terms=str(row.get("channel") or "") != "email_cold",
    )
    if not audit:
        return {}

    metadata["latest_specificity_audit"] = {
        **audit,
        "boundary": "backfill",
    }
    return {"metadata": metadata}


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
