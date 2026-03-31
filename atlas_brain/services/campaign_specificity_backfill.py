"""Helpers for backfilling campaign specificity audits on legacy rows."""

from __future__ import annotations

import json
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


def _inferred_boundary_from_status(status: Any) -> str | None:
    normalized = str(status or "").strip().lower()
    if normalized == "draft":
        return "generation"
    if normalized == "approved":
        return "manual_approval"
    if normalized == "queued":
        return "queue_send"
    if normalized == "sent":
        return "send"
    return None


def _latest_audit_needs_backfill(
    metadata: Mapping[str, Any],
    *,
    status: Any,
) -> bool:
    audit = metadata.get("latest_specificity_audit")
    if not isinstance(audit, dict):
        return True
    failure_explanation = audit.get("failure_explanation")
    if not isinstance(failure_explanation, dict):
        return True
    required_keys = (
        "boundary",
        "primary_blocker",
        "cause_type",
        "missing_inputs",
        "context_sources",
    )
    if any(key not in failure_explanation for key in required_keys):
        return True
    inferred_boundary = _inferred_boundary_from_status(status)
    if not inferred_boundary:
        return False
    audit_boundary = str(audit.get("boundary") or "").strip()
    explanation_boundary = str(failure_explanation.get("boundary") or "").strip()
    return audit_boundary == "backfill" or explanation_boundary == "backfill"


def _backfill_boundary(
    metadata: Mapping[str, Any],
    *,
    status: Any,
) -> str:
    audit = metadata.get("latest_specificity_audit")
    if isinstance(audit, dict):
        boundary = str(audit.get("boundary") or "").strip()
        if boundary and boundary != "backfill":
            return boundary
    inferred_boundary = _inferred_boundary_from_status(status)
    if inferred_boundary:
        return inferred_boundary
    return "backfill"


def derive_campaign_specificity_patch(
    row: Mapping[str, Any],
    *,
    min_anchor_hits: int,
    require_anchor_support: bool,
    require_timing_or_numeric_when_available: bool,
) -> dict[str, Any]:
    metadata = coerce_json_dict(row.get("metadata"))
    status = row.get("status")
    if not _latest_audit_needs_backfill(metadata, status=status):
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
        boundary=_backfill_boundary(metadata, status=status),
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
            bc.subject,
            bc.body,
            bc.cta,
            bc.metadata,
            bc.target_mode,
            cs.company_context
        FROM b2b_campaigns bc
        LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
        WHERE bc.status = ANY($1::text[])
          AND (
              COALESCE(jsonb_typeof(bc.metadata -> 'latest_specificity_audit'), 'null') <> 'object'
              OR COALESCE(
                  jsonb_typeof(bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation'),
                  'null'
              ) <> 'object'
              OR (
                  COALESCE(
                      jsonb_typeof(bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation'),
                      'null'
                  ) = 'object'
                  AND (
                      NOT ((bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation') ? 'boundary')
                      OR NOT ((bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation') ? 'primary_blocker')
                      OR NOT ((bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation') ? 'cause_type')
                      OR NOT ((bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation') ? 'missing_inputs')
                      OR NOT ((bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation') ? 'context_sources')
                  )
              )
              OR COALESCE(bc.metadata -> 'latest_specificity_audit' ->> 'boundary', '') = 'backfill'
              OR COALESCE(
                  bc.metadata -> 'latest_specificity_audit' -> 'failure_explanation' ->> 'boundary',
                  ''
              ) = 'backfill'
          )
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
