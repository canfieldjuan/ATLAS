"""Postgres draft review helpers for the standalone campaign product."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from .campaign_ports import TenantScope
from .campaign_postgres_generation import tenant_scope_from_mapping


JsonDict = dict[str, Any]

_ACCOUNT_ID_FILTER_EXPR = "metadata -> 'scope' ->> 'account_id'"
_REVIEW_STATUSES = frozenset({"approved", "queued", "cancelled", "expired"})


@dataclass(frozen=True)
class CampaignDraftReviewResult:
    """Result for host draft review/status update workflows."""

    rows: tuple[JsonDict, ...]
    requested_ids: tuple[str, ...]
    status: str
    dry_run: bool
    filters: Mapping[str, Any]

    @property
    def updated(self) -> int:
        return len(self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "requested": len(self.requested_ids),
            "updated": self.updated,
            "status": self.status,
            "filters": dict(self.filters),
            "rows": [dict(row) for row in self.rows],
        }


async def review_campaign_drafts(
    pool: Any,
    *,
    campaign_ids: Sequence[str],
    status: str = "approved",
    scope: TenantScope | Mapping[str, Any] | None = None,
    campaign_table: str = "b2b_campaigns",
    target_mode: str | None = None,
    from_statuses: Sequence[str] = ("draft",),
    from_email: str | None = None,
    reason: str | None = None,
    reviewed_by: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> CampaignDraftReviewResult:
    """Approve, queue, cancel, or expire generated campaign draft rows."""

    normalized_ids = _normalize_campaign_ids(campaign_ids)
    target_status = _normalize_status(status)
    table = _identifier(campaign_table)
    tenant = tenant_scope_from_mapping(scope)
    status_guard = tuple(
        item
        for item in (_clean(value) for value in from_statuses)
        if item
    )
    patch = {
        **dict(metadata or {}),
        "review_status": target_status,
    }
    if reason:
        patch["review_reason"] = _clean(reason)
    if reviewed_by:
        patch["reviewed_by"] = _clean(reviewed_by)

    params: list[Any] = [list(normalized_ids)]
    where = ["id = ANY($1::uuid[])"]
    update_where = ["campaign.id = matched.id"]
    filters: dict[str, Any] = {"campaign_ids": normalized_ids}
    if status_guard:
        params.append(list(status_guard))
        where.append(f"status = ANY(${len(params)}::text[])")
        update_where.append("campaign.status = matched.previous_status")
        filters["from_statuses"] = status_guard
    if tenant.account_id:
        params.append(tenant.account_id)
        where.append(f"{_ACCOUNT_ID_FILTER_EXPR} = ${len(params)}")
        update_where.append(f"campaign.{_ACCOUNT_ID_FILTER_EXPR} = ${len(params)}")
        filters["account_id"] = tenant.account_id
    if target_mode:
        params.append(_clean(target_mode))
        where.append(f"target_mode = ${len(params)}")
        update_where.append(f"campaign.target_mode = ${len(params)}")
        filters["target_mode"] = _clean(target_mode)

    if dry_run:
        rows = await pool.fetch(
            f"""
            SELECT
                id::text AS id,
                status AS previous_status,
                status,
                company_name,
                vendor_name,
                channel,
                recipient_email,
                from_email,
                COALESCE(metadata, '{{}}'::jsonb) AS metadata
              FROM {table}
             WHERE {' AND '.join(where)}
             ORDER BY created_at DESC
            """,
            *params,
        )
    else:
        params.append(target_status)
        status_position = len(params)
        params.append(_jsonb(patch))
        patch_position = len(params)
        params.append((_clean(from_email) or None) if target_status == "queued" else None)
        from_email_position = len(params)
        rows = await pool.fetch(
            f"""
            WITH matched AS (
                SELECT id, status AS previous_status
                  FROM {table}
                 WHERE {' AND '.join(where)}
            )
            UPDATE {table} AS campaign
               SET status = ${status_position},
                   approved_at = CASE
                       WHEN ${status_position} IN ('approved', 'queued')
                       THEN COALESCE(campaign.approved_at, NOW())
                       ELSE campaign.approved_at
                   END,
                   from_email = CASE
                       WHEN ${status_position} = 'queued'
                       THEN COALESCE(${from_email_position}::text, campaign.from_email)
                       ELSE campaign.from_email
                   END,
                   metadata = COALESCE(campaign.metadata, '{{}}'::jsonb) || ${patch_position}::jsonb,
                   updated_at = NOW()
              FROM matched
             WHERE {' AND '.join(update_where)}
             RETURNING
                   campaign.id::text AS id,
                   matched.previous_status,
                   campaign.status,
                   campaign.company_name,
                   campaign.vendor_name,
                   campaign.channel,
                   campaign.recipient_email,
                   campaign.from_email,
                   COALESCE(campaign.metadata, '{{}}'::jsonb) AS metadata
            """,
            *params,
        )

    return CampaignDraftReviewResult(
        rows=tuple(_serializable_row(_row_dict(row)) for row in rows),
        requested_ids=normalized_ids,
        status=target_status,
        dry_run=dry_run,
        filters=filters,
    )


def _normalize_campaign_ids(values: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(item for item in (_clean(value) for value in values) if item)
    if not ids:
        raise ValueError("at least one campaign id is required")
    return ids


def _normalize_status(value: str) -> str:
    status = _clean(value).lower()
    if status not in _REVIEW_STATUSES:
        allowed = ", ".join(sorted(_REVIEW_STATUSES))
        raise ValueError(f"unsupported review status: {value!r}; expected one of {allowed}")
    return status


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _serializable_row(row: Mapping[str, Any]) -> JsonDict:
    return {key: _json_ready(value) for key, value in row.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


__all__ = [
    "CampaignDraftReviewResult",
    "review_campaign_drafts",
]
