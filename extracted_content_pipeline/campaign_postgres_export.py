"""Postgres draft export helpers for the standalone campaign product."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any

from .campaign_ports import TenantScope


JsonDict = dict[str, Any]


_EXPORT_COLUMNS = (
    "id",
    "company_name",
    "vendor_name",
    "target_mode",
    "channel",
    "status",
    "recipient_email",
    "subject",
    "body",
    "cta",
    "llm_model",
    "created_at",
    "metadata",
)


@dataclass(frozen=True)
class CampaignDraftExportResult:
    rows: tuple[JsonDict, ...]
    limit: int
    filters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.rows),
            "limit": self.limit,
            "filters": dict(self.filters),
            "rows": [dict(row) for row in self.rows],
        }

    def as_csv(self) -> str:
        handle = StringIO()
        writer = csv.DictWriter(handle, fieldnames=list(_EXPORT_COLUMNS))
        writer.writeheader()
        for row in self.rows:
            writer.writerow(
                {
                    column: _csv_value(row.get(column))
                    for column in _EXPORT_COLUMNS
                }
            )
        return handle.getvalue()


async def list_campaign_drafts(
    pool: Any,
    *,
    scope: TenantScope | Mapping[str, Any] | None = None,
    campaign_table: str = "b2b_campaigns",
    statuses: Sequence[str] = ("draft",),
    target_mode: str | None = None,
    channel: str | None = None,
    vendor_name: str | None = None,
    company_name: str | None = None,
    limit: int = 20,
) -> CampaignDraftExportResult:
    """Return generated campaign drafts for host review/export workflows."""

    table = _identifier(campaign_table)
    tenant = _tenant_scope(scope)
    filters: dict[str, Any] = {
        "statuses": tuple(_clean(status) for status in statuses if _clean(status)),
    }
    where: list[str] = []
    params: list[Any] = []

    if filters["statuses"]:
        params.append(list(filters["statuses"]))
        where.append(f"status = ANY(${len(params)}::text[])")
    if tenant.account_id:
        params.append(tenant.account_id)
        where.append(f"metadata -> 'scope' ->> 'account_id' = ${len(params)}")
        filters["account_id"] = tenant.account_id
    if target_mode:
        params.append(_clean(target_mode))
        where.append(f"target_mode = ${len(params)}")
        filters["target_mode"] = _clean(target_mode)
    if channel:
        params.append(_clean(channel))
        where.append(f"channel = ${len(params)}")
        filters["channel"] = _clean(channel)
    if vendor_name:
        params.append(_clean(vendor_name))
        where.append(f"LOWER(vendor_name) = LOWER(${len(params)})")
        filters["vendor_name"] = _clean(vendor_name)
    if company_name:
        params.append(_clean(company_name))
        where.append(f"LOWER(company_name) = LOWER(${len(params)})")
        filters["company_name"] = _clean(company_name)

    params.append(max(1, int(limit)))
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = await pool.fetch(
        f"""
        SELECT
            id, company_name, vendor_name, target_mode, channel, status,
            recipient_email, subject, body, cta, llm_model, created_at,
            COALESCE(metadata, '{{}}'::jsonb) AS metadata
          FROM {table}
          {where_sql}
         ORDER BY created_at DESC
         LIMIT ${len(params)}
        """,
        *params,
    )
    return CampaignDraftExportResult(
        rows=tuple(_serializable_row(_row_dict(row)) for row in rows),
        limit=max(1, int(limit)),
        filters=filters,
    )


def _tenant_scope(value: TenantScope | Mapping[str, Any] | None) -> TenantScope:
    if isinstance(value, TenantScope):
        return value
    if isinstance(value, Mapping):
        return TenantScope(
            account_id=_clean(value.get("account_id")) or None,
            user_id=_clean(value.get("user_id")) or None,
        )
    return TenantScope()


def _serializable_row(row: Mapping[str, Any]) -> JsonDict:
    return {
        key: _json_ready(value)
        for key, value in row.items()
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, separators=(",", ":"))
    return "" if value is None else value


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
    "CampaignDraftExportResult",
    "list_campaign_drafts",
]
