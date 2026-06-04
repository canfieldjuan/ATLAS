"""Read-only stat-card draft export helpers for AI Content Ops hosts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from io import StringIO
from typing import Any

from .campaign_ports import TenantScope
from .csv_export import csv_cell_value as _csv_value
from .stat_card_ports import StatCardDraft, StatCardRepository


JsonDict = dict[str, Any]


_EXPORT_COLUMNS = (
    "target_id",
    "target_mode",
    "theme",
    "company_name",
    "vendor_name",
    "source_id",
    "source_type",
    "metric_label",
    "metric_value",
    "metric_display",
    "claim",
    "headline",
    "supporting_text",
    "evidence",
    "pain_point_count",
    "pain_points",
    "metadata",
    "id",
    "status",
)


@dataclass(frozen=True)
class StatCardDraftExportResult:
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
            writer.writerow({
                column: _csv_value(row.get(column))
                for column in _EXPORT_COLUMNS
            })
        return handle.getvalue()


async def export_stat_card_drafts(
    repository: StatCardRepository,
    *,
    scope: TenantScope | Mapping[str, Any] | None = None,
    status: str | None = "draft",
    target_mode: str | None = None,
    theme: str | None = None,
    limit: int = 20,
) -> StatCardDraftExportResult:
    """Return generated stat-card drafts for host review/export workflows."""

    tenant = _tenant_scope(scope)
    normalized_limit = _normalize_limit(limit)
    filters: dict[str, Any] = {}
    if status:
        filters["status"] = status
    if tenant.account_id:
        filters["account_id"] = tenant.account_id
    if target_mode:
        filters["target_mode"] = target_mode
    if theme:
        filters["theme"] = theme
    drafts = await repository.list_drafts(
        scope=tenant,
        status=status,
        target_mode=target_mode,
        theme=theme,
        limit=normalized_limit,
    )
    return StatCardDraftExportResult(
        rows=tuple(_draft_row(draft) for draft in drafts),
        limit=normalized_limit,
        filters=filters,
    )


def _draft_row(draft: StatCardDraft) -> JsonDict:
    row = draft.as_dict()
    row["pain_point_count"] = len(draft.pain_points)
    return row


def _normalize_limit(value: Any) -> int:
    limit = int(value)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return limit


def _tenant_scope(value: TenantScope | Mapping[str, Any] | None) -> TenantScope:
    if isinstance(value, TenantScope):
        return value
    if isinstance(value, Mapping):
        return TenantScope(
            account_id=str(value.get("account_id") or "") or None,
            user_id=str(value.get("user_id") or "") or None,
        )
    return TenantScope()


__all__ = [
    "StatCardDraftExportResult",
    "export_stat_card_drafts",
]
