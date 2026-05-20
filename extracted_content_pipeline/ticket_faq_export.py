"""Read-only ticket FAQ Markdown draft export helpers for AI Content Ops hosts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any

from .campaign_ports import TenantScope
from .ticket_faq_ports import TicketFAQDraft, TicketFAQRepository


JsonDict = dict[str, Any]


_EXPORT_COLUMNS = (
    "target_id",
    "target_mode",
    "title",
    "source_count",
    "ticket_source_count",
    "passed_output_checks",
    "warning_count",
    "markdown",
    "items",
    "output_checks",
    "warnings",
    "metadata",
    "id",
    "status",
)


@dataclass(frozen=True)
class TicketFAQDraftExportResult:
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


async def export_ticket_faq_drafts(
    repository: TicketFAQRepository,
    *,
    scope: TenantScope | Mapping[str, Any] | None = None,
    status: str | None = "draft",
    target_mode: str | None = None,
    limit: int = 20,
) -> TicketFAQDraftExportResult:
    """Return generated ticket FAQ Markdown drafts for host review/export."""

    tenant = _tenant_scope(scope)
    normalized_limit = _normalize_limit(limit)
    filters: dict[str, Any] = {}
    if status:
        filters["status"] = status
    if tenant.account_id:
        filters["account_id"] = tenant.account_id
    if target_mode:
        filters["target_mode"] = target_mode
    drafts = await repository.list_drafts(
        scope=tenant,
        status=status,
        target_mode=target_mode,
        limit=normalized_limit,
    )
    return TicketFAQDraftExportResult(
        rows=tuple(_draft_row(draft) for draft in drafts),
        limit=normalized_limit,
        filters=filters,
    )


def _draft_row(draft: TicketFAQDraft) -> JsonDict:
    row = draft.as_dict()
    row["warning_count"] = len(draft.warnings)
    row["passed_output_checks"] = _passed_output_checks(draft.output_checks)
    return row


def _passed_output_checks(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    return sum(1 for item in value.values() if item is True)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, separators=(",", ":"))
    return "" if value is None else value


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
    "TicketFAQDraftExportResult",
    "export_ticket_faq_drafts",
]
