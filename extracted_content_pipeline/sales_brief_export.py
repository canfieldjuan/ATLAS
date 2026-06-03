"""Read-only sales brief draft export helpers for AI Content Ops hosts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any

from .campaign_ports import TenantScope
from .csv_export import csv_cell_value as _csv_value
from .sales_brief_ports import SalesBriefDraft, SalesBriefRepository


JsonDict = dict[str, Any]


_EXPORT_COLUMNS = (
    "target_id",
    "target_mode",
    "brief_type",
    "title",
    "headline",
    "section_count",
    "reference_count",
    "generation_input_tokens",
    "generation_output_tokens",
    "generation_total_tokens",
    "generation_parse_attempts",
    "reasoning_context_used",
    "reasoning_wedge",
    "reasoning_confidence",
    "sections",
    "reference_ids",
    "metadata",
    "id",
    "status",
)


@dataclass(frozen=True)
class SalesBriefDraftExportResult:
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


async def export_sales_brief_drafts(
    repository: SalesBriefRepository,
    *,
    scope: TenantScope | Mapping[str, Any] | None = None,
    status: str | None = "draft",
    target_mode: str | None = None,
    brief_type: str | None = None,
    limit: int = 20,
) -> SalesBriefDraftExportResult:
    """Return generated sales brief drafts for host review/export workflows."""

    tenant = _tenant_scope(scope)
    normalized_limit = _normalize_limit(limit)
    filters: dict[str, Any] = {}
    if status:
        filters["status"] = status
    if tenant.account_id:
        filters["account_id"] = tenant.account_id
    if target_mode:
        filters["target_mode"] = target_mode
    if brief_type:
        filters["brief_type"] = brief_type
    drafts = await repository.list_drafts(
        scope=tenant,
        status=status,
        target_mode=target_mode,
        brief_type=brief_type,
        limit=normalized_limit,
    )
    return SalesBriefDraftExportResult(
        rows=tuple(_draft_row(draft) for draft in drafts),
        limit=normalized_limit,
        filters=filters,
    )


def _draft_row(draft: SalesBriefDraft) -> JsonDict:
    row = draft.as_dict()
    row["section_count"] = len(draft.sections)
    row["reference_count"] = len(draft.reference_ids)
    row.update(_metadata_summary(draft.metadata))
    return row


def _metadata_summary(value: Any) -> JsonDict:
    metadata = _metadata_mapping(value)
    usage = _metadata_mapping(metadata.get("generation_usage"))
    reasoning = _metadata_mapping(metadata.get("reasoning_context"))
    return {
        "generation_input_tokens": usage.get("input_tokens"),
        "generation_output_tokens": usage.get("output_tokens"),
        "generation_total_tokens": usage.get("total_tokens"),
        "generation_parse_attempts": metadata.get("generation_parse_attempts"),
        "reasoning_context_used": bool(reasoning),
        "reasoning_wedge": reasoning.get("wedge"),
        "reasoning_confidence": reasoning.get("confidence"),
    }


def _metadata_mapping(value: Any) -> JsonDict:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


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
    "SalesBriefDraftExportResult",
    "export_sales_brief_drafts",
]
