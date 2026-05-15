"""Read-only blog-post draft export helpers for AI Content Ops hosts."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from io import StringIO
import json
from typing import Any

from .blog_ports import BlogPostDraft, BlogPostRepository
from .campaign_ports import TenantScope


JsonDict = dict[str, Any]


_EXPORT_COLUMNS = (
    "slug",
    "title",
    "description",
    "topic_type",
    "tag_count",
    "chart_count",
    "generation_input_tokens",
    "generation_output_tokens",
    "generation_total_tokens",
    "generation_parse_attempts",
    "reasoning_context_used",
    "reasoning_wedge",
    "reasoning_confidence",
    "tags",
    "content",
    "charts",
    "data_context",
    "metadata",
    "id",
    "status",
)


@dataclass(frozen=True)
class BlogPostDraftExportResult:
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


async def export_blog_post_drafts(
    repository: BlogPostRepository,
    *,
    scope: TenantScope | Mapping[str, Any] | None = None,
    status: str | None = "draft",
    topic_type: str | None = None,
    limit: int = 20,
) -> BlogPostDraftExportResult:
    """Return generated blog-post drafts for host review/export workflows."""

    tenant = _tenant_scope(scope)
    normalized_limit = _normalize_limit(limit)
    filters: dict[str, Any] = {}
    if status:
        filters["status"] = status
    if tenant.account_id:
        filters["account_id"] = tenant.account_id
    if topic_type:
        filters["topic_type"] = topic_type
    drafts = await repository.list_drafts(
        scope=tenant,
        status=status,
        topic_type=topic_type,
        limit=normalized_limit,
    )
    return BlogPostDraftExportResult(
        rows=tuple(_draft_row(draft) for draft in drafts),
        limit=normalized_limit,
        filters=filters,
    )


def _draft_row(draft: BlogPostDraft) -> JsonDict:
    row = draft.as_dict()
    row["tag_count"] = len(draft.tags)
    row["chart_count"] = len(draft.charts)
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
    "BlogPostDraftExportResult",
    "export_blog_post_drafts",
]
