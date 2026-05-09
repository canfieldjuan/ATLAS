"""Postgres repository adapter for AI Content Ops blog posts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

from .blog_ports import BlogPostDraft
from .campaign_ports import JsonDict, TenantScope
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    parse_command_tag,
    row_to_dict,
)


_METADATA_KEY = "_metadata"


def _draft_data_context(draft: BlogPostDraft, scope: TenantScope) -> JsonDict:
    data_context = dict(draft.data_context or {})
    data_context[_METADATA_KEY] = dict(draft.metadata or {})
    data_context["scope"] = {
        "account_id": scope.account_id,
        "user_id": scope.user_id,
    }
    return data_context


def _row_to_draft(row: Mapping[str, Any]) -> BlogPostDraft:
    tags_raw = decode_jsonb_field(row.get("tags"), default=[])
    if not isinstance(tags_raw, Sequence) or isinstance(tags_raw, (str, bytes)):
        tags_raw = []

    charts_raw = decode_jsonb_field(row.get("charts"), default=[])
    if not isinstance(charts_raw, Sequence) or isinstance(charts_raw, (str, bytes)):
        charts_raw = []

    data_context_raw = decode_jsonb_field(row.get("data_context"), default={})
    if not isinstance(data_context_raw, Mapping):
        data_context_raw = {}

    data_context = dict(data_context_raw)
    metadata_raw = data_context.pop(_METADATA_KEY, {})
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
    if row.get("llm_model") and "generation_model" not in metadata:
        metadata["generation_model"] = str(row.get("llm_model"))

    return BlogPostDraft(
        slug=str(row.get("slug") or ""),
        title=str(row.get("title") or ""),
        description=str(row.get("description") or ""),
        topic_type=str(row.get("topic_type") or ""),
        tags=tuple(str(tag) for tag in tags_raw if str(tag).strip()),
        content=str(row.get("content") or ""),
        charts=tuple(dict(chart) for chart in charts_raw if isinstance(chart, Mapping)),
        data_context=data_context,
        metadata=metadata,
    )


def _source_report_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class PostgresBlogPostRepository:
    """Async Postgres adapter for generated blog-post drafts."""

    pool: Any

    async def save_drafts(
        self,
        drafts: Sequence[BlogPostDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        saved: list[str] = []
        account_id = scope.account_id or ""
        for draft in drafts:
            data_context = _draft_data_context(draft, scope)
            blog_post_id = await self.pool.fetchval(
                """
                INSERT INTO blog_posts (
                    account_id, slug, title, description, topic_type,
                    tags, content, charts, data_context, status,
                    llm_model, source_report_date
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6::jsonb, $7, $8::jsonb, $9::jsonb, 'draft',
                    $10, $11
                )
                ON CONFLICT (slug) DO UPDATE SET
                    account_id = EXCLUDED.account_id,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    topic_type = EXCLUDED.topic_type,
                    tags = EXCLUDED.tags,
                    content = EXCLUDED.content,
                    charts = EXCLUDED.charts,
                    data_context = EXCLUDED.data_context,
                    status = EXCLUDED.status,
                    llm_model = EXCLUDED.llm_model,
                    source_report_date = EXCLUDED.source_report_date
                WHERE blog_posts.status != 'published'
                RETURNING id
                """,
                account_id,
                draft.slug,
                draft.title,
                draft.description,
                draft.topic_type,
                json_dump_jsonb(list(draft.tags or ())),
                draft.content,
                json_dump_jsonb([dict(chart) for chart in draft.charts or ()]),
                json_dump_jsonb(data_context),
                str(draft.metadata.get("generation_model") or "") or None,
                _source_report_date(data_context.get("source_report_date")),
            )
            if blog_post_id:
                saved.append(str(blog_post_id))
        return tuple(saved)

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        topic_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[BlogPostDraft]:
        clauses: list[str] = ["account_id = $1"]
        params: list[Any] = [scope.account_id or ""]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if topic_type is not None:
            params.append(topic_type)
            clauses.append(f"topic_type = ${len(params)}")
        sql = (
            "SELECT slug, title, description, topic_type, tags, content, "
            "charts, data_context, llm_model "
            "FROM blog_posts WHERE " + " AND ".join(clauses) + " "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            params.append(int(limit))
            sql += f" LIMIT ${len(params)}"
        rows = await self.pool.fetch(sql, *params)
        return tuple(_row_to_draft(row_to_dict(row)) for row in rows)

    async def update_status(
        self,
        blog_post_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        result = await self.pool.execute(
            """
            UPDATE blog_posts
               SET status = $2,
                   published_at = CASE
                       WHEN $2 = 'published' THEN COALESCE(published_at, NOW())
                       ELSE published_at
                   END
             WHERE id = $1
               AND account_id = $3
            """,
            blog_post_id,
            status,
            scope.account_id or "",
        )
        return parse_command_tag(result)


__all__ = [
    "PostgresBlogPostRepository",
]
