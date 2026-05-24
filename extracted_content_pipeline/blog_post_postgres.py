"""Postgres repository adapter for AI Content Ops blog posts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
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
_MAX_SLUG_ATTEMPTS = 5
logger = logging.getLogger(__name__)


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
    _hydrate_seo_metadata(metadata, row)

    return BlogPostDraft(
        id=str(row.get("id") or ""),
        slug=str(row.get("slug") or ""),
        title=str(row.get("title") or ""),
        description=str(row.get("description") or ""),
        topic_type=str(row.get("topic_type") or ""),
        tags=tuple(str(tag) for tag in tags_raw if str(tag).strip()),
        content=str(row.get("content") or ""),
        charts=tuple(dict(chart) for chart in charts_raw if isinstance(chart, Mapping)),
        data_context=data_context,
        metadata=metadata,
        status=str(row.get("status") or ""),
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


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _slug_part(value: Any, *, fallback: str) -> str:
    chars: list[str] = []
    previous_dash = False
    for char in str(value or "").strip().lower():
        if "a" <= char <= "z" or "0" <= char <= "9":
            chars.append(char)
            previous_dash = False
        elif chars and not previous_dash:
            chars.append("-")
            previous_dash = True
    slug = "".join(chars).strip("-")
    return slug or fallback


def _fallback_slug(base_slug: str, account_id: str, attempt: int) -> str:
    suffix = _slug_part(account_id, fallback="draft")[:40].strip("-") or "draft"
    if attempt > 1:
        suffix = f"{suffix}-{attempt}"
    base = _slug_part(base_slug, fallback="blog-post")
    max_base_length = max(1, 180 - len(suffix) - 1)
    return f"{base[:max_base_length].strip('-')}-{suffix}"


def _metadata_json_list(metadata: Mapping[str, Any], key: str) -> tuple[bool, list[Any]]:
    if key not in metadata:
        return False, []
    value = metadata.get(key)
    if isinstance(value, str):
        decoded = decode_jsonb_field(value, default=[])
        value = decoded
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return True, list(value)
    return True, []


def _hydrate_seo_metadata(metadata: JsonDict, row: Mapping[str, Any]) -> None:
    for key in ("seo_title", "seo_description", "target_keyword"):
        value = _optional_text(row.get(key))
        if value:
            metadata[key] = value
    for key in ("secondary_keywords", "faq"):
        if key not in row:
            continue
        value = decode_jsonb_field(row.get(key), default=[])
        items = (
            list(value)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
            else []
        )
        if items or key not in metadata:
            metadata[key] = items


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
            blog_post_id = await self._save_draft_with_slug(
                draft,
                scope=scope,
                slug=draft.slug,
            )
            attempt = 1
            while not blog_post_id and attempt < _MAX_SLUG_ATTEMPTS:
                blog_post_id = await self._save_draft_with_slug(
                    draft,
                    scope=scope,
                    slug=_fallback_slug(draft.slug, account_id, attempt),
                )
                attempt += 1
            if blog_post_id:
                saved.append(str(blog_post_id))
            else:
                logger.warning(
                    "blog_post_draft_save_exhausted_slug_attempts",
                    extra={
                        "account_id": account_id,
                        "slug": draft.slug,
                        "attempts": _MAX_SLUG_ATTEMPTS,
                    },
                )
        return tuple(saved)

    async def _save_draft_with_slug(
        self,
        draft: BlogPostDraft,
        *,
        scope: TenantScope,
        slug: str,
    ) -> Any:
        account_id = scope.account_id or ""
        data_context = _draft_data_context(draft, scope)
        has_secondary_keywords, secondary_keywords = _metadata_json_list(
            draft.metadata,
            "secondary_keywords",
        )
        has_faq, faq = _metadata_json_list(draft.metadata, "faq")
        return await self.pool.fetchval(
            """
            INSERT INTO blog_posts (
                account_id, slug, title, description, topic_type,
                tags, content, charts, data_context, status,
                llm_model, source_report_date,
                seo_title, seo_description, target_keyword,
                secondary_keywords, faq
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6::jsonb, $7, $8::jsonb, $9::jsonb, 'draft',
                $10, $11,
                $12, $13, $14, $15::jsonb, $16::jsonb
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
                source_report_date = EXCLUDED.source_report_date,
                seo_title = COALESCE(EXCLUDED.seo_title, blog_posts.seo_title),
                seo_description = COALESCE(
                    EXCLUDED.seo_description,
                    blog_posts.seo_description
                ),
                target_keyword = COALESCE(
                    EXCLUDED.target_keyword,
                    blog_posts.target_keyword
                ),
                secondary_keywords = CASE
                    WHEN $17 THEN EXCLUDED.secondary_keywords
                    ELSE blog_posts.secondary_keywords
                END,
                faq = CASE
                    WHEN $18 THEN EXCLUDED.faq
                    ELSE blog_posts.faq
                END
            WHERE blog_posts.status != 'published'
              AND blog_posts.account_id = EXCLUDED.account_id
            RETURNING id
            """,
            account_id,
            slug,
            draft.title,
            draft.description,
            draft.topic_type,
            json_dump_jsonb(list(draft.tags or ())),
            draft.content,
            json_dump_jsonb([dict(chart) for chart in draft.charts or ()]),
            json_dump_jsonb(data_context),
            str(draft.metadata.get("generation_model") or "") or None,
            _source_report_date(data_context.get("source_report_date")),
            _optional_text(draft.metadata.get("seo_title")),
            _optional_text(draft.metadata.get("seo_description")),
            _optional_text(draft.metadata.get("target_keyword")),
            json_dump_jsonb(secondary_keywords),
            json_dump_jsonb(faq),
            has_secondary_keywords,
            has_faq,
        )

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        topic_type: str | None = None,
        ids: Sequence[str] | None = None,
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
        normalized_ids = [str(item).strip() for item in ids or () if str(item).strip()]
        if normalized_ids:
            params.append(normalized_ids)
            clauses.append(f"id = ANY(${len(params)}::uuid[])")
        sql = (
            "SELECT id, slug, title, description, topic_type, tags, content, "
            "charts, data_context, llm_model, status, "
            "seo_title, seo_description, target_keyword, secondary_keywords, faq "
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

    async def update_statuses(
        self,
        blog_post_ids: Sequence[str],
        status: str,
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        ids = [str(item).strip() for item in blog_post_ids if str(item).strip()]
        if not ids:
            return ()
        rows = await self.pool.fetch(
            """
            UPDATE blog_posts
               SET status = $2,
                   published_at = CASE
                       WHEN $2 = 'published' THEN COALESCE(published_at, NOW())
                       ELSE published_at
                   END
             WHERE id = ANY($1::uuid[])
               AND account_id = $3
            RETURNING id
            """,
            ids,
            status,
            scope.account_id or "",
        )
        return tuple(str(row_to_dict(row).get("id") or "") for row in rows)


__all__ = [
    "PostgresBlogPostRepository",
]
