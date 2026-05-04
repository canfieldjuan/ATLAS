"""Postgres repository adapters for podcast repurposing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from .podcast_ports import (
    PodcastFormatDraft,
    PodcastIdea,
    PodcastTranscript,
    TenantScope,
)


JsonDict = dict[str, Any]


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _json_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    parsed = _json_value(value)
    if parsed in (None, "", [], {}):
        return ()
    if isinstance(parsed, (list, tuple)):
        return tuple(str(item).strip() for item in parsed if str(item or "").strip())
    return (str(parsed),)


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _transcript_from_row(row: Mapping[str, Any]) -> PodcastTranscript:
    raw = _json_value(row.get("raw_payload")) or {}
    if not isinstance(raw, Mapping):
        raw = {}
    publish_date = row.get("publish_date")
    if publish_date is not None and not isinstance(publish_date, str):
        publish_date = publish_date.isoformat() if hasattr(publish_date, "isoformat") else str(publish_date)
    episode_id = str(row.get("episode_id") or "")
    # Fall back to episode_id when no explicit title was provided so
    # downstream prompts always see a non-empty title field.
    title = str(row.get("title") or "").strip() or episode_id
    return PodcastTranscript(
        episode_id=episode_id,
        title=title,
        transcript_text=str(row.get("transcript_text") or ""),
        duration_seconds=_coerce_int(row.get("duration_seconds")),
        publish_date=publish_date,
        host_name=row.get("host_name") or None,
        guest_name=row.get("guest_name") or None,
        source_url=row.get("source_url") or None,
        raw_payload=dict(raw),
    )


def _idea_from_row(row: Mapping[str, Any]) -> PodcastIdea:
    metadata = _json_value(row.get("metadata")) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return PodcastIdea(
        idea_id=str(row.get("id")) if row.get("id") is not None else None,
        episode_id=str(row.get("episode_id") or ""),
        rank=int(row.get("rank") or 0),
        summary=str(row.get("summary") or ""),
        arguments=_string_tuple(row.get("arguments")),
        hooks=_string_tuple(row.get("hooks")),
        key_quotes=_string_tuple(row.get("key_quotes")),
        teaching_moments=_string_tuple(row.get("teaching_moments")),
        metadata=dict(metadata),
    )


@dataclass(frozen=True)
class PostgresTranscriptRepository:
    pool: Any
    transcripts_table: str = "podcast_transcripts"

    async def read_transcripts(
        self,
        *,
        scope: TenantScope,
        episode_id: str | None = None,
        limit: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[PodcastTranscript]:
        del filters
        table = _identifier(self.transcripts_table)
        # Empty string sentinel for "no tenant scope" — matches the migration
        # default (NOT NULL DEFAULT '').
        account_id = scope.account_id or ""
        params: list[Any] = [account_id]
        where = ["status = 'active'", "account_id = $1"]
        if episode_id is not None:
            params.append(str(episode_id))
            where.append(f"episode_id = ${len(params)}")
        params.append(int(max(0, limit)))
        rows = await self.pool.fetch(
            f"""
            SELECT
                id, account_id, episode_id, title, transcript_text,
                duration_seconds, publish_date, host_name, guest_name,
                source_url, raw_payload
              FROM {table}
             WHERE {' AND '.join(where)}
             ORDER BY created_at DESC NULLS LAST
             LIMIT ${len(params)}
            """,
            *params,
        )
        return tuple(_transcript_from_row(_row_dict(row)) for row in rows)


@dataclass(frozen=True)
class PostgresIdeaRepository:
    pool: Any
    ideas_table: str = "podcast_extracted_ideas"

    async def save_ideas(
        self,
        ideas: Sequence[PodcastIdea],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        if not ideas:
            return ()
        table = _identifier(self.ideas_table)
        # Coerce to '' so the ON CONFLICT (account_id, episode_id, rank)
        # upsert actually fires for single-tenant inserts.
        account_id = scope.account_id or ""
        saved: list[str] = []
        for idea in ideas:
            row = await self.pool.fetchrow(
                f"""
                INSERT INTO {table} (
                    account_id, episode_id, rank, summary,
                    arguments, hooks, key_quotes, teaching_moments,
                    metadata, status
                )
                VALUES (
                    $1, $2, $3, $4,
                    $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb,
                    $9::jsonb,
                    'active'
                )
                ON CONFLICT (account_id, episode_id, rank) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    arguments = EXCLUDED.arguments,
                    hooks = EXCLUDED.hooks,
                    key_quotes = EXCLUDED.key_quotes,
                    teaching_moments = EXCLUDED.teaching_moments,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                RETURNING id
                """,
                account_id,
                idea.episode_id,
                int(idea.rank),
                idea.summary,
                _jsonb(list(idea.arguments)),
                _jsonb(list(idea.hooks)),
                _jsonb(list(idea.key_quotes)),
                _jsonb(list(idea.teaching_moments)),
                _jsonb(dict(idea.metadata)),
            )
            saved.append(str(_row_dict(row).get("id") or ""))
        return tuple(saved)

    async def read_ideas(
        self,
        *,
        scope: TenantScope,
        episode_id: str,
        limit: int = 10,
    ) -> Sequence[PodcastIdea]:
        table = _identifier(self.ideas_table)
        account_id = scope.account_id or ""
        params: list[Any] = [account_id, episode_id]
        where = [
            "status = 'active'",
            "account_id = $1",
            "episode_id = $2",
        ]
        params.append(int(max(0, limit)))
        rows = await self.pool.fetch(
            f"""
            SELECT
                id, account_id, episode_id, rank, summary,
                arguments, hooks, key_quotes, teaching_moments,
                metadata
              FROM {table}
             WHERE {' AND '.join(where)}
             ORDER BY rank ASC
             LIMIT ${len(params)}
            """,
            *params,
        )
        return tuple(_idea_from_row(_row_dict(row)) for row in rows)


@dataclass(frozen=True)
class PostgresFormatDraftRepository:
    pool: Any
    drafts_table: str = "podcast_format_drafts"

    async def save_drafts(
        self,
        drafts: Sequence[PodcastFormatDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        if not drafts:
            return ()
        table = _identifier(self.drafts_table)
        account_id = scope.account_id or ""
        saved: list[str] = []
        for draft in drafts:
            audit = dict(draft.quality_audit)
            audit_status = str(audit.get("status") or "").strip().lower()
            row_status = "needs_review" if audit_status == "fail" else "draft"
            row = await self.pool.fetchrow(
                f"""
                INSERT INTO {table} (
                    account_id, idea_id, episode_id, format_type,
                    title, body, metadata, quality_audit, status
                )
                VALUES (
                    $1, $2::uuid, $3, $4,
                    $5, $6, $7::jsonb, $8::jsonb, $9
                )
                RETURNING id
                """,
                account_id,
                draft.idea_id,
                draft.episode_id,
                draft.format_type,
                _clean(draft.title),
                draft.body,
                _jsonb(dict(draft.metadata)),
                _jsonb(audit),
                row_status,
            )
            saved.append(str(_row_dict(row).get("id") or ""))
        return tuple(saved)


__all__ = [
    "PostgresTranscriptRepository",
    "PostgresIdeaRepository",
    "PostgresFormatDraftRepository",
]
