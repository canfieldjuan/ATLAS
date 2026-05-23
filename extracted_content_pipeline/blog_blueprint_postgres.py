"""Postgres adapter for AI Content Ops blog blueprints.

Backs the package's ``BlogBlueprintRepository`` Protocol
(``extracted_content_pipeline/blog_ports.py``) with a concrete
table implementation. Hosts that maintain blueprints in their
own storage can implement the Protocol against that store
instead; this adapter is the default for hosts using the
extracted package's own migrations.

Read path (the Protocol method): returns blueprints filtered
by ``(account_id, target_mode)`` with ``consumed_at IS NULL``,
ordered by recency, payload merged with row metadata so the
generator sees a single self-contained dict.

Write path (concrete extension): ``save_blueprints`` for hosts
landing blueprints in this table; ``mark_consumed`` to flag
already-generated blueprints so duplicate runs skip them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from .blog_ports import BlogBlueprint
from .campaign_ports import JsonDict, TenantScope
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    row_to_dict,
)


def _row_to_blueprint(row: Mapping[str, Any]) -> JsonDict:
    """Decode a row into the dict shape the generator consumes.

    The ``payload`` JSONB carries the rich blueprint dict
    (sections / charts / tags / data_context / etc.). Row-level
    metadata (``id``, ``target_mode``, ``topic_type``, ``slug``,
    ``suggested_title``) is merged in so callers see a single
    self-contained dict regardless of which fields lived in
    typed columns vs. the JSONB blob.
    """

    payload_raw = decode_jsonb_field(row.get("payload"), default={})
    if not isinstance(payload_raw, Mapping):
        payload_raw = {}
    merged: JsonDict = dict(payload_raw)
    merged.setdefault("id", str(row.get("id") or ""))
    merged.setdefault("target_mode", str(row.get("target_mode") or ""))
    merged.setdefault("topic_type", str(row.get("topic_type") or ""))
    merged.setdefault("slug", str(row.get("slug") or ""))
    merged.setdefault("suggested_title", str(row.get("suggested_title") or ""))
    return merged


@dataclass(frozen=True)
class PostgresBlogBlueprintRepository:
    """Async Postgres adapter for blog blueprints."""

    pool: Any
    table: str = "blog_blueprints"

    async def read_blog_blueprints(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Return unconsumed blueprints for the tenant + target_mode.

        ``filters`` is accepted for protocol parity and currently honors
        ``topic_type`` and ``slug``. The smoke path uses both so a freshly
        seeded custom blueprint is selected uniquely instead of relying on
        recency within a broad topic type.
        """

        topic_type = None
        slug = None
        if filters:
            raw = filters.get("topic_type")
            if isinstance(raw, str) and raw.strip():
                topic_type = raw.strip()
            raw_slug = filters.get("slug")
            if isinstance(raw_slug, str) and raw_slug.strip():
                slug = raw_slug.strip()

        query = f"""
            SELECT id, target_mode, topic_type, slug, suggested_title, payload
            FROM {self.table}
            WHERE account_id = $1
              AND target_mode = $2
              AND consumed_at IS NULL
        """
        args: list[Any] = [scope.account_id or "", target_mode]
        if topic_type is not None:
            args.append(topic_type)
            query += f" AND topic_type = ${len(args)}"
        if slug is not None:
            args.append(slug)
            query += f" AND slug = ${len(args)}"
        query += f" ORDER BY created_at DESC LIMIT ${len(args) + 1}"
        args.append(int(limit))

        rows = await self.pool.fetch(query, *args)
        return tuple(_row_to_blueprint(row_to_dict(row)) for row in rows)

    async def save_blueprints(
        self,
        blueprints: Sequence[BlogBlueprint],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist blueprints and return assigned ids.

        Outside the upstream Protocol -- hosts with their own
        blueprint store don't need this. Hosts using the
        extracted package's table call this from their blueprint
        ETL path (or extend the autonomous task to write
        through here).

        Conflicts on the unique ``(account_id, target_mode, slug)``
        index resolve to UPDATE: payload + suggested_title are
        refreshed and ``consumed_at`` is cleared so the blueprint
        is eligible for regeneration. Returns the resulting id.
        """

        saved: list[str] = []
        account_id = scope.account_id or ""
        for blueprint in blueprints:
            blueprint_id = await self.pool.fetchval(
                f"""
                INSERT INTO {self.table} (
                    account_id, target_mode, topic_type, slug,
                    suggested_title, payload
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (account_id, target_mode, slug) DO UPDATE
                SET topic_type = EXCLUDED.topic_type,
                    suggested_title = EXCLUDED.suggested_title,
                    payload = EXCLUDED.payload,
                    consumed_at = NULL
                RETURNING id
                """,
                account_id,
                blueprint.target_mode,
                blueprint.topic_type,
                blueprint.slug,
                blueprint.suggested_title,
                json_dump_jsonb(dict(blueprint.payload or {})),
            )
            saved.append(str(blueprint_id))
        return tuple(saved)

    async def mark_consumed(
        self,
        blueprint_ids: Sequence[str],
        *,
        scope: TenantScope,
        consumed_at: datetime | None = None,
    ) -> int:
        """Flag blueprints as consumed so they drop out of reads.

        Returns the count of rows actually updated, parsed from
        asyncpg's ``"UPDATE N"`` command tag. Mismatched ids
        (already-consumed, wrong tenant) are silently skipped --
        callers can compare the return to ``len(blueprint_ids)``
        to detect partial-batch failures. Test fakes that return
        a non-string command tag fall back to
        ``len(blueprint_ids)``. ``consumed_at`` defaults to
        ``NOW()`` server-side.
        """

        if not blueprint_ids:
            return 0
        account_id = scope.account_id or ""
        result = await self.pool.execute(
            f"""
            UPDATE {self.table}
            SET consumed_at = COALESCE($3, NOW())
            WHERE account_id = $1
              AND id = ANY($2::uuid[])
              AND consumed_at IS NULL
            """,
            account_id,
            list(blueprint_ids),
            consumed_at,
        )
        if isinstance(result, str):
            try:
                return int(result.rsplit(" ", 1)[-1])
            except (ValueError, IndexError):
                return len(blueprint_ids)
        return len(blueprint_ids)


__all__ = [
    "BlogBlueprint",
    "PostgresBlogBlueprintRepository",
]
