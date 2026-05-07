"""Postgres repository adapter for the AI Content Ops Landing Pages product.

Mirrors the shape of ``PostgresReportRepository`` (see
``report_postgres.py``) but persists ``LandingPageDraft`` rows into the
``landing_pages`` table from migration 274. Hosts inject an
asyncpg-style pool; the adapter does no connection management itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .campaign_ports import JsonDict, TenantScope
from .landing_page_ports import (
    LandingPageDraft,
    LandingPageSection,
)


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _draft_metadata(draft: LandingPageDraft, scope: TenantScope) -> JsonDict:
    return {
        **dict(draft.metadata or {}),
        "campaign_name": draft.campaign_name,
        "scope": {
            "account_id": scope.account_id,
            "user_id": scope.user_id,
        },
    }


def _coerce_section(value: Any) -> LandingPageSection:
    """Coerce a host-supplied section into ``LandingPageSection``.

    Accepts an existing ``LandingPageSection`` or a Mapping with the
    expected keys. Anything else (str, int, list, None, ...) raises
    ``TypeError`` so host-side bugs surface at the persistence
    boundary rather than getting silently coerced into an empty
    section that the quality pack later rejects.
    """
    if isinstance(value, LandingPageSection):
        return value
    if isinstance(value, Mapping):
        return LandingPageSection(
            id=str(value.get("id") or ""),
            title=str(value.get("title") or ""),
            body_markdown=str(value.get("body_markdown") or ""),
            metadata=dict(value.get("metadata") or {}),
        )
    raise TypeError(
        "LandingPageDraft.sections entries must be LandingPageSection "
        f"instances or Mappings; got {type(value).__name__}: {value!r}"
    )


def _decode_jsonb(raw: Any, *, default: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default
    if raw is None:
        return default
    return raw


def _row_to_draft(row: Mapping[str, Any]) -> LandingPageDraft:
    sections_raw = _decode_jsonb(row.get("sections"), default=[])
    if not isinstance(sections_raw, Sequence) or isinstance(sections_raw, (str, bytes)):
        sections_raw = []

    reference_ids_raw = _decode_jsonb(row.get("reference_ids"), default=[])
    if not isinstance(reference_ids_raw, Sequence) or isinstance(reference_ids_raw, (str, bytes)):
        reference_ids_raw = []

    hero_raw = _decode_jsonb(row.get("hero"), default={})
    if not isinstance(hero_raw, Mapping):
        hero_raw = {}

    cta_raw = _decode_jsonb(row.get("cta"), default={})
    if not isinstance(cta_raw, Mapping):
        cta_raw = {}

    meta_raw = _decode_jsonb(row.get("meta"), default={})
    if not isinstance(meta_raw, Mapping):
        meta_raw = {}

    metadata_raw = _decode_jsonb(row.get("metadata"), default={})
    if not isinstance(metadata_raw, Mapping):
        metadata_raw = {}

    return LandingPageDraft(
        campaign_name=str(row.get("campaign_name") or ""),
        persona=str(row.get("persona") or ""),
        value_prop=str(row.get("value_prop") or ""),
        title=str(row.get("title") or ""),
        slug=str(row.get("slug") or ""),
        hero=dict(hero_raw),
        sections=tuple(_coerce_section(s) for s in sections_raw),
        cta=dict(cta_raw),
        meta=dict(meta_raw),
        reference_ids=tuple(str(r) for r in reference_ids_raw),
        metadata=dict(metadata_raw),
    )


@dataclass(frozen=True)
class PostgresLandingPageRepository:
    """Async Postgres adapter for generated landing pages."""

    pool: Any

    async def save_drafts(
        self,
        drafts: Sequence[LandingPageDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist drafts; one INSERT per draft (sequential).

        Per-call batches are typically small (1-3 drafts) for the
        per-campaign trigger shape, so the round-trip count is bounded
        in practice. Kept sequential for parity with
        ``PostgresReportRepository`` / ``PostgresCampaignRepository``;
        a future batch-INSERT optimization would migrate all three for
        consistency rather than diverging here.
        """
        saved: list[str] = []
        account_id = scope.account_id or ""
        for draft in drafts:
            sections_payload = [_coerce_section(s).as_dict() for s in draft.sections]
            reference_ids_payload = [str(r) for r in draft.reference_ids]
            metadata_payload = _draft_metadata(draft, scope)
            page_id = await self.pool.fetchval(
                """
                INSERT INTO landing_pages (
                    account_id, campaign_name, persona, value_prop,
                    title, slug,
                    hero, sections, cta, meta, reference_ids, metadata, status
                )
                VALUES (
                    $1, $2, $3, $4,
                    $5, $6,
                    $7::jsonb, $8::jsonb, $9::jsonb, $10::jsonb,
                    $11::jsonb, $12::jsonb, 'draft'
                )
                RETURNING id
                """,
                account_id,
                draft.campaign_name,
                draft.persona,
                draft.value_prop,
                draft.title,
                draft.slug,
                _jsonb(dict(draft.hero or {})),
                _jsonb(sections_payload),
                _jsonb(dict(draft.cta or {})),
                _jsonb(dict(draft.meta or {})),
                _jsonb(reference_ids_payload),
                _jsonb(metadata_payload),
            )
            saved.append(str(page_id))
        return tuple(saved)

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        campaign_name: str | None = None,
        slug: str | None = None,
        limit: int | None = None,
    ) -> Sequence[LandingPageDraft]:
        clauses: list[str] = ["account_id = $1"]
        params: list[Any] = [scope.account_id or ""]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if campaign_name is not None:
            params.append(campaign_name)
            clauses.append(f"campaign_name = ${len(params)}")
        if slug is not None:
            params.append(slug)
            clauses.append(f"slug = ${len(params)}")
        sql = (
            "SELECT campaign_name, persona, value_prop, title, slug, "
            "hero, sections, cta, meta, reference_ids, metadata "
            "FROM landing_pages WHERE " + " AND ".join(clauses) + " "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            params.append(int(limit))
            sql += f" LIMIT ${len(params)}"
        rows = await self.pool.fetch(sql, *params)
        return tuple(_row_to_draft(_row_dict(row)) for row in rows)

    async def update_status(
        self,
        landing_page_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        """Scoped status update. Returns True on hit, False on miss.

        Parses asyncpg's command tag (e.g., ``"UPDATE 1"``) so a
        wrong-id or wrong-tenant call surfaces as a False return at
        the call site rather than a silent no-op.
        """
        result = await self.pool.execute(
            """
            UPDATE landing_pages
               SET status = $2,
                   updated_at = NOW()
             WHERE id = $1
               AND account_id = $3
            """,
            landing_page_id,
            status,
            scope.account_id or "",
        )
        # asyncpg returns "UPDATE <n>"; 1 == hit, 0 == miss. Defensive
        # parse so non-asyncpg pools (test fakes, alternative drivers)
        # that return e.g. "OK" or None don't crash -- treat unknown
        # shapes as success (matches the prior silent-no-op default).
        if not isinstance(result, str):
            return True
        try:
            return int(result.rsplit(" ", 1)[-1]) > 0
        except (ValueError, IndexError):
            return True


__all__ = [
    "PostgresLandingPageRepository",
]
