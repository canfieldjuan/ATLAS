"""Postgres repository adapter for the AI Content Ops Sales Briefs product.

Mirrors the shape of ``PostgresLandingPageRepository`` (see
``landing_page_postgres.py``) but persists ``SalesBriefDraft`` rows into
the ``sales_briefs`` table from migration 275. Hosts inject an
asyncpg-style pool; the adapter does no connection management itself.

Carries forward the ergonomic improvements added during the
landing-pages slice: ``_coerce_section`` raises ``TypeError`` on
non-Mapping input rather than silently producing an empty section, and
``update_status`` returns a boolean parsed from asyncpg's command tag
so wrong-id / wrong-tenant calls surface as ``False`` at the call site
rather than a silent no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .campaign_ports import JsonDict, TenantScope
from .sales_brief_ports import SalesBriefDraft, SalesBriefSection


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _draft_metadata(draft: SalesBriefDraft, scope: TenantScope) -> JsonDict:
    return {
        **dict(draft.metadata or {}),
        "target_id": draft.target_id,
        "target_mode": draft.target_mode,
        "scope": {
            "account_id": scope.account_id,
            "user_id": scope.user_id,
        },
    }


def _coerce_section(value: Any) -> SalesBriefSection:
    """Coerce a host-supplied section into ``SalesBriefSection``.

    Accepts an existing ``SalesBriefSection`` or a Mapping with the
    expected keys. Anything else raises ``TypeError`` so host-side bugs
    surface at the persistence boundary rather than getting silently
    coerced into an empty section the quality pack later rejects.
    """
    if isinstance(value, SalesBriefSection):
        return value
    if isinstance(value, Mapping):
        return SalesBriefSection(
            id=str(value.get("id") or ""),
            title=str(value.get("title") or ""),
            body_markdown=str(value.get("body_markdown") or ""),
            claim_ids=tuple(str(c) for c in (value.get("claim_ids") or ())),
            evidence_ids=tuple(str(e) for e in (value.get("evidence_ids") or ())),
            metadata=dict(value.get("metadata") or {}),
        )
    raise TypeError(
        "SalesBriefDraft.sections entries must be SalesBriefSection "
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


def _row_to_draft(row: Mapping[str, Any]) -> SalesBriefDraft:
    sections_raw = _decode_jsonb(row.get("sections"), default=[])
    if not isinstance(sections_raw, Sequence) or isinstance(sections_raw, (str, bytes)):
        sections_raw = []

    reference_ids_raw = _decode_jsonb(row.get("reference_ids"), default=[])
    if not isinstance(reference_ids_raw, Sequence) or isinstance(reference_ids_raw, (str, bytes)):
        reference_ids_raw = []

    metadata_raw = _decode_jsonb(row.get("metadata"), default={})
    if not isinstance(metadata_raw, Mapping):
        metadata_raw = {}

    return SalesBriefDraft(
        target_id=str(row.get("target_id") or ""),
        target_mode=str(row.get("target_mode") or ""),
        brief_type=str(row.get("brief_type") or ""),
        title=str(row.get("title") or ""),
        headline=str(row.get("headline") or ""),
        sections=tuple(_coerce_section(s) for s in sections_raw),
        reference_ids=tuple(str(r) for r in reference_ids_raw),
        metadata=dict(metadata_raw),
    )


@dataclass(frozen=True)
class PostgresSalesBriefRepository:
    """Async Postgres adapter for generated sales briefs."""

    pool: Any

    async def save_drafts(
        self,
        drafts: Sequence[SalesBriefDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        saved: list[str] = []
        account_id = scope.account_id or ""
        for draft in drafts:
            sections_payload = [_coerce_section(s).as_dict() for s in draft.sections]
            reference_ids_payload = [str(r) for r in draft.reference_ids]
            metadata_payload = _draft_metadata(draft, scope)
            brief_id = await self.pool.fetchval(
                """
                INSERT INTO sales_briefs (
                    account_id, target_id, target_mode, brief_type,
                    title, headline,
                    sections, reference_ids, metadata, status
                )
                VALUES (
                    $1, $2, $3, $4,
                    $5, $6,
                    $7::jsonb, $8::jsonb, $9::jsonb, 'draft'
                )
                RETURNING id
                """,
                account_id,
                draft.target_id,
                draft.target_mode,
                draft.brief_type,
                draft.title,
                draft.headline,
                _jsonb(sections_payload),
                _jsonb(reference_ids_payload),
                _jsonb(metadata_payload),
            )
            saved.append(str(brief_id))
        return tuple(saved)

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        brief_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[SalesBriefDraft]:
        clauses: list[str] = ["account_id = $1"]
        params: list[Any] = [scope.account_id or ""]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if target_mode is not None:
            params.append(target_mode)
            clauses.append(f"target_mode = ${len(params)}")
        if brief_type is not None:
            params.append(brief_type)
            clauses.append(f"brief_type = ${len(params)}")
        sql = (
            "SELECT target_id, target_mode, brief_type, title, headline, "
            "sections, reference_ids, metadata "
            "FROM sales_briefs WHERE " + " AND ".join(clauses) + " "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            params.append(int(limit))
            sql += f" LIMIT ${len(params)}"
        rows = await self.pool.fetch(sql, *params)
        return tuple(_row_to_draft(_row_dict(row)) for row in rows)

    async def update_status(
        self,
        brief_id: str,
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
            UPDATE sales_briefs
               SET status = $2,
                   updated_at = NOW()
             WHERE id = $1
               AND account_id = $3
            """,
            brief_id,
            status,
            scope.account_id or "",
        )
        if not isinstance(result, str):
            return True
        try:
            return int(result.rsplit(" ", 1)[-1]) > 0
        except (ValueError, IndexError):
            return True


__all__ = [
    "PostgresSalesBriefRepository",
]
