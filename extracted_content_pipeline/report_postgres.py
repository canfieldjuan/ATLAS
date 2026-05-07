"""Postgres repository adapter for the AI Content Ops Reports product.

Mirrors the shape of ``PostgresCampaignRepository`` (see
``campaign_postgres.py:240-292``) but persists ``ReportDraft`` rows into
the ``reports`` table from migration 273. Hosts inject an asyncpg-style
pool; the adapter does no connection management itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .campaign_ports import JsonDict, TenantScope
from .report_ports import ReportDraft, ReportSection


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _row_dict(row: Mapping[str, Any] | Any) -> JsonDict:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def _draft_metadata(draft: ReportDraft, scope: TenantScope) -> JsonDict:
    """Merge tenant scope into the draft's metadata before persisting."""
    return {
        **dict(draft.metadata or {}),
        "target_id": draft.target_id,
        "target_mode": draft.target_mode,
        "scope": {
            "account_id": scope.account_id,
            "user_id": scope.user_id,
        },
    }


def _coerce_section(value: Any) -> ReportSection:
    if isinstance(value, ReportSection):
        return value
    if isinstance(value, Mapping):
        return ReportSection(
            id=str(value.get("id") or ""),
            title=str(value.get("title") or ""),
            body_markdown=str(value.get("body_markdown") or ""),
            claim_ids=tuple(str(c) for c in (value.get("claim_ids") or ())),
            evidence_ids=tuple(str(e) for e in (value.get("evidence_ids") or ())),
            metadata=dict(value.get("metadata") or {}),
        )
    return ReportSection(id="", title="", body_markdown=str(value or ""))


def _row_to_draft(row: Mapping[str, Any]) -> ReportDraft:
    sections_raw = row.get("sections")
    if isinstance(sections_raw, str):
        try:
            sections_raw = json.loads(sections_raw)
        except (TypeError, ValueError):
            sections_raw = []
    if not isinstance(sections_raw, Sequence) or isinstance(sections_raw, (str, bytes)):
        sections_raw = []

    reference_ids_raw = row.get("reference_ids")
    if isinstance(reference_ids_raw, str):
        try:
            reference_ids_raw = json.loads(reference_ids_raw)
        except (TypeError, ValueError):
            reference_ids_raw = []
    if not isinstance(reference_ids_raw, Sequence) or isinstance(reference_ids_raw, (str, bytes)):
        reference_ids_raw = []

    metadata_raw = row.get("metadata")
    if isinstance(metadata_raw, str):
        try:
            metadata_raw = json.loads(metadata_raw)
        except (TypeError, ValueError):
            metadata_raw = {}
    if not isinstance(metadata_raw, Mapping):
        metadata_raw = {}

    return ReportDraft(
        target_id=str(row.get("target_id") or ""),
        target_mode=str(row.get("target_mode") or ""),
        report_type=str(row.get("report_type") or ""),
        title=str(row.get("title") or ""),
        summary=str(row.get("summary") or ""),
        sections=tuple(_coerce_section(s) for s in sections_raw),
        reference_ids=tuple(str(r) for r in reference_ids_raw),
        metadata=dict(metadata_raw),
    )


@dataclass(frozen=True)
class PostgresReportRepository:
    """Async Postgres adapter for generated structured reports."""

    pool: Any

    async def save_drafts(
        self,
        drafts: Sequence[ReportDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        saved: list[str] = []
        account_id = scope.account_id or ""
        for draft in drafts:
            sections_payload = [_coerce_section(s).as_dict() for s in draft.sections]
            reference_ids_payload = [str(r) for r in draft.reference_ids]
            metadata_payload = _draft_metadata(draft, scope)
            report_id = await self.pool.fetchval(
                """
                INSERT INTO reports (
                    account_id, target_id, target_mode, report_type,
                    title, summary, sections, reference_ids, metadata, status
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, 'draft'
                )
                RETURNING id
                """,
                account_id,
                draft.target_id,
                draft.target_mode,
                draft.report_type,
                draft.title,
                draft.summary,
                _jsonb(sections_payload),
                _jsonb(reference_ids_payload),
                _jsonb(metadata_payload),
            )
            saved.append(str(report_id))
        return tuple(saved)

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        report_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[ReportDraft]:
        clauses: list[str] = ["account_id = $1"]
        params: list[Any] = [scope.account_id or ""]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if target_mode is not None:
            params.append(target_mode)
            clauses.append(f"target_mode = ${len(params)}")
        if report_type is not None:
            params.append(report_type)
            clauses.append(f"report_type = ${len(params)}")
        sql = (
            "SELECT target_id, target_mode, report_type, title, summary, "
            "sections, reference_ids, metadata "
            "FROM reports WHERE " + " AND ".join(clauses) + " "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            params.append(int(limit))
            sql += f" LIMIT ${len(params)}"
        rows = await self.pool.fetch(sql, *params)
        return tuple(_row_to_draft(_row_dict(row)) for row in rows)

    async def update_status(
        self,
        report_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> None:
        await self.pool.execute(
            """
            UPDATE reports
               SET status = $2,
                   updated_at = NOW()
             WHERE id = $1
               AND account_id = $3
            """,
            report_id,
            status,
            scope.account_id or "",
        )


__all__ = [
    "PostgresReportRepository",
]
