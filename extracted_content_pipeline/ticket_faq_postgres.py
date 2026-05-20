"""Postgres repository adapter for ticket FAQ Markdown drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .campaign_ports import JsonDict, TenantScope
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    parse_command_tag,
    row_to_dict,
)
from .ticket_faq_ports import TicketFAQDraft


def _draft_metadata(draft: TicketFAQDraft, scope: TenantScope) -> JsonDict:
    return {
        **dict(draft.metadata or {}),
        "target_id": draft.target_id,
        "target_mode": draft.target_mode,
        "scope": {
            "account_id": scope.account_id,
            "user_id": scope.user_id,
        },
    }


def _json_mapping(value: Any) -> dict[str, Any]:
    decoded = decode_jsonb_field(value, default={})
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _json_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    decoded = decode_jsonb_field(value, default=[])
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        return ()
    return tuple(item for item in decoded if isinstance(item, Mapping))


def _row_to_draft(row: Mapping[str, Any]) -> TicketFAQDraft:
    return TicketFAQDraft(
        id=str(row.get("id") or ""),
        target_id=str(row.get("target_id") or ""),
        target_mode=str(row.get("target_mode") or ""),
        title=str(row.get("title") or ""),
        markdown=str(row.get("markdown") or ""),
        items=_json_sequence(row.get("items")),
        source_count=int(row.get("source_count") or 0),
        ticket_source_count=int(row.get("ticket_source_count") or 0),
        output_checks=_json_mapping(row.get("output_checks")),
        warnings=_json_sequence(row.get("warnings")),
        metadata=_json_mapping(row.get("metadata")),
        status=str(row.get("status") or ""),
    )


@dataclass(frozen=True)
class PostgresTicketFAQRepository:
    """Async Postgres adapter for generated ticket FAQ Markdown."""

    pool: Any

    async def save_drafts(
        self,
        drafts: Sequence[TicketFAQDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        saved: list[str] = []
        account_id = scope.account_id or ""
        for draft in drafts:
            faq_id = await self.pool.fetchval(
                """
                INSERT INTO ticket_faq_markdown (
                    account_id, target_id, target_mode, title, markdown,
                    items, source_count, ticket_source_count, output_checks,
                    warnings, metadata, status
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6::jsonb, $7, $8, $9::jsonb,
                    $10::jsonb, $11::jsonb, 'draft'
                )
                RETURNING id
                """,
                account_id,
                draft.target_id,
                draft.target_mode,
                draft.title,
                draft.markdown,
                json_dump_jsonb([dict(item) for item in draft.items]),
                int(draft.source_count),
                int(draft.ticket_source_count),
                json_dump_jsonb(dict(draft.output_checks or {})),
                json_dump_jsonb([dict(warning) for warning in draft.warnings]),
                json_dump_jsonb(_draft_metadata(draft, scope)),
            )
            saved.append(str(faq_id))
        return tuple(saved)

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        limit: int | None = None,
    ) -> Sequence[TicketFAQDraft]:
        clauses: list[str] = ["account_id = $1"]
        params: list[Any] = [scope.account_id or ""]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if target_mode is not None:
            params.append(target_mode)
            clauses.append(f"target_mode = ${len(params)}")
        sql = (
            "SELECT id, target_id, target_mode, title, markdown, items, "
            "source_count, ticket_source_count, output_checks, warnings, "
            "metadata, status "
            "FROM ticket_faq_markdown WHERE " + " AND ".join(clauses) + " "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            params.append(int(limit))
            sql += f" LIMIT ${len(params)}"
        rows = await self.pool.fetch(sql, *params)
        return tuple(_row_to_draft(row_to_dict(row)) for row in rows)

    async def update_status(
        self,
        faq_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        result = await self.pool.execute(
            """
            UPDATE ticket_faq_markdown
               SET status = $2,
                   updated_at = NOW()
             WHERE id = $1
               AND account_id = $3
            """,
            faq_id,
            status,
            scope.account_id or "",
        )
        return parse_command_tag(result)

    async def update_statuses(
        self,
        faq_ids: Sequence[str],
        status: str,
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        ids = [str(item).strip() for item in faq_ids if str(item).strip()]
        if not ids:
            return ()
        rows = await self.pool.fetch(
            """
            UPDATE ticket_faq_markdown
               SET status = $2,
                   updated_at = NOW()
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
    "PostgresTicketFAQRepository",
]
