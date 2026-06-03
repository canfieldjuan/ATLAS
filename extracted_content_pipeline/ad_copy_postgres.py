"""Postgres repository adapter for generated ad-copy drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ad_copy_ports import AdCopyDraft
from .campaign_ports import JsonDict, TenantScope
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    parse_command_tag,
    row_to_dict,
)


_AD_COPY_COLUMNS = (
    "id, target_id, target_mode, channel, format, headline, primary_text, cta, "
    "source_id, source_type, company_name, vendor_name, pain_points, metadata, "
    "status"
)


def _draft_metadata(draft: AdCopyDraft, scope: TenantScope) -> JsonDict:
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


def _json_string_sequence(value: Any) -> tuple[str, ...]:
    decoded = decode_jsonb_field(value, default=[])
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        return ()
    return tuple(str(item) for item in decoded if str(item).strip())


def _row_to_draft(row: Mapping[str, Any]) -> AdCopyDraft:
    return AdCopyDraft(
        id=str(row.get("id") or ""),
        target_id=str(row.get("target_id") or ""),
        target_mode=str(row.get("target_mode") or ""),
        channel=str(row.get("channel") or ""),
        format=str(row.get("format") or ""),
        headline=str(row.get("headline") or ""),
        primary_text=str(row.get("primary_text") or ""),
        cta=str(row.get("cta") or ""),
        source_id=str(row.get("source_id") or ""),
        source_type=str(row.get("source_type") or ""),
        company_name=str(row.get("company_name") or ""),
        vendor_name=str(row.get("vendor_name") or ""),
        pain_points=_json_string_sequence(row.get("pain_points")),
        metadata=_json_mapping(row.get("metadata")),
        status=str(row.get("status") or ""),
    )


@dataclass(frozen=True)
class PostgresAdCopyRepository:
    """Async Postgres adapter for generated ad copy."""

    pool: Any

    async def save_drafts(
        self,
        drafts: Sequence[AdCopyDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        saved: list[str] = []
        account_id = scope.account_id or ""
        for draft in drafts:
            draft_id = await self.pool.fetchval(
                """
                INSERT INTO ad_copy_drafts (
                    account_id, target_id, target_mode, channel, format,
                    headline, primary_text, cta, source_id, source_type,
                    company_name, vendor_name, pain_points, metadata, status
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10,
                    $11, $12, $13::jsonb, $14::jsonb, 'draft'
                )
                RETURNING id
                """,
                account_id,
                draft.target_id,
                draft.target_mode,
                draft.channel,
                draft.format,
                draft.headline,
                draft.primary_text,
                draft.cta,
                draft.source_id,
                draft.source_type,
                draft.company_name,
                draft.vendor_name,
                json_dump_jsonb([str(item) for item in draft.pain_points]),
                json_dump_jsonb(_draft_metadata(draft, scope)),
            )
            saved.append(str(draft_id))
        return tuple(saved)

    async def list_drafts(
        self,
        *,
        scope: TenantScope,
        status: str | None = None,
        target_mode: str | None = None,
        channel: str | None = None,
        limit: int | None = None,
    ) -> Sequence[AdCopyDraft]:
        clauses: list[str] = ["account_id = $1"]
        params: list[Any] = [scope.account_id or ""]
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        if target_mode is not None:
            params.append(target_mode)
            clauses.append(f"target_mode = ${len(params)}")
        if channel is not None:
            params.append(channel)
            clauses.append(f"channel = ${len(params)}")
        sql = (
            f"SELECT {_AD_COPY_COLUMNS} "
            "FROM ad_copy_drafts WHERE " + " AND ".join(clauses) + " "
            "ORDER BY created_at DESC"
        )
        if limit is not None:
            params.append(int(limit))
            sql += f" LIMIT ${len(params)}"
        rows = await self.pool.fetch(sql, *params)
        return tuple(_row_to_draft(row_to_dict(row)) for row in rows)

    async def update_status(
        self,
        draft_id: str,
        status: str,
        *,
        scope: TenantScope,
    ) -> bool:
        result = await self.pool.execute(
            """
            UPDATE ad_copy_drafts
               SET status = $2,
                   updated_at = NOW()
             WHERE id = $1::uuid
               AND account_id = $3
            """,
            draft_id,
            status,
            scope.account_id or "",
        )
        return parse_command_tag(result)

    async def update_statuses(
        self,
        draft_ids: Sequence[str],
        status: str,
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        ids = [str(item).strip() for item in draft_ids if str(item).strip()]
        if not ids:
            return ()
        rows = await self.pool.fetch(
            """
            UPDATE ad_copy_drafts
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
    "PostgresAdCopyRepository",
]
