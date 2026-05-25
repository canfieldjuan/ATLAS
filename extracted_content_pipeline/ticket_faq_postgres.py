"""Postgres repository adapter for ticket FAQ Markdown drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .campaign_ports import JsonDict, TenantScope
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    row_to_dict,
)
from .ticket_faq_ports import TicketFAQDraft
from .ticket_faq_search import (
    PostgresTicketFAQSearchRepository,
    build_ticket_faq_search_documents,
    build_ticket_faq_search_projection_key,
)


_TICKET_FAQ_COLUMNS = (
    "id, target_id, target_mode, title, markdown, items, "
    "source_count, ticket_source_count, output_checks, warnings, "
    "metadata, status"
)


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
class TicketFAQSearchBackfillResult:
    """Summary for a ticket FAQ search projection backfill run."""

    apply: bool
    status: str
    account_id: str | None
    limit: int | None
    scanned: int
    eligible_rows: int
    skipped_missing_key: int
    projected_documents: int
    applied_rows: int
    applied_documents: int

    def as_dict(self) -> JsonDict:
        return {
            "apply": self.apply,
            "status": self.status,
            "account_id": self.account_id,
            "limit": self.limit,
            "scanned": self.scanned,
            "eligible_rows": self.eligible_rows,
            "skipped_missing_key": self.skipped_missing_key,
            "projected_documents": self.projected_documents,
            "applied_rows": self.applied_rows,
            "applied_documents": self.applied_documents,
        }


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
            f"SELECT {_TICKET_FAQ_COLUMNS} "
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
        async def _update(db: Any) -> bool:
            rows = await db.fetch(
                f"""
                UPDATE ticket_faq_markdown
                   SET status = $2,
                       updated_at = NOW()
                 WHERE id = $1
                   AND account_id = $3
                RETURNING {_TICKET_FAQ_COLUMNS}
                """,
                faq_id,
                status,
                scope.account_id or "",
            )
            drafts = tuple(_row_to_draft(row_to_dict(row)) for row in rows)
            if not drafts:
                return False
            await self._replace_search_projection(drafts, scope=scope, db=db)
            return True

        return await _with_write_connection(self.pool, _update)

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
        async def _update(db: Any) -> Sequence[str]:
            rows = await db.fetch(
                f"""
                UPDATE ticket_faq_markdown
                   SET status = $2,
                       updated_at = NOW()
                 WHERE id = ANY($1::uuid[])
                   AND account_id = $3
                RETURNING {_TICKET_FAQ_COLUMNS}
                """,
                ids,
                status,
                scope.account_id or "",
            )
            drafts = tuple(_row_to_draft(row_to_dict(row)) for row in rows)
            if not drafts:
                return ()
            await self._replace_search_projection(drafts, scope=scope, db=db)
            return tuple(draft.id for draft in drafts)

        return await _with_write_connection(self.pool, _update)

    async def _replace_search_projection(
        self,
        drafts: Sequence[TicketFAQDraft],
        *,
        scope: TenantScope,
        db: Any | None = None,
    ) -> None:
        documents = []
        replace_keys = []
        for draft in drafts:
            key = build_ticket_faq_search_projection_key(
                draft,
                account_id=scope.account_id,
            )
            if not key.account_id or not key.corpus_id or not key.faq_id:
                continue
            replace_keys.append(key)
            documents.extend(
                build_ticket_faq_search_documents(
                    draft,
                    account_id=key.account_id,
                    corpus_id=key.corpus_id,
                )
            )
        if not replace_keys:
            return
        await PostgresTicketFAQSearchRepository(db or self.pool).replace_documents(
            tuple(documents),
            replace_keys=tuple(replace_keys),
        )


async def backfill_ticket_faq_search_documents(
    pool: Any,
    *,
    status: str = "approved",
    account_id: str | None = None,
    limit: int | None = None,
    apply: bool = False,
) -> TicketFAQSearchBackfillResult:
    """Backfill persisted FAQ drafts into the search projection table."""

    normalized_status = str(status or "").strip()
    if not normalized_status:
        raise ValueError("ticket FAQ search backfill requires status")

    normalized_account_id = str(account_id or "").strip() or None
    normalized_limit = None if limit is None else max(0, int(limit))
    clauses = ["status = $1"]
    params: list[Any] = [normalized_status]
    if normalized_account_id is not None:
        params.append(normalized_account_id)
        clauses.append(f"account_id = ${len(params)}")
    sql = (
        f"SELECT account_id, {_TICKET_FAQ_COLUMNS} "
        "FROM ticket_faq_markdown WHERE " + " AND ".join(clauses) + " "
        "ORDER BY updated_at DESC, created_at DESC"
    )
    if normalized_limit is not None:
        params.append(normalized_limit)
        sql += f" LIMIT ${len(params)}"

    rows = await pool.fetch(sql, *params)
    search_repo = PostgresTicketFAQSearchRepository(pool)
    scanned = 0
    eligible_rows = 0
    skipped_missing_key = 0
    projected_documents = 0
    applied_rows = 0
    applied_documents = 0

    for row in rows:
        scanned += 1
        row_dict = row_to_dict(row)
        draft = _row_to_draft(row_dict)
        key = build_ticket_faq_search_projection_key(
            draft,
            account_id=str(row_dict.get("account_id") or "").strip(),
        )
        if not key.account_id or not key.corpus_id or not key.faq_id:
            skipped_missing_key += 1
            continue
        documents = build_ticket_faq_search_documents(
            draft,
            account_id=key.account_id,
            corpus_id=key.corpus_id,
        )
        eligible_rows += 1
        projected_documents += len(documents)
        if apply:
            applied_documents += await search_repo.replace_documents(
                documents,
                replace_keys=(key,),
            )
            applied_rows += 1

    return TicketFAQSearchBackfillResult(
        apply=apply,
        status=normalized_status,
        account_id=normalized_account_id,
        limit=normalized_limit,
        scanned=scanned,
        eligible_rows=eligible_rows,
        skipped_missing_key=skipped_missing_key,
        projected_documents=projected_documents,
        applied_rows=applied_rows,
        applied_documents=applied_documents,
    )


async def _with_write_connection(db: Any, callback: Any) -> Any:
    transaction = getattr(db, "transaction", None)
    if callable(transaction):
        async with transaction():
            return await callback(db)
    acquire = getattr(db, "acquire", None)
    if callable(acquire):
        async with acquire() as connection:
            connection_transaction = getattr(connection, "transaction", None)
            if callable(connection_transaction):
                async with connection_transaction():
                    return await callback(connection)
            return await callback(connection)
    return await callback(db)


__all__ = [
    "PostgresTicketFAQRepository",
    "TicketFAQSearchBackfillResult",
    "backfill_ticket_faq_search_documents",
]
