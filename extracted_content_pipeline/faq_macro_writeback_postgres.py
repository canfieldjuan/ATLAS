"""Postgres storage for FAQ macro writeback idempotency mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .campaign_ports import TenantScope
from .faq_macro_writeback import MacroWritebackMapping
from .faq_macro_writeback_publish import FAQMacroPublishSummary
from .storage._jsonb_helpers import decode_jsonb_field, json_dump_jsonb, row_to_dict


_MAPPING_COLUMNS = (
    "platform, faq_draft_id, faq_item_id, external_id, external_url, "
    "publish_status, metadata"
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _metadata(value: Any) -> dict[str, Any]:
    decoded = decode_jsonb_field(value, default={})
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _row_to_mapping(row: Mapping[str, Any]) -> MacroWritebackMapping:
    return MacroWritebackMapping(
        platform=str(row.get("platform") or ""),
        faq_draft_id=str(row.get("faq_draft_id") or ""),
        faq_item_id=str(row.get("faq_item_id") or ""),
        external_id=str(row.get("external_id") or ""),
        external_url=str(row.get("external_url") or ""),
        publish_status=str(row.get("publish_status") or "published"),
        metadata=_metadata(row.get("metadata")),
    )


@dataclass(frozen=True)
class PostgresFAQMacroWritebackMappingRepository:
    """Async Postgres adapter for FAQ macro writeback idempotency mappings."""

    pool: Any

    async def get_mapping(
        self,
        *,
        platform: str,
        faq_draft_id: str,
        faq_item_id: str,
        scope: TenantScope,
    ) -> MacroWritebackMapping | None:
        rows = await self.pool.fetch(
            f"""
            SELECT {_MAPPING_COLUMNS}
              FROM ticket_faq_macro_writebacks
             WHERE account_id = $1
               AND platform = $2
               AND faq_draft_id = $3
               AND faq_item_id = $4
             LIMIT 1
            """,
            scope.account_id or "",
            _clean(platform),
            _clean(faq_draft_id),
            _clean(faq_item_id),
        )
        if not rows:
            return None
        return _row_to_mapping(row_to_dict(rows[0]))

    async def upsert_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> MacroWritebackMapping:
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO ticket_faq_macro_writebacks (
                account_id, platform, faq_draft_id, faq_item_id,
                external_id, external_url, publish_status, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            ON CONFLICT (account_id, platform, faq_draft_id, faq_item_id)
            DO UPDATE SET
                external_id = EXCLUDED.external_id,
                external_url = EXCLUDED.external_url,
                publish_status = EXCLUDED.publish_status,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING {_MAPPING_COLUMNS}
            """,
            scope.account_id or "",
            _clean(mapping.platform),
            _clean(mapping.faq_draft_id),
            _clean(mapping.faq_item_id),
            _clean(mapping.external_id),
            _clean(mapping.external_url),
            _clean(mapping.publish_status) or "published",
            json_dump_jsonb(dict(mapping.metadata or {})),
        )
        return _row_to_mapping(row_to_dict(row))

    async def reserve_mapping(
        self,
        mapping: MacroWritebackMapping,
        *,
        scope: TenantScope,
    ) -> MacroWritebackMapping:
        row = await self.pool.fetchrow(
            f"""
            WITH inserted AS (
                INSERT INTO ticket_faq_macro_writebacks (
                    account_id, platform, faq_draft_id, faq_item_id,
                    external_id, external_url, publish_status, metadata
                )
                VALUES ($1, $2, $3, $4, '', '', 'pending', $5::jsonb)
                ON CONFLICT (account_id, platform, faq_draft_id, faq_item_id)
                DO NOTHING
                RETURNING {_MAPPING_COLUMNS}
            )
            SELECT {_MAPPING_COLUMNS} FROM inserted
            UNION ALL
            SELECT {_MAPPING_COLUMNS}
              FROM ticket_faq_macro_writebacks
             WHERE account_id = $1
               AND platform = $2
               AND faq_draft_id = $3
               AND faq_item_id = $4
            LIMIT 1
            """,
            scope.account_id or "",
            _clean(mapping.platform),
            _clean(mapping.faq_draft_id),
            _clean(mapping.faq_item_id),
            json_dump_jsonb(dict(mapping.metadata or {})),
        )
        return _row_to_mapping(row_to_dict(row))

    async def list_pending_mappings(
        self,
        *,
        platform: str,
        scope: TenantScope,
        limit: int,
    ) -> tuple[MacroWritebackMapping, ...]:
        rows = await self.pool.fetch(
            f"""
            SELECT {_MAPPING_COLUMNS}
              FROM ticket_faq_macro_writebacks
             WHERE account_id = $1
               AND platform = $2
               AND publish_status = 'pending'
               AND btrim(external_id) = ''
             ORDER BY updated_at ASC, faq_draft_id ASC, faq_item_id ASC
             LIMIT $3
            """,
            scope.account_id or "",
            _clean(platform),
            max(1, int(limit)),
        )
        return tuple(_row_to_mapping(row_to_dict(row)) for row in rows)


@dataclass(frozen=True)
class PostgresFAQMacroPublishAttemptRepository:
    """Async Postgres adapter for append-only FAQ macro publish attempts."""

    pool: Any

    async def record_attempt(
        self,
        summary: FAQMacroPublishSummary,
        *,
        scope: TenantScope,
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO ticket_faq_macro_publish_attempts (
                account_id, faq_draft_id, draft_status, ok,
                publishable_count, skipped_count, published_count,
                updated_count, failed_count, pending_reconcile_count,
                draft_status_updated, skipped, results
            )
            VALUES (
                $1, $2, $3, $4,
                $5, $6, $7,
                $8, $9, $10,
                $11, $12::jsonb, $13::jsonb
            )
            """,
            scope.account_id or "",
            _clean(summary.faq_id),
            _clean(summary.draft_status),
            bool(summary.ok),
            int(summary.publishable_count),
            int(summary.skipped_count),
            int(summary.published_count),
            int(summary.updated_count),
            int(summary.failed_count),
            int(summary.pending_reconcile_count),
            bool(summary.draft_status_updated),
            json_dump_jsonb([dict(item) for item in summary.skipped]),
            json_dump_jsonb([dict(item) for item in summary.results]),
        )


__all__ = [
    "PostgresFAQMacroPublishAttemptRepository",
    "PostgresFAQMacroWritebackMappingRepository",
]
