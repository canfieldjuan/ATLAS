"""Autonomous task for scheduled FAQ-to-Zendesk macro writeback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import (
    APPROVED_FAQ_STATUS,
    MacroWritebackMappingRepository,
    SupportMacroDraft,
    build_macro_writeback_preview,
    macro_content_hash,
)
from extracted_content_pipeline.faq_macro_writeback_postgres import (
    PostgresFAQMacroPublishAttemptRepository,
    PostgresFAQMacroWritebackMappingRepository,
)
from extracted_content_pipeline.faq_macro_writeback_publish import FAQMacroWritebackPublishService
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZENDESK_PLATFORM,
    ZendeskMacroCredentials,
    ZendeskMacroPublishProvider,
)
from extracted_content_pipeline.ticket_faq_ports import TicketFAQRepository
from extracted_content_pipeline.ticket_faq_postgres import PostgresTicketFAQRepository

from ..._content_ops_zendesk_credentials import lookup_zendesk_credentials
from ...config import settings
from ...storage.database import get_db_pool

logger = logging.getLogger("atlas.autonomous.tasks.faq_macro_writeback_scheduled_publish")
MAX_TENANTS_PER_RUN = 25
MAX_DRAFTS_PER_TENANT = 25


@dataclass(frozen=True)
class ZendeskTenant:
    account_id: str
    has_zendesk_credentials: bool = True


@dataclass(frozen=True)
class StoredOnlyZendeskMacroCredentialsProvider:
    pool: Any

    async def credentials_for_scope(self, scope: TenantScope) -> ZendeskMacroCredentials | None:
        if not scope.account_id:
            return None
        return await lookup_zendesk_credentials(self.pool, account_id=scope.account_id)


@dataclass
class PostgresScheduledFAQMacroCandidateRepository:
    pool: Any
    faq_repository: TicketFAQRepository
    mapping_repository: MacroWritebackMappingRepository
    last_enrolled_tenant_count: int = 0

    async def list_tenants(self, *, limit: int) -> Sequence[ZendeskTenant]:
        rows = await self.pool.fetch(
            """
            WITH enrolled AS (
                SELECT DISTINCT z.account_id::text AS account_id
                  FROM content_ops_zendesk_credentials z
                 WHERE z.revoked_at IS NULL
            ),
            last_publish AS (
                SELECT account_id::text AS account_id, MAX(updated_at) AS last_published_at
                  FROM ticket_faq_macro_writebacks
                 WHERE platform = $2
                   AND publish_status = 'published'
                 GROUP BY account_id
            )
            SELECT e.account_id,
                   COUNT(*) OVER() AS enrolled_count
              FROM enrolled e
              LEFT JOIN last_publish p ON p.account_id = e.account_id
             ORDER BY p.last_published_at ASC NULLS FIRST,
                      hashtext(e.account_id || date_trunc('hour', NOW())::text) ASC,
                      e.account_id ASC
             LIMIT $1
            """,
            max(1, int(limit)),
            ZENDESK_PLATFORM,
        )
        self.last_enrolled_tenant_count = int(rows[0]["enrolled_count"] or 0) if rows else 0
        return tuple(ZendeskTenant(account_id=str(row["account_id"] or "")) for row in rows)

    async def list_candidate_faq_ids(self, tenant: ZendeskTenant, *, limit: int) -> Sequence[str]:
        scope = TenantScope(account_id=tenant.account_id)
        drafts = await self.faq_repository.list_drafts(
            scope=scope,
            status=APPROVED_FAQ_STATUS,
            limit=max(1, int(limit)),
        )
        selected: list[str] = []
        for draft in drafts:
            preview = build_macro_writeback_preview([draft])
            if not preview.macros:
                continue
            if await _has_unpublished_macro(preview.macros, self.mapping_repository, scope=scope):
                selected.append(draft.id)
        return tuple(selected)


async def run_scheduled_faq_macro_writeback(*, candidate_repository: Any, publish_service: Any, max_tenants: int, max_drafts_per_tenant: int) -> dict[str, Any]:
    tenants = await candidate_repository.list_tenants(limit=max_tenants)
    enrolled_tenant_count = int(getattr(candidate_repository, "last_enrolled_tenant_count", len(tenants)) or len(tenants))
    tenants_deferred = max(0, enrolled_tenant_count - len(tenants))
    result: dict[str, Any] = {
        "enrolled_tenants": enrolled_tenant_count,
        "tenants_checked": len(tenants),
        "tenants_deferred_by_limit": tenants_deferred,
        "tenants_skipped_no_credentials": 0,
        "drafts_selected": 0,
        "drafts_published_ok": 0,
        "drafts_failed": 0,
        "tenants": [],
        "_skip_synthesis": "Scheduled FAQ macro writeback complete",
    }
    if tenants_deferred:
        logger.warning(
            "scheduled FAQ macro publish deferred %s of %s enrolled tenants due to max_tenants=%s",
            tenants_deferred,
            enrolled_tenant_count,
            max_tenants,
        )
    for tenant in tenants:
        tenant_result: dict[str, Any] = {"account_id": tenant.account_id, "drafts": []}
        if not tenant.has_zendesk_credentials:
            result["tenants_skipped_no_credentials"] += 1
            tenant_result["status"] = "skipped_no_zendesk_credentials"
            result["tenants"].append(tenant_result)
            continue
        scope = TenantScope(account_id=tenant.account_id)
        faq_ids = tuple(await candidate_repository.list_candidate_faq_ids(tenant, limit=max_drafts_per_tenant))
        result["drafts_selected"] += len(faq_ids)
        for faq_id in faq_ids:
            try:
                summary = await publish_service.publish_faq_draft(faq_id, scope=scope)
                item = summary.as_dict()
            except Exception as exc:
                logger.warning("scheduled FAQ macro publish failed account_id=%s faq_id=%s: %s", tenant.account_id, faq_id, exc)
                item = {"faq_id": faq_id, "ok": False, "error": exc.__class__.__name__}
            if item.get("ok"):
                result["drafts_published_ok"] += 1
            else:
                result["drafts_failed"] += 1
            tenant_result["drafts"].append(item)
        tenant_result["status"] = "processed" if faq_ids else "no_candidates"
        result["tenants"].append(tenant_result)
    return result


async def run(task: Any) -> dict[str, Any]:
    del task
    cfg = settings.b2b_campaign
    if not cfg.content_ops_faq_macro_writeback_scheduled_enabled:
        return {"_skip_synthesis": "Scheduled FAQ macro writeback disabled"}
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database pool not initialized"}
    mapping_repository = PostgresFAQMacroWritebackMappingRepository(pool)
    faq_repository = PostgresTicketFAQRepository(pool=pool)
    provider = ZendeskMacroPublishProvider(
        credentials_provider=StoredOnlyZendeskMacroCredentialsProvider(pool),
        mapping_repository=mapping_repository,
    )
    publish_service = FAQMacroWritebackPublishService(
        faq_repository=faq_repository,
        provider=provider,
        attempt_repository=PostgresFAQMacroPublishAttemptRepository(pool),
    )
    return await run_scheduled_faq_macro_writeback(
        candidate_repository=PostgresScheduledFAQMacroCandidateRepository(
            pool=pool,
            faq_repository=faq_repository,
            mapping_repository=mapping_repository,
        ),
        publish_service=publish_service,
        max_tenants=MAX_TENANTS_PER_RUN,
        max_drafts_per_tenant=MAX_DRAFTS_PER_TENANT,
    )


async def _has_unpublished_macro(
    macros: Sequence[SupportMacroDraft],
    mapping_repository: MacroWritebackMappingRepository,
    *,
    scope: TenantScope,
) -> bool:
    for macro in macros:
        mapping = await mapping_repository.get_mapping(
            platform=ZENDESK_PLATFORM,
            faq_draft_id=macro.faq_draft_id,
            faq_item_id=macro.faq_item_id,
            scope=scope,
        )
        if (
            mapping is None
            or mapping.publish_status != "published"
            or not mapping.external_id
            or _clean(mapping.metadata.get("title")) != _clean(macro.title)
            or _clean(mapping.metadata.get("category")) != _clean(macro.category)
            or _clean(mapping.metadata.get("content_hash")) != macro_content_hash(macro)
        ):
            return True
    return False


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
