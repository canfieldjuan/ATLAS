"""Postgres-backed campaign draft generation runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .campaign_generation import (
    CampaignGenerationConfig,
    CampaignGenerationResult,
    CampaignGenerationService,
)
from .campaign_llm_client import create_pipeline_llm_client
from .campaign_ports import (
    CampaignReasoningContextProvider,
    LLMClient,
    SkillStore,
    TenantScope,
)
from .campaign_postgres import (
    PostgresCampaignRepository,
    PostgresIntelligenceRepository,
)
from .skills.registry import get_skill_registry


def tenant_scope_from_mapping(value: Mapping[str, Any] | TenantScope | None) -> TenantScope:
    """Build a TenantScope from CLI/host-provided mapping data."""

    if isinstance(value, TenantScope):
        return value
    if not isinstance(value, Mapping):
        return TenantScope()
    return TenantScope(
        account_id=str(value.get("account_id") or "") or None,
        user_id=str(value.get("user_id") or "") or None,
        allowed_vendors=tuple(str(item) for item in value.get("allowed_vendors") or ()),
        roles=tuple(str(item) for item in value.get("roles") or ()),
    )


async def generate_campaign_drafts_from_postgres(
    pool: Any,
    *,
    scope: Mapping[str, Any] | TenantScope | None = None,
    target_mode: str = "vendor_retention",
    channel: str = "email",
    channels: tuple[str, ...] = (),
    limit: int = 20,
    filters: Mapping[str, Any] | None = None,
    llm: LLMClient | None = None,
    skills: SkillStore | None = None,
    reasoning_context: CampaignReasoningContextProvider | None = None,
    config: CampaignGenerationConfig | None = None,
    opportunity_table: str = "campaign_opportunities",
    vendor_targets_table: str = "vendor_targets",
) -> CampaignGenerationResult:
    """Generate and persist campaign drafts from Postgres opportunity rows."""

    generation_config = config or CampaignGenerationConfig(
        channel=channel,
        channels=channels,
        limit=limit,
    )
    service = CampaignGenerationService(
        intelligence=PostgresIntelligenceRepository(
            pool,
            opportunity_table=opportunity_table,
            vendor_targets_table=vendor_targets_table,
        ),
        campaigns=PostgresCampaignRepository(pool),
        llm=llm or create_pipeline_llm_client(),
        skills=skills or get_skill_registry(),
        reasoning_context=reasoning_context,
        config=generation_config,
    )
    return await service.generate(
        scope=tenant_scope_from_mapping(scope),
        target_mode=target_mode,
        limit=limit,
        filters=filters,
    )


__all__ = [
    "generate_campaign_drafts_from_postgres",
    "tenant_scope_from_mapping",
]
