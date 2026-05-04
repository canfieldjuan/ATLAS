"""Postgres-backed campaign sequence progression runner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .campaign_llm_client import create_pipeline_llm_client
from .campaign_ports import LLMClient, SkillStore
from .campaign_postgres import (
    PostgresCampaignAuditSink,
    PostgresCampaignSequenceRepository,
)
from .campaign_sequence_progression import (
    CampaignSequenceProgressionConfig,
    CampaignSequenceProgressionResult,
    CampaignSequenceProgressionService,
)
from .skills.registry import get_skill_registry


async def progress_campaign_sequences_from_postgres(
    pool: Any,
    *,
    llm: LLMClient | None = None,
    skills: SkillStore | None = None,
    config: CampaignSequenceProgressionConfig | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    from_email: str | None = None,
) -> CampaignSequenceProgressionResult:
    """Generate and queue due sequence follow-ups from Postgres rows."""

    effective_config = _sequence_progression_config(
        config=config,
        limit=limit,
        max_steps=max_steps,
        from_email=from_email,
    )
    if effective_config.enabled and effective_config.batch_limit <= 0:
        return CampaignSequenceProgressionResult()

    service = CampaignSequenceProgressionService(
        sequences=PostgresCampaignSequenceRepository(pool),
        llm=llm or create_pipeline_llm_client(),
        skills=skills or get_skill_registry(),
        audit=PostgresCampaignAuditSink(pool),
        config=effective_config,
    )
    return await service.progress_due()


def _sequence_progression_config(
    *,
    config: CampaignSequenceProgressionConfig | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    from_email: str | None = None,
) -> CampaignSequenceProgressionConfig:
    base = config or CampaignSequenceProgressionConfig()
    updates: dict[str, Any] = {}
    if limit is not None:
        updates["batch_limit"] = int(limit)
    if max_steps is not None:
        updates["max_steps"] = int(max_steps)
    if from_email is not None:
        updates["from_email"] = str(from_email)
    return replace(base, **updates) if updates else base


__all__ = [
    "progress_campaign_sequences_from_postgres",
]
