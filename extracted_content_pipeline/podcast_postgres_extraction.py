"""Postgres-backed runner for podcast idea extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .campaign_llm_client import create_pipeline_llm_client
from .campaign_postgres_generation import tenant_scope_from_mapping
from .podcast_extraction import (
    PodcastExtractionConfig,
    PodcastExtractionResult,
    PodcastExtractionService,
)
from .podcast_ports import (
    IdeaExtractor,
    LLMClient,
    SkillStore,
    TenantScope,
)
from .podcast_postgres import (
    PostgresIdeaRepository,
    PostgresTranscriptRepository,
)
from .skills.registry import get_skill_registry


async def extract_podcast_ideas_from_postgres(
    pool: Any,
    *,
    scope: Mapping[str, Any] | TenantScope | None = None,
    episode_id: str | None = None,
    limit: int = 1,
    target_idea_count: int = 3,
    llm: LLMClient | None = None,
    skills: SkillStore | None = None,
    extractor: IdeaExtractor | None = None,
    config: PodcastExtractionConfig | None = None,
    transcripts_table: str = "podcast_transcripts",
    ideas_table: str = "podcast_extracted_ideas",
) -> PodcastExtractionResult:
    """Extract and persist podcast ideas from Postgres transcript rows."""

    extraction_config = config or PodcastExtractionConfig(
        target_idea_count=target_idea_count,
    )
    service = PodcastExtractionService(
        transcripts=PostgresTranscriptRepository(
            pool=pool,
            transcripts_table=transcripts_table,
        ),
        ideas=PostgresIdeaRepository(pool=pool, ideas_table=ideas_table),
        llm=llm or create_pipeline_llm_client(),
        skills=skills or get_skill_registry(),
        extractor=extractor,
        config=extraction_config,
    )
    return await service.extract(
        scope=tenant_scope_from_mapping(scope),
        episode_id=episode_id,
        limit=limit,
    )


__all__ = ["extract_podcast_ideas_from_postgres"]
