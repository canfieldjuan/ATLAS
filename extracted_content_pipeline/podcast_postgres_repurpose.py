"""Postgres-backed runner for podcast multi-format repurposing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .campaign_llm_client import create_pipeline_llm_client
from .campaign_postgres_generation import tenant_scope_from_mapping
from .podcast_ports import (
    LLMClient,
    SkillStore,
    TenantScope,
)
from .podcast_postgres import (
    PostgresFormatDraftRepository,
    PostgresIdeaRepository,
    PostgresTranscriptRepository,
)
from .podcast_repurpose_generation import (
    SUPPORTED_FORMATS,
    PodcastRepurposeConfig,
    PodcastRepurposeResult,
    PodcastRepurposeService,
)
from .skills.registry import get_skill_registry


async def repurpose_podcast_episode_from_postgres(
    pool: Any,
    *,
    scope: Mapping[str, Any] | TenantScope | None = None,
    episode_id: str,
    formats: Sequence[str] = SUPPORTED_FORMATS,
    idea_limit: int = 3,
    voice_anchors: Mapping[str, Any] | None = None,
    llm: LLMClient | None = None,
    skills: SkillStore | None = None,
    config: PodcastRepurposeConfig | None = None,
    ideas_table: str = "podcast_extracted_ideas",
    transcripts_table: str = "podcast_transcripts",
    drafts_table: str = "podcast_format_drafts",
) -> PodcastRepurposeResult:
    """Generate and persist per-format drafts for one episode."""

    repurpose_config = config or PodcastRepurposeConfig(
        formats=tuple(formats) if formats else SUPPORTED_FORMATS,
        idea_limit=idea_limit,
        voice_anchors=dict(voice_anchors or {}),
    )
    service = PodcastRepurposeService(
        ideas=PostgresIdeaRepository(pool=pool, ideas_table=ideas_table),
        transcripts=PostgresTranscriptRepository(
            pool=pool,
            transcripts_table=transcripts_table,
        ),
        drafts=PostgresFormatDraftRepository(
            pool=pool,
            drafts_table=drafts_table,
        ),
        llm=llm or create_pipeline_llm_client(),
        skills=skills or get_skill_registry(),
        config=repurpose_config,
    )
    return await service.repurpose(
        scope=tenant_scope_from_mapping(scope),
        episode_id=episode_id,
        formats=tuple(formats) if formats else None,
        idea_limit=idea_limit,
    )


__all__ = ["repurpose_podcast_episode_from_postgres"]
