"""Standalone ports for the podcast repurposing product.

Reuses LLMClient, LLMMessage, LLMResponse, TenantScope, JsonDict, and
SkillStore from campaign_ports so the two products share the same host
contracts. Defines podcast-specific dataclasses and Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .campaign_ports import (
    JsonDict,
    LLMClient,
    LLMMessage,
    LLMResponse,
    SkillStore,
    TenantScope,
)

__all__ = [
    "JsonDict",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "SkillStore",
    "TenantScope",
    "PodcastTranscript",
    "PodcastIdea",
    "PodcastFormatDraft",
    "TranscriptRepository",
    "IdeaRepository",
    "FormatDraftRepository",
    "IdeaExtractor",
]


@dataclass(frozen=True)
class PodcastTranscript:
    """One episode's pre-transcribed text plus metadata."""

    episode_id: str
    title: str = ""
    transcript_text: str = ""
    duration_seconds: int | None = None
    publish_date: str | None = None
    host_name: str | None = None
    guest_name: str | None = None
    source_url: str | None = None
    raw_payload: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PodcastIdea:
    """One ranked idea extracted from a podcast transcript."""

    episode_id: str
    rank: int
    summary: str
    arguments: tuple[str, ...] = ()
    hooks: tuple[str, ...] = ()
    key_quotes: tuple[str, ...] = ()
    teaching_moments: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)
    idea_id: str | None = None


@dataclass(frozen=True)
class PodcastFormatDraft:
    """One per-format draft generated from an extracted idea."""

    episode_id: str
    format_type: str
    title: str
    body: str
    metadata: JsonDict = field(default_factory=dict)
    quality_audit: JsonDict = field(default_factory=dict)
    idea_id: str | None = None


class TranscriptRepository(Protocol):
    async def read_transcripts(
        self,
        *,
        scope: TenantScope,
        episode_id: str | None = None,
        limit: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[PodcastTranscript]:
        """Return transcripts matching the scope and optional episode filter."""


class IdeaRepository(Protocol):
    async def save_ideas(
        self,
        ideas: Sequence[PodcastIdea],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist extracted ideas and return their database ids."""

    async def read_ideas(
        self,
        *,
        scope: TenantScope,
        episode_id: str,
        limit: int = 10,
    ) -> Sequence[PodcastIdea]:
        """Return previously saved ideas for an episode in rank order."""


class FormatDraftRepository(Protocol):
    async def save_drafts(
        self,
        drafts: Sequence[PodcastFormatDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[str]:
        """Persist per-format drafts and return their database ids."""


class IdeaExtractor(Protocol):
    """Optional BYO seam.

    A host that already has its own idea-extraction pipeline (a reasoning
    engine, a curator workflow, anything) can implement this Protocol and
    inject it into PodcastExtractionService to bypass the built-in single-
    pass LLM extractor.
    """

    async def extract_ideas(
        self,
        *,
        scope: TenantScope,
        transcript: PodcastTranscript,
        target_idea_count: int,
    ) -> Sequence[PodcastIdea]:
        """Return ranked ideas for a transcript."""
