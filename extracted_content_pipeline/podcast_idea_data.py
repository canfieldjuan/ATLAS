"""File-backed idea provider for podcast extraction (BYO seam).

Mirrors campaign_reasoning_data.FileCampaignReasoningContextProvider. A
host that already has its own idea-extraction pipeline can persist the
output as JSON and inject FilePodcastIdeaProvider into
PodcastExtractionService to bypass the built-in single-pass LLM
extractor entirely.

Expected JSON shape:

    {
      "ideas": [
        {
          "episode_id": "ep-42",
          "rank": 1,
          "summary": "...",
          "arguments": [...],
          "hooks": [...],
          "key_quotes": [...],
          "teaching_moments": [...],
          "metadata": {...}
        },
        ...
      ]
    }

A bare top-level array is also accepted. Ideas without an ``episode_id``
are dropped silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .podcast_ports import (
    PodcastIdea,
    PodcastTranscript,
    TenantScope,
)


_ROW_KEYS = ("ideas", "rows", "data")


@dataclass(frozen=True)
class FilePodcastIdeaProvider:
    """IdeaExtractor backed by pre-baked ideas indexed by episode_id."""

    ideas_by_episode: Mapping[str, tuple[PodcastIdea, ...]]
    source: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "FilePodcastIdeaProvider":
        source = Path(path)
        return cls.from_payload(
            json.loads(source.read_text(encoding="utf-8")),
            source=str(source),
        )

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        source: str | None = None,
    ) -> "FilePodcastIdeaProvider":
        rows = _idea_rows(payload)
        # Track per-episode insertion order so missing/invalid ranks fall
        # back to a 1-based index instead of 0 (which would collide on the
        # (account_id, episode_id, rank) UNIQUE constraint at persist time).
        per_episode_counter: dict[str, int] = {}
        indexed: dict[str, list[PodcastIdea]] = {}
        for row in rows:
            episode_id = str(row.get("episode_id") or "").strip()
            if not episode_id:
                continue
            per_episode_counter[episode_id] = per_episode_counter.get(episode_id, 0) + 1
            fallback_rank = per_episode_counter[episode_id]
            idea = _idea_from_row(
                row,
                episode_id=episode_id,
                fallback_rank=fallback_rank,
            )
            indexed.setdefault(episode_id, []).append(idea)
        sorted_index: dict[str, tuple[PodcastIdea, ...]] = {}
        for episode_id, ideas in indexed.items():
            ideas.sort(key=lambda idea: idea.rank)
            sorted_index[episode_id] = tuple(ideas)
        return cls(ideas_by_episode=sorted_index, source=source)

    async def extract_ideas(
        self,
        *,
        scope: TenantScope,
        transcript: PodcastTranscript,
        target_idea_count: int,
    ) -> Sequence[PodcastIdea]:
        del scope
        ideas = self.ideas_by_episode.get(str(transcript.episode_id), ())
        return ideas[: max(0, int(target_idea_count))]


def load_podcast_idea_provider(path: str | Path) -> FilePodcastIdeaProvider:
    """Load a file-backed podcast idea provider from JSON."""

    return FilePodcastIdeaProvider.from_file(path)


def _idea_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("podcast idea JSON must be an object or array")
    for key in _ROW_KEYS:
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [row for row in value if isinstance(row, Mapping)]
    return [payload]


def _idea_from_row(
    row: Mapping[str, Any],
    *,
    episode_id: str,
    fallback_rank: int = 1,
) -> PodcastIdea:
    rank = row.get("rank")
    try:
        rank_int = int(rank) if rank is not None else fallback_rank
    except (TypeError, ValueError):
        rank_int = fallback_rank
    # Rank is 1-indexed by contract. A rank of 0 / negative / missing
    # would collapse to 0 and collide on the
    # (account_id, episode_id, rank) UNIQUE constraint at persist time;
    # fall back to the per-episode insertion order instead.
    if rank_int < 1:
        rank_int = fallback_rank
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return PodcastIdea(
        episode_id=episode_id,
        rank=rank_int,
        summary=str(row.get("summary") or "").strip(),
        arguments=tuple(_string_list(row.get("arguments"))),
        hooks=tuple(_string_list(row.get("hooks"))),
        key_quotes=tuple(_string_list(row.get("key_quotes") or row.get("quotes"))),
        teaching_moments=tuple(_string_list(row.get("teaching_moments"))),
        metadata=dict(metadata),
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        cleaned = text.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value or "").strip()
    return [text] if text else []


__all__ = [
    "FilePodcastIdeaProvider",
    "load_podcast_idea_provider",
]
