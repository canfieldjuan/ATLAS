"""Podcast idea extraction orchestration.

Single-pass LLM call per transcript; produces ranked PodcastIdea rows.
The service can be configured with an optional IdeaExtractor that bypasses
the LLM path entirely (the BYO seam used by FilePodcastIdeaProvider).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import re
from typing import Any

from .podcast_ports import (
    IdeaExtractor,
    IdeaRepository,
    LLMClient,
    LLMMessage,
    PodcastIdea,
    PodcastTranscript,
    SkillStore,
    TenantScope,
    TranscriptRepository,
)


@dataclass(frozen=True)
class PodcastExtractionConfig:
    skill_name: str = "digest/podcast_idea_extraction"
    target_idea_count: int = 3
    max_tokens: int = 4000
    temperature: float = 0.3


@dataclass(frozen=True)
class PodcastExtractionResult:
    requested: int = 0
    ideas_extracted: int = 0
    skipped: int = 0
    saved_ids: tuple[str, ...] = ()
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "ideas_extracted": self.ideas_extracted,
            "skipped": self.skipped,
            "saved_ids": list(self.saved_ids),
            "errors": list(self.errors),
        }


def parse_podcast_idea_response(text: str) -> list[dict[str, Any]] | None:
    """Extract a JSON array of ideas from a model response.

    Mirrors parse_campaign_draft_response: strips ``<think>`` blocks and
    ``json`` code fences, then walks for a balanced top-level array.
    """

    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    candidates: list[Any] = []
    try:
        candidates.append(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    depth = 0
    start = -1
    for index, char in enumerate(cleaned):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidates.append(json.loads(cleaned[start : index + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1

    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("ideas"), list):
            candidate = candidate["ideas"]
        if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
            return candidate
    return None


def _ideas_from_response(
    response_items: Sequence[dict[str, Any]],
    *,
    episode_id: str,
    target_idea_count: int,
    model: str | None,
    usage: dict[str, Any] | None,
) -> list[PodcastIdea]:
    ideas: list[PodcastIdea] = []
    for index, item in enumerate(response_items[:target_idea_count], start=1):
        rank = item.get("rank")
        try:
            rank_int = int(rank) if rank is not None else index
        except (TypeError, ValueError):
            rank_int = index
        # Rank is 1-indexed by contract. Fall back to the model-supplied
        # ordering (1-based) if the LLM emits 0 / negative / missing rank.
        if rank_int < 1:
            rank_int = index
        ideas.append(
            PodcastIdea(
                episode_id=episode_id,
                rank=rank_int,
                summary=str(item.get("summary") or "").strip(),
                arguments=tuple(_string_list(item.get("arguments"))),
                hooks=tuple(_string_list(item.get("hooks"))),
                key_quotes=tuple(_string_list(item.get("key_quotes") or item.get("quotes"))),
                teaching_moments=tuple(_string_list(item.get("teaching_moments"))),
                metadata={
                    key: value
                    for key, value in {
                        "generation_model": model,
                        "generation_usage": dict(usage or {}),
                    }.items()
                    if value not in (None, "", {})
                },
            )
        )
    return ideas


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


class PodcastExtractionService:
    """Extract ranked ideas from podcast transcripts via LLM or BYO extractor."""

    def __init__(
        self,
        *,
        transcripts: TranscriptRepository,
        ideas: IdeaRepository,
        llm: LLMClient,
        skills: SkillStore,
        extractor: IdeaExtractor | None = None,
        config: PodcastExtractionConfig | None = None,
    ):
        self._transcripts = transcripts
        self._ideas = ideas
        self._llm = llm
        self._skills = skills
        self._extractor = extractor
        self._config = config or PodcastExtractionConfig()

    async def extract(
        self,
        *,
        scope: TenantScope,
        episode_id: str | None = None,
        limit: int = 1,
    ) -> PodcastExtractionResult:
        target_idea_count = max(1, int(self._config.target_idea_count))

        prompt_template: str | None = None
        if self._extractor is None:
            prompt_template = self._skills.get_prompt(self._config.skill_name)
            if not prompt_template:
                raise ValueError(
                    f"Podcast idea extraction skill not found: {self._config.skill_name}"
                )

        transcripts = list(
            await self._transcripts.read_transcripts(
                scope=scope,
                episode_id=episode_id,
                limit=limit,
            )
        )
        if not transcripts:
            return PodcastExtractionResult(requested=0)

        all_ideas: list[PodcastIdea] = []
        errors: list[dict[str, Any]] = []
        skipped = 0

        for transcript in transcripts:
            try:
                ideas = await self._extract_one(
                    transcript=transcript,
                    target_idea_count=target_idea_count,
                    prompt_template=prompt_template,
                    scope=scope,
                )
            except Exception as exc:
                skipped += 1
                errors.append({"episode_id": transcript.episode_id, "reason": str(exc)})
                continue
            if not ideas:
                skipped += 1
                errors.append(
                    {
                        "episode_id": transcript.episode_id,
                        "reason": "no_ideas_extracted",
                    }
                )
                continue
            all_ideas.extend(ideas)

        saved_ids: tuple[str, ...] = ()
        if all_ideas:
            saved_ids = tuple(
                str(item) for item in await self._ideas.save_ideas(all_ideas, scope=scope)
            )
        return PodcastExtractionResult(
            requested=len(transcripts),
            ideas_extracted=len(all_ideas),
            skipped=skipped,
            saved_ids=saved_ids,
            errors=tuple(errors),
        )

    async def _extract_one(
        self,
        *,
        transcript: PodcastTranscript,
        target_idea_count: int,
        prompt_template: str | None,
        scope: TenantScope,
    ) -> list[PodcastIdea]:
        if self._extractor is not None:
            extracted = await self._extractor.extract_ideas(
                scope=scope,
                transcript=transcript,
                target_idea_count=target_idea_count,
            )
            return [
                idea if isinstance(idea, PodcastIdea) else PodcastIdea(**dict(idea))
                for idea in extracted
            ]

        assert prompt_template is not None
        episode_metadata = {
            "episode_id": transcript.episode_id,
            "title": transcript.title,
            "host_name": transcript.host_name,
            "guest_name": transcript.guest_name,
            "duration_seconds": transcript.duration_seconds,
            "publish_date": transcript.publish_date,
            "source_url": transcript.source_url,
        }
        episode_metadata = {
            key: value for key, value in episode_metadata.items() if value not in (None, "", {})
        }
        episode_metadata_json = json.dumps(
            episode_metadata, separators=(",", ":"), default=str
        )
        system_prompt = (
            prompt_template
            .replace("{episode_metadata_json}", episode_metadata_json)
            .replace("{transcript_text}", transcript.transcript_text)
            .replace("{target_idea_count}", str(target_idea_count))
        )
        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=(
                        "Extract the strongest standalone ideas from this transcript.\n"
                        f"target_idea_count={target_idea_count}\n"
                        f"episode_metadata={episode_metadata_json}\n\n"
                        f"transcript_text:\n{transcript.transcript_text}"
                    ),
                ),
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            metadata={
                "episode_id": transcript.episode_id,
                "skill_name": self._config.skill_name,
            },
        )
        items = parse_podcast_idea_response(response.content)
        if not items:
            return []
        return _ideas_from_response(
            items,
            episode_id=transcript.episode_id,
            target_idea_count=target_idea_count,
            model=response.model,
            usage=dict(response.usage or {}),
        )


__all__ = [
    "PodcastExtractionConfig",
    "PodcastExtractionResult",
    "PodcastExtractionService",
    "parse_podcast_idea_response",
]
