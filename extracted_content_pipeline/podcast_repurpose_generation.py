"""Multi-format generation orchestration for podcast repurposing.

Per episode, loops per idea × per format and makes one LLM call each.
Results pass through podcast_quality_revalidation; drafts whose audit
fails are still persisted with status='needs_review' (set in the
repository), and the audit is stashed in metadata.quality_audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
from typing import Any

from .podcast_ports import (
    FormatDraftRepository,
    IdeaRepository,
    LLMClient,
    LLMMessage,
    PodcastFormatDraft,
    PodcastIdea,
    PodcastTranscript,
    SkillStore,
    TenantScope,
    TranscriptRepository,
)
from .services.podcast_quality import podcast_quality_revalidation


SUPPORTED_FORMATS: tuple[str, ...] = (
    "newsletter",
    "blog",
    "linkedin",
    "x_thread",
    "shorts",
)

_DEFAULT_MAX_TOKENS: dict[str, int] = {
    "newsletter": 3500,
    "blog": 6000,
    "linkedin": 800,
    "x_thread": 1500,
    "shorts": 800,
}


@dataclass(frozen=True)
class PodcastRepurposeConfig:
    skill_name: str = "digest/podcast_format_repurpose"
    formats: tuple[str, ...] = SUPPORTED_FORMATS
    idea_limit: int = 3
    max_tokens_per_format: Mapping[str, int] = field(default_factory=lambda: dict(_DEFAULT_MAX_TOKENS))
    temperature: float = 0.5
    voice_anchors: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PodcastRepurposeResult:
    requested_ideas: int = 0
    drafts_generated: int = 0
    skipped: int = 0
    saved_ids: tuple[str, ...] = ()
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_ideas": self.requested_ideas,
            "drafts_generated": self.drafts_generated,
            "skipped": self.skipped,
            "saved_ids": list(self.saved_ids),
            "errors": list(self.errors),
        }


def parse_podcast_format_response(text: str) -> dict[str, Any] | None:
    """Extract a single JSON object from a model response."""

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
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidates.append(json.loads(cleaned[start : index + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1

    for candidate in candidates:
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
        if isinstance(candidate, dict) and (candidate.get("body") or candidate.get("content")):
            body = str(candidate.get("body") or candidate.get("content") or "").strip()
            title = str(candidate.get("title") or "").strip()
            metadata = candidate.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            return {
                "title": title,
                "body": body,
                "metadata": dict(metadata),
                "format_type": str(candidate.get("format_type") or "").strip(),
                "_raw": candidate,
            }
    return None


class PodcastRepurposeService:
    """Generate per-format drafts for an episode's extracted ideas."""

    def __init__(
        self,
        *,
        ideas: IdeaRepository,
        transcripts: TranscriptRepository,
        drafts: FormatDraftRepository,
        llm: LLMClient,
        skills: SkillStore,
        config: PodcastRepurposeConfig | None = None,
    ):
        self._ideas = ideas
        self._transcripts = transcripts
        self._drafts = drafts
        self._llm = llm
        self._skills = skills
        self._config = config or PodcastRepurposeConfig()

    async def repurpose(
        self,
        *,
        scope: TenantScope,
        episode_id: str,
        formats: Sequence[str] | None = None,
        idea_limit: int | None = None,
    ) -> PodcastRepurposeResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(
                f"Podcast format repurpose skill not found: {self._config.skill_name}"
            )

        target_formats = self._normalize_formats(formats)
        target_idea_limit = max(1, int(idea_limit or self._config.idea_limit))

        ideas = list(
            await self._ideas.read_ideas(
                scope=scope,
                episode_id=episode_id,
                limit=target_idea_limit,
            )
        )
        ideas = ideas[:target_idea_limit]

        episode_metadata = await self._episode_metadata(scope=scope, episode_id=episode_id)

        drafts: list[PodcastFormatDraft] = []
        errors: list[dict[str, Any]] = []
        skipped = 0

        for idea in ideas:
            for format_type in target_formats:
                try:
                    parsed = await self._generate_one(
                        prompt_template=prompt_template,
                        idea=idea,
                        episode_metadata=episode_metadata,
                        format_type=format_type,
                    )
                except Exception as exc:
                    skipped += 1
                    errors.append(
                        {
                            "episode_id": episode_id,
                            "idea_id": idea.idea_id,
                            "rank": idea.rank,
                            "format_type": format_type,
                            "reason": str(exc),
                        }
                    )
                    continue
                if not parsed:
                    skipped += 1
                    errors.append(
                        {
                            "episode_id": episode_id,
                            "idea_id": idea.idea_id,
                            "rank": idea.rank,
                            "format_type": format_type,
                            "reason": "unparseable_response",
                        }
                    )
                    continue
                audit_envelope = podcast_quality_revalidation(
                    draft=parsed,
                    format_type=format_type,
                    idea=_idea_to_dict(idea),
                    voice_anchors=self._config.voice_anchors,
                )
                metadata = dict(parsed.get("metadata") or {})
                metadata["quality_audit"] = audit_envelope.get("audit", {})
                if idea.idea_id:
                    metadata.setdefault("source_idea_id", idea.idea_id)
                metadata.setdefault("source_idea_rank", idea.rank)
                drafts.append(
                    PodcastFormatDraft(
                        episode_id=episode_id,
                        format_type=format_type,
                        title=str(parsed.get("title") or ""),
                        body=str(parsed.get("body") or ""),
                        metadata=metadata,
                        quality_audit=dict(audit_envelope.get("audit") or {}),
                        idea_id=idea.idea_id,
                    )
                )

        saved_ids: tuple[str, ...] = ()
        if drafts:
            saved_ids = tuple(
                str(item) for item in await self._drafts.save_drafts(drafts, scope=scope)
            )
        return PodcastRepurposeResult(
            requested_ideas=len(ideas),
            drafts_generated=len(drafts),
            skipped=skipped,
            saved_ids=saved_ids,
            errors=tuple(errors),
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _normalize_formats(self, formats: Sequence[str] | None) -> tuple[str, ...]:
        configured = self._config.formats or SUPPORTED_FORMATS
        source: Sequence[str]
        if formats is None:
            source = configured
        elif isinstance(formats, str):
            source = formats.split(",")
        else:
            source = formats
        out: list[str] = []
        for item in source:
            value = str(item or "").strip().lower()
            if value and value in SUPPORTED_FORMATS and value not in out:
                out.append(value)
        return tuple(out)

    async def _episode_metadata(
        self,
        *,
        scope: TenantScope,
        episode_id: str,
    ) -> dict[str, Any]:
        try:
            transcripts = list(
                await self._transcripts.read_transcripts(
                    scope=scope,
                    episode_id=episode_id,
                    limit=1,
                )
            )
        except Exception:
            transcripts = []
        if not transcripts:
            return {"episode_id": episode_id}
        transcript: PodcastTranscript = transcripts[0]
        meta = {
            "episode_id": transcript.episode_id,
            "title": transcript.title,
            "host_name": transcript.host_name,
            "guest_name": transcript.guest_name,
            "duration_seconds": transcript.duration_seconds,
            "publish_date": transcript.publish_date,
            "source_url": transcript.source_url,
        }
        return {key: value for key, value in meta.items() if value not in (None, "", {})}

    async def _generate_one(
        self,
        *,
        prompt_template: str,
        idea: PodcastIdea,
        episode_metadata: Mapping[str, Any],
        format_type: str,
    ) -> dict[str, Any] | None:
        idea_dict = _idea_to_dict(idea)
        idea_json = json.dumps(idea_dict, separators=(",", ":"), default=str)
        episode_json = json.dumps(dict(episode_metadata), separators=(",", ":"), default=str)
        voice_json = json.dumps(dict(self._config.voice_anchors or {}), separators=(",", ":"), default=str)
        system_prompt = (
            prompt_template
            .replace("{format_type}", format_type)
            .replace("{idea_json}", idea_json)
            .replace("{episode_metadata_json}", episode_json)
            .replace("{voice_anchors_json}", voice_json)
        )
        max_tokens = int(
            self._config.max_tokens_per_format.get(
                format_type,
                _DEFAULT_MAX_TOKENS.get(format_type, 2000),
            )
        )
        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=(
                        f"Repurpose this idea into format_type={format_type}.\n"
                        f"episode_metadata={episode_json}\n"
                        f"idea={idea_json}\n"
                        f"voice_anchors={voice_json}"
                    ),
                ),
            ],
            max_tokens=max_tokens,
            temperature=self._config.temperature,
            metadata={
                "episode_id": idea.episode_id,
                "idea_id": idea.idea_id,
                "rank": idea.rank,
                "format_type": format_type,
                "skill_name": self._config.skill_name,
            },
        )
        parsed = parse_podcast_format_response(response.content)
        if not parsed:
            return None
        parsed = dict(parsed)
        meta = dict(parsed.get("metadata") or {})
        meta.setdefault("generation_model", response.model)
        meta.setdefault("generation_usage", dict(response.usage or {}))
        parsed["metadata"] = meta
        return parsed


def _idea_to_dict(idea: PodcastIdea) -> dict[str, Any]:
    return {
        "idea_id": idea.idea_id,
        "episode_id": idea.episode_id,
        "rank": idea.rank,
        "summary": idea.summary,
        "arguments": list(idea.arguments),
        "hooks": list(idea.hooks),
        "key_quotes": list(idea.key_quotes),
        "teaching_moments": list(idea.teaching_moments),
        "metadata": dict(idea.metadata),
    }


__all__ = [
    "SUPPORTED_FORMATS",
    "PodcastRepurposeConfig",
    "PodcastRepurposeResult",
    "PodcastRepurposeService",
    "parse_podcast_format_response",
]
