"""Offline / deterministic helpers for podcast repurposing.

Mirrors the role of campaign_example.DeterministicCampaignLLM. Produces
plausible-shape JSON for both the idea-extraction skill and the format-
repurpose skill, so the runners and tests can exercise the pipeline
end-to-end without a real LLM.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from .podcast_ports import LLMMessage, LLMResponse


class StaticPodcastSkillStore:
    """Host skill store stub for offline runs.

    Returns a fixed marker string for any skill name. The deterministic
    LLM looks at the metadata['skill_name'] passed in by the service to
    decide which output shape to produce.
    """

    def __init__(self, prompt: str = "OFFLINE_DETERMINISTIC_PODCAST_SKILL") -> None:
        self._prompt = prompt

    def get_prompt(self, name: str) -> str | None:
        del name
        return self._prompt


class DeterministicPodcastLLM:
    """Offline LLM stand-in that returns predictable JSON for podcast skills."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "metadata": dict(metadata or {}),
            }
        )
        meta = dict(metadata or {})
        skill_name = str(meta.get("skill_name") or "")
        if skill_name == "digest/podcast_idea_extraction":
            return self._extraction_response(meta, messages)
        return self._format_response(meta, messages)

    # ------------------------------------------------------------------
    # response builders
    # ------------------------------------------------------------------

    def _extraction_response(
        self,
        meta: Mapping[str, Any],
        messages: Sequence[LLMMessage],
    ) -> LLMResponse:
        target_count = _coerce_int(_payload_field(messages, "target_idea_count")) or 3
        episode_id = str(meta.get("episode_id") or "ep-offline")
        ideas = []
        for rank in range(1, target_count + 1):
            ideas.append(
                {
                    "rank": rank,
                    "summary": (
                        f"Idea {rank}: an offline-deterministic standalone idea "
                        f"extracted from episode {episode_id}."
                    ),
                    "arguments": [
                        f"Argument A for idea {rank}.",
                        f"Argument B for idea {rank}.",
                        f"Argument C for idea {rank}.",
                    ],
                    "hooks": [
                        f"Hook line A for idea {rank}.",
                        f"Hook line B for idea {rank}.",
                    ],
                    "key_quotes": [
                        f"Verbatim quote about idea {rank} that runs longer than eight words.",
                    ],
                    "teaching_moments": [
                        f"What the listener takes away from idea {rank}.",
                    ],
                }
            )
        return LLMResponse(
            content=json.dumps(ideas, separators=(",", ":")),
            model="offline-deterministic",
            usage={"input_tokens": 0, "output_tokens": 0},
        )

    def _format_response(
        self,
        meta: Mapping[str, Any],
        messages: Sequence[LLMMessage],
    ) -> LLMResponse:
        format_type = str(meta.get("format_type") or "newsletter")
        episode_id = str(meta.get("episode_id") or "ep-offline")
        idea_id = str(meta.get("idea_id") or "")
        if format_type == "x_thread":
            tweets = [
                f"Tweet {i}/6: deterministic offline x_thread tweet about {episode_id}."
                for i in range(1, 7)
            ]
            tweets[0] = f"Hook tweet about {episode_id} (offline deterministic body)."
            body = "\n\n---\n\n".join(tweets)
            metadata_extra = {"tweet_count": 6}
            title = f"Offline thread for {episode_id}"
        elif format_type == "shorts":
            body = (
                "HOOK: Listen to this surprising claim now.\n"
                "BODY: This is the deterministic body of a shorts script generated "
                "offline. It walks through one supporting argument step by step, "
                "describing the texture and the stakes. The narrator slows down at "
                "the right moments and avoids revealing the ending. The whole arc "
                "covers more than one hundred words so the validator's length band "
                "is satisfied while the spoiler rule still holds, with the closing "
                "punch reserved entirely for the very last sentence below.\n"
                "CTA: Subscribe for more."
            )
            metadata_extra = {"section_count": 3}
            title = f"Offline shorts script for {episode_id}"
        elif format_type == "blog":
            body = (
                f"# Offline blog draft for {episode_id}\n\n"
                "Lede paragraph that is at least eighty words long for a deterministic offline draft. "
                "The lede explains why the topic matters, sets the stake, and previews the structure of the post. "
                "It is intentionally verbose to exceed the minimum word band that the quality validator enforces. "
                "After this lede the post moves into named sub-sections each of which can be expanded later.\n\n"
                "## Section one\n\nDeterministic body for section one. " * 30 + "\n\n"
                "## Section two\n\nDeterministic body for section two. " * 30 + "\n\n"
                "## Section three\n\nDeterministic body for section three. " * 30 + "\n\n"
                "## Conclusion\n\nDeterministic conclusion. " * 20
            )
            metadata_extra = {
                "meta_description": (
                    "Offline deterministic blog draft for the podcast repurposing pipeline."
                )
            }
            title = f"Offline blog post for {episode_id}"
        elif format_type == "linkedin":
            body = (
                "Hook line under one twenty characters that grabs attention.\n\n"
                "Deterministic offline LinkedIn body paragraph one walks through the "
                "first supporting beat with concrete language and grounded examples "
                "without using any hashtags inside the post itself.\n\n"
                "Deterministic offline LinkedIn body paragraph two layers another "
                "supporting beat on top, building the second half of the argument "
                "in a few clean sentences before the close.\n\n"
                "Deterministic offline LinkedIn body paragraph three lands the "
                "reflection and sets up the question that ends the post in a way "
                "that invites a real response from the reader.\n\n"
                "Does this resonate?"
            )
            metadata_extra = {"hashtags": ["#offline", "#deterministic"]}
            title = f"Offline LinkedIn post for {episode_id}"
        else:  # newsletter and any unknown fallback
            body = (
                "Hook sentence one. Hook sentence two.\n\n"
                "Context paragraph that frames the idea for the reader and sets the "
                "stakes with a couple of concrete details that ground the rest of "
                "the piece in something real.\n\n"
                + ("Argument paragraph with a supporting quote and a concrete example. " * 60)
                + "\n\nReflection on what this means going forward.\n\n"
                "Subscribe to listen to the full episode."
            )
            metadata_extra = {"section_count": 4}
            title = f"Offline newsletter for {episode_id}"

        payload = {
            "title": title,
            "body": body,
            "format_type": format_type,
            "metadata": {
                "word_count": len(body.split()),
                **metadata_extra,
            },
        }
        if idea_id:
            payload["metadata"]["idea_id"] = idea_id
        return LLMResponse(
            content=json.dumps(payload, separators=(",", ":")),
            model="offline-deterministic",
            usage={"input_tokens": 0, "output_tokens": 0},
        )


def _payload_field(messages: Sequence[LLMMessage], field: str) -> Any:
    """Extract a ``field=value`` line from any message content."""

    needle = f"{field}="
    for message in messages:
        for line in str(message.content or "").splitlines():
            if needle in line:
                _, _, value = line.partition(needle)
                return value.strip()
    return None


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


__all__ = [
    "DeterministicPodcastLLM",
    "StaticPodcastSkillStore",
]
