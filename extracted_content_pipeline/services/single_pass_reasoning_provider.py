"""Single-pass LLM reasoning provider for campaign generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from typing import Any

from ..campaign_ports import (
    CampaignReasoningContext,
    LLMClient,
    LLMMessage,
    SkillStore,
    TenantScope,
)
from .campaign_reasoning_context import normalize_campaign_reasoning_context


DEFAULT_REASONING_SKILL_NAME = "digest/b2b_campaign_reasoning_context"


def _clean(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class SinglePassReasoningConfig:
    """Host-owned defaults for single-pass campaign reasoning."""

    skill_name: str = DEFAULT_REASONING_SKILL_NAME
    max_tokens: int = 900
    temperature: float = 0.2
    include_source_opportunity: bool = True

    def __post_init__(self) -> None:
        if not _clean(self.skill_name):
            raise ValueError("skill_name is required")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class SinglePassCampaignReasoningProvider:
    """Generate compact campaign reasoning context with one LLM call."""

    llm: LLMClient
    skills: SkillStore
    config: SinglePassReasoningConfig = SinglePassReasoningConfig()

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | None:
        prompt_template = self.skills.get_prompt(self.config.skill_name)
        if not prompt_template:
            raise ValueError(f"Campaign reasoning skill not found: {self.config.skill_name}")

        scope_payload = _scope_payload(scope)
        opportunity_payload = dict(opportunity) if self.config.include_source_opportunity else {}
        opportunity_json = json.dumps(opportunity_payload, separators=(",", ":"), default=str)
        scope_json = json.dumps(scope_payload, separators=(",", ":"), default=str)
        prompt = (
            prompt_template
            .replace("{target_mode}", _clean(target_mode))
            .replace("{target_id}", _clean(target_id))
            .replace("{scope}", scope_json)
            .replace("{opportunity}", opportunity_json)
            .replace("{opportunity_json}", opportunity_json)
        )
        response = await self.llm.complete(
            [
                LLMMessage(role="system", content=prompt),
                LLMMessage(
                    role="user",
                    content=(
                        "Return one compact campaign reasoning context as JSON.\n"
                        f"target_mode={_clean(target_mode)}\n"
                        f"target_id={_clean(target_id)}\n"
                        f"scope={scope_json}\n"
                        f"opportunity={opportunity_json}"
                    ),
                ),
            ],
            max_tokens=self.config.max_tokens,
            temperature=float(self.config.temperature),
            metadata={
                "target_mode": _clean(target_mode),
                "target_id": _clean(target_id),
                "skill_name": self.config.skill_name,
                "reasoning_provider": "single_pass",
            },
        )
        return parse_reasoning_context_response(response.content)


def parse_reasoning_context_response(text: str) -> CampaignReasoningContext | None:
    """Parse an LLM JSON response into normalized campaign reasoning context."""

    for candidate in _json_candidates(text):
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
        if not isinstance(candidate, Mapping):
            continue
        context = normalize_campaign_reasoning_context(candidate)
        if context.has_content():
            return context
    return None


def _json_candidates(text: str) -> list[Any]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
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
    return candidates


def _scope_payload(scope: TenantScope) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "account_id": scope.account_id,
            "user_id": scope.user_id,
            "allowed_vendors": list(scope.allowed_vendors),
            "roles": list(scope.roles),
        }.items()
        if value not in (None, "", [], {})
    }


__all__ = [
    "DEFAULT_REASONING_SKILL_NAME",
    "SinglePassCampaignReasoningProvider",
    "SinglePassReasoningConfig",
    "parse_reasoning_context_response",
]
