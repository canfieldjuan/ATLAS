"""Standalone campaign draft generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping

from .campaign_ports import (
    CampaignDraft,
    CampaignRepository,
    IntelligenceRepository,
    LLMClient,
    LLMMessage,
    SkillStore,
    TenantScope,
)


@dataclass(frozen=True)
class CampaignGenerationConfig:
    skill_name: str = "digest/b2b_campaign_generation"
    channel: str = "email"
    limit: int = 20
    max_tokens: int = 1200
    temperature: float = 0.4
    include_source_opportunity: bool = True


@dataclass(frozen=True)
class CampaignGenerationResult:
    requested: int = 0
    generated: int = 0
    skipped: int = 0
    saved_ids: tuple[str, ...] = ()
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "generated": self.generated,
            "skipped": self.skipped,
            "saved_ids": list(self.saved_ids),
            "errors": list(self.errors),
        }


def parse_campaign_draft_response(text: str) -> dict[str, Any] | None:
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
        if not isinstance(candidate, dict):
            continue
        subject = str(candidate.get("subject") or "").strip()
        body = str(
            candidate.get("body")
            or candidate.get("email_body")
            or candidate.get("content")
            or ""
        ).strip()
        if subject and body:
            return {**candidate, "subject": subject, "body": body}
    return None


def opportunity_target_id(opportunity: Mapping[str, Any]) -> str:
    for key in ("target_id", "id", "company_id", "vendor_id", "email"):
        value = str(opportunity.get(key) or "").strip()
        if value:
            return value
    for key in ("company_name", "vendor_name", "name"):
        value = str(opportunity.get(key) or "").strip()
        if value:
            return value
    return ""


class CampaignGenerationService:
    """Generate campaign drafts through product-owned ports."""

    def __init__(
        self,
        *,
        intelligence: IntelligenceRepository,
        campaigns: CampaignRepository,
        llm: LLMClient,
        skills: SkillStore,
        config: CampaignGenerationConfig | None = None,
    ):
        self._intelligence = intelligence
        self._campaigns = campaigns
        self._llm = llm
        self._skills = skills
        self._config = config or CampaignGenerationConfig()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> CampaignGenerationResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(f"Campaign generation skill not found: {self._config.skill_name}")

        requested = int(limit or self._config.limit)
        opportunities = [
            dict(row)
            for row in await self._intelligence.read_campaign_opportunities(
                scope=scope,
                target_mode=target_mode,
                limit=requested,
                filters=filters,
            )
        ]

        drafts: list[CampaignDraft] = []
        errors: list[dict[str, Any]] = []
        skipped = 0
        for opportunity in opportunities:
            target_id = opportunity_target_id(opportunity)
            if not target_id:
                skipped += 1
                errors.append({"reason": "missing_target_id", "opportunity": opportunity})
                continue
            try:
                parsed = await self._generate_one(
                    prompt_template,
                    opportunity=opportunity,
                    target_mode=target_mode,
                )
            except Exception as exc:
                skipped += 1
                errors.append({"target_id": target_id, "reason": str(exc)})
                continue
            if not parsed:
                skipped += 1
                errors.append({"target_id": target_id, "reason": "unparseable_response"})
                continue
            drafts.append(
                CampaignDraft(
                    target_id=target_id,
                    target_mode=target_mode,
                    channel=self._config.channel,
                    subject=parsed["subject"],
                    body=parsed["body"],
                    metadata=self._metadata(parsed, opportunity),
                )
            )

        saved_ids: tuple[str, ...] = ()
        if drafts:
            saved_ids = tuple(
                str(item)
                for item in await self._campaigns.save_drafts(drafts, scope=scope)
            )
        return CampaignGenerationResult(
            requested=len(opportunities),
            generated=len(drafts),
            skipped=skipped,
            saved_ids=saved_ids,
            errors=tuple(errors),
        )

    async def _generate_one(
        self,
        prompt_template: str,
        *,
        opportunity: Mapping[str, Any],
        target_mode: str,
    ) -> dict[str, Any] | None:
        opportunity_json = json.dumps(dict(opportunity), separators=(",", ":"), default=str)
        system_prompt = (
            prompt_template
            .replace("{target_mode}", target_mode)
            .replace("{opportunity}", opportunity_json)
            .replace("{opportunity_json}", opportunity_json)
        )
        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content="Generate one campaign draft."),
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            metadata={
                "target_mode": target_mode,
                "target_id": opportunity_target_id(opportunity),
                "skill_name": self._config.skill_name,
            },
        )
        parsed = parse_campaign_draft_response(response.content)
        if not parsed:
            return None
        return {
            **parsed,
            "_model": response.model,
            "_usage": dict(response.usage or {}),
        }

    def _metadata(
        self,
        parsed: Mapping[str, Any],
        opportunity: Mapping[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "cta": parsed.get("cta"),
            "angle_reasoning": parsed.get("angle_reasoning"),
            "generation_model": parsed.get("_model"),
            "generation_usage": parsed.get("_usage") or {},
        }
        if self._config.include_source_opportunity:
            metadata["source_opportunity"] = dict(opportunity)
        return {key: value for key, value in metadata.items() if value not in (None, "", {})}
