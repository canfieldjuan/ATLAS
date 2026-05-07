"""Standalone campaign draft generation orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import re
from typing import Any

from .campaign_opportunities import (
    normalize_campaign_opportunity,
    opportunity_target_id,
)
from .campaign_ports import (
    CampaignDraft,
    CampaignRepository,
    IntelligenceRepository,
    LLMClient,
    LLMMessage,
    CampaignReasoningContextProvider,
    SkillStore,
    TenantScope,
)
from .services.campaign_reasoning_context import (
    campaign_reasoning_context_metadata,
    campaign_reasoning_context_payload,
    normalize_campaign_reasoning_context,
)
from .services.campaign_quality import campaign_quality_revalidation


_PROOF_TERM_TEXT_KEYS = ("excerpt_text", "quote", "text", "anchor", "value")


def _dedupe_terms(terms: Sequence[str], *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    clean: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        if text and text not in clean:
            clean.append(text)
        if len(clean) >= limit:
            break
    return clean


def _clean_term_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return _dedupe_terms([str(item or "").strip() for item in value], limit=limit)


def _terms_from_rows(value: Any, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    terms: list[str] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        for key in _PROOF_TERM_TEXT_KEYS:
            text = str(row.get(key) or "").strip()
            if text:
                terms.append(text)
                break
        terms = _dedupe_terms(terms, limit=limit)
        if len(terms) >= limit:
            break
    return terms


@dataclass(frozen=True)
class CampaignGenerationConfig:
    skill_name: str = "digest/b2b_campaign_generation"
    channel: str = "email"
    limit: int = 20
    max_tokens: int = 1200
    temperature: float = 0.4
    include_source_opportunity: bool = True
    channels: tuple[str, ...] = ()
    quality_revalidation_enabled: bool = False
    quality_prompt_proof_term_limit: int = 5


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


class CampaignGenerationService:
    """Generate campaign drafts through product-owned ports."""

    def __init__(
        self,
        *,
        intelligence: IntelligenceRepository,
        campaigns: CampaignRepository,
        llm: LLMClient,
        skills: SkillStore,
        reasoning_context: CampaignReasoningContextProvider | None = None,
        config: CampaignGenerationConfig | None = None,
    ):
        self._intelligence = intelligence
        self._campaigns = campaigns
        self._llm = llm
        self._skills = skills
        self._reasoning_context = reasoning_context
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
            normalize_campaign_opportunity(row, target_mode=target_mode)
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
        channels = self._channels()
        for opportunity in opportunities:
            target_id = opportunity_target_id(opportunity)
            if not target_id:
                skipped += 1
                errors.append({"reason": "missing_target_id", "opportunity": opportunity})
                continue
            try:
                opportunity = await self._opportunity_with_reasoning_context(
                    scope=scope,
                    target_mode=target_mode,
                    target_id=target_id,
                    opportunity=opportunity,
                )
            except Exception as exc:
                skipped += 1
                errors.append({"target_id": target_id, "reason": str(exc)})
                continue
            cold_email_context: dict[str, str] | None = None
            for channel in channels:
                channel_opportunity = self._opportunity_for_channel(
                    opportunity,
                    channel=channel,
                    cold_email_context=cold_email_context,
                )
                try:
                    parsed = await self._generate_one(
                        prompt_template,
                        opportunity=channel_opportunity,
                        target_mode=target_mode,
                        channel=channel,
                    )
                except Exception as exc:
                    skipped += 1
                    errors.append({
                        "target_id": target_id,
                        "channel": channel,
                        "reason": str(exc),
                    })
                    continue
                if not parsed:
                    skipped += 1
                    errors.append({
                        "target_id": target_id,
                        "channel": channel,
                        "reason": "unparseable_response",
                    })
                    continue
                parsed = self._revalidated_parsed(
                    parsed,
                    opportunity=channel_opportunity,
                    target_mode=target_mode,
                    channel=channel,
                )
                if not parsed:
                    skipped += 1
                    errors.append({
                        "target_id": target_id,
                        "channel": channel,
                        "reason": "quality_revalidation_failed",
                    })
                    continue
                if channel == "email_cold":
                    cold_email_context = {
                        "subject": parsed["subject"],
                        "body": parsed["body"],
                    }
                drafts.append(
                    CampaignDraft(
                        target_id=target_id,
                        target_mode=target_mode,
                        channel=channel,
                        subject=parsed["subject"],
                        body=parsed["body"],
                        metadata=self._metadata(parsed, channel_opportunity),
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

    def _channels(self) -> tuple[str, ...]:
        raw_value = self._config.channels or (self._config.channel,)
        raw: Sequence[str]
        if isinstance(raw_value, str):
            raw = raw_value.split(",")
        else:
            raw = raw_value
        channels: list[str] = []
        for item in raw:
            channel = str(item or "").strip()
            if channel and channel not in channels:
                channels.append(channel)
        return tuple(channels or ("email",))

    def _opportunity_for_channel(
        self,
        opportunity: Mapping[str, Any],
        *,
        channel: str,
        cold_email_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        enriched = dict(opportunity)
        enriched["channel"] = channel
        if channel == "email_followup" and cold_email_context:
            enriched["cold_email_context"] = {
                "subject": str(cold_email_context.get("subject") or ""),
                "body": str(cold_email_context.get("body") or ""),
            }
        if self._config.quality_revalidation_enabled:
            enriched = self._with_quality_prompt_terms(enriched)
        return enriched

    def _with_quality_prompt_terms(
        self,
        opportunity: Mapping[str, Any],
    ) -> dict[str, Any]:
        enriched = dict(opportunity)
        context = normalize_campaign_reasoning_context(enriched)
        terms = self._campaign_proof_terms(
            context.as_dict(),
            existing=enriched.get("campaign_proof_terms"),
        )
        if terms:
            enriched["campaign_proof_terms"] = terms
        return enriched

    def _campaign_proof_terms(
        self,
        context: Mapping[str, Any],
        *,
        existing: Any = None,
    ) -> list[str]:
        limit = max(0, int(self._config.quality_prompt_proof_term_limit or 0))
        terms = _clean_term_list(existing, limit=limit)
        if len(terms) >= limit:
            return terms
        anchors = context.get("anchor_examples")
        if isinstance(anchors, Mapping):
            for rows in anchors.values():
                terms.extend(_terms_from_rows(rows, limit=limit - len(terms)))
                terms = _dedupe_terms(terms, limit=limit)
                if len(terms) >= limit:
                    return terms
        for key in ("witness_highlights", "proof_points", "timing_windows"):
            terms.extend(_terms_from_rows(context.get(key), limit=limit - len(terms)))
            terms = _dedupe_terms(terms, limit=limit)
            if len(terms) >= limit:
                break
        return terms

    async def _opportunity_with_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        target_id: str,
        opportunity: Mapping[str, Any],
    ) -> dict[str, Any]:
        context = None
        if self._reasoning_context is not None:
            provided = await self._reasoning_context.read_campaign_reasoning_context(
                scope=scope,
                target_id=target_id,
                target_mode=target_mode,
                opportunity=opportunity,
            )
            provided_context = normalize_campaign_reasoning_context(provided)
            if provided_context.has_content():
                context = provided_context
        if context is None:
            context = normalize_campaign_reasoning_context(opportunity)
        if not context.has_content():
            return dict(opportunity)
        enriched = dict(opportunity)
        payload = campaign_reasoning_context_payload(context)
        enriched.update(campaign_reasoning_context_metadata(context))
        existing_reasoning_context = opportunity.get("reasoning_context")
        if isinstance(existing_reasoning_context, Mapping):
            enriched["reasoning_context"] = {
                **dict(existing_reasoning_context),
                "campaign_reasoning_context": payload,
            }
        else:
            enriched["reasoning_context"] = payload
        enriched["campaign_reasoning_context"] = payload
        return enriched

    async def _generate_one(
        self,
        prompt_template: str,
        *,
        opportunity: Mapping[str, Any],
        target_mode: str,
        channel: str,
    ) -> dict[str, Any] | None:
        opportunity_json = json.dumps(dict(opportunity), separators=(",", ":"), default=str)
        system_prompt = (
            prompt_template
            .replace("{target_mode}", target_mode)
            .replace("{channel}", channel)
            .replace("{opportunity}", opportunity_json)
            .replace("{opportunity_json}", opportunity_json)
        )
        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=(
                        "Generate one campaign draft from this normalized "
                        f"opportunity.\ntarget_mode={target_mode}\n"
                        f"channel={channel}\n"
                        f"opportunity={opportunity_json}"
                    ),
                ),
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            metadata={
                "target_mode": target_mode,
                "target_id": opportunity_target_id(opportunity),
                "channel": channel,
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

    def _revalidated_parsed(
        self,
        parsed: Mapping[str, Any],
        *,
        opportunity: Mapping[str, Any],
        target_mode: str,
        channel: str,
    ) -> dict[str, Any] | None:
        if not self._config.quality_revalidation_enabled:
            return dict(parsed)
        context = normalize_campaign_reasoning_context(opportunity)
        specificity_context = context.as_dict()
        proof_terms = _clean_term_list(
            opportunity.get("campaign_proof_terms"),
            limit=max(0, int(self._config.quality_prompt_proof_term_limit or 0)),
        )
        if proof_terms:
            specificity_context = {
                **specificity_context,
                "campaign_proof_terms": proof_terms,
            }
        revalidation = campaign_quality_revalidation(
            campaign={
                **dict(opportunity),
                "subject": parsed.get("subject") or "",
                "body": parsed.get("body") or "",
                "cta": parsed.get("cta") or "",
                "channel": channel,
                "target_mode": target_mode,
            },
            boundary="generation",
            specificity_context=specificity_context,
        )
        audit = revalidation.get("audit")
        if isinstance(audit, Mapping) and audit.get("blocking_issues"):
            return None
        return {
            **dict(parsed),
            "_quality_revalidation": revalidation,
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
            "campaign_revalidation": parsed.get("_quality_revalidation"),
        }
        context = normalize_campaign_reasoning_context(opportunity)
        metadata.update(campaign_reasoning_context_metadata(context))
        if self._config.include_source_opportunity:
            metadata["source_opportunity"] = dict(opportunity)
        return {key: value for key, value in metadata.items() if value not in (None, "", {})}
