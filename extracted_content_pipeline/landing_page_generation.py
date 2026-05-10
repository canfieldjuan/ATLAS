"""Standalone Landing Pages generator orchestration.

Sibling of ``CampaignGenerationService`` (per-opportunity emails) and
``ReportGenerationService`` (per-opportunity structured reports). Where
those generators iterate ``read_campaign_opportunities``, this one
takes a :class:`MarketingCampaign` directly and produces ONE
``LandingPageDraft``. The trigger shape is the primary structural
divergence: landing pages are marketing-campaign-driven, not
opportunity-driven.

Optional reasoning context: hosts may pass a
``CampaignReasoningContextProvider``. When configured, the marketing
campaign payload is handed to the provider as the "opportunity" dict
(keyed by ``target_id=campaign.name``, ``target_mode=marketing_campaign``)
so the existing multi-pass reasoning bridge can produce a narrative
plan + claims that the LLM consumes when structuring the page.

Quality gating runs through ``extracted_quality_gate.landing_page_pack``
-- pure deterministic, no LLM. Failures skip persistence and surface
in the result's ``errors`` payload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from .campaign_ports import (
    CampaignReasoningContextProvider,
    LLMClient,
    LLMMessage,
    SkillStore,
    TenantScope,
)
from .landing_page_ports import (
    LandingPageDraft,
    LandingPageRepository,
    LandingPageSection,
    MarketingCampaign,
)
from .services.campaign_reasoning_context import (
    campaign_reasoning_context_metadata,
    campaign_reasoning_context_payload,
    consumed_campaign_reasoning_contexts,
    normalize_campaign_reasoning_context,
)
from .services._parse_retry_helpers import (
    accumulate_usage,
    clip_invalid_response,
    parse_attempt_limit,
    retry_prompt_with_invalid_response,
)
from extracted_quality_gate.landing_page_pack import evaluate_landing_page
from extracted_quality_gate.types import QualityInput, QualityPolicy


_TARGET_MODE = "marketing_campaign"


@dataclass(frozen=True)
class LandingPageGenerationConfig:
    """Tunable defaults for ``LandingPageGenerationService``."""

    skill_name: str = "digest/landing_page_generation"
    max_tokens: int = 4096
    temperature: float = 0.3
    quality_policy: QualityPolicy | None = None
    quality_gates_enabled: bool = True
    parse_retry_attempts: int = 1
    parse_retry_response_excerpt_chars: int = 800


@dataclass(frozen=True)
class LandingPageGenerationResult:
    requested: int
    generated: int
    skipped: int
    reasoning_contexts_used: int = 0
    consumed_reasoning_contexts: tuple[Mapping[str, Any], ...] = ()
    saved_ids: tuple[str, ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = {
            "requested": self.requested,
            "generated": self.generated,
            "skipped": self.skipped,
            "reasoning_contexts_used": self.reasoning_contexts_used,
            "saved_ids": list(self.saved_ids),
            "errors": list(self.errors),
        }
        if self.consumed_reasoning_contexts:
            data["consumed_reasoning_contexts"] = [
                dict(item) for item in self.consumed_reasoning_contexts
            ]
        return data


def parse_landing_page_response(text: str) -> dict[str, Any] | None:
    """Extract the first well-formed landing-page JSON object.

    Mirrors ``parse_report_response``: strips ``<think>`` blocks + code
    fences, then walks the cleaned text with
    ``json.JSONDecoder.raw_decode`` so braces inside string values
    (markdown templates, CSS, etc.) don't trip the parser.

    The parser only enforces what's needed to identify the candidate as
    a landing-page payload: ``title`` (non-empty) and a non-empty
    ``sections`` list. Missing or malformed ``hero`` / ``cta`` / ``meta``
    are NOT rejected here -- the quality pack's specific blockers
    (``no_hero_headline``, ``no_cta``, etc.) are the right place to
    surface those failures so callers see exactly what was wrong rather
    than a generic ``unparseable_response``.
    """

    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    index = 0
    length = len(cleaned)
    while index < length:
        if cleaned[index] != "{":
            index += 1
            continue
        try:
            decoded, end = decoder.raw_decode(cleaned, index)
        except json.JSONDecodeError:
            index += 1
            continue
        candidates.append(decoded)
        index = end if end > index else index + 1

    for candidate in candidates:
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
        if not isinstance(candidate, dict):
            continue
        title = str(candidate.get("title") or "").strip()
        sections = candidate.get("sections")
        if not title:
            continue
        if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
            continue
        coerced_sections = [s for s in sections if isinstance(s, Mapping)]
        if not coerced_sections:
            continue
        # hero / cta / meta validation deferred to the quality pack.
        return {
            **candidate,
            "title": title,
            "sections": coerced_sections,
        }
    return None


def _has_prompt_reasoning_context(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload.get("campaign_reasoning_context"), Mapping)


def _landing_page_user_prompt(prior_invalid_response: str = "") -> str:
    prompt = "Generate one landing page from the marketing campaign above."
    return retry_prompt_with_invalid_response(
        prompt,
        prior_invalid_response=prior_invalid_response,
        instruction=(
            "The previous response could not be parsed as the required JSON object. "
            "Return one JSON object with non-empty title and sections."
        ),
    )


class LandingPageGenerationService:
    """Generate one structured landing-page draft per marketing campaign."""

    def __init__(
        self,
        *,
        landing_pages: LandingPageRepository,
        llm: LLMClient,
        skills: SkillStore,
        reasoning_context: CampaignReasoningContextProvider | None = None,
        config: LandingPageGenerationConfig | None = None,
    ):
        self._landing_pages = landing_pages
        self._llm = llm
        self._skills = skills
        self._reasoning_context = reasoning_context
        self._config = config or LandingPageGenerationConfig()

    def with_reasoning_context(
        self,
        provider: CampaignReasoningContextProvider | None,
    ) -> "LandingPageGenerationService":
        # PR-ControlSurfaces-Reasoning-Provider: route-level seam.
        return LandingPageGenerationService(
            landing_pages=self._landing_pages,
            llm=self._llm,
            skills=self._skills,
            reasoning_context=provider,
            config=self._config,
        )

    async def generate(
        self,
        *,
        scope: TenantScope,
        campaign: MarketingCampaign,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        quality_gates_enabled: bool | None = None,
    ) -> LandingPageGenerationResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(
                f"Landing-page generation skill not found: {self._config.skill_name}"
            )

        # PR-OptionA-2: per-call LLM-tuning overrides; None falls through.
        resolved_temperature = (
            self._config.temperature if temperature is None else float(temperature)
        )
        resolved_max_tokens = (
            self._config.max_tokens if max_tokens is None else int(max_tokens)
        )
        resolved_parse_retry_attempts = (
            self._config.parse_retry_attempts
            if parse_retry_attempts is None
            else int(parse_retry_attempts)
        )
        resolved_parse_retry_response_excerpt_chars = (
            self._config.parse_retry_response_excerpt_chars
            if parse_retry_response_excerpt_chars is None
            else int(parse_retry_response_excerpt_chars)
        )
        # PR-OptionA-4: when False, skip the quality gate entirely. Lets
        # operators opt out of gating per call without changing config.
        resolved_quality_gates_enabled = (
            self._config.quality_gates_enabled
            if quality_gates_enabled is None
            else bool(quality_gates_enabled)
        )

        if not str(campaign.name or "").strip():
            return LandingPageGenerationResult(
                requested=1,
                generated=0,
                skipped=1,
                errors=({"reason": "missing_campaign_name"},),
            )

        try:
            campaign_payload = await self._campaign_with_reasoning_context(
                scope=scope,
                campaign=campaign,
            )
        except Exception as exc:
            return LandingPageGenerationResult(
                requested=1,
                generated=0,
                skipped=1,
                errors=({"campaign_name": campaign.name, "reason": str(exc)},),
            )

        try:
            parsed = await self._generate_one(
                prompt_template,
                campaign_payload=campaign_payload,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                parse_retry_attempts=resolved_parse_retry_attempts,
                parse_retry_response_excerpt_chars=resolved_parse_retry_response_excerpt_chars,
            )
        except Exception as exc:
            return LandingPageGenerationResult(
                requested=1,
                generated=0,
                skipped=1,
                errors=({"campaign_name": campaign.name, "reason": str(exc)},),
            )

        if not parsed:
            return LandingPageGenerationResult(
                requested=1,
                generated=0,
                skipped=1,
                errors=({"campaign_name": campaign.name, "reason": "unparseable_response"},),
            )

        quality = self._quality_check(
            parsed,
            quality_gates_enabled=resolved_quality_gates_enabled,
        )
        if not quality["passed"]:
            return LandingPageGenerationResult(
                requested=1,
                generated=0,
                skipped=1,
                errors=({
                    "campaign_name": campaign.name,
                    "reason": "quality_blocked",
                    "blockers": quality["blockers"],
                },),
            )

        draft = self._build_draft(
            parsed,
            campaign=campaign,
            campaign_payload=campaign_payload,
        )
        saved_ids = tuple(
            str(item) for item in await self._landing_pages.save_drafts([draft], scope=scope)
        )
        return LandingPageGenerationResult(
            requested=1,
            generated=1,
            skipped=0,
            reasoning_contexts_used=(
                1 if _has_prompt_reasoning_context(campaign_payload) else 0
            ),
            consumed_reasoning_contexts=consumed_campaign_reasoning_contexts(
                campaign_payload
            ),
            saved_ids=saved_ids,
        )

    async def _campaign_with_reasoning_context(
        self,
        *,
        scope: TenantScope,
        campaign: MarketingCampaign,
    ) -> dict[str, Any]:
        """Pass the marketing campaign through the reasoning provider port.

        The existing ``CampaignReasoningContextProvider`` port keys on
        (target_id, target_mode, opportunity). We adapt by treating the
        campaign as the "opportunity" payload, with target_id =
        campaign.name and target_mode = "marketing_campaign". Hosts
        that don't configure a provider get the bare campaign payload.
        """
        payload = campaign.as_dict()
        if self._reasoning_context is None:
            return payload
        provided = await self._reasoning_context.read_campaign_reasoning_context(
            scope=scope,
            target_id=campaign.name,
            target_mode=_TARGET_MODE,
            opportunity=payload,
        )
        provided_context = normalize_campaign_reasoning_context(provided)
        if not provided_context.has_content():
            return payload
        enriched = dict(payload)
        reasoning_payload = campaign_reasoning_context_payload(provided_context)
        enriched.update(campaign_reasoning_context_metadata(provided_context))
        enriched["reasoning_context"] = reasoning_payload
        enriched["campaign_reasoning_context"] = reasoning_payload
        return enriched

    async def _generate_one(
        self,
        prompt_template: str,
        *,
        campaign_payload: Mapping[str, Any],
        temperature: float,
        max_tokens: int,
        parse_retry_attempts: int,
        parse_retry_response_excerpt_chars: int,
    ) -> dict[str, Any] | None:
        campaign_json = json.dumps(dict(campaign_payload), separators=(",", ":"), default=str)
        # Single source for the campaign payload: in the system prompt
        # via {campaign_json}. User message is structural-only so the
        # campaign isn't sent twice (matches the report-generator fix).
        system_prompt = prompt_template.replace("{campaign_json}", campaign_json)
        attempts = parse_attempt_limit(parse_retry_attempts)
        last_response = ""
        total_usage: dict[str, Any] = {}
        for attempt_no in range(1, attempts + 1):
            response = await self._llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(
                        role="user",
                        content=_landing_page_user_prompt(last_response),
                    ),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={
                    "campaign_name": campaign_payload.get("name"),
                    "skill_name": self._config.skill_name,
                    "asset_type": "landing_page",
                    "target_mode": _TARGET_MODE,
                    "attempt_no": attempt_no,
                },
            )
            total_usage = accumulate_usage(total_usage, response.usage)
            parsed = parse_landing_page_response(response.content)
            if parsed:
                return {
                    **parsed,
                    "_model": response.model,
                    "_usage": total_usage,
                    "_parse_attempts": attempt_no,
                }
            last_response = clip_invalid_response(
                response.content,
                limit=max(0, int(parse_retry_response_excerpt_chars or 0)),
            )
        return None

    def _quality_check(
        self,
        parsed: Mapping[str, Any],
        *,
        quality_gates_enabled: bool = True,
    ) -> dict[str, Any]:
        # PR-OptionA-4: opt out of the quality gate per call. The plan
        # emits ``quality_gates_enabled`` in step.config so the executor
        # can route the operator's choice to the service. False short-
        # circuits the gate; True (the default) preserves prior behavior.
        if not quality_gates_enabled:
            return {"passed": True, "blockers": ()}
        report_input = QualityInput(
            artifact_type="landing_page",
            context={
                "title": parsed.get("title"),
                "slug": parsed.get("slug"),
                "hero": parsed.get("hero") or {},
                "sections": parsed.get("sections") or (),
                "cta": parsed.get("cta") or {},
                "meta": parsed.get("meta") or {},
            },
        )
        report = evaluate_landing_page(report_input, policy=self._config.quality_policy)
        return {
            "passed": report.passed,
            "blockers": tuple(f.message for f in report.blockers),
        }

    def _build_draft(
        self,
        parsed: Mapping[str, Any],
        *,
        campaign: MarketingCampaign,
        campaign_payload: Mapping[str, Any] | None = None,
    ) -> LandingPageDraft:
        sections = tuple(
            LandingPageSection(
                id=str(s.get("id") or "").strip(),
                title=str(s.get("title") or "").strip(),
                body_markdown=str(s.get("body_markdown") or "").strip(),
                metadata=dict(s.get("metadata") or {}),
            )
            for s in parsed.get("sections") or ()
            if isinstance(s, Mapping)
        )
        title = str(parsed.get("title") or "").strip()
        slug = str(parsed.get("slug") or "").strip()
        hero = (
            dict(parsed.get("hero") or {}) if isinstance(parsed.get("hero"), Mapping) else {}
        )
        cta = (
            dict(parsed.get("cta") or {}) if isinstance(parsed.get("cta"), Mapping) else {}
        )
        meta = (
            dict(parsed.get("meta") or {}) if isinstance(parsed.get("meta"), Mapping) else {}
        )
        reference_ids = tuple(
            str(r) for r in (parsed.get("reference_ids") or ())
            if str(r).strip()
        )
        metadata: dict[str, Any] = {
            "generation_model": parsed.get("_model"),
            "generation_usage": parsed.get("_usage") or {},
            "generation_parse_attempts": parsed.get("_parse_attempts"),
        }
        if campaign_payload is not None:
            context = normalize_campaign_reasoning_context(campaign_payload)
            metadata.update(campaign_reasoning_context_metadata(context))
        return LandingPageDraft(
            campaign_name=campaign.name,
            persona=campaign.persona,
            value_prop=campaign.value_prop,
            title=title,
            slug=slug,
            hero=hero,
            sections=sections,
            cta=cta,
            meta=meta,
            reference_ids=reference_ids,
            metadata=metadata,
        )


__all__ = [
    "LandingPageGenerationConfig",
    "LandingPageGenerationResult",
    "LandingPageGenerationService",
    "parse_landing_page_response",
]
