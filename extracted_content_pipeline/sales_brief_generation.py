"""Standalone Sales Briefs generator orchestration.

Sibling of ``CampaignGenerationService`` (per-opportunity emails) and
``ReportGenerationService`` (per-opportunity structured reports). Same
per-opportunity trigger shape as Reports: the service iterates
``intelligence.read_campaign_opportunities`` and produces one
``SalesBriefDraft`` per opportunity row.

Distinct from Reports:
- Output carries a one-line ``headline`` (elevator pitch, ~140 chars)
  instead of a longer paragraph ``summary``.
- ``brief_type`` (pre_call / renewal / displacement / discovery)
  classifies the brief shape; defaults to ``pre_call`` when the LLM
  doesn't supply one.
- The skill prompt frames the output as sales-facing internal copy
  rather than a customer-facing analytical document.

Reasoning-context aware: when the host provides a multi-pass
``CampaignReasoningContextProvider`` whose context exposes a
``canonical_reasoning["narrative_plan"]`` block, the generator hands
that pre-structured plan to the LLM as the section spine so the model
writes prose for each section rather than inventing the structure.
Without a narrative plan, the LLM structures the brief itself.

Quality gating runs through ``extracted_quality_gate.sales_brief_pack``
-- pure deterministic, no LLM. Failures skip persistence and surface
in the result's ``errors`` payload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from .campaign_opportunities import (
    normalize_campaign_opportunity,
    opportunity_target_id,
)
from .campaign_ports import (
    CampaignReasoningContextProvider,
    IntelligenceRepository,
    LLMClient,
    LLMMessage,
    SkillStore,
    TenantScope,
)
from .brand_voice import (
    BrandVoiceProfile,
    apply_brand_voice_to_system_prompt,
    brand_voice_profile_from_mapping,
    brand_voice_result_metadata,
)
from .campaign_source_adapters import (
    source_material_to_source_rows,
    source_rows_to_campaign_opportunities,
)
from .sales_brief_ports import (
    SalesBriefDraft,
    SalesBriefRepository,
    SalesBriefSection,
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
from extracted_quality_gate.sales_brief_pack import evaluate_sales_brief
from extracted_quality_gate.types import QualityInput, QualityPolicy


@dataclass(frozen=True)
class SalesBriefGenerationConfig:
    """Tunable defaults for ``SalesBriefGenerationService``."""

    skill_name: str = "digest/sales_brief_generation"
    default_brief_type: str = "pre_call"
    limit: int = 10
    max_tokens: int = 4096
    temperature: float = 0.3
    quality_policy: QualityPolicy | None = None
    quality_gates_enabled: bool = True
    parse_retry_attempts: int = 1
    parse_retry_response_excerpt_chars: int = 800


@dataclass(frozen=True)
class SalesBriefGenerationResult:
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


def _opportunities_from_source_material(
    source_material: Any,
    *,
    target_mode: str,
    limit: int,
) -> list[dict[str, Any]] | None:
    rows = source_material_to_source_rows(source_material)
    if not rows:
        return None
    loaded = source_rows_to_campaign_opportunities(rows, target_mode=target_mode)
    return [
        normalize_campaign_opportunity(row, target_mode=target_mode)
        for row in loaded.opportunities[: max(0, int(limit))]
    ]


def parse_sales_brief_response(text: str) -> dict[str, Any] | None:
    """Extract the first well-formed sales-brief JSON object.

    Mirrors ``parse_report_response``: strips ``<think>`` blocks + code
    fences, then walks the cleaned text with
    ``json.JSONDecoder.raw_decode`` so braces inside string values
    (markdown templates, etc.) don't trip the parser.

    The parser only enforces what's needed to identify the candidate as
    a sales-brief payload: ``title`` (non-empty) and a non-empty
    ``sections`` list. Missing or malformed ``headline`` / ``brief_type``
    are NOT rejected here -- the quality pack's specific blockers
    (``no_headline``, etc.) surface those failures so callers see
    exactly what was wrong rather than a generic ``unparseable_response``.
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
        return {
            **candidate,
            "title": title,
            "sections": coerced_sections,
        }
    return None


def _has_prompt_reasoning_context(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload.get("campaign_reasoning_context"), Mapping)


def _sales_brief_user_prompt(
    prior_invalid_response: str = "",
    *,
    variant_angle: str | None = None,
) -> str:
    prompt = "Generate one sales brief from the opportunity above."
    resolved_variant_angle = str(variant_angle or "").strip()
    if resolved_variant_angle:
        prompt = (
            f"{prompt}\n\n"
            "Variant angle:\n"
            f"- {resolved_variant_angle}\n"
            "Use this angle to change framing and emphasis only. Keep the same "
            "opportunity facts, supplied evidence, buying context, and "
            "truthfulness limits."
        )
    return retry_prompt_with_invalid_response(
        prompt,
        prior_invalid_response=prior_invalid_response,
        instruction=(
            "The previous response could not be parsed as the required JSON object. "
            "Return one JSON object with non-empty title and sections."
        ),
    )


class SalesBriefGenerationService:
    """Generate sales-brief drafts through product-owned ports."""

    def __init__(
        self,
        *,
        intelligence: IntelligenceRepository,
        sales_briefs: SalesBriefRepository,
        llm: LLMClient,
        skills: SkillStore,
        reasoning_context: CampaignReasoningContextProvider | None = None,
        config: SalesBriefGenerationConfig | None = None,
    ):
        self._intelligence = intelligence
        self._sales_briefs = sales_briefs
        self._llm = llm
        self._skills = skills
        self._reasoning_context = reasoning_context
        self._config = config or SalesBriefGenerationConfig()

    def with_reasoning_context(
        self,
        provider: CampaignReasoningContextProvider | None,
    ) -> "SalesBriefGenerationService":
        # PR-ControlSurfaces-Reasoning-Provider: route-level seam.
        return SalesBriefGenerationService(
            intelligence=self._intelligence,
            sales_briefs=self._sales_briefs,
            llm=self._llm,
            skills=self._skills,
            reasoning_context=provider,
            config=self._config,
        )

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        default_brief_type: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        quality_gates_enabled: bool | None = None,
        source_material: Any = None,
        brand_voice: Mapping[str, Any] | BrandVoiceProfile | None = None,
        variant_angle: str | None = None,
    ) -> SalesBriefGenerationResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(
                f"Sales-brief generation skill not found: {self._config.skill_name}"
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
        # PR-OptionA-4: per-call quality gate opt-out.
        resolved_quality_gates_enabled = (
            self._config.quality_gates_enabled
            if quality_gates_enabled is None
            else bool(quality_gates_enabled)
        )
        resolved_brand_voice = brand_voice_profile_from_mapping(
            brand_voice,
            scope=scope,
        )
        resolved_variant_angle = str(variant_angle or "").strip() or None

        requested = int(limit or self._config.limit)
        source_opportunities = _opportunities_from_source_material(
            source_material,
            target_mode=target_mode,
            limit=requested,
        )
        if source_opportunities is None:
            opportunities = [
                normalize_campaign_opportunity(row, target_mode=target_mode)
                for row in await self._intelligence.read_campaign_opportunities(
                    scope=scope,
                    target_mode=target_mode,
                    limit=requested,
                    filters=filters,
                )
            ]
        else:
            opportunities = source_opportunities

        drafts: list[SalesBriefDraft] = []
        errors: list[dict[str, Any]] = []
        skipped = 0
        reasoning_contexts_used = 0
        consumed_reasoning_contexts: list[dict[str, Any]] = []
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

            try:
                parsed = await self._generate_one(
                    prompt_template,
                    opportunity=opportunity,
                    target_mode=target_mode,
                    temperature=resolved_temperature,
                    max_tokens=resolved_max_tokens,
                    parse_retry_attempts=resolved_parse_retry_attempts,
                    parse_retry_response_excerpt_chars=resolved_parse_retry_response_excerpt_chars,
                    brand_voice=resolved_brand_voice,
                    variant_angle=resolved_variant_angle,
                )
            except Exception as exc:
                skipped += 1
                errors.append({"target_id": target_id, "reason": str(exc)})
                continue

            if not parsed:
                skipped += 1
                errors.append({"target_id": target_id, "reason": "unparseable_response"})
                continue

            quality = self._quality_check(
                parsed,
                quality_gates_enabled=resolved_quality_gates_enabled,
            )
            if not quality["passed"]:
                skipped += 1
                errors.append({
                    "target_id": target_id,
                    "reason": "quality_blocked",
                    "blockers": quality["blockers"],
                })
                continue

            if _has_prompt_reasoning_context(opportunity):
                reasoning_contexts_used += 1
                consumed_reasoning_contexts.extend(
                    consumed_campaign_reasoning_contexts(opportunity)
                )
            drafts.append(
                self._build_draft(
                    parsed,
                    target_id=target_id,
                    target_mode=target_mode,
                    default_brief_type=default_brief_type,
                    opportunity=opportunity,
                )
            )

        saved_ids: tuple[str, ...] = ()
        if drafts:
            saved_ids = tuple(
                str(item)
                for item in await self._sales_briefs.save_drafts(drafts, scope=scope)
            )
        return SalesBriefGenerationResult(
            requested=len(opportunities),
            generated=len(drafts),
            skipped=skipped,
            reasoning_contexts_used=reasoning_contexts_used,
            consumed_reasoning_contexts=tuple(consumed_reasoning_contexts),
            saved_ids=saved_ids,
            errors=tuple(errors),
        )

    async def _opportunity_with_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        target_id: str,
        opportunity: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Mirrors ReportGenerationService._opportunity_with_reasoning_context.
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
        temperature: float,
        max_tokens: int,
        parse_retry_attempts: int,
        parse_retry_response_excerpt_chars: int,
        brand_voice: BrandVoiceProfile | None = None,
        variant_angle: str | None = None,
    ) -> dict[str, Any] | None:
        opportunity_json = json.dumps(dict(opportunity), separators=(",", ":"), default=str)
        # Single source for the opportunity payload: in the system prompt
        # via {opportunity_json}. User message is structural-only so the
        # opportunity isn't sent twice (matches the report-generator fix).
        system_prompt = (
            prompt_template
            .replace("{target_mode}", target_mode)
            .replace("{opportunity_json}", opportunity_json)
        )
        system_prompt = apply_brand_voice_to_system_prompt(system_prompt, brand_voice)
        attempts = parse_attempt_limit(parse_retry_attempts)
        last_response = ""
        total_usage: dict[str, Any] = {}
        resolved_variant_angle = str(variant_angle or "").strip()
        for attempt_no in range(1, attempts + 1):
            metadata: dict[str, Any] = {
                "target_mode": target_mode,
                "target_id": opportunity_target_id(opportunity),
                "skill_name": self._config.skill_name,
                "asset_type": "sales_brief",
                "attempt_no": attempt_no,
            }
            if resolved_variant_angle:
                metadata["variant_angle"] = resolved_variant_angle
            response = await self._llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(
                        role="user",
                        content=_sales_brief_user_prompt(
                            last_response,
                            variant_angle=resolved_variant_angle,
                        ),
                    ),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            total_usage = accumulate_usage(total_usage, response.usage)
            parsed = parse_sales_brief_response(response.content)
            if parsed:
                result_payload = {
                    **parsed,
                    "_model": response.model,
                    "_usage": total_usage,
                    "_parse_attempts": attempt_no,
                }
                if resolved_variant_angle:
                    result_payload["_variant_angle"] = resolved_variant_angle
                return brand_voice_result_metadata(result_payload, brand_voice)
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
        # PR-OptionA-4: opt out of the quality gate per call. Same shape
        # as the landing-page service.
        if not quality_gates_enabled:
            return {"passed": True, "blockers": ()}
        brief_input = QualityInput(
            artifact_type="sales_brief",
            context={
                "title": parsed.get("title"),
                "headline": parsed.get("headline"),
                "sections": parsed.get("sections") or (),
                "reference_ids": parsed.get("reference_ids") or (),
                "metadata": {
                    "confidence": parsed.get("confidence"),
                },
            },
        )
        report = evaluate_sales_brief(brief_input, policy=self._config.quality_policy)
        return {
            "passed": report.passed,
            "blockers": tuple(f.message for f in report.blockers),
        }

    def _build_draft(
        self,
        parsed: Mapping[str, Any],
        *,
        target_id: str,
        target_mode: str,
        default_brief_type: str | None = None,
        opportunity: Mapping[str, Any] | None = None,
    ) -> SalesBriefDraft:
        sections = tuple(
            SalesBriefSection(
                id=str(s.get("id") or "").strip(),
                title=str(s.get("title") or "").strip(),
                body_markdown=str(s.get("body_markdown") or "").strip(),
                claim_ids=tuple(str(c) for c in (s.get("claim_ids") or ())),
                evidence_ids=tuple(str(e) for e in (s.get("evidence_ids") or ())),
                metadata=dict(s.get("metadata") or {}),
            )
            for s in parsed.get("sections") or ()
            if isinstance(s, Mapping)
        )
        # Aggregate reference_ids: explicit list union per-section evidence ids.
        ref_seen: list[str] = []
        for value in parsed.get("reference_ids") or ():
            v = str(value).strip()
            if v and v not in ref_seen:
                ref_seen.append(v)
        for section in sections:
            for evidence_id in section.evidence_ids:
                v = str(evidence_id).strip()
                if v and v not in ref_seen:
                    ref_seen.append(v)
        # Per-call override (when present and non-empty) is the requested
        # output contract from the run plan. The model field remains only a
        # fallback for callers that did not request a specific brief type.
        requested_default = (default_brief_type or "").strip()
        model_brief_type = str(parsed.get("brief_type") or "").strip()
        configured_default = str(self._config.default_brief_type or "").strip()
        brief_type = requested_default or model_brief_type or configured_default
        metadata: dict[str, Any] = {
            "generation_model": parsed.get("_model"),
            "generation_usage": parsed.get("_usage") or {},
            "generation_parse_attempts": parsed.get("_parse_attempts"),
            "brand_voice_profile": parsed.get("_brand_voice_profile"),
            "brand_voice_audit": parsed.get("_brand_voice_audit"),
        }
        variant_angle = str(parsed.get("_variant_angle") or "").strip()
        if variant_angle:
            metadata["variant_angle"] = variant_angle
        if opportunity is not None:
            context = normalize_campaign_reasoning_context(opportunity)
            metadata.update(campaign_reasoning_context_metadata(context))
        if "confidence" in parsed:
            metadata["confidence"] = parsed["confidence"]
        return SalesBriefDraft(
            target_id=target_id,
            target_mode=target_mode,
            brief_type=brief_type,
            title=str(parsed.get("title") or "").strip(),
            headline=str(parsed.get("headline") or "").strip(),
            sections=sections,
            reference_ids=tuple(ref_seen),
            metadata=metadata,
        )


__all__ = [
    "SalesBriefGenerationConfig",
    "SalesBriefGenerationResult",
    "SalesBriefGenerationService",
    "parse_sales_brief_response",
]
