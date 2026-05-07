"""Standalone Reports generator orchestration.

Parallel to ``CampaignGenerationService``, but produces structured
``ReportDraft`` rows (title + summary + ordered sections + reference
ids) instead of email subject/body drafts. Per-opportunity trigger
shape: each opportunity yields one report.

Reasoning-context aware: when the host provides a multi-pass
``CampaignReasoningContextProvider`` whose context exposes a
``canonical_reasoning["narrative_plan"]`` block, the generator hands
that pre-structured plan to the LLM as the section spine so the model
writes prose for each section rather than inventing the structure.
Without a narrative plan, the LLM structures the report itself.

Quality gating runs through ``extracted_quality_gate.report_pack`` —
deterministic, no LLM, no DB. Drafts that fail the pack are skipped
and the failure is recorded in the result's ``errors`` payload; the
caller decides what to do with the rejection (retry, surface to
operator, etc.).
"""

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
    CampaignReasoningContextProvider,
    IntelligenceRepository,
    LLMClient,
    LLMMessage,
    SkillStore,
    TenantScope,
)
from .report_ports import ReportDraft, ReportRepository, ReportSection
from .services.campaign_reasoning_context import (
    campaign_reasoning_context_metadata,
    campaign_reasoning_context_payload,
    normalize_campaign_reasoning_context,
)
from extracted_quality_gate.report_pack import evaluate_report
from extracted_quality_gate.types import QualityInput


@dataclass(frozen=True)
class ReportGenerationConfig:
    """Tunable defaults for ``ReportGenerationService``."""

    skill_name: str = "digest/report_generation"
    default_report_type: str = "vendor_pressure"
    limit: int = 10
    max_tokens: int = 4096
    temperature: float = 0.3


@dataclass(frozen=True)
class ReportGenerationResult:
    requested: int
    generated: int
    skipped: int
    saved_ids: tuple[str, ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "generated": self.generated,
            "skipped": self.skipped,
            "saved_ids": list(self.saved_ids),
            "errors": list(self.errors),
        }


def parse_report_response(text: str) -> dict[str, Any] | None:
    """Mirror of ``parse_campaign_draft_response`` for the report shape.

    Strips ``<think>`` blocks and code fences, then walks the cleaned
    text with ``json.JSONDecoder.raw_decode`` (a real JSON parser, so
    braces inside string values such as ``body_markdown`` template
    syntax don't trip it up). Returns the first decoded object that
    carries the minimum report fields (``title`` + ``summary`` +
    non-empty ``sections``).
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
        summary = str(candidate.get("summary") or "").strip()
        sections = candidate.get("sections")
        if not title or not summary:
            continue
        if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
            continue
        coerced_sections = [s for s in sections if isinstance(s, Mapping)]
        if not coerced_sections:
            continue
        return {
            **candidate,
            "title": title,
            "summary": summary,
            "sections": coerced_sections,
        }
    return None


class ReportGenerationService:
    """Generate structured report drafts through product-owned ports."""

    def __init__(
        self,
        *,
        intelligence: IntelligenceRepository,
        reports: ReportRepository,
        llm: LLMClient,
        skills: SkillStore,
        reasoning_context: CampaignReasoningContextProvider | None = None,
        config: ReportGenerationConfig | None = None,
    ):
        self._intelligence = intelligence
        self._reports = reports
        self._llm = llm
        self._skills = skills
        self._reasoning_context = reasoning_context
        self._config = config or ReportGenerationConfig()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> ReportGenerationResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(f"Report generation skill not found: {self._config.skill_name}")

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

        drafts: list[ReportDraft] = []
        errors: list[dict[str, Any]] = []
        skipped = 0
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
                )
            except Exception as exc:
                skipped += 1
                errors.append({"target_id": target_id, "reason": str(exc)})
                continue

            if not parsed:
                skipped += 1
                errors.append({"target_id": target_id, "reason": "unparseable_response"})
                continue

            quality = self._quality_check(parsed)
            if not quality["passed"]:
                skipped += 1
                errors.append({
                    "target_id": target_id,
                    "reason": "quality_blocked",
                    "blockers": quality["blockers"],
                })
                continue

            drafts.append(self._build_draft(parsed, target_id=target_id, target_mode=target_mode))

        saved_ids: tuple[str, ...] = ()
        if drafts:
            saved_ids = tuple(
                str(item)
                for item in await self._reports.save_drafts(drafts, scope=scope)
            )
        return ReportGenerationResult(
            requested=len(opportunities),
            generated=len(drafts),
            skipped=skipped,
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
        # Mirrors CampaignGenerationService._opportunity_with_reasoning_context.
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
    ) -> dict[str, Any] | None:
        opportunity_json = json.dumps(dict(opportunity), separators=(",", ":"), default=str)
        # Single source for the opportunity payload: in the system prompt
        # via {opportunity_json}. The user message is structural-only so
        # the opportunity isn't sent twice (the reasoning-context block
        # alone can be the largest part of the request).
        system_prompt = (
            prompt_template
            .replace("{target_mode}", target_mode)
            .replace("{opportunity_json}", opportunity_json)
        )
        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content="Generate one structured report from the opportunity above.",
                ),
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            metadata={
                "target_mode": target_mode,
                "target_id": opportunity_target_id(opportunity),
                "skill_name": self._config.skill_name,
                "asset_type": "report",
            },
        )
        parsed = parse_report_response(response.content)
        if not parsed:
            return None
        return {
            **parsed,
            "_model": response.model,
            "_usage": dict(response.usage or {}),
        }

    def _quality_check(self, parsed: Mapping[str, Any]) -> dict[str, Any]:
        report_input = QualityInput(
            artifact_type="report",
            context={
                "title": parsed.get("title"),
                "summary": parsed.get("summary"),
                "sections": parsed.get("sections") or (),
                "reference_ids": parsed.get("reference_ids") or (),
                "metadata": {
                    "confidence": parsed.get("confidence"),
                },
            },
        )
        quality = evaluate_report(report_input)
        return {
            "passed": quality.passed,
            "blockers": tuple(f.message for f in quality.blockers),
        }

    def _build_draft(
        self,
        parsed: Mapping[str, Any],
        *,
        target_id: str,
        target_mode: str,
    ) -> ReportDraft:
        sections = tuple(
            ReportSection(
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
        report_type = str(parsed.get("report_type") or self._config.default_report_type)
        metadata: dict[str, Any] = {
            "generation_model": parsed.get("_model"),
            "generation_usage": parsed.get("_usage") or {},
        }
        if "confidence" in parsed:
            metadata["confidence"] = parsed["confidence"]
        return ReportDraft(
            target_id=target_id,
            target_mode=target_mode,
            report_type=report_type,
            title=str(parsed.get("title") or "").strip(),
            summary=str(parsed.get("summary") or "").strip(),
            sections=sections,
            reference_ids=tuple(ref_seen),
            metadata=metadata,
        )


__all__ = [
    "ReportGenerationConfig",
    "ReportGenerationResult",
    "ReportGenerationService",
    "parse_report_response",
]
