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
    consumed_campaign_reasoning_contexts,
    normalize_campaign_reasoning_context,
)
from .services.campaign_quality import campaign_quality_revalidation
from .services._parse_retry_helpers import (
    accumulate_usage,
    clip_invalid_response,
    parse_attempt_limit,
    retry_prompt_with_invalid_response,
)


_PROOF_TERM_TEXT_KEYS = ("excerpt_text", "quote", "text", "anchor", "value")
_PLACEHOLDER_URL_RE = re.compile(
    r"(?<![A-Za-z0-9.@-])(?:https?://)?(?:www\.)?"
    r"(?:(?:[A-Za-z0-9-]+\.)*example\.(?:com|org|net)|localhost)"
    r"(?::\d+)?(?=$|[\s/?#:.,;!?)\]}>'\"])",
    re.IGNORECASE,
)


def _normalize_channels(items: Sequence[Any]) -> tuple[str, ...]:
    """Strip + dedupe channel ids while preserving insertion order.

    Shared between the per-call override path and the construction-time
    config fallback so both apply identical normalization. Empty / blank
    items are dropped silently. Empty input yields an empty tuple.
    """
    seen: list[str] = []
    for item in items:
        channel = str(item or "").strip()
        if channel and channel not in seen:
            seen.append(channel)
    return tuple(seen)


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


def _revalidation_error_details(revalidation: Mapping[str, Any]) -> dict[str, Any]:
    audit = revalidation.get("audit")
    if not isinstance(audit, Mapping):
        return {}
    details = {
        "status": audit.get("status"),
        "blocking_issues": list(audit.get("blocking_issues") or []),
        "warnings": list(audit.get("warnings") or []),
        "primary_blocker": audit.get("primary_blocker"),
        "used_proof_terms": list(audit.get("used_proof_terms") or []),
        "unused_proof_terms": list(audit.get("unused_proof_terms") or []),
    }
    return {
        key: value
        for key, value in details.items()
        if value not in (None, "", [], {})
    }


def _campaign_generation_user_prompt(
    *,
    target_mode: str,
    channel: str,
    opportunity_json: str,
    prior_invalid_response: str = "",
) -> str:
    prompt = (
        "Generate one campaign draft from this normalized "
        f"opportunity.\ntarget_mode={target_mode}\n"
        f"channel={channel}\n"
        f"opportunity={opportunity_json}"
    )
    return retry_prompt_with_invalid_response(
        prompt,
        prior_invalid_response=prior_invalid_response,
        instruction=(
            "The previous response was not valid campaign JSON. "
            "Return one JSON object with non-empty subject and body."
        ),
    )


def _contains_placeholder_url(value: Any) -> bool:
    return bool(_PLACEHOLDER_URL_RE.search(str(value or "")))


@dataclass(frozen=True)
class CampaignGenerationConfig:
    skill_name: str = "digest/b2b_campaign_generation"
    limit: int = 20
    max_tokens: int = 1200
    temperature: float = 0.4
    include_source_opportunity: bool = True
    channels: tuple[str, ...] = ()
    quality_revalidation_enabled: bool = False
    quality_prompt_proof_term_limit: int = 5
    parse_retry_attempts: int = 1
    parse_retry_response_excerpt_chars: int = 800


@dataclass(frozen=True)
class CampaignGenerationResult:
    requested: int = 0
    generated: int = 0
    skipped: int = 0
    reasoning_contexts_used: int = 0
    consumed_reasoning_contexts: tuple[Mapping[str, Any], ...] = ()
    saved_ids: tuple[str, ...] = ()
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

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


def _has_prompt_reasoning_context(opportunity: Mapping[str, Any]) -> bool:
    return isinstance(opportunity.get("campaign_reasoning_context"), Mapping)


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

    def with_reasoning_context(
        self,
        provider: CampaignReasoningContextProvider | None,
    ) -> "CampaignGenerationService":
        # PR-ControlSurfaces-Reasoning-Provider: route-level seam.
        return CampaignGenerationService(
            intelligence=self._intelligence,
            campaigns=self._campaigns,
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
        channels: Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        quality_revalidation_enabled: bool | None = None,
        quality_prompt_proof_term_limit: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
    ) -> CampaignGenerationResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(f"Campaign generation skill not found: {self._config.skill_name}")

        # PR-OptionA-2: per-call LLM-tuning overrides win over construction
        # config; None falls through. ``parse_retry_attempts`` accepts 0
        # (disable retries), so we have to check None explicitly rather than
        # truthiness.
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
        # PR-OptionA-3: quality + retry-excerpt knobs. Same shape.
        resolved_quality_revalidation_enabled = (
            self._config.quality_revalidation_enabled
            if quality_revalidation_enabled is None
            else bool(quality_revalidation_enabled)
        )
        resolved_quality_prompt_proof_term_limit = (
            self._config.quality_prompt_proof_term_limit
            if quality_prompt_proof_term_limit is None
            else int(quality_prompt_proof_term_limit)
        )
        resolved_parse_retry_response_excerpt_chars = (
            self._config.parse_retry_response_excerpt_chars
            if parse_retry_response_excerpt_chars is None
            else int(parse_retry_response_excerpt_chars)
        )

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
        reasoning_contexts_used = 0
        consumed_reasoning_contexts: list[dict[str, Any]] = []
        resolved_channels = self._channels(override=channels)
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
            for channel in resolved_channels:
                channel_opportunity = self._opportunity_for_channel(
                    opportunity,
                    channel=channel,
                    cold_email_context=cold_email_context,
                    quality_revalidation_enabled=resolved_quality_revalidation_enabled,
                    quality_prompt_proof_term_limit=resolved_quality_prompt_proof_term_limit,
                )
                try:
                    parsed = await self._generate_one(
                        prompt_template,
                        opportunity=channel_opportunity,
                        target_mode=target_mode,
                        channel=channel,
                        temperature=resolved_temperature,
                        max_tokens=resolved_max_tokens,
                        parse_retry_attempts=resolved_parse_retry_attempts,
                        parse_retry_response_excerpt_chars=resolved_parse_retry_response_excerpt_chars,
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
                if _contains_placeholder_url(parsed.get("body")) or _contains_placeholder_url(
                    parsed.get("cta")
                ):
                    skipped += 1
                    errors.append({
                        "target_id": target_id,
                        "channel": channel,
                        "reason": "placeholder_url",
                    })
                    continue
                parsed, revalidation_error = self._revalidated_parsed(
                    parsed,
                    opportunity=channel_opportunity,
                    target_mode=target_mode,
                    channel=channel,
                    quality_revalidation_enabled=resolved_quality_revalidation_enabled,
                    quality_prompt_proof_term_limit=resolved_quality_prompt_proof_term_limit,
                )
                if not parsed:
                    skipped += 1
                    error = {
                        "target_id": target_id,
                        "channel": channel,
                        "reason": "quality_revalidation_failed",
                    }
                    if revalidation_error:
                        error["quality_revalidation"] = revalidation_error
                    errors.append(error)
                    continue
                if channel == "email_cold":
                    cold_email_context = {
                        "subject": parsed["subject"],
                        "body": parsed["body"],
                    }
                if _has_prompt_reasoning_context(channel_opportunity):
                    reasoning_contexts_used += 1
                    consumed_reasoning_contexts.extend(
                        consumed_campaign_reasoning_contexts(channel_opportunity)
                    )
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
            reasoning_contexts_used=reasoning_contexts_used,
            consumed_reasoning_contexts=tuple(consumed_reasoning_contexts),
            saved_ids=saved_ids,
            errors=tuple(errors),
        )

    def _channels(self, *, override: Sequence[str] | None = None) -> tuple[str, ...]:
        # Per-call override (when present) wins over the construction-time
        # config. PR-OptionA-1: makes the plan's step.config["channels"]
        # load-bearing at dispatch time so the control surface preview's
        # channel selection actually reaches the service. None or an empty
        # override falls through to the existing config-then-default chain.
        if override is not None:
            override_channels = _normalize_channels(override)
            if override_channels:
                return override_channels
        raw_value = self._config.channels
        if isinstance(raw_value, str):
            raw: Sequence[str] = raw_value.split(",")
        else:
            raw = raw_value
        normalized = _normalize_channels(raw)
        return normalized or ("email",)

    def _opportunity_for_channel(
        self,
        opportunity: Mapping[str, Any],
        *,
        channel: str,
        cold_email_context: Mapping[str, Any] | None = None,
        quality_revalidation_enabled: bool,
        quality_prompt_proof_term_limit: int,
    ) -> dict[str, Any]:
        enriched = dict(opportunity)
        enriched["channel"] = channel
        if channel == "email_followup" and cold_email_context:
            enriched["cold_email_context"] = {
                "subject": str(cold_email_context.get("subject") or ""),
                "body": str(cold_email_context.get("body") or ""),
            }
        if quality_revalidation_enabled:
            enriched = self._with_quality_prompt_terms(
                enriched,
                quality_prompt_proof_term_limit=quality_prompt_proof_term_limit,
            )
        return enriched

    def _with_quality_prompt_terms(
        self,
        opportunity: Mapping[str, Any],
        *,
        quality_prompt_proof_term_limit: int,
    ) -> dict[str, Any]:
        enriched = dict(opportunity)
        context = normalize_campaign_reasoning_context(enriched)
        terms = self._campaign_proof_terms(
            context.as_dict(),
            existing=enriched.get("campaign_proof_terms"),
            quality_prompt_proof_term_limit=quality_prompt_proof_term_limit,
        )
        enriched.pop("campaign_proof_terms", None)
        if terms:
            enriched["campaign_proof_terms"] = terms
        return enriched

    def _campaign_proof_terms(
        self,
        context: Mapping[str, Any],
        *,
        existing: Any = None,
        quality_prompt_proof_term_limit: int,
    ) -> list[str]:
        limit = max(0, int(quality_prompt_proof_term_limit or 0))
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
        temperature: float,
        max_tokens: int,
        parse_retry_attempts: int,
        parse_retry_response_excerpt_chars: int,
    ) -> dict[str, Any] | None:
        opportunity_json = json.dumps(dict(opportunity), separators=(",", ":"), default=str)
        system_prompt = (
            prompt_template
            .replace("{target_mode}", target_mode)
            .replace("{channel}", channel)
            .replace("{opportunity}", opportunity_json)
            .replace("{opportunity_json}", opportunity_json)
        )
        attempts = parse_attempt_limit(parse_retry_attempts)
        last_response = ""
        total_usage: dict[str, Any] = {}
        for attempt_no in range(1, attempts + 1):
            response = await self._llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(
                        role="user",
                        content=_campaign_generation_user_prompt(
                            target_mode=target_mode,
                            channel=channel,
                            opportunity_json=opportunity_json,
                            prior_invalid_response=last_response,
                        ),
                    ),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={
                    "target_mode": target_mode,
                    "target_id": opportunity_target_id(opportunity),
                    "channel": channel,
                    "skill_name": self._config.skill_name,
                    "attempt_no": attempt_no,
                },
            )
            total_usage = accumulate_usage(total_usage, response.usage)
            parsed = parse_campaign_draft_response(response.content)
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

    def _revalidated_parsed(
        self,
        parsed: Mapping[str, Any],
        *,
        opportunity: Mapping[str, Any],
        target_mode: str,
        channel: str,
        quality_revalidation_enabled: bool,
        quality_prompt_proof_term_limit: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not quality_revalidation_enabled:
            return dict(parsed), None
        context = normalize_campaign_reasoning_context(opportunity)
        specificity_context = context.as_dict()
        proof_terms = _clean_term_list(
            opportunity.get("campaign_proof_terms"),
            limit=max(0, int(quality_prompt_proof_term_limit or 0)),
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
            return None, _revalidation_error_details(revalidation)
        return {
            **dict(parsed),
            "_quality_revalidation": revalidation,
        }, None

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
            "generation_parse_attempts": parsed.get("_parse_attempts"),
            "campaign_revalidation": parsed.get("_quality_revalidation"),
        }
        context = normalize_campaign_reasoning_context(opportunity)
        metadata.update(campaign_reasoning_context_metadata(context))
        if self._config.include_source_opportunity:
            metadata["source_opportunity"] = dict(opportunity)
        return {key: value for key, value in metadata.items() if value not in (None, "", {})}
