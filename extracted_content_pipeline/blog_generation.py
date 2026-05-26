"""Standalone blog-post generator orchestration for AI Content Ops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from .blog_ports import BlogBlueprintRepository, BlogPostDraft, BlogPostRepository
from .campaign_ports import (
    CampaignReasoningContextProvider,
    LLMClient,
    LLMMessage,
    SkillStore,
    TenantScope,
)
from .services._parse_retry_helpers import (
    accumulate_usage,
    clip_invalid_response,
    parse_attempt_limit,
    retry_prompt_with_invalid_response,
)
from .services.campaign_reasoning_context import (
    campaign_reasoning_context_metadata,
    campaign_reasoning_context_payload,
    consumed_campaign_reasoning_contexts,
    normalize_campaign_reasoning_context,
)
from .support_ticket_context_contract import is_support_ticket_context
from .support_ticket_generated_content_eval import (
    evaluate_support_ticket_generated_content,
)
from extracted_quality_gate.blog_pack import evaluate_blog_post
from extracted_quality_gate.types import QualityInput, QualityPolicy


# PR-Blog-Reasoning-Parity: fixed lookup mode for the reasoning-context
# port. Mirrors landing_page_generation's _TARGET_MODE constant; the
# call-site target_mode (e.g. "vendor_retention") is an unrelated
# tenant-scope concept.
_BLOG_REASONING_TARGET_MODE = "blog_blueprint"
_BLOG_FAILURE_EXCERPT_CHARS = 1500
_SMALL_SUPPORT_TICKET_BLOG_MAX_ROWS = 25
_SMALL_SUPPORT_TICKET_BLOG_MIN_WORDS = 700
_SMALL_SUPPORT_TICKET_BLOG_TARGET_WORDS = 1100


@dataclass(frozen=True)
class BlogPostGenerationConfig:
    """Tunable defaults for ``BlogPostGenerationService``."""

    skill_name: str = "digest/blog_post_generation"
    limit: int = 10
    max_tokens: int = 4096
    temperature: float = 0.3
    quality_policy: QualityPolicy | None = None
    quality_gates_enabled: bool = True
    quality_repair_attempts: int = 2
    parse_retry_attempts: int = 1
    parse_retry_response_excerpt_chars: int = 800


@dataclass(frozen=True)
class BlogPostGenerationResult:
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


def parse_blog_post_response(text: str) -> dict[str, Any] | None:
    """Extract the first valid blog-post JSON object from LLM output."""

    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    decoder = json.JSONDecoder()
    index = 0
    while index < len(cleaned):
        if cleaned[index] != "{":
            index += 1
            continue
        try:
            decoded, end = decoder.raw_decode(cleaned, index)
        except json.JSONDecodeError:
            index += 1
            continue
        index = end if end > index else index + 1
        if not isinstance(decoded, Mapping):
            continue
        title = str(decoded.get("title") or "").strip()
        # PR-Audit-MINOR-Batch-2: parser identifies a candidate (non-empty
        # ``title``); the quality pack judges the rest. Pre-fix required
        # both ``title`` AND ``content`` so a missing-content draft would
        # collapse to ``unparseable_response`` -- a missing-content blocker
        # at the quality-pack layer is more useful. (Empty content fires
        # ``content_too_short`` because zero words is below any min_words.)
        # ``content`` is still stripped here when present so downstream
        # consumers see normalized whitespace.
        if title:
            content = str(decoded.get("content") or "").strip()
            return {**dict(decoded), "title": title, "content": content}
    return None


def _has_prompt_reasoning_context(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload.get("campaign_reasoning_context"), Mapping)


def _blog_generation_user_prompt(
    *,
    base_prompt: str,
    prior_invalid_response: str = "",
) -> str:
    return retry_prompt_with_invalid_response(
        base_prompt,
        prior_invalid_response=prior_invalid_response,
        instruction=(
            "The previous response could not be parsed as the required JSON object. "
            "Return one JSON object with a non-empty title."
        ),
    )


def _blog_quality_repair_user_prompt(
    *,
    base_prompt: str,
    blockers: Sequence[str],
    previous_json: str,
) -> str:
    blocker_text = "\n".join(f"- {item}" for item in blockers)
    guidance = _blog_quality_repair_guidance(blockers)
    return (
        f"{base_prompt}\n\n"
        "The previous blog JSON parsed, but it failed the blog quality gate.\n"
        "Quality blockers:\n"
        f"{blocker_text}\n\n"
        f"Repair instructions:\n{guidance}\n\n"
        "Return the full blog JSON object again. Keep the valid fields that already "
        "worked, but revise the draft so it fixes every blocker above. Do not return "
        "commentary, markdown fences, or partial JSON.\n\n"
        "Previous parsed JSON:\n"
        f"{previous_json}"
    )


def _blog_quality_repair_guidance(blockers: Sequence[str]) -> str:
    """Translate gate blocker codes into concrete blog-edit instructions."""

    instructions: list[str] = []
    for blocker in blockers:
        code = str(blocker or "").strip()
        if code.startswith("content_too_short:"):
            min_words_match = re.search(r"_need_(\d+)", code)
            min_words = int(min_words_match.group(1)) if min_words_match else 1500
            if min_words == _SMALL_SUPPORT_TICKET_BLOG_MIN_WORDS:
                instructions.append(
                    f"- Expand `content` to at least {min_words} words while "
                    "keeping the compact support-ticket brief shape. Use 3-4 "
                    "H2 sections, no H3 subsections, and no broad scaling or "
                    "process section."
                )
                continue
            instructions.append(
                "- Expand `content` to at least 1500 words and keep it in the "
                "1500-2200 word range. Add useful H2 sections and supporting "
                "paragraphs instead of filler."
            )
        elif code.startswith("seo_title_too_long:"):
            instructions.append(
                "- Shorten `seo_title` to 60 characters or fewer while keeping "
                "the target keyword near the front."
            )
        elif code == "geo_entity_clarity_missing":
            instructions.append(
                "- Update the display `title` to include the exact current "
                "`target_keyword` string from the previous JSON. Also repeat "
                "that exact phrase naturally in the first 40-60 words of `content`."
            )
        elif code == "geo_citable_section_structure_missing":
            instructions.append(
                "- Make at least two H2 sections independently citable. Each of "
                "those sections must start with a 40-120 word answer paragraph "
                "that includes the exact `target_keyword` or clearest named subject."
            )
        elif code == "geo_citation_safety_failed":
            instructions.append(
                "- Remove unresolved placeholders, placeholder links, unknown chart "
                "placeholders, and unsupported claims. Keep only claims grounded in "
                "the blueprint data, source wording, or visible chart IDs."
            )
        elif code.startswith("support_ticket_generated_content:"):
            instructions.append(
                "- Fix the support-ticket generated-content issue exactly. Use only "
                "counts, timeframes, clusters, and customer wording present in the "
                "blueprint. Do not invent calendar windows, ticket-reduction "
                "percentages, ROI math, future impact claims, or claims that "
                "customers will or could find answers without opening support tickets."
            )
    if not instructions:
        instructions.append(
            "- Fix each blocker directly while preserving the required blog JSON "
            "schema and the factual constraints from the blueprint."
        )
    return "\n".join(dict.fromkeys(instructions))


def _normalize_blog_metadata(
    parsed: Mapping[str, Any],
    *,
    quality_policy: QualityPolicy | None,
) -> dict[str, Any]:
    """Apply deterministic metadata limits before quality validation."""

    normalized = dict(parsed)
    seo_title = str(normalized.get("seo_title") or "").strip()
    seo_title_max = 60
    if quality_policy is not None:
        raw_max = quality_policy.thresholds.get("seo_title_max_chars")
        if isinstance(raw_max, (int, float)) and not isinstance(raw_max, bool):
            seo_title_max = int(raw_max)
    if seo_title and len(seo_title) > seo_title_max:
        normalized["seo_title"] = _truncate_text_at_word_boundary(
            seo_title,
            max_chars=seo_title_max,
        )
    return normalized


def _truncate_text_at_word_boundary(value: str, *, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    trimmed = text[:limit].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0].rstrip()
    return trimmed or text[:limit].rstrip()


def _blog_failure_candidate_snapshot(
    parsed: Mapping[str, Any],
    *,
    excerpt_chars: int = _BLOG_FAILURE_EXCERPT_CHARS,
) -> dict[str, Any]:
    """Return bounded diagnostics for a parsed draft that cannot be saved."""

    content = str(parsed.get("content") or "").strip()
    limit = max(1, int(excerpt_chars))
    return {
        "title": str(parsed.get("title") or "").strip(),
        "slug": str(parsed.get("slug") or "").strip(),
        "seo_title": str(parsed.get("seo_title") or "").strip(),
        "target_keyword": str(parsed.get("target_keyword") or "").strip(),
        "topic_type": str(parsed.get("topic_type") or "").strip(),
        "word_count": len(content.split()),
        "generation_parse_attempts": parsed.get("_parse_attempts"),
        "generation_quality_repair_attempts": (
            parsed.get("_quality_repair_attempts") or 0
        ),
        "content_excerpt_head": content[:limit],
        "content_excerpt_tail": content[-limit:] if len(content) > limit else "",
        "content_truncated": len(content) > limit,
    }


class BlogPostGenerationService:
    """Generate blog-post drafts through product-owned ports."""

    def __init__(
        self,
        *,
        blueprints: BlogBlueprintRepository,
        blog_posts: BlogPostRepository,
        llm: LLMClient,
        skills: SkillStore,
        reasoning_context: CampaignReasoningContextProvider | None = None,
        config: BlogPostGenerationConfig | None = None,
    ):
        self._blueprints = blueprints
        self._blog_posts = blog_posts
        self._llm = llm
        self._skills = skills
        # PR-Blog-Reasoning-Parity: brings blog to constructor parity
        # with the other 4 generators. When wired, supplemental
        # reasoning context is merged into each blueprint before the
        # LLM call -- additive enrichment, not replacement.
        self._reasoning_context = reasoning_context
        self._config = config or BlogPostGenerationConfig()

    def with_reasoning_context(
        self,
        provider: CampaignReasoningContextProvider | None,
    ) -> "BlogPostGenerationService":
        # PR-ControlSurfaces-Reasoning-Provider: route-level seam. The
        # /execute route derives a fresh services bundle per request
        # so a host-supplied reasoning provider can be wired without
        # mutating the cached service instance.
        return BlogPostGenerationService(
            blueprints=self._blueprints,
            blog_posts=self._blog_posts,
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
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        quality_gates_enabled: bool | None = None,
        quality_repair_attempts: int | None = None,
        topic: str | None = None,
        data_context: Mapping[str, Any] | None = None,
    ) -> BlogPostGenerationResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(f"Blog generation skill not found: {self._config.skill_name}")

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
        # PR-OptionA-5: per-call quality gate opt-out (symmetry with
        # the other content-asset services).
        resolved_quality_gates_enabled = (
            self._config.quality_gates_enabled
            if quality_gates_enabled is None
            else bool(quality_gates_enabled)
        )
        resolved_quality_repair_attempts = (
            self._config.quality_repair_attempts
            if quality_repair_attempts is None
            else int(quality_repair_attempts)
        )
        if not resolved_quality_gates_enabled:
            resolved_quality_repair_attempts = 0
        resolved_quality_repair_attempts = max(0, resolved_quality_repair_attempts)
        # PR-Blog-Topic-Per-Call: operator-supplied topic for this run.
        # Empty string when None so prompt substitution is a clean no-op.
        resolved_topic = (topic or "").strip()
        trusted_data_context = _mapping_dict(data_context)

        requested = int(limit or self._config.limit)
        rows = await self._blueprints.read_blog_blueprints(
            scope=scope,
            target_mode=target_mode,
            limit=requested,
            filters=filters,
        )

        drafts: list[BlogPostDraft] = []
        errors: list[dict[str, Any]] = []
        skipped = 0
        reasoning_contexts_used = 0
        consumed_reasoning_contexts: list[dict[str, Any]] = []
        for row in rows:
            blueprint = await self._blueprint_with_reasoning_context(
                scope=scope,
                blueprint=dict(row),
            )
            blueprint = _blueprint_with_data_context(blueprint, trusted_data_context)
            blueprint_id = _blueprint_id(blueprint)
            try:
                parsed = await self._generate_one(
                    prompt_template,
                    blueprint=blueprint,
                    target_mode=target_mode,
                    temperature=resolved_temperature,
                    max_tokens=resolved_max_tokens,
                    parse_retry_attempts=resolved_parse_retry_attempts,
                    parse_retry_response_excerpt_chars=resolved_parse_retry_response_excerpt_chars,
                    topic=resolved_topic,
                )
            except Exception as exc:
                skipped += 1
                errors.append({"blueprint_id": blueprint_id, "reason": str(exc)})
                continue
            if not parsed:
                skipped += 1
                errors.append({"blueprint_id": blueprint_id, "reason": "unparseable_response"})
                continue
            parsed = _normalize_blog_metadata(
                parsed,
                quality_policy=self._config.quality_policy,
            )
            quality = self._quality_check(
                parsed,
                blueprint=blueprint,
                quality_gates_enabled=resolved_quality_gates_enabled,
            )
            if not quality["passed"]:
                repair_failed = False
                for repair_attempt_no in range(1, resolved_quality_repair_attempts + 1):
                    try:
                        repaired = await self._repair_quality_once(
                            prompt_template,
                            parsed=parsed,
                            quality=quality,
                            blueprint=blueprint,
                            target_mode=target_mode,
                            temperature=resolved_temperature,
                            max_tokens=resolved_max_tokens,
                            quality_repair_attempt_no=repair_attempt_no,
                            topic=resolved_topic,
                        )
                    except Exception as exc:
                        skipped += 1
                        errors.append({
                            "blueprint_id": blueprint_id,
                            "reason": "quality_repair_failed",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        })
                        repair_failed = True
                        break
                    if not repaired:
                        skipped += 1
                        errors.append({
                            "blueprint_id": blueprint_id,
                            "reason": "quality_repair_unparseable",
                            "blockers": quality["blockers"],
                            "quality_repair_attempt_no": repair_attempt_no,
                            "failed_candidate": _blog_failure_candidate_snapshot(parsed),
                        })
                        repair_failed = True
                        break
                    parsed = _normalize_blog_metadata(
                        repaired,
                        quality_policy=self._config.quality_policy,
                    )
                    quality = self._quality_check(
                        parsed,
                        blueprint=blueprint,
                        quality_gates_enabled=resolved_quality_gates_enabled,
                    )
                    if quality["passed"]:
                        break
                if repair_failed:
                    continue
                if not quality["passed"]:
                    skipped += 1
                    errors.append({
                        "blueprint_id": blueprint_id,
                        "reason": "quality_blocked",
                        "blockers": quality["blockers"],
                        "failed_candidate": _blog_failure_candidate_snapshot(parsed),
                    })
                    continue
            if _has_prompt_reasoning_context(blueprint):
                reasoning_contexts_used += 1
                consumed_reasoning_contexts.extend(
                    consumed_campaign_reasoning_contexts(blueprint)
                )
            drafts.append(self._build_draft(parsed, blueprint=blueprint))

        saved_ids: tuple[str, ...] = ()
        if drafts:
            saved_ids = tuple(
                str(item)
                for item in await self._blog_posts.save_drafts(drafts, scope=scope)
            )
        return BlogPostGenerationResult(
            requested=len(rows),
            generated=len(drafts),
            skipped=skipped,
            reasoning_contexts_used=reasoning_contexts_used,
            consumed_reasoning_contexts=tuple(consumed_reasoning_contexts),
            saved_ids=saved_ids,
            errors=tuple(errors),
        )

    async def _blueprint_with_reasoning_context(
        self,
        *,
        scope: TenantScope,
        blueprint: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        # PR-Blog-Reasoning-Parity: mirrors landing_page_generation's
        # _payload_with_reasoning_context shape. When no provider is
        # wired the blueprint passes through unchanged. When the
        # provider returns no content the blueprint also passes
        # through unchanged so empty / missing reasoning is a no-op.
        if self._reasoning_context is None:
            return blueprint
        provided = await self._reasoning_context.read_campaign_reasoning_context(
            scope=scope,
            target_id=_blueprint_id(blueprint),
            target_mode=_BLOG_REASONING_TARGET_MODE,
            opportunity=blueprint,
        )
        provided_context = normalize_campaign_reasoning_context(provided)
        if not provided_context.has_content():
            return blueprint
        enriched = dict(blueprint)
        reasoning_payload = campaign_reasoning_context_payload(provided_context)
        enriched.update(campaign_reasoning_context_metadata(provided_context))
        enriched["reasoning_context"] = reasoning_payload
        enriched["campaign_reasoning_context"] = reasoning_payload
        return enriched

    async def _generate_one(
        self,
        prompt_template: str,
        *,
        blueprint: Mapping[str, Any],
        target_mode: str,
        temperature: float,
        max_tokens: int,
        parse_retry_attempts: int,
        parse_retry_response_excerpt_chars: int,
        topic: str = "",
    ) -> dict[str, Any] | None:
        system_prompt, base_user_prompt = _blog_generation_prompts(
            prompt_template,
            blueprint=blueprint,
            topic=topic,
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
                        content=_blog_generation_user_prompt(
                            base_prompt=base_user_prompt,
                            prior_invalid_response=last_response,
                        ),
                    ),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={
                    "target_mode": target_mode,
                    "blueprint_id": _blueprint_id(blueprint),
                    "skill_name": self._config.skill_name,
                    "asset_type": "blog_post",
                    "attempt_no": attempt_no,
                },
            )
            total_usage = accumulate_usage(total_usage, response.usage)
            parsed = parse_blog_post_response(response.content)
            if parsed:
                return {
                    **parsed,
                    "_model": response.model,
                    "_usage": total_usage,
                    "_parse_attempts": attempt_no,
                    "_quality_repair_attempts": 0,
                }
            last_response = clip_invalid_response(
                response.content,
                limit=max(0, int(parse_retry_response_excerpt_chars or 0)),
            )
        return None

    async def _repair_quality_once(
        self,
        prompt_template: str,
        *,
        parsed: Mapping[str, Any],
        quality: Mapping[str, Any],
        blueprint: Mapping[str, Any],
        target_mode: str,
        temperature: float,
        max_tokens: int,
        quality_repair_attempt_no: int,
        topic: str = "",
    ) -> dict[str, Any] | None:
        blockers = tuple(str(item) for item in quality.get("blockers") or () if item)
        if not blockers:
            return None
        parse_attempts_used = int(parsed.get("_parse_attempts") or 1)
        system_prompt, base_user_prompt = _blog_generation_prompts(
            prompt_template,
            blueprint=blueprint,
            topic=topic,
        )
        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=_blog_quality_repair_user_prompt(
                        base_prompt=base_user_prompt,
                        blockers=blockers,
                        previous_json=_public_blog_json(parsed),
                    ),
                ),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            metadata={
                "target_mode": target_mode,
                "blueprint_id": _blueprint_id(blueprint),
                "skill_name": self._config.skill_name,
                "asset_type": "blog_post",
                "attempt_no": parse_attempts_used + quality_repair_attempt_no,
                "quality_repair_attempt_no": quality_repair_attempt_no,
            },
        )
        repaired = parse_blog_post_response(response.content)
        if not repaired:
            return None
        return {
            **repaired,
            "_model": response.model,
            "_usage": accumulate_usage(
                dict(parsed.get("_usage") or {}),
                response.usage,
            ),
            "_parse_attempts": parse_attempts_used,
            "_quality_repair_attempts": int(parsed.get("_quality_repair_attempts") or 0) + 1,
        }

    def _quality_check(
        self,
        parsed: Mapping[str, Any],
        *,
        blueprint: Mapping[str, Any],
        quality_gates_enabled: bool = True,
    ) -> dict[str, Any]:
        # PR-OptionA-5: opt out of the quality gate per call. Same shape
        # as report / sales_brief / landing_page.
        if not quality_gates_enabled:
            return {"passed": True, "blockers": ()}
        context = _quality_context(parsed, blueprint)
        quality = evaluate_blog_post(
            QualityInput(
                artifact_type="blog_post",
                artifact_id=str(parsed.get("slug") or blueprint.get("slug") or ""),
                content=str(parsed.get("content") or ""),
                context=context,
            ),
            policy=_quality_policy_for_context(
                context,
                base_policy=self._config.quality_policy,
            ),
        )
        support_ticket_blockers = _support_ticket_generated_content_blockers(
            parsed,
            blueprint=blueprint,
        )
        return {
            "passed": quality.passed and not support_ticket_blockers,
            "blockers": tuple(f.message for f in quality.blockers) + support_ticket_blockers,
        }

    def _build_draft(
        self,
        parsed: Mapping[str, Any],
        *,
        blueprint: Mapping[str, Any],
    ) -> BlogPostDraft:
        title = str(parsed.get("title") or "").strip()
        slug = _slugify(parsed.get("slug") or blueprint.get("slug") or title)
        topic_type = str(
            parsed.get("topic_type")
            or blueprint.get("topic_type")
            or "blog_post"
        ).strip() or "blog_post"
        data_context = _merged_blog_data_context(parsed, blueprint)
        metadata = {
            "seo_title": parsed.get("seo_title"),
            "seo_description": parsed.get("seo_description"),
            "target_keyword": parsed.get("target_keyword"),
            "secondary_keywords": list(_string_tuple(parsed.get("secondary_keywords"))),
            "faq": _mapping_list(parsed.get("faq")),
            "generation_model": parsed.get("_model"),
            "generation_usage": parsed.get("_usage") or {},
            "generation_parse_attempts": parsed.get("_parse_attempts"),
            "generation_quality_repair_attempts": parsed.get("_quality_repair_attempts") or 0,
        }
        # PR-Blog-Reasoning-Parity: surface reasoning audit fields on
        # the draft metadata when the blueprint carried merged context.
        # Mirrors campaign / report / sales_brief metadata threading.
        for reasoning_key in (
            "reasoning_context",
            "reasoning_anchor_examples",
            "reasoning_witness_highlights",
            "reasoning_reference_ids",
            "reasoning_provider",
        ):
            if reasoning_key in blueprint:
                metadata[reasoning_key] = blueprint[reasoning_key]
        return BlogPostDraft(
            slug=slug,
            title=title,
            description=str(
                parsed.get("description")
                or parsed.get("seo_description")
                or ""
            ).strip(),
            topic_type=topic_type,
            tags=_string_tuple(parsed.get("tags") or parsed.get("secondary_keywords")),
            content=str(parsed.get("content") or "").strip(),
            charts=_chart_tuple(parsed.get("charts") or blueprint.get("available_charts")),
            data_context=data_context,
            metadata=metadata,
        )


def _quality_context(
    parsed: Mapping[str, Any],
    blueprint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "topic_type": parsed.get("topic_type") or blueprint.get("topic_type"),
        "slug": parsed.get("slug") or blueprint.get("slug"),
        "suggested_title": parsed.get("title") or blueprint.get("suggested_title"),
        "data_context": _mapping_dict(blueprint.get("data_context")),
        "charts": parsed.get("charts") or blueprint.get("available_charts") or (),
        "source_quotes": _source_quote_tuple(blueprint.get("quotable_phrases")),
        "required_vendors": _string_tuple(blueprint.get("required_vendors")),
        "grounded_vendors": frozenset(_string_tuple(blueprint.get("grounded_vendors"))),
        "require_seo_aeo": True,
        "require_geo": True,
        "title": parsed.get("title"),
        "seo_title": parsed.get("seo_title"),
        "seo_description": parsed.get("seo_description"),
        "target_keyword": parsed.get("target_keyword"),
        "secondary_keywords": parsed.get("secondary_keywords"),
        "faq": parsed.get("faq"),
    }


def _quality_policy_for_context(
    context: Mapping[str, Any],
    *,
    base_policy: QualityPolicy | None,
) -> QualityPolicy | None:
    data_context = _mapping_dict(context.get("data_context"))
    if not _uses_small_support_ticket_blog_policy(data_context):
        return base_policy

    thresholds = dict(base_policy.thresholds) if base_policy is not None else {}
    thresholds.setdefault("min_words", _SMALL_SUPPORT_TICKET_BLOG_MIN_WORDS)
    thresholds.setdefault("target_words", _SMALL_SUPPORT_TICKET_BLOG_TARGET_WORDS)
    metadata = dict(base_policy.metadata) if base_policy is not None else {}
    metadata["support_ticket_small_upload"] = True
    return QualityPolicy(
        name=base_policy.name if base_policy is not None else "support_ticket_blog",
        version=base_policy.version if base_policy is not None else "v1",
        thresholds=thresholds,
        metadata=metadata,
    )


def _uses_small_support_ticket_blog_policy(data_context: Mapping[str, Any]) -> bool:
    if not _is_support_ticket_blog_context(data_context):
        return False
    if _truthy_context(data_context.get("has_measured_outcomes")):
        return False
    if _truthy_context(data_context.get("support_ticket_resolution_evidence_present")):
        return False
    row_counts = (
        _positive_int_context(data_context.get("source_row_count")),
        _positive_int_context(data_context.get("included_ticket_row_count")),
    )
    return any(
        count is not None and count <= _SMALL_SUPPORT_TICKET_BLOG_MAX_ROWS
        for count in row_counts
    )


def _positive_int_context(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _truthy_context(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _support_ticket_generated_content_blockers(
    parsed: Mapping[str, Any],
    *,
    blueprint: Mapping[str, Any],
) -> tuple[str, ...]:
    data_context = _merged_blog_data_context(parsed, blueprint)
    if not _is_support_ticket_blog_context(data_context):
        return ()
    row = {
        "id": str(parsed.get("slug") or blueprint.get("slug") or blueprint.get("id") or ""),
        "title": parsed.get("title"),
        "description": parsed.get("description") or parsed.get("seo_description"),
        "content": parsed.get("content"),
        "tags": parsed.get("tags") or parsed.get("secondary_keywords"),
        "charts": parsed.get("charts") or blueprint.get("available_charts"),
        "data_context": data_context,
    }
    result = evaluate_support_ticket_generated_content(
        {"count": 1, "rows": [row]},
        output="blog_post",
    )
    errors = tuple(str(error).strip() for error in result.get("errors") or ())
    if result.get("ok") is False and not errors:
        errors = ("support-ticket generated-content evaluation failed",)
    return tuple(
        f"support_ticket_generated_content:{error}"
        for error in errors
        if error
    )


def _merged_blog_data_context(
    parsed: Mapping[str, Any],
    blueprint: Mapping[str, Any],
) -> dict[str, Any]:
    data_context = _mapping_dict(parsed.get("data_context"))
    data_context.update(_mapping_dict(blueprint.get("data_context")))
    return data_context


def _blueprint_with_data_context(
    blueprint: Mapping[str, Any],
    data_context: Mapping[str, Any],
) -> Mapping[str, Any]:
    override = _mapping_dict(data_context)
    if not override:
        return blueprint
    enriched = dict(blueprint)
    merged = _mapping_dict(enriched.get("data_context"))
    merged.update(override)
    enriched["data_context"] = merged
    return enriched


def _is_support_ticket_blog_context(data_context: Mapping[str, Any]) -> bool:
    return is_support_ticket_context(data_context)


def _blog_generation_prompts(
    prompt_template: str,
    *,
    blueprint: Mapping[str, Any],
    topic: str = "",
) -> tuple[str, str]:
    blueprint_json = json.dumps(dict(blueprint), separators=(",", ":"), default=str)
    if "{blueprint_json}" in prompt_template:
        system_prompt = prompt_template.replace("{blueprint_json}", blueprint_json)
        base_user_prompt = "Generate one blog post from the blueprint above."
    else:
        system_prompt = prompt_template
        base_user_prompt = f"Generate one blog post from this blueprint JSON:\n{blueprint_json}"
    # PR-Blog-Topic-Per-Call: operator-supplied topic substitutes into
    # the ``{topic}`` placeholder. Empty topic resolves to "" so hosts
    # on the prior prompt (without ``{topic}``) are unaffected -- the
    # ``replace()`` is a no-op when the placeholder isn't present.
    system_prompt = system_prompt.replace("{topic}", topic)
    return system_prompt, base_user_prompt


def _public_blog_json(parsed: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in parsed.items()
        if not str(key).startswith("_")
    }
    return json.dumps(payload, separators=(",", ":"), default=str)


def _blueprint_id(blueprint: Mapping[str, Any]) -> str:
    return str(
        blueprint.get("id")
        or blueprint.get("slug")
        or blueprint.get("topic")
        or blueprint.get("suggested_title")
        or ""
    ).strip()


# PR-Audit-MINOR-Batch-2: cap slug length at 100 chars. Pre-fix, a
# 2000-char title produced a 2000-char slug -- valid but unwieldy for
# CMS / URL routing (most expect <100). Truncation rather than
# rejection: a long-title draft is still valid; the slug just gets
# clipped. ``rstrip("-")`` after the cut prevents trailing-hyphen
# artifacts (``"a-very-long--"``).
_MAX_SLUG_CHARS = 100


def _slugify(value: Any) -> str:
    text = str(value or "blog-post").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(slug) > _MAX_SLUG_CHARS:
        slug = slug[:_MAX_SLUG_CHARS].rstrip("-")
    return slug or "blog-post"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _source_quote_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    quotes: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("phrase") or item.get("quote") or "").strip()
        if text:
            quotes.append(text)
    return tuple(quotes)


def _chart_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


__all__ = [
    "BlogPostGenerationConfig",
    "BlogPostGenerationResult",
    "BlogPostGenerationService",
    "parse_blog_post_response",
]
