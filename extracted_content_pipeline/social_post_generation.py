"""Deterministic social-post drafts from Content Ops source material."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .brand_voice import (
    BrandVoiceProfile,
    apply_brand_voice_to_system_prompt,
    brand_voice_profile_from_mapping,
    brand_voice_result_metadata,
)
from .campaign_customer_data import CampaignOpportunityWarning
from .campaign_ports import LLMClient, LLMMessage, SkillStore, TenantScope
from .campaign_source_adapters import source_row_to_campaign_opportunity
from .services._parse_retry_helpers import (
    accumulate_usage,
    clip_invalid_response,
    parse_attempt_limit,
    retry_prompt_with_invalid_response,
)
from .social_post_ports import SocialPostDraft, SocialPostRepository

_ROW_LIST_KEYS = ("sources", "opportunities", "reviews", "documents", "rows", "data")


@dataclass(frozen=True)
class SocialPostGenerationConfig:
    """Config for deterministic source-material social posts."""

    skill_name: str = "digest/social_post_generation"
    limit: int = 3
    max_text_chars: int = 600
    max_post_chars: int = 420
    max_tokens: int = 700
    temperature: float = 0.4
    parse_retry_attempts: int = 1
    parse_retry_response_excerpt_chars: int = 800


@dataclass(frozen=True)
class SocialPostGenerationResult:
    """Generated social posts plus non-fatal source-material warnings."""

    posts: tuple[dict[str, Any], ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    target_mode: str = "vendor_retention"
    saved_ids: tuple[str, ...] = ()

    @property
    def generated(self) -> int:
        return len(self.posts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "target_mode": self.target_mode,
            "posts": [dict(post) for post in self.posts],
            "warnings": [warning.as_dict() for warning in self.warnings],
            "saved_ids": list(self.saved_ids),
        }


class SocialPostGenerationService:
    """Build short, evidence-backed social posts from source material."""

    def __init__(
        self,
        config: SocialPostGenerationConfig | None = None,
        *,
        social_posts: SocialPostRepository | None = None,
        llm: LLMClient | None = None,
        skills: SkillStore | None = None,
    ) -> None:
        self.config = config or SocialPostGenerationConfig()
        self._social_posts = social_posts
        self._llm = llm
        self._skills = skills

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        limit: int | None = None,
        max_text_chars: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry_attempts: int | None = None,
        parse_retry_response_excerpt_chars: int | None = None,
        brand_voice: Mapping[str, Any] | BrandVoiceProfile | None = None,
        **kwargs: Any,
    ) -> SocialPostGenerationResult:
        del kwargs
        resolved_brand_voice = brand_voice_profile_from_mapping(
            brand_voice,
            scope=scope,
        )
        resolved_limit = int(limit) if limit is not None else self.config.limit
        if resolved_limit < 1:
            raise ValueError("limit must be at least 1")
        resolved_max_text_chars = (
            int(max_text_chars)
            if max_text_chars is not None
            else self.config.max_text_chars
        )
        if resolved_max_text_chars < 1:
            raise ValueError("max_text_chars must be at least 1")
        result = _generate_social_posts(
            _rows_from_source_material(source_material),
            target_mode=target_mode,
            limit=resolved_limit,
            max_text_chars=resolved_max_text_chars,
            max_post_chars=self.config.max_post_chars,
        )
        if resolved_brand_voice is not None and result.posts:
            prompt_template = self._social_post_prompt()
            posts, warnings = await self._rewrite_posts_with_brand_voice(
                result.posts,
                target_mode=target_mode,
                prompt_template=prompt_template,
                brand_voice=resolved_brand_voice,
                temperature=(
                    self.config.temperature
                    if temperature is None
                    else float(temperature)
                ),
                max_tokens=(
                    self.config.max_tokens
                    if max_tokens is None
                    else int(max_tokens)
                ),
                parse_retry_attempts=(
                    self.config.parse_retry_attempts
                    if parse_retry_attempts is None
                    else int(parse_retry_attempts)
                ),
                parse_retry_response_excerpt_chars=(
                    self.config.parse_retry_response_excerpt_chars
                    if parse_retry_response_excerpt_chars is None
                    else int(parse_retry_response_excerpt_chars)
                ),
            )
            result = replace(
                result,
                posts=tuple(posts),
                warnings=tuple(result.warnings) + tuple(warnings),
            )
        return await self._persist_result(result, scope=scope, target_mode=target_mode)

    def _social_post_prompt(self) -> str:
        if self._llm is None or self._skills is None:
            raise ValueError(
                "brand voice social_post generation requires configured LLM and skill store"
            )
        prompt_template = self._skills.get_prompt(self.config.skill_name)
        if not prompt_template:
            raise ValueError(
                f"Social post generation skill not found: {self.config.skill_name}"
            )
        return prompt_template

    async def _rewrite_posts_with_brand_voice(
        self,
        posts: Sequence[Mapping[str, Any]],
        *,
        target_mode: str,
        prompt_template: str,
        brand_voice: BrandVoiceProfile,
        temperature: float,
        max_tokens: int,
        parse_retry_attempts: int,
        parse_retry_response_excerpt_chars: int,
    ) -> tuple[list[dict[str, Any]], list[CampaignOpportunityWarning]]:
        rewritten: list[dict[str, Any]] = []
        warnings: list[CampaignOpportunityWarning] = []
        system_prompt = apply_brand_voice_to_system_prompt(prompt_template, brand_voice)
        for index, post in enumerate(posts, start=1):
            try:
                parsed = await self._rewrite_one_post(
                    post,
                    target_mode=target_mode,
                    system_prompt=system_prompt,
                    brand_voice=brand_voice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    parse_retry_attempts=parse_retry_attempts,
                    parse_retry_response_excerpt_chars=parse_retry_response_excerpt_chars,
                )
            except Exception as exc:
                warnings.append(CampaignOpportunityWarning(
                    code="social_post_llm_error",
                    row_index=index,
                    message=(
                        "Skipped social post rewrite because LLM generation failed: "
                        f"{exc}"
                    ),
                ))
                continue
            if parsed is None:
                warnings.append(CampaignOpportunityWarning(
                    code="social_post_llm_unparseable_response",
                    row_index=index,
                    message=(
                        "Skipped social post rewrite because the LLM did not return "
                        "valid social-post JSON."
                    ),
                ))
                continue
            rewritten.append(parsed)
        return rewritten, warnings

    async def _rewrite_one_post(
        self,
        post: Mapping[str, Any],
        *,
        target_mode: str,
        system_prompt: str,
        brand_voice: BrandVoiceProfile,
        temperature: float,
        max_tokens: int,
        parse_retry_attempts: int,
        parse_retry_response_excerpt_chars: int,
    ) -> dict[str, Any] | None:
        if self._llm is None:
            raise ValueError("brand voice social_post generation requires configured LLM")
        prior_invalid_response = ""
        usage: dict[str, Any] = {}
        for _attempt in range(parse_attempt_limit(parse_retry_attempts)):
            response = await self._llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(
                        role="user",
                        content=_social_post_user_prompt(
                            target_mode=target_mode,
                            source_post=post,
                            max_post_chars=self.config.max_post_chars,
                            prior_invalid_response=prior_invalid_response,
                        ),
                    ),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={
                    "asset_type": "social_post",
                    "target_mode": target_mode,
                    "source_id": _clean(post.get("source_id") or post.get("id")),
                },
            )
            usage = accumulate_usage(usage, response.usage)
            parsed = parse_social_post_response(response.content)
            if parsed is None:
                prior_invalid_response = clip_invalid_response(
                    response.content,
                    limit=parse_retry_response_excerpt_chars,
                )
                continue
            text = _truncate(parsed["text"], self.config.max_post_chars)
            if not text:
                prior_invalid_response = clip_invalid_response(
                    response.content,
                    limit=parse_retry_response_excerpt_chars,
                )
                continue
            channel = (
                _clean(parsed.get("channel"))
                or _clean(post.get("channel"))
                or "linkedin"
            )
            rewritten = {
                **dict(post),
                "channel": channel,
                "text": text,
            }
            voice_metadata = brand_voice_result_metadata(
                {"channel": channel, "text": text},
                brand_voice,
            )
            if voice_metadata.get("_brand_voice_profile"):
                rewritten["_brand_voice_profile"] = voice_metadata[
                    "_brand_voice_profile"
                ]
            if voice_metadata.get("_brand_voice_audit"):
                rewritten["_brand_voice_audit"] = voice_metadata["_brand_voice_audit"]
            if usage:
                rewritten["_generation_usage"] = dict(usage)
            return rewritten
        return None

    async def _persist_result(
        self,
        result: SocialPostGenerationResult,
        *,
        scope: TenantScope,
        target_mode: str,
    ) -> SocialPostGenerationResult:
        if self._social_posts is None or not result.posts:
            return result
        saved_ids = tuple(
            str(item)
            for item in await self._social_posts.save_drafts(
                _drafts_from_posts(result.posts, target_mode=target_mode),
                scope=scope,
            )
        )
        return replace(result, saved_ids=saved_ids)


def parse_social_post_response(text: str) -> dict[str, str] | None:
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
        if not isinstance(candidate, Mapping):
            continue
        post_text = _clean(
            candidate.get("text")
            or candidate.get("body")
            or candidate.get("content")
        )
        if not post_text:
            continue
        return {
            "channel": _clean(candidate.get("channel")),
            "text": post_text,
        }
    return None


def _social_post_user_prompt(
    *,
    target_mode: str,
    source_post: Mapping[str, Any],
    max_post_chars: int,
    prior_invalid_response: str = "",
) -> str:
    prompt = (
        "Rewrite this source-backed social post.\n"
        f"target_mode={target_mode}\n"
        f"max_post_chars={max_post_chars}\n"
        "source_post="
        + json.dumps(dict(source_post), sort_keys=True, separators=(",", ":"))
    )
    return retry_prompt_with_invalid_response(
        prompt,
        prior_invalid_response=prior_invalid_response,
        instruction=(
            "The previous response was not valid social-post JSON. "
            "Return one JSON object with non-empty text."
        ),
    )


def _generate_social_posts(
    rows: Sequence[Any],
    *,
    target_mode: str,
    limit: int,
    max_text_chars: int,
    max_post_chars: int,
) -> SocialPostGenerationResult:
    posts: list[dict[str, Any]] = []
    warnings: list[CampaignOpportunityWarning] = []
    for index, row in enumerate(rows, start=1):
        if len(posts) >= limit:
            break
        if not isinstance(row, Mapping):
            warnings.append(CampaignOpportunityWarning(
                code="row_not_object",
                row_index=index,
                message="Skipped source row because it is not an object.",
            ))
            continue
        opportunity, row_warnings = source_row_to_campaign_opportunity(
            row,
            row_index=index,
            max_text_chars=max_text_chars,
        )
        warnings.extend(row_warnings)
        post = _post_from_opportunity(
            opportunity,
            index=len(posts) + 1,
            max_post_chars=max_post_chars,
        )
        if post is None:
            warnings.append(CampaignOpportunityWarning(
                code="missing_social_post_evidence",
                row_index=index,
                message="Skipped source row because it did not contain usable evidence.",
            ))
            continue
        posts.append(post)
    return SocialPostGenerationResult(
        posts=tuple(posts),
        warnings=tuple(warnings),
        target_mode=target_mode,
    )


def _post_from_opportunity(
    opportunity: Mapping[str, Any],
    *,
    index: int,
    max_post_chars: int,
) -> dict[str, Any] | None:
    if not opportunity:
        return None
    evidence = _first_evidence(opportunity)
    if not evidence:
        return None
    vendor = _clean(opportunity.get("vendor_name") or opportunity.get("vendor"))
    pain = _first_text(opportunity.get("pain_points"))
    hook_parts = ["Customer evidence"]
    if vendor:
        hook_parts.append(f"for {vendor}")
    if pain:
        hook_parts.append(f"flags {pain}")
    hook = " ".join(hook_parts) + "."
    body = (
        f'{hook} Source note: "{evidence}" Use this proof point to sharpen '
        "the next landing page, blog post, or sales brief."
    )
    source_id = _clean(opportunity.get("source_id") or opportunity.get("target_id"))
    return {
        "id": source_id or f"social-post-{index}",
        "channel": "linkedin",
        "text": _truncate(body, max_post_chars),
        "source_id": source_id,
        "source_type": _clean(opportunity.get("source_type")),
        "target_id": _clean(opportunity.get("target_id")),
        "company_name": _clean(opportunity.get("company_name")),
        "vendor_name": vendor,
        "pain_points": list(opportunity.get("pain_points") or ()),
    }


def _drafts_from_posts(
    posts: Sequence[Mapping[str, Any]],
    *,
    target_mode: str,
) -> tuple[SocialPostDraft, ...]:
    drafts: list[SocialPostDraft] = []
    for post in posts:
        source_id = _clean(post.get("source_id") or post.get("id"))
        target_id = _clean(post.get("target_id")) or source_id
        drafts.append(
            SocialPostDraft(
                target_id=target_id,
                target_mode=target_mode,
                channel=_clean(post.get("channel")) or "linkedin",
                text=_clean(post.get("text")),
                source_id=source_id,
                source_type=_clean(post.get("source_type")),
                company_name=_clean(post.get("company_name")),
                vendor_name=_clean(post.get("vendor_name")),
                pain_points=_pain_points_from_post(post.get("pain_points")),
                metadata=_post_metadata(post),
            )
        )
    return tuple(drafts)


def _post_metadata(post: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {"source_post": dict(post)}
    if isinstance(post.get("_brand_voice_profile"), Mapping):
        metadata["brand_voice_profile"] = dict(post["_brand_voice_profile"])
    if isinstance(post.get("_brand_voice_audit"), Mapping):
        metadata["brand_voice_audit"] = dict(post["_brand_voice_audit"])
    if isinstance(post.get("_generation_usage"), Mapping):
        metadata["generation_usage"] = dict(post["_generation_usage"])
    return metadata


def _pain_points_from_post(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(_clean(item) for item in value if _clean(item))


def _first_evidence(opportunity: Mapping[str, Any]) -> str:
    raw = opportunity.get("evidence")
    if isinstance(raw, Mapping):
        return _clean(raw.get("text"))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, Mapping):
                text = _clean(item.get("text"))
                if text:
                    return text
            else:
                text = _clean(item)
                if text:
                    return text
    return _clean(raw)


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            text = _clean(item)
            if text:
                return text
    return _clean(value)


def _rows_from_source_material(source_material: Any) -> list[Any]:
    if isinstance(source_material, str):
        text = source_material.strip()
        return [{"text": text}] if text else []
    if isinstance(source_material, Mapping):
        for key in _ROW_LIST_KEYS:
            value = source_material.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return list(value)
        return [dict(source_material)]
    if isinstance(source_material, Sequence) and not isinstance(source_material, (bytes, bytearray)):
        return list(source_material)
    return []


def _truncate(value: str, max_chars: int) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "SocialPostGenerationConfig",
    "SocialPostGenerationResult",
    "SocialPostGenerationService",
    "SocialPostDraft",
    "SocialPostRepository",
    "parse_social_post_response",
]
