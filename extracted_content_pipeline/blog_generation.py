"""Standalone blog-post generator orchestration for AI Content Ops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from .blog_ports import BlogBlueprintRepository, BlogPostDraft, BlogPostRepository
from .campaign_ports import LLMClient, LLMMessage, SkillStore, TenantScope
from extracted_quality_gate.blog_pack import evaluate_blog_post
from extracted_quality_gate.types import QualityInput, QualityPolicy


@dataclass(frozen=True)
class BlogPostGenerationConfig:
    """Tunable defaults for ``BlogPostGenerationService``."""

    skill_name: str = "digest/blog_post_generation"
    limit: int = 10
    max_tokens: int = 4096
    temperature: float = 0.3
    quality_policy: QualityPolicy | None = None


@dataclass(frozen=True)
class BlogPostGenerationResult:
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
        content = str(decoded.get("content") or "").strip()
        if title and content:
            return {**dict(decoded), "title": title, "content": content}
    return None


class BlogPostGenerationService:
    """Generate blog-post drafts through product-owned ports."""

    def __init__(
        self,
        *,
        blueprints: BlogBlueprintRepository,
        blog_posts: BlogPostRepository,
        llm: LLMClient,
        skills: SkillStore,
        config: BlogPostGenerationConfig | None = None,
    ):
        self._blueprints = blueprints
        self._blog_posts = blog_posts
        self._llm = llm
        self._skills = skills
        self._config = config or BlogPostGenerationConfig()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> BlogPostGenerationResult:
        prompt_template = self._skills.get_prompt(self._config.skill_name)
        if not prompt_template:
            raise ValueError(f"Blog generation skill not found: {self._config.skill_name}")

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
        for row in rows:
            blueprint = dict(row)
            blueprint_id = _blueprint_id(blueprint)
            try:
                parsed = await self._generate_one(
                    prompt_template,
                    blueprint=blueprint,
                    target_mode=target_mode,
                )
            except Exception as exc:
                skipped += 1
                errors.append({"blueprint_id": blueprint_id, "reason": str(exc)})
                continue
            if not parsed:
                skipped += 1
                errors.append({"blueprint_id": blueprint_id, "reason": "unparseable_response"})
                continue
            quality = self._quality_check(parsed, blueprint=blueprint)
            if not quality["passed"]:
                skipped += 1
                errors.append({
                    "blueprint_id": blueprint_id,
                    "reason": "quality_blocked",
                    "blockers": quality["blockers"],
                })
                continue
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
            saved_ids=saved_ids,
            errors=tuple(errors),
        )

    async def _generate_one(
        self,
        prompt_template: str,
        *,
        blueprint: Mapping[str, Any],
        target_mode: str,
    ) -> dict[str, Any] | None:
        blueprint_json = json.dumps(dict(blueprint), separators=(",", ":"), default=str)
        if "{blueprint_json}" in prompt_template:
            system_prompt = prompt_template.replace("{blueprint_json}", blueprint_json)
            user_prompt = "Generate one blog post from the blueprint above."
        else:
            system_prompt = prompt_template
            user_prompt = f"Generate one blog post from this blueprint JSON:\n{blueprint_json}"
        response = await self._llm.complete(
            [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            metadata={
                "target_mode": target_mode,
                "blueprint_id": _blueprint_id(blueprint),
                "skill_name": self._config.skill_name,
                "asset_type": "blog_post",
            },
        )
        parsed = parse_blog_post_response(response.content)
        if not parsed:
            return None
        return {
            **parsed,
            "_model": response.model,
            "_usage": dict(response.usage or {}),
        }

    def _quality_check(
        self,
        parsed: Mapping[str, Any],
        *,
        blueprint: Mapping[str, Any],
    ) -> dict[str, Any]:
        context = _quality_context(parsed, blueprint)
        quality = evaluate_blog_post(
            QualityInput(
                artifact_type="blog_post",
                artifact_id=str(parsed.get("slug") or blueprint.get("slug") or ""),
                content=str(parsed.get("content") or ""),
                context=context,
            ),
            policy=self._config.quality_policy,
        )
        return {
            "passed": quality.passed,
            "blockers": tuple(f.message for f in quality.blockers),
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
        data_context = _mapping_dict(blueprint.get("data_context"))
        data_context.update(_mapping_dict(parsed.get("data_context")))
        metadata = {
            "seo_title": parsed.get("seo_title"),
            "seo_description": parsed.get("seo_description"),
            "target_keyword": parsed.get("target_keyword"),
            "secondary_keywords": list(_string_tuple(parsed.get("secondary_keywords"))),
            "faq": _mapping_list(parsed.get("faq")),
            "generation_model": parsed.get("_model"),
            "generation_usage": parsed.get("_usage") or {},
        }
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
    }


def _blueprint_id(blueprint: Mapping[str, Any]) -> str:
    return str(
        blueprint.get("id")
        or blueprint.get("slug")
        or blueprint.get("topic")
        or blueprint.get("suggested_title")
        or ""
    ).strip()


def _slugify(value: Any) -> str:
    text = str(value or "blog-post").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
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
