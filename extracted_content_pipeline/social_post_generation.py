"""Deterministic social-post drafts from Content Ops source material."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .campaign_customer_data import CampaignOpportunityWarning
from .campaign_ports import TenantScope
from .campaign_source_adapters import source_row_to_campaign_opportunity

_ROW_LIST_KEYS = ("sources", "opportunities", "reviews", "documents", "rows", "data")


@dataclass(frozen=True)
class SocialPostGenerationConfig:
    """Config for deterministic source-material social posts."""

    limit: int = 3
    max_text_chars: int = 600
    max_post_chars: int = 420


@dataclass(frozen=True)
class SocialPostGenerationResult:
    """Generated social posts plus non-fatal source-material warnings."""

    posts: tuple[dict[str, Any], ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    target_mode: str = "vendor_retention"

    @property
    def generated(self) -> int:
        return len(self.posts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "target_mode": self.target_mode,
            "posts": [dict(post) for post in self.posts],
            "warnings": [warning.as_dict() for warning in self.warnings],
        }


class SocialPostGenerationService:
    """Build short, evidence-backed social posts from source material."""

    def __init__(self, config: SocialPostGenerationConfig | None = None) -> None:
        self.config = config or SocialPostGenerationConfig()

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        limit: int | None = None,
        max_text_chars: int | None = None,
        **kwargs: Any,
    ) -> SocialPostGenerationResult:
        del scope
        del kwargs
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
        return _generate_social_posts(
            _rows_from_source_material(source_material),
            target_mode=target_mode,
            limit=resolved_limit,
            max_text_chars=resolved_max_text_chars,
            max_post_chars=self.config.max_post_chars,
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
]
