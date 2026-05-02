"""Compatibility exports for campaign sequence storage and prompt assembly.

The product-owned implementation lives in
``extracted_content_pipeline.campaign_sequence_context``. This task-local module
keeps copied Atlas task imports stable while routing public helpers through the
standalone implementation with explicit ``SequenceContextLimits`` defaults.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

from ...config import settings

_STORAGE_DROP_COMPANY_CONTEXT_KEYS = frozenset({
    "selling",
    "comparison_asset",
    "reasoning_contracts",
    "qualification",
    "partner",
    "primary_blog_post",
    "supporting_blog_posts",
})
_PROMPT_ONLY_DROP_COMPANY_CONTEXT_KEYS = frozenset({
    "reasoning_anchor_examples",
    "reasoning_witness_highlights",
    "reasoning_reference_ids",
})
_SKIP = object()


def prompt_max_tokens() -> int:
    return int(settings.campaign_sequence.prompt_max_tokens)


def prompt_email_body_preview_chars() -> int:
    return int(settings.campaign_sequence.prompt_email_body_preview_chars)


def _prompt_list_limit() -> int:
    return int(settings.campaign_sequence.prompt_list_limit)


def _prompt_quote_limit() -> int:
    return int(settings.campaign_sequence.prompt_quote_limit)


def _prompt_blog_post_limit() -> int:
    return int(settings.campaign_sequence.prompt_blog_post_limit)


def _has_context_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _parse_context_blob(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _compact_scalar_list(items: Any, *, max_items: int) -> list[str]:
    compact: list[str] = []
    for item in items or []:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            compact.append(text)
        if len(compact) >= max_items:
            break
    return compact


def _compact_object_list(items: Any, *, max_items: int) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        entry = {
            key: value
            for key, value in item.items()
            if isinstance(value, (str, int, float, bool)) and _has_context_value(value)
        }
        if entry:
            compact.append(entry)
        if len(compact) >= max_items:
            break
    return compact


def _compact_named_rows(
    items: Any,
    *,
    keys: tuple[str, ...],
    max_items: int,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        entry = {key: item[key] for key in keys if _has_context_value(item.get(key))}
        if entry:
            compact.append(entry)
        if len(compact) >= max_items:
            break
    return compact


def _compact_scalar_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool)) and _has_context_value(item)
    }


def _compact_blog_posts(posts: Any) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for post in posts or []:
        if not isinstance(post, dict):
            continue
        entry = {
            key: str(post.get(key) or "").strip()
            for key in ("title", "url", "topic_type")
            if str(post.get(key) or "").strip()
        }
        if entry:
            compact.append(entry)
        if len(compact) >= _prompt_blog_post_limit():
            break
    return compact


def _compact_briefing_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) and _has_context_value(item):
            compact[key] = item
        elif isinstance(item, list):
            compact_list = _compact_scalar_list(item, max_items=_prompt_list_limit())
            if compact_list:
                compact[key] = compact_list
    return compact


def _compact_reasoning_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    compact: dict[str, Any] = {}
    for key in ("wedge", "confidence", "summary", "account_summary"):
        item = value.get(key)
        if _has_context_value(item):
            compact[key] = item

    key_signals = _compact_scalar_list(
        value.get("key_signals"),
        max_items=_prompt_quote_limit(),
    )
    if key_signals:
        compact["key_signals"] = key_signals

    why_they_stay = value.get("why_they_stay")
    if isinstance(why_they_stay, dict):
        summary = str(why_they_stay.get("summary") or "").strip()
        if summary:
            compact["why_they_stay"] = {"summary": summary}

    timing = value.get("timing")
    if isinstance(timing, dict):
        timing_compact = {
            key: timing[key]
            for key in ("best_window", "trigger_count")
            if _has_context_value(timing.get(key))
        }
        if timing_compact:
            compact["timing"] = timing_compact

    switch_triggers = _compact_object_list(
        value.get("switch_triggers"),
        max_items=_prompt_quote_limit(),
    )
    if switch_triggers:
        compact["switch_triggers"] = switch_triggers

    return compact


def _compact_signal_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    compact: dict[str, Any] = {}
    row_keys = {
        "pain_distribution": ("category", "count"),
        "competitor_distribution": ("name", "count"),
        "role_distribution": ("role", "count"),
        "pain_driving_switch": ("category", "count"),
        "incumbents_losing": ("name", "count"),
    }
    scalar_lists = {"feature_gaps", "feature_mentions"}

    for key, item in value.items():
        if key in row_keys:
            rows = _compact_named_rows(
                item,
                keys=row_keys[key],
                max_items=_prompt_list_limit(),
            )
            if rows:
                compact[key] = rows
        elif key in scalar_lists:
            rows = _compact_scalar_list(item, max_items=_prompt_list_limit())
            if rows:
                compact[key] = rows
        elif isinstance(item, dict):
            nested = _compact_scalar_mapping(item)
            if nested:
                compact[key] = nested
        elif _has_context_value(item):
            compact[key] = item

    return compact


def _compact_incumbent_reasoning(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    compact: dict[str, Any] = {}
    for vendor_name, item in value.items():
        if not isinstance(item, dict):
            continue
        entry = {
            key: item[key]
            for key in ("wedge", "summary", "why_they_stay")
            if _has_context_value(item.get(key))
        }
        switch_triggers = _compact_scalar_list(
            item.get("switch_triggers"),
            max_items=_prompt_quote_limit(),
        )
        if switch_triggers:
            entry["switch_triggers"] = switch_triggers
        if entry:
            compact[str(vendor_name)] = entry
        if len(compact) >= _prompt_list_limit():
            break
    return compact


def _compact_incumbent_archetypes(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}

    compact: dict[str, list[str]] = {}
    for group, items in value.items():
        compact_items = _compact_scalar_list(items, max_items=_prompt_quote_limit())
        if compact_items:
            compact[str(group)] = compact_items
    return compact


def _compact_category_intelligence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    compact: dict[str, Any] = {}
    stats = _compact_scalar_mapping(value.get("category_stats"))
    if stats:
        compact["category_stats"] = stats

    for key in (
        "top_pain_points",
        "feature_gaps",
        "competitive_flows",
        "brand_health",
        "safety_signals",
        "top_root_causes",
    ):
        rows = _compact_object_list(value.get(key), max_items=_prompt_list_limit())
        if rows:
            compact[key] = rows

    return compact


def _company_context_drop_keys(*, prompt_safe: bool) -> frozenset[str]:
    if prompt_safe:
        return _STORAGE_DROP_COMPANY_CONTEXT_KEYS | _PROMPT_ONLY_DROP_COMPANY_CONTEXT_KEYS
    return _STORAGE_DROP_COMPANY_CONTEXT_KEYS


def _compact_company_context_value(key: str, value: Any, *, prompt_safe: bool) -> Any:
    if key in _company_context_drop_keys(prompt_safe=prompt_safe):
        return _SKIP
    if key == "key_quotes":
        return _compact_scalar_list(value, max_items=_prompt_quote_limit())
    if key == "pain_categories":
        return _compact_named_rows(
            value,
            keys=("category", "severity"),
            max_items=_prompt_list_limit(),
        )
    if key == "competitors_considering":
        return _compact_named_rows(
            value,
            keys=("name", "reason"),
            max_items=_prompt_list_limit(),
        )
    if key in {"feature_gaps", "integration_stack"}:
        return _compact_scalar_list(value, max_items=_prompt_list_limit())
    if key == "signal_summary":
        return _compact_signal_summary(value)
    if key == "briefing_context":
        return _compact_briefing_context(value)
    if key == "reasoning_context":
        return _compact_reasoning_context(value)
    if key == "incumbent_reasoning":
        return _compact_incumbent_reasoning(value)
    if key == "incumbent_archetypes":
        return _compact_incumbent_archetypes(value)
    if key == "category_intelligence":
        return _compact_category_intelligence(value)
    return value


def _compact_company_context(
    company_context: dict[str, Any],
    *,
    prompt_safe: bool,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in company_context.items():
        compact_value = _compact_company_context_value(
            key,
            value,
            prompt_safe=prompt_safe,
        )
        if compact_value is _SKIP or not _has_context_value(compact_value):
            continue
        compact[key] = compact_value
    return compact


def _compact_selling_context(selling_context: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in selling_context.items():
        if key in {"blog_posts", "primary_blog_post"}:
            continue
        if isinstance(value, (str, int, float, bool)) and _has_context_value(value):
            compact[key] = value

    blog_posts = selling_context.get("blog_posts")
    if not blog_posts and isinstance(selling_context.get("primary_blog_post"), dict):
        blog_posts = [selling_context["primary_blog_post"]]
    compact_posts = _compact_blog_posts(blog_posts)
    if compact_posts:
        compact["blog_posts"] = compact_posts
    return compact


def compact_sequence_contexts(
    company_context: Any,
    selling_context: Any,
    *,
    prompt_safe: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed_company_context = _parse_context_blob(company_context)
    parsed_selling_context = _parse_context_blob(selling_context)

    legacy_selling = parsed_company_context.get("selling")
    if not parsed_selling_context and isinstance(legacy_selling, dict):
        parsed_selling_context = legacy_selling

    return (
        _compact_company_context(parsed_company_context, prompt_safe=prompt_safe),
        _compact_selling_context(parsed_selling_context),
    )


def prepare_sequence_prompt_contexts(seq: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return compact_sequence_contexts(
        seq.get("company_context"),
        seq.get("selling_context"),
        prompt_safe=True,
    )


def prepare_sequence_storage_contexts(
    company_context: Any,
    selling_context: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return compact_sequence_contexts(
        company_context,
        selling_context,
        prompt_safe=False,
    )


def plain_text_preview(body: str, *, limit: int | None = None) -> str:
    max_chars = int(limit or prompt_email_body_preview_chars())
    text = re.sub(r"<[^>]+>", " ", body or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


from ...campaign_sequence_context import (  # noqa: E402
    DEFAULT_LIMITS,
    SequenceContextLimits,
    compact_sequence_contexts,
    plain_text_preview,
    prepare_sequence_prompt_contexts,
    prepare_sequence_storage_contexts,
    prompt_email_body_preview_chars,
    prompt_max_tokens,
)

__all__ = [
    "DEFAULT_LIMITS",
    "SequenceContextLimits",
    "compact_sequence_contexts",
    "plain_text_preview",
    "prepare_sequence_prompt_contexts",
    "prepare_sequence_storage_contexts",
    "prompt_email_body_preview_chars",
    "prompt_max_tokens",
]
