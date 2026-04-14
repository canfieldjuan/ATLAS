"""
B2B blog post generation: picks the best data story each night from B2B
churn signals, product profiles, and affiliate partners. Builds a
deterministic blueprint, generates prose via LLM, and stores the
assembled draft in blog_posts.

Runs daily after b2b_product_profiles (default 11 PM).

Pipeline stages:
  1. Topic selection  -- score candidates from 4 B2B topic types
  2. Data gathering   -- parallel SQL from b2b_* tables + affiliate_partners
  3. Blueprint build  -- deterministic section/chart layout
  4. Content gen      -- single LLM call with blueprint as input
  5. Assembly/store   -- draft in blog_posts, affiliate link injection, ntfy

Returns _skip_synthesis.
"""

import asyncio
import copy
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ...services.scraping.sources import (
    VERIFIED_SOURCES,
    ReviewSource,
    display_name as source_display_name,
    filter_deprecated_sources,
    parse_source_allowlist,
    with_required_sources,
)
from ...reasoning.wedge_registry import Wedge, get_wedge_meta
from ._execution_progress import task_run_id as _task_execution_run_id
from ._b2b_shared import (
    read_vendor_intelligence_map as _read_vendor_intelligence_map,
    read_vendor_intelligence_records as _read_vendor_intelligence_records,
    read_vendor_scorecard_details as _read_vendor_scorecard_details,
    _align_vendor_intelligence_records_to_scorecards,
    _segment_targeting_summary,
    _timing_summary_payload,
    fetch_all_pool_layers,
)
from ._b2b_specificity import (
    evaluate_specificity_support,
    specificity_signal_terms,
    surface_specificity_context,
)

logger = logging.getLogger("atlas.autonomous.tasks.b2b_blog_post_generation")


async def _fetch_latest_evidence_vault(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compatibility seam delegating canonical reads to the shared adapter."""
    return await _read_vendor_intelligence_map(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
        vendor_names=vendor_names,
    )


async def _fetch_latest_evidence_vault_records(
    pool,
    *,
    as_of: date,
    analysis_window_days: int,
    vendor_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Read canonical vendor-intelligence records with run metadata."""
    return await _read_vendor_intelligence_records(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
        vendor_names=vendor_names,
    )


def _blog_vendor_lookup_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _blog_review_vendor_join(
    review_alias: str = "r",
    mention_alias: str = "vm",
) -> str:
    """Join canonical review-to-vendor mentions for vendor-scoped reads."""
    return (
        f"JOIN b2b_review_vendor_mentions {mention_alias} "
        f"ON {mention_alias}.review_id = {review_alias}.id"
    )


def _blog_primary_vendor_join(
    review_alias: str = "r",
    mention_alias: str = "primary_vm",
) -> str:
    """Project one canonical vendor label per review for category-scoped reads."""
    return (
        "LEFT JOIN LATERAL ("
        f" SELECT vm.vendor_name"
        " FROM b2b_review_vendor_mentions vm"
        f" WHERE vm.review_id = {review_alias}.id"
        " ORDER BY vm.is_primary DESC, vm.id ASC"
        f" LIMIT 1) AS {mention_alias} ON TRUE"
    )


async def _fetch_vendor_scorecard_details(
    pool,
    *,
    vendor_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Read detailed scorecard rows keyed by requested vendor label."""
    requested_vendors = [str(name or "").strip() for name in (vendor_names or []) if str(name or "").strip()]
    if not requested_vendors:
        return {}
    rows = await _read_vendor_scorecard_details(
        pool,
        vendor_names=requested_vendors,
    )
    requested_by_key = {
        _blog_vendor_lookup_key(name): name
        for name in requested_vendors
    }
    detail_lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        vendor_name = str(row.get("vendor_name") or "").strip()
        if not vendor_name:
            continue
        detail_lookup[vendor_name] = row
        requested_name = requested_by_key.get(_blog_vendor_lookup_key(vendor_name))
        if requested_name:
            detail_lookup[requested_name] = row
    return detail_lookup

_BLOG_GENERATION_CACHE_STAGE = "b2b_blog_post_generation.content"
_BLOG_GENERATION_TRACE_SPAN = "task.b2b_blog_post_generation"
_BLOG_BATCH_SELECTION_MULTIPLIER = 4
_BLOG_BATCH_SELECTION_MIN_ATTEMPTS = 8

_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s*(.+?)\s*$")
_ANSWER_PREFIX_RE = re.compile(r"(?im)^(\s*)answer:\s*")
_CRITICAL_BLOG_WARNINGS = {
    "review_period_not_explicitly_mentioned",
    "methodology_disclaimer_missing_self_selected",
    "unsupported_data_claim",
    "chart_scope_ambiguity",
    "migration_direction_drift",
}
_DATA_CLAIM_MARKERS = (
    "most common",
    "top migration",
    "top source",
    "primary source",
    "primary driver",
    "leading source",
    "data shows",
    "reviews mention",
    "switched from",
    "stories analyzed",
)
_DATA_CLAIM_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(m) for m in _DATA_CLAIM_MARKERS)
    + r"|\d+\s*%|\d+\s+reviews?\b|\d+\s+stories?\b)",
    re.IGNORECASE,
)
_VENDORISH_NAME_PATTERN = re.compile(
    r"\b("
    r"(?:[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+|[A-Z]{2,}|[a-z]+[A-Z][A-Za-z0-9]*)"
    r"(?:\s+(?:[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+|[A-Z]{2,}|[a-z]+[A-Z][A-Za-z0-9]*)){0,3}"
    r")\b"
)
_VENDORISH_SKIP_WORDS = frozenset(
    " ".join(re.findall(r"[a-z0-9]+", w.lower())) for w in (
        # Common English words that match the capitalized pattern
        "The", "This", "That", "When", "What", "Where", "Which", "While",
        "Most", "Top", "Data", "Teams", "Users", "Some", "Each", "Both",
        "Many", "Other", "These", "Those", "After", "Before", "Between",
        "About", "Since", "Until", "During", "However", "Although",
        # Document structure words
        "Introduction", "Conclusion", "Overview", "Analysis", "Guide",
        "Report", "Summary", "Review", "Reviews", "Reviewers", "Reviewer",
        "Section", "Chart", "Table", "Source", "Sources",
        "Note", "Key", "Figure", "Methodology", "Decision",
        "Rating", "Ratings", "Platform", "Platforms", "Price", "Pricing",
        "Feature", "Features", "Support", "Integration", "Performance",
        # Time/date words
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        # Tech/domain terms that aren't vendors
        "API", "SEO", "SaaS", "CRM", "ERP", "DNS", "SSL", "CSS",
        "HTML", "REST", "SDK", "ROI", "KPI", "B2B", "SMB",
    )
)
_BLOG_SKIP_SENTENCE_PREFIXES = (
    "#",
    "_methodology note:",
    "methodology note:",
    "analysis methodology:",
    "*analysis methodology:",
)
_BLOG_SKIP_SENTENCE_CONTAINS = (
    "this analysis draws on",
    "this post draws on",
    "analysis based on self-selected reviewer feedback",
    "analysis reflects self-selected feedback from",
)
_SOURCE_NAME_ALIASES = {
    "gartner peer insights",
    "public b2b software review platforms",
    "public software reviews",
}
_SOURCEISH_SKIP_WORDS = frozenset(
    {
        *(
            " ".join(re.findall(r"[a-z0-9]+", source_display_name(source).lower()))
            for source in ReviewSource
        ),
        *(
            " ".join(re.findall(r"[a-z0-9]+", alias.lower()))
            for alias in _SOURCE_NAME_ALIASES
        ),
    }
)


def _task_run_id(task: ScheduledTask | Any) -> str | None:
    """Return a stable execution-aware run id for scheduled blog jobs."""
    return _task_execution_run_id(task)


def _blog_post_max_per_run(task: ScheduledTask | Any, cfg: Any) -> int:
    """Resolve max posts per run from task metadata before config defaults."""
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("blog_post_max_per_run", "max_posts_per_run", "max_per_run"):
            value = metadata.get(key)
            try:
                if value is not None:
                    return max(1, int(value))
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid %s=%r on blog task metadata", key, value)
    return max(1, int(cfg.blog_post_max_per_run))
_WEDGEISH_SKIP_WORDS = frozenset(
    " ".join(re.findall(r"[a-z0-9]+", phrase.lower()))
    for phrase in {
        *(wedge.value.replace("_", " ") for wedge in Wedge),
        *(get_wedge_meta(wedge).label for wedge in Wedge),
    }
)
_BLOG_NON_VENDOR_SKIP_WORDS = frozenset(
    set(_VENDORISH_SKIP_WORDS) | set(_SOURCEISH_SKIP_WORDS) | set(_WEDGEISH_SKIP_WORDS)
)


def _build_grounded_vendor_set(blueprint: "PostBlueprint") -> set[str]:
    """Build the set of vendor/product names supported by chart + context data."""
    names: set[str] = set()
    ctx = blueprint.data_context if isinstance(blueprint.data_context, dict) else {}
    # Topic vendors
    for key in ("vendor", "vendor_a", "vendor_b", "from_vendor", "to_vendor"):
        v = str(ctx.get(key) or "").strip()
        nv = _normalized_vendor_text(v)
        if nv and len(nv) > 1:
            names.add(nv)
    # Chart labels and series values
    for chart in blueprint.charts:
        for row in chart.data:
            if not isinstance(row, dict):
                continue
            for val in row.values():
                if isinstance(val, str):
                    nv = _normalized_vendor_text(val)
                    if nv and len(nv) > 1:
                        names.add(nv)
    # Explicit vendor lists in data_context
    for key in ("displacement_targets", "competitors", "commonly_switched_from",
                "top_displacement_targets", "vendor_profiles"):
        items = ctx.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for vk in ("vendor", "name", "competitor", "vendor_name"):
                        v = str(item.get(vk) or "").strip()
                        nv = _normalized_vendor_text(v)
                        if nv and len(nv) > 1:
                            names.add(nv)
                elif isinstance(item, str):
                    nv = _normalized_vendor_text(item)
                    if nv and len(nv) > 1:
                        names.add(nv)
    return names


def _find_unsupported_data_claims(
    body: str,
    grounded: set[str],
    known_vendors: list[str] | None = None,
) -> list[str]:
    """Return sentences with data-claim markers that reference ungrounded vendors.

    Uses two detection strategies:
    1. Known-vendor lookup: check if any known vendor (from DB) appears in a
       claim sentence but is NOT in the grounded set.
    2. Regex fallback: extract capitalized name patterns for vendors not in
       the known universe (catches novel names the LLM invented).
    """
    sentences = re.split(r'(?<=[.!?])\s+|\n+', body)
    flagged: list[str] = []

    # Build case-insensitive lookup for known vendors not in grounded set
    ungrounded_known: list[tuple[str, re.Pattern[str]]] = []
    for v in (known_vendors or []):
        nv = _normalized_vendor_text(v)
        if nv and nv not in grounded and len(v) > 2:
            # Match the vendor name as a whole word, case-insensitive
            pattern = re.compile(r"\b" + re.escape(v) + r"\b", re.IGNORECASE)
            ungrounded_known.append((v, pattern))

    for sentence in sentences:
        if not _DATA_CLAIM_PATTERN.search(sentence):
            continue
        trimmed_sentence = sentence.strip()
        trimmed_lower = trimmed_sentence.lower()
        if any(trimmed_lower.startswith(prefix) for prefix in _BLOG_SKIP_SENTENCE_PREFIXES):
            continue
        if any(fragment in trimmed_lower for fragment in _BLOG_SKIP_SENTENCE_CONTAINS):
            continue
        # Strategy 1: known vendor lookup
        found_known = False
        for vendor_name, pattern in ungrounded_known:
            if pattern.search(sentence):
                flagged.append(f"{vendor_name}: {sentence.strip()[:120]}")
                found_known = True
                break
        if found_known:
            continue
        # Strategy 2: regex fallback for multi-word names not in known universe.
        # Single capitalized words are too noisy (Among, Outcome, Causal, etc.)
        # -- those are only caught via the known-vendor DB path above.
        candidates = _VENDORISH_NAME_PATTERN.findall(sentence)
        for name in candidates:
            if " " not in name.strip():
                continue  # single words handled by known-vendor lookup only
            normalized_name = _normalized_vendor_text(name)
            if normalized_name not in grounded and len(normalized_name) > 2:
                if normalized_name in _BLOG_NON_VENDOR_SKIP_WORDS:
                    continue
                flagged.append(f"{name}: {sentence.strip()[:120]}")
                break
    return flagged


def _unsupported_claim_entries(report: dict[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for warning in report.get("warnings", []) or []:
        warning_text = str(warning)
        if not warning_text.startswith("unsupported_data_claim:"):
            continue
        claim_text = warning_text[len("unsupported_data_claim:"):].strip()
        vendor, _, snippet = claim_text.partition(":")
        entries.append((vendor.strip(), snippet.strip()))
    return entries


def _line_contains_unsupported_claim(line: str, vendor: str, snippet: str) -> bool:
    if not line.strip():
        return False
    if snippet and snippet in line:
        return True
    if vendor and re.search(r"\b" + re.escape(vendor) + r"\b", line, re.IGNORECASE):
        return bool(_DATA_CLAIM_PATTERN.search(line))
    return False


def _remove_unsupported_claim_lines(body: str, report: dict[str, Any]) -> tuple[str, int]:
    lines = body.splitlines()
    if not lines:
        return body, 0
    removed = 0
    for vendor, snippet in _unsupported_claim_entries(report):
        if not vendor and not snippet:
            continue
        next_lines: list[str] = []
        claim_removed = False
        for line in lines:
            if _line_contains_unsupported_claim(line, vendor, snippet):
                removed += 1
                claim_removed = True
                continue
            next_lines.append(line)
        lines = next_lines
        if claim_removed:
            continue
        text = "\n".join(lines)
        sentences = re.split(r"(?<=[.!?])(\s+)", text)
        kept: list[str] = []
        for segment in sentences:
            if _line_contains_unsupported_claim(segment, vendor, snippet):
                removed += 1
                continue
            kept.append(segment)
        lines = "".join(kept).splitlines()
    return "\n".join(lines), removed


def _strip_nonexistent_internal_links(
    body: str,
    *,
    valid_slugs: set[str],
    current_slug: str,
) -> tuple[str, int]:
    updated = str(body or "")
    if not updated:
        return updated, 0

    removed = 0

    markdown_pattern = re.compile(
        r"\[([^\]]+)\]\(((?:https?://[^/)\s]+)?/blog/([a-z0-9\-]+))\)"
    )

    def _replace_markdown(match: re.Match[str]) -> str:
        nonlocal removed
        slug = str(match.group(3) or "")
        if slug in valid_slugs or slug == current_slug:
            return match.group(0)
        removed += 1
        return str(match.group(1) or "")

    updated = markdown_pattern.sub(_replace_markdown, updated)

    html_pattern = re.compile(
        r"<a\s+[^>]*href=[\"']((?:https?://[^/\"']+)?/blog/([a-z0-9\-]+))[\"'][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )

    def _replace_html(match: re.Match[str]) -> str:
        nonlocal removed
        slug = str(match.group(2) or "")
        if slug in valid_slugs or slug == current_slug:
            return match.group(0)
        removed += 1
        return str(match.group(3) or "")

    updated = html_pattern.sub(_replace_html, updated)

    path_pattern = re.compile(r"(?<![A-Za-z0-9_-])/blog/([a-z0-9\-]+)")

    def _replace_path(match: re.Match[str]) -> str:
        nonlocal removed
        slug = str(match.group(1) or "")
        if slug in valid_slugs or slug == current_slug:
            return match.group(0)
        removed += 1
        return slug.replace("-", " ")

    updated = path_pattern.sub(_replace_path, updated)
    return updated, removed


def _blog_source_allowlist() -> list[str]:
    """Return the configured source allowlist as a list for SQL ANY() binding."""
    return filter_deprecated_sources(
        with_required_sources(parse_source_allowlist(settings.b2b_churn.blog_source_allowlist)),
        settings.b2b_churn.deprecated_review_sources,
    )


# -- dataclasses (same structure as consumer blog pipeline) --------

@dataclass
class ChartSpec:
    chart_id: str
    chart_type: str  # bar | horizontal_bar | radar | line
    title: str
    data: list[dict[str, Any]]
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionSpec:
    id: str
    heading: str
    goal: str
    key_stats: dict[str, Any] = field(default_factory=dict)
    chart_ids: list[str] = field(default_factory=list)
    data_summary: str = ""


@dataclass
class PostBlueprint:
    topic_type: str
    slug: str
    suggested_title: str
    tags: list[str]
    data_context: dict[str, Any]
    sections: list[SectionSpec]
    charts: list[ChartSpec]
    quotable_phrases: list[dict[str, Any]] = field(default_factory=list)
    cta: dict[str, Any] | None = None


def _reasoning_scalar(value: Any) -> Any:
    """Unwrap a traced reasoning value when present."""
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _reasoning_int(value: Any) -> int | None:
    """Unwrap a traced reasoning value into an integer when possible."""
    raw = _reasoning_scalar(value)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None


def _blog_account_reasoning_stats(
    account_reasoning: dict[str, Any] | None,
    account_preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic account-pressure stats for blog blueprints."""
    if not isinstance(account_reasoning, dict):
        account_reasoning = {}
    if not isinstance(account_preview, dict):
        account_preview = {}
    stats: dict[str, Any] = {}
    summary = str(account_reasoning.get("market_summary") or "").strip()
    if not summary:
        summary = str(account_preview.get("account_pressure_summary") or "").strip()
    if summary:
        stats["account_pressure_summary"] = summary
    disclaimer = str(
        account_reasoning.get("account_actionability_note")
        or account_preview.get("account_actionability_note")
        or account_preview.get("disclaimer")
        or ""
    ).strip()
    if disclaimer:
        stats["account_pressure_disclaimer"] = disclaimer
    actionability_tier = str(
        account_reasoning.get("account_actionability_tier")
        or account_preview.get("account_actionability_tier")
        or ""
    ).strip()
    if actionability_tier:
        stats["account_actionability_tier"] = actionability_tier
    for key in ("total_accounts", "high_intent_count", "active_eval_count"):
        value = _reasoning_int(account_reasoning.get(key))
        if value is None:
            preview_metrics = account_preview.get("account_pressure_metrics") or {}
            if isinstance(preview_metrics, dict):
                value = _reasoning_int(preview_metrics.get(key))
        if value is not None:
            stats[key] = value
    names: list[str] = []
    for item in account_reasoning.get("top_accounts") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    if not names:
        for item in account_preview.get("priority_account_names") or []:
            name = str(item or "").strip()
            if name and name not in names:
                names.append(name)
    if names:
        stats["priority_accounts"] = names[:3]
    return stats


def _blog_timing_reasoning_stats(timing_intelligence: dict[str, Any] | None) -> dict[str, Any]:
    """Build deterministic timing stats for blog blueprints."""
    if not isinstance(timing_intelligence, dict):
        return {}
    summary, metrics, triggers = _timing_summary_payload(timing_intelligence)
    stats: dict[str, Any] = {}
    if summary:
        stats["timing_summary"] = summary
    if metrics:
        stats.update(metrics)
    if triggers:
        stats["priority_timing_triggers"] = triggers[:3]
    sentiment_direction = str(timing_intelligence.get("sentiment_direction") or "").strip()
    if sentiment_direction:
        stats["sentiment_direction"] = sentiment_direction
    best_window = str(timing_intelligence.get("best_timing_window") or "").strip()
    if best_window:
        stats["best_timing_window"] = best_window
    return stats


def _blog_segment_reasoning_stats(
    segment_playbook: dict[str, Any] | None,
    timing_intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic segment-targeting stats for blog blueprints."""
    if not isinstance(segment_playbook, dict):
        return {}
    stats: dict[str, Any] = {}
    summary = _segment_targeting_summary(segment_playbook, timing_intelligence)
    if summary:
        stats["segment_targeting_summary"] = summary
    segments: list[dict[str, Any]] = []
    for item in segment_playbook.get("priority_segments") or []:
        if not isinstance(item, dict):
            continue
        segment = str(item.get("segment") or "").strip()
        if not segment:
            continue
        row: dict[str, Any] = {"segment": segment}
        reach = _reasoning_scalar(item.get("estimated_reach"))
        if reach not in (None, ""):
            row["estimated_reach"] = reach
        angle = str(item.get("best_opening_angle") or "").strip()
        if angle:
            row["best_opening_angle"] = angle
        segments.append(row)
    if segments:
        stats["priority_segments"] = segments[:3]
    return stats


def _blog_migration_proof_stats(displacement_reasoning: dict[str, Any] | None) -> dict[str, Any]:
    """Build deterministic migration-proof stats for blog blueprints."""
    if not isinstance(displacement_reasoning, dict):
        return {}
    proof = displacement_reasoning.get("migration_proof")
    if not isinstance(proof, dict):
        return {}
    stats: dict[str, Any] = {}
    if "switching_is_real" in proof:
        stats["switching_is_real"] = bool(proof.get("switching_is_real"))
    evidence_type = str(proof.get("evidence_type") or "").strip()
    if evidence_type:
        stats["evidence_type"] = evidence_type
    for key in (
        "switch_volume",
        "active_evaluation_volume",
        "displacement_mention_volume",
        "top_destination",
        "primary_switch_driver",
    ):
        value = _reasoning_scalar(proof.get(key))
        if value not in (None, "", []):
            stats[key] = value
    examples: list[str] = []
    for item in proof.get("named_examples") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("company") or "").strip()
        if name and name not in examples:
            examples.append(name)
    if examples:
        stats["named_examples"] = examples[:3]
    return stats


def _blog_category_reasoning_stats(category_reasoning: dict[str, Any] | None) -> dict[str, Any]:
    """Build deterministic category-reasoning stats for blog blueprints."""
    if not isinstance(category_reasoning, dict):
        return {}
    stats: dict[str, Any] = {}
    for key in ("market_regime", "narrative", "winner", "loser"):
        value = str(category_reasoning.get(key) or "").strip()
        if value:
            stats[key] = value
    return stats


def _build_blog_claim_plan(
    contracts: dict[str, Any] | None,
    specificity_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(contracts, dict):
        return {}

    plan: dict[str, Any] = {}
    vendor_core = contracts.get("vendor_core_reasoning")
    if isinstance(vendor_core, dict):
        causal = vendor_core.get("causal_narrative")
        if isinstance(causal, dict):
            summary = str(causal.get("summary") or "").strip()
            trigger = str(causal.get("trigger") or "").strip()
            why_now = str(causal.get("why_now") or "").strip()
            if summary:
                plan["primary_thesis"] = summary
            elif trigger or why_now:
                parts = [part for part in (trigger, why_now) if part]
                if parts:
                    plan["primary_thesis"] = ". ".join(parts)

        timing = vendor_core.get("timing_intelligence")
        if isinstance(timing, dict):
            best_window = str(timing.get("best_timing_window") or "").strip()
            if best_window:
                plan["timing_hook"] = best_window
            else:
                for trigger_row in timing.get("immediate_triggers") or []:
                    if not isinstance(trigger_row, dict):
                        continue
                    trigger = str(
                        trigger_row.get("trigger")
                        or trigger_row.get("description")
                        or ""
                    ).strip()
                    if trigger:
                        plan["timing_hook"] = trigger
                        break

        retention = vendor_core.get("why_they_stay")
        if isinstance(retention, dict):
            summary = str(retention.get("summary") or "").strip()
            if summary:
                plan["counterevidence"] = summary

    category_reasoning = contracts.get("category_reasoning")
    if isinstance(category_reasoning, dict):
        market_regime = str(category_reasoning.get("market_regime") or "").strip()
        if market_regime:
            plan["market_regime"] = market_regime

    specificity = specificity_context if isinstance(specificity_context, dict) else {}
    anchors = specificity.get("anchor_examples")
    highlights = specificity.get("witness_highlights")
    proof_anchors: list[dict[str, Any]] = []
    preferred_labels = (
        "outlier_or_named_account",
        "common_pattern",
        "counterevidence",
    )
    if isinstance(anchors, dict):
        for label in preferred_labels:
            rows = anchors.get(label) or []
            if not isinstance(rows, list) or not rows:
                continue
            row = rows[0]
            if not isinstance(row, dict):
                continue
            anchor = {"anchor_type": label}
            for key in (
                "excerpt_text",
                "time_anchor",
                "competitor",
                "pain_category",
                "signal_type",
                "replacement_mode",
                "operating_model_shift",
                "productivity_delta_claim",
                "org_pressure_type",
                "reviewer_title",
                "company_size",
                "industry",
                "numeric_literals",
            ):
                value = row.get(key)
                if value not in (None, "", [], {}):
                    anchor[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
            if len(anchor) > 1:
                proof_anchors.append(anchor)
    if not proof_anchors and isinstance(highlights, list):
        for row in highlights[:2]:
            if not isinstance(row, dict):
                continue
            anchor: dict[str, Any] = {"anchor_type": "witness_highlight"}
            for key in (
                "excerpt_text",
                "time_anchor",
                "competitor",
                "pain_category",
                "signal_type",
                "replacement_mode",
                "operating_model_shift",
                "productivity_delta_claim",
                "org_pressure_type",
                "numeric_literals",
            ):
                value = row.get(key)
                if value not in (None, "", [], {}):
                    anchor[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
            if len(anchor) > 1:
                proof_anchors.append(anchor)
    if proof_anchors:
        plan["proof_anchors"] = proof_anchors[:3]
    return plan


def _blog_scope_summary(scope_manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scope_manifest, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "selection_strategy",
        "reviews_considered_total",
        "reviews_in_scope",
        "witnesses_in_scope",
        "witness_mix",
    ):
        value = scope_manifest.get(key)
        if value not in (None, "", [], {}):
            summary[key] = copy.deepcopy(value)
    return summary


def _blog_atom_context(consumer_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(consumer_context, dict):
        return {}
    context: dict[str, Any] = {}
    theses = [
        {
            "wedge": str(item.get("wedge") or "").strip(),
            "summary": str(item.get("summary") or "").strip(),
            "why_now": str(item.get("why_now") or "").strip(),
            "confidence": str(item.get("confidence") or "").strip(),
        }
        for item in (consumer_context.get("theses") or [])[:3]
        if isinstance(item, dict)
    ]
    theses = [item for item in theses if item["summary"] or item["why_now"]]
    if theses:
        context["top_theses"] = theses
    timing_windows = [
        {
            "window_type": str(item.get("window_type") or "").strip(),
            "anchor": str(item.get("start_or_anchor") or "").strip(),
            "urgency": str(item.get("urgency") or "").strip(),
            "recommended_action": str(item.get("recommended_action") or "").strip(),
        }
        for item in (consumer_context.get("timing_windows") or [])[:3]
        if isinstance(item, dict)
    ]
    timing_windows = [item for item in timing_windows if item["anchor"]]
    if timing_windows:
        context["timing_windows"] = timing_windows
    proof_points = [
        {
            "label": str(item.get("label") or "").strip(),
            "value": _reasoning_scalar(item.get("value")),
            "interpretation": str(item.get("interpretation") or "").strip(),
            "confidence": str(item.get("confidence") or "").strip(),
        }
        for item in (consumer_context.get("proof_points") or [])[:3]
        if isinstance(item, dict)
    ]
    proof_points = [item for item in proof_points if item["label"]]
    if proof_points:
        context["proof_points"] = proof_points
    account_signals = [
        {
            "company": str(item.get("company") or "").strip(),
            "buying_stage": str(item.get("buying_stage") or "").strip(),
            "competitor_context": str(item.get("competitor_context") or "").strip(),
            "primary_pain": str(item.get("primary_pain") or "").strip(),
            "quote": str(item.get("quote") or "").strip(),
        }
        for item in (consumer_context.get("account_signals") or [])[:3]
        if isinstance(item, dict)
    ]
    account_signals = [
        item for item in account_signals
        if item["company"] or item["primary_pain"] or item["quote"]
    ]
    if account_signals:
        context["account_signals"] = account_signals
    coverage_limits = [
        str(item.get("label") or "").strip()
        for item in (consumer_context.get("coverage_limits") or [])[:4]
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    ]
    if coverage_limits:
        context["coverage_limits"] = coverage_limits
    counterevidence = [
        str(item.get("statement") or "").strip()
        for item in (consumer_context.get("counterevidence") or [])[:2]
        if isinstance(item, dict) and str(item.get("statement") or "").strip()
    ]
    if counterevidence:
        context["counterevidence"] = counterevidence
    return context


def _blog_delta_summary(reasoning_delta: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(reasoning_delta, dict):
        return {}
    summary: dict[str, Any] = {"changed": bool(reasoning_delta.get("changed"))}
    for key in (
        "wedge_changed",
        "confidence_changed",
        "top_destination_changed",
    ):
        if key in reasoning_delta:
            summary[key] = bool(reasoning_delta.get(key))
    for key in (
        "theses_added",
        "new_timing_windows",
        "new_account_signals",
        "coverage_worsened",
    ):
        value = reasoning_delta.get(key)
        if isinstance(value, list) and value:
            summary[key] = copy.deepcopy(value[:3])
    return summary


def _inject_blog_reasoning_specificity(
    data: dict[str, Any],
    *,
    suffix: str,
    view: Any,
) -> None:
    data_context = data.setdefault("data_context", {})
    if not isinstance(data_context, dict):
        data_context = {}
        data["data_context"] = data_context

    consumer_context = view.filtered_consumer_context("blog_reranker")
    specificity = surface_specificity_context(consumer_context, surface="blog")
    if specificity.get("anchor_examples"):
        data[f"reasoning_anchor_examples{suffix}"] = specificity["anchor_examples"]
        data_context[f"reasoning_anchor_examples{suffix}"] = copy.deepcopy(
            specificity["anchor_examples"]
        )
    if specificity.get("witness_highlights"):
        data[f"reasoning_witness_highlights{suffix}"] = specificity["witness_highlights"]
        data_context[f"reasoning_witness_highlights{suffix}"] = copy.deepcopy(
            specificity["witness_highlights"]
        )
    if specificity.get("reference_ids"):
        data[f"reasoning_reference_ids{suffix}"] = specificity["reference_ids"]
        data_context[f"reasoning_reference_ids{suffix}"] = copy.deepcopy(
            specificity["reference_ids"]
        )
    disclaimers = consumer_context.get("reasoning_section_disclaimers")
    if isinstance(disclaimers, dict) and disclaimers:
        data[f"reasoning_section_disclaimers{suffix}"] = disclaimers
        data_context[f"reasoning_section_disclaimers{suffix}"] = copy.deepcopy(
            disclaimers
        )

    claim_plan = _build_blog_claim_plan(
        consumer_context.get("reasoning_contracts") or view.materialized_contracts(),
        specificity,
    )
    if claim_plan:
        data[f"blog_claim_plan{suffix}"] = claim_plan
        data_context[f"blog_claim_plan{suffix}"] = copy.deepcopy(claim_plan)
    scope_summary = _blog_scope_summary(consumer_context.get("scope_manifest"))
    if scope_summary:
        data[f"reasoning_scope_summary{suffix}"] = scope_summary
        data_context[f"reasoning_scope_summary{suffix}"] = copy.deepcopy(scope_summary)
    atom_context = _blog_atom_context(consumer_context)
    if atom_context:
        data[f"reasoning_atom_context{suffix}"] = atom_context
        data_context[f"reasoning_atom_context{suffix}"] = copy.deepcopy(atom_context)
    delta_summary = _blog_delta_summary(consumer_context.get("reasoning_delta"))
    if delta_summary:
        data[f"reasoning_delta_summary{suffix}"] = delta_summary
        data_context[f"reasoning_delta_summary{suffix}"] = copy.deepcopy(delta_summary)


# -- Cross-vendor synthesis resolution helpers ----------------------

def _resolve_blog_battle_summary(
    vendor_a: str,
    vendor_b: str,
    xv_lookup: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a pairwise battle from cross-vendor synthesis or pool fallback.

    Returns a blog-compatible summary dict with ``conclusion``, ``winner``,
    ``loser``, ``confidence``, ``durability_assessment``, ``key_insights``.
    """
    battles = xv_lookup.get("battles") or {}
    needle = tuple(sorted([
        vendor_a.strip().lower(), vendor_b.strip().lower(),
    ]))
    # Try exact key first, then case-insensitive scan
    entry = battles.get(needle)
    if entry is None:
        entry = battles.get(tuple(sorted([vendor_a.strip(), vendor_b.strip()])))
    if entry is None:
        for k, v in battles.items():
            if tuple(sorted(s.lower() for s in k)) == needle:
                entry = v
                break
    if entry is None:
        return fallback or {}
    conclusion = entry.get("conclusion") or {}
    if not isinstance(conclusion, dict):
        return fallback or {}
    result = {
        "conclusion": conclusion.get("conclusion") or "",
        "winner": conclusion.get("winner") or "",
        "loser": conclusion.get("loser") or "",
        "confidence": conclusion.get("confidence"),
        "durability_assessment": conclusion.get("durability_assessment") or "",
        "key_insights": conclusion.get("key_insights") or [],
        "source": "synthesis",
    }
    if not result["conclusion"]:
        return fallback or {}
    return result


def _resolve_blog_council_summary(
    category: str,
    xv_lookup: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a category council from cross-vendor synthesis or pool fallback.

    Returns a blog-compatible summary dict with ``conclusion``,
    ``market_regime``, ``winner``, ``loser``, ``confidence``.
    """
    councils = xv_lookup.get("councils") or {}
    cat_norm = (category or "").strip().lower()
    entry = councils.get(category)
    if entry is None:
        # Try case-insensitive match
        for k, v in councils.items():
            if (k or "").strip().lower() == cat_norm:
                entry = v
                break
    if entry is None:
        return fallback or {}
    conclusion = entry.get("conclusion") or {}
    if not isinstance(conclusion, dict):
        return fallback or {}
    result = {
        "conclusion": conclusion.get("conclusion") or "",
        "market_regime": conclusion.get("market_regime") or "",
        "winner": conclusion.get("winner"),
        "loser": conclusion.get("loser"),
        "confidence": conclusion.get("confidence"),
        "durability_assessment": conclusion.get("durability_assessment") or "",
        "key_insights": conclusion.get("key_insights") or [],
        "source": "synthesis",
    }
    if not result["conclusion"] and not result["market_regime"]:
        return fallback or {}
    return result


# -- CTA configuration --------------------------------------------

_CTA_CONFIG: dict[str, dict[str, str]] = {
    "vendor_showdown":       {"report_type": "vendor_comparison", "button_text": "Download the full benchmark report"},
    "vendor_deep_dive":      {"report_type": "vendor_deep_dive",  "button_text": "Get the exclusive deep dive report"},
    "churn_report":          {"report_type": "weekly_churn_feed", "button_text": "See which vendors are most at risk"},
    "vendor_alternative":    {"report_type": "battle_card",       "button_text": "Compare alternatives side by side"},
    "market_landscape":      {"report_type": "category_overview", "button_text": "Get the full industry report"},
    "pricing_reality_check": {"report_type": "vendor_scorecard",  "button_text": "See the full pricing analysis"},
    "switching_story":       {"report_type": "battle_card",       "button_text": "Get the switching playbook"},
    "migration_guide":       {"report_type": "vendor_comparison", "button_text": "Download the migration comparison"},
    "pain_point_roundup":    {"report_type": "category_overview", "button_text": "See the full category breakdown"},
    "best_fit_guide":        {"report_type": "category_overview", "button_text": "Check your vendor's risk score"},
}


def _blog_length_policy(topic_type: str) -> dict[str, int]:
    default_min_by_topic = {
        "vendor_showdown": 2000,
        "market_landscape": 1900,
        "best_fit_guide": 1900,
        "vendor_deep_dive": 1800,
        "vendor_alternative": 1700,
        "churn_report": 1700,
        "pain_point_roundup": 1700,
        "pricing_reality_check": 1600,
        "migration_guide": 1500,
        "switching_story": 1500,
    }
    default_target_by_topic = {
        "vendor_showdown": 2600,
        "market_landscape": 2600,
        "best_fit_guide": 2500,
        "vendor_deep_dive": 2400,
        "vendor_alternative": 2300,
        "churn_report": 2300,
        "pain_point_roundup": 2300,
        "pricing_reality_check": 2200,
        "migration_guide": 2100,
        "switching_story": 2100,
    }

    def _coerce_int_or_default(value: Any, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value != value:
                return default
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                return int(text)
            except ValueError:
                try:
                    return int(float(text))
                except ValueError:
                    return default
        return default

    cfg = settings.b2b_churn
    normalized_topic = str(topic_type or "").strip().lower()
    default_min = max(
        1,
        _coerce_int_or_default(getattr(cfg, "blog_post_min_words_default", 1800), 1800),
    )
    default_target = max(
        default_min,
        _coerce_int_or_default(getattr(cfg, "blog_post_target_words_default", default_min), default_min),
    )
    raw_min_by_topic = getattr(cfg, "blog_post_min_words_by_topic", default_min_by_topic) or default_min_by_topic
    raw_target_by_topic = getattr(cfg, "blog_post_target_words_by_topic", default_target_by_topic) or default_target_by_topic
    if not isinstance(raw_min_by_topic, dict):
        raw_min_by_topic = default_min_by_topic
    if not isinstance(raw_target_by_topic, dict):
        raw_target_by_topic = default_target_by_topic
    min_by_topic = {
        str(key or "").strip().lower(): _coerce_int_or_default(value, default_min)
        for key, value in raw_min_by_topic.items()
        if str(key or "").strip()
    }
    target_by_topic = {
        str(key or "").strip().lower(): _coerce_int_or_default(value, default_target)
        for key, value in raw_target_by_topic.items()
        if str(key or "").strip()
    }
    min_words = max(
        1,
        _coerce_int_or_default(min_by_topic.get(normalized_topic, default_min), default_min),
    )
    target_words = max(
        min_words,
        _coerce_int_or_default(target_by_topic.get(normalized_topic, default_target), default_target),
    )
    return {
        "min_words": min_words,
        "target_words": target_words,
    }


def _blog_topic_threshold(raw_map: Any, topic_type: str, *, default: int = 0) -> int:
    if not isinstance(raw_map, dict):
        return max(0, int(default))
    normalized_topic = str(topic_type or "").strip().lower()
    for key, value in raw_map.items():
        if str(key or "").strip().lower() != normalized_topic:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return max(0, int(default))
    return max(0, int(default))


def _blog_min_vendor_profiles(topic_type: str) -> int:
    return _blog_topic_threshold(
        getattr(settings.b2b_churn, "blog_min_vendor_profiles_by_topic", {}) or {},
        topic_type,
        default=2,
    )


def _blog_section_word_budget(blueprint: "PostBlueprint") -> dict[str, int]:
    length_policy = _blog_length_policy(blueprint.topic_type)
    section_count = max(1, len(blueprint.sections or []))
    min_words = max(1, int(length_policy["min_words"]))
    target_words = max(min_words, int(length_policy["target_words"]))
    return {
        "section_count": section_count,
        "min_words_per_section": (min_words + section_count - 1) // section_count,
        "target_words_per_section": (target_words + section_count - 1) // section_count,
    }


def _blog_sections_manifest(blueprint: "PostBlueprint") -> list[dict[str, Any]]:
    return [
        {
            "id": section.id,
            "heading": section.heading,
            "goal": section.goal,
            "chart_ids": list(section.chart_ids or []),
        }
        for section in (blueprint.sections or [])
    ]


def _attach_blog_blueprint_runtime_metadata(blueprint: "PostBlueprint") -> None:
    if not isinstance(blueprint.data_context, dict):
        blueprint.data_context = {}
    blueprint.data_context["generation_length_policy"] = _blog_length_policy(
        blueprint.topic_type
    )
    blueprint.data_context["section_word_budget"] = _blog_section_word_budget(
        blueprint
    )
    blueprint.data_context["sections_manifest"] = _blog_sections_manifest(blueprint)


def _build_cta(topic_type: str, data_context: dict[str, Any]) -> dict[str, Any] | None:
    """Build structured CTA from topic type and data context."""
    cfg = _CTA_CONFIG.get(topic_type)
    if not cfg:
        return None
    tc = data_context.get("topic_ctx", {})
    vendor = (
        data_context.get("vendor")
        or data_context.get("vendor_a")
        or tc.get("vendor")
        or tc.get("vendor_a")
    )
    category = data_context.get("category") or tc.get("category")
    return {
        "headline": "Want the full picture?",
        "body": "",
        "button_text": cfg["button_text"],
        "report_type": cfg["report_type"],
        "vendor_filter": vendor,
        "category_filter": category,
    }


def _inject_affiliate_links(blueprint: PostBlueprint, content: dict[str, Any]) -> None:
    """Inject affiliate placeholders/URLs into markdown as proper links."""
    body = str(content.get("content") or "")
    if not body:
        return

    affiliate_url = str(blueprint.data_context.get("affiliate_url") or "").strip()
    affiliate_slug = str(blueprint.data_context.get("affiliate_slug") or "").strip()
    partner_info = blueprint.data_context.get("affiliate_partner", {})
    if not isinstance(partner_info, dict):
        partner_info = {}
    partner_name = str(partner_info.get("name") or partner_info.get("product_name") or "").strip()

    if not (affiliate_slug and affiliate_url):
        return

    md_link = f"[{partner_name}]({affiliate_url})" if partner_name else affiliate_url
    body = body.replace(f"{{{{affiliate:{affiliate_slug}}}}}", md_link)
    if affiliate_url in body:
        body = re.sub(
            r'(?<!\]\()' + re.escape(affiliate_url) + r'(?!\))',
            md_link,
            body,
        )
    content["content"] = body


def _normalize_quote_text(text: Any) -> str:
    raw = str(text or "")
    raw = raw.replace('\u201c', '"').replace('\u201d', '"').replace('\u2019', "'")
    raw = raw.strip().strip('"').strip("'")
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"[^a-z0-9 ]+", " ", raw.lower())
    return re.sub(r"\s+", " ", raw).strip()


def _extract_quote_body(line: str) -> str:
    body = str(line or "").strip()
    if "--" in body:
        body = body.split("--", 1)[0].strip()
    return body.strip().strip('"').strip("'").strip()


def _extract_blockquote_quotes(markdown: str) -> list[str]:
    quotes: list[str] = []
    for line in str(markdown or "").splitlines():
        match = _BLOCKQUOTE_RE.match(line)
        if not match:
            continue
        quote = _extract_quote_body(match.group(1))
        if len(quote) >= 12:
            quotes.append(quote)
    return quotes


def _source_quote_texts(blueprint: PostBlueprint) -> list[str]:
    phrases = blueprint.quotable_phrases or []
    expected_vendors = _expected_quote_vendors(blueprint)
    candidate_vendors = {
        str(item.get("vendor") or "").strip()
        for item in phrases
        if isinstance(item, dict) and str(item.get("vendor") or "").strip()
    }
    out: list[str] = []
    seen: set[str] = set()
    for item in phrases:
        if not isinstance(item, dict):
            continue
        item_vendor = str(item.get("vendor") or "").strip()
        if expected_vendors and item_vendor and item_vendor.lower() not in expected_vendors:
            continue
        for key in ("phrase", "text", "quote", "best_quote"):
            value = str(item.get(key) or "").strip()
            if not value:
                continue
            if expected_vendors and _quote_mentions_unexpected_vendor(
                value, expected_vendors, candidate_vendors,
            ):
                continue
            marker = _normalize_quote_text(value)
            if not marker or marker in seen:
                continue
            seen.add(marker)
            out.append(value)
            break
    return out


def _expected_quote_vendors(blueprint: PostBlueprint) -> set[str]:
    data_context = blueprint.data_context or {}
    expected: set[str] = set()
    for key in ("vendor", "vendor_a", "vendor_b", "from_vendor", "to_vendor"):
        value = str(data_context.get(key) or "").strip().lower()
        if value:
            expected.add(value)
    return expected


def _normalized_vendor_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _quote_mentions_unexpected_vendor(
    quote_text: str,
    expected_vendors: set[str],
    candidate_vendors: set[str],
) -> bool:
    if not expected_vendors or not candidate_vendors:
        return False
    normalized_quote = f" {_normalized_vendor_text(quote_text)} "
    if not normalized_quote.strip():
        return False
    for vendor in candidate_vendors:
        if not vendor:
            continue
        normalized_vendor = _normalized_vendor_text(vendor)
        if not normalized_vendor:
            continue
        if vendor.lower() in expected_vendors:
            continue
        if f" {normalized_vendor} " in normalized_quote:
            return True
    return False


def _quote_matches_source(quote_text: str, source_quotes: list[str]) -> bool:
    normalized_quote = _normalize_quote_text(quote_text)
    if not normalized_quote:
        return True
    quote_tokens = set(normalized_quote.split())
    if not quote_tokens:
        return True

    for source_text in source_quotes:
        normalized_source = _normalize_quote_text(source_text)
        if not normalized_source:
            continue
        if normalized_quote in normalized_source or normalized_source in normalized_quote:
            return True
        source_tokens = set(normalized_source.split())
        overlap = len(quote_tokens & source_tokens)
        min_required = max(5, int(0.6 * min(len(quote_tokens), len(source_tokens))))
        if overlap >= min_required:
            return True
    return False


def _remove_unmatched_quote_lines(markdown: str, source_quotes: list[str]) -> tuple[str, int]:
    if not source_quotes:
        return markdown, 0
    removed = 0
    output: list[str] = []
    for line in str(markdown or "").splitlines():
        match = _BLOCKQUOTE_RE.match(line)
        if not match:
            output.append(line)
            continue
        quote = _extract_quote_body(match.group(1))
        if quote and not _quote_matches_source(quote, source_quotes):
            removed += 1
            continue
        output.append(line)
    return "\n".join(output), removed


def _sanitize_blog_markdown(markdown: str) -> tuple[str, dict[str, int]]:
    """Apply deterministic cleanup for common LLM artifacts."""
    text = str(markdown or "")
    answer_hits = len(_ANSWER_PREFIX_RE.findall(text))
    if answer_hits:
        text = _ANSWER_PREFIX_RE.sub(r"\1", text)
    return text, {"answer_prefix_removed": answer_hits}


def _required_vendor_mentions(blueprint: PostBlueprint, content_text: str) -> list[str]:
    ctx = blueprint.data_context if isinstance(blueprint.data_context, dict) else {}
    required: list[str] = []
    for key in ("vendor", "vendor_a", "vendor_b"):
        value = str(ctx.get(key) or "").strip()
        if value:
            required.append(value)
    if not required:
        topic_ctx = ctx.get("topic_ctx")
        if isinstance(topic_ctx, dict):
            for key in ("vendor", "vendor_a", "vendor_b"):
                value = str(topic_ctx.get(key) or "").strip()
                if value:
                    required.append(value)
    missing: list[str] = []
    haystack = str(content_text or "")
    for vendor in required:
        if not re.search(rf"\b{re.escape(vendor)}\b", haystack, re.IGNORECASE):
            missing.append(vendor)
    return missing


def _apply_blog_quality_gate(
    blueprint: PostBlueprint,
    content: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run deterministic quality checks and cleanup.

    Returns (possibly cleaned content, quality report).
    """
    cleaned = dict(content or {})
    body = str(cleaned.get("content") or "")
    body, sanitize_counts = _sanitize_blog_markdown(body)

    source_quotes = _source_quote_texts(blueprint)
    body, removed_quotes = _remove_unmatched_quote_lines(body, source_quotes)
    cleaned["content"] = body

    blocking_issues: list[str] = []
    warnings: list[str] = []
    fixes_applied: list[str] = []

    if sanitize_counts.get("answer_prefix_removed", 0) > 0:
        fixes_applied.append(f"removed_answer_prefix:{sanitize_counts['answer_prefix_removed']}")
    if removed_quotes > 0:
        fixes_applied.append(f"removed_unmatched_quotes:{removed_quotes}")

    length_policy = _blog_length_policy(blueprint.topic_type)
    min_words = int(length_policy["min_words"])
    target_words = int(length_policy["target_words"])
    word_count = len(body.split())
    if word_count < min_words:
        blocking_issues.append(f"content_too_short:{word_count}_words_need_{min_words}")
    elif word_count < target_words:
        warnings.append(f"content_below_seo_target_{target_words}_words")

    chart_ids = [c.chart_id for c in blueprint.charts]
    chart_mentions = re.findall(r"\{\{chart:([a-zA-Z0-9\-_]+)\}\}", body)
    chart_counts = {chart_id: chart_mentions.count(chart_id) for chart_id in chart_ids}
    for chart_id in chart_ids:
        count = chart_counts.get(chart_id, 0)
        if count == 0:
            blocking_issues.append(f"missing_chart_placeholder:{chart_id}")
        elif count > 1:
            blocking_issues.append(f"duplicate_chart_placeholder:{chart_id}")
    unknown_chart_ids = sorted({cid for cid in chart_mentions if cid not in set(chart_ids)})
    if unknown_chart_ids:
        blocking_issues.append(f"unknown_chart_placeholders:{','.join(unknown_chart_ids)}")

    unresolved_tokens: list[str] = []
    for token in _PLACEHOLDER_RE.findall(body):
        token = token.strip()
        if token.startswith("chart:"):
            continue
        unresolved_tokens.append(token)
    if unresolved_tokens:
        blocking_issues.append(
            f"unresolved_placeholders:{','.join(sorted(set(unresolved_tokens))[:6])}"
        )

    quoted = _extract_blockquote_quotes(body)
    if source_quotes and len(quoted) < 2:
        blocking_issues.append(f"too_few_sourced_quotes:{len(quoted)}")
    if not source_quotes and len(quoted) == 0:
        warnings.append("no_quotes_present")

    review_period = str((blueprint.data_context or {}).get("review_period") or "").strip()
    if review_period and review_period not in body:
        warnings.append("review_period_not_explicitly_mentioned")
    if "self-selected" not in body.lower():
        warnings.append("methodology_disclaimer_missing_self_selected")

    missing_vendors = _required_vendor_mentions(blueprint, body)
    if missing_vendors:
        blocking_issues.append(f"missing_vendor_mentions:{','.join(missing_vendors)}")

    specificity_context = surface_specificity_context(
        blueprint.data_context if isinstance(blueprint.data_context, dict) else {},
        surface="blog",
    )
    if specificity_context:
        specificity = evaluate_specificity_support(
            body,
            anchor_examples=specificity_context.get("anchor_examples"),
            witness_highlights=specificity_context.get("witness_highlights"),
            reference_ids=specificity_context.get("reference_ids"),
            allow_company_names=False,
            min_anchor_hits=int(settings.b2b_churn.blog_specificity_min_anchor_hits),
            require_anchor_support=bool(
                settings.b2b_churn.blog_specificity_require_anchor_support
            ),
            require_timing_or_numeric_when_available=bool(
                settings.b2b_churn.blog_specificity_require_timing_or_numeric_when_available
            ),
            include_competitor_terms=True,
            numeric_term_filter=_is_meaningful_numeric_anchor,
        )
        blocking_issues.extend(
            f"witness_specificity:{issue}"
            for issue in specificity.get("blocking_issues", []) or []
        )
        warnings.extend(
            f"witness_specificity:{warning}"
            for warning in specificity.get("warnings", []) or []
        )

    # Block placeholder links
    if 'href="#"' in body or "href='#'" in body:
        blocking_issues.append("placeholder_links_href_hash")

    # Block nonexistent internal blog links
    internal_links = re.findall(r'/blog/([a-z0-9\-]+)', body)
    if internal_links:
        known = set((blueprint.data_context or {}).get("_valid_internal_slugs") or [])
        fake = [lk for lk in internal_links if lk not in known and lk != blueprint.slug]
        if fake:
            blocking_issues.append(f"nonexistent_internal_links:{','.join(fake[:4])}")

    # Warn on title/slug mismatch
    title_lower = str(content.get("title") or blueprint.suggested_title or "").lower()
    for vk in ("vendor", "vendor_a", "vendor_b"):
        v = str((blueprint.data_context or {}).get(vk) or "").strip()
        if v and len(v) > 2 and v.lower() not in title_lower:
            warnings.append(f"title_missing_expected_vendor:{v}")
            break

    body_lower = body.lower()
    if "category winner" in body_lower or "category loser" in body_lower:
        dc = blueprint.data_context or {}
        if not (dc.get("category_winner") or dc.get("category_loser")):
            blocking_issues.append("unsupported_category_outcome_assertion")

    # Unsupported data claims: vendor names in claim-bearing sentences not in data
    grounded = _build_grounded_vendor_set(blueprint)
    ctx = blueprint.data_context if isinstance(blueprint.data_context, dict) else {}
    unsupported = _find_unsupported_data_claims(
        body, grounded, known_vendors=ctx.get("_known_vendors"),
    )
    for claim in unsupported[:3]:
        warnings.append(f"unsupported_data_claim:{claim}")

    # Chart-scope ambiguity: strongest claim phrases must match chart data_labels
    _CHART_SCOPE_PHRASES = (
        "most common source", "top migration source", "top source",
        "primary source", "where users come from", "where teams come from",
        "most common migration",
    )
    chart_labels_lower: set[str] = set()
    for chart in blueprint.charts:
        for row in chart.data:
            if isinstance(row, dict):
                for vk in ("name", "label", "category"):
                    v = str(row.get(vk) or "").strip()
                    if v:
                        chart_labels_lower.add(v.lower())
    if chart_labels_lower:
        sentences = re.split(r'(?<=[.!?])\s+|\n+', body)
        for sentence in sentences:
            s_lower = sentence.lower()
            if not any(phrase in s_lower for phrase in _CHART_SCOPE_PHRASES):
                continue
            # Check if sentence names a vendor NOT in chart labels
            for v in (ctx.get("_known_vendors") or []):
                if len(v) > 2 and v.lower() not in chart_labels_lower and re.search(r"\b" + re.escape(v) + r"\b", sentence, re.IGNORECASE):
                    warnings.append(
                        f"chart_scope_ambiguity:{v}: {sentence.strip()[:100]}"
                    )
                    break

    # Numeric consistency: check that sub-counts don't exceed headline counts
    _COUNT_PATTERN = re.compile(
        r"(\d[\d,]*)\s+(?:switching|churn|migration|displacement)\s+"
        r"(?:signals?|stories|reviews?|mentions?)",
        re.IGNORECASE,
    )
    count_values = [int(m.replace(",", "")) for m in _COUNT_PATTERN.findall(body)]
    if len(count_values) >= 2:
        headline = max(count_values)
        sub_total = sum(v for v in count_values if v != headline)
        if sub_total > headline and headline > 0:
            warnings.append(
                f"numeric_inconsistency:sub-counts ({sub_total}) exceed headline ({headline})"
            )

    # Direction guard: migration_guide should stay inbound-focused
    if blueprint.topic_type == "migration_guide":
        dc = blueprint.data_context if isinstance(blueprint.data_context, dict) else {}
        topic_vendor = str(dc.get("vendor") or dc.get("to_vendor") or "").strip()
        if topic_vendor:
            _OUTBOUND_PHRASES = (
                f"switching from {topic_vendor}".lower(),
                f"leaving {topic_vendor}".lower(),
                f"moving away from {topic_vendor}".lower(),
                f"migrating from {topic_vendor}".lower(),
                f"abandoning {topic_vendor}".lower(),
            )
            outbound_count = sum(
                1 for phrase in _OUTBOUND_PHRASES
                if phrase in body_lower
            )
            if outbound_count >= 2:
                warnings.append(
                    f"migration_direction_drift:too much outbound prose about leaving {topic_vendor} "
                    f"in a switch-to-{topic_vendor} article ({outbound_count} outbound phrases)"
                )

    threshold = int(settings.b2b_churn.blog_quality_pass_score)
    score = max(0, 100 - (18 * len(blocking_issues)) - (6 * len(warnings)))
    report = {
        "score": score,
        "threshold": threshold,
        "status": "pass" if (not blocking_issues and score >= threshold) else "fail",
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "fixes_applied": fixes_applied,
        "quote_count": len(quoted),
        "word_count": word_count,
        "min_words_required": min_words,
        "target_words": target_words,
    }
    return cleaned, report


def _quality_feedback(report: dict[str, Any]) -> list[str]:
    feedback: list[str] = []
    for issue in report.get("blocking_issues", []) or []:
        feedback.append(f"Fix: {issue}")
    if (
        report.get("status") == "fail"
        and not (report.get("blocking_issues") or [])
        and report.get("score") is not None
        and report.get("threshold") is not None
        and report.get("score") < report.get("threshold")
    ):
        feedback.append(
            "Fix: Reduce warning-level quality issues so the final score "
            f"meets the passing threshold of {report['threshold']}."
        )
    for warning in (report.get("warnings", []) or []):
        if warning == "review_period_not_explicitly_mentioned":
            feedback.append("Fix: Explicitly state the exact review period from data_context.review_period.")
        elif warning == "methodology_disclaimer_missing_self_selected":
            feedback.append("Fix: Include a plain-language methodology line that reviewers are self-selected and signals reflect perception.")
        elif str(warning).startswith("chart_scope_ambiguity:"):
            feedback.append(
                "Fix: Differentiate charted source data from broader displacement/evaluation signals. "
                "Use 'most common' and 'top sources' only for vendors visible in the chart's data_labels. "
                "For vendors from broader data_context, use hedged language like 'broader displacement signals also mention' or 'evaluation alternatives include'."
            )
        elif str(warning).startswith("unsupported_data_claim:"):
            feedback.append(
                "Fix: A data claim references a vendor not present in any chart or data_context. "
                "Either remove the claim or reword to cite the actual data source."
            )
        elif str(warning).startswith("migration_direction_drift:"):
            feedback.append(
                "Fix: This is a switch-TO article. Keep the primary narrative about inbound migration. "
                "Outbound switching (people leaving the vendor) may be acknowledged in 1-2 sentences as a brief caveat, "
                "but must not become a narrative thread. Remove or compress outbound-focused passages."
            )
        elif str(warning).startswith("witness_specificity:"):
            feedback.append(
                "Fix: Use the witness-backed proof anchors already provided. "
                "Ground key claims in the concrete timing, numeric, competitor, or pain details instead of generic summary language."
            )
        else:
            feedback.append(f"Improve: {warning}")
    return feedback[:8]


def _critical_quality_warnings(report: dict[str, Any]) -> list[str]:
    return [
        str(w)
        for w in (report.get("warnings", []) or [])
        if any(str(w) == cw or str(w).startswith(cw + ":") for cw in _CRITICAL_BLOG_WARNINGS)
    ]


def _with_unresolved_critical_warnings(report: dict[str, Any]) -> dict[str, Any]:
    critical = _critical_quality_warnings(report)
    if not critical:
        return report
    enriched = dict(report)
    blocking = list(enriched.get("blocking_issues", []) or [])
    for warning in critical:
        marker = f"critical_warning_unresolved:{warning}"
        if marker not in blocking:
            blocking.append(marker)
    enriched["blocking_issues"] = blocking
    enriched["status"] = "fail"
    return enriched


def _insert_blog_note(body: str, note: str, *, marker: str) -> str:
    text = str(body or "").strip()
    note_text = str(note or "").strip()
    if not text or not note_text:
        return text

    blocks = text.split("\n\n")
    marker_prefix = marker.lower()
    for idx, block in enumerate(blocks):
        if block.strip().lower().startswith(marker_prefix):
            if block.strip() == note_text:
                return text
            blocks[idx] = note_text
            return "\n\n".join(blocks)
    insert_at = 0
    if blocks and blocks[0].lstrip().startswith("#"):
        insert_at = 1
        if len(blocks) > 1 and blocks[1].strip().lower().startswith("_methodology note:"):
            insert_at = 2
    blocks.insert(min(insert_at, len(blocks)), note_text)
    return "\n\n".join(blocks)


def _has_blog_note(body: str, *, marker: str) -> bool:
    text = str(body or "").strip()
    if not text:
        return False
    marker_prefix = str(marker or "").strip().lower()
    if not marker_prefix:
        return False
    return any(
        block.strip().lower().startswith(marker_prefix)
        for block in text.split("\n\n")
    )


def _remove_blog_note(body: str, *, marker: str) -> str:
    text = str(body or "").strip()
    if not text:
        return text
    marker_prefix = str(marker or "").strip().lower()
    if not marker_prefix:
        return text
    blocks = [
        block
        for block in text.split("\n\n")
        if not block.strip().lower().startswith(marker_prefix)
    ]
    return "\n\n".join(blocks).strip()


def _blog_specificity_context(blueprint: PostBlueprint) -> dict[str, Any]:
    return surface_specificity_context(
        blueprint.data_context if isinstance(blueprint.data_context, dict) else {},
        surface="blog",
    )


def _specificity_rows(blueprint: PostBlueprint) -> list[dict[str, Any]]:
    specificity = _blog_specificity_context(blueprint)
    collected: list[dict[str, Any]] = []
    anchors = specificity.get("anchor_examples")
    preferred_labels = (
        "outlier_or_named_account",
        "common_pattern",
        "counterevidence",
    )
    if isinstance(anchors, dict):
        for label in preferred_labels:
            group_rows = anchors.get(label) or []
            if isinstance(group_rows, list):
                for row in group_rows:
                    if isinstance(row, dict):
                        collected.append(row)
        for group_rows in anchors.values():
            if isinstance(group_rows, list):
                for row in group_rows:
                    if isinstance(row, dict):
                        collected.append(row)
    highlights = specificity.get("witness_highlights") or []
    if isinstance(highlights, list):
        for row in highlights:
            if isinstance(row, dict):
                collected.append(row)
    return collected


def _first_numeric_literal(witness: dict[str, Any]) -> str:
    numeric_literals = witness.get("numeric_literals")
    if not isinstance(numeric_literals, dict):
        return ""
    for values in numeric_literals.values():
        if isinstance(values, list):
            for value in values:
                text = str(value or "").strip()
                if _is_meaningful_numeric_anchor(text):
                    return text
            continue
        text = str(values or "").strip()
        if _is_meaningful_numeric_anchor(text):
            return text
    return ""


def _first_term_from_rows(
    rows: list[dict[str, Any]],
    extractor,
) -> str:
    for row in rows:
        value = str(extractor(row) or "").strip()
        if value:
            return value
    return ""


def _preferred_signal_term(
    values: set[str] | None,
    *,
    skip: set[str] | None = None,
    predicate=None,
) -> str:
    candidates = [
        str(value or "").strip()
        for value in (values or set())
        if str(value or "").strip()
    ]
    if skip:
        lowered_skip = {str(value).strip().lower() for value in skip if str(value).strip()}
        candidates = [value for value in candidates if value.lower() not in lowered_skip]
    if predicate is not None:
        candidates = [value for value in candidates if predicate(value)]
    if not candidates:
        return ""
    candidates.sort(key=lambda token: (-len(token), token.lower()))
    return candidates[0]


def _is_meaningful_numeric_anchor(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if "%" in text:
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*[kmbx]\b", text, re.IGNORECASE):
        return True
    if re.search(r"[$\u20ac\u00a3]\s*\d", text):
        numeric_text = re.sub(r"[^0-9.]", "", text)
        try:
            return float(numeric_text) >= 1.0
        except (TypeError, ValueError):
            return False
    numeric_groups = [
        chunk.replace(",", "")
        for chunk in re.findall(r"\d[\d,]*", text)
    ]
    return any(len(chunk) >= 3 for chunk in numeric_groups)


def _blog_low_signal_anchor_labels() -> set[str]:
    values = getattr(settings.b2b_churn, "blog_evidence_anchor_low_signal_labels", []) or []
    labels: set[str] = set()
    for value in values:
        label = str(value or "").strip().lower().replace("_", " ")
        if label:
            labels.add(label)
    return labels


def _humanize_blog_anchor_label(value: Any) -> str:
    return " ".join(str(value or "").strip().replace("_", " ").split())


def _is_meaningful_blog_anchor_label(value: Any) -> bool:
    label = _humanize_blog_anchor_label(value).lower()
    if not label:
        return False
    if label in {"unknown", "none", "recommendation"}:
        return False
    if label in _blog_low_signal_anchor_labels():
        return False
    if any(pattern in label for pattern in (settings.b2b_churn.witness_specificity_generic_patterns or [])):
        return False
    condensed = re.sub(r"[^a-z0-9]", "", label)
    if condensed.isalpha() and len(condensed) <= 2:
        return False
    return True


def _meaningful_pain_label(witness: dict[str, Any]) -> str:
    for key in ("pain_category", "signal_type", "org_pressure_type"):
        value = _humanize_blog_anchor_label(witness.get(key))
        if not _is_meaningful_blog_anchor_label(value):
            continue
        return value
    return ""


def _meaningful_workflow_label(witness: dict[str, Any]) -> str:
    for key in ("replacement_mode", "operating_model_shift", "productivity_delta_claim"):
        value = _humanize_blog_anchor_label(witness.get(key))
        if not _is_meaningful_blog_anchor_label(value):
            continue
        return value
    return ""


def _needs_timing_or_numeric_anchor(report: dict[str, Any]) -> bool:
    blockers = [str(issue) for issue in (report.get("blocking_issues") or [])]
    return any(
        issue.startswith("witness_specificity:")
        and "timing or numeric anchor" in issue
        for issue in blockers
    )


def _select_specificity_anchor_terms(
    blueprint: PostBlueprint,
    report: dict[str, Any],
) -> dict[str, str]:
    specificity = _blog_specificity_context(blueprint)
    rows = _specificity_rows(blueprint)
    if not rows:
        return {}

    needs_timing_numeric = _needs_timing_or_numeric_anchor(report)
    timing_anchor = _first_term_from_rows(rows, lambda row: row.get("time_anchor"))
    numeric_anchor = _first_term_from_rows(rows, _first_numeric_literal)
    competitor = _first_term_from_rows(rows, lambda row: row.get("competitor"))
    pain = _first_term_from_rows(rows, _meaningful_pain_label)
    workflow = _first_term_from_rows(rows, _meaningful_workflow_label)

    signal_terms = specificity_signal_terms(
        anchor_examples=specificity.get("anchor_examples"),
        witness_highlights=specificity.get("witness_highlights"),
        allow_company_names=False,
    )
    if not timing_anchor:
        timing_anchor = _preferred_signal_term(signal_terms.get("timing_terms"))
    if not numeric_anchor:
        numeric_anchor = _preferred_signal_term(
            signal_terms.get("numeric_terms"),
            predicate=_is_meaningful_numeric_anchor,
        )
    if not competitor:
        competitor = _preferred_signal_term(signal_terms.get("competitor_terms"))
    if not pain:
        pain = _preferred_signal_term(
            signal_terms.get("pain_terms"),
            skip={
                "recommendation",
                "unknown",
                "none",
                *(_blog_low_signal_anchor_labels()),
            },
        )
        if pain and not _is_meaningful_blog_anchor_label(pain):
            pain = ""
        else:
            pain = _humanize_blog_anchor_label(pain)
    if not workflow:
        workflow = _preferred_signal_term(
            signal_terms.get("workflow_terms"),
            skip={"unknown", "none"},
        )
        if workflow and not _is_meaningful_blog_anchor_label(workflow):
            workflow = ""
        else:
            workflow = _humanize_blog_anchor_label(workflow)

    if needs_timing_numeric and not (timing_anchor or numeric_anchor):
        return {}

    return {
        "time_anchor": timing_anchor,
        "numeric_anchor": numeric_anchor,
        "competitor": competitor,
        "pain": pain,
        "workflow": workflow,
    }


def _build_specificity_anchor_note(
    blueprint: PostBlueprint,
    report: dict[str, Any],
) -> str:
    terms = _select_specificity_anchor_terms(blueprint, report)
    if not terms:
        return ""

    detail_bits: list[str] = []
    time_anchor = terms["time_anchor"]
    numeric_anchor = terms["numeric_anchor"]
    competitor = terms["competitor"]
    pain = terms["pain"]
    workflow = terms["workflow"]

    if time_anchor:
        detail_bits.append(f"{time_anchor} is the live timing trigger")
    if numeric_anchor:
        detail_bits.append(f"{numeric_anchor} is the concrete spend anchor")
    if competitor:
        detail_bits.append(f"{competitor} is the competitive alternative in the witness-backed record")
    if pain:
        detail_bits.append(f"the core pressure showing up in the evidence is {pain}")
    if workflow:
        detail_bits.append(f"the workflow shift in play is {workflow}")

    if not detail_bits:
        return ""

    note = "Evidence anchor: " + ", ".join(detail_bits[:-1])
    if len(detail_bits) > 1:
        note += f", and {detail_bits[-1]}."
    else:
        note += f"{detail_bits[-1]}."
    if len(detail_bits) == 1:
        note = "Evidence anchor: " + detail_bits[0] + "."
    return note


def _has_specificity_issues(report: dict[str, Any]) -> bool:
    blockers = [str(issue) for issue in (report.get("blocking_issues") or [])]
    warnings = [str(warning) for warning in (report.get("warnings") or [])]
    return any(issue.startswith("witness_specificity:") for issue in blockers + warnings)


def _apply_specificity_anchor_repair(
    blueprint: PostBlueprint,
    content: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    updated = dict(content or {})
    body = str(updated.get("content") or "")
    has_existing_note = _has_blog_note(body, marker="Evidence anchor:")
    if not _has_specificity_issues(report) and not has_existing_note:
        return updated, report, False

    note = _build_specificity_anchor_note(blueprint, report)
    if not note:
        if not has_existing_note:
            return updated, report, False
        repaired_body = _remove_blog_note(body, marker="Evidence anchor:")
        if repaired_body == body:
            return updated, report, False
        updated["content"] = repaired_body
        updated, repaired_report = _apply_blog_quality_gate(blueprint, updated)
        repaired_report = dict(repaired_report)
        fixes = list(repaired_report.get("fixes_applied", []) or [])
        fixes.append("removed_low_signal_witness_anchor_note")
        repaired_report["fixes_applied"] = fixes
        return updated, repaired_report, True

    repaired_body = _insert_blog_note(body, note, marker="Evidence anchor:")
    if repaired_body == body:
        return updated, report, False

    updated["content"] = repaired_body
    updated, repaired_report = _apply_blog_quality_gate(blueprint, updated)
    repaired_report = dict(repaired_report)
    fixes = list(repaired_report.get("fixes_applied", []) or [])
    fixes.append("added_witness_anchor_note")
    repaired_report["fixes_applied"] = fixes
    return updated, repaired_report, True


def _only_content_too_short_blockers(report: dict[str, Any]) -> bool:
    blockers = [str(issue) for issue in (report.get("blocking_issues") or [])]
    return bool(blockers) and all(issue.startswith("content_too_short:") for issue in blockers)


def _shortfall_words(report: dict[str, Any]) -> int:
    try:
        min_words = int(report.get("min_words_required") or 0)
        word_count = int(report.get("word_count") or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min_words - word_count)


def _top_source_summary(source_distribution: Any, *, limit: int = 3) -> str:
    if not isinstance(source_distribution, dict):
        return ""
    ranked: list[tuple[str, int]] = []
    for key, value in source_distribution.items():
        name = str(key or "").strip()
        if not name:
            continue
        try:
            count = int(value or 0)
        except (TypeError, ValueError):
            continue
        ranked.append((name, count))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: (-item[1], item[0].lower()))
    parts = [f"{name} ({count})" for name, count in ranked[: max(1, int(limit))]]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _build_coverage_snapshot_note(blueprint: PostBlueprint) -> str:
    data_context = blueprint.data_context if isinstance(blueprint.data_context, dict) else {}
    enriched_count = int(data_context.get("enriched_count") or 0)
    churn_intent_count = int(data_context.get("churn_intent_count") or 0)
    review_period = str(data_context.get("review_period") or "").strip()
    source_summary = _top_source_summary(data_context.get("source_distribution"))
    data_quality = data_context.get("data_quality")
    confidence = ""
    if isinstance(data_quality, dict):
        confidence = str(data_quality.get("confidence") or "").strip()
    scope_summary = data_context.get("reasoning_scope_summary")
    witnesses_in_scope = 0
    if isinstance(scope_summary, dict):
        try:
            witnesses_in_scope = int(scope_summary.get("witnesses_in_scope") or 0)
        except (TypeError, ValueError):
            witnesses_in_scope = 0

    details: list[str] = []
    if enriched_count > 0:
        details.append(f"this piece draws from {enriched_count} enriched reviews")
    if churn_intent_count > 0:
        details.append(f"including {churn_intent_count} churn-intent signals")
    if witnesses_in_scope > 0:
        details.append(f"with {witnesses_in_scope} witness-backed examples in scope")
    if review_period:
        details.append(f"collected between {review_period}")
    if source_summary:
        details.append(f"with the visible source mix led by {source_summary}")
    if confidence:
        details.append(f"and a current data-quality posture of {confidence} confidence")

    if not details:
        return ""

    note = "Coverage snapshot: " + ", ".join(details[:-1])
    if len(details) > 1:
        note += f", and {details[-1]}."
    else:
        note += f"{details[-1]}."
    if len(details) == 1:
        note = "Coverage snapshot: " + details[0] + "."
    note += " Treat the analysis as directional evidence rather than a full market census."
    return note


def _apply_borderline_shortfall_repair(
    blueprint: PostBlueprint,
    content: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    if not _only_content_too_short_blockers(report):
        return dict(content or {}), report, False

    shortfall = _shortfall_words(report)
    threshold = int(getattr(settings.b2b_churn, "blog_post_borderline_shortfall_max_words", 0) or 0)
    if shortfall <= 0 or shortfall > threshold:
        return dict(content or {}), report, False

    note = _build_coverage_snapshot_note(blueprint)
    if not note:
        return dict(content or {}), report, False

    updated = dict(content or {})
    body = str(updated.get("content") or "")
    repaired_body = _insert_blog_note(body, note, marker="Coverage snapshot:")
    if repaired_body == body:
        return updated, report, False

    updated["content"] = repaired_body
    updated, repaired_report = _apply_blog_quality_gate(blueprint, updated)
    repaired_report = dict(repaired_report)
    fixes = list(repaired_report.get("fixes_applied", []) or [])
    fixes.append("added_coverage_snapshot_note")
    repaired_report["fixes_applied"] = fixes
    return updated, repaired_report, True


def _has_unsupported_category_outcome_issue(report: dict[str, Any]) -> bool:
    return any(
        str(issue) == "unsupported_category_outcome_assertion"
        for issue in (report.get("blocking_issues") or [])
    )


def _line_contains_unsupported_category_outcome(line: str) -> bool:
    return bool(re.search(r"\bcategory\s+(winner|loser)\b", str(line or ""), re.IGNORECASE))


def _remove_unsupported_category_outcome_lines(body: str, report: dict[str, Any]) -> tuple[str, int]:
    if not _has_unsupported_category_outcome_issue(report):
        return str(body or ""), 0
    lines = str(body or "").splitlines()
    if not lines:
        return str(body or ""), 0
    removed = 0
    next_lines: list[str] = []
    for line in lines:
        if _line_contains_unsupported_category_outcome(line):
            removed += 1
            continue
        next_lines.append(line)
    if removed > 0:
        return "\n".join(next_lines), removed
    text = "\n".join(lines)
    sentences = re.split(r"(?<=[.!?])(\s+)", text)
    kept: list[str] = []
    for segment in sentences:
        if _line_contains_unsupported_category_outcome(segment):
            removed += 1
            continue
        kept.append(segment)
    return "".join(kept), removed


def _apply_blog_deterministic_repairs(
    blueprint: PostBlueprint,
    content: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(content or {})
    body = str(updated.get("content") or "")
    repaired_body, removed_claim_lines = _remove_unsupported_claim_lines(body, report)
    repaired_body, removed_category_outcome_lines = _remove_unsupported_category_outcome_lines(
        repaired_body,
        report,
    )
    valid_slugs = {
        str(slug).strip()
        for slug in ((blueprint.data_context or {}).get("_valid_internal_slugs") or [])
        if str(slug or "").strip()
    }
    repaired_body, removed_bad_links = _strip_nonexistent_internal_links(
        repaired_body,
        valid_slugs=valid_slugs,
        current_slug=str(blueprint.slug or "").strip(),
    )
    repaired_report = dict(report or {})
    if (
        removed_claim_lines > 0
        or removed_category_outcome_lines > 0
        or removed_bad_links > 0
    ) and repaired_body != body:
        updated["content"] = repaired_body
        updated, repaired_report = _apply_blog_quality_gate(blueprint, updated)
        repaired_report = dict(repaired_report)
        fixes = list(repaired_report.get("fixes_applied", []) or [])
        if removed_claim_lines > 0:
            fixes.append(f"removed_unsupported_claim_lines:{removed_claim_lines}")
        if removed_category_outcome_lines > 0:
            fixes.append(
                f"removed_unsupported_category_outcome_lines:{removed_category_outcome_lines}"
            )
        if removed_bad_links > 0:
            fixes.append(f"removed_nonexistent_internal_links:{removed_bad_links}")
        repaired_report["fixes_applied"] = fixes

    updated, repaired_report, _ = _apply_specificity_anchor_repair(
        blueprint,
        updated,
        repaired_report,
    )
    updated, repaired_report, _ = _apply_borderline_shortfall_repair(
        blueprint,
        updated,
        repaired_report,
    )
    return updated, repaired_report


def _run_blog_quality_pass(
    blueprint: PostBlueprint,
    content: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = _ensure_methodology_context(blueprint, dict(content or {}))
    _inject_affiliate_links(blueprint, current)
    current, report = _apply_blog_quality_gate(blueprint, current)
    current, report = _apply_blog_deterministic_repairs(blueprint, current, report)
    return current, report


def _with_first_pass_quality_metadata(
    report: dict[str, Any],
    *,
    first_pass_report: dict[str, Any],
    retry_requested: bool,
) -> dict[str, Any]:
    enriched = dict(report)
    enriched["_first_pass_report"] = dict(first_pass_report)
    enriched["_retry_requested"] = retry_requested
    return enriched


def _with_rejected_content_snapshot(
    report: dict[str, Any],
    *,
    content_obj: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(report)
    finalized = _finalize_blog_generation_content(content_obj)
    if isinstance(finalized, dict):
        enriched["_rejected_content"] = finalized
    return enriched


def _ensure_methodology_context(
    blueprint: PostBlueprint,
    content: dict[str, Any],
) -> dict[str, Any]:
    """Inject a deterministic methodology note when the draft omits it."""
    body = str((content or {}).get("content") or "").strip()
    if not body:
        return dict(content or {})

    review_period = str((blueprint.data_context or {}).get("review_period") or "").strip()
    if not review_period:
        return dict(content or {})

    has_review_period = review_period in body
    has_self_selected = "self-selected" in body.lower()
    if has_review_period and has_self_selected:
        return dict(content or {})

    source_label = str((blueprint.data_context or {}).get("data_source_label") or "").strip()
    source_text = source_label or "public software reviews"
    note = (
        f"_Methodology note: This analysis reflects self-selected feedback from {source_text} "
        f"collected between {review_period}. It captures reviewer perception, not a census of all users._"
    )

    updated = dict(content or {})
    if body.startswith("# "):
        head, _, remainder = body.partition("\n")
        remainder = remainder.lstrip()
        updated["content"] = f"{head}\n\n{note}\n\n{remainder}" if remainder else f"{head}\n\n{note}"
    else:
        updated["content"] = f"{note}\n\n{body}"
    return updated


def _finalize_blog_generation_content(
    content_obj: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip cache-only fields before returning assembled blog content."""
    if content_obj is None:
        return None
    return {
        key: value
        for key, value in content_obj.items()
        if not str(key).startswith("_cache_")
    }


async def _store_blog_generation_cache(
    blueprint: PostBlueprint,
    content_obj: dict[str, Any] | None,
    *,
    pool: Any | None = None,
) -> None:
    """Persist the final approved blog generation result in the exact cache."""
    from ...services.b2b.cache_runner import (
        bind_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )

    if not isinstance(content_obj, dict):
        return
    request_envelope = content_obj.get("_cache_request_envelope")
    provider_name = str(content_obj.get("_cache_provider") or "")
    model_name = str(content_obj.get("_cache_model") or "")
    if not isinstance(request_envelope, dict) or not provider_name or not model_name:
        return
    request = bind_b2b_exact_stage_request(
        "b2b_blog_post_generation.content",
        provider=provider_name,
        model=model_name,
        request_envelope=request_envelope,
    )
    await store_b2b_exact_stage_text(
        request,
        response_text=json.dumps(
            _finalize_blog_generation_content(content_obj),
            separators=(",", ":"),
            default=str,
        ),
        metadata={
            "slug": blueprint.slug,
            "topic_type": blueprint.topic_type,
        },
        pool=pool,
    )


async def _enforce_blog_quality_async(
    llm,
    blueprint: PostBlueprint,
    content: dict[str, Any],
    max_tokens: int,
    related_posts: list[dict[str, str]] | None = None,
    run_id: str | None = None,
    pool: Any | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Apply quality gate and perform one retry with feedback when needed."""
    current, report = _run_blog_quality_pass(blueprint, content)
    initial_critical = _critical_quality_warnings(report)
    needs_retry = report.get("status") != "pass" or bool(initial_critical)
    first_pass_report = dict(report)
    if not needs_retry:
        await _store_blog_generation_cache(blueprint, current, pool=pool)
        return _finalize_blog_generation_content(current), _with_first_pass_quality_metadata(
            report,
            first_pass_report=first_pass_report,
            retry_requested=False,
        )

    retry = await _generate_content_async(
        llm,
        blueprint,
        max_tokens,
        related_posts=related_posts,
        quality_feedback=_quality_feedback(report),
        run_id=run_id,
        pool=pool,
    )
    if retry is None:
        if initial_critical:
            return None, _with_first_pass_quality_metadata(
                _with_rejected_content_snapshot(
                    _with_unresolved_critical_warnings(report),
                    content_obj=current,
                ),
                first_pass_report=first_pass_report,
                retry_requested=True,
            )
        return None, _with_first_pass_quality_metadata(
            _with_rejected_content_snapshot(report, content_obj=current),
            first_pass_report=first_pass_report,
            retry_requested=True,
        )

    retry, retry_report = _run_blog_quality_pass(blueprint, retry)
    retry_report = _with_unresolved_critical_warnings(retry_report)
    if retry_report.get("status") != "pass":
        return None, _with_first_pass_quality_metadata(
            _with_rejected_content_snapshot(retry_report, content_obj=retry),
            first_pass_report=first_pass_report,
            retry_requested=True,
        )
    await _store_blog_generation_cache(blueprint, retry, pool=pool)
    return _finalize_blog_generation_content(retry), _with_first_pass_quality_metadata(
        retry_report,
        first_pass_report=first_pass_report,
        retry_requested=True,
    )


def _enforce_blog_quality(
    llm,
    blueprint: PostBlueprint,
    content: dict[str, Any],
    max_tokens: int,
    related_posts: list[dict[str, str]] | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Sync wrapper retained for local/unit-test callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Use _enforce_blog_quality_async() from async contexts")

    current, report = _run_blog_quality_pass(blueprint, content)
    initial_critical = _critical_quality_warnings(report)
    needs_retry = report.get("status") != "pass" or bool(initial_critical)
    first_pass_report = dict(report)
    if not needs_retry:
        return _finalize_blog_generation_content(current), _with_first_pass_quality_metadata(
            report,
            first_pass_report=first_pass_report,
            retry_requested=False,
        )

    retry = _generate_content(
        llm,
        blueprint,
        max_tokens,
        related_posts=related_posts,
        quality_feedback=_quality_feedback(report),
        run_id=run_id,
    )
    if retry is None:
        if initial_critical:
            return None, _with_first_pass_quality_metadata(
                _with_rejected_content_snapshot(
                    _with_unresolved_critical_warnings(report),
                    content_obj=current,
                ),
                first_pass_report=first_pass_report,
                retry_requested=True,
            )
        return None, _with_first_pass_quality_metadata(
            _with_rejected_content_snapshot(report, content_obj=current),
            first_pass_report=first_pass_report,
            retry_requested=True,
        )

    retry, retry_report = _run_blog_quality_pass(blueprint, retry)
    retry_report = _with_unresolved_critical_warnings(retry_report)
    if retry_report.get("status") != "pass":
        return None, _with_first_pass_quality_metadata(
            _with_rejected_content_snapshot(retry_report, content_obj=retry),
            first_pass_report=first_pass_report,
            retry_requested=True,
        )
    return _finalize_blog_generation_content(retry), _with_first_pass_quality_metadata(
        retry_report,
        first_pass_report=first_pass_report,
        retry_requested=True,
    )


def _canonicalize_blog_quality(
    blueprint: PostBlueprint,
    *,
    content: dict[str, Any] | None,
    report: dict[str, Any],
    boundary: str,
) -> dict[str, Any]:
    from ...services.blog_quality import blog_quality_revalidation

    result = blog_quality_revalidation(
        blueprint=blueprint,
        content=content,
        boundary=boundary,
        report=report,
    )
    blueprint.data_context = result["data_context"]
    return result["audit"]


def _persist_first_pass_blog_quality(
    blueprint: PostBlueprint,
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    from ...services.blog_quality import (
        blog_quality_revalidation,
        merge_blog_first_pass_quality_data_context,
    )

    first_pass_report = report.get("_first_pass_report") if isinstance(report, dict) else None
    if not isinstance(first_pass_report, dict):
        return {}

    result = blog_quality_revalidation(
        blueprint=blueprint,
        content=None,
        boundary="generation",
        report=first_pass_report,
    )
    first_pass_audit = dict(result["audit"])
    blueprint.data_context = merge_blog_first_pass_quality_data_context(
        data_context=result["data_context"],
        audit=first_pass_audit,
    )
    return first_pass_audit


def _normalize_string_scope(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = [raw]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


# -- entry point --------------------------------------------------

async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate B2B data-backed blog posts.

    Loops up to ``max_per_run`` times, picking a fresh topic each iteration.
    All posts are stored as drafts.
    """
    cfg = settings.b2b_churn
    run_id = _task_run_id(task) or str(task.id)
    maintenance_run = bool((task.metadata or {}).get("maintenance_run"))
    if not cfg.blog_post_enabled and not maintenance_run:
        return {"_skip_synthesis": "B2B blog post generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    from ...pipelines.llm import get_pipeline_llm
    from ...pipelines.notify import send_pipeline_notification
    from ...services.b2b.anthropic_batch import (
        AnthropicBatchItem,
        mark_batch_fallback_result,
        run_anthropic_message_batch,
    )
    from ...services.b2b.cache_runner import lookup_b2b_exact_stage_text
    from ...services.llm.anthropic import AnthropicLLM
    from ...services.protocols import Message

    llm = get_pipeline_llm(
        workload="synthesis",
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model=cfg.blog_post_openrouter_model,
    )
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    batch_requested = bool(
        getattr(cfg, "anthropic_batch_enabled", False)
        and getattr(cfg, "blog_post_anthropic_batch_enabled", True)
    )
    batch_llm = get_pipeline_llm(workload="anthropic") if batch_requested else None
    if batch_llm is None and isinstance(llm, AnthropicLLM):
        batch_llm = llm
    generation_llm = batch_llm if isinstance(batch_llm, AnthropicLLM) else llm
    blog_batch_enabled = isinstance(batch_llm, AnthropicLLM)

    if generation_llm is None:
        return {"_skip_synthesis": "No LLM available for B2B blog post generation"}

    max_posts = _blog_post_max_per_run(task, cfg)
    results: list[dict[str, Any]] = []
    # Track vendors and topic types across iterations to prevent dominance
    run_seen_vendors: set[str] = set()
    run_seen_types: dict[str, int] = {}
    attempted_slugs: set[str] = set()
    blog_batch_metrics = {
        "jobs": 0,
        "submitted_items": 0,
        "cache_prefiltered_items": 0,
        "fallback_single_call_items": 0,
        "completed_items": 0,
        "failed_items": 0,
    }

    def _batch_payload() -> dict[str, int]:
        return {
            "blog_batch_jobs": int(blog_batch_metrics["jobs"]),
            "blog_batch_items_submitted": int(blog_batch_metrics["submitted_items"]),
            "blog_batch_cache_prefiltered": int(blog_batch_metrics["cache_prefiltered_items"]),
            "blog_batch_fallback_single_call": int(blog_batch_metrics["fallback_single_call_items"]),
            "blog_batch_completed_items": int(blog_batch_metrics["completed_items"]),
            "blog_batch_failed_items": int(blog_batch_metrics["failed_items"]),
        }

    def _apply_topic_scope(
        topic_type: str,
        topic_ctx: dict[str, Any],
        *,
        seen_vendors: set[str],
        seen_types: dict[str, int],
    ) -> None:
        for vendor_key in ("vendor", "vendor_a", "vendor_b", "from_vendor"):
            vendor_value = str(topic_ctx.get(vendor_key) or "").lower().strip()
            if vendor_value:
                seen_vendors.add(vendor_value)
        seen_types[topic_type] = seen_types.get(topic_type, 0) + 1

    async def _prepare_post_candidate(
        topic_type: str,
        topic_ctx: dict[str, Any],
    ) -> dict[str, Any] | None:
        topic_slug = str(topic_ctx.get("slug") or "").strip()
        if topic_slug and topic_slug in attempted_slugs:
            return None

        data = await _gather_data(pool, topic_type, topic_ctx)
        await _load_pool_layers_for_blog(pool, topic_type, topic_ctx, data)

        sufficiency = _check_data_sufficiency(topic_type, data)
        if not sufficiency["sufficient"]:
            logger.warning(
                "Data insufficiency for %s (%s): %s",
                topic_ctx.get("slug", "?"), topic_type, sufficiency["reason"],
            )
            return None

        blueprint = _build_blueprint(topic_type, topic_ctx, data)
        if blueprint.slug in attempted_slugs:
            return None
        blueprint_sufficiency = _check_blueprint_sufficiency(blueprint)
        if not blueprint_sufficiency["sufficient"]:
            logger.warning(
                "Blueprint insufficiency for %s (%s): %s",
                blueprint.slug,
                topic_type,
                blueprint_sufficiency["reason"],
            )
            return None

        link_posts = await _fetch_related_for_linking(
            pool, blueprint.tags, blueprint.slug,
        )
        blueprint.data_context["_valid_internal_slugs"] = [
            p["slug"] for p in (link_posts or []) if isinstance(p, dict) and p.get("slug")
        ]
        if "_known_vendors" not in blueprint.data_context:
            try:
                from ._b2b_shared import read_known_vendor_names

                blueprint.data_context["_known_vendors"] = await read_known_vendor_names(pool)
            except Exception:
                blueprint.data_context["_known_vendors"] = []

        return {
            "topic_type": topic_type,
            "topic_ctx": topic_ctx,
            "blueprint": blueprint,
            "link_posts": link_posts,
        }

    async def _generate_first_pass_batch(
        candidates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any] | None]:
        prepared_entries: list[dict[str, Any]] = []
        first_pass_by_slug: dict[str, dict[str, Any] | None] = {}

        for index, candidate in enumerate(candidates):
            blueprint = candidate["blueprint"]
            prepared = _prepare_blog_generation_request(
                generation_llm,
                blueprint,
                cfg.blog_post_max_tokens,
                related_posts=candidate["link_posts"],
            )
            if prepared is None:
                first_pass_by_slug[blueprint.slug] = None
                continue

            cached = await lookup_b2b_exact_stage_text(prepared["request"], pool=pool)
            prepared_entries.append({
                "custom_id": f"blog_post:{index}:{blueprint.slug}",
                "candidate": candidate,
                "prepared": prepared,
                "cached_response_text": (
                    str(cached["response_text"] or "")
                    if cached is not None
                    else None
                ),
                "cached_usage": (
                    dict(cached.get("usage") or {})
                    if cached is not None
                    else {}
                ),
            })

        if not prepared_entries:
            return first_pass_by_slug

        execution = await run_anthropic_message_batch(
            llm=generation_llm,
            stage_id=_BLOG_GENERATION_CACHE_STAGE,
            task_name="b2b_blog_post_generation",
            items=[
                AnthropicBatchItem(
                    custom_id=str(entry["custom_id"]),
                    artifact_type="blog_post",
                    artifact_id=str(entry["candidate"]["blueprint"].slug),
                    vendor_name=(
                        str(
                            entry["candidate"]["blueprint"].data_context.get("vendor")
                            or ""
                        )
                        or None
                    ),
                    messages=[
                        Message(
                            role=str(getattr(message, "role", "") or ""),
                            content=str(getattr(message, "content", "") or ""),
                        )
                        for message in entry["prepared"]["messages"]
                    ],
                    max_tokens=cfg.blog_post_max_tokens,
                    temperature=float(cfg.blog_post_temperature),
                    trace_span_name=_BLOG_GENERATION_TRACE_SPAN,
                    trace_metadata={
                        **_blog_generation_trace_metadata(
                            entry["candidate"]["blueprint"],
                            run_id=run_id,
                        ),
                        "workload": "anthropic_batch",
                    },
                    request_metadata={
                        "topic_type": entry["candidate"]["blueprint"].topic_type,
                        "slug": entry["candidate"]["blueprint"].slug,
                    },
                    cached_response_text=entry["cached_response_text"],
                    cached_usage=entry["cached_usage"],
                )
                for entry in prepared_entries
            ],
            run_id=run_id,
            min_batch_size=int(getattr(cfg, "blog_post_anthropic_batch_min_items", 2)),
            batch_metadata={
                "report_type": "blog_post",
            },
            pool=pool,
        )
        blog_batch_metrics["jobs"] += 1 if execution.provider_batch_id else 0
        blog_batch_metrics["submitted_items"] += execution.submitted_items
        blog_batch_metrics["cache_prefiltered_items"] += execution.cache_prefiltered_items
        blog_batch_metrics["fallback_single_call_items"] += execution.fallback_single_call_items
        blog_batch_metrics["completed_items"] += execution.completed_items
        blog_batch_metrics["failed_items"] += execution.failed_items

        for entry in prepared_entries:
            candidate = entry["candidate"]
            blueprint = candidate["blueprint"]
            prepared = entry["prepared"]
            outcome = execution.results_by_custom_id.get(str(entry["custom_id"]))
            parse_error_text: str | None = None
            content: dict[str, Any] | None = None

            if outcome is not None and outcome.response_text:
                content = _parse_blog_generation_response_text(
                    blueprint,
                    outcome.response_text,
                    request_envelope=prepared["request_envelope"],
                    provider_name=str(prepared["provider_name"]),
                    model_name=str(prepared["model_name"]),
                )
                if content is None:
                    parse_error_text = "failed_to_parse_batch_response"

            used_fallback_single_call = False
            if content is None:
                used_fallback_single_call = True
                fallback_usage: dict[str, Any] = {}
                content = await _generate_content_async(
                    generation_llm,
                    blueprint,
                    cfg.blog_post_max_tokens,
                    related_posts=candidate["link_posts"],
                    run_id=run_id,
                    pool=pool,
                    usage_out=fallback_usage,
                )
                await mark_batch_fallback_result(
                    batch_id=execution.local_batch_id,
                    custom_id=str(entry["custom_id"]),
                    succeeded=content is not None,
                    error_text=(
                        parse_error_text
                        or (
                            outcome.error_text
                            if outcome is not None and outcome.error_text and content is None
                            else None
                        )
                        or ("llm_returned_none" if content is None else None)
                    ),
                    response_text=(
                        json.dumps(
                            _finalize_blog_generation_content(content),
                            separators=(",", ":"),
                            default=str,
                        )
                        if content is not None
                        else None
                    ),
                    usage=fallback_usage,
                    provider=str(fallback_usage.get("provider") or "") or None,
                    model=str(fallback_usage.get("model") or "") or None,
                    provider_request_id=(
                        str(fallback_usage.get("provider_request_id") or "") or None
                    ),
                    pool=pool,
                )

            if used_fallback_single_call:
                logger.debug(
                    "Blog batch fallback used for %s",
                    blueprint.slug,
                )
            first_pass_by_slug[blueprint.slug] = content

        return first_pass_by_slug

    # Regeneration mode: re-process existing drafts through fixed pipeline
    if cfg.blog_post_regenerate_mode or (
        maintenance_run and bool((task.metadata or {}).get("regenerate_existing_posts"))
    ):
        return await _regenerate_existing_posts(pool, generation_llm, cfg, task, max_posts)

    while len(results) < max_posts:
        remaining_slots = max_posts - len(results)
        candidate_entries: list[dict[str, Any]] = []
        reserved_vendors = set(run_seen_vendors)
        reserved_types = dict(run_seen_types)
        selection_attempts = 0
        selection_limit = max(
            remaining_slots * _BLOG_BATCH_SELECTION_MULTIPLIER,
            _BLOG_BATCH_SELECTION_MIN_ATTEMPTS,
        )

        while len(candidate_entries) < remaining_slots and selection_attempts < selection_limit:
            selection_attempts += 1
            topic = await _select_topic(
                pool,
                max_posts,
                exclude_vendors=reserved_vendors,
                exclude_types=reserved_types,
            )
            if topic is None:
                break
            topic_type, topic_ctx = topic
            prepared_candidate = await _prepare_post_candidate(topic_type, topic_ctx)
            _apply_topic_scope(
                topic_type,
                topic_ctx,
                seen_vendors=reserved_vendors,
                seen_types=reserved_types,
            )
            if prepared_candidate is None:
                continue
            attempted_slugs.add(prepared_candidate["blueprint"].slug)
            candidate_entries.append(prepared_candidate)

        if not candidate_entries:
            logger.info("No more viable B2B topics after %d posts", len(results))
            break

        first_pass_by_slug: dict[str, dict[str, Any] | None] = {}
        if blog_batch_enabled:
            first_pass_by_slug = await _generate_first_pass_batch(candidate_entries)

        for candidate in candidate_entries:
            topic_type = candidate["topic_type"]
            topic_ctx = candidate["topic_ctx"]
            blueprint = candidate["blueprint"]
            link_posts = candidate["link_posts"]
            if blog_batch_enabled:
                content = first_pass_by_slug.get(blueprint.slug)
            else:
                content = await _generate_content_async(
                    generation_llm,
                    blueprint,
                    cfg.blog_post_max_tokens,
                    related_posts=link_posts,
                    run_id=run_id,
                    pool=pool,
                )

            if content is None:
                logger.warning("LLM failed for B2B topic %s, skipping", blueprint.slug)
                from ..visibility import record_attempt, emit_event
                await _upsert_blog_post_state(
                    pool,
                    blueprint,
                    generation_llm,
                    status="failed",
                    run_id=run_id,
                    attempt_no=1,
                    failure_step="llm_call",
                    error_code="llm_returned_none",
                    error_summary="LLM returned None",
                )
                await record_attempt(
                    pool, artifact_type="blog_post", artifact_id=blueprint.slug,
                    run_id=run_id, attempt_no=1, stage="generation",
                    status="failed", failure_step="llm_call",
                    error_message="LLM returned None",
                )
                await emit_event(
                    pool, stage="blog", event_type="generation_failure",
                    entity_type="blog_post", entity_id=blueprint.slug,
                    summary=f"LLM failed for {blueprint.slug} ({topic_type})",
                    severity="error", actionable=True,
                    artifact_type="blog_post", run_id=run_id,
                    reason_code="llm_returned_none",
                )
                continue

            content, quality_report = await _enforce_blog_quality_async(
                generation_llm,
                blueprint,
                content,
                cfg.blog_post_max_tokens,
                related_posts=link_posts,
                run_id=run_id,
                pool=pool,
            )
            first_pass_audit = _persist_first_pass_blog_quality(blueprint, quality_report)
            if quality_report.get("_retry_requested"):
                from ..visibility import record_attempt
                await record_attempt(
                    pool,
                    artifact_type="blog_post",
                    artifact_id=blueprint.slug,
                    run_id=run_id,
                    attempt_no=1,
                    stage="quality_gate_first_pass",
                    status="retry_requested",
                    score=first_pass_audit.get("score"),
                    threshold=first_pass_audit.get("threshold"),
                    blocker_count=len(first_pass_audit.get("blocking_issues", [])),
                    warning_count=len(first_pass_audit.get("warnings", [])),
                    blocking_issues=first_pass_audit.get("blocking_issues"),
                    warnings=first_pass_audit.get("warnings"),
                    failure_step="quality_gate_first_pass",
                    error_message=", ".join(first_pass_audit.get("blocking_issues", [])[:3]) or None,
                )
            if content is None:
                rejected_content = quality_report.get("_rejected_content") if isinstance(quality_report, dict) else None
                quality_audit = _canonicalize_blog_quality(
                    blueprint,
                    content=rejected_content if isinstance(rejected_content, dict) else None,
                    report=quality_report,
                    boundary="generation",
                )
                logger.warning(
                    "Quality gate failed for %s (%s): %s",
                    blueprint.slug,
                    blueprint.topic_type,
                    ", ".join(quality_audit.get("blocking_issues", [])[:4]) or "unknown issues",
                )
                from ..visibility import record_attempt, emit_event
                await _upsert_blog_post_state(
                    pool,
                    blueprint,
                    generation_llm,
                    status="rejected",
                    run_id=run_id,
                    attempt_no=1,
                    score=quality_audit.get("score"),
                    threshold=quality_audit.get("threshold"),
                    blocker_count=len(quality_audit.get("blocking_issues", [])),
                    warning_count=len(quality_audit.get("warnings", [])),
                    failure_step="quality_gate",
                    error_code="quality_gate_rejection",
                    error_summary=", ".join(quality_audit.get("blocking_issues", [])[:3]),
                    rejection_reason=", ".join(quality_audit.get("blocking_issues", [])[:3]),
                    content=rejected_content if isinstance(rejected_content, dict) else None,
                )
                await record_attempt(
                    pool, artifact_type="blog_post", artifact_id=blueprint.slug,
                    run_id=run_id, attempt_no=1, stage="quality_gate",
                    status="rejected",
                    score=quality_audit.get("score"),
                    threshold=quality_audit.get("threshold"),
                    blocker_count=len(quality_audit.get("blocking_issues", [])),
                    warning_count=len(quality_audit.get("warnings", [])),
                    blocking_issues=quality_audit.get("blocking_issues"),
                    warnings=quality_audit.get("warnings"),
                    failure_step="quality_gate",
                    error_message=", ".join(quality_audit.get("blocking_issues", [])[:3]),
                )
                await emit_event(
                    pool, stage="blog", event_type="quality_gate_rejection",
                    entity_type="blog_post", entity_id=blueprint.slug,
                    summary=f"Quality gate rejected {blueprint.slug} ({topic_type})",
                    severity="warning", actionable=True,
                    artifact_type="blog_post", run_id=run_id,
                    reason_code=quality_audit.get("blocking_issues", ["unknown"])[0][:80] if quality_audit.get("blocking_issues") else "unknown",
                    decision="rejected",
                    detail=quality_audit,
                )
                continue
            quality_audit = _canonicalize_blog_quality(
                blueprint,
                content=content,
                report=quality_report,
                boundary="generation",
            )

            post_id = await _assemble_and_store(
                pool,
                blueprint,
                content,
                generation_llm,
                run_id=run_id,
                attempt_no=1,
            )
            if not post_id:
                logger.info("Slug %s already published, skipping", blueprint.slug)
                from ..visibility import record_dedup
                await record_dedup(
                    pool, stage="blog", entity_type="blog_post",
                    entity_id=blueprint.slug, run_id=run_id,
                    reason=f"Slug {blueprint.slug} already published",
                )
                continue

            from ..visibility import record_attempt
            await record_attempt(
                pool, artifact_type="blog_post", artifact_id=blueprint.slug,
                run_id=run_id, attempt_no=1, stage="quality_gate",
                status="succeeded",
                score=quality_audit.get("score"),
                threshold=quality_audit.get("threshold"),
                blocker_count=len(quality_audit.get("blocking_issues", [])),
                warning_count=len(quality_audit.get("warnings", [])),
                warnings=quality_audit.get("warnings"),
            )

            n_charts = len(blueprint.charts)
            results.append({
                "post_id": str(post_id),
                "topic_type": blueprint.topic_type,
                "slug": blueprint.slug,
                "charts": n_charts,
            })
            _apply_topic_scope(
                topic_type,
                topic_ctx,
                seen_vendors=run_seen_vendors,
                seen_types=run_seen_types,
            )

            if len(results) >= max_posts:
                break

    if not results:
        return {
            "_skip_synthesis": "No B2B blog posts generated this run",
            **_batch_payload(),
        }

    slugs = ", ".join(r["slug"] for r in results)
    msg = f"B2B Blog: {len(results)} draft(s) created -- {slugs}"
    await send_pipeline_notification(
        msg, task, title="Atlas: B2B Blog Post Drafts",
        default_tags="brain,newspaper",
    )

    return {
        "_skip_synthesis": msg,
        "posts": results,
        "count": len(results),
        **_batch_payload(),
    }


# -- Regeneration Mode ---------------------------------------------

async def _regenerate_existing_posts(
    pool, llm, cfg, task, max_posts: int
) -> dict[str, Any]:
    """Re-process existing draft posts through the fixed pipeline.

    Queries blog_posts with status='draft' ordered by topic_type priority
    (showdowns and deep dives first).  Uses the topic_ctx stored in
    data_context to reconstruct blueprints.  Posts without stored topic_ctx
    (legacy) are skipped.
    """
    from ...pipelines.notify import send_pipeline_notification
    run_id = _task_run_id(task) or str(task.id)

    scoped_vendors = [
        value.lower()
        for value in _normalize_string_scope((task.metadata or {}).get("test_vendors"))
    ]
    scoped_topic_types = _normalize_string_scope((task.metadata or {}).get("test_topic_types"))
    scoped_slugs = _normalize_string_scope((task.metadata or {}).get("test_slugs"))

    params: list[Any] = [list(_B2B_TOPIC_TYPES)]
    where_clauses = [
        "status = 'draft'",
        f"topic_type = ANY(${len(params)}::text[])",
    ]
    if scoped_vendors:
        params.append(scoped_vendors)
        where_clauses.append(
            "LOWER(COALESCE(data_context->>'vendor_name', data_context->>'vendor_a', "
            f"data_context->>'vendor', '')) = ANY(${len(params)}::text[])"
        )
    if scoped_topic_types:
        params.append(scoped_topic_types)
        where_clauses.append(f"topic_type = ANY(${len(params)}::text[])")
    if scoped_slugs:
        params.append(scoped_slugs)
        where_clauses.append(f"slug = ANY(${len(params)}::text[])")
    params.append(max_posts)

    rows = await pool.fetch(
        f"""
        SELECT id, slug, topic_type, data_context, created_at
        FROM blog_posts
        WHERE {' AND '.join(where_clauses)}
        ORDER BY
            CASE topic_type
                WHEN 'vendor_showdown' THEN 1
                WHEN 'vendor_deep_dive' THEN 2
                WHEN 'churn_report' THEN 3
                WHEN 'pricing_reality_check' THEN 4
                WHEN 'vendor_alternative' THEN 5
                WHEN 'switching_story' THEN 6
                WHEN 'migration_guide' THEN 7
                WHEN 'market_landscape' THEN 8
                WHEN 'pain_point_roundup' THEN 9
                WHEN 'best_fit_guide' THEN 10
                ELSE 11
            END,
            created_at ASC
        LIMIT ${len(params)}
        """,
        *params,
    )

    if not rows:
        return {"_skip_synthesis": "No draft posts to regenerate"}

    results: list[dict[str, Any]] = []
    for row in rows:
        slug = row["slug"]
        topic_type = row["topic_type"]
        old_ctx = row["data_context"] or {}
        if isinstance(old_ctx, str):
            try:
                old_ctx = json.loads(old_ctx)
            except (json.JSONDecodeError, TypeError):
                old_ctx = {}

        try:
            stored_ctx = old_ctx.get("topic_ctx")
            if not stored_ctx or not isinstance(stored_ctx, dict):
                logger.warning("Regen: no stored topic_ctx for %s, skipping", slug)
                continue
            topic_ctx = {**stored_ctx, "slug": slug}

            data = await _gather_data(pool, topic_type, topic_ctx)
            await _load_pool_layers_for_blog(pool, topic_type, topic_ctx, data)
            sufficiency = _check_data_sufficiency(topic_type, data)
            if not sufficiency["sufficient"]:
                logger.warning("Regen: data insufficiency for %s: %s", slug, sufficiency["reason"])
                continue
            blueprint = _build_blueprint(topic_type, topic_ctx, data)
            blueprint_sufficiency = _check_blueprint_sufficiency(blueprint)
            if not blueprint_sufficiency["sufficient"]:
                logger.warning(
                    "Regen: blueprint insufficiency for %s: %s",
                    slug,
                    blueprint_sufficiency["reason"],
                )
                continue
            link_posts = await _fetch_related_for_linking(
                pool, blueprint.tags, blueprint.slug,
            )
            content = await _generate_content_async(
                llm, blueprint, cfg.blog_post_max_tokens,
                related_posts=link_posts,
                run_id=run_id,
            )
            if content is None:
                logger.warning("Regen: LLM failed for %s, skipping", slug)
                await _upsert_blog_post_state(
                    pool,
                    blueprint,
                    llm,
                    status="failed",
                    run_id=run_id,
                    attempt_no=1,
                    failure_step="llm_call",
                    error_code="llm_returned_none",
                    error_summary="LLM returned None",
                )
                continue

            content, quality_report = await _enforce_blog_quality_async(
                llm,
                blueprint,
                content,
                cfg.blog_post_max_tokens,
                related_posts=link_posts,
                run_id=run_id,
            )
            _persist_first_pass_blog_quality(blueprint, quality_report)
            if content is None:
                rejected_content = quality_report.get("_rejected_content") if isinstance(quality_report, dict) else None
                quality_audit = _canonicalize_blog_quality(
                    blueprint,
                    content=rejected_content if isinstance(rejected_content, dict) else None,
                    report=quality_report,
                    boundary="generation",
                )
                logger.warning(
                    "Regen quality gate failed for %s: %s",
                    slug,
                    ", ".join(quality_audit.get("blocking_issues", [])[:4]) or "unknown issues",
                )
                await _upsert_blog_post_state(
                    pool,
                    blueprint,
                    llm,
                    status="rejected",
                    run_id=run_id,
                    attempt_no=1,
                    score=quality_audit.get("score"),
                    threshold=quality_audit.get("threshold"),
                    blocker_count=len(quality_audit.get("blocking_issues", [])),
                    warning_count=len(quality_audit.get("warnings", [])),
                    failure_step="quality_gate",
                    error_code="quality_gate_rejection",
                    error_summary=", ".join(quality_audit.get("blocking_issues", [])[:3]),
                    rejection_reason=", ".join(quality_audit.get("blocking_issues", [])[:3]),
                    content=rejected_content if isinstance(rejected_content, dict) else None,
                )
                continue
            _canonicalize_blog_quality(
                blueprint,
                content=content,
                report=quality_report,
                boundary="generation",
            )

            post_id = await _assemble_and_store(
                pool,
                blueprint,
                content,
                llm,
                run_id=run_id,
                attempt_no=1,
            )
            if post_id:
                results.append({
                    "post_id": str(post_id),
                    "topic_type": topic_type,
                    "slug": slug,
                    "charts": len(blueprint.charts),
                    "regenerated": True,
                })
                logger.info("Regenerated post: slug=%s", slug)
        except Exception:
            logger.exception("Regen failed for slug=%s", slug)

    if not results:
        return {"_skip_synthesis": "No posts regenerated this run"}

    slugs = ", ".join(r["slug"] for r in results)
    msg = f"B2B Blog Regen: {len(results)} post(s) regenerated -- {slugs}"
    await send_pipeline_notification(
        msg, task, title="Atlas: B2B Blog Regeneration",
        default_tags="brain,recycle",
    )

    return {
        "_skip_synthesis": msg,
        "posts": results,
        "count": len(results),
        "regenerated": True,
    }


# -- Stage 1: Topic Selection -------------------------------------

_CAMPAIGN_GAP_LOOKBACK_DAYS = 14
_CAMPAIGN_GAP_BONUS_PCT = 0.30
_OUTBOUND_SHOWDOWN_GAP_BONUS_PCT = 0.75
_OUTBOUND_SHOWDOWN_CANDIDATE_LIMIT = 50
_OUTBOUND_SHOWDOWN_MAX_SCORE = 100
_B2B_TOPIC_TYPES = (
    "vendor_alternative",
    "vendor_showdown",
    "churn_report",
    "migration_guide",
    "vendor_deep_dive",
    "market_landscape",
    "pricing_reality_check",
    "switching_story",
    "pain_point_roundup",
    "best_fit_guide",
)


def _normalize_pain_label(value: Any) -> str:
    """Normalize pain labels for deterministic overlap checks."""
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _collect_pain_labels(value: Any) -> set[str]:
    """Collect normalized pain labels from nested payloads."""
    labels: set[str] = set()
    if value is None:
        return labels
    if isinstance(value, dict):
        for key in ("category", "pain_category", "top_pain", "label", "name"):
            norm = _normalize_pain_label(value.get(key))
            if len(norm) >= 3:
                labels.add(norm)
        for inner in value.values():
            labels |= _collect_pain_labels(inner)
        return labels
    if isinstance(value, list):
        for item in value:
            labels |= _collect_pain_labels(item)
        return labels
    norm = _normalize_pain_label(value)
    if len(norm) >= 3:
        labels.add(norm)
    return labels


def _extract_candidate_pain_labels(topic_type: str, ctx: dict[str, Any]) -> set[str]:
    """Extract pain labels from candidate context without whole-object substring scans."""
    labels: set[str] = set()
    for key in ("pain_category", "top_pain", "pain_categories", "pain_distribution", "pain_breakdown"):
        labels |= _collect_pain_labels(ctx.get(key))

    # Topic-type inferred pain coverage for deterministic matching.
    if topic_type == "pricing_reality_check":
        labels |= {"pricing", "price", "billing", "cost"}
    return labels


def _candidate_overlaps_gap_pain(
    topic_type: str,
    ctx: dict[str, Any],
    gap_pains: set[str],
) -> bool:
    """Return True when a candidate explicitly overlaps one of the gap pain labels."""
    candidate_labels = _extract_candidate_pain_labels(topic_type, ctx)
    if not candidate_labels:
        return False
    for pain in gap_pains:
        norm = _normalize_pain_label(pain)
        if len(norm) < 3:
            continue
        if norm in candidate_labels:
            return True
        if any(norm in cand or cand in norm for cand in candidate_labels):
            return True
    return False


def _extract_blog_coverage_vendors(
    data_context: dict[str, Any],
    topic_ctx: dict[str, Any],
) -> set[str]:
    """Extract all vendor identities that a blog post covers."""
    vendors: set[str] = set()
    for key in ("vendor", "vendor_a", "vendor_b", "from_vendor", "to_vendor"):
        for source in (data_context, topic_ctx):
            value = str(source.get(key) or "").strip().lower()
            if value:
                vendors.add(value)
    return vendors


def _normalize_vendor_pair(
    vendor_a: Any,
    vendor_b: Any,
) -> tuple[str, str] | None:
    """Return a stable vendor pair ordering for deterministic showdown keys."""
    left = str(vendor_a or "").strip()
    right = str(vendor_b or "").strip()
    if not left or not right:
        return None
    ordered = sorted((left, right), key=lambda item: item.lower())
    if ordered[0].lower() == ordered[1].lower():
        return None
    return ordered[0], ordered[1]


def _showdown_pair_key(vendor_a: Any, vendor_b: Any) -> str:
    pair = _normalize_vendor_pair(vendor_a, vendor_b)
    if not pair:
        return ""
    return f"{pair[0].lower()}::{pair[1].lower()}"


def _normalized_text_contains_term(text: Any, term: Any) -> bool:
    haystack = f" {_normalize_pain_label(text)} "
    needle = _normalize_pain_label(term)
    if len(needle) < 2:
        return False
    return f" {needle} " in haystack


def _blog_post_covers_showdown_pair(
    post: dict[str, Any],
    vendor_a: Any,
    vendor_b: Any,
) -> bool:
    """Return True when a blog post already covers a specific showdown pair."""
    if str(post.get("topic_type") or "").strip() != "vendor_showdown":
        return False

    data_context = post.get("data_context")
    if isinstance(data_context, str):
        try:
            data_context = json.loads(data_context)
        except Exception:
            data_context = {}
    if not isinstance(data_context, dict):
        data_context = {}
    topic_ctx = data_context.get("topic_ctx") if isinstance(data_context.get("topic_ctx"), dict) else {}

    pair_key = _showdown_pair_key(vendor_a, vendor_b)
    covered_key = _showdown_pair_key(
        topic_ctx.get("vendor_a") or data_context.get("vendor_a"),
        topic_ctx.get("vendor_b") or data_context.get("vendor_b"),
    )
    if pair_key and pair_key == covered_key:
        return True

    text = " ".join(
        str(post.get(field) or "").strip()
        for field in ("title", "slug", "url")
    )
    return (
        _normalized_text_contains_term(text, vendor_a)
        and _normalized_text_contains_term(text, vendor_b)
    )


def _blog_post_showdown_pair_key(post: dict[str, Any]) -> str:
    """Extract a normalized showdown pair key from stored blog post context."""
    data_context = post.get("data_context")
    if isinstance(data_context, str):
        try:
            data_context = json.loads(data_context)
        except Exception:
            data_context = {}
    if not isinstance(data_context, dict):
        data_context = {}
    topic_ctx = data_context.get("topic_ctx") if isinstance(data_context.get("topic_ctx"), dict) else {}
    return _showdown_pair_key(
        topic_ctx.get("vendor_a") or data_context.get("vendor_a"),
        topic_ctx.get("vendor_b") or data_context.get("vendor_b"),
    )


def _canonicalize_showdown_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize showdown pair ordering while keeping metrics attached to the right vendor."""
    vendor_a = str(candidate.get("vendor_a") or "").strip()
    vendor_b = str(candidate.get("vendor_b") or "").strip()
    pair = _normalize_vendor_pair(vendor_a, vendor_b)
    if not pair:
        return None

    normalized = dict(candidate)
    if pair != (vendor_a, vendor_b):
        normalized["vendor_a"], normalized["vendor_b"] = pair
        normalized["reviews_a"], normalized["reviews_b"] = (
            candidate.get("reviews_b"),
            candidate.get("reviews_a"),
        )
        normalized["urgency_a"], normalized["urgency_b"] = (
            candidate.get("urgency_b"),
            candidate.get("urgency_a"),
        )
    else:
        normalized["vendor_a"], normalized["vendor_b"] = pair
    return normalized


def _merge_showdown_candidates(
    organic: list[dict[str, Any]],
    outbound: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge organic showdown rows with outbound-driven pair demand."""
    merged: dict[str, dict[str, Any]] = {}

    for source_rows in (organic, outbound):
        for row in source_rows:
            normalized = _canonicalize_showdown_candidate(row)
            if not normalized:
                continue
            key = _showdown_pair_key(normalized["vendor_a"], normalized["vendor_b"])
            existing = merged.get(key)
            if not existing:
                merged[key] = normalized
                continue

            combined = dict(existing)
            for field in ("category", "slug"):
                if not combined.get(field) and normalized.get(field):
                    combined[field] = normalized[field]
            for field in ("reviews_a", "reviews_b", "total_reviews", "urgency_a", "urgency_b", "pain_diff"):
                left = combined.get(field)
                right = normalized.get(field)
                if left in (None, ""):
                    combined[field] = right
                elif right not in (None, ""):
                    combined[field] = max(left, right)
            if normalized.get("outbound_gap_priority"):
                combined["outbound_gap_priority"] = True
            if normalized.get("outbound_gap_company_count"):
                combined["outbound_gap_company_count"] = max(
                    int(combined.get("outbound_gap_company_count") or 0),
                    int(normalized.get("outbound_gap_company_count") or 0),
                )
            companies = [
                str(name).strip()
                for name in (combined.get("outbound_gap_companies") or [])
                + (normalized.get("outbound_gap_companies") or [])
                if str(name).strip()
            ]
            if companies:
                combined["outbound_gap_companies"] = list(dict.fromkeys(companies))[:5]
            merged[key] = combined

    return list(merged.values())


async def _fetch_existing_showdown_posts(pool) -> list[dict[str, Any]]:
    """Return existing showdown posts so the selector can avoid duplicate pair coverage."""
    return await pool.fetch(
        """
        SELECT title, slug, topic_type, data_context
        FROM blog_posts
        WHERE status = ANY($1::text[])
          AND topic_type = 'vendor_showdown'
        """,
        ["draft", "published"],
    )


async def _fetch_outbound_review_queue_candidates(pool) -> list[dict[str, Any]]:
    """Fetch active account-backed outbound candidates to identify missing comparison assets."""
    from .b2b_campaign_generation import list_churning_company_review_candidates

    result = await list_churning_company_review_candidates(
        pool,
        min_score=max(int(settings.b2b_campaign.review_queue_min_score), 1),
        max_score=_OUTBOUND_SHOWDOWN_MAX_SCORE,
        limit=_OUTBOUND_SHOWDOWN_CANDIDATE_LIMIT,
        qualified_only=False,
        ignore_recent_dedup=True,
    )
    return result.get("candidates") or []


async def _build_outbound_showdown_candidate(
    pool,
    *,
    vendor_a: str,
    vendor_b: str,
    category: str | None,
) -> dict[str, Any] | None:
    """Build a showdown candidate shape from outbound demand."""
    stats_a, stats_b = await asyncio.gather(
        _fetch_vendor_stats(pool, vendor_a),
        _fetch_vendor_stats(pool, vendor_b),
    )
    if not stats_a or not stats_b:
        return None

    return {
        "vendor_a": vendor_a,
        "vendor_b": vendor_b,
        "category": category or stats_a.get("category") or stats_b.get("category") or "software",
        "reviews_a": int(stats_a.get("total") or 0),
        "reviews_b": int(stats_b.get("total") or 0),
        "total_reviews": int(stats_a.get("total") or 0) + int(stats_b.get("total") or 0),
        "urgency_a": round(float(stats_a.get("avg_urgency") or 0), 1),
        "urgency_b": round(float(stats_b.get("avg_urgency") or 0), 1),
        "pain_diff": round(
            abs(float(stats_a.get("avg_urgency") or 0) - float(stats_b.get("avg_urgency") or 0)),
            1,
        ),
    }


async def _find_outbound_showdown_gap_candidates(pool) -> list[dict[str, Any]]:
    """Return missing incumbent-vs-alternative showdowns from the live outbound queue."""
    existing_posts, review_candidates = await asyncio.gather(
        _fetch_existing_showdown_posts(pool),
        _fetch_outbound_review_queue_candidates(pool),
    )
    covered_pairs = {
        _blog_post_showdown_pair_key(post)
        for post in existing_posts
        if _blog_post_showdown_pair_key(post)
    }

    pair_demand: dict[str, dict[str, Any]] = {}
    for item in review_candidates:
        comparison_asset = item.get("comparison_asset") or {}
        incumbent = str(
            comparison_asset.get("incumbent_vendor") or item.get("vendor_name") or ""
        ).strip()
        alternative = str(comparison_asset.get("alternative_vendor") or "").strip()
        if not incumbent or not alternative:
            continue
        if not comparison_asset.get("company_safe"):
            continue
        if not comparison_asset.get("pain_categories"):
            continue

        pair_key = _showdown_pair_key(incumbent, alternative)
        if not pair_key or pair_key in covered_pairs:
            continue

        primary_post = comparison_asset.get("primary_blog_post") or item.get("primary_blog_post") or {}
        if isinstance(primary_post, dict) and _blog_post_covers_showdown_pair(primary_post, incumbent, alternative):
            continue

        company_name = str(item.get("company_name") or "").strip()
        current = pair_demand.setdefault(
            pair_key,
            {
                "vendor_a": incumbent,
                "vendor_b": alternative,
                "category": str(item.get("product_category") or "").strip(),
                "companies": [],
                "max_score": 0,
            },
        )
        if company_name and company_name not in current["companies"]:
            current["companies"].append(company_name)
        current["max_score"] = max(current["max_score"], int(item.get("opportunity_score") or 0))

    candidates: list[dict[str, Any]] = []
    for demand in pair_demand.values():
        built = await _build_outbound_showdown_candidate(
            pool,
            vendor_a=demand["vendor_a"],
            vendor_b=demand["vendor_b"],
            category=demand.get("category"),
        )
        if not built:
            continue
        built["outbound_gap_priority"] = True
        built["outbound_gap_company_count"] = len(demand["companies"])
        built["outbound_gap_companies"] = demand["companies"][:5]
        built["outbound_gap_max_score"] = demand["max_score"]
        candidates.append(built)

    return candidates


async def _detect_campaign_content_gaps(pool) -> dict[str, set[str]]:
    """Find (vendor, pain) pairs with recent campaigns but no matching blog post.

    Returns ``{vendor_lower: {pain_lower, ...}}`` for vendors that have
    campaigns in the last N days but no draft/published blog post mentioning
    both the vendor and that pain category.
    """
    try:
        rows = await pool.fetch(
            """
            SELECT LOWER(vendor_name) AS vendor,
                   LOWER(
                       COALESCE(p.value->>'category', p.value #>> '{}')
                   ) AS pain
            FROM b2b_campaigns,
                 LATERAL jsonb_array_elements(
                     CASE WHEN jsonb_typeof(pain_categories) = 'array'
                          THEN pain_categories
                          ELSE '[]'::jsonb END
                 ) AS p(value)
            WHERE created_at >= NOW() - make_interval(days => $1)
              AND status NOT IN ('cancelled', 'failed')
              AND vendor_name IS NOT NULL
            GROUP BY 1, 2
            HAVING count(*) >= 1
            """,
            _CAMPAIGN_GAP_LOOKBACK_DAYS,
        )
    except Exception:
        logger.warning("Failed to query campaign content gaps", exc_info=True)
        return {}

    if not rows:
        return {}

    # Build campaign demand: {vendor: {pain, ...}}
    demand: dict[str, set[str]] = {}
    for r in rows:
        vendor = (r["vendor"] or "").strip()
        pain = (r["pain"] or "").strip()
        if vendor and pain:
            demand.setdefault(vendor, set()).add(pain)

    if not demand:
        return {}

    # Check which (vendor, pain) pairs already have blog coverage
    try:
        covered_rows = await pool.fetch(
            """
            SELECT title, slug, topic_type, tags, data_context
            FROM blog_posts
            WHERE status = ANY($1::text[])
              AND topic_type = ANY($2::text[])
            """,
            ["draft", "published"],
            list(_B2B_TOPIC_TYPES),
        )
    except Exception:
        logger.warning("Failed to query blog coverage for gap detection", exc_info=True)
        return demand  # Assume no coverage -- treat all as gaps

    # Remove covered pairs
    for cr in covered_rows:
        dc = cr.get("data_context")
        if isinstance(dc, str):
            try:
                dc = json.loads(dc)
            except Exception:
                dc = {}
        if not isinstance(dc, dict):
            dc = {}
        tc = dc.get("topic_ctx") if isinstance(dc.get("topic_ctx"), dict) else {}
        covered_vendors = _extract_blog_coverage_vendors(dc, tc)
        covered_candidates = covered_vendors & set(demand.keys())
        if not covered_candidates:
            continue
        pain_labels: set[str] = set()
        for key in ("pain_distribution", "pain_breakdown", "pain_categories", "top_pain"):
            pain_labels |= _collect_pain_labels(dc.get(key))
            pain_labels |= _collect_pain_labels(tc.get(key))
        tags = cr.get("tags") or []
        if isinstance(tags, list):
            tags_text = " ".join(str(tag).lower() for tag in tags if tag)
        else:
            tags_text = str(tags).lower()
        text_haystack = " ".join(
            filter(
                None,
                [
                    str(cr.get("title") or "").lower(),
                    str(cr.get("slug") or "").lower(),
                    str(cr.get("topic_type") or "").lower(),
                    tags_text,
                ],
            )
        )
        for vendor in covered_candidates:
            covered_pains: set[str] = set()
            for pain in demand.get(vendor, set()):
                norm = _normalize_pain_label(pain)
                if len(norm) < 3:
                    continue
                if norm in pain_labels or norm in text_haystack:
                    covered_pains.add(pain)
            demand[vendor] -= covered_pains
            if not demand[vendor]:
                del demand[vendor]

    return demand


# ---------------------------------------------------------------------------
# Reasoning-aware topic reranker (Phase 7)
# ---------------------------------------------------------------------------

# Score adjustments for reasoning-backed reranking
_REASONING_TIMING_BOOST = 10.0       # topic has active timing intelligence
_REASONING_ACCOUNT_BOOST = 8.0       # topic has account-level intent signals
_REASONING_SWITCH_TRIGGER_BOOST = 6.0  # displacement has switch triggers
_REASONING_COVERAGE_GAP_PENALTY = -15.0  # evidence is thin
_REASONING_RETENTION_BOOST = 3.0     # why_they_stay available (enables balanced copy)
_REASONING_CONTRADICTION_PENALTY = -5.0  # contradictory evidence reduces confidence


async def _rerank_topic_candidates_with_reasoning(
    pool,
    candidates: list[tuple[str, float, str, dict[str, Any]]],
    *,
    as_of: date | None = None,
    analysis_window_days: int = 90,
) -> list[tuple[str, float, str, dict[str, Any]]]:
    """Rerank topic candidates using synthesis reasoning signals.

    Adjusts normalized scores based on reasoning quality so topics backed
    by strong synthesis outrank raw-volume topics, and thin-evidence topics
    are deprioritized even when deterministic scores are high.
    """
    if not candidates:
        return candidates

    # Collect vendor names and category names from candidates
    vendor_set: set[str] = set()
    category_set: set[str] = set()
    for _, _, _, ctx in candidates:
        for key in ("vendor", "from_vendor", "vendor_a", "vendor_b"):
            v = str(ctx.get(key) or "").strip()
            if v:
                vendor_set.add(v)
        cat = str(ctx.get("category") or "").strip()
        if cat:
            category_set.add(cat)

    from ._b2b_synthesis_reader import load_best_reasoning_views

    # For category-only candidates, resolve categories to vendor names
    vendor_to_category: dict[str, str] = {}
    if category_set:
        cat_vendors_needed = category_set - {v.lower() for v in vendor_set}
        if cat_vendors_needed:
            try:
                from ._b2b_shared import read_category_vendor_rows

                _as_of_filter = as_of or date.today()
                cat_vendor_rows = await read_category_vendor_rows(
                    pool,
                    category_names=cat_vendors_needed,
                )
                for row in cat_vendor_rows:
                    vn = row.get("vendor_name")
                    pc = row.get("product_category")
                    if vn:
                        vendor_set.add(vn)
                        if pc:
                            vendor_to_category[vn] = pc.lower().strip()
            except Exception:
                logger.debug("Category vendor resolution failed", exc_info=True)

    views: dict = {}
    if vendor_set:
        try:
            views = await load_best_reasoning_views(
                pool, list(vendor_set),
                as_of=as_of,
                analysis_window_days=analysis_window_days,
            )
        except Exception:
            logger.debug("Reasoning reranker: synthesis lookup failed, skipping", exc_info=True)
            return candidates

    # Build category index from loaded views
    category_views: dict[str, list] = {}
    for vname, view in views.items():
        cat_name = ""
        # Source 1: DB-resolved category from vendor resolution query
        cat_name = vendor_to_category.get(vname, "")
        # Source 2: evidence_vault product_category
        if not cat_name:
            ev = view.raw.get("evidence_vault") or {}
            if isinstance(ev, dict):
                cat_name = str(ev.get("product_category") or "").strip()
        # Source 3: category_reasoning fields
        if not cat_name:
            cat_contract = view.contract("category_reasoning")
            if isinstance(cat_contract, dict):
                for cat_key in ("category", "product_category"):
                    cat_name = str(cat_contract.get(cat_key) or "").strip()
                    if cat_name:
                        break
        if cat_name:
            category_views.setdefault(cat_name.lower().strip(), []).append(view)

    if not views and not category_views:
        return candidates

    reranked: list[tuple[str, float, str, dict[str, Any]]] = []
    for slug, score, topic_type, ctx in candidates:
        adjustment = 0.0
        reasons: list[str] = []

        # Get primary vendor's reasoning view
        primary = (
            str(ctx.get("vendor") or ctx.get("from_vendor") or "").strip()
            or str(ctx.get("vendor_a") or "").strip()
        )
        view = views.get(primary) if primary else None

        if view is not None:
            # Timing intelligence boost
            timing = view.section("timing_intelligence")
            if timing and timing.get("best_timing_window"):
                triggers = timing.get("immediate_triggers") or []
                if triggers:
                    adjustment += _REASONING_TIMING_BOOST
                    reasons.append("timing_boost")

            # Account-level intent boost
            acct = view.contract("account_reasoning")
            if acct and acct.get("market_summary"):
                high_intent = acct.get("high_intent_count")
                if isinstance(high_intent, dict):
                    high_intent = high_intent.get("value")
                if high_intent and int(high_intent or 0) > 0:
                    adjustment += _REASONING_ACCOUNT_BOOST
                    reasons.append("account_intent_boost")

            # Switch trigger boost
            triggers = view.switch_triggers
            if triggers:
                adjustment += _REASONING_SWITCH_TRIGGER_BOOST
                reasons.append("switch_trigger_boost")

            # Why they stay boost (enables balanced copy)
            if view.why_they_stay:
                adjustment += _REASONING_RETENTION_BOOST
                reasons.append("retention_context")

            # Coverage gap penalty
            gaps = view.coverage_gaps
            if gaps:
                # Scale penalty by number of gaps
                gap_penalty = _REASONING_COVERAGE_GAP_PENALTY * min(len(gaps), 3) / 3
                adjustment += gap_penalty
                reasons.append(f"coverage_gap_penalty({len(gaps)})")

            # Contradiction penalty
            contradictions = view.contradictions
            if contradictions:
                adjustment += _REASONING_CONTRADICTION_PENALTY
                reasons.append("contradiction_penalty")

        # For showdowns, apply symmetric reasoning for vendor_b at half weight
        vendor_b = str(ctx.get("vendor_b") or "").strip()
        view_b = views.get(vendor_b) if vendor_b else None
        if view_b is not None:
            _SECONDARY_WEIGHT = 0.5
            timing_b = view_b.section("timing_intelligence")
            if timing_b and (timing_b.get("immediate_triggers") or []):
                adjustment += _REASONING_TIMING_BOOST * _SECONDARY_WEIGHT
                reasons.append("vendor_b_timing_boost")
            if view_b.switch_triggers:
                adjustment += _REASONING_SWITCH_TRIGGER_BOOST * _SECONDARY_WEIGHT
                reasons.append("vendor_b_trigger_boost")
            if view_b.why_they_stay:
                adjustment += _REASONING_RETENTION_BOOST * _SECONDARY_WEIGHT
                reasons.append("vendor_b_retention_context")
            gaps_b = view_b.coverage_gaps
            if gaps_b:
                adjustment += _REASONING_COVERAGE_GAP_PENALTY * min(len(gaps_b), 3) / 6
                reasons.append(f"vendor_b_gap_penalty({len(gaps_b)})")
            if view_b.contradictions:
                adjustment += _REASONING_CONTRADICTION_PENALTY * _SECONDARY_WEIGHT
                reasons.append("vendor_b_contradiction_penalty")

        # Category-level reasoning for topics without vendor keys
        if view is None and view_b is None:
            cat_name = str(ctx.get("category") or "").lower().strip()
            cat_matches = category_views.get(cat_name, [])
            if cat_matches:
                # Use the best-informed category view
                best_cat_view = cat_matches[0]
                cat_contract = best_cat_view.contract("category_reasoning")
                if isinstance(cat_contract, dict):
                    regime = cat_contract.get("market_regime", "")
                    if regime:
                        adjustment += 4.0  # category has regime context
                        reasons.append("category_regime_boost")
                    cat_confidence = cat_contract.get("confidence_score")
                    if cat_confidence is not None:
                        try:
                            if float(cat_confidence) < 0.3:
                                adjustment += _REASONING_COVERAGE_GAP_PENALTY / 3
                                reasons.append("category_low_confidence_penalty")
                        except (TypeError, ValueError):
                            pass
                cat_gaps = best_cat_view.coverage_gaps
                if cat_gaps:
                    adjustment += _REASONING_COVERAGE_GAP_PENALTY * min(len(cat_gaps), 3) / 6
                    reasons.append(f"category_gap_penalty({len(cat_gaps)})")

        new_score = max(0.0, score + adjustment)
        if reasons:
            ctx = {**ctx, "_reasoning_adjustments": reasons}
        reranked.append((slug, new_score, topic_type, ctx))

    reranked.sort(key=lambda x: x[1], reverse=True)
    boosted = sum(1 for _, _, _, c in reranked if c.get("_reasoning_adjustments"))
    if boosted:
        logger.info(
            "Reasoning reranker adjusted %d/%d candidates", boosted, len(reranked),
        )
    return reranked


async def _select_topic(
    pool,
    max_per_run: int = 1,
    *,
    exclude_vendors: set[str] | None = None,
    exclude_types: dict[str, int] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Score candidates and pick the best unwritten B2B topic."""
    today = date.today()
    month_suffix = today.strftime("%Y-%m")

    (alternatives, showdowns, outbound_showdowns, churn_reports, migrations,
     deep_dives, landscapes,
     pricing_checks, switching_stories, pain_roundups, fit_guides,
    ) = await asyncio.gather(
        _find_vendor_alternative_candidates(pool),
        _find_vendor_showdown_candidates(pool),
        _find_outbound_showdown_gap_candidates(pool),
        _find_churn_report_candidates(pool),
        _find_migration_guide_candidates(pool),
        _find_vendor_deep_dive_candidates(pool),
        _find_market_landscape_candidates(pool),
        _find_pricing_reality_check_candidates(pool),
        _find_switching_story_candidates(pool),
        _find_pain_point_roundup_candidates(pool),
        _find_best_fit_guide_candidates(pool),
        return_exceptions=True,
    )
    alternatives = alternatives if not isinstance(alternatives, Exception) else []
    showdowns = showdowns if not isinstance(showdowns, Exception) else []
    outbound_showdowns = outbound_showdowns if not isinstance(outbound_showdowns, Exception) else []
    churn_reports = churn_reports if not isinstance(churn_reports, Exception) else []
    migrations = migrations if not isinstance(migrations, Exception) else []
    deep_dives = deep_dives if not isinstance(deep_dives, Exception) else []
    landscapes = landscapes if not isinstance(landscapes, Exception) else []
    pricing_checks = pricing_checks if not isinstance(pricing_checks, Exception) else []
    switching_stories = switching_stories if not isinstance(switching_stories, Exception) else []
    pain_roundups = pain_roundups if not isinstance(pain_roundups, Exception) else []
    fit_guides = fit_guides if not isinstance(fit_guides, Exception) else []
    showdowns = _merge_showdown_candidates(showdowns, outbound_showdowns)

    raw_candidates: list[tuple[str, float, str, dict[str, Any]]] = []

    for alt in alternatives:
        slug = f"{_slugify(alt['vendor'])}-alternatives-{month_suffix}"
        score = alt["urgency"] * alt["review_count"]
        raw_candidates.append((slug, score, "vendor_alternative", {**alt, "slug": slug}))

    for pair in showdowns:
        slug = f"{_slugify(pair['vendor_a'])}-vs-{_slugify(pair['vendor_b'])}-{month_suffix}"
        # Weight reviews heavily -- popular pairs are most interesting to readers.
        # pain_diff is a bonus, not the primary driver.
        score = (pair["total_reviews"] + pair["pain_diff"] * 50) * 1.5
        raw_candidates.append((slug, score, "vendor_showdown", {**pair, "slug": slug}))

    for cr in churn_reports:
        slug = f"{_slugify(cr['vendor'])}-churn-report-{month_suffix}"
        score = cr["negative_reviews"] * cr["avg_urgency"]
        raw_candidates.append((slug, score, "churn_report", {**cr, "slug": slug}))

    for mig in migrations:
        slug = f"switch-to-{_slugify(mig['vendor'])}-{month_suffix}"
        score = mig["switch_count"] * mig["review_total"] * 1.5
        raw_candidates.append((slug, score, "migration_guide", {**mig, "slug": slug}))

    for dd in deep_dives:
        slug = f"{_slugify(dd['vendor'])}-deep-dive-{month_suffix}"
        score = dd["review_count"] * 1.5 + dd["profile_richness"] * 5
        raw_candidates.append((slug, score, "vendor_deep_dive", {**dd, "slug": slug}))

    for ml in landscapes:
        slug = f"{_slugify(ml['category'])}-landscape-{month_suffix}"
        score = ml["vendor_count"] * ml["total_reviews"] * 0.5
        raw_candidates.append((slug, score, "market_landscape", {**ml, "slug": slug}))

    for pc in pricing_checks:
        slug = f"real-cost-of-{_slugify(pc['vendor'])}-{month_suffix}"
        score = pc["pricing_complaints"] * 10 + pc["total_reviews"] * 0.5
        raw_candidates.append((slug, score, "pricing_reality_check", {**pc, "slug": slug}))

    for ss in switching_stories:
        slug = f"why-teams-leave-{_slugify(ss['from_vendor'])}-{month_suffix}"
        score = ss["switch_mentions"] * 8 + ss["avg_urgency"] * 2
        raw_candidates.append((slug, score, "switching_story", {**ss, "slug": slug}))

    for pr in pain_roundups:
        slug = f"top-complaint-every-{_slugify(pr['category'])}-{month_suffix}"
        score = pr["vendor_count"] * pr["total_complaints"] * 0.3
        raw_candidates.append((slug, score, "pain_point_roundup", {**pr, "slug": slug}))

    for fg in fit_guides:
        slug = f"best-{_slugify(fg['category'])}-for-{_slugify(fg['company_size'])}-{month_suffix}"
        score = fg["vendor_count"] * fg["total_reviews"] * 0.8
        raw_candidates.append((slug, score, "best_fit_guide", {**fg, "slug": slug}))

    if not raw_candidates:
        return None

    # --- Normalize scores within each topic type (0-100 scale) ---
    # Without normalization, deep_dives (score ~900) always beat showdowns (~400).
    # Normalizing ensures the *best* candidate of each type competes fairly.
    by_type: dict[str, list[tuple[str, float, str, dict]]] = {}
    for slug, score, topic_type, ctx in raw_candidates:
        by_type.setdefault(topic_type, []).append((slug, score, topic_type, ctx))

    normalized: list[tuple[str, float, str, dict]] = []
    for topic_type, entries in by_type.items():
        max_score = max(e[1] for e in entries) or 1.0
        for slug, score, tt, ctx in entries:
            norm = (score / max_score) * 100.0
            normalized.append((slug, norm, tt, ctx))

    raw_candidates = normalized

    # --- Campaign content gap bonus: boost topics filling campaign demand ---
    content_gaps = await _detect_campaign_content_gaps(pool)
    if content_gaps:
        boosted: list[tuple[str, float, str, dict]] = []
        gap_boost_count = 0
        for slug, score, topic_type, ctx in raw_candidates:
            vendor_keys = {
                str(ctx.get(key) or "").lower().strip()
                for key in ("vendor", "from_vendor", "vendor_a", "vendor_b")
                if str(ctx.get(key) or "").strip()
            }
            should_boost = False
            for vendor_key in vendor_keys:
                gap_pains = content_gaps.get(vendor_key)
                if gap_pains and _candidate_overlaps_gap_pain(topic_type, ctx, gap_pains):
                    should_boost = True
                    break
            if should_boost:
                score *= (1.0 + _CAMPAIGN_GAP_BONUS_PCT)
                gap_boost_count += 1
            boosted.append((slug, score, topic_type, ctx))
        raw_candidates = boosted
        if gap_boost_count:
            logger.info(
                "Campaign gap bonus applied to %d/%d candidates",
                gap_boost_count, len(raw_candidates),
            )

    boosted_showdowns: list[tuple[str, float, str, dict]] = []
    showdown_gap_boost_count = 0
    for slug, score, topic_type, ctx in raw_candidates:
        if topic_type == "vendor_showdown" and ctx.get("outbound_gap_priority"):
            score *= (1.0 + _OUTBOUND_SHOWDOWN_GAP_BONUS_PCT)
            showdown_gap_boost_count += 1
        boosted_showdowns.append((slug, score, topic_type, ctx))
    raw_candidates = boosted_showdowns
    if showdown_gap_boost_count:
        logger.info(
            "Outbound showdown gap bonus applied to %d candidate(s)",
            showdown_gap_boost_count,
        )

    # --- Reasoning-aware reranking (Phase 7) ---
    raw_candidates = await _rerank_topic_candidates_with_reasoning(
        pool, raw_candidates, as_of=today,
    )

    # --- Data sufficiency gate: filter candidates below minimum review counts ---
    sources = _blog_source_allowlist()
    vendor_counts = await _batch_vendor_review_counts(pool, raw_candidates, sources)
    _MIN_REVIEWS_BY_TYPE = {
        "vendor_showdown": 10,
        "vendor_deep_dive": 8,
        "churn_report": 8,
        "best_fit_guide": 8,
    }
    _DEFAULT_MIN_REVIEWS = 5
    before_sufficiency = len(raw_candidates)
    sufficient: list[tuple[str, float, str, dict]] = []
    for slug, score, topic_type, ctx in raw_candidates:
        min_required = _MIN_REVIEWS_BY_TYPE.get(topic_type, _DEFAULT_MIN_REVIEWS)
        # For showdowns, check both vendors
        if topic_type == "vendor_showdown":
            va = (ctx.get("vendor_a") or "").lower().strip()
            vb = (ctx.get("vendor_b") or "").lower().strip()
            if vendor_counts.get(va, 0) < min_required or vendor_counts.get(vb, 0) < min_required:
                logger.debug(
                    "Sufficiency gate: %s skipped (%s=%d, %s=%d, need %d)",
                    slug, va, vendor_counts.get(va, 0), vb, vendor_counts.get(vb, 0), min_required,
                )
                continue
        else:
            vk = (ctx.get("vendor") or ctx.get("from_vendor") or "").lower().strip()
            if vk and vendor_counts.get(vk, 0) < min_required:
                logger.debug(
                    "Sufficiency gate: %s skipped (vendor=%s, count=%d, need %d)",
                    slug, vk, vendor_counts.get(vk, 0), min_required,
                )
                continue
        sufficient.append((slug, score, topic_type, ctx))
    raw_candidates = sufficient
    if before_sufficiency != len(raw_candidates):
        logger.info(
            "Sufficiency gate filtered %d -> %d candidates",
            before_sufficiency, len(raw_candidates),
        )

    if not raw_candidates:
        return None

    # --- Dedup layer 1: exact slug match (same topic+vendor+month) ---
    all_slugs = list({c[0] for c in raw_candidates})
    existing_slugs = await _batch_slug_check(pool, all_slugs)

    # --- Dedup layer 2: vendor-level cooldown (any topic type, 7 days) ---
    # A vendor needs 3+ posts within the window to be considered "covered",
    # allowing a deep-dive and a showdown without triggering cooldown.
    covered_vendors = await _recently_covered_vendors(pool, days=7)

    def _vendor_keys(ctx: dict) -> set[str]:
        """Return all vendor names from a candidate (normalized for dedup).

        Showdowns have vendor_a + vendor_b; others have vendor.
        """
        keys = set()
        for k in ("vendor", "vendor_a", "vendor_b", "from_vendor"):
            v = ctx.get(k, "")
            if v:
                keys.add(v.lower().strip())
        return keys

    candidates = [
        (score, topic_type, ctx)
        for slug, score, topic_type, ctx in raw_candidates
        if slug not in existing_slugs
        and (
            ctx.get("outbound_gap_priority")
            or not _vendor_keys(ctx) & covered_vendors
        )
    ]

    if not candidates:
        logger.info(
            "No candidates survived filters. raw=%d, slug_dupes=%d, covered_vendors=%d",
            len(raw_candidates), len(existing_slugs), len(covered_vendors),
        )
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    logger.info(
        "Topic candidates after filtering: %d (top: %s)",
        len(candidates),
        [(c[1], c[2].get("slug", "?"), f"{c[0]:.0f}") for c in candidates[:5]],
    )

    # --- Dedup layer 3: cross-iteration vendor + type diversity ---
    _max_per_type = 2
    _exclude_vendors = exclude_vendors or set()
    _exclude_types = exclude_types or {}
    best = None
    for score, topic_type, ctx in candidates:
        # Skip vendors already used in this run
        vks = _vendor_keys(ctx)
        if vks & _exclude_vendors:
            continue
        # Skip topic types that hit the per-run cap
        if _exclude_types.get(topic_type, 0) >= _max_per_type:
            continue
        best = (score, topic_type, ctx)
        break

    if best is None:
        return None
    logger.info(
        "Selected B2B topic: %s (score=%.1f, slug=%s)",
        best[1], best[0], best[2].get("slug"),
    )
    return best[1], best[2]


async def _find_vendor_alternative_candidates(pool) -> list[dict[str, Any]]:
    """Vendors with high churn + affiliate partner covering the category."""
    from ._b2b_shared import read_vendor_alternative_candidates

    return await read_vendor_alternative_candidates(pool)


async def _find_vendor_showdown_candidates(pool) -> list[dict[str, Any]]:
    """Pairs of vendors in the same category with contrasting pain profiles."""
    from ._b2b_shared import read_vendor_showdown_candidates

    return await read_vendor_showdown_candidates(pool)


async def _find_churn_report_candidates(pool) -> list[dict[str, Any]]:
    """Single vendor with high urgency + many negative reviews."""
    from ._b2b_shared import read_churn_report_candidates

    return await read_churn_report_candidates(pool)


async def _find_migration_guide_candidates(pool) -> list[dict[str, Any]]:
    """Vendors with high switched_from counts in product profiles."""
    rows = await pool.fetch(
        """
        SELECT
            pp.vendor_name AS vendor,
            pp.product_category AS category,
            COALESCE(jsonb_array_length(pp.commonly_switched_from), 0) AS switch_count,
            (
                SELECT COUNT(DISTINCT br.id)
                FROM b2b_reviews br
                JOIN b2b_review_vendor_mentions vm ON vm.review_id = br.id
                WHERE vm.vendor_name = pp.vendor_name
                  AND br.duplicate_of_review_id IS NULL
            ) AS review_total
        FROM b2b_product_profiles pp
        WHERE jsonb_array_length(COALESCE(pp.commonly_switched_from, '[]'::jsonb)) >= 2
        ORDER BY jsonb_array_length(pp.commonly_switched_from) DESC
        LIMIT 10
        """
    )
    return [
        {
            "vendor": r["vendor"],
            "category": r["category"],
            "switch_count": r["switch_count"],
            "review_total": r["review_total"],
        }
        for r in rows
    ]


async def _find_pricing_reality_check_candidates(pool) -> list[dict[str, Any]]:
    """Vendors where users complain about pricing -- bait-and-switch, hidden costs, etc."""
    # APPROVED-ENRICHMENT-READ: pain_categories, urgency_score
    # Reason: inline aggregate query, structurally coupled to product output
    sources = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT
            vm.vendor_name AS vendor,
            r.product_category AS category,
            COUNT(DISTINCT r.id) AS total_reviews,
            COUNT(DISTINCT r.id) FILTER (
                WHERE r.enrichment->>'pain_categories' ILIKE '%pricing%'
            ) AS pricing_complaints,
            ROUND(AVG(
                CASE WHEN r.enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (r.enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.source = ANY($1)
        GROUP BY vm.vendor_name, r.product_category
        HAVING COUNT(DISTINCT r.id) FILTER (
            WHERE r.enrichment->>'pain_categories' ILIKE '%pricing%'
        ) >= 2
        ORDER BY COUNT(DISTINCT r.id) FILTER (
            WHERE r.enrichment->>'pain_categories' ILIKE '%pricing%'
        ) DESC
        LIMIT 10
        """,
        sources,
    )
    return [
        {
            "vendor": r["vendor"],
            "category": r["category"],
            "total_reviews": r["total_reviews"],
            "pricing_complaints": r["pricing_complaints"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
        }
        for r in rows
    ]


async def _find_switching_story_candidates(pool) -> list[dict[str, Any]]:
    """Vendors users are actively leaving -- real switching stories from reviews."""
    # APPROVED-ENRICHMENT-READ: urgency_score
    # Reason: inline aggregate query, structurally coupled to product output
    sources = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT
            vm.vendor_name AS from_vendor,
            r.product_category AS category,
            COUNT(DISTINCT r.id) AS total_reviews,
            COUNT(DISTINCT r.id) FILTER (
                WHERE (r.enrichment->>'urgency_score')::numeric >= 7
            ) AS high_urgency_count,
            COUNT(DISTINCT r.id) FILTER (
                WHERE r.review_text ILIKE '%switch%' OR r.review_text ILIKE '%migrat%'
                   OR r.review_text ILIKE '%moved to%' OR r.review_text ILIKE '%moving to%'
                   OR r.review_text ILIKE '%left for%' OR r.review_text ILIKE '%leaving for%'
            ) AS switch_mentions,
            ROUND(AVG(
                CASE WHEN r.enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (r.enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.source = ANY($1)
        GROUP BY vm.vendor_name, r.product_category
        HAVING COUNT(DISTINCT r.id) FILTER (
            WHERE r.review_text ILIKE '%switch%' OR r.review_text ILIKE '%migrat%'
               OR r.review_text ILIKE '%moved to%' OR r.review_text ILIKE '%moving to%'
               OR r.review_text ILIKE '%left for%' OR r.review_text ILIKE '%leaving for%'
        ) >= 2
        ORDER BY switch_mentions DESC
        LIMIT 10
        """,
        sources,
    )
    return [
        {
            "from_vendor": r["from_vendor"],
            "category": r["category"],
            "total_reviews": r["total_reviews"],
            "high_urgency_count": r["high_urgency_count"],
            "switch_mentions": r["switch_mentions"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
        }
        for r in rows
    ]


async def _find_pain_point_roundup_candidates(pool) -> list[dict[str, Any]]:
    """Categories with enough vendors to do a cross-vendor complaint comparison."""
    # APPROVED-ENRICHMENT-READ: urgency_score
    # Reason: inline aggregate query, structurally coupled to product output
    sources = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT
            r.product_category AS category,
            COUNT(DISTINCT vm.vendor_name) AS vendor_count,
            COUNT(DISTINCT r.id) AS total_complaints,
            ROUND(AVG(
                CASE WHEN r.enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (r.enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.product_category IS NOT NULL AND r.product_category != ''
          AND r.source = ANY($1)
        GROUP BY r.product_category
        HAVING COUNT(DISTINCT vm.vendor_name) >= 3
        ORDER BY COUNT(DISTINCT vm.vendor_name) DESC
        LIMIT 10
        """,
        sources,
    )
    return [
        {
            "category": r["category"],
            "vendor_count": r["vendor_count"],
            "total_complaints": r["total_complaints"],
            "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
        }
        for r in rows
    ]


async def _find_best_fit_guide_candidates(pool) -> list[dict[str, Any]]:
    """Categories with vendors serving different company sizes -- recommend by fit."""
    min_vendor_profiles = _blog_min_vendor_profiles("best_fit_guide")
    rows = await pool.fetch(
        """
        SELECT
            pp.product_category AS category,
            COUNT(DISTINCT pp.vendor_name) AS vendor_count,
            (SELECT COUNT(*) FROM b2b_reviews br
             WHERE br.product_category = pp.product_category
               AND br.enrichment_status = 'enriched'
               AND br.duplicate_of_review_id IS NULL) AS total_reviews,
            MODE() WITHIN GROUP (
                ORDER BY COALESCE(
                    (
                        SELECT key
                        FROM jsonb_each_text(COALESCE(pp.typical_company_size, '{}'::jsonb))
                        ORDER BY value::numeric DESC
                        LIMIT 1
                    ),
                    'unknown'
                )
            ) AS dominant_size
        FROM b2b_product_profiles pp
        WHERE pp.product_category IS NOT NULL AND pp.product_category != ''
        GROUP BY pp.product_category
        HAVING COUNT(DISTINCT pp.vendor_name) >= $1
        ORDER BY COUNT(DISTINCT pp.vendor_name) DESC
        LIMIT 10
        """,
        min_vendor_profiles,
    )
    return [
        {
            "category": r["category"],
            "vendor_count": r["vendor_count"],
            "total_reviews": r["total_reviews"],
            "company_size": r["dominant_size"] or "small-teams",
        }
        for r in rows
    ]


async def _find_vendor_deep_dive_candidates(pool) -> list[dict[str, Any]]:
    """Any vendor with a product profile -- showcase what we know about them."""
    rows = await pool.fetch(
        """
        SELECT
            pp.vendor_name AS vendor,
            pp.product_category AS category,
            (
                SELECT COUNT(DISTINCT br.id)
                FROM b2b_reviews br
                JOIN b2b_review_vendor_mentions vm ON vm.review_id = br.id
                WHERE vm.vendor_name = pp.vendor_name
                  AND br.duplicate_of_review_id IS NULL
            ) AS review_count,
            (CASE
                WHEN pp.strengths IS NOT NULL AND jsonb_array_length(COALESCE(pp.strengths, '[]'::jsonb)) > 0 THEN 1 ELSE 0
            END +
            CASE
                WHEN pp.weaknesses IS NOT NULL AND jsonb_array_length(COALESCE(pp.weaknesses, '[]'::jsonb)) > 0 THEN 1 ELSE 0
            END +
            CASE
                WHEN pp.top_integrations IS NOT NULL AND jsonb_array_length(COALESCE(pp.top_integrations, '[]'::jsonb)) > 0 THEN 1 ELSE 0
            END +
            CASE
                WHEN pp.commonly_compared_to IS NOT NULL AND jsonb_array_length(COALESCE(pp.commonly_compared_to, '[]'::jsonb)) > 0 THEN 1 ELSE 0
            END +
            CASE
                WHEN pp.commonly_switched_from IS NOT NULL AND jsonb_array_length(COALESCE(pp.commonly_switched_from, '[]'::jsonb)) > 0 THEN 1 ELSE 0
            END) AS profile_richness
        FROM b2b_product_profiles pp
        ORDER BY profile_richness DESC, review_count DESC
        LIMIT 60
        """
    )
    return [
        {
            "vendor": r["vendor"],
            "category": r["category"],
            "review_count": r["review_count"],
            "profile_richness": r["profile_richness"],
        }
        for r in rows
    ]


async def _find_market_landscape_candidates(pool) -> list[dict[str, Any]]:
    """Categories with multiple vendors -- write a landscape overview."""
    from ._b2b_shared import read_market_landscape_candidates

    min_vendor_profiles = _blog_min_vendor_profiles("market_landscape")
    return await read_market_landscape_candidates(
        pool,
        min_vendor_profiles=min_vendor_profiles,
    )


async def _batch_vendor_review_counts(
    pool, candidates: list[tuple[str, float, str, dict]], sources: list[str]
) -> dict[str, int]:
    """Single SQL query counting blog-eligible reviews per vendor.

    Returns {vendor_name_lower: count}.
    """
    vendor_set: set[str] = set()
    for _, _, _, ctx in candidates:
        for k in ("vendor", "vendor_a", "vendor_b", "from_vendor"):
            v = ctx.get(k, "")
            if v:
                vendor_set.add(v)
    if not vendor_set:
        return {}
    rows = await pool.fetch(
        """
        SELECT LOWER(vm.vendor_name) AS vn, COUNT(DISTINCT r.id) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE vm.vendor_name = ANY($1)
          AND r.duplicate_of_review_id IS NULL
          AND r.source = ANY($2)
        GROUP BY LOWER(vm.vendor_name)
        """,
        list(vendor_set), sources,
    )
    return {r["vn"]: r["cnt"] for r in rows}


def _blog_slug_block_reason(
    row: dict[str, Any] | Any,
    *,
    now: datetime | None = None,
) -> str | None:
    """Return the reason a blog slug should be blocked from regeneration."""
    def _coerce_int_or_default(value: Any, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value != value:
                return default
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                return int(text)
            except ValueError:
                try:
                    return int(float(text))
                except ValueError:
                    return default
        return default

    if not row:
        return None
    status = str(row.get("status") or "").strip().lower()
    slug = str(row.get("slug") or "").strip()
    if not slug or not status:
        return None
    if status == "failed":
        return None
    if status != "rejected":
        return f"status:{status}"

    max_retries = _coerce_int_or_default(
        getattr(settings.b2b_churn, "blog_post_max_rejection_retries", 1),
        1,
    )
    rejection_count = int(row.get("rejection_count") or 0)
    if rejection_count > max_retries:
        return "retry_limit"

    cooldown_hours = _coerce_int_or_default(
        getattr(settings.b2b_churn, "blog_post_rejection_cooldown_hours", 24),
        24,
    )
    rejected_at = row.get("rejected_at")
    if (
        cooldown_hours > 0
        and isinstance(rejected_at, datetime)
        and rejected_at >= (now or datetime.now(rejected_at.tzinfo)) - timedelta(hours=cooldown_hours)
    ):
        return "rejection_cooldown"
    return None


async def _get_blog_slug_block_reason(pool, slug: str) -> str | None:
    """Return the current block reason for a single blog slug, if any."""
    row = await pool.fetchrow(
        """
        SELECT slug, status, COALESCE(rejection_count, 0) AS rejection_count, rejected_at
        FROM blog_posts
        WHERE slug = $1
        """,
        slug,
    )
    return _blog_slug_block_reason(row)


async def _batch_slug_check(pool, slugs: list[str]) -> set[str]:
    """Return slugs that should NOT be regenerated."""
    if not slugs:
        return set()
    rows = await pool.fetch(
        """
        SELECT slug, status, COALESCE(rejection_count, 0) AS rejection_count, rejected_at
        FROM blog_posts
        WHERE slug = ANY($1)
        """,
        slugs,
    )
    blocked: set[str] = set()
    for row in rows:
        reason = _blog_slug_block_reason(row)
        if not reason:
            continue
        blocked.add(str(row["slug"]))
        if reason == "retry_limit":
            logger.info(
                "Permanently blocked slug %s after %d rejections",
                row["slug"],
                int(row.get("rejection_count") or 0),
            )
        elif reason == "rejection_cooldown":
            logger.info("Cooling down rejected slug %s before another retry", row["slug"])
    return blocked


async def _recently_covered_vendors(pool, days: int = 90) -> set[str]:
    """Return vendor names that already have *multiple* B2B blog posts recently.

    A vendor is considered "covered" when it appears in 2+ posts within the
    cooldown window.  This allows a single deep-dive to coexist with a showdown
    or pricing check for the same vendor, while still preventing one vendor
    from dominating the blog.
    """
    rows = await pool.fetch(
        """
        SELECT LOWER(vendor) AS vendor, COUNT(*) AS cnt FROM (
            SELECT data_context->>'vendor' AS vendor
            FROM blog_posts
            WHERE created_at > NOW() - make_interval(days => $1)
              AND data_context->>'vendor' IS NOT NULL
            UNION ALL
            SELECT data_context->>'vendor_a' AS vendor
            FROM blog_posts
            WHERE created_at > NOW() - make_interval(days => $1)
              AND data_context->>'vendor_a' IS NOT NULL
            UNION ALL
            SELECT data_context->>'vendor_b' AS vendor
            FROM blog_posts
            WHERE created_at > NOW() - make_interval(days => $1)
              AND data_context->>'vendor_b' IS NOT NULL
        ) sub
        WHERE vendor != ''
        GROUP BY LOWER(vendor)
        HAVING COUNT(*) >= 3
        """,
        days,
    )
    return {r["vendor"] for r in rows if r["vendor"]}


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")[:60]


def _merge_blog_signals_with_evidence_vault(
    raw_signals: list[dict[str, Any]],
    vault: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Prefer canonical vault pain rows while preserving raw fallback rows."""
    if not isinstance(vault, dict):
        return raw_signals
    weakness_rows = vault.get("weakness_evidence") or []
    gap_labels = [
        str(item.get("label") or item.get("key") or "").strip()
        for item in weakness_rows
        if isinstance(item, dict) and str(item.get("evidence_type") or "") == "feature_gap"
    ]
    vault_rows = [
        item for item in weakness_rows
        if isinstance(item, dict) and str(item.get("evidence_type") or "") == "pain_category"
    ]
    if not vault_rows:
        vault_rows = [
            item for item in weakness_rows
            if isinstance(item, dict) and str(item.get("evidence_type") or "") != "feature_gap"
        ]
    raw_by_key = {
        str(item.get("pain_category") or "").strip().lower(): item
        for item in (raw_signals or [])
        if isinstance(item, dict) and str(item.get("pain_category") or "").strip()
    }
    merged: list[dict[str, Any]] = []
    for item in vault_rows:
        match_key = str(item.get("key") or item.get("label") or "").strip().lower()
        raw = raw_by_key.pop(match_key, {})
        metrics = item.get("supporting_metrics") if isinstance(item.get("supporting_metrics"), dict) else {}
        merged.append({
            "pain_category": str(item.get("label") or item.get("key") or "").strip(),
            "signal_count": int(item.get("mention_count_total") or raw.get("signal_count") or 0),
            "avg_urgency": metrics.get("avg_urgency_when_mentioned") or raw.get("avg_urgency") or 0,
            "feature_gaps": gap_labels[:5] or list(raw.get("feature_gaps") or []),
        })
    merged.extend(raw_by_key.values())
    merged.sort(key=lambda item: (int(item.get("signal_count") or 0), float(item.get("avg_urgency") or 0)), reverse=True)
    return merged


def _merge_blog_quotes_with_evidence_vault(
    raw_quotes: list[dict[str, Any]],
    vault: dict[str, Any] | None,
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Prefer canonical vault quotes while preserving raw quote fallback."""
    if not isinstance(vault, dict):
        return raw_quotes
    candidates: list[dict[str, Any]] = []
    for section, sentiment in (("weakness_evidence", "negative"), ("strength_evidence", "positive")):
        for item in vault.get(section) or []:
            if not isinstance(item, dict):
                continue
            phrase = str(item.get("best_quote") or "").strip()
            if not phrase:
                continue
            source = item.get("quote_source") if isinstance(item.get("quote_source"), dict) else {}
            metrics = item.get("supporting_metrics") if isinstance(item.get("supporting_metrics"), dict) else {}
            candidates.append({
                "phrase": phrase,
                "vendor": vault.get("vendor_name") or "",
                "urgency": metrics.get("avg_urgency_when_mentioned") or 0,
                "role": source.get("reviewer_title") or "",
                "company": source.get("company") or "",
                "company_size": source.get("company_size") or "",
                "industry": source.get("industry") or "",
                "source_name": source.get("source") or "",
                "sentiment": sentiment,
                "_priority": int(item.get("mention_count_total") or 0),
            })
    ordered = sorted(candidates, key=lambda item: (item["_priority"], len(str(item.get("phrase") or ""))), reverse=True)
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ordered + list(raw_quotes or []):
        phrase = " ".join(str(item.get("phrase") or "").lower().split())
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        merged.append({k: v for k, v in item.items() if k != "_priority"})
        if len(merged) >= limit:
            break
    return merged


def _merge_specialized_blog_review_rows(
    raw_rows: list[dict[str, Any]],
    vault_rows: list[dict[str, Any]],
    *,
    limit: int,
    prefer_raw: bool,
) -> list[dict[str, Any]]:
    """Merge specialized review-row payloads with exact-text dedupe."""
    ordered = (list(raw_rows or []) + list(vault_rows or [])) if prefer_raw else (list(vault_rows or []) + list(raw_rows or []))
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ordered:
        text = " ".join(str(item.get("text") or "").lower().split())
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _build_specialized_blog_review_rows_from_evidence_vault(
    vault: dict[str, Any] | None,
    *,
    mode: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Build pricing/positive/switching review rows from canonical vault quotes."""
    if not isinstance(vault, dict):
        return []

    def _matches_pricing(item: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(item.get("key") or "").lower(),
                str(item.get("label") or "").lower(),
                str(item.get("evidence_type") or "").lower(),
            ]
        )
        return any(token in text for token in ("pricing", "price", "billing", "cost", "seat", "contract", "fee"))

    section = "strength_evidence" if mode == "positive" else "weakness_evidence"
    candidates: list[dict[str, Any]] = []
    for item in vault.get(section) or []:
        if not isinstance(item, dict):
            continue
        if mode == "pricing" and not _matches_pricing(item):
            continue
        quote = str(item.get("best_quote") or "").strip()
        if not quote:
            continue
        source = item.get("quote_source") if isinstance(item.get("quote_source"), dict) else {}
        metrics = item.get("supporting_metrics") if isinstance(item.get("supporting_metrics"), dict) else {}
        candidates.append({
            "text": quote[:400],
            "vendor": vault.get("vendor_name") or "",
            "role": source.get("reviewer_title") or "",
            "rating": source.get("rating"),
            "urgency": metrics.get("avg_urgency_when_mentioned") or 0,
            "source_name": source.get("source") or "",
            "company": source.get("company") or "",
            "_priority": (
                int(item.get("mention_count_total") or 0),
                int(item.get("mention_count_recent") or 0),
                str(source.get("reviewed_at") or ""),
            ),
        })
    candidates.sort(key=lambda item: (item["_priority"], len(str(item.get("text") or ""))), reverse=True)
    return [{k: v for k, v in item.items() if k != "_priority"} for item in candidates[:limit]]


# -- Stage 1b: Reasoning Pool Loading --------------------------------


async def _load_pool_layers_for_blog(
    pool,
    topic_type: str,
    topic_ctx: dict[str, Any],
    data: dict[str, Any],
) -> None:
    """Inject reasoning pool data into the blog data dict.

    Loads the 6 reasoning pools via ``fetch_all_pool_layers`` and extracts
    vendor-specific layers based on the topic context.  Also loads reasoning
    synthesis views when available for causal narrative injection.

    Mutates *data* in place -- adds keys like ``pool_layers``,
    ``synthesis_views``, and topic-specific shortcuts (``displacement_a_to_b``,
    ``category_regime``, etc.) that blueprint functions can reference.
    """
    vendor_keys = [
        str(topic_ctx.get(k) or "").strip()
        for k in ("vendor", "vendor_a", "vendor_b", "from_vendor", "to_vendor")
    ]
    vendor_names = [v for v in vendor_keys if v]
    category_name = str(topic_ctx.get("category") or "").strip()
    if not vendor_names and not category_name:
        return

    window_days = settings.b2b_churn.intelligence_window_days
    today = date.today()

    try:
        all_layers = await fetch_all_pool_layers(
            pool, as_of=today, analysis_window_days=window_days,
        )
    except Exception:
        logger.warning("Failed to load pool layers for blog generation", exc_info=True)
        all_layers = {}

    if not all_layers:
        return

    data["pool_layers"] = all_layers

    # Load reasoning views (synthesis-first, legacy fallback)
    from ._b2b_synthesis_reader import load_best_reasoning_views

    view_names = [v.strip() for v in vendor_names if v and v.strip()]

    # For category-only topics, resolve category to vendors
    if not view_names:
        category_name = str(
            topic_ctx.get("category") or ""
        ).strip()
        if category_name:
            try:
                from ._b2b_shared import read_category_vendor_rows

                cat_rows = await read_category_vendor_rows(
                    pool,
                    category_names=[category_name],
                    limit=10,
                )
                view_names = [r["vendor_name"] for r in cat_rows if r.get("vendor_name")]
            except Exception:
                logger.debug("Category vendor resolution for blog render failed", exc_info=True)

    if view_names:
        try:
            synth_views = await load_best_reasoning_views(
                pool, view_names,
                as_of=today,
                analysis_window_days=window_days,
            )
            data["synthesis_views"] = synth_views
        except Exception:
            logger.warning("Failed to load synthesis views for blog generation", exc_info=True)
            data["synthesis_views"] = {}
    else:
        data["synthesis_views"] = {}

    # Load cross-vendor synthesis (battles, councils, asymmetries)
    try:
        from ._b2b_cross_vendor_synthesis import load_cross_vendor_synthesis_lookup
        xv_synth = await load_cross_vendor_synthesis_lookup(
            pool, as_of=today, analysis_window_days=window_days,
        )
        data["xv_synthesis_lookup"] = xv_synth
    except Exception:
        logger.debug("Cross-vendor synthesis load failed, using pool fallback")
        data["xv_synthesis_lookup"] = {"battles": {}, "councils": {}, "asymmetries": {}}

    # Extract topic-specific shortcuts for blueprint convenience
    vendor = str(
        topic_ctx.get("vendor") or topic_ctx.get("from_vendor") or ""
    ).strip()
    vendor_a = str(topic_ctx.get("vendor_a") or "").strip()
    vendor_b = str(topic_ctx.get("vendor_b") or "").strip()

    if vendor:
        vl = all_layers.get(vendor) or {}
        data["pool_segment"] = vl.get("segment")
        data["pool_temporal"] = vl.get("temporal")
        data["pool_accounts"] = vl.get("accounts")
        data["pool_category"] = vl.get("category")
        data["pool_displacement"] = vl.get("displacement") or []

    if vendor_a and vendor_b:
        vl_a = all_layers.get(vendor_a) or {}
        vl_b = all_layers.get(vendor_b) or {}
        data["pool_segment_a"] = vl_a.get("segment")
        data["pool_segment_b"] = vl_b.get("segment")
        data["pool_temporal_a"] = vl_a.get("temporal")
        data["pool_temporal_b"] = vl_b.get("temporal")

        # Find the displacement edge A->B
        for edge in vl_a.get("displacement") or []:
            if isinstance(edge, dict):
                to = str(edge.get("to_vendor") or "").strip().lower()
                if to == vendor_b.lower():
                    data["displacement_a_to_b"] = edge
                    break

        # Find the displacement edge B->A
        for edge in vl_b.get("displacement") or []:
            if isinstance(edge, dict):
                to = str(edge.get("to_vendor") or "").strip().lower()
                if to == vendor_a.lower():
                    data["displacement_b_to_a"] = edge
                    break

        # Category dynamics (shared by vendors in same category)
        data["pool_category"] = vl_a.get("category") or vl_b.get("category")

    if category_name and not data.get("pool_category"):
        category_lower = _normalize_pain_label(category_name)
        for layers in all_layers.values():
            if not isinstance(layers, dict):
                continue
            cat_layer = layers.get("category")
            if not isinstance(cat_layer, dict):
                continue
            layer_category = _normalize_pain_label(cat_layer.get("category"))
            evidence_category = _normalize_pain_label(
                (layers.get("evidence_vault") or {}).get("product_category"),
            )
            if category_lower == layer_category or category_lower == evidence_category:
                data["pool_category"] = cat_layer
                break

    # Extract synthesis contracts for primary vendor(s)
    synth = data.get("synthesis_views") or {}
    primary = vendor or vendor_a
    # For category-only topics, pick the best-informed view from loaded views
    if not primary and synth:
        # Prefer view with category_reasoning.market_regime
        for vn, v in synth.items():
            cat_c = v.contract("category_reasoning")
            if isinstance(cat_c, dict) and cat_c.get("market_regime"):
                primary = vn
                break
        # Fallback: first view with any contracts
        if not primary:
            primary = next(iter(synth), "")
    if primary and synth.get(primary):
        view = synth[primary]
        consumer_context = view.filtered_consumer_context("blog_reranker")
        contracts = consumer_context.get("reasoning_contracts") or view.materialized_contracts()
        if contracts:
            data["synthesis_contracts"] = contracts
            data["synthesis_wedge"] = (
                view.primary_wedge.value if view.primary_wedge else None
            )
            data["synthesis_wedge_label"] = view.wedge_label
        account_preview = consumer_context.get("account_reasoning_preview") or {}
        if isinstance(account_preview, dict) and account_preview:
            data["account_reasoning_preview"] = copy.deepcopy(account_preview)
        # Phase 3 governance sections for blueprint context
        if view.why_they_stay:
            data["why_they_stay"] = view.why_they_stay
        if view.confidence_posture:
            data["confidence_posture"] = view.confidence_posture
        eg = view.evidence_governance
        if eg:
            data["evidence_governance"] = eg
        _inject_blog_reasoning_specificity(data, suffix="", view=view)

    # Balanced multi-vendor synthesis for showdown/comparison topics
    if vendor_a and vendor_b:
        view_b = synth.get(vendor_b)
        if view_b:
            consumer_context_b = view_b.filtered_consumer_context("blog_reranker")
            contracts_b = (
                consumer_context_b.get("reasoning_contracts")
                or view_b.materialized_contracts()
            )
            if contracts_b:
                data["synthesis_contracts_b"] = contracts_b
                data["synthesis_wedge_b"] = (
                    view_b.primary_wedge.value if view_b.primary_wedge else None
                )
                data["synthesis_wedge_label_b"] = view_b.wedge_label
            account_preview_b = consumer_context_b.get("account_reasoning_preview") or {}
            if isinstance(account_preview_b, dict) and account_preview_b:
                data["account_reasoning_preview_b"] = copy.deepcopy(account_preview_b)
            if view_b.why_they_stay:
                data["why_they_stay_b"] = view_b.why_they_stay
            if view_b.confidence_posture:
                data["confidence_posture_b"] = view_b.confidence_posture
            _inject_blog_reasoning_specificity(data, suffix="_b", view=view_b)


# -- Stage 2: Data Gathering --------------------------------------

async def _gather_data(
    pool, topic_type: str, topic_ctx: dict[str, Any]
) -> dict[str, Any]:
    """Fetch data needed for the blueprint from B2B tables."""
    data: dict[str, Any] = {}
    vendor_names = [
        str(topic_ctx.get(key) or "").strip()
        for key in ("vendor", "vendor_a", "vendor_b", "from_vendor")
        if str(topic_ctx.get(key) or "").strip()
    ]
    scorecard_detail_lookup: dict[str, dict[str, Any]] = {}
    try:
        if vendor_names:
            evidence_vault_records, scorecard_detail_lookup = await asyncio.gather(
                _fetch_latest_evidence_vault_records(
                    pool,
                    as_of=date.today(),
                    analysis_window_days=settings.b2b_churn.intelligence_window_days,
                    vendor_names=vendor_names,
                ),
                _fetch_vendor_scorecard_details(
                    pool,
                    vendor_names=vendor_names,
                ),
            )
            evidence_vault_lookup, vault_alignment = (
                _align_vendor_intelligence_records_to_scorecards(
                    list(scorecard_detail_lookup.values()),
                    evidence_vault_records,
                )
            )
            if vault_alignment["mismatched_vendor_count"] > 0:
                logger.info(
                    "Suppressed stale blog evidence-vault overlays for %s",
                    ", ".join(vault_alignment["mismatched_vendors"]),
                )
        else:
            evidence_vault_lookup = {}
    except Exception:
        logger.warning("Failed to load evidence vault for blog data gathering", exc_info=True)
        evidence_vault_lookup = {}
        scorecard_detail_lookup = {}

    if topic_type == "vendor_alternative":
        vendor = topic_ctx["vendor"]
        category = topic_ctx["category"]
        profile, signals, reviews, partner, extended_ctx = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_blog_signal_rows(
                pool,
                vendor,
                detail_row=scorecard_detail_lookup.get(vendor),
            ),
            _fetch_quotable_reviews(pool, vendor_name=vendor),
            _fetch_affiliate_partner(pool, topic_ctx.get("affiliate_id")),
            _fetch_vendor_extended_context(pool, vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = _merge_blog_signals_with_evidence_vault(
            signals if not isinstance(signals, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        data["quotes"] = _merge_blog_quotes_with_evidence_vault(
            reviews if not isinstance(reviews, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        data["partner"] = partner if not isinstance(partner, Exception) else None
        data["extended_ctx"] = extended_ctx if not isinstance(extended_ctx, Exception) else {}

    elif topic_type == "vendor_showdown":
        vendor_a, vendor_b = topic_ctx["vendor_a"], topic_ctx["vendor_b"]
        prof_a, prof_b, sigs_a, sigs_b, quotes = await asyncio.gather(
            _fetch_product_profile(pool, vendor_a),
            _fetch_product_profile(pool, vendor_b),
            _fetch_blog_signal_rows(
                pool,
                vendor_a,
                detail_row=scorecard_detail_lookup.get(vendor_a),
            ),
            _fetch_blog_signal_rows(
                pool,
                vendor_b,
                detail_row=scorecard_detail_lookup.get(vendor_b),
            ),
            _fetch_quotable_reviews(pool, category=topic_ctx["category"]),
            return_exceptions=True,
        )
        data["profile_a"] = prof_a if not isinstance(prof_a, Exception) else {}
        data["profile_b"] = prof_b if not isinstance(prof_b, Exception) else {}
        data["signals_a"] = _merge_blog_signals_with_evidence_vault(
            sigs_a if not isinstance(sigs_a, Exception) else [],
            evidence_vault_lookup.get(vendor_a),
        )
        data["signals_b"] = _merge_blog_signals_with_evidence_vault(
            sigs_b if not isinstance(sigs_b, Exception) else [],
            evidence_vault_lookup.get(vendor_b),
        )
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    elif topic_type == "churn_report":
        vendor = topic_ctx["vendor"]
        profile, signals, quotes = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_blog_signal_rows(
                pool,
                vendor,
                detail_row=scorecard_detail_lookup.get(vendor),
            ),
            _fetch_quotable_reviews(pool, vendor_name=vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = _merge_blog_signals_with_evidence_vault(
            signals if not isinstance(signals, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        data["quotes"] = _merge_blog_quotes_with_evidence_vault(
            quotes if not isinstance(quotes, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )

    elif topic_type == "migration_guide":
        vendor = topic_ctx["vendor"]
        profile, signals, quotes, extended_ctx = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_blog_signal_rows(
                pool,
                vendor,
                detail_row=scorecard_detail_lookup.get(vendor),
            ),
            _fetch_quotable_reviews(pool, vendor_name=vendor),
            _fetch_vendor_extended_context(pool, vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = _merge_blog_signals_with_evidence_vault(
            signals if not isinstance(signals, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        data["quotes"] = _merge_blog_quotes_with_evidence_vault(
            quotes if not isinstance(quotes, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        data["extended_ctx"] = extended_ctx if not isinstance(extended_ctx, Exception) else {}

    elif topic_type == "pricing_reality_check":
        vendor = topic_ctx["vendor"]
        profile, signals = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_blog_signal_rows(
                pool,
                vendor,
                detail_row=scorecard_detail_lookup.get(vendor),
            ),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = _merge_blog_signals_with_evidence_vault(
            signals if not isinstance(signals, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        # Pull actual pricing complaint reviews directly
        from atlas_brain.autonomous.tasks._b2b_shared import read_vendor_quote_evidence
        sources = _blog_source_allowlist()
        pricing_rows = await read_vendor_quote_evidence(
            pool,
            vendor_name=vendor,
            window_days=3650,
            min_urgency=0,
            limit=10,
            sources=sources,
            pain_filter="pricing",
        )
        raw_pricing_reviews = [
            {
                "text": (r["review_text"] or "")[:300],
                "vendor": r["vendor_name"],
                "role": r["reviewer_title"],
                "rating": r["rating"],
                "urgency": r["urgency"] or 0,
                "source_name": r["source"],
            }
            for r in pricing_rows
        ]
        vault_pricing_reviews = _build_specialized_blog_review_rows_from_evidence_vault(
            evidence_vault_lookup.get(vendor),
            mode="pricing",
            limit=10,
        )
        data["pricing_reviews"] = _merge_specialized_blog_review_rows(
            raw_pricing_reviews,
            vault_pricing_reviews,
            limit=10,
            prefer_raw=False,
        )
        # Also pull positive reviews for balance
        positive_reviews = await pool.fetch(
            """
            SELECT
                r.review_text,
                r.reviewer_title,
                r.rating,
                vm.vendor_name AS vendor_name
            FROM b2b_reviews r
            JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
            WHERE vm.vendor_name = $1 AND r.enrichment_status = 'enriched'
              AND r.duplicate_of_review_id IS NULL
              AND rating >= 4
              AND r.source = ANY($2)
            ORDER BY rating DESC
            LIMIT 5
            """,
            vendor, sources,
        )
        raw_positive_reviews = [
            {"text": r["review_text"][:300], "role": r["reviewer_title"], "rating": float(r["rating"]) if r["rating"] else None}
            for r in positive_reviews
        ]
        vault_positive_reviews = _build_specialized_blog_review_rows_from_evidence_vault(
            evidence_vault_lookup.get(vendor),
            mode="positive",
            limit=5,
        )
        data["positive_reviews"] = _merge_specialized_blog_review_rows(
            raw_positive_reviews,
            vault_positive_reviews,
            limit=5,
            prefer_raw=False,
        )

    elif topic_type == "switching_story":
        vendor = topic_ctx["from_vendor"]
        profile, signals = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_blog_signal_rows(
                pool,
                vendor,
                detail_row=scorecard_detail_lookup.get(vendor),
            ),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = _merge_blog_signals_with_evidence_vault(
            signals if not isinstance(signals, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        # Pull actual switching reviews
        # APPROVED-ENRICHMENT-READ: urgency_score
        # Reason: review_text ILIKE filter is domain-specific, only urgency in ORDER BY
        sources = _blog_source_allowlist()
        switch_reviews = await pool.fetch(
            """
            SELECT
                r.review_text,
                vm.vendor_name AS vendor_name,
                r.reviewer_title,
                r.rating,
                r.source,
                r.enrichment->>'urgency_score' AS urgency
            FROM b2b_reviews r
            JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
            WHERE vm.vendor_name = $1 AND r.enrichment_status = 'enriched'
              AND r.duplicate_of_review_id IS NULL
              AND (r.review_text ILIKE '%switch%' OR r.review_text ILIKE '%migrat%'
                   OR r.review_text ILIKE '%moved to%' OR r.review_text ILIKE '%moving to%'
                   OR r.review_text ILIKE '%left for%' OR r.review_text ILIKE '%leaving for%')
              AND r.source = ANY($2)
            ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT 10
            """,
            vendor, sources,
        )
        raw_switch_reviews = [
            {
                "text": r["review_text"][:400],
                "vendor": r["vendor_name"],
                "role": r["reviewer_title"],
                "rating": float(r["rating"]) if r["rating"] else None,
                "urgency": float(r["urgency"]) if r["urgency"] else 0,
                "source_name": r["source"],
            }
            for r in switch_reviews
        ]
        vault_switch_reviews = _build_specialized_blog_review_rows_from_evidence_vault(
            evidence_vault_lookup.get(vendor),
            mode="switching",
            limit=10,
        )
        data["switch_reviews"] = _merge_specialized_blog_review_rows(
            raw_switch_reviews,
            vault_switch_reviews,
            limit=10,
            prefer_raw=True,
        )
        data["quotes"] = data["switch_reviews"]

    elif topic_type == "pain_point_roundup":
        category = topic_ctx["category"]
        # Per-vendor pain breakdown from raw reviews
        # APPROVED-ENRICHMENT-READ: pain_categories, urgency_score
        # Reason: inline aggregate query, structurally coupled to product output
        sources = _blog_source_allowlist()
        vendor_pains = await pool.fetch(
            """
            SELECT
                vm.vendor_name AS vendor_name,
                COUNT(DISTINCT r.id) AS review_count,
                r.enrichment->>'pain_categories' AS pains,
                ROUND(AVG(
                    CASE WHEN r.enrichment->>'urgency_score' ~ '^[0-9]'
                         THEN (r.enrichment->>'urgency_score')::numeric ELSE NULL END
                )::numeric, 1) AS avg_urgency
            FROM b2b_reviews r
            JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
            WHERE r.product_category = $1 AND r.enrichment_status = 'enriched'
              AND r.duplicate_of_review_id IS NULL
              AND r.source = ANY($2)
            GROUP BY vm.vendor_name, r.enrichment->>'pain_categories'
            ORDER BY review_count DESC
            LIMIT 30
            """,
            category, sources,
        )
        # Aggregate top pain per vendor
        vendor_pain_map: dict[str, dict] = {}
        for r in vendor_pains:
            vn = r["vendor_name"]
            if vn not in vendor_pain_map:
                vendor_pain_map[vn] = {
                    "vendor": vn, "review_count": 0, "top_pain": "overall_dissatisfaction",
                    "avg_urgency": float(r["avg_urgency"]) if r["avg_urgency"] else 0,
                }
            vendor_pain_map[vn]["review_count"] += r["review_count"]
            pain_str = r["pains"] or ""
            if "pricing" in pain_str:
                vendor_pain_map[vn]["top_pain"] = "pricing"
            elif "ux" in pain_str:
                vendor_pain_map[vn]["top_pain"] = "ux"
            elif "support" in pain_str:
                vendor_pain_map[vn]["top_pain"] = "support"
            elif "reliability" in pain_str:
                vendor_pain_map[vn]["top_pain"] = "reliability"
            elif "features" in pain_str:
                vendor_pain_map[vn]["top_pain"] = "features"
        data["vendor_pains"] = list(vendor_pain_map.values())
        # Pull quotable reviews across the category
        quotes = await _fetch_quotable_reviews(pool, category=category)
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    elif topic_type == "best_fit_guide":
        category = topic_ctx["category"]
        sources = _blog_source_allowlist()
        # Fetch all profiles in category
        from ._b2b_shared import read_category_vendor_rows

        vendor_rows = await read_category_vendor_rows(
            pool,
            category_names=[category],
        )
        profiles = []
        vendor_signals = []
        for vr in vendor_rows[:10]:
            vn = vr["vendor_name"]
            p = await _fetch_product_profile(pool, vn)
            s = await _fetch_blog_signal_rows(
                pool,
                vn,
                detail_row=scorecard_detail_lookup.get(vn),
            )
            if p:
                # Also pull avg rating from raw reviews
                rating_row = await pool.fetchrow(
                    "SELECT ROUND(AVG(r.rating)::numeric, 1) AS avg_rating, COUNT(*) AS cnt "
                    "FROM b2b_reviews r "
                    "JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id "
                    "WHERE vm.vendor_name = $1 AND r.rating IS NOT NULL "
                    "AND r.duplicate_of_review_id IS NULL AND r.source = ANY($2)",
                    vn, sources,
                )
                profiles.append({
                    "vendor": vn,
                    "profile": p,
                    "avg_rating": float(rating_row["avg_rating"]) if rating_row and rating_row["avg_rating"] else None,
                    "review_count": rating_row["cnt"] if rating_row else 0,
                })
                vendor_signals.append(
                    {
                        "vendor": vn,
                        "signals": _merge_blog_signals_with_evidence_vault(
                            s,
                            evidence_vault_lookup.get(vn),
                        ),
                    }
                )
        data["vendor_profiles"] = profiles
        data["vendor_signals"] = vendor_signals
        quotes = await _fetch_quotable_reviews(pool, category=category)
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    elif topic_type == "vendor_deep_dive":
        vendor = topic_ctx["vendor"]
        profile, signals, quotes, extended_ctx = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_blog_signal_rows(
                pool,
                vendor,
                detail_row=scorecard_detail_lookup.get(vendor),
            ),
            _fetch_quotable_reviews(pool, vendor_name=vendor),
            _fetch_vendor_extended_context(pool, vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = _merge_blog_signals_with_evidence_vault(
            signals if not isinstance(signals, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        data["quotes"] = _merge_blog_quotes_with_evidence_vault(
            quotes if not isinstance(quotes, Exception) else [],
            evidence_vault_lookup.get(vendor),
        )
        data["extended_ctx"] = extended_ctx if not isinstance(extended_ctx, Exception) else {}
        # Fetch competitors for context
        compared = (data["profile"].get("commonly_compared_to") or [])[:5]
        competitor_profiles = []
        for comp in compared:
            comp_name = comp.get("vendor", comp) if isinstance(comp, dict) else str(comp)
            try:
                cp = await _fetch_product_profile(pool, comp_name)
                if cp:
                    competitor_profiles.append(cp)
            except Exception:
                pass
        data["competitor_profiles"] = competitor_profiles

    elif topic_type == "market_landscape":
        category = topic_ctx["category"]
        sources = _blog_source_allowlist()
        # Fetch all vendors in this category
        from ._b2b_shared import read_category_vendor_rows

        vendor_rows = await read_category_vendor_rows(
            pool,
            category_names=[category],
            require_scorecard_match=True,
        )
        vendor_names = [r["vendor_name"] for r in vendor_rows]
        profiles = []
        signals_list = []
        for vn in vendor_names[:10]:
            try:
                p = await _fetch_product_profile(pool, vn)
                s = await _fetch_blog_signal_rows(
                    pool,
                    vn,
                    detail_row=scorecard_detail_lookup.get(vn),
                )
                rating_row = await pool.fetchrow(
                    "SELECT ROUND(AVG(r.rating)::numeric, 1) AS avg_rating, COUNT(*) AS cnt "
                    "FROM b2b_reviews r "
                    "JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id "
                    "WHERE vm.vendor_name = $1 AND r.rating IS NOT NULL "
                    "AND r.duplicate_of_review_id IS NULL AND r.source = ANY($2)",
                    vn, sources,
                )
                profiles.append({
                    "vendor": vn,
                    "profile": p,
                    "avg_rating": float(rating_row["avg_rating"]) if rating_row and rating_row["avg_rating"] else None,
                    "review_count": rating_row["cnt"] if rating_row else 0,
                })
                signals_list.append({
                    "vendor": vn,
                    "signals": _merge_blog_signals_with_evidence_vault(
                        s,
                        evidence_vault_lookup.get(vn),
                    ),
                })
            except Exception:
                pass
        data["vendor_profiles"] = profiles
        data["vendor_signals"] = signals_list
        quotes = await _fetch_quotable_reviews(pool, category=category)
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    # Data context metadata -- scoped to vendor(s) from this topic
    ctx_sources = _blog_source_allowlist()
    vendor_names: list[str] = []
    if topic_ctx.get("vendor"):
        vendor_names.append(topic_ctx["vendor"])
    if topic_ctx.get("vendor_a"):
        vendor_names.append(topic_ctx["vendor_a"])
    if topic_ctx.get("vendor_b"):
        vendor_names.append(topic_ctx["vendor_b"])
    if topic_ctx.get("from_vendor"):
        vendor_names.append(topic_ctx["from_vendor"])
    # For category-level topics, pull all vendors in the category
    if not vendor_names and topic_ctx.get("category"):
        cat_rows = await pool.fetch(
            "SELECT DISTINCT vm.vendor_name "
            "FROM b2b_reviews r "
            "JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id "
            "WHERE r.product_category = $1 AND r.duplicate_of_review_id IS NULL AND r.source = ANY($2)",
            topic_ctx["category"], ctx_sources,
        )
        vendor_names = [r["vendor_name"] for r in cat_rows]

    # APPROVED-ENRICHMENT-READ: churn_signals.intent_to_leave
    # Reason: inline aggregate query, structurally coupled to product output
    if vendor_names:
        ctx_row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_reviews,
                COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched,
                COUNT(*) FILTER (WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true) AS churn_intent,
                MIN(imported_at)::date AS earliest,
                MAX(imported_at)::date AS latest
            FROM b2b_reviews
            WHERE EXISTS (
                SELECT 1
                FROM b2b_review_vendor_mentions vm
                WHERE vm.review_id = b2b_reviews.id
                  AND vm.vendor_name = ANY($1)
            )
              AND duplicate_of_review_id IS NULL
              AND source = ANY($2)
            """,
            vendor_names, ctx_sources,
        )
    else:
        # APPROVED-ENRICHMENT-READ: churn_signals.intent_to_leave
        # Reason: inline aggregate query, structurally coupled to product output
        ctx_row = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total_reviews,
                COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched,
                COUNT(*) FILTER (WHERE (enrichment->'churn_signals'->>'intent_to_leave')::boolean = true) AS churn_intent,
                MIN(imported_at)::date AS earliest,
                MAX(imported_at)::date AS latest
            FROM b2b_reviews
            WHERE source = ANY($1)
              AND duplicate_of_review_id IS NULL
            """,
            ctx_sources,
        )
    data["data_context"] = {
        "total_reviews_analyzed": ctx_row["total_reviews"] if ctx_row else 0,
        "enriched_count": ctx_row["enriched"] if ctx_row else 0,
        "churn_intent_count": ctx_row["churn_intent"] if ctx_row else 0,
        "review_period": (
            f"{ctx_row['earliest']} to {ctx_row['latest']}"
            if ctx_row and ctx_row["earliest"]
            else "dates unavailable"
        ),
        "report_date": str(date.today()),
        "booking_url": settings.b2b_campaign.default_booking_url,
    }

    # Store the full topic_ctx so regeneration can reconstruct blueprints
    # without re-deriving scoring stats from the DB.
    data["data_context"]["topic_ctx"] = {
        k: v for k, v in topic_ctx.items()
        if v is not None and k != "slug"
    }

    # Keep vendor names as top-level keys for the dedup SQL in
    # _recently_covered_vendors() which queries data_context->>'vendor' etc.
    for vk in ("vendor", "vendor_a", "vendor_b"):
        if topic_ctx.get(vk):
            data["data_context"][vk] = topic_ctx[vk]

    # Source distribution and data quality metadata
    source_dist = await _fetch_source_distribution(pool, vendor_names)
    data["data_context"]["source_distribution"] = source_dist
    data["data_context"]["data_source_label"] = "Public B2B software review platforms"
    data["data_context"]["data_disclaimer"] = (
        "Analysis based on self-selected reviewer feedback. "
        "Results reflect reviewer perception, not product capability."
    )
    review_count = data["data_context"]["enriched_count"]
    data["data_context"]["data_quality"] = {
        "sample_size": review_count,
        "confidence": "high" if review_count >= 50 else "moderate" if review_count >= 20 else "low",
        "note": f"Based on {review_count} enriched reviews" + (
            " (small sample)" if review_count < 20 else ""
        ),
    }
    vault_vendors = [vn for vn in vendor_names if evidence_vault_lookup.get(vn)]
    data["data_context"]["evidence_vault_used"] = bool(vault_vendors)
    if vault_vendors:
        data["data_context"]["evidence_vault_vendors"] = vault_vendors

    category = (
        topic_ctx.get("category")
        or topic_ctx.get("product_category")
        or data.get("profile", {}).get("product_category")
    )
    if category:
        overview = await _fetch_category_overview_entry(pool, str(category))
        if overview:
            data["category_overview"] = overview
            regime = (overview.get("cross_vendor_analysis") or {}).get("market_regime")
            if regime:
                data["data_context"]["market_regime"] = regime

    # Attach affiliate info to data_context if available.
    # For topic types that don't explicitly fetch a partner (everything except
    # vendor_alternative), look up a matching partner by product category so
    # comparison/landscape/deep-dive posts can include the affiliate link.
    partner = data.get("partner")
    if not partner:
        category = topic_ctx.get("category") or topic_ctx.get("product_category")
        if category:
            partner = await _fetch_affiliate_partner_by_category(pool, category)
            if partner:
                data["partner"] = partner
    if partner:
        data["data_context"]["affiliate_partner"] = {
            "name": partner["name"],
            "product_name": partner["product_name"],
            "slug": _slugify(partner["product_name"]),
        }
        data["data_context"]["affiliate_url"] = partner["affiliate_url"]
        data["data_context"]["affiliate_slug"] = _slugify(partner["product_name"])

    return data


async def _fetch_product_profile(pool, vendor_name: str) -> dict[str, Any]:
    """Fetch the product profile for a vendor."""
    row = await pool.fetchrow(
        "SELECT * FROM b2b_product_profiles WHERE vendor_name = $1 ORDER BY last_computed_at DESC LIMIT 1",
        vendor_name,
    )
    if not row:
        return {}
    result = dict(row)
    for key in ("strengths", "weaknesses", "pain_addressed", "primary_use_cases",
                "top_integrations", "commonly_compared_to", "commonly_switched_from",
                "typical_company_size", "typical_industries"):
        if key in result and isinstance(result[key], str):
            try:
                result[key] = json.loads(result[key])
            except (json.JSONDecodeError, TypeError):
                pass
    # Normalize field names for blueprint compatibility
    result["integrations"] = result.get("top_integrations", [])
    result["use_cases"] = result.get("primary_use_cases", [])
    return result


async def _fetch_vendor_extended_context(pool, vendor_name: str) -> dict[str, Any]:
    """Fetch extended vendor context from use case, integration, and buyer profile tables.

    Returns ``{"use_cases": [...], "integrations": [...], "buyer_profiles": [...]}``.
    Each list entry includes mention counts and confidence scores from aggregated review data.
    Empty dicts on missing tables or errors -- callers use profile fallbacks.
    """
    try:
        use_case_rows, integration_rows, buyer_rows = await asyncio.gather(
            pool.fetch(
                "SELECT use_case_name, mention_count, avg_urgency, confidence_score "
                "FROM b2b_vendor_use_cases WHERE vendor_name = $1 "
                "ORDER BY mention_count DESC LIMIT 8",
                vendor_name,
            ),
            pool.fetch(
                "SELECT integration_name, mention_count, confidence_score "
                "FROM b2b_vendor_integrations WHERE vendor_name = $1 "
                "ORDER BY mention_count DESC LIMIT 10",
                vendor_name,
            ),
            pool.fetch(
                "SELECT role_type, buying_stage, review_count, dm_count, avg_urgency "
                "FROM b2b_vendor_buyer_profiles WHERE vendor_name = $1 "
                "ORDER BY review_count DESC LIMIT 8",
                vendor_name,
            ),
        )
        return {
            "use_cases": [dict(r) for r in use_case_rows],
            "integrations": [dict(r) for r in integration_rows],
            "buyer_profiles": [dict(r) for r in buyer_rows],
        }
    except Exception:
        logger.debug("Extended context unavailable for %s", vendor_name)
        return {}


async def _fetch_pain_category_urgency(pool, vendor_name: str) -> dict[str, float]:
    """Query per-category average urgency directly from b2b_reviews."""
    # APPROVED-ENRICHMENT-READ: pain_category, urgency_score
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT r.enrichment->>'pain_category' AS pain_cat,
               AVG((r.enrichment->>'urgency_score')::numeric) AS avg_urg,
               COUNT(*) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE vm.vendor_name = $1 AND r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enrichment->>'pain_category' IS NOT NULL
        GROUP BY r.enrichment->>'pain_category'
        """,
        vendor_name,
    )
    return {
        r["pain_cat"]: round(float(r["avg_urg"]), 1)
        for r in rows
        if r["pain_cat"] and r["avg_urg"] is not None
    }


def _build_blog_signal_rows_from_scorecard_detail(
    row: dict[str, Any],
    *,
    per_cat_urgency: dict[str, float],
) -> list[dict[str, Any]]:
    """Project one scorecard-detail row into blog signal rows."""
    if not row:
        return []
    vendor_avg = round(float(row["avg_urgency_score"]), 1)

    # Unpack JSONB pain categories into structured list
    pain_cats = row["top_pain_categories"] or []
    if isinstance(pain_cats, str):
        try:
            pain_cats = json.loads(pain_cats)
        except (json.JSONDecodeError, TypeError):
            pain_cats = []

    feature_gaps = row["top_feature_gaps"] or []
    if isinstance(feature_gaps, str):
        try:
            feature_gaps = json.loads(feature_gaps)
        except (json.JSONDecodeError, TypeError):
            feature_gaps = []

    results = []
    seen_cats: set[str] = set()
    for pc in pain_cats[:10]:
        raw_cat = pc.get("category", pc) if isinstance(pc, dict) else str(pc)
        # Handle double-encoded JSON (e.g. '{"category": "features", "severity": "primary"}')
        if isinstance(raw_cat, str) and raw_cat.startswith("{"):
            try:
                inner = json.loads(raw_cat)
                raw_cat = inner.get("category", raw_cat) if isinstance(inner, dict) else raw_cat
            except (json.JSONDecodeError, TypeError):
                pass
        cat_name = str(raw_cat)
        # Skip null/None/empty categories
        if not cat_name or cat_name in ("None", "null", "none", ""):
            continue
        count = pc.get("count", 1) if isinstance(pc, dict) else 1
        # Use per-category urgency when available, fall back to vendor average
        cat_urgency = per_cat_urgency.get(cat_name, vendor_avg)
        results.append({
            "pain_category": cat_name,
            "signal_count": count,
            "avg_urgency": cat_urgency,
            "feature_gaps": [
                fg.get("feature", fg) if isinstance(fg, dict) else str(fg)
                for fg in feature_gaps[:5]
            ],
        })
        seen_cats.add(cat_name)

    # Supplement with categories from enriched reviews not in the aggregate
    for cat_name, urgency in per_cat_urgency.items():
        if cat_name in seen_cats or not cat_name or cat_name in ("None", "null", "none"):
            continue
        results.append({
            "pain_category": cat_name,
            "signal_count": 1,
            "avg_urgency": urgency,
            "feature_gaps": [],
        })
    return results


async def _fetch_blog_signal_rows(
    pool,
    vendor_name: str,
    *,
    detail_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch blog signal rows for one vendor from scorecard detail."""
    row = detail_row
    if row is None:
        from ._b2b_shared import read_vendor_scorecard_detail

        row = await read_vendor_scorecard_detail(
            pool,
            vendor_name,
        )
    if not row:
        return []

    # Fetch per-category urgency from raw reviews so category charts reflect
    # the actual category average, not the rolled-up vendor average.
    per_cat_urgency = await _fetch_pain_category_urgency(pool, vendor_name)
    return _build_blog_signal_rows_from_scorecard_detail(
        row,
        per_cat_urgency=per_cat_urgency,
    )


async def _fetch_churn_signals(pool, vendor_name: str) -> list[dict[str, Any]]:
    """Compatibility seam for callers that still fetch one vendor at a time."""
    return await _fetch_blog_signal_rows(pool, vendor_name)


async def _fetch_quotable_reviews(
    pool, vendor_name: str | None = None, category: str | None = None
) -> list[dict[str, Any]]:
    """Pull balanced positive + negative review excerpts for the topic.

    Returns up to 15 quotes interleaved with a ``sentiment`` field so the
    LLM skill prompt can place them in the right sections.
    """
    sources = _blog_source_allowlist()
    negative = await _fetch_negative_quotes(pool, vendor_name, category, sources, limit=9)
    positive = await _fetch_positive_quotes(pool, vendor_name, category, sources, limit=6)

    # Interleave: negative, positive, negative, positive, ...
    merged: list[dict[str, Any]] = []
    ni, pi = 0, 0
    while ni < len(negative) or pi < len(positive):
        if ni < len(negative):
            merged.append(negative[ni])
            ni += 1
        if pi < len(positive):
            merged.append(positive[pi])
            pi += 1
    return merged[:15]


def _extract_phrase(text: str) -> str:
    """Extract the most impactful sentence from review text."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
    return (sentences[0] if sentences else text[:150])[:200]


async def _fetch_negative_quotes(
    pool, vendor_name: str | None, category: str | None,
    sources: list[str], limit: int = 9,
) -> list[dict[str, Any]]:
    """High-urgency enriched reviews (negative signal)."""
    # APPROVED-ENRICHMENT-READ: urgency_score
    # Reason: urgency used only in ORDER BY; query is structurally tied to account_resolution + prospect_org_cache JOINs
    vendor_projection = (
        "matched_vm.vendor_name AS matched_vendor_name"
        if vendor_name
        else "COALESCE(primary_vm.vendor_name, r.vendor_name) AS matched_vendor_name"
    )
    vendor_join = (
        _blog_review_vendor_join("r", "matched_vm")
        if vendor_name
        else _blog_primary_vendor_join("r", "primary_vm")
    )
    _quote_cols = f"""
        r.review_text, {vendor_projection}, r.reviewer_title, r.rating,
        r.enrichment->>'urgency_score' AS urgency,
        r.source,
        COALESCE(
            CASE
                WHEN ar.confidence_label IN ('high', 'medium')
                THEN ar.resolved_company_name
                ELSE NULL
            END,
            r.reviewer_company
        ) AS company,
        r.company_size_raw,
        COALESCE(poc.industry, r.reviewer_industry) AS industry,
        poc.employee_count AS verified_employee_count,
        poc.country AS company_country
    """
    _quote_joins = f"""
        {vendor_join}
        LEFT JOIN b2b_account_resolution ar
            ON ar.review_id = r.id AND ar.resolution_status = 'resolved'
        LEFT JOIN prospect_org_cache poc
            ON poc.company_name_norm = CASE
                WHEN ar.confidence_label IN ('high', 'medium')
                THEN ar.normalized_company_name
                ELSE NULL
            END
    """
    if vendor_name:
        rows = await pool.fetch(
            f"""
            SELECT {_quote_cols}
            FROM b2b_reviews r {_quote_joins}
            WHERE matched_vm.vendor_name = $1
              AND r.enrichment_status = 'enriched'
              AND r.duplicate_of_review_id IS NULL
              AND r.source = ANY($2)
            ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT $3
            """,
            vendor_name, sources, limit,
        )
    elif category:
        rows = await pool.fetch(
            f"""
            SELECT {_quote_cols}
            FROM b2b_reviews r {_quote_joins}
            WHERE r.product_category = $1
              AND r.enrichment_status = 'enriched'
              AND r.duplicate_of_review_id IS NULL
              AND r.source = ANY($2)
            ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT $3
            """,
            category, sources, limit,
        )
    else:
        return []

    results = []
    for r in rows:
        urg = 0.0
        try:
            urg = float(r["urgency"]) if r["urgency"] else 0.0
        except (ValueError, TypeError):
            pass
        results.append({
            "phrase": _extract_phrase(r["review_text"] or ""),
            "vendor": r["matched_vendor_name"],
            "urgency": urg,
            "role": r["reviewer_title"],
            "company": r["company"],
            "company_size": r["company_size_raw"],
            "industry": r["industry"],
            "verified_employee_count": r["verified_employee_count"],
            "company_country": r["company_country"],
            "source_name": r["source"],
            "sentiment": "negative",
        })
    return results


async def _fetch_positive_quotes(
    pool, vendor_name: str | None, category: str | None,
    sources: list[str], limit: int = 6,
) -> list[dict[str, Any]]:
    """High-rated reviews from raw columns (no enrichment required)."""
    vendor_projection = (
        "matched_vm.vendor_name AS matched_vendor_name"
        if vendor_name
        else "COALESCE(primary_vm.vendor_name, r.vendor_name) AS matched_vendor_name"
    )
    vendor_join = (
        _blog_review_vendor_join("r", "matched_vm")
        if vendor_name
        else _blog_primary_vendor_join("r", "primary_vm")
    )
    _pos_cols = f"""
        COALESCE(r.pros, r.review_text) AS text, {vendor_projection},
        r.reviewer_title, r.rating, r.source,
        COALESCE(
            CASE
                WHEN ar.confidence_label IN ('high', 'medium')
                THEN ar.resolved_company_name
                ELSE NULL
            END,
            r.reviewer_company
        ) AS company,
        r.company_size_raw,
        COALESCE(poc.industry, r.reviewer_industry) AS industry,
        poc.employee_count AS verified_employee_count,
        poc.country AS company_country
    """
    _pos_joins = f"""
        {vendor_join}
        LEFT JOIN b2b_account_resolution ar
            ON ar.review_id = r.id AND ar.resolution_status = 'resolved'
        LEFT JOIN prospect_org_cache poc
            ON poc.company_name_norm = CASE
                WHEN ar.confidence_label IN ('high', 'medium')
                THEN ar.normalized_company_name
                ELSE NULL
            END
    """
    if vendor_name:
        rows = await pool.fetch(
            f"""
            SELECT {_pos_cols}
            FROM b2b_reviews r {_pos_joins}
            WHERE matched_vm.vendor_name = $1
              AND r.duplicate_of_review_id IS NULL
              AND r.rating >= 4
              AND r.source = ANY($2)
              AND COALESCE(r.pros, r.review_text) IS NOT NULL
              AND LENGTH(COALESCE(r.pros, r.review_text)) > 20
            ORDER BY r.rating DESC, r.imported_at DESC
            LIMIT $3
            """,
            vendor_name, sources, limit,
        )
    elif category:
        rows = await pool.fetch(
            f"""
            SELECT {_pos_cols}
            FROM b2b_reviews r {_pos_joins}
            WHERE r.product_category = $1
              AND r.duplicate_of_review_id IS NULL
              AND r.rating >= 4
              AND r.source = ANY($2)
              AND COALESCE(r.pros, r.review_text) IS NOT NULL
              AND LENGTH(COALESCE(r.pros, r.review_text)) > 20
            ORDER BY r.rating DESC, r.imported_at DESC
            LIMIT $3
            """,
            category, sources, limit,
        )
    else:
        return []

    results = []
    for r in rows:
        results.append({
            "phrase": _extract_phrase(r["text"] or ""),
            "vendor": r["matched_vendor_name"],
            "urgency": 0.0,
            "role": r["reviewer_title"],
            "company": r["company"],
            "company_size": r["company_size_raw"],
            "industry": r["industry"],
            "verified_employee_count": r["verified_employee_count"],
            "company_country": r["company_country"],
            "source_name": r["source"],
            "sentiment": "positive",
        })
    return results


async def _fetch_affiliate_partner(pool, partner_id: str | None) -> dict[str, Any] | None:
    """Fetch affiliate partner details."""
    if not partner_id:
        return None
    import uuid as _uuid
    pid = _uuid.UUID(partner_id) if isinstance(partner_id, str) else partner_id
    row = await pool.fetchrow(
        "SELECT id, name, product_name, affiliate_url, category FROM affiliate_partners WHERE id = $1",
        pid,
    )
    if not row:
        return None
    return dict(row)


async def _fetch_affiliate_partner_by_category(pool, category: str) -> dict[str, Any] | None:
    """Fetch the first enabled affiliate partner matching a product category."""
    row = await pool.fetchrow(
        "SELECT id, name, product_name, affiliate_url, category "
        "FROM affiliate_partners WHERE enabled = true AND LOWER(category) = LOWER($1) "
        "LIMIT 1",
        category,
    )
    if not row:
        return None
    return dict(row)



async def _fetch_category_overview_entry(pool, category: str) -> dict[str, Any] | None:
    """Fetch the latest persisted category overview entry for a category."""
    if not category:
        return None

    row = await pool.fetchrow(
        """
        SELECT intelligence_data
        FROM b2b_intelligence
        WHERE report_type = 'category_overview'
        ORDER BY report_date DESC, created_at DESC
        LIMIT 1
        """,
    )
    if not row:
        return None

    data = row["intelligence_data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    entries = data if isinstance(data, list) else data.get("category_overview", data)
    if not isinstance(entries, list):
        return None

    wanted = category.strip().lower()
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("category", "")).strip().lower() == wanted:
            return entry
    return None


async def _fetch_source_distribution(pool, vendor_names: list[str]) -> dict[str, Any]:
    """Return review counts by source platform for the given vendors."""
    if not vendor_names:
        return {"sources": [], "verified_count": 0, "community_count": 0}
    allowed = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT COALESCE(r.source, 'unknown') AS src, COUNT(DISTINCT r.id) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE vm.vendor_name = ANY($1) AND r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.source = ANY($2)
        GROUP BY r.source
        ORDER BY cnt DESC
        """,
        vendor_names, allowed,
    )
    sources = [{"name": r["src"], "count": r["cnt"]} for r in rows]
    verified = sum(r["cnt"] for r in rows if r["src"].lower().replace(" ", "_") in VERIFIED_SOURCES)
    community = sum(r["cnt"] for r in rows if r["src"].lower().replace(" ", "_") not in VERIFIED_SOURCES)
    return {"sources": sources, "verified_count": verified, "community_count": community}


# -- Data Sufficiency Check ----------------------------------------

# Topic types that focus on a single vendor
_SINGLE_VENDOR_TYPES = {
    "vendor_alternative", "vendor_deep_dive", "churn_report",
    "migration_guide", "pricing_reality_check", "switching_story",
}
# Topic types that need multiple vendor profiles
_MULTI_VENDOR_TYPES = {"market_landscape", "best_fit_guide"}


def _check_data_sufficiency(topic_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Validate gathered data meets minimum requirements for a quality post.

    Returns {"sufficient": bool, "reason": str}.
    """
    quotes = data.get("quotes", [])
    min_quotes = _blog_topic_threshold(
        getattr(settings.b2b_churn, "blog_min_quotes_by_topic", {}) or {},
        topic_type,
        default=2,
    )

    # Universal: at least 2 quotable reviews
    # (pricing_reality_check builds quotes from pricing_reviews in the blueprint)
    if topic_type != "pricing_reality_check" and len(quotes) < min_quotes:
        return {
            "sufficient": False,
            "reason": f"Only {len(quotes)} quotable reviews (need {min_quotes}+)",
        }

    # Single-vendor types: product profile must exist
    if topic_type in _SINGLE_VENDOR_TYPES:
        profile = data.get("profile", {})
        if not profile:
            return {"sufficient": False, "reason": "No product profile found"}

    # Churn-focused types: at least 1 signal category
    if topic_type in ("churn_report", "vendor_alternative", "migration_guide"):
        signals = data.get("signals", [])
        if not signals:
            return {"sufficient": False, "reason": "No churn signal categories found"}

    # Pricing / switching story need their specific reviews
    if topic_type == "pricing_reality_check":
        if not data.get("pricing_reviews"):
            return {"sufficient": False, "reason": "No pricing complaint reviews found"}

    if topic_type == "switching_story":
        if not data.get("switch_reviews"):
            return {"sufficient": False, "reason": "No switching reviews found"}

    # Showdowns: both vendor profiles must exist
    if topic_type == "vendor_showdown":
        if not data.get("profile_a"):
            return {"sufficient": False, "reason": "No product profile for vendor A"}
        if not data.get("profile_b"):
            return {"sufficient": False, "reason": "No product profile for vendor B"}

    # Multi-vendor types: at least 2 vendor profiles
    if topic_type in _MULTI_VENDOR_TYPES:
        vendor_profiles = data.get("vendor_profiles", [])
        min_vendor_profiles = _blog_min_vendor_profiles(topic_type)
        if len(vendor_profiles) < min_vendor_profiles:
            return {
                "sufficient": False,
                "reason": (
                    f"Only {len(vendor_profiles)} vendor profiles "
                    f"(need {min_vendor_profiles}+)"
                ),
            }

    return {"sufficient": True, "reason": ""}


def _check_blueprint_sufficiency(blueprint: PostBlueprint) -> dict[str, Any]:
    min_sections = _blog_topic_threshold(
        getattr(settings.b2b_churn, "blog_min_sections_by_topic", {}) or {},
        blueprint.topic_type,
        default=0,
    )
    section_count = len(blueprint.sections or [])
    if section_count < min_sections:
        return {
            "sufficient": False,
            "reason": f"Only {section_count} blueprint sections (need {min_sections}+)",
        }
    return {"sufficient": True, "reason": ""}


# -- Stage 3: Blueprint Construction ------------------------------

def _build_blueprint(
    topic_type: str, topic_ctx: dict[str, Any], data: dict[str, Any]
) -> PostBlueprint:
    """Build a structured post blueprint deterministically from data."""
    builder = {
        "vendor_alternative": _blueprint_vendor_alternative,
        "vendor_showdown": _blueprint_vendor_showdown,
        "churn_report": _blueprint_churn_report,
        "migration_guide": _blueprint_migration_guide,
        "vendor_deep_dive": _blueprint_vendor_deep_dive,
        "market_landscape": _blueprint_market_landscape,
        "pricing_reality_check": _blueprint_pricing_reality_check,
        "switching_story": _blueprint_switching_story,
        "pain_point_roundup": _blueprint_pain_point_roundup,
        "best_fit_guide": _blueprint_best_fit_guide,
    }[topic_type]
    bp = builder(topic_ctx, data)
    bp.cta = _build_cta(bp.topic_type, bp.data_context)
    return bp


def _blog_quote_highlights(
    quotes: list[dict[str, Any]] | None,
    *,
    limit: int = 4,
    vendors: list[str] | None = None,
) -> list[dict[str, Any]]:
    allowed_vendors = {
        str(value or "").strip().lower()
        for value in (vendors or [])
        if str(value or "").strip()
    }
    seen: set[str] = set()
    highlights: list[dict[str, Any]] = []
    for item in quotes or []:
        if not isinstance(item, dict):
            continue
        vendor = str(item.get("vendor") or "").strip()
        if allowed_vendors and vendor.lower() not in allowed_vendors:
            continue
        phrase = " ".join(str(item.get("phrase") or "").split())
        if not phrase:
            continue
        marker = phrase.lower()
        if marker in seen:
            continue
        seen.add(marker)
        row: dict[str, Any] = {
            "vendor": vendor,
            "phrase": phrase[:220],
            "sentiment": str(item.get("sentiment") or "").strip(),
        }
        for key in ("role", "company", "source_name"):
            value = str(item.get(key) or "").strip()
            if value:
                row[key] = value[:120]
        highlights.append(row)
        if len(highlights) >= max(1, int(limit)):
            break
    return highlights


def _aggregate_category_pain_rows(
    vendor_signals: list[dict[str, Any]] | None,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for vendor_row in vendor_signals or []:
        if not isinstance(vendor_row, dict):
            continue
        vendor = str(vendor_row.get("vendor") or "").strip()
        for signal in vendor_row.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            pain = str(signal.get("pain_category") or "").strip()
            if not pain or pain.lower() in {"none", "null"}:
                continue
            bucket = aggregated.setdefault(
                pain,
                {
                    "name": pain,
                    "vendor_count": 0,
                    "signal_count": 0,
                    "avg_urgency_total": 0.0,
                    "_vendors": set(),
                },
            )
            vendors = bucket["_vendors"]
            if vendor and vendor not in vendors:
                vendors.add(vendor)
                bucket["vendor_count"] += 1
            try:
                bucket["signal_count"] += int(signal.get("signal_count") or 0)
            except (TypeError, ValueError):
                pass
            try:
                bucket["avg_urgency_total"] += float(signal.get("avg_urgency") or 0.0)
            except (TypeError, ValueError):
                pass
    rows: list[dict[str, Any]] = []
    for bucket in aggregated.values():
        vendor_count = int(bucket.get("vendor_count") or 0)
        avg_total = float(bucket.get("avg_urgency_total") or 0.0)
        rows.append(
            {
                "name": bucket["name"],
                "vendor_count": vendor_count,
                "signal_count": int(bucket.get("signal_count") or 0),
                "avg_urgency": round(avg_total / max(vendor_count, 1), 1),
            }
        )
    rows.sort(
        key=lambda item: (
            -int(item.get("vendor_count") or 0),
            -int(item.get("signal_count") or 0),
            -float(item.get("avg_urgency") or 0.0),
            str(item.get("name") or "").lower(),
        )
    )
    return rows[: max(1, int(limit))]


def _blueprint_vendor_alternative(ctx: dict, data: dict) -> PostBlueprint:
    vendor = ctx["vendor"]
    category = ctx.get("category", "software")
    profile = data.get("profile", {})
    signals = data.get("signals", [])
    partner = data.get("partner")

    # Pain radar chart
    pain_data = [
        {"name": s["pain_category"] or "Other", vendor: s["avg_urgency"]}
        for s in signals[:6]
    ]
    pain_chart = ChartSpec(
        chart_id="pain-radar",
        chart_type="radar",
        title=f"Pain Distribution: {vendor}",
        data=pain_data,
        config={
            "x_key": "name",
            "bars": [{"dataKey": vendor, "color": "#f87171"}],
        },
    )

    # Feature gaps chart
    all_gaps: dict[str, int] = {}
    for s in signals:
        for gap in s.get("feature_gaps", []):
            if gap:
                all_gaps[gap] = all_gaps.get(gap, 0) + s["signal_count"]
    top_gaps = sorted(all_gaps.items(), key=lambda x: x[1], reverse=True)[:6]
    gap_chart_data = [{"name": g[:30], "mentions": c} for g, c in top_gaps]

    charts = [pain_chart]
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Hook with the scale of churn signals for {vendor}",
            key_stats={
                "vendor": vendor,
                "category": category,
                "urgency": ctx["urgency"],
                "review_count": ctx["review_count"],
            },
            data_summary=(
                f"{vendor} has {ctx['review_count']} reviews with churn signals "
                f"(avg urgency {ctx['urgency']}/10) in the {category} category."
            ),
        ),
        SectionSpec(
            id="pain_analysis",
            heading=f"What's Driving Users Away from {vendor}?",
            goal="Break down the pain categories causing churn",
            chart_ids=["pain-radar"],
            data_summary=f"Top pain areas: {', '.join(s['pain_category'] for s in signals[:3] if s['pain_category'])}.",
        ),
    ]

    if gap_chart_data:
        gap_chart = ChartSpec(
            chart_id="gaps-bar",
            chart_type="horizontal_bar",
            title=f"Most Requested Features Missing from {vendor}",
            data=gap_chart_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "mentions", "color": "#a78bfa"}],
            },
        )
        charts.append(gap_chart)
        sections.append(SectionSpec(
            id="feature_gaps",
            heading="What Users Wish They Had",
            goal="List the most requested missing features",
            chart_ids=["gaps-bar"],
            data_summary=f"Top {len(gap_chart_data)} feature gaps users mention.",
        ))

    # Alternative spotlight
    alt_name = partner["product_name"] if partner else f"alternatives in {category}"
    sections.append(SectionSpec(
        id="alternative",
        heading=f"The Alternative: {alt_name}" if partner else f"Alternatives in {category}",
        goal="Present the alternative with data-backed strengths",
        key_stats={
            "alternative": alt_name,
            "vendor": vendor,
            **({"affiliate_slug": _slugify(partner["product_name"])} if partner else {}),
        },
        data_summary=(
            f"Profile strengths for {alt_name}: "
            f"{', '.join(s.get('area', str(s)) if isinstance(s, dict) else str(s) for s in profile.get('strengths', [])[:3]) or 'N/A'}."
        ),
    ))

    # Displacement and temporal context (from reasoning pools)
    pool_disp = data.get("pool_displacement") or []
    pool_temporal = data.get("pool_temporal") or {}
    pool_segment = data.get("pool_segment") or {}
    synth_contracts = data.get("synthesis_contracts") or {}
    account_preview = data.get("account_reasoning_preview") or {}
    vendor_core = synth_contracts.get("vendor_core_reasoning") or {}
    displacement_reasoning = synth_contracts.get("displacement_reasoning") or {}
    category_reasoning = synth_contracts.get("category_reasoning") or {}
    account_reasoning = synth_contracts.get("account_reasoning") or {}
    segment_playbook = vendor_core.get("segment_playbook") if isinstance(vendor_core, dict) else {}
    timing_intelligence = vendor_core.get("timing_intelligence") if isinstance(vendor_core, dict) else {}
    causal = vendor_core if isinstance(vendor_core, dict) else {}
    contract_disp = _blog_migration_proof_stats(displacement_reasoning)
    contract_timing = _blog_timing_reasoning_stats(timing_intelligence)
    contract_segment = _blog_segment_reasoning_stats(segment_playbook, timing_intelligence)
    contract_account = _blog_account_reasoning_stats(
        account_reasoning,
        account_preview,
    )
    contract_category = _blog_category_reasoning_stats(category_reasoning)

    has_pool_context = bool(
        pool_disp or pool_temporal or causal or contract_disp or contract_timing
        or contract_segment or contract_account or contract_category
    )
    if has_pool_context:
        context_stats: dict[str, Any] = {"vendor": vendor}
        # Top displacement targets
        if pool_disp:
            top_targets = sorted(
                pool_disp,
                key=lambda e: (e.get("edge_metrics") or {}).get("mention_count") or 0,
                reverse=True,
            )[:3]
            context_stats["displacement_targets"] = [
                {
                    "to_vendor": e.get("to_vendor"),
                    "mentions": (e.get("edge_metrics") or {}).get("mention_count") or 0,
                    "primary_driver": (e.get("edge_metrics") or {}).get("primary_driver"),
                }
                for e in top_targets
            ]
        # Temporal triggers
        tl_summary = pool_temporal.get("timeline_signal_summary") or {}
        if tl_summary.get("renewal_signals") or tl_summary.get("evaluation_deadline_signals"):
            context_stats["renewal_signals"] = tl_summary.get("renewal_signals") or 0
            context_stats["evaluation_deadlines"] = tl_summary.get("evaluation_deadline_signals") or 0
        # Keyword spikes
        spikes = pool_temporal.get("keyword_spikes") or {}
        if spikes.get("spike_count"):
            context_stats["keyword_spike_count"] = spikes["spike_count"]
            context_stats["spike_keywords"] = (spikes.get("spike_keywords") or [])[:5]
        # Segment: decision maker churn
        budget = pool_segment.get("budget_pressure") or {}
        if budget.get("dm_churn_rate") is not None:
            context_stats["dm_churn_rate"] = budget["dm_churn_rate"]
        # Causal narrative
        cn = causal.get("causal_narrative")
        if isinstance(cn, dict) and cn.get("trigger"):
            context_stats["causal_trigger"] = cn["trigger"]
            context_stats["causal_why_now"] = cn.get("why_now")
        context_stats.update(contract_disp)
        context_stats.update(contract_timing)
        context_stats.update(contract_segment)
        context_stats.update(contract_account)
        if contract_category.get("market_regime"):
            context_stats["category_market_regime"] = contract_category["market_regime"]
        if contract_category.get("winner"):
            context_stats["category_winner"] = contract_category["winner"]
        if contract_category.get("loser"):
            context_stats["category_loser"] = contract_category["loser"]
        # Buyer persona distribution from extended context
        _ext_alt = data.get("extended_ctx") or {}
        _buyer_profiles_alt = _ext_alt.get("buyer_profiles") or []
        if _buyer_profiles_alt:
            context_stats["top_buyer_personas"] = [
                {"role": p["role_type"], "stage": p["buying_stage"], "reviews": p["review_count"]}
                for p in _buyer_profiles_alt[:3]
            ]

        sections.append(SectionSpec(
            id="market_context",
            heading=f"Why Users Are Leaving {vendor} Right Now",
            goal="Show displacement patterns, temporal triggers, and buyer context",
            key_stats=context_stats,
        ))

    verdict_stats: dict[str, Any] = {"vendor": vendor, "urgency": ctx["urgency"]}
    wedge = data.get("synthesis_wedge")
    if wedge:
        verdict_stats["synthesis_wedge"] = wedge
        verdict_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    if contract_category.get("market_regime"):
        verdict_stats["market_regime"] = contract_category["market_regime"]
    if contract_account.get("account_pressure_summary"):
        verdict_stats["account_pressure_summary"] = contract_account["account_pressure_summary"]
    if contract_account.get("account_pressure_disclaimer"):
        verdict_stats["account_pressure_disclaimer"] = contract_account["account_pressure_disclaimer"]
    if contract_account.get("account_actionability_tier"):
        verdict_stats["account_actionability_tier"] = contract_account["account_actionability_tier"]

    sections.append(SectionSpec(
        id="verdict",
        heading="The Verdict",
        goal="Summarize findings and recommend action",
        key_stats=verdict_stats,
    ))

    # Build affiliate context
    data_context = {**data["data_context"]}
    if partner:
        data_context["affiliate_url"] = partner["affiliate_url"]
        data_context["affiliate_slug"] = _slugify(partner["product_name"])

    return PostBlueprint(
        topic_type="vendor_alternative",
        slug=ctx["slug"],
        suggested_title=f"{vendor} Alternatives: {ctx['review_count']} Churn Signals Analyzed",
        tags=[category, vendor.lower(), "alternatives", "churn-analysis"],
        data_context=data_context,
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_vendor_showdown(ctx: dict, data: dict) -> PostBlueprint:
    vendor_a, vendor_b = ctx["vendor_a"], ctx["vendor_b"]
    category = ctx.get("category", "software")
    quote_highlights = _blog_quote_highlights(
        data.get("quotes", []),
        limit=4,
        vendors=[vendor_a, vendor_b],
    )

    # Head-to-head comparison chart
    h2h_data = [
        {"name": "Avg Urgency", vendor_a: ctx["urgency_a"], vendor_b: ctx["urgency_b"]},
        {"name": "Review Count", vendor_a: ctx["reviews_a"], vendor_b: ctx["reviews_b"]},
    ]

    # Add pain category comparison from signals
    signals_a = {s["pain_category"]: s["avg_urgency"] for s in data.get("signals_a", [])}
    signals_b = {s["pain_category"]: s["avg_urgency"] for s in data.get("signals_b", [])}
    all_cats = set(signals_a.keys()) | set(signals_b.keys())
    pain_comparison = [
        {"name": cat, vendor_a: signals_a.get(cat, 0), vendor_b: signals_b.get(cat, 0)}
        for cat in sorted(all_cats)
    ][:6]

    h2h_chart = ChartSpec(
        chart_id="head2head-bar",
        chart_type="horizontal_bar",
        title=f"{vendor_a} vs {vendor_b}: Key Metrics",
        data=h2h_data,
        config={
            "x_key": "name",
            "bars": [
                {"dataKey": vendor_a, "color": "#22d3ee"},
                {"dataKey": vendor_b, "color": "#f472b6"},
            ],
        },
    )

    charts = [h2h_chart]
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal="Hook with the contrast between the two vendors",
            key_stats={
                "vendor_a": vendor_a,
                "vendor_b": vendor_b,
                "category": category,
                "reviews_a": ctx["reviews_a"],
                "reviews_b": ctx["reviews_b"],
                "urgency_a": ctx["urgency_a"],
                "urgency_b": ctx["urgency_b"],
                "pain_diff": ctx["pain_diff"],
            },
            data_summary=(
                f"{vendor_a} ({ctx['reviews_a']} signals, urgency {ctx['urgency_a']}) "
                f"vs {vendor_b} ({ctx['reviews_b']} signals, urgency {ctx['urgency_b']}). "
                f"Urgency difference: {ctx['pain_diff']}."
            ),
        ),
        SectionSpec(
            id="head2head",
            heading=f"{vendor_a} vs {vendor_b}: By the Numbers",
            goal="Present core metrics side by side",
            chart_ids=["head2head-bar"],
            data_summary=f"Comparing churn signals and urgency across both vendors.",
        ),
    ]

    if pain_comparison:
        pain_chart = ChartSpec(
            chart_id="pain-comparison-bar",
            chart_type="bar",
            title=f"Pain Categories: {vendor_a} vs {vendor_b}",
            data=pain_comparison,
            config={
                "x_key": "name",
                "bars": [
                    {"dataKey": vendor_a, "color": "#22d3ee"},
                    {"dataKey": vendor_b, "color": "#f472b6"},
                ],
            },
        )
        charts.append(pain_chart)
        sections.append(SectionSpec(
            id="pain_breakdown",
            heading="Where Each Vendor Falls Short",
            goal="Compare pain categories between both vendors",
            chart_ids=["pain-comparison-bar"],
            data_summary=f"Pain category comparison across {len(pain_comparison)} categories.",
        ))

    # Displacement dynamics section (from reasoning pools)
    disp_a_to_b = data.get("displacement_a_to_b") or {}
    disp_b_to_a = data.get("displacement_b_to_a") or {}
    edge_a = disp_a_to_b.get("edge_metrics") or {}
    edge_b = disp_b_to_a.get("edge_metrics") or {}
    xv_lookup = data.get("xv_synthesis_lookup") or {}
    battle_a = _resolve_blog_battle_summary(
        vendor_a, vendor_b, xv_lookup,
        disp_a_to_b.get("battle_summary") or {},
    )
    flow_a = disp_a_to_b.get("flow_summary") or {}
    switch_reasons = disp_a_to_b.get("switch_reasons") or []

    has_displacement = bool(edge_a.get("mention_count") or edge_b.get("mention_count"))
    if has_displacement:
        disp_stats: dict[str, Any] = {
            "a_to_b_mentions": edge_a.get("mention_count") or 0,
            "b_to_a_mentions": edge_b.get("mention_count") or 0,
            "a_to_b_signal_strength": edge_a.get("signal_strength"),
            "a_to_b_primary_driver": edge_a.get("primary_driver"),
            "explicit_switches": flow_a.get("explicit_switch_count") or 0,
            "active_evaluations": flow_a.get("active_evaluation_count") or 0,
        }
        if switch_reasons:
            disp_stats["top_switch_reasons"] = [
                {"reason": r.get("reason_category") or r.get("reason"), "count": r.get("mention_count", 0)}
                for r in switch_reasons[:5]
            ]
        if battle_a.get("conclusion"):
            disp_stats["battle_conclusion"] = battle_a["conclusion"]
            disp_stats["battle_winner"] = battle_a.get("winner")
            disp_stats["battle_confidence"] = battle_a.get("confidence")
            disp_stats["battle_durability"] = battle_a.get("durability_assessment")
        sections.append(SectionSpec(
            id="displacement",
            heading="Who Is Actually Switching?",
            goal="Show displacement patterns: who leaves whom, why, and how fast",
            key_stats=disp_stats,
            data_summary=(
                f"{edge_a.get('mention_count') or 0} displacement signals from {vendor_a} to {vendor_b}, "
                f"{edge_b.get('mention_count') or 0} from {vendor_b} to {vendor_a}."
            ),
        ))

    # Segment intelligence (from reasoning pools)
    seg_a = data.get("pool_segment_a") or {}
    seg_b = data.get("pool_segment_b") or {}
    roles_a = seg_a.get("affected_roles") or []
    roles_b = seg_b.get("affected_roles") or []
    has_segments = bool(roles_a or roles_b)
    if has_segments:
        seg_stats: dict[str, Any] = {}
        if roles_a:
            seg_stats["roles_a"] = [
                {"role": r.get("role_type"), "count": r.get("review_count", 0), "churn_rate": r.get("churn_rate")}
                for r in roles_a[:5]
            ]
        if roles_b:
            seg_stats["roles_b"] = [
                {"role": r.get("role_type"), "count": r.get("review_count", 0), "churn_rate": r.get("churn_rate")}
                for r in roles_b[:5]
            ]
        budget_a = seg_a.get("budget_pressure") or {}
        budget_b = seg_b.get("budget_pressure") or {}
        if budget_a.get("dm_churn_rate") is not None:
            seg_stats["dm_churn_rate_a"] = budget_a["dm_churn_rate"]
        if budget_b.get("dm_churn_rate") is not None:
            seg_stats["dm_churn_rate_b"] = budget_b["dm_churn_rate"]
        sections.append(SectionSpec(
            id="buyer_segments",
            heading="Who Is Churning? Buyer Profile Breakdown",
            goal="Show which buyer roles and segments are most affected",
            key_stats=seg_stats,
        ))

    # Category dynamics + synthesis (from reasoning pools + cross-vendor synthesis)
    cat_dyn = data.get("pool_category") or {}
    regime = cat_dyn.get("market_regime") or {}
    council = _resolve_blog_council_summary(
        category, xv_lookup, cat_dyn.get("council_summary") or {},
    )
    synth_contracts = data.get("synthesis_contracts") or {}
    vendor_core = synth_contracts.get("vendor_core_reasoning") or {}
    category_reasoning = synth_contracts.get("category_reasoning") or {}
    contract_category = _blog_category_reasoning_stats(category_reasoning)
    causal = vendor_core if isinstance(vendor_core, dict) else {}

    verdict_stats: dict[str, Any] = {
        "vendor_a": vendor_a,
        "vendor_b": vendor_b,
        "urgency_a": ctx["urgency_a"],
        "urgency_b": ctx["urgency_b"],
    }
    if regime.get("regime_type"):
        verdict_stats["market_regime"] = regime["regime_type"]
    if council.get("conclusion"):
        verdict_stats["category_conclusion"] = council["conclusion"]
    elif contract_category.get("narrative"):
        verdict_stats["category_conclusion"] = contract_category["narrative"]
    if battle_a.get("winner"):
        verdict_stats["displacement_winner"] = battle_a["winner"]
        verdict_stats["displacement_confidence"] = battle_a.get("confidence")
    if causal.get("causal_narrative"):
        cn = causal["causal_narrative"]
        if isinstance(cn, dict):
            verdict_stats["causal_trigger"] = cn.get("trigger")
            verdict_stats["causal_why_now"] = cn.get("why_now")
    wedge = data.get("synthesis_wedge")
    if wedge:
        verdict_stats["synthesis_wedge"] = wedge
        verdict_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    if contract_category.get("market_regime"):
        verdict_stats["category_market_regime"] = contract_category["market_regime"]
    if contract_category.get("winner"):
        verdict_stats["category_winner"] = contract_category["winner"]
    if contract_category.get("loser"):
        verdict_stats["category_loser"] = contract_category["loser"]

    sections.append(SectionSpec(
        id="verdict",
        heading="The Verdict",
        goal="Declare which vendor fares better and the decisive factor",
        key_stats=verdict_stats,
        data_summary=f"Final comparison of urgency, displacement, and buyer-fit signals for {vendor_a} and {vendor_b}.",
    ))

    if quote_highlights:
        sections.append(SectionSpec(
            id="reviewer_voice",
            heading=f"What Reviewers Say About {vendor_a} and {vendor_b}",
            goal="Keep the comparison grounded in direct reviewer language from both vendors",
            key_stats={"quote_highlights": quote_highlights},
            data_summary=f"{len(quote_highlights)} quote-backed snippets comparing {vendor_a} and {vendor_b}.",
        ))

    return PostBlueprint(
        topic_type="vendor_showdown",
        slug=ctx["slug"],
        suggested_title=f"{vendor_a} vs {vendor_b}: Comparing Reviewer Complaints Across {ctx['total_reviews']} Reviews",
        tags=[category, vendor_a.lower(), vendor_b.lower(), "comparison", "churn-analysis"],
        data_context=data["data_context"],
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_churn_report(ctx: dict, data: dict) -> PostBlueprint:
    vendor = ctx["vendor"]
    category = ctx.get("category", "software")
    signals = data.get("signals", [])
    profile = data.get("profile", {})
    market_regime = (data.get("category_overview", {}).get("cross_vendor_analysis") or {}).get("market_regime")
    synth_contracts = data.get("synthesis_contracts") or {}
    account_preview = data.get("account_reasoning_preview") or {}
    vendor_core = synth_contracts.get("vendor_core_reasoning") or {}
    displacement_reasoning = synth_contracts.get("displacement_reasoning") or {}
    category_reasoning = synth_contracts.get("category_reasoning") or {}
    account_reasoning = synth_contracts.get("account_reasoning") or {}
    segment_playbook = vendor_core.get("segment_playbook") if isinstance(vendor_core, dict) else {}
    timing_intelligence = vendor_core.get("timing_intelligence") if isinstance(vendor_core, dict) else {}
    contract_disp = _blog_migration_proof_stats(displacement_reasoning)
    contract_segment = _blog_segment_reasoning_stats(segment_playbook, timing_intelligence)
    contract_timing = _blog_timing_reasoning_stats(timing_intelligence)
    contract_category = _blog_category_reasoning_stats(category_reasoning)
    contract_account = _blog_account_reasoning_stats(
        account_reasoning,
        account_preview,
    )
    if not market_regime:
        market_regime = contract_category.get("market_regime")

    # Pain distribution chart
    pain_data = [
        {"name": s["pain_category"] or "Other", "signals": s["signal_count"], "urgency": s["avg_urgency"]}
        for s in signals[:8]
    ]
    pain_chart = ChartSpec(
        chart_id="pain-bar",
        chart_type="bar",
        title=f"Churn Pain Categories: {vendor}",
        data=pain_data,
        config={
            "x_key": "name",
            "bars": [
                {"dataKey": "signals", "color": "#f87171"},
                {"dataKey": "urgency", "color": "#fbbf24"},
            ],
        },
    )

    # Feature gaps
    all_gaps: dict[str, int] = {}
    for s in signals:
        for gap in s.get("feature_gaps", []):
            if gap:
                all_gaps[gap] = all_gaps.get(gap, 0) + 1
    top_gaps = sorted(all_gaps.items(), key=lambda x: x[1], reverse=True)[:6]
    gap_data = [{"name": g[:30], "mentions": c} for g, c in top_gaps]

    charts = [pain_chart]
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Lead with the scale of churn signals for {vendor}",
            key_stats={
                "vendor": vendor,
                "category": category,
                "negative_reviews": ctx["negative_reviews"],
                "avg_urgency": ctx["avg_urgency"],
                "total_reviews": ctx["total_reviews"],
                "market_regime": market_regime,
            },
            data_summary=(
                f"{vendor} has {ctx['negative_reviews']} negative reviews out of "
                f"{ctx['total_reviews']} total (avg urgency {ctx['avg_urgency']}/10)."
            ),
        ),
        SectionSpec(
            id="pain_breakdown",
            heading="What's Causing the Churn?",
            goal="Group pain points by category",
            chart_ids=["pain-bar"],
            data_summary=f"Top pain categories: {', '.join(s['pain_category'] for s in signals[:3] if s['pain_category'])}.",
        ),
    ]
    if market_regime:
        sections.append(SectionSpec(
            id="market_context",
            heading=f"Market Context for {category}",
            goal="Explain how the broader category regime changes the interpretation of this vendor's churn signals",
            key_stats={"market_regime": market_regime},
            data_summary=f"Current market regime: {market_regime}.",
        ))

    if gap_data:
        gap_chart = ChartSpec(
            chart_id="gaps-bar",
            chart_type="horizontal_bar",
            title=f"Feature Gaps Driving Churn: {vendor}",
            data=gap_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "mentions", "color": "#a78bfa"}],
            },
        )
        charts.append(gap_chart)
        sections.append(SectionSpec(
            id="feature_gaps",
            heading="What's Missing?",
            goal="List the feature gaps driving users away",
            chart_ids=["gaps-bar"],
            data_summary=f"Top {len(gap_data)} missing features.",
        ))

    # Reasoning pool enrichment: displacement, segment, temporal
    pool_disp = data.get("pool_displacement") or []
    pool_segment = data.get("pool_segment") or {}
    pool_temporal = data.get("pool_temporal") or {}
    if pool_disp:
        top_targets = sorted(
            pool_disp,
            key=lambda e: (e.get("edge_metrics") or {}).get("mention_count") or 0,
            reverse=True,
        )[:5]
        disp_stats: dict[str, Any] = {
            "targets": [
                {
                    "to_vendor": e.get("to_vendor"),
                    "mentions": (e.get("edge_metrics") or {}).get("mention_count") or 0,
                    "primary_driver": (e.get("edge_metrics") or {}).get("primary_driver"),
                    "signal_strength": (e.get("edge_metrics") or {}).get("signal_strength"),
                }
                for e in top_targets
            ],
        }
        sections.append(SectionSpec(
            id="displacement",
            heading=f"Where {vendor} Users Are Going",
            goal="Show which alternatives are gaining traction and why",
            key_stats=disp_stats,
        ))
    elif contract_disp:
        sections.append(SectionSpec(
            id="displacement",
            heading=f"Where {vendor} Users Are Going",
            goal="Show which alternatives are gaining traction and why",
            key_stats=contract_disp,
        ))

    # Buyer segment breakdown
    roles = pool_segment.get("affected_roles") or []
    if roles:
        seg_stats: dict[str, Any] = {
            "roles": [
                {"role": r.get("role_type"), "count": r.get("review_count", 0), "churn_rate": r.get("churn_rate")}
                for r in roles[:5]
            ],
        }
        budget = pool_segment.get("budget_pressure") or {}
        if budget.get("dm_churn_rate") is not None:
            seg_stats["dm_churn_rate"] = budget["dm_churn_rate"]
        sections.append(SectionSpec(
            id="buyer_segments",
            heading="Who Is Churning?",
            goal="Break down churn by buyer role and seniority",
            key_stats=seg_stats,
        ))
    elif contract_segment:
        sections.append(SectionSpec(
            id="buyer_segments",
            heading="Who Is Churning?",
            goal="Break down churn by buyer role and seniority",
            key_stats=contract_segment,
        ))

    # Temporal context
    tl_summary = pool_temporal.get("timeline_signal_summary") or {}
    sentiment = pool_temporal.get("sentiment_trajectory") or {}
    has_temporal = tl_summary.get("renewal_signals") or sentiment.get("declining_pct")
    if has_temporal:
        temporal_stats: dict[str, Any] = {}
        if tl_summary.get("renewal_signals"):
            temporal_stats["renewal_signals"] = tl_summary["renewal_signals"]
        if tl_summary.get("evaluation_deadline_signals"):
            temporal_stats["evaluation_deadlines"] = tl_summary["evaluation_deadline_signals"]
        if sentiment.get("declining_pct") is not None:
            temporal_stats["declining_pct"] = sentiment["declining_pct"]
            temporal_stats["improving_pct"] = sentiment.get("improving_pct")
        sections.append(SectionSpec(
            id="timing",
            heading="Timing Signals: When to Act",
            goal="Show contract renewal windows and sentiment trajectory",
            key_stats=temporal_stats,
        ))
    elif contract_timing:
        sections.append(SectionSpec(
            id="timing",
            heading="Timing Signals: When to Act",
            goal="Show contract renewal windows and sentiment trajectory",
            key_stats=contract_timing,
        ))

    outlook_stats: dict[str, Any] = {"vendor": vendor, "avg_urgency": ctx["avg_urgency"]}
    causal = vendor_core.get("causal_narrative") if isinstance(vendor_core, dict) else {}
    if isinstance(causal, dict) and causal.get("trigger"):
        outlook_stats["causal_trigger"] = causal["trigger"]
        outlook_stats["causal_why_now"] = causal.get("why_now")
    wedge = data.get("synthesis_wedge")
    if wedge:
        outlook_stats["synthesis_wedge"] = wedge
    if contract_account.get("account_pressure_summary"):
        outlook_stats["account_pressure_summary"] = contract_account["account_pressure_summary"]
    if contract_account.get("account_pressure_disclaimer"):
        outlook_stats["account_pressure_disclaimer"] = contract_account["account_pressure_disclaimer"]
    if contract_account.get("account_actionability_tier"):
        outlook_stats["account_actionability_tier"] = contract_account["account_actionability_tier"]
    if contract_category.get("narrative"):
        outlook_stats["category_narrative"] = contract_category["narrative"]

    sections.append(SectionSpec(
        id="outlook",
        heading="What This Means for Teams Using " + vendor,
        goal="Provide actionable guidance for current users",
        key_stats=outlook_stats,
    ))

    return PostBlueprint(
        topic_type="churn_report",
        slug=ctx["slug"],
        suggested_title=f"{vendor} Churn Report: {ctx['negative_reviews']} Negative Reviews Analyzed",
        tags=[category, vendor.lower(), "churn-report", "enterprise-software"],
        data_context=data["data_context"],
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_migration_guide(ctx: dict, data: dict) -> PostBlueprint:
    vendor = ctx["vendor"]
    category = ctx.get("category", "software")
    profile = data.get("profile", {})
    signals = data.get("signals", [])
    synth_contracts = data.get("synthesis_contracts") or {}
    account_preview = data.get("account_reasoning_preview") or {}
    vendor_core = synth_contracts.get("vendor_core_reasoning") or {}
    displacement_reasoning = synth_contracts.get("displacement_reasoning") or {}
    category_reasoning = synth_contracts.get("category_reasoning") or {}
    account_reasoning = synth_contracts.get("account_reasoning") or {}
    timing_intelligence = vendor_core.get("timing_intelligence") if isinstance(vendor_core, dict) else {}
    # contract_disp intentionally omitted: migration_proof is outbound-oriented,
    # wrong direction for a switch-to guide.
    contract_category = _blog_category_reasoning_stats(category_reasoning)

    # Migration sources chart
    switched_from = profile.get("commonly_switched_from", [])
    if isinstance(switched_from, str):
        try:
            switched_from = json.loads(switched_from)
        except (json.JSONDecodeError, TypeError):
            switched_from = []

    source_data = [
        {
            "name": (src.get("vendor", "Unknown") if isinstance(src, dict) else str(src))[:25],
            "migrations": src.get("count", 1) if isinstance(src, dict) else 1,
        }
        for src in switched_from[:8]
        if (src.get("vendor", "") if isinstance(src, dict) else str(src)).lower().strip() != vendor.lower().strip()
    ]

    inbound_source_count = len(source_data)
    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Highlight the volume of inbound migrations to {vendor}",
            key_stats={
                "vendor": vendor,
                "category": category,
                "inbound_source_count": inbound_source_count,
                "review_total": ctx["review_total"],
            },
            data_summary=(
                f"{vendor} attracts users from {inbound_source_count} documented competitor platforms "
                f"based on analysis of {ctx['review_total']} total reviews."
            ),
        ),
    ]

    if source_data:
        source_chart = ChartSpec(
            chart_id="sources-bar",
            chart_type="horizontal_bar",
            title=f"Where {vendor} Users Come From",
            data=source_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "migrations", "color": "#34d399"}],
            },
        )
        charts.append(source_chart)
        sections.append(SectionSpec(
            id="sources",
            heading=f"Where Are {vendor} Users Coming From?",
            goal="Show the top migration sources",
            chart_ids=["sources-bar"],
            data_summary=f"Top {len(source_data)} competitors users are leaving for {vendor}.",
        ))

    # Pain of origin chart (what drove them away from competitors)
    if signals:
        pain_data = [
            {"name": s["pain_category"] or "Other", "signals": s["signal_count"]}
            for s in signals[:6]
        ]
        pain_chart = ChartSpec(
            chart_id="pain-bar",
            chart_type="bar",
            title=f"Pain Categories That Drive Migration to {vendor}",
            data=pain_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "signals", "color": "#f87171"}],
            },
        )
        charts.append(pain_chart)
        sections.append(SectionSpec(
            id="triggers",
            heading="What Triggers the Switch?",
            goal="Explain the common pain categories behind migration",
            chart_ids=["pain-bar"],
            data_summary=f"Top pain categories driving migration.",
        ))

    # Use mention-counted integrations from extended context when available
    _ext_ctx = data.get("extended_ctx") or {}
    _ext_ints = _ext_ctx.get("integrations") or []
    _migration_integrations = (
        [{"name": r["integration_name"], "mentions": r["mention_count"]} for r in _ext_ints[:5]]
        if _ext_ints
        else (profile.get("integrations", [])[:5] if isinstance(profile.get("integrations"), list) else [])
    )
    sections.append(SectionSpec(
        id="practical",
        heading="Making the Switch: What to Expect",
        goal="Practical migration considerations (integrations, learning curve)",
        key_stats={
            "vendor": vendor,
            "integrations": _migration_integrations,
        },
    ))

    # Inbound migration count from commonly_switched_from
    inbound_migration_count = sum(
        (src.get("count", 1) if isinstance(src, dict) else 1)
        for src in switched_from
        if (src.get("vendor", "") if isinstance(src, dict) else str(src)).lower().strip() != vendor.lower().strip()
    )
    takeaway_stats: dict[str, Any] = {
        "vendor": vendor,
        "inbound_source_count": len(source_data),
        "inbound_migration_mentions": inbound_migration_count,
    }

    # Pool displacement: filter to INBOUND edges only (to_vendor = this vendor)
    pool_disp = data.get("pool_displacement") or []
    inbound_disp = [
        e for e in pool_disp
        if str(e.get("to_vendor") or "").lower().strip() == vendor.lower().strip()
    ]
    if inbound_disp:
        top = sorted(inbound_disp, key=lambda e: (e.get("edge_metrics") or {}).get("mention_count") or 0, reverse=True)
        if top:
            em = top[0].get("edge_metrics") or {}
            takeaway_stats["top_inbound_driver"] = em.get("primary_driver")

    # Drop outbound-oriented contract_disp (migration_proof describes why
    # customers LEAVE, not why they JOIN -- wrong direction for this article).
    # Only inject inbound-safe fields from reasoning.
    causal = (data.get("synthesis_contracts") or {}).get("vendor_core_reasoning") or {}

    # --- Reasoning wedge injection (AEO authority) ---
    wedge = data.get("synthesis_wedge")
    if wedge:
        takeaway_stats["synthesis_wedge"] = wedge
        takeaway_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    cn = causal.get("causal_narrative")
    if isinstance(cn, dict) and cn.get("trigger"):
        takeaway_stats["causal_trigger"] = cn["trigger"]
        takeaway_stats["causal_why_now"] = cn.get("why_now")
    if contract_category.get("market_regime"):
        takeaway_stats["market_regime"] = contract_category["market_regime"]

    sections.append(SectionSpec(
        id="takeaway",
        heading="Key Takeaways",
        goal="Summary and recommendations",
        key_stats=takeaway_stats,
    ))

    return PostBlueprint(
        topic_type="migration_guide",
        slug=ctx["slug"],
        suggested_title=f"Migration Guide: Why Teams Are Switching to {vendor}",
        tags=[category, vendor.lower(), "migration", "switching-guide"],
        data_context=data["data_context"],
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_vendor_deep_dive(ctx: dict, data: dict) -> PostBlueprint:
    """In-depth profile of a single vendor -- showcase data gathering capabilities."""
    vendor = ctx["vendor"]
    category = ctx.get("category", "software")
    profile = data.get("profile", {})
    signals = data.get("signals", [])
    competitor_profiles = data.get("competitor_profiles", [])
    synth_contracts = data.get("synthesis_contracts") or {}
    account_preview = data.get("account_reasoning_preview") or {}
    vendor_core = synth_contracts.get("vendor_core_reasoning") or {}
    category_reasoning = synth_contracts.get("category_reasoning") or {}
    account_reasoning = synth_contracts.get("account_reasoning") or {}
    segment_playbook = vendor_core.get("segment_playbook") if isinstance(vendor_core, dict) else {}
    timing_intelligence = vendor_core.get("timing_intelligence") if isinstance(vendor_core, dict) else {}
    contract_segment = _blog_segment_reasoning_stats(segment_playbook, timing_intelligence)
    contract_timing = _blog_timing_reasoning_stats(timing_intelligence)
    contract_account = _blog_account_reasoning_stats(
        account_reasoning,
        account_preview,
    )
    contract_category = _blog_category_reasoning_stats(category_reasoning)
    quote_highlights = _blog_quote_highlights(
        data.get("quotes", []),
        limit=4,
        vendors=[vendor],
    )

    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Position this as a comprehensive, data-driven look at {vendor}",
            key_stats={
                "vendor": vendor,
                "category": category,
                "review_count": ctx["review_count"],
                "profile_richness": ctx["profile_richness"],
            },
            data_summary=(
                f"A deep dive into {vendor} based on {ctx['review_count']} reviews "
                f"and cross-referenced data from multiple B2B intelligence sources."
            ),
        ),
    ]

    # Strengths vs weaknesses chart
    strengths = profile.get("strengths", [])
    weaknesses = profile.get("weaknesses", [])
    # When the product profile is too thin, build from review sentiment
    if len(strengths) + len(weaknesses) < 3 and signals:
        area_map: dict[str, dict] = {}
        for s in signals:
            cat = s.get("pain_category", "")
            if not cat or cat in ("None", "null", "none"):
                continue
            urg = float(s.get("avg_urgency", 0))
            cnt = int(s.get("signal_count", 1))
            area_map.setdefault(cat, {"name": cat, "strengths": 0, "weaknesses": 0})
            if urg >= 3.0:
                area_map[cat]["weaknesses"] += cnt
            else:
                area_map[cat]["strengths"] += cnt
        sw_data = sorted(area_map.values(), key=lambda x: x["strengths"] + x["weaknesses"], reverse=True)[:8]
    elif strengths or weaknesses:
        # Merge by area so each bar shows strength vs weakness evidence
        area_map: dict[str, dict] = {}
        for s in strengths[:8]:
            name = str(s.get("area", s) if isinstance(s, dict) else s)[:30]
            count = int(s.get("evidence_count", 1)) if isinstance(s, dict) else 1
            area_map.setdefault(name, {"name": name, "strengths": 0, "weaknesses": 0})
            area_map[name]["strengths"] += count
        for w in weaknesses[:8]:
            name = str(w.get("area", w) if isinstance(w, dict) else w)[:30]
            count = int(w.get("evidence_count", 1)) if isinstance(w, dict) else 1
            area_map.setdefault(name, {"name": name, "strengths": 0, "weaknesses": 0})
            area_map[name]["weaknesses"] += count
        sw_data = sorted(area_map.values(), key=lambda x: x["strengths"] + x["weaknesses"], reverse=True)[:8]
    else:
        sw_data = []
    if sw_data:
        sw_chart = ChartSpec(
            chart_id="strengths-weaknesses",
            chart_type="horizontal_bar",
            title=f"{vendor}: Strengths vs Weaknesses",
            data=sw_data,
            config={
                "x_key": "name",
                "bars": [
                    {"dataKey": "strengths", "color": "#34d399"},
                    {"dataKey": "weaknesses", "color": "#f87171"},
                ],
            },
        )
        charts.append(sw_chart)
        sections.append(SectionSpec(
            id="strengths_weaknesses",
            heading=f"What {vendor} Does Well -- and Where It Falls Short",
            goal="Present strengths and weaknesses from real user data",
            chart_ids=["strengths-weaknesses"],
            data_summary=f"{len(strengths)} strengths and {len(weaknesses)} weaknesses identified.",
        ))

    # Pain signals chart
    if signals:
        pain_data = [
            {"name": s["pain_category"] or "Other", "urgency": s["avg_urgency"]}
            for s in signals[:6]
        ]
        pain_chart = ChartSpec(
            chart_id="pain-radar",
            chart_type="radar",
            title=f"User Pain Areas: {vendor}",
            data=pain_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "urgency", "color": "#f87171"}],
            },
        )
        charts.append(pain_chart)
        sections.append(SectionSpec(
            id="pain_analysis",
            heading=f"Where {vendor} Users Feel the Most Pain",
            goal="Break down the top pain categories from review analysis",
            chart_ids=["pain-radar"],
        ))

    # Integrations and use cases -- prefer mention-counted extended context over flat profile arrays
    extended_ctx = data.get("extended_ctx") or {}
    ext_integrations = extended_ctx.get("integrations") or []
    ext_use_cases = extended_ctx.get("use_cases") or []
    ext_buyer_profiles = extended_ctx.get("buyer_profiles") or []
    integrations = profile.get("integrations", [])
    use_cases = profile.get("use_cases", [])
    if ext_integrations or ext_use_cases or integrations or use_cases:
        ecosystem_integrations = (
            [{"name": r["integration_name"], "mentions": r["mention_count"]}
             for r in ext_integrations[:8]]
            if ext_integrations
            else [str(i)[:30] for i in integrations[:8]] if isinstance(integrations, list) else []
        )
        ecosystem_use_cases = (
            [{"name": r["use_case_name"], "mentions": r["mention_count"],
              "urgency": round(float(r.get("avg_urgency") or 0), 1)}
             for r in ext_use_cases[:6]]
            if ext_use_cases
            else [str(u)[:40] for u in use_cases[:6]] if isinstance(use_cases, list) else []
        )
        effective_int_count = len(ext_integrations) if ext_integrations else len(integrations) if isinstance(integrations, list) else 0
        effective_uc_count = len(ext_use_cases) if ext_use_cases else len(use_cases) if isinstance(use_cases, list) else 0
        sections.append(SectionSpec(
            id="ecosystem",
            heading=f"The {vendor} Ecosystem: Integrations & Use Cases",
            goal="Show the product ecosystem and typical deployment scenarios",
            key_stats={
                "integrations": ecosystem_integrations,
                "use_cases": ecosystem_use_cases,
            },
            data_summary=f"{effective_int_count} integrations and {effective_uc_count} primary use cases.",
        ))

    if ext_buyer_profiles:
        sections.append(SectionSpec(
            id="buyer_profiles",
            heading=f"Who Reviews {vendor}: Buyer Personas",
            goal="Show the distribution of buyer roles and purchase stages to anchor persona targeting",
            key_stats={
                "top_buyer_roles": [
                    {"role": p["role_type"], "stage": p["buying_stage"],
                     "reviews": p["review_count"]}
                    for p in ext_buyer_profiles[:5]
                ],
            },
            data_summary=f"Top buyer roles: {', '.join(p['role_type'] for p in ext_buyer_profiles[:3] if p.get('role_type'))}.",
        ))

    if contract_segment:
        sections.append(SectionSpec(
            id="segment_timing",
            heading=f"Which Teams Feel {vendor} Pain First",
            goal="Translate segment targeting and timing intelligence into who feels the pressure first and why",
            key_stats=contract_segment,
            data_summary=(
                str(contract_segment.get("segment_targeting_summary") or "").strip()
                or f"Priority segments surfaced from reasoning for {vendor}."
            ),
        ))

    if contract_timing:
        sections.append(SectionSpec(
            id="timing_signals",
            heading=f"When {vendor} Friction Turns Into Action",
            goal="Show renewal windows, sentiment direction, and timing triggers that make dissatisfaction operational",
            key_stats=contract_timing,
            data_summary=(
                str(contract_timing.get("timing_summary") or "").strip()
                or f"Timing intelligence surfaced for {vendor}."
            ),
        ))

    if contract_account:
        sections.append(SectionSpec(
            id="account_pressure",
            heading=f"Where {vendor} Pressure Shows Up in Accounts",
            goal="Surface named-account and account-pattern pressure without overstating certainty",
            key_stats=contract_account,
            data_summary=(
                str(contract_account.get("account_pressure_summary") or "").strip()
                or f"Account-pressure patterns were surfaced for {vendor}."
            ),
        ))

    # Competitive landscape
    compared = profile.get("commonly_compared_to", [])
    if compared:
        comp_names = [
            (c.get("vendor", c) if isinstance(c, dict) else str(c))[:25]
            for c in compared[:6]
        ]
        sections.append(SectionSpec(
            id="competitive_landscape",
            heading=f"How {vendor} Stacks Up Against Competitors",
            goal="Position the vendor relative to frequently compared alternatives",
            key_stats={"competitors": comp_names},
            data_summary=f"Commonly compared to: {', '.join(comp_names)}.",
        ))

    market_position_stats: dict[str, Any] = {}
    if contract_category:
        market_position_stats.update(contract_category)
    if competitor_profiles:
        market_position_stats["competitor_snapshots"] = [
            {
                "vendor": str(profile_row.get("vendor_name") or profile_row.get("vendor") or "").strip(),
                "strengths": [
                    str(item.get("area", item)) if isinstance(item, dict) else str(item)
                    for item in (profile_row.get("strengths") or [])[:2]
                ],
                "weaknesses": [
                    str(item.get("area", item)) if isinstance(item, dict) else str(item)
                    for item in (profile_row.get("weaknesses") or [])[:2]
                ],
            }
            for profile_row in competitor_profiles[:3]
            if str(profile_row.get("vendor_name") or profile_row.get("vendor") or "").strip()
        ]
    if market_position_stats:
        sections.append(SectionSpec(
            id="market_position",
            heading=f"Where {vendor} Sits in the {category} Market",
            goal="Connect the vendor profile to broader category dynamics and the nearest competitive alternatives",
            key_stats=market_position_stats,
            data_summary=(
                str(contract_category.get("narrative") or "").strip()
                or f"Competitive context for {vendor} in the {category} market."
            ),
        ))

    if quote_highlights:
        sections.append(SectionSpec(
            id="reviewer_voice",
            heading=f"What Reviewers Actually Say About {vendor}",
            goal="Anchor the analysis in direct review language so the conclusions stay evidence-backed",
            key_stats={"quote_highlights": quote_highlights},
            data_summary=f"{len(quote_highlights)} representative quote-backed snippets for {vendor}.",
        ))

    verdict_stats: dict[str, Any] = {"vendor": vendor, "review_count": ctx["review_count"]}
    pool_segment = data.get("pool_segment") or {}
    pool_temporal = data.get("pool_temporal") or {}
    roles = pool_segment.get("affected_roles") or []
    if roles:
        verdict_stats["top_churning_role"] = roles[0].get("role_type")
        verdict_stats["top_role_churn_rate"] = roles[0].get("churn_rate")
    sentiment = pool_temporal.get("sentiment_trajectory") or {}
    if sentiment.get("declining_pct") is not None:
        verdict_stats["declining_pct"] = sentiment["declining_pct"]
    wedge = data.get("synthesis_wedge")
    if wedge:
        verdict_stats["synthesis_wedge"] = wedge
    if contract_segment.get("segment_targeting_summary"):
        verdict_stats["segment_targeting_summary"] = contract_segment["segment_targeting_summary"]
    if contract_timing.get("timing_summary"):
        verdict_stats["timing_summary"] = contract_timing["timing_summary"]
    if contract_account.get("account_pressure_summary"):
        verdict_stats["account_pressure_summary"] = contract_account["account_pressure_summary"]
    if contract_account.get("account_pressure_disclaimer"):
        verdict_stats["account_pressure_disclaimer"] = contract_account["account_pressure_disclaimer"]
    if contract_account.get("account_actionability_tier"):
        verdict_stats["account_actionability_tier"] = contract_account["account_actionability_tier"]
    if contract_category.get("market_regime"):
        verdict_stats["market_regime"] = contract_category["market_regime"]

    sections.append(SectionSpec(
        id="verdict",
        heading=f"The Bottom Line on {vendor}",
        goal="Synthesize all data into actionable guidance for potential buyers",
        key_stats=verdict_stats,
        data_summary=f"Final synthesis of category position, buyer pressure, and timing risk for {vendor}.",
    ))

    return PostBlueprint(
        topic_type="vendor_deep_dive",
        slug=ctx["slug"],
        suggested_title=f"{vendor} Deep Dive: Reviewer Sentiment Across {ctx['review_count']} Reviews",
        tags=[category, vendor.lower(), "deep-dive", "vendor-profile", "b2b-intelligence"],
        data_context=data["data_context"],
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_market_landscape(ctx: dict, data: dict) -> PostBlueprint:
    """Category-wide overview comparing all vendors in a space."""
    category = ctx["category"]
    vendor_count = ctx["vendor_count"]
    vendor_profiles = data.get("vendor_profiles", [])
    vendor_signals = data.get("vendor_signals", [])
    market_regime = (data.get("category_overview", {}).get("cross_vendor_analysis") or {}).get("market_regime")
    pain_rows = _aggregate_category_pain_rows(vendor_signals, limit=6)
    quote_highlights = _blog_quote_highlights(data.get("quotes", []), limit=4)
    signals_by_vendor = {
        str(item.get("vendor") or "").strip(): item.get("signals") or []
        for item in vendor_signals
        if isinstance(item, dict)
    }

    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Frame this as a comprehensive market overview of the {category} space",
            key_stats={
                "category": category,
                "vendor_count": vendor_count,
                "total_reviews": ctx["total_reviews"],
                "avg_urgency": ctx["avg_urgency"],
                "market_regime": market_regime,
            },
            data_summary=(
                f"The {category} landscape has {vendor_count} major vendors "
                f"with {ctx['total_reviews']} total churn signals analyzed."
            ),
        ),
    ]
    if market_regime:
        sections.append(SectionSpec(
            id="market_regime",
            heading="What Market Regime Are We In?",
            goal="Anchor the landscape analysis in the current category regime before comparing vendors",
            key_stats={"market_regime": market_regime},
            data_summary=f"Current market regime: {market_regime}.",
        ))

    # Urgency comparison chart across vendors
    urgency_data = []
    for vs in vendor_signals:
        vendor = vs["vendor"]
        sigs = vs.get("signals", [])
        if sigs:
            avg_urg = sum(s.get("avg_urgency", 0) for s in sigs) / len(sigs) if sigs else 0
            urgency_data.append({"name": vendor[:20], "urgency": round(avg_urg, 1)})
    if urgency_data:
        urgency_chart = ChartSpec(
            chart_id="vendor-urgency",
            chart_type="horizontal_bar",
            title=f"Churn Urgency by Vendor: {category}",
            data=sorted(urgency_data, key=lambda x: x["urgency"], reverse=True),
            config={
                "x_key": "name",
                "bars": [{"dataKey": "urgency", "color": "#f87171"}],
            },
        )
        charts.append(urgency_chart)
        sections.append(SectionSpec(
            id="urgency_ranking",
            heading="Which Vendors Face the Highest Churn Risk?",
            goal="Rank vendors by churn urgency",
            chart_ids=["vendor-urgency"],
            data_summary=f"Urgency scores across {len(urgency_data)} vendors.",
        ))

    if pain_rows:
        pain_chart = ChartSpec(
            chart_id="category-pain-map",
            chart_type="horizontal_bar",
            title=f"Common Pain Patterns Across {category}",
            data=pain_rows,
            config={
                "x_key": "name",
                "bars": [
                    {"dataKey": "vendor_count", "color": "#f87171"},
                    {"dataKey": "avg_urgency", "color": "#fbbf24"},
                ],
            },
        )
        charts.append(pain_chart)
        sections.append(SectionSpec(
            id="category_pain_patterns",
            heading=f"What Keeps Coming Up Across {category} Vendors",
            goal="Show the pain categories that recur across vendors, not just inside one product",
            chart_ids=["category-pain-map"],
            data_summary=(
                f"Recurring pain patterns across {len(pain_rows)} category themes, "
                f"led by {', '.join(row['name'] for row in pain_rows[:3])}."
            ),
        ))

    # Per-vendor breakdowns
    for vp in vendor_profiles[:5]:
        vendor = vp["vendor"]
        profile = vp.get("profile", {})
        strengths = profile.get("strengths", [])
        weaknesses = profile.get("weaknesses", [])
        vendor_signal_rows = signals_by_vendor.get(vendor, [])
        top_pains = [
            str(item.get("pain_category") or "").strip()
            for item in vendor_signal_rows[:3]
            if str(item.get("pain_category") or "").strip()
        ]
        avg_rating = vp.get("avg_rating")
        review_count = vp.get("review_count")
        if strengths or weaknesses:
            sections.append(SectionSpec(
                id=f"vendor-{_slugify(vendor)}",
                heading=f"{vendor}: Strengths & Weaknesses",
                goal=f"Brief profile of {vendor} in the {category} space",
                key_stats={
                    "vendor": vendor,
                    "avg_rating": avg_rating,
                    "review_count": review_count,
                    "top_pains": top_pains,
                    "strengths": [str(s.get("area", s)) if isinstance(s, dict) else str(s) for s in strengths[:3]],
                    "weaknesses": [str(w.get("area", w)) if isinstance(w, dict) else str(w) for w in weaknesses[:3]],
                },
                data_summary=(
                    f"{vendor} has {review_count or 0} reviews in scope"
                    + (f", rating {avg_rating}" if avg_rating is not None else "")
                    + (
                        f", with pain pressure concentrated in {', '.join(top_pains)}."
                        if top_pains
                        else "."
                    )
                ),
            ))

    takeaway_stats: dict[str, Any] = {"category": category, "vendor_count": vendor_count}
    cat_dyn = data.get("pool_category") or {}
    regime = cat_dyn.get("market_regime") or {}
    xv_lookup_ml = data.get("xv_synthesis_lookup") or {}
    council = _resolve_blog_council_summary(
        category, xv_lookup_ml, cat_dyn.get("council_summary") or {},
    )
    if regime.get("regime_type"):
        takeaway_stats["market_regime"] = regime["regime_type"]
    if council.get("conclusion"):
        takeaway_stats["category_conclusion"] = council["conclusion"]
    if council.get("winner"):
        takeaway_stats["category_winner"] = council["winner"]

    # --- Reasoning wedge injection (AEO authority) ---
    wedge = data.get("synthesis_wedge")
    if wedge:
        takeaway_stats["synthesis_wedge"] = wedge
        takeaway_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    causal = (data.get("synthesis_contracts") or {}).get("vendor_core_reasoning") or {}
    cn = causal.get("causal_narrative")
    if isinstance(cn, dict) and cn.get("trigger"):
        takeaway_stats["causal_trigger"] = cn["trigger"]
        takeaway_stats["causal_why_now"] = cn.get("why_now")
    _seg = data.get("pool_segment") or {}
    _budget = _seg.get("budget_pressure") or {}
    if _budget.get("dm_churn_rate") is not None and "dm_churn_rate" not in takeaway_stats:
        takeaway_stats["dm_churn_rate"] = _budget["dm_churn_rate"]

        takeaway_stats["category_winner"] = council["winner"]

    sections.append(SectionSpec(
        id="takeaway",
        heading=f"Choosing the Right {category} Platform",
        goal="Synthesize the landscape and help readers pick the right tool",
        key_stats=takeaway_stats,
        data_summary=f"Decision framework across vendor breadth, churn urgency, and category-level pressure in {category}.",
    ))

    if quote_highlights:
        sections.append(SectionSpec(
            id="reviewer_voice",
            heading=f"What Reviewers Say Across the {category} Market",
            goal="Use direct reviewer snippets to keep the market overview grounded in visible evidence",
            key_stats={"quote_highlights": quote_highlights},
            data_summary=f"{len(quote_highlights)} quote-backed examples spanning the {category} market.",
        ))

    vendor_names = [vp["vendor"] for vp in vendor_profiles[:5]]
    return PostBlueprint(
        topic_type="market_landscape",
        slug=ctx["slug"],
        suggested_title=f"{category} Landscape {date.today().year}: {vendor_count} Vendors Compared by Real User Data",
        tags=[category.lower(), "market-landscape", "comparison", "b2b-intelligence"],
        data_context={**data["data_context"], "category": category},
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_pricing_reality_check(ctx: dict, data: dict) -> PostBlueprint:
    """Honest breakdown of a vendor's pricing -- the good, the bad, and the bait-and-switch."""
    vendor = ctx["vendor"]
    category = ctx.get("category", "software")
    pricing_reviews = data.get("pricing_reviews", [])
    positive_reviews = data.get("positive_reviews", [])
    profile = data.get("profile", {})

    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Lead with the pricing pain -- how many users flagged pricing as a problem with {vendor}",
            key_stats={
                "vendor": vendor,
                "category": category,
                "pricing_complaints": ctx["pricing_complaints"],
                "total_reviews": ctx["total_reviews"],
                "avg_urgency": ctx["avg_urgency"],
            },
            data_summary=(
                f"{ctx['pricing_complaints']} out of {ctx['total_reviews']} {vendor} reviews "
                f"flag pricing as a pain point (avg urgency {ctx['avg_urgency']}/10)."
            ),
        ),
        SectionSpec(
            id="what_users_say",
            heading=f"What {vendor} Users Actually Say About Pricing",
            goal="Present real quotes from users who got hit by price increases, hidden costs, or bait-and-switch tactics",
            key_stats={"pricing_review_count": len(pricing_reviews)},
            data_summary=f"{len(pricing_reviews)} reviews specifically mention pricing frustrations.",
        ),
    ]

    # Pricing complaint urgency distribution
    if pricing_reviews:
        urgency_buckets = {"Critical (8-10)": 0, "High (6-7)": 0, "Moderate (4-5)": 0, "Low (1-3)": 0}
        for pr in pricing_reviews:
            u = pr.get("urgency", 0)
            if u >= 8: urgency_buckets["Critical (8-10)"] += 1
            elif u >= 6: urgency_buckets["High (6-7)"] += 1
            elif u >= 4: urgency_buckets["Moderate (4-5)"] += 1
            else: urgency_buckets["Low (1-3)"] += 1
        urgency_data = [{"name": k, "count": v} for k, v in urgency_buckets.items() if v > 0]
        if urgency_data:
            charts.append(ChartSpec(
                chart_id="pricing-urgency",
                chart_type="bar",
                title=f"Pricing Complaint Severity: {vendor}",
                data=urgency_data,
                config={"x_key": "name", "bars": [{"dataKey": "count", "color": "#f87171"}]},
            ))
            sections.append(SectionSpec(
                id="severity",
                heading="How Bad Is It?",
                goal="Show the severity distribution of pricing complaints",
                chart_ids=["pricing-urgency"],
            ))

    # Credit where it's due
    if positive_reviews:
        sections.append(SectionSpec(
            id="credit_where_due",
            heading=f"Where {vendor} Genuinely Delivers",
            goal="Be fair -- highlight what users love about the product despite pricing concerns",
            key_stats={"positive_count": len(positive_reviews)},
            data_summary=f"{len(positive_reviews)} positive reviews highlight genuine strengths.",
        ))

    bl_stats: dict[str, Any] = {"vendor": vendor, "pricing_complaints": ctx["pricing_complaints"]}
    pool_segment = data.get("pool_segment") or {}
    budget = pool_segment.get("budget_pressure") or {}
    if budget.get("price_increase_rate") is not None:
        bl_stats["price_increase_rate"] = budget["price_increase_rate"]
    if budget.get("dm_churn_rate") is not None:
        bl_stats["dm_churn_rate"] = budget["dm_churn_rate"]
    roles = pool_segment.get("affected_roles") or []

    # --- Reasoning wedge injection (AEO authority) ---
    wedge = data.get("synthesis_wedge")
    if wedge:
        bl_stats["synthesis_wedge"] = wedge
        bl_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    causal = (data.get("synthesis_contracts") or {}).get("vendor_core_reasoning") or {}
    cn = causal.get("causal_narrative")
    if isinstance(cn, dict) and cn.get("trigger"):
        bl_stats["causal_trigger"] = cn["trigger"]
        bl_stats["causal_why_now"] = cn.get("why_now")
    _seg = data.get("pool_segment") or {}
    _budget = _seg.get("budget_pressure") or {}
    if _budget.get("dm_churn_rate") is not None and "dm_churn_rate" not in bl_stats:
        bl_stats["dm_churn_rate"] = _budget["dm_churn_rate"]

    if roles:
        bl_stats["top_churning_role"] = roles[0].get("role_type")

    sections.append(SectionSpec(
        id="bottom_line",
        heading="The Bottom Line: Is It Worth the Price?",
        goal="Honest verdict -- who should pay for it and who should look elsewhere",
        key_stats=bl_stats,
    ))

    # Quotable phrases from pricing reviews
    quotes = [
        {"phrase": r["text"][:200], "vendor": r["vendor"], "urgency": r["urgency"], "role": r.get("role", "")}
        for r in pricing_reviews[:5]
    ]

    return PostBlueprint(
        topic_type="pricing_reality_check",
        slug=ctx["slug"],
        suggested_title=f"The Real Cost of {vendor}: Pricing Complaints in {ctx['pricing_complaints']} Reviews",
        tags=[category, vendor.lower(), "pricing", "honest-review", "cost-analysis"],
        data_context={**data.get("data_context", {}), "vendor": vendor},
        sections=sections,
        charts=charts,
        quotable_phrases=quotes,
    )


def _blueprint_switching_story(ctx: dict, data: dict) -> PostBlueprint:
    """Real stories of teams leaving a vendor -- why they left and where they went."""
    vendor = ctx["from_vendor"]
    category = ctx.get("category", "software")
    switch_reviews = data.get("switch_reviews", [])
    profile = data.get("profile", {})

    compared_to = profile.get("commonly_compared_to", [])
    comp_names = [
        (c.get("vendor", c) if isinstance(c, dict) else str(c))[:25]
        for c in compared_to[:6]
    ]

    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Lead with the switching volume -- real teams actively leaving {vendor}",
            key_stats={
                "vendor": vendor,
                "category": category,
                "switch_mentions": ctx["switch_mentions"],
                "total_reviews": ctx["total_reviews"],
                "avg_urgency": ctx["avg_urgency"],
            },
            data_summary=(
                f"{ctx['switch_mentions']} reviewers mention switching away from {vendor}. "
                f"Avg urgency among all reviews: {ctx['avg_urgency']}/10."
            ),
        ),
        SectionSpec(
            id="breaking_points",
            heading=f"The Breaking Points: Why Teams Leave {vendor}",
            goal="Present the real reasons from actual reviews -- be specific and honest",
            key_stats={"switch_review_count": len(switch_reviews)},
            data_summary=f"{len(switch_reviews)} reviews describe their switching experience.",
        ),
    ]

    if comp_names:
        sections.append(SectionSpec(
            id="where_they_go",
            heading="Where Are They Going?",
            goal="Show the alternatives teams are choosing and why",
            key_stats={"alternatives": comp_names},
            data_summary=f"Commonly compared to: {', '.join(comp_names)}.",
        ))

    # Strengths they're giving up
    strengths = profile.get("strengths", [])
    if strengths:
        sections.append(SectionSpec(
            id="what_youll_miss",
            heading=f"What You'll Miss: {vendor}'s Genuine Strengths",
            goal="Be honest about what the vendor does well -- switching has trade-offs",
            key_stats={
                "strengths": [str(s.get("area", s)) if isinstance(s, dict) else str(s) for s in strengths[:4]],
            },
        ))

    sv_stats: dict[str, Any] = {"vendor": vendor, "avg_urgency": ctx["avg_urgency"]}
    pool_disp = data.get("pool_displacement") or []
    if pool_disp:
        top_edge = max(pool_disp, key=lambda e: (e.get("edge_metrics") or {}).get("mention_count") or 0)
        em = top_edge.get("edge_metrics") or {}
        sv_stats["top_destination"] = top_edge.get("to_vendor")
        sv_stats["displacement_driver"] = em.get("primary_driver")
        switch_reasons = top_edge.get("switch_reasons") or []

    # --- Reasoning wedge injection (AEO authority) ---
    wedge = data.get("synthesis_wedge")
    if wedge:
        sv_stats["synthesis_wedge"] = wedge
        sv_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    causal = (data.get("synthesis_contracts") or {}).get("vendor_core_reasoning") or {}
    cn = causal.get("causal_narrative")
    if isinstance(cn, dict) and cn.get("trigger"):
        sv_stats["causal_trigger"] = cn["trigger"]
        sv_stats["causal_why_now"] = cn.get("why_now")
    _seg = data.get("pool_segment") or {}
    _budget = _seg.get("budget_pressure") or {}
    if _budget.get("dm_churn_rate") is not None and "dm_churn_rate" not in sv_stats:
        sv_stats["dm_churn_rate"] = _budget["dm_churn_rate"]

        if switch_reasons:
            sv_stats["top_switch_reasons"] = [r.get("reason_category") or r.get("reason") for r in switch_reasons[:3]]

    sections.append(SectionSpec(
        id="verdict",
        heading="Should You Stay or Switch?",
        goal="Honest framework for making the decision -- not everyone should switch",
        key_stats=sv_stats,
    ))

    quotes = [
        {"phrase": r["text"][:200], "vendor": r["vendor"], "urgency": r["urgency"], "role": r.get("role", "")}
        for r in switch_reviews[:5]
    ]

    return PostBlueprint(
        topic_type="switching_story",
        slug=ctx["slug"],
        suggested_title=f"Why Teams Are Leaving {vendor}: {ctx['switch_mentions']} Switching Stories Analyzed",
        tags=[category, vendor.lower(), "switching", "migration", "honest-review"],
        data_context={**data.get("data_context", {}), "vendor": vendor},
        sections=sections,
        charts=charts,
        quotable_phrases=quotes,
    )


def _blueprint_pain_point_roundup(ctx: dict, data: dict) -> PostBlueprint:
    """Cross-vendor pain comparison -- the #1 complaint about every vendor in a category."""
    category = ctx["category"]
    vendor_pains = data.get("vendor_pains", [])

    # Chart: top pain per vendor
    pain_chart_data = [
        {"name": vp["vendor"][:20], "reviews": vp["review_count"], "urgency": vp["avg_urgency"]}
        for vp in sorted(vendor_pains, key=lambda x: x["review_count"], reverse=True)[:8]
    ]

    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Frame as a no-BS comparison -- every {category} tool has flaws, here they are",
            key_stats={
                "category": category,
                "vendor_count": ctx["vendor_count"],
                "total_complaints": ctx["total_complaints"],
            },
            data_summary=(
                f"We analyzed {ctx['total_complaints']} reviews across {ctx['vendor_count']} "
                f"{category} vendors. Every single one has a #1 complaint."
            ),
        ),
    ]

    if pain_chart_data:
        charts.append(ChartSpec(
            chart_id="vendor-urgency",
            chart_type="horizontal_bar",
            title=f"Review Volume & Urgency by Vendor: {category}",
            data=pain_chart_data,
            config={
                "x_key": "name",
                "bars": [
                    {"dataKey": "reviews", "color": "#22d3ee"},
                    {"dataKey": "urgency", "color": "#f87171"},
                ],
            },
        ))
        sections.append(SectionSpec(
            id="overview",
            heading="The Landscape at a Glance",
            goal="Show which vendors have the most complaints and highest urgency",
            chart_ids=["vendor-urgency"],
        ))

    # Per-vendor sections
    for vp in vendor_pains[:6]:
        sections.append(SectionSpec(
            id=f"vendor-{_slugify(vp['vendor'])}",
            heading=f"{vp['vendor']}: The #1 Complaint Is {vp['top_pain'].title()}",
            goal=f"Honest breakdown of {vp['vendor']}'s biggest weakness AND what it does well",
            key_stats={
                "vendor": vp["vendor"],
                "top_pain": vp["top_pain"],
                "review_count": vp["review_count"],
                "avg_urgency": vp["avg_urgency"],
            },
        ))

    pp_stats: dict[str, Any] = {"category": category, "vendor_count": ctx["vendor_count"]}
    cat_dyn = data.get("pool_category") or {}
    regime = cat_dyn.get("market_regime") or {}
    if regime.get("regime_type"):
        pp_stats["market_regime"] = regime["regime_type"]
    if regime.get("outlier_vendors"):
        pp_stats["outlier_vendors"] = regime["outlier_vendors"][:3]


    # --- Reasoning wedge injection (AEO authority) ---
    wedge = data.get("synthesis_wedge")
    if wedge:
        pp_stats["synthesis_wedge"] = wedge
        pp_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    causal = (data.get("synthesis_contracts") or {}).get("vendor_core_reasoning") or {}
    cn = causal.get("causal_narrative")
    if isinstance(cn, dict) and cn.get("trigger"):
        pp_stats["causal_trigger"] = cn["trigger"]
        pp_stats["causal_why_now"] = cn.get("why_now")
    _seg = data.get("pool_segment") or {}
    _budget = _seg.get("budget_pressure") or {}
    if _budget.get("dm_churn_rate") is not None and "dm_churn_rate" not in pp_stats:
        pp_stats["dm_churn_rate"] = _budget["dm_churn_rate"]

    sections.append(SectionSpec(
        id="takeaway",
        heading="Every Tool Has a Flaw -- Pick the One You Can Live With",
        goal="Honest summary -- there's no perfect tool, help readers pick the right trade-off",
        key_stats=pp_stats,
    ))

    return PostBlueprint(
        topic_type="pain_point_roundup",
        slug=ctx["slug"],
        suggested_title=f"The #1 Complaint About Every Major {category} Tool in {date.today().year}",
        tags=[category.lower(), "complaints", "comparison", "honest-review", "b2b-intelligence"],
        data_context={**data.get("data_context", {}), "category": category},
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_best_fit_guide(ctx: dict, data: dict) -> PostBlueprint:
    """Recommend the right tool based on team size, needs, and budget -- not commissions."""
    category = ctx["category"]
    vendor_profiles = data.get("vendor_profiles", [])
    vendor_signals = data.get("vendor_signals", [])
    pain_rows = _aggregate_category_pain_rows(vendor_signals, limit=6)
    quote_highlights = _blog_quote_highlights(data.get("quotes", []), limit=4)
    signals_by_vendor = {
        str(item.get("vendor") or "").strip(): item.get("signals") or []
        for item in vendor_signals
        if isinstance(item, dict)
    }

    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Position as an honest buyer's guide for {category} -- based on real user data, not marketing",
            key_stats={
                "category": category,
                "vendor_count": ctx["vendor_count"],
                "total_reviews": ctx["total_reviews"],
            },
            data_summary=(
                f"We analyzed {ctx['total_reviews']} real user reviews across "
                f"{ctx['vendor_count']} {category} tools to find who's actually best for what."
            ),
        ),
    ]

    # Rating comparison chart
    rated_vendors = [vp for vp in vendor_profiles if vp.get("avg_rating") is not None]
    if rated_vendors:
        rating_data = sorted(
            [{"name": vp["vendor"][:20], "rating": vp["avg_rating"], "reviews": vp["review_count"]}
             for vp in rated_vendors],
            key=lambda x: x["rating"], reverse=True,
        )
        charts.append(ChartSpec(
            chart_id="ratings",
            chart_type="horizontal_bar",
            title=f"Average Rating by Vendor: {category}",
            data=rating_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "rating", "color": "#34d399"}],
            },
        ))
        sections.append(SectionSpec(
            id="ratings_overview",
            heading="Ratings at a Glance (But Don't Stop Here)",
            goal="Show ratings but warn that averages hide important nuances",
            chart_ids=["ratings"],
            data_summary=f"Ratings and review volume across {len(rating_data)} vendors in the {category} set.",
        ))

    if pain_rows:
        charts.append(ChartSpec(
            chart_id="category-pain-fit-map",
            chart_type="horizontal_bar",
            title=f"Common Pain Patterns in {category}",
            data=pain_rows,
            config={
                "x_key": "name",
                "bars": [
                    {"dataKey": "vendor_count", "color": "#f87171"},
                    {"dataKey": "avg_urgency", "color": "#fbbf24"},
                ],
            },
        ))
        sections.append(SectionSpec(
            id="category_tradeoffs",
            heading=f"What Teams Consistently Struggle With in {category}",
            goal="Explain the tradeoffs that keep showing up across vendors before recommending specific fits",
            chart_ids=["category-pain-fit-map"],
            data_summary=(
                f"Category-wide pain patterns led by {', '.join(row['name'] for row in pain_rows[:3])}."
            ),
        ))

    # Per-vendor recommendation sections
    for vp in vendor_profiles[:6]:
        profile = vp.get("profile", {})
        company_size = profile.get("typical_company_size", {})
        size_str = ", ".join(f"{k}" for k, v in sorted(company_size.items(), key=lambda x: x[1], reverse=True)[:2]) if isinstance(company_size, dict) and company_size else "all sizes"
        strengths = profile.get("strengths", [])
        weaknesses = profile.get("weaknesses", [])
        vendor_signal_rows = signals_by_vendor.get(vp["vendor"], [])
        top_pains = [
            str(item.get("pain_category") or "").strip()
            for item in vendor_signal_rows[:3]
            if str(item.get("pain_category") or "").strip()
        ]
        sections.append(SectionSpec(
            id=f"vendor-{_slugify(vp['vendor'])}",
            heading=f"{vp['vendor']}: Best For {size_str} Teams",
            goal=f"Honest recommendation -- who should use {vp['vendor']} and who shouldn't",
            key_stats={
                "vendor": vp["vendor"],
                "company_size": size_str,
                "avg_rating": vp.get("avg_rating"),
                "review_count": vp.get("review_count"),
                "top_pains": top_pains,
                "strengths": [str(s.get("area", s)) if isinstance(s, dict) else str(s) for s in strengths[:3]],
                "weaknesses": [str(w.get("area", w)) if isinstance(w, dict) else str(w) for w in weaknesses[:3]],
            },
            data_summary=(
                f"{vp['vendor']} is strongest for {size_str} teams"
                + (f", with rating {vp.get('avg_rating')}" if vp.get("avg_rating") is not None else "")
                + (
                    f", but recurring pain shows up around {', '.join(top_pains)}."
                    if top_pains
                    else "."
                )
            ),
        ))

    bf_stats: dict[str, Any] = {"category": category, "vendor_count": ctx["vendor_count"]}
    cat_dyn = data.get("pool_category") or {}
    xv_lookup_bf = data.get("xv_synthesis_lookup") or {}
    council = _resolve_blog_council_summary(
        category, xv_lookup_bf, cat_dyn.get("council_summary") or {},
    )
    if council.get("winner"):
        bf_stats["category_winner"] = council["winner"]
    if council.get("conclusion"):
        bf_stats["category_conclusion"] = council["conclusion"]


    # --- Reasoning wedge injection (AEO authority) ---
    wedge = data.get("synthesis_wedge")
    if wedge:
        bf_stats["synthesis_wedge"] = wedge
        bf_stats["synthesis_wedge_label"] = data.get("synthesis_wedge_label") or ""
    causal = (data.get("synthesis_contracts") or {}).get("vendor_core_reasoning") or {}
    cn = causal.get("causal_narrative")
    if isinstance(cn, dict) and cn.get("trigger"):
        bf_stats["causal_trigger"] = cn["trigger"]
        bf_stats["causal_why_now"] = cn.get("why_now")
    _seg = data.get("pool_segment") or {}
    _budget = _seg.get("budget_pressure") or {}
    if _budget.get("dm_churn_rate") is not None and "dm_churn_rate" not in bf_stats:
        bf_stats["dm_churn_rate"] = _budget["dm_churn_rate"]

    sections.append(SectionSpec(
        id="decision_framework",
        heading="How to Actually Choose",
        goal="Give a clear decision framework based on budget, team size, and must-have features",
        key_stats=bf_stats,
        data_summary=f"Decision framework for narrowing {category} options by fit, tradeoffs, and current market pressure.",
    ))

    if quote_highlights:
        sections.append(SectionSpec(
            id="reviewer_voice",
            heading=f"What Reviewers Actually Say in {category}",
            goal="Ground the buyer's guide in direct reviewer evidence, not just rating averages",
            key_stats={"quote_highlights": quote_highlights},
            data_summary=f"{len(quote_highlights)} quote-backed snippets from the {category} review set.",
        ))

    company_size = ctx.get("company_size") or ctx.get("dominant_size") or ""
    size_label = company_size.replace("_", " ").replace("-", " ").strip() if company_size else ""
    size_suffix = f" for {size_label} Teams" if size_label and size_label != "unknown" else ""
    return PostBlueprint(
        topic_type="best_fit_guide",
        slug=ctx["slug"],
        suggested_title=f"Best {category} Tools{size_suffix}: {ctx['vendor_count']} Vendors Compared Across {ctx['total_reviews']} Reviews",
        tags=[category.lower(), "buyers-guide", "comparison", "honest-review", *(["team-size"] if size_label else [])],
        data_context={**data.get("data_context", {}), "category": category, "company_size": company_size},
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


# -- Stage 4: Content Generation ----------------------------------

def _blog_generation_trace_metadata(
    blueprint: PostBlueprint,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "workflow": "b2b_blog_post_generation",
        "report_type": blueprint.topic_type,
        "topic_type": blueprint.topic_type,
        "slug": blueprint.slug,
        "source_name": "b2b_blog_post_generation",
        "event_type": "llm_overlay",
        "entity_type": "blog_post",
        "entity_id": blueprint.slug,
        "suggested_title": blueprint.suggested_title[:200],
        "tags": blueprint.tags[:10],
    }
    if run_id:
        metadata["run_id"] = run_id
    data_context = blueprint.data_context if isinstance(blueprint.data_context, dict) else {}
    if data_context.get("vendor"):
        metadata["vendor_name"] = str(data_context["vendor"])[:200]
    if data_context.get("vendor_a"):
        metadata["vendor_a"] = str(data_context["vendor_a"])[:200]
    if data_context.get("vendor_b"):
        metadata["vendor_b"] = str(data_context["vendor_b"])[:200]
    if data_context.get("category"):
        metadata["category"] = str(data_context["category"])[:200]
    return metadata


def _build_blog_generation_payload(
    blueprint: PostBlueprint,
    *,
    related_posts: list[dict[str, str]] | None = None,
    quality_feedback: list[str] | None = None,
) -> dict[str, Any]:
    _attach_blog_blueprint_runtime_metadata(blueprint)
    length_policy = _blog_length_policy(blueprint.topic_type)
    section_word_budget = _blog_section_word_budget(blueprint)
    payload: dict[str, Any] = {
        "topic_type": blueprint.topic_type,
        "suggested_title": blueprint.suggested_title,
        "data_context": blueprint.data_context,
        "length_policy": length_policy,
        "section_word_budget": section_word_budget,
        "sections": [
            {
                "id": section.id,
                "heading": section.heading,
                "goal": section.goal,
                "key_stats": section.key_stats,
                "chart_ids": section.chart_ids,
                "data_summary": section.data_summary,
            }
            for section in blueprint.sections
        ],
        "available_charts": [
            {
                "chart_id": chart.chart_id,
                "chart_type": chart.chart_type,
                "title": chart.title,
                "data_labels": [
                    str(row.get("name") or row.get("label") or row.get("category") or "")
                    for row in chart.data
                    if isinstance(row, dict)
                ][:10],
            }
            for chart in blueprint.charts
        ],
        "quotable_phrases": blueprint.quotable_phrases[:5],
    }
    data_context = blueprint.data_context if isinstance(blueprint.data_context, dict) else {}
    for src_key, dest_key in (
        ("reasoning_anchor_examples", "anchor_examples"),
        ("reasoning_witness_highlights", "witness_highlights"),
        ("reasoning_reference_ids", "reference_ids"),
        ("blog_claim_plan", "claim_plan"),
        ("reasoning_anchor_examples_b", "anchor_examples_b"),
        ("reasoning_witness_highlights_b", "witness_highlights_b"),
        ("reasoning_reference_ids_b", "reference_ids_b"),
        ("blog_claim_plan_b", "claim_plan_b"),
    ):
        value = data_context.get(src_key)
        if value not in (None, "", [], {}):
            payload[dest_key] = value
    if blueprint.cta:
        payload["cta_context"] = {
            "button_text": blueprint.cta["button_text"],
            "report_type": blueprint.cta["report_type"],
            "vendor": blueprint.cta.get("vendor_filter"),
        }
    if related_posts:
        payload["related_posts"] = related_posts
    if quality_feedback:
        payload["quality_feedback"] = quality_feedback[:10]
    return payload


def _prepare_blog_generation_request(
    llm,
    blueprint: PostBlueprint,
    max_tokens: int,
    *,
    related_posts: list[dict[str, str]] | None = None,
    quality_feedback: list[str] | None = None,
) -> dict[str, Any] | None:
    from ...services.b2b.cache_runner import prepare_b2b_exact_stage_request
    from ...services.protocols import Message
    from ...skills.registry import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_blog_post_generation")
    if skill is None:
        logger.error("Skill digest/b2b_blog_post_generation not found")
        return None

    payload = _build_blog_generation_payload(
        blueprint,
        related_posts=related_posts,
        quality_feedback=quality_feedback,
    )
    messages = [
        Message(role="system", content=skill.content),
        Message(
            role="user",
            content=json.dumps(payload, separators=(",", ":"), default=str),
        ),
    ]
    request = prepare_b2b_exact_stage_request(
        _BLOG_GENERATION_CACHE_STAGE,
        llm=llm,
        messages=messages,
        max_tokens=max_tokens,
        temperature=float(settings.b2b_churn.blog_post_temperature),
    )
    return {
        "payload": payload,
        "messages": messages,
        "request": request,
        "provider_name": request.provider,
        "model_name": request.model,
        "request_envelope": request.request_envelope,
    }


def _trace_blog_generation_call(
    llm,
    blueprint: PostBlueprint,
    *,
    messages: list[Any],
    result: Any,
    run_id: str | None = None,
) -> None:
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    if not usage.get("input_tokens"):
        return

    logger.info(
        "b2b_blog_post_generation LLM tokens: in=%d out=%d",
        usage["input_tokens"],
        usage.get("output_tokens", 0),
    )
    from ...pipelines.llm import trace_llm_call

    trace_meta = result.get("_trace_meta", {}) if isinstance(result, dict) else {}
    response_text = (result.get("response", "") if isinstance(result, dict) else str(result)) or ""
    trace_llm_call(
        _BLOG_GENERATION_TRACE_SPAN,
        input_tokens=usage["input_tokens"],
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=trace_meta.get("cached_tokens") or trace_meta.get("cache_read_tokens"),
        cache_write_tokens=trace_meta.get("cache_write_tokens") or trace_meta.get("cache_creation_tokens"),
        billable_input_tokens=trace_meta.get("billable_input_tokens"),
        model=getattr(llm, "model", ""),
        provider=getattr(llm, "name", ""),
        input_data={
            "messages": [
                {
                    "role": getattr(message, "role", ""),
                    "content": str(getattr(message, "content", "") or "")[:500],
                }
                for message in messages
            ]
        },
        output_data={"response": response_text[:2000]},
        api_endpoint=trace_meta.get("api_endpoint"),
        provider_request_id=trace_meta.get("provider_request_id"),
        ttft_ms=trace_meta.get("ttft_ms"),
        inference_time_ms=trace_meta.get("inference_time_ms"),
        queue_time_ms=trace_meta.get("queue_time_ms"),
        metadata=_blog_generation_trace_metadata(
            blueprint,
            run_id=run_id,
        ),
    )


def _parse_blog_generation_response_text(
    blueprint: PostBlueprint,
    text: str | None,
    *,
    request_envelope: dict[str, Any],
    provider_name: str,
    model_name: str,
    debug_result: Any | None = None,
) -> dict[str, Any] | None:
    from ...pipelines.llm import clean_llm_output, parse_json_response

    logger.info("Blog LLM raw response length: %d chars", len(text or ""))
    if not text:
        logger.error("Blog LLM returned empty response for %s", blueprint.slug)
        if debug_result is not None:
            try:
                with open("/tmp/blog_empty_response.txt", "w") as handle:
                    handle.write(json.dumps(debug_result, indent=2, default=str)[:5000])
            except Exception:
                pass
        return None

    cleaned_text = clean_llm_output(text)
    parsed = parse_json_response(cleaned_text, recover_truncated=True)
    if parsed.get("_parse_fallback"):
        logger.error(
            "Failed to parse LLM response as JSON (text[:500]=%s)",
            cleaned_text[:500],
        )
        try:
            with open("/tmp/blog_llm_fail.txt", "w") as handle:
                handle.write(
                    f"PARSE FALLBACK\ntext_len={len(cleaned_text)}\ntext[:2000]={cleaned_text[:2000]}\n"
                )
        except Exception:
            pass
        return None

    if not all(key in parsed for key in ("title", "description", "content")):
        logger.error(
            "LLM response missing required keys: %s (text[:300]=%s)",
            list(parsed.keys()),
            cleaned_text[:300],
        )
        return None

    if "seo_title" not in parsed or not parsed["seo_title"]:
        parsed["seo_title"] = parsed["title"][:60]
    if "seo_description" not in parsed or not parsed["seo_description"]:
        parsed["seo_description"] = parsed["description"][:155]
    if "target_keyword" not in parsed:
        parsed["target_keyword"] = ""
    if "secondary_keywords" not in parsed:
        parsed["secondary_keywords"] = []
    if "faq" not in parsed or not isinstance(parsed["faq"], list):
        parsed["faq"] = []

    if blueprint.cta and parsed.get("cta_body"):
        blueprint.cta["body"] = str(parsed["cta_body"])[:200]

    parsed["_cache_request_envelope"] = request_envelope
    parsed["_cache_provider"] = provider_name
    parsed["_cache_model"] = model_name
    return parsed


async def _generate_content_async(
    llm, blueprint: PostBlueprint, max_tokens: int,
    related_posts: list[dict[str, str]] | None = None,
    quality_feedback: list[str] | None = None,
    run_id: str | None = None,
    pool: Any | None = None,
    usage_out: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Single LLM call: blueprint in, {title, description, content} out."""
    from ...services.b2b.cache_runner import lookup_b2b_exact_stage_text

    prepared = _prepare_blog_generation_request(
        llm,
        blueprint,
        max_tokens,
        related_posts=related_posts,
        quality_feedback=quality_feedback,
    )
    if prepared is None:
        return None

    request = prepared["request"]
    messages = prepared["messages"]

    try:
        cached = await lookup_b2b_exact_stage_text(request, pool=pool)
        if cached is not None:
            if usage_out is not None:
                usage_out.clear()
                usage_out.update(
                    {
                        "input_tokens": 0,
                        "billable_input_tokens": 0,
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                        "output_tokens": 0,
                        "provider": str(cached.get("provider") or prepared["provider_name"]),
                        "model": str(cached.get("model") or prepared["model_name"]),
                    }
                )
            return _parse_blog_generation_response_text(
                blueprint,
                str(cached["response_text"] or ""),
                request_envelope=prepared["request_envelope"],
                provider_name=str(prepared["provider_name"]),
                model_name=str(prepared["model_name"]),
            )

        result = await asyncio.to_thread(
            llm.chat,
            messages=messages,
            max_tokens=max_tokens,
            temperature=float(settings.b2b_churn.blog_post_temperature),
        )
        if usage_out is not None:
            usage = result.get("usage", {}) if isinstance(result, dict) else {}
            trace_meta = result.get("_trace_meta", {}) if isinstance(result, dict) else {}
            usage_out.clear()
            usage_out.update(
                {
                    "input_tokens": int((usage or {}).get("input_tokens") or 0),
                    "billable_input_tokens": int(
                        (trace_meta or {}).get("billable_input_tokens")
                        or (usage or {}).get("input_tokens")
                        or 0
                    ),
                    "cached_tokens": int(
                        (trace_meta or {}).get("cached_tokens")
                        or (trace_meta or {}).get("cache_read_tokens")
                        or 0
                    ),
                    "cache_write_tokens": int(
                        (trace_meta or {}).get("cache_write_tokens")
                        or (trace_meta or {}).get("cache_creation_tokens")
                        or 0
                    ),
                    "output_tokens": int((usage or {}).get("output_tokens") or 0),
                    "provider": str(getattr(llm, "name", "") or ""),
                    "model": str(getattr(llm, "model", "") or ""),
                    "provider_request_id": (
                        str((trace_meta or {}).get("provider_request_id") or "") or None
                    ),
                }
            )
        _trace_blog_generation_call(
            llm,
            blueprint,
            messages=messages,
            result=result,
            run_id=run_id,
        )
        text = result.get("response", "") if isinstance(result, dict) else str(result)
        return _parse_blog_generation_response_text(
            blueprint,
            text,
            request_envelope=prepared["request_envelope"],
            provider_name=str(prepared["provider_name"]),
            model_name=str(prepared["model_name"]),
            debug_result=result,
        )
    except Exception:
        logger.exception("LLM content generation failed")
        return None


def _generate_content(
    llm, blueprint: PostBlueprint, max_tokens: int,
    related_posts: list[dict[str, str]] | None = None,
    quality_feedback: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Sync wrapper retained for local/unit-test callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _generate_content_async(
                llm,
                blueprint,
                max_tokens,
                related_posts=related_posts,
                quality_feedback=quality_feedback,
                run_id=run_id,
            )
        )
    raise RuntimeError("Use _generate_content_async() from async contexts")


# -- Stage 5: Assembly & Storage ----------------------------------


async def _compute_related_slugs(
    pool, current_slug: str, tags: list[str], limit: int = 4
) -> list[str]:
    """Find related blog posts by overlapping tags/category."""
    if not tags:
        return []
    rows = await pool.fetch(
        """
        SELECT slug FROM blog_posts
        WHERE slug != $1
          AND status IN ('draft', 'published')
          AND tags::jsonb ?| $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        current_slug, tags[:3], limit,
    )
    return [r["slug"] for r in rows]


async def _fetch_related_for_linking(
    pool, tags: list[str], current_slug: str = "", limit: int = 6
) -> list[dict[str, str]]:
    """Fetch published/draft posts with overlapping tags for internal linking."""
    if not tags:
        return []
    try:
        rows = await pool.fetch(
            """
            SELECT slug, title FROM blog_posts
            WHERE slug != $1
              AND status IN ('draft', 'published')
              AND tags::jsonb ?| $2
            ORDER BY created_at DESC
            LIMIT $3
            """,
            current_slug, tags[:3], limit,
        )
        return [{"slug": r["slug"], "title": r["title"]} for r in rows]
    except Exception:
        logger.debug("Failed to fetch related posts for linking", exc_info=True)
        return []


def _fallback_target_keyword(blueprint: PostBlueprint) -> str:
    """Generate a deterministic target keyword when the LLM omits it."""
    ctx = blueprint.data_context.get("topic_ctx") or blueprint.data_context
    tt = blueprint.topic_type
    vendor = str(ctx.get("vendor") or ctx.get("vendor_a") or ctx.get("from_vendor") or "").strip()
    vendor_b = str(ctx.get("vendor_b") or "").strip()
    category = str(ctx.get("category") or "").strip()
    kw_map = {
        "vendor_showdown": f"{vendor} vs {vendor_b}".strip(),
        "vendor_alternative": f"{vendor} alternatives".strip(),
        "churn_report": f"{vendor} churn rate".strip(),
        "pricing_reality_check": f"{vendor} pricing".strip(),
        "migration_guide": f"switch to {vendor}".strip(),
        "switching_story": f"why teams leave {vendor}".strip(),
        "vendor_deep_dive": f"{vendor} reviews".strip(),
        "market_landscape": f"{category} software comparison".strip(),
        "pain_point_roundup": f"{category} software complaints".strip(),
        "best_fit_guide": (
            f"best {category} tools for {ctx.get('company_size', '').replace('_', ' ')}".strip()
            if ctx.get("company_size") and ctx["company_size"] != "unknown"
            else f"best {category} tools".strip()
        ),
    }
    return kw_map.get(tt, vendor).lower() or blueprint.slug.replace("-", " ")[:50]


def _fallback_seo_title(display_title: str, blueprint: PostBlueprint) -> str:
    """Generate a deterministic SEO title when the LLM omits it."""
    kw = _fallback_target_keyword(blueprint)
    year = date.today().year
    candidate = f"{kw.title()} {year}"
    if len(candidate) <= 60:
        return candidate
    return display_title[:60]


async def _upsert_blog_post_state(
    pool,
    blueprint: PostBlueprint,
    llm,
    *,
    status: str,
    run_id: str | None = None,
    attempt_no: int | None = None,
    score: int | None = None,
    threshold: int | None = None,
    blocker_count: int = 0,
    warning_count: int = 0,
    failure_step: str | None = None,
    error_code: str | None = None,
    error_summary: str | None = None,
    rejection_reason: str | None = None,
    content: dict[str, Any] | None = None,
) -> str:
    """Upsert the canonical blog row so failures/rejections do not vanish."""
    _attach_blog_blueprint_runtime_metadata(blueprint)
    charts_json = [asdict(c) for c in blueprint.charts]
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    title = (content or {}).get("title") or blueprint.suggested_title
    description = (content or {}).get("description", "")
    body = (content or {}).get("content", "")
    source_report_date = blueprint.data_context.get("source_report_date") or date.today()
    row = await pool.fetchrow(
        """
        INSERT INTO blog_posts (
            slug, title, description, topic_type, tags,
            content, charts, data_context, status, llm_model, source_report_date,
            seo_title, seo_description, target_keyword,
            secondary_keywords, faq, cta,
            latest_run_id, latest_attempt_no, latest_failure_step,
            latest_error_code, latest_error_summary,
            quality_score, quality_threshold, blocker_count, warning_count,
            rejected_at, rejection_reason
        ) VALUES (
            $1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8::jsonb, $9, $10, $11,
            $12, $13, $14, $15::jsonb, $16::jsonb, $17::jsonb,
            $18, $19, $20, $21, $22, $23, $24, $25, $26,
            CASE WHEN $9 = 'rejected' THEN NOW() ELSE NULL END,
            CASE WHEN $9 = 'rejected' THEN $27 ELSE NULL END
        )
        ON CONFLICT (slug) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            topic_type = EXCLUDED.topic_type,
            tags = EXCLUDED.tags,
            content = EXCLUDED.content,
            charts = EXCLUDED.charts,
            data_context = EXCLUDED.data_context,
            status = EXCLUDED.status,
            llm_model = EXCLUDED.llm_model,
            source_report_date = EXCLUDED.source_report_date,
            seo_title = EXCLUDED.seo_title,
            seo_description = EXCLUDED.seo_description,
            target_keyword = EXCLUDED.target_keyword,
            secondary_keywords = EXCLUDED.secondary_keywords,
            faq = EXCLUDED.faq,
            cta = EXCLUDED.cta,
            latest_run_id = EXCLUDED.latest_run_id,
            latest_attempt_no = EXCLUDED.latest_attempt_no,
            latest_failure_step = EXCLUDED.latest_failure_step,
            latest_error_code = EXCLUDED.latest_error_code,
            latest_error_summary = EXCLUDED.latest_error_summary,
            quality_score = EXCLUDED.quality_score,
            quality_threshold = EXCLUDED.quality_threshold,
            blocker_count = EXCLUDED.blocker_count,
            warning_count = EXCLUDED.warning_count,
            rejected_at = EXCLUDED.rejected_at,
            rejection_reason = EXCLUDED.rejection_reason,
            rejection_count = CASE
                WHEN EXCLUDED.status = 'rejected'
                THEN COALESCE(blog_posts.rejection_count, 0) + 1
                ELSE 0
            END,
            published_at = CASE
                WHEN EXCLUDED.status = 'published' THEN COALESCE(blog_posts.published_at, NOW())
                ELSE blog_posts.published_at
            END
        WHERE blog_posts.status != 'published'
        RETURNING id
        """,
        blueprint.slug,
        title,
        description,
        blueprint.topic_type,
        json.dumps(blueprint.tags),
        body,
        json.dumps(charts_json, default=str),
        json.dumps(blueprint.data_context, default=str),
        status,
        str(model_name),
        source_report_date,
        (content or {}).get("seo_title") or _fallback_seo_title(title, blueprint),
        (content or {}).get("seo_description") or description[:155],
        (content or {}).get("target_keyword") or _fallback_target_keyword(blueprint),
        json.dumps((content or {}).get("secondary_keywords", []), default=str),
        json.dumps((content or {}).get("faq", []), default=str),
        json.dumps(blueprint.cta, default=str) if blueprint.cta else None,
        run_id,
        attempt_no,
        failure_step,
        error_code,
        error_summary,
        score,
        threshold,
        blocker_count,
        warning_count,
        rejection_reason,
    )
    return str(row["id"]) if row else ""


async def _assemble_and_store(
    pool,
    blueprint: PostBlueprint,
    content: dict[str, Any],
    llm,
    *,
    run_id: str | None = None,
    attempt_no: int | None = None,
) -> str:
    """Store the assembled post as a draft in blog_posts."""
    from ...services.blog_quality import blog_quality_summary

    charts_json = [asdict(c) for c in blueprint.charts]
    quality = blog_quality_summary(blueprint.data_context)
    post_id = await _upsert_blog_post_state(
        pool,
        blueprint,
        llm,
        status="draft",
        run_id=run_id,
        attempt_no=attempt_no,
        score=quality.get("score"),
        threshold=quality.get("threshold"),
        blocker_count=len(quality.get("blocking_issues", []) or []),
        warning_count=len(quality.get("warnings", []) or []),
        content=content,
    )
    if not post_id:
        logger.warning(
            "Skipped overwrite of published post: slug=%s", blueprint.slug
        )
        return ""
    logger.info("Stored B2B blog draft: slug=%s, id=%s", blueprint.slug, post_id)

    # Compute related posts (same category/vendor overlap)
    related: list[str] = []
    try:
        related = await _compute_related_slugs(pool, blueprint.slug, blueprint.tags)
        if related:
            await pool.execute(
                "UPDATE blog_posts SET related_slugs = $1 WHERE id = $2",
                json.dumps(related), post_id,
            )
    except Exception:
        logger.debug("Related slug computation skipped", exc_info=True)

    # Write .ts file for the frontend if ui_path is configured
    cfg = settings.b2b_churn
    if cfg.blog_post_ui_path:
        try:
            _write_ui_post(
                cfg.blog_post_ui_path,
                blueprint,
                content,
                charts_json,
                related_slugs=related,
            )
        except Exception:
            logger.warning("Failed to write B2B blog UI file", exc_info=True)
        else:
            try:
                from ._blog_deploy import auto_deploy_blog
                await auto_deploy_blog(
                    cfg.blog_post_ui_path,
                    blueprint.slug,
                    enabled=cfg.blog_auto_deploy_enabled,
                    branch=cfg.blog_auto_deploy_branch,
                    hook_url=cfg.blog_auto_deploy_hook_url,
                )
            except Exception:
                logger.warning("B2B blog auto-deploy failed", exc_info=True)

    return post_id


def _write_ui_post(
    ui_path: str,
    blueprint: PostBlueprint,
    content: dict[str, Any],
    charts_json: list[dict[str, Any]],
    related_slugs: list[str] | None = None,
) -> None:
    """Write a .ts post file and register it in index.ts."""
    from pathlib import Path
    from ._blog_ts import build_post_ts, update_blog_index

    blog_dir = Path(ui_path)
    if not blog_dir.is_dir():
        logger.warning("blog_post_ui_path does not exist: %s", ui_path)
        return

    slug = blueprint.slug
    var_name, ts_content = build_post_ts(
        slug=slug,
        title=content["title"],
        description=content.get("description", ""),
        date_str=date.today().isoformat(),
        author="Churn Signals Team",
        tags=blueprint.tags,
        topic_type=blueprint.topic_type,
        charts_json=charts_json,
        content=content["content"],
        data_context=blueprint.data_context,
        seo_title=content.get("seo_title", ""),
        seo_description=content.get("seo_description", ""),
        target_keyword=content.get("target_keyword", ""),
        secondary_keywords=content.get("secondary_keywords"),
        faq=content.get("faq"),
        related_slugs=related_slugs,
        cta=blueprint.cta,
    )

    post_path = blog_dir / (slug + ".ts")
    post_path.write_text(ts_content, encoding="utf-8")
    logger.info("Wrote B2B blog UI file: %s", post_path)

    update_blog_index(blog_dir / "index.ts", slug, var_name)


# -- Manual generation helpers ------------------------------------

_KNOWN_TOPIC_TYPES = {
    "vendor_alternative", "vendor_showdown", "churn_report",
    "migration_guide", "vendor_deep_dive", "market_landscape",
    "pricing_reality_check", "switching_story", "pain_point_roundup",
    "best_fit_guide",
}


async def _fetch_vendor_stats(pool, vendor_name: str) -> dict[str, Any]:
    """Return review counts and urgency for a single vendor."""
    # APPROVED-ENRICHMENT-READ: urgency_score
    # Reason: inline aggregate query, structurally coupled to product output
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE r.enrichment_status = 'enriched') AS enriched,
            COUNT(*) FILTER (WHERE r.rating IS NOT NULL AND r.rating < 3) AS negative,
            ROUND(AVG(
                CASE WHEN r.enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (r.enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency,
            MODE() WITHIN GROUP (ORDER BY r.product_category) AS category
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE LOWER(vm.vendor_name) = LOWER($1)
          AND r.duplicate_of_review_id IS NULL
        """,
        vendor_name,
    )
    if not row or row["total"] == 0:
        return {}
    return {
        "total": row["total"],
        "enriched": row["enriched"],
        "negative": row["negative"],
        "avg_urgency": float(row["avg_urgency"]) if row["avg_urgency"] else 0,
        "category": row["category"] or "",
    }


async def _fetch_category_topic_stats(pool, category: str) -> dict[str, Any]:
    sources = _blog_source_allowlist()
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(DISTINCT vm.vendor_name) AS vendor_count,
            COUNT(DISTINCT r.id) FILTER (WHERE r.enrichment_status = 'enriched') AS enriched_reviews,
            ROUND(AVG(
                CASE WHEN r.enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (r.enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.product_category = $1
          AND r.duplicate_of_review_id IS NULL
          AND r.source = ANY($2)
        """,
        category,
        sources,
    )
    dominant_size = await pool.fetchval(
        """
        SELECT MODE() WITHIN GROUP (
            ORDER BY COALESCE(
                (
                    SELECT key
                    FROM jsonb_each_text(COALESCE(pp.typical_company_size, '{}'::jsonb))
                    ORDER BY value::numeric DESC
                    LIMIT 1
                ),
                'unknown'
            )
        ) AS dominant_size
        FROM b2b_product_profiles pp
        WHERE pp.product_category = $1
        """,
        category,
    )
    return {
        "vendor_count": int(row["vendor_count"] or 0) if row else 0,
        "enriched_reviews": int(row["enriched_reviews"] or 0) if row else 0,
        "avg_urgency": float(row["avg_urgency"]) if row and row["avg_urgency"] else 0.0,
        "dominant_size": str(dominant_size or "unknown"),
    }


async def build_manual_topic_ctx(
    pool,
    vendor_name: str,
    topic_type: str,
    vendor_b: str | None = None,
    category: str | None = None,
    company_size: str | None = None,
) -> dict[str, Any]:
    """Construct topic_ctx for a manually requested blog post.

    Bypasses _select_topic() dedup -- always builds context even if a post
    for this vendor+month already exists.
    """
    month_suffix = date.today().strftime("%Y-%m")
    stats = await _fetch_vendor_stats(pool, vendor_name)
    if not category:
        category = stats.get("category", "software") or "software"
    category_stats = await _fetch_category_topic_stats(pool, category)

    ctx: dict[str, Any] = {
        "category": category,
    }

    if topic_type == "vendor_showdown":
        if not vendor_b:
            raise ValueError("vendor_showdown requires vendor_b")
        stats_b = await _fetch_vendor_stats(pool, vendor_b)
        slug = f"{_slugify(vendor_name)}-vs-{_slugify(vendor_b)}-{month_suffix}"
        ctx.update({
            "vendor_a": vendor_name,
            "vendor_b": vendor_b,
            "reviews_a": stats.get("total", 0),
            "reviews_b": stats_b.get("total", 0),
            "total_reviews": stats.get("total", 0) + stats_b.get("total", 0),
            "urgency_a": stats.get("avg_urgency", 0),
            "urgency_b": stats_b.get("avg_urgency", 0),
            "pain_diff": abs(stats.get("avg_urgency", 0) - stats_b.get("avg_urgency", 0)),
            "slug": slug,
        })
    elif topic_type == "switching_story":
        slug = f"why-teams-leave-{_slugify(vendor_name)}-{month_suffix}"
        ctx.update({
            "from_vendor": vendor_name,
            "total_reviews": stats.get("total", 0),
            "high_urgency_count": stats.get("negative", 0),
            "switch_mentions": 0,
            "avg_urgency": stats.get("avg_urgency", 0),
            "slug": slug,
        })
    elif topic_type == "market_landscape":
        slug = f"{_slugify(category)}-landscape-{month_suffix}"
        ctx.update({
            "vendor_count": category_stats.get("vendor_count", 0),
            "total_reviews": category_stats.get("enriched_reviews", 0),
            "avg_urgency": category_stats.get("avg_urgency", 0),
            "slug": slug,
        })
    elif topic_type == "pain_point_roundup":
        slug = f"top-complaint-every-{_slugify(category)}-{month_suffix}"
        ctx.update({
            "vendor_count": category_stats.get("vendor_count", 0),
            "total_complaints": category_stats.get("enriched_reviews", 0),
            "avg_urgency": category_stats.get("avg_urgency", 0),
            "slug": slug,
        })
    elif topic_type == "best_fit_guide":
        size = (
            company_size
            or category_stats.get("dominant_size")
            or ctx.get("company_size")
            or "teams"
        )
        slug = f"best-{_slugify(category)}-for-{_slugify(size)}-{month_suffix}"
        ctx.update({
            "vendor_count": category_stats.get("vendor_count", 0),
            "total_reviews": category_stats.get("enriched_reviews", 0),
            "company_size": size,
            "slug": slug,
        })
    elif topic_type == "pricing_reality_check":
        slug = f"real-cost-of-{_slugify(vendor_name)}-{month_suffix}"
        ctx.update({
            "vendor": vendor_name,
            "total_reviews": stats.get("total", 0),
            "pricing_complaints": stats.get("negative", 0),
            "avg_urgency": stats.get("avg_urgency", 0),
            "slug": slug,
        })
    elif topic_type == "migration_guide":
        slug = f"switch-to-{_slugify(vendor_name)}-{month_suffix}"
        ctx.update({
            "vendor": vendor_name,
            "switch_count": 0,
            "review_total": stats.get("total", 0),
            "slug": slug,
        })
    elif topic_type == "vendor_alternative":
        slug = f"{_slugify(vendor_name)}-alternatives-{month_suffix}"
        ctx.update({
            "vendor": vendor_name,
            "urgency": stats.get("avg_urgency", 0),
            "review_count": stats.get("total", 0),
            "has_affiliate": False,
            "affiliate_id": None,
            "affiliate_name": None,
            "affiliate_product": None,
            "affiliate_url": None,
            "slug": slug,
        })
    elif topic_type == "churn_report":
        slug = f"{_slugify(vendor_name)}-churn-report-{month_suffix}"
        ctx.update({
            "vendor": vendor_name,
            "negative_reviews": stats.get("negative", 0),
            "avg_urgency": stats.get("avg_urgency", 0),
            "total_reviews": stats.get("total", 0),
            "slug": slug,
        })
    elif topic_type == "vendor_deep_dive":
        slug = f"{_slugify(vendor_name)}-deep-dive-{month_suffix}"
        ctx.update({
            "vendor": vendor_name,
            "review_count": stats.get("total", 0),
            "profile_richness": 0,
            "slug": slug,
        })
    else:
        raise ValueError(f"Unknown topic_type: {topic_type}")

    return ctx
