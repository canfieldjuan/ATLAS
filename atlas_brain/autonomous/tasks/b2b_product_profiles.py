"""
B2B Product Profile Generation: aggregate enriched review data into
pre-computed vendor knowledge cards for product matching.

Runs daily after b2b_churn_intelligence (9:30 PM). Reads enriched
b2b_reviews (last 90 days), aggregates strengths/weaknesses/competitive
flows per vendor, calls LLM for summary synthesis, upserts to
b2b_product_profiles.

Returns _skip_synthesis.
"""

import asyncio
import json
import logging
import re
from types import SimpleNamespace
from typing import Any

import math

import httpx

from ...config import settings
from ...services.scraping.sources import VERIFIED_SOURCES
from ...services.vendor_registry import resolve_vendor_name
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from ._execution_progress import task_run_id as _task_run_id


def _compute_profile_confidence(
    total_reviews: int,
    source_distribution: dict[str, int],
) -> float:
    """Evidence-based confidence score for product profiles.

    Same 3-signal formula as b2b_churn_intelligence._compute_evidence_confidence.
    """
    _VERIFIED = {s.value for s in VERIFIED_SOURCES}
    mention_weight = min(math.log2(max(total_reviews, 1)) / math.log2(20), 1.0)
    n_sources = len(source_distribution)
    source_weight = min(n_sources / 3.0, 1.0)
    total = sum(source_distribution.values()) or 1
    verified_total = sum(cnt for src, cnt in source_distribution.items() if src in _VERIFIED)
    quality_weight = verified_total / total
    return round((mention_weight + source_weight + quality_weight) / 3.0, 2)

logger = logging.getLogger("atlas.autonomous.tasks.b2b_product_profiles")

# Pain categories from the enrichment extraction schema
_PAIN_CATEGORIES = [
    "pricing",
    "features",
    "reliability",
    "support",
    "integration",
    "performance",
    "security",
    "ux",
    "onboarding",
    "other",
]


def _normalize_test_vendors(raw: Any) -> list[str]:
    """Normalize optional runtime vendor scope from task metadata."""
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
        return [value for value in values if value]
    if isinstance(raw, (list, tuple, set)):
        normalized: list[str] = []
        for item in raw:
            value = str(item or "").strip()
            if value:
                normalized.append(value)
        return normalized
    value = str(raw).strip()
    return [value] if value else []


def _profile_batch_custom_id(raw_vendor: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_vendor or "").strip())
    normalized = normalized.strip("_") or "vendor"
    return f"profile_{normalized[:56]}"


def _safe_json(value: Any, default: Any = None) -> Any:
    """Safely deserialize a JSON value."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _profile_review_vendor_join(
    review_alias: str = "r",
    mention_alias: str = "vm",
) -> str:
    """Join canonical review-to-vendor mentions for profile aggregates."""
    return (
        f"JOIN b2b_review_vendor_mentions {mention_alias} "
        f"ON {mention_alias}.review_id = {review_alias}.id"
    )


# ------------------------------------------------------------------
# SQL fetchers -- one per data dimension, all run in parallel
# ------------------------------------------------------------------


async def _fetch_satisfaction_by_area(pool, window_days: int) -> dict[str, list[dict]]:
    """Per-vendor feature satisfaction: invert pain categories + use rating as proxy.

    Returns {vendor_name: [{"area": str, "score": float, "evidence_count": int}, ...]}.
    Strengths = high-rated areas, Weaknesses = low-rated areas.
    """
    # APPROVED-ENRICHMENT-READ: pain_category
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name,
               r.enrichment->>'pain_category' AS pain_cat,
               AVG(r.rating) AS avg_rating,
               COUNT(*) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND r.enrichment->>'pain_category' IS NOT NULL
        GROUP BY vm.vendor_name, r.enrichment->>'pain_category'
        HAVING COUNT(*) >= 2
        ORDER BY vm.vendor_name, avg_rating DESC
        """,
        window_days,
    )

    result: dict[str, list[dict]] = {}
    for r in rows:
        vendor = r["vendor_name"]
        result.setdefault(vendor, []).append({
            "area": r["pain_cat"],
            "score": round(_safe_float(r["avg_rating"]), 2),
            "evidence_count": r["cnt"],
        })
    return result


async def _fetch_pain_distribution(pool, window_days: int) -> dict[str, dict[str, int]]:
    """Pain frequency per vendor: {vendor: {pain_cat: count}}."""
    # APPROVED-ENRICHMENT-READ: pain_category
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name,
               r.enrichment->>'pain_category' AS pain_cat,
               COUNT(*) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND r.enrichment->>'pain_category' IS NOT NULL
        GROUP BY vm.vendor_name, r.enrichment->>'pain_category'
        ORDER BY vm.vendor_name, cnt DESC
        """,
        window_days,
    )
    result: dict[str, dict[str, int]] = {}
    for r in rows:
        result.setdefault(r["vendor_name"], {})[r["pain_cat"]] = r["cnt"]
    return result


async def _fetch_use_case_distribution(pool, window_days: int) -> dict[str, list[dict]]:
    """Primary workflows per vendor."""
    # APPROVED-ENRICHMENT-READ: use_case.primary_workflow
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name,
               r.enrichment->'use_case'->>'primary_workflow' AS workflow,
               COUNT(*) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND r.enrichment->'use_case'->>'primary_workflow' IS NOT NULL
        GROUP BY vm.vendor_name, r.enrichment->'use_case'->>'primary_workflow'
        ORDER BY vm.vendor_name, cnt DESC
        """,
        window_days,
    )
    result: dict[str, list[dict]] = {}
    for r in rows:
        result.setdefault(r["vendor_name"], []).append({
            "use_case": r["workflow"],
            "count": r["cnt"],
        })
    return result


async def _fetch_company_size_distribution(pool, window_days: int) -> dict[str, dict[str, int]]:
    """Company size segments per vendor."""
    # APPROVED-ENRICHMENT-READ: reviewer_context.company_size_segment
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name,
               r.enrichment->'reviewer_context'->>'company_size_segment' AS seg,
               COUNT(*) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND r.enrichment->'reviewer_context'->>'company_size_segment' IS NOT NULL
          AND r.enrichment->'reviewer_context'->>'company_size_segment' != 'unknown'
        GROUP BY vm.vendor_name, r.enrichment->'reviewer_context'->>'company_size_segment'
        ORDER BY vm.vendor_name, cnt DESC
        """,
        window_days,
    )
    result: dict[str, dict[str, int]] = {}
    for r in rows:
        result.setdefault(r["vendor_name"], {})[r["seg"]] = r["cnt"]
    return result


async def _fetch_competitive_flows(pool, window_days: int) -> dict[str, dict]:
    """Competitive switching data per vendor.

    Returns {vendor: {"compared_to": [...], "switched_from": [...]}}.
    """
    # APPROVED-ENRICHMENT-READ: competitors_mentioned
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name,
               comp->>'name' AS comp_name,
               comp->>'context' AS comp_context,
               comp->>'reason' AS comp_reason
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id,
             jsonb_array_elements(r.enrichment->'competitors_mentioned') AS comp
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND jsonb_typeof(r.enrichment->'competitors_mentioned') = 'array'
        """,
        window_days,
    )

    result: dict[str, dict] = {}
    for r in rows:
        vendor = r["vendor_name"]
        data = result.setdefault(vendor, {"compared_to": {}, "switched_from": {}})
        comp_name = r["comp_name"]
        context = r["comp_context"] or ""

        if not comp_name:
            continue

        if context in ("considering", "compared", "switched_to"):
            entry = data["compared_to"].setdefault(comp_name, {"mentions": 0, "reasons": []})
            entry["mentions"] += 1
            if r["comp_reason"] and r["comp_reason"] not in entry["reasons"]:
                entry["reasons"].append(r["comp_reason"])
        elif context == "switched_from":
            entry = data["switched_from"].setdefault(comp_name, {"count": 0, "reasons": []})
            entry["count"] += 1
            if r["comp_reason"] and r["comp_reason"] not in entry["reasons"]:
                entry["reasons"].append(r["comp_reason"])

    return result


async def _fetch_integration_stacks(pool, window_days: int) -> dict[str, dict[str, int]]:
    """Top integrations per vendor."""
    # APPROVED-ENRICHMENT-READ: use_case.integration_stack
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name,
               integ::text AS integration
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id,
             jsonb_array_elements_text(r.enrichment->'use_case'->'integration_stack') AS integ
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
          AND jsonb_typeof(r.enrichment->'use_case'->'integration_stack') = 'array'
        """,
        window_days,
    )
    result: dict[str, dict[str, int]] = {}
    for r in rows:
        vendor_data = result.setdefault(r["vendor_name"], {})
        name = r["integration"]
        vendor_data[name] = vendor_data.get(name, 0) + 1
    return result


async def _fetch_aggregate_metrics(pool, window_days: int, min_reviews: int) -> dict[str, dict]:
    """Per-vendor aggregate metrics (rating, recommend rate, urgency, review count).

    Groups by vendor_name only (not product_category) to produce one profile
    per vendor. The most common product_category is selected via MODE().
    Also collects provenance: sample_review_ids, review_window_start/end.
    """
    # APPROVED-ENRICHMENT-READ: would_recommend, urgency_score
    # Reason: inline aggregate query, structurally coupled to product output
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name,
               MODE() WITHIN GROUP (ORDER BY r.product_category) AS product_category,
               COUNT(*) AS total_reviews,
               AVG(r.rating) AS avg_rating,
               AVG(CASE WHEN (r.enrichment->>'would_recommend')::boolean THEN 1.0 ELSE 0.0 END)
                   FILTER (WHERE r.enrichment->>'would_recommend' IS NOT NULL) AS recommend_rate,
               AVG((r.enrichment->>'urgency_score')::numeric)
                   FILTER (WHERE r.enrichment->>'urgency_score' IS NOT NULL) AS avg_urgency,
               (ARRAY_AGG(r.id ORDER BY (r.enrichment->>'urgency_score')::numeric DESC NULLS LAST))[1:50]
                   AS sample_review_ids,
               MIN(r.enriched_at) AS review_window_start,
               MAX(r.enriched_at) AS review_window_end
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vm.vendor_name
        HAVING COUNT(*) >= $2
        ORDER BY total_reviews DESC
        """,
        window_days, min_reviews,
    )
    result: dict[str, dict] = {}
    for r in rows:
        result[r["vendor_name"]] = {
            "product_category": r["product_category"],
            "total_reviews": r["total_reviews"],
            "avg_rating": round(_safe_float(r["avg_rating"]), 2),
            "recommend_rate": round(_safe_float(r["recommend_rate"]), 2) if r["recommend_rate"] is not None else None,
            "avg_urgency": round(_safe_float(r["avg_urgency"]), 1) if r["avg_urgency"] is not None else None,
            "sample_review_ids": r["sample_review_ids"] or [],
            "review_window_start": r["review_window_start"],
            "review_window_end": r["review_window_end"],
        }
    return result


async def _fetch_source_distribution(pool, window_days: int) -> dict[str, dict[str, int]]:
    """Per-vendor review source distribution: {vendor: {source: count}}."""
    rows = await pool.fetch(
        """
        SELECT vm.vendor_name, r.source, count(*) AS cnt
        FROM b2b_reviews r
        JOIN b2b_review_vendor_mentions vm ON vm.review_id = r.id
        WHERE r.enrichment_status = 'enriched'
          AND r.duplicate_of_review_id IS NULL
          AND r.enriched_at > NOW() - make_interval(days => $1)
        GROUP BY vm.vendor_name, r.source
        """,
        window_days,
    )
    result: dict[str, dict[str, int]] = {}
    for r in rows:
        result.setdefault(r["vendor_name"], {})[r["source"]] = r["cnt"]
    return result


# ------------------------------------------------------------------
# Profile assembly
# ------------------------------------------------------------------


def _build_strengths_weaknesses(
    satisfaction: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split satisfaction data into strengths (score >= 3.5) and weaknesses (score < 3.5)."""
    strengths = []
    weaknesses = []
    for item in satisfaction:
        if item["score"] >= 3.5:
            strengths.append(item)
        else:
            weaknesses.append(item)
    strengths.sort(
        key=lambda x: (
            -_safe_float(x.get("score")),
            -int(x.get("evidence_count") or 0),
            str(x.get("area") or ""),
        )
    )
    weaknesses.sort(
        key=lambda x: (
            _safe_float(x.get("score")),
            -int(x.get("evidence_count") or 0),
            str(x.get("area") or ""),
        )
    )
    return strengths[:10], weaknesses[:10]


def _build_use_cases(use_case_data: list[dict], total_reviews: int) -> list[dict]:
    """Convert use case counts to fit scores."""
    if not use_case_data or total_reviews == 0:
        return []
    result = []
    ranked = sorted(
        use_case_data,
        key=lambda uc: (
            -int(uc.get("count") or 0),
            str(uc.get("use_case") or ""),
        ),
    )
    for uc in ranked[:10]:
        fit_score = round(min(uc["count"] / total_reviews, 1.0), 2)
        result.append({"use_case": uc["use_case"], "fit_score": fit_score})
    return result


def _build_company_size(size_data: dict[str, int]) -> dict[str, float]:
    """Convert size counts to proportions."""
    total = sum(size_data.values())
    if total == 0:
        return {}
    # Map enrichment segments to display buckets
    segment_map = {
        "startup": "1-50",
        "smb": "51-200",
        "mid_market": "201-1000",
        "enterprise": "1000+",
    }
    result = {}
    for seg, count in size_data.items():
        bucket = segment_map.get(seg, seg)
        result[bucket] = round(count / total, 2)
    return result


def _build_competitive_positioning(comp_data: dict) -> tuple[list[dict], list[dict]]:
    """Build commonly_compared_to and commonly_switched_from."""
    compared_to = []
    for comp_name, data in comp_data.get("compared_to", {}).items():
        mentions = data["mentions"]
        compared_to.append({
            "vendor": comp_name,
            "mentions": mentions,
        })
    compared_to.sort(
        key=lambda x: (
            -int(x.get("mentions") or 0),
            str(x.get("vendor") or ""),
        )
    )

    switched_from = []
    for comp_name, data in comp_data.get("switched_from", {}).items():
        top_reason = data["reasons"][0] if data["reasons"] else None
        switched_from.append({
            "vendor": comp_name,
            "count": data["count"],
            "top_reason": top_reason,
        })
    switched_from.sort(
        key=lambda x: (
            -int(x.get("count") or 0),
            str(x.get("vendor") or ""),
            str(x.get("top_reason") or ""),
        )
    )

    return compared_to[:10], switched_from[:10]


def _build_top_integrations(integration_data: dict[str, int]) -> list[str]:
    """Top integrations sorted by frequency."""
    sorted_items = sorted(
        integration_data.items(),
        key=lambda x: (
            -int(x[1] or 0),
            str(x[0] or ""),
        ),
    )
    return [name for name, _ in sorted_items[:15]]


def _compute_pain_addressed_heuristic(
    pain_dist: dict[str, int],
    satisfaction: list[dict],
    total_reviews: int,
) -> dict[str, float]:
    """Heuristic pain_addressed scores before LLM validation.

    High score = this vendor does NOT suffer from this pain (it's a strength).
    Low score = this vendor suffers from this pain (it's a weakness).
    """
    if total_reviews == 0:
        return {}

    # Build a score map from satisfaction data
    area_scores: dict[str, float] = {}
    for item in satisfaction:
        area_scores[item["area"]] = item["score"]

    result = {}
    for cat in _PAIN_CATEGORIES:
        pain_count = pain_dist.get(cat, 0)
        pain_rate = pain_count / total_reviews

        # If this vendor has high pain rate in this category, they DON'T address it well
        # Invert: score = 1 - pain_rate (capped)
        score = max(0.0, min(1.0, 1.0 - (pain_rate * 2)))

        # Boost if there's positive satisfaction data for this area
        if cat in area_scores and area_scores[cat] >= 4.0:
            score = min(1.0, score + 0.15)

        result[cat] = round(score, 2)

    return result


# ------------------------------------------------------------------
# LLM synthesis
# ------------------------------------------------------------------


async def _call_vllm(messages: list[dict], max_tokens: int, cfg, client: httpx.AsyncClient) -> str:
    """Call local vLLM server for profile synthesis."""
    resp = await client.post(
        f"{cfg.product_profile_vllm_url}/v1/chat/completions",
        json={
            "model": cfg.product_profile_vllm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _product_profile_openrouter_model(cfg: Any) -> str:
    from ...pipelines.llm import normalize_openrouter_model

    return normalize_openrouter_model(
        getattr(cfg, "product_profile_openrouter_model", ""),
        context="product profile synthesis",
    )


async def _call_openrouter(messages: list[dict], max_tokens: int, cfg, client: httpx.AsyncClient) -> str:
    """Call OpenRouter API for profile synthesis."""
    import os

    api_key = cfg.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OpenRouter API key not configured for product profiles")

    resp = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": _product_profile_openrouter_model(cfg),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _log_product_profile_cache_trace(
    *,
    cfg,
    phase: str,
    request,
    vendor_name: str,
    payload: dict[str, Any],
    cached: dict[str, Any] | None = None,
    response_text: str | None = None,
) -> None:
    """Emit targeted exact-cache trace logs for product profile synthesis."""
    if not bool(getattr(cfg, "product_profile_cache_trace_enabled", False)):
        return

    from ...services.b2b import llm_exact_cache

    envelope_hash = llm_exact_cache.compute_cache_key("__trace__", request.request_envelope)
    cache_key = llm_exact_cache.compute_cache_key(request.namespace, request.request_envelope)
    payload_hash = llm_exact_cache.canonicalize_for_cache(payload)
    logger.info(
        "product_profile_cache_trace phase=%s vendor=%s namespace=%s key=%s env=%s payload=%s hit_count=%s response_len=%s provider=%s model=%s",
        phase,
        vendor_name,
        request.namespace,
        cache_key[:16],
        envelope_hash[:16],
        payload_hash[:16],
        cached.get("hit_count") if isinstance(cached, dict) else None,
        len(str(response_text or "")) if response_text is not None else None,
        request.provider,
        request.model,
    )


def _build_profile_synthesis_payload(
    *,
    vendor_name: str,
    metrics: dict[str, Any],
    strengths: list[dict],
    weaknesses: list[dict],
    use_cases: list[dict],
    integrations: list[str],
    competitive: dict[str, Any],
) -> dict[str, Any]:
    return {
        "vendor_name": vendor_name,
        "product_category": metrics.get("product_category"),
        "total_reviews": metrics.get("total_reviews", 0),
        "avg_rating": metrics.get("avg_rating"),
        "strengths": strengths[:8],
        "weaknesses": weaknesses[:8],
        "use_cases": use_cases[:5],
        "integrations": integrations[:10],
        "competitive_data": {
            "commonly_compared_to": competitive.get("compared_to", {})
            if isinstance(competitive.get("compared_to"), dict)
            else competitive.get("compared_to", []),
            "commonly_switched_from": competitive.get("switched_from", {})
            if isinstance(competitive.get("switched_from"), dict)
            else competitive.get("switched_from", []),
        },
        "pain_categories": _PAIN_CATEGORIES,
    }


def _clean_profile_synthesis_text(text: str | None) -> str | None:
    if not text:
        return None
    candidate = str(text or "").strip()
    candidate = re.sub(r"<think>[\s\S]*?</think>", "", candidate).strip()
    candidate = re.sub(r"^```(?:json)?\s*\n?", "", candidate)
    candidate = re.sub(r"\n?```\s*$", "", candidate)
    candidate = candidate.strip()
    return candidate or None


def _validated_profile_pain_scores(
    llm_pain: dict[str, Any],
    pain_heuristic: dict[str, float],
) -> dict[str, float]:
    validated_pain: dict[str, float] = {}
    for cat in _PAIN_CATEGORIES:
        if cat in llm_pain:
            score = _safe_float(llm_pain[cat], -1)
            if 0.0 <= score <= 1.0:
                validated_pain[cat] = round(score, 2)
            else:
                validated_pain[cat] = pain_heuristic.get(cat, 0.5)
        else:
            validated_pain[cat] = pain_heuristic.get(cat, 0.5)
    return validated_pain


def _parse_profile_synthesis_text(
    text: str | None,
    *,
    pain_heuristic: dict[str, float],
) -> tuple[str | None, dict[str, float]] | None:
    cleaned = _clean_profile_synthesis_text(text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return (
        parsed.get("summary"),
        _validated_profile_pain_scores(
            parsed.get("pain_addressed", {}) if isinstance(parsed.get("pain_addressed"), dict) else {},
            pain_heuristic,
        ),
    )


async def _synthesize_profile(
    vendor_name: str,
    metrics: dict,
    strengths: list[dict],
    weaknesses: list[dict],
    use_cases: list[dict],
    integrations: list[str],
    competitive: dict,
    pain_heuristic: dict[str, float],
    max_tokens: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[str | None, dict[str, float]]:
    """Call LLM to generate profile summary and validate pain_addressed scores.

    Returns (summary, pain_addressed).
    """
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("b2b/product_profile_synthesis")
    if not skill:
        logger.warning("Skill 'b2b/product_profile_synthesis' not found, using heuristic only")
        return None, pain_heuristic

    payload = _build_profile_synthesis_payload(
        vendor_name=vendor_name,
        metrics=metrics,
        strengths=strengths,
        weaknesses=weaknesses,
        use_cases=use_cases,
        integrations=integrations,
        competitive=competitive,
    )

    cfg = settings.b2b_churn
    backend = cfg.product_profile_llm_backend
    model_id = (
        _product_profile_openrouter_model(cfg)
        if backend == "openrouter"
        else cfg.product_profile_vllm_model
    )

    messages = [
        {"role": "system", "content": skill.content},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]

    _client = client or httpx.AsyncClient(timeout=120)
    _owns_client = client is None

    try:
        request = prepare_b2b_exact_stage_request(
            "b2b_product_profiles.synthesis",
            provider=backend,
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        cached = await lookup_b2b_exact_stage_text(request)
        _log_product_profile_cache_trace(
            cfg=cfg,
            phase="lookup_hit" if cached is not None else "lookup_miss",
            request=request,
            vendor_name=vendor_name,
            payload=payload,
            cached=cached,
        )
        cache_hit = cached is not None
        text: str | None = None
        if cached is not None:
            text = _clean_profile_synthesis_text(str(cached["response_text"] or ""))
            if _parse_profile_synthesis_text(text, pain_heuristic=pain_heuristic) is None:
                text = None

        if backend == "openrouter":
            if text is None:
                text = await _call_openrouter(messages, max_tokens, cfg, _client)
        elif text is None:
            text = await _call_vllm(messages, max_tokens, cfg, _client)

        if not text:
            return None, pain_heuristic

        parsed = _parse_profile_synthesis_text(text, pain_heuristic=pain_heuristic)
        if parsed is None:
            raise json.JSONDecodeError("Invalid product profile synthesis JSON", text, 0)
        summary, validated_pain = parsed

        if not cache_hit:
            await store_b2b_exact_stage_text(
                request,
                response_text=_clean_profile_synthesis_text(text) or text,
                metadata={
                    "vendor_name": vendor_name,
                    "product_category": metrics.get("product_category"),
                },
            )
            _log_product_profile_cache_trace(
                cfg=cfg,
                phase="store",
                request=request,
                vendor_name=vendor_name,
                payload=payload,
                response_text=text,
            )
        return summary, validated_pain

    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM synthesis for %s", vendor_name)
        return None, pain_heuristic
    except Exception:
        logger.exception("LLM synthesis failed for %s", vendor_name)
        return None, pain_heuristic
    finally:
        if _owns_client:
            await _client.aclose()


# ------------------------------------------------------------------
# Upsert
# ------------------------------------------------------------------


async def _upsert_profile(pool, profile: dict) -> None:
    """Insert or update a product profile row (20 columns incl. provenance)."""
    await pool.execute(
        """
        INSERT INTO b2b_product_profiles (
            vendor_name, product_category,
            strengths, weaknesses, pain_addressed,
            total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
            primary_use_cases, typical_company_size, typical_industries,
            top_integrations, commonly_compared_to, commonly_switched_from,
            profile_summary,
            source_distribution, sample_review_ids,
            review_window_start, review_window_end,
            confidence_score,
            last_computed_at
        ) VALUES (
            $1, $2,
            $3::jsonb, $4::jsonb, $5::jsonb,
            $6, $7, $8, $9,
            $10::jsonb, $11::jsonb, $12::jsonb,
            $13::jsonb, $14::jsonb, $15::jsonb,
            $16,
            $17::jsonb, $18,
            $19, $20,
            $21,
            NOW()
        )
        ON CONFLICT (vendor_name, COALESCE(product_category, ''))
        DO UPDATE SET
            strengths = EXCLUDED.strengths,
            weaknesses = EXCLUDED.weaknesses,
            pain_addressed = EXCLUDED.pain_addressed,
            total_reviews_analyzed = EXCLUDED.total_reviews_analyzed,
            avg_rating = EXCLUDED.avg_rating,
            recommend_rate = EXCLUDED.recommend_rate,
            avg_urgency = EXCLUDED.avg_urgency,
            primary_use_cases = EXCLUDED.primary_use_cases,
            typical_company_size = EXCLUDED.typical_company_size,
            typical_industries = EXCLUDED.typical_industries,
            top_integrations = EXCLUDED.top_integrations,
            commonly_compared_to = EXCLUDED.commonly_compared_to,
            commonly_switched_from = EXCLUDED.commonly_switched_from,
            profile_summary = EXCLUDED.profile_summary,
            source_distribution = EXCLUDED.source_distribution,
            sample_review_ids = EXCLUDED.sample_review_ids,
            review_window_start = EXCLUDED.review_window_start,
            review_window_end = EXCLUDED.review_window_end,
            confidence_score = EXCLUDED.confidence_score,
            last_computed_at = NOW()
        """,
        profile["vendor_name"],
        profile.get("product_category"),
        json.dumps(profile["strengths"]),
        json.dumps(profile["weaknesses"]),
        json.dumps(profile["pain_addressed"]),
        profile["total_reviews_analyzed"],
        profile.get("avg_rating"),
        profile.get("recommend_rate"),
        profile.get("avg_urgency"),
        json.dumps(profile["primary_use_cases"]),
        json.dumps(profile["typical_company_size"]),
        json.dumps(profile["typical_industries"]),
        json.dumps(profile["top_integrations"]),
        json.dumps(profile["commonly_compared_to"]),
        json.dumps(profile["commonly_switched_from"]),
        profile.get("profile_summary"),
        json.dumps(profile.get("source_distribution", {})),
        profile.get("sample_review_ids", []),
        profile.get("review_window_start"),
        profile.get("review_window_end"),
        profile.get("confidence_score", 0),
    )


# ------------------------------------------------------------------
# Snapshot capture
# ------------------------------------------------------------------


async def _persist_profile_snapshots(pool, profiles_generated: int) -> int:
    """Capture daily snapshots of all product profiles (append-only)."""
    if profiles_generated == 0:
        return 0

    try:
        rows = await pool.fetch(
            """
            SELECT vendor_name, total_reviews_analyzed, avg_rating,
                   recommend_rate, avg_urgency, strengths, weaknesses,
                   primary_use_cases, top_integrations,
                   commonly_compared_to, commonly_switched_from,
                   pain_addressed, profile_summary
            FROM b2b_product_profiles
            """
        )
    except Exception:
        logger.exception("Failed to fetch profiles for snapshot")
        return 0

    persisted = 0
    for r in rows:
        try:
            strengths = r["strengths"] if isinstance(r["strengths"], list) else json.loads(r["strengths"] or "[]")
            weaknesses = r["weaknesses"] if isinstance(r["weaknesses"], list) else json.loads(r["weaknesses"] or "[]")
            use_cases = r["primary_use_cases"] if isinstance(r["primary_use_cases"], list) else json.loads(r["primary_use_cases"] or "[]")
            integrations = r["top_integrations"] if isinstance(r["top_integrations"], list) else json.loads(r["top_integrations"] or "[]")
            compared_to = r["commonly_compared_to"] if isinstance(r["commonly_compared_to"], list) else json.loads(r["commonly_compared_to"] or "[]")
            switched_from = r["commonly_switched_from"] if isinstance(r["commonly_switched_from"], list) else json.loads(r["commonly_switched_from"] or "[]")
            pain_addressed = r["pain_addressed"] if isinstance(r["pain_addressed"], dict) else json.loads(r["pain_addressed"] or "{}")

            top_strength = strengths[0]["area"] if strengths else None
            top_weakness = weaknesses[0]["area"] if weaknesses else None
            top_use_case = use_cases[0]["use_case"] if use_cases and isinstance(use_cases[0], dict) else (use_cases[0] if use_cases else None)
            top_integration = integrations[0]["tool"] if integrations and isinstance(integrations[0], dict) else (integrations[0] if integrations else None)
            summary_len = len(r["profile_summary"] or "")

            await pool.execute(
                """
                INSERT INTO b2b_product_profile_snapshots (
                    vendor_name, snapshot_date,
                    total_reviews_analyzed, avg_rating, recommend_rate, avg_urgency,
                    strength_count, weakness_count, top_strength, top_weakness,
                    top_use_case, top_integration,
                    compared_to_count, switched_from_count,
                    pain_categories_covered, profile_summary_len
                ) VALUES (
                    $1, CURRENT_DATE,
                    $2, $3, $4, $5,
                    $6, $7, $8, $9,
                    $10, $11,
                    $12, $13,
                    $14, $15
                )
                ON CONFLICT (vendor_name, snapshot_date) DO UPDATE SET
                    total_reviews_analyzed = EXCLUDED.total_reviews_analyzed,
                    avg_rating = EXCLUDED.avg_rating,
                    recommend_rate = EXCLUDED.recommend_rate,
                    avg_urgency = EXCLUDED.avg_urgency,
                    strength_count = EXCLUDED.strength_count,
                    weakness_count = EXCLUDED.weakness_count,
                    top_strength = EXCLUDED.top_strength,
                    top_weakness = EXCLUDED.top_weakness,
                    top_use_case = EXCLUDED.top_use_case,
                    top_integration = EXCLUDED.top_integration,
                    compared_to_count = EXCLUDED.compared_to_count,
                    switched_from_count = EXCLUDED.switched_from_count,
                    pain_categories_covered = EXCLUDED.pain_categories_covered,
                    profile_summary_len = EXCLUDED.profile_summary_len
                """,
                r["vendor_name"],
                r["total_reviews_analyzed"],
                r["avg_rating"],
                r["recommend_rate"],
                r["avg_urgency"],
                len(strengths),
                len(weaknesses),
                top_strength,
                top_weakness,
                top_use_case,
                top_integration,
                len(compared_to),
                len(switched_from),
                len(pain_addressed),
                summary_len,
            )
            persisted += 1
        except Exception:
            logger.warning("Failed to snapshot profile for %s", r["vendor_name"])

    logger.info("Product profile snapshots: %d persisted", persisted)
    return persisted


# ------------------------------------------------------------------
# Main handler
# ------------------------------------------------------------------


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate/refresh B2B product profile knowledge cards."""
    cfg = settings.b2b_churn
    if not cfg.enabled or not cfg.product_profile_enabled:
        return {"_skip_synthesis": "B2B product profiles disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    window_days = 90
    min_reviews = cfg.product_profile_min_reviews
    max_tokens = cfg.product_profile_max_tokens

    logger.info("Starting product profile generation (window=%dd, min_reviews=%d)", window_days, min_reviews)
    run_id = _task_run_id(task)

    # 1. Fetch all data dimensions in parallel
    (
        satisfaction_data,
        pain_data,
        use_case_data,
        company_size_data,
        competitive_data,
        integration_data,
        aggregate_metrics,
        source_dist_data,
    ) = await asyncio.gather(
        _fetch_satisfaction_by_area(pool, window_days),
        _fetch_pain_distribution(pool, window_days),
        _fetch_use_case_distribution(pool, window_days),
        _fetch_company_size_distribution(pool, window_days),
        _fetch_competitive_flows(pool, window_days),
        _fetch_integration_stacks(pool, window_days),
        _fetch_aggregate_metrics(pool, window_days, min_reviews),
        _fetch_source_distribution(pool, window_days),
    )

    scoped_vendors = _normalize_test_vendors((task.metadata or {}).get("test_vendors"))
    if scoped_vendors:
        scoped_vendor_keys = {vendor.lower().strip() for vendor in scoped_vendors if vendor}
        aggregate_metrics = {
            vendor: metrics
            for vendor, metrics in aggregate_metrics.items()
            if vendor.lower().strip() in scoped_vendor_keys
        }
        logger.info(
            "Product profile generation scoped to %d vendor(s): %s",
            len(aggregate_metrics),
            sorted(aggregate_metrics),
        )

    if not aggregate_metrics:
        logger.info("No vendors with >= %d enriched reviews, nothing to generate", min_reviews)
        return {"_skip_synthesis": "No vendors meet min review threshold", "vendors": 0}

    # 2. Build vendor contexts, then synthesize/upsert profiles.
    generated = 0
    failed = 0
    batch_metrics = {
        "anthropic_batch_jobs": 0,
        "anthropic_batch_items_submitted": 0,
        "anthropic_batch_cache_prefiltered_items": 0,
        "anthropic_batch_fallback_single_call_items": 0,
        "anthropic_batch_completed_items": 0,
        "anthropic_batch_failed_items": 0,
        "anthropic_batch_reused_completed_items": 0,
        "anthropic_batch_reused_pending_items": 0,
    }
    vendor_entries: list[dict[str, Any]] = []
    for raw_vendor, metrics in aggregate_metrics.items():
        vendor_name = await resolve_vendor_name(raw_vendor)
        satisfaction = satisfaction_data.get(raw_vendor, [])
        strengths, weaknesses = _build_strengths_weaknesses(satisfaction)
        use_cases = _build_use_cases(
            use_case_data.get(raw_vendor, []),
            metrics["total_reviews"],
        )
        company_size = _build_company_size(
            company_size_data.get(raw_vendor, {}),
        )
        compared_to, switched_from = _build_competitive_positioning(
            competitive_data.get(raw_vendor, {}),
        )
        integrations = _build_top_integrations(
            integration_data.get(raw_vendor, {}),
        )
        pain_heuristic = _compute_pain_addressed_heuristic(
            pain_data.get(raw_vendor, {}),
            satisfaction,
            metrics["total_reviews"],
        )
        vendor_entries.append(
            {
                "raw_vendor": raw_vendor,
                "vendor_name": vendor_name,
                "metrics": metrics,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "use_cases": use_cases,
                "company_size": company_size,
                "compared_to": compared_to,
                "switched_from": switched_from,
                "integrations": integrations,
                "pain_heuristic": pain_heuristic,
                "source_distribution": source_dist_data.get(raw_vendor, {}),
            }
        )

    from ...services.b2b.anthropic_batch import (
        AnthropicBatchItem,
        mark_batch_fallback_result,
        run_anthropic_message_batch,
    )
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )
    from ...services.llm.anthropic import AnthropicBatchableLLM
    from ...services.protocols import Message
    from ...skills import get_skill_registry
    from ._b2b_batch_utils import (
        anthropic_batch_min_items,
        anthropic_batch_requested,
        reconcile_existing_batch_artifacts,
        resolve_anthropic_batch_llm,
    )

    backend = str(cfg.product_profile_llm_backend or "").strip().lower()
    batch_requested = (
        backend == "openrouter"
        and anthropic_batch_requested(
            task,
            global_default=bool(getattr(settings.b2b_churn, "anthropic_batch_enabled", False)),
            task_default=bool(getattr(cfg, "product_profile_anthropic_batch_enabled", True)),
            task_keys=("product_profile_anthropic_batch_enabled",),
        )
    )
    batch_llm = (
        resolve_anthropic_batch_llm(
            current_llm=SimpleNamespace(name="openrouter", model=_product_profile_openrouter_model(cfg)),
            target_model_candidates=(_product_profile_openrouter_model(cfg),),
        )
        if batch_requested
        else None
    )
    if not isinstance(batch_llm, AnthropicBatchableLLM):
        batch_llm = None
    skill = get_skill_registry().get("b2b/product_profile_synthesis")

    async def _upsert_vendor_profile(
        entry: dict[str, Any],
        *,
        summary: str | None,
        pain_addressed: dict[str, float],
    ) -> None:
        profile = {
            "vendor_name": entry["vendor_name"],
            "product_category": entry["metrics"].get("product_category"),
            "strengths": entry["strengths"],
            "weaknesses": entry["weaknesses"],
            "pain_addressed": pain_addressed,
            "total_reviews_analyzed": entry["metrics"]["total_reviews"],
            "avg_rating": entry["metrics"].get("avg_rating"),
            "recommend_rate": entry["metrics"].get("recommend_rate"),
            "avg_urgency": entry["metrics"].get("avg_urgency"),
            "primary_use_cases": entry["use_cases"],
            "typical_company_size": entry["company_size"],
            "typical_industries": [],
            "top_integrations": entry["integrations"],
            "commonly_compared_to": entry["compared_to"],
            "commonly_switched_from": entry["switched_from"],
            "profile_summary": summary,
            "source_distribution": entry["source_distribution"],
            "sample_review_ids": entry["metrics"].get("sample_review_ids", []),
            "review_window_start": entry["metrics"].get("review_window_start"),
            "review_window_end": entry["metrics"].get("review_window_end"),
            "confidence_score": _compute_profile_confidence(
                entry["metrics"]["total_reviews"],
                entry["source_distribution"],
            ),
        }
        await _upsert_profile(pool, profile)
        logger.info(
            "Generated profile for %s (%d reviews, %d strengths, %d weaknesses)",
            entry["vendor_name"],
            entry["metrics"]["total_reviews"],
            len(entry["strengths"]),
            len(entry["weaknesses"]),
        )

    _sem = asyncio.Semaphore(5)
    _http_client = httpx.AsyncClient(timeout=120)

    async def _fallback_process_entry(entry: dict[str, Any]) -> bool:
        nonlocal generated, failed
        try:
            async with _sem:
                summary, pain_addressed = await _synthesize_profile(
                    vendor_name=entry["vendor_name"],
                    metrics=entry["metrics"],
                    strengths=entry["strengths"],
                    weaknesses=entry["weaknesses"],
                    use_cases=entry["use_cases"],
                    integrations=entry["integrations"],
                    competitive={
                        "compared_to": entry["compared_to"],
                        "switched_from": entry["switched_from"],
                    },
                    pain_heuristic=entry["pain_heuristic"],
                    max_tokens=max_tokens,
                    client=_http_client,
                )
            await _upsert_vendor_profile(entry, summary=summary, pain_addressed=pain_addressed)
            generated += 1
            return True
        except Exception:
            logger.exception("Failed to generate profile for %s", entry["raw_vendor"])
            failed += 1
            return False

    try:
        if batch_llm is not None and skill is not None:
            existing_batch_results = await reconcile_existing_batch_artifacts(
                pool=pool,
                llm=batch_llm,
                task_name="b2b_product_profiles",
                artifact_type="product_profile",
                artifact_ids=[str(entry["raw_vendor"]) for entry in vendor_entries],
            )
            pending_existing_items = 0
            remaining_vendor_entries: list[dict[str, Any]] = []
            for entry in vendor_entries:
                existing = existing_batch_results.get(str(entry["raw_vendor"]))
                if existing and existing.get("state") == "succeeded":
                    parsed = _parse_profile_synthesis_text(
                        existing.get("response_text"),
                        pain_heuristic=entry["pain_heuristic"],
                    )
                    if parsed is not None:
                        summary, pain_addressed = parsed
                        await _upsert_vendor_profile(
                            entry,
                            summary=summary,
                            pain_addressed=pain_addressed,
                        )
                        generated += 1
                        continue
                if existing and existing.get("state") == "pending":
                    pending_existing_items += 1
                    logger.info(
                        "Skipping duplicate product-profile submission for %s; existing Anthropic batch item %s is still pending",
                        entry["vendor_name"],
                        existing.get("custom_id"),
                    )
                    continue
                remaining_vendor_entries.append(entry)

            vendor_entries = remaining_vendor_entries
            batch_metrics["anthropic_batch_reused_completed_items"] = generated
            batch_metrics["anthropic_batch_reused_pending_items"] = pending_existing_items

            if not vendor_entries:
                snapshots_persisted = await _persist_profile_snapshots(pool, generated)
                return {
                    "vendors_processed": generated,
                    "vendors_failed": failed,
                    "total_eligible": len(aggregate_metrics),
                    "snapshots_persisted": snapshots_persisted,
                    **batch_metrics,
                }

            prepared_entries: list[dict[str, Any]] = []
            for entry in vendor_entries:
                payload = _build_profile_synthesis_payload(
                    vendor_name=entry["vendor_name"],
                    metrics=entry["metrics"],
                    strengths=entry["strengths"],
                    weaknesses=entry["weaknesses"],
                    use_cases=entry["use_cases"],
                    integrations=entry["integrations"],
                    competitive={
                        "compared_to": entry["compared_to"],
                        "switched_from": entry["switched_from"],
                    },
                )
                messages = [
                    {"role": "system", "content": skill.content},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ]
                request = prepare_b2b_exact_stage_request(
                    "b2b_product_profiles.synthesis",
                    provider="openrouter",
                    model=_product_profile_openrouter_model(cfg),
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                cached = await lookup_b2b_exact_stage_text(request)
                _log_product_profile_cache_trace(
                    cfg=cfg,
                    phase="lookup_hit" if cached is not None else "lookup_miss",
                    request=request,
                    vendor_name=entry["vendor_name"],
                    payload=payload,
                    cached=cached,
                )
                prepared_entries.append(
                    {
                        **entry,
                        "payload": payload,
                        "request": request,
                        "messages": messages,
                        "cached_response_text": str(cached["response_text"] or "") if cached is not None else None,
                        "cached_usage": dict(cached.get("usage") or {}) if cached is not None else {},
                    }
                )

            execution = await run_anthropic_message_batch(
                llm=batch_llm,
                stage_id="b2b_product_profiles.synthesis",
                task_name="b2b_product_profiles",
                items=[
                    AnthropicBatchItem(
                        custom_id=_profile_batch_custom_id(str(entry["raw_vendor"])),
                        artifact_type="product_profile",
                        artifact_id=str(entry["raw_vendor"]),
                        vendor_name=entry["vendor_name"],
                        messages=[
                            Message(role=str(message["role"]), content=str(message["content"]))
                            for message in entry["messages"]
                        ],
                        max_tokens=max_tokens,
                        temperature=0.3,
                        trace_span_name="task.b2b_product_profiles.synthesis",
                        trace_metadata={
                            "vendor_name": entry["vendor_name"],
                            "raw_vendor": entry["raw_vendor"],
                            "workload": "anthropic_batch",
                        },
                        request_metadata={"vendor_name": entry["vendor_name"]},
                        cached_response_text=entry["cached_response_text"],
                        cached_usage=entry["cached_usage"],
                    )
                    for entry in prepared_entries
                ],
                run_id=run_id,
                min_batch_size=anthropic_batch_min_items(
                    task,
                    default=int(getattr(cfg, "product_profile_anthropic_batch_min_items", 2)),
                    keys=("product_profile_anthropic_batch_min_items",),
                ),
                batch_metadata={"stage": "profile_synthesis"},
                pool=pool,
            )
            batch_metrics["anthropic_batch_jobs"] += 1 if execution.provider_batch_id else 0
            batch_metrics["anthropic_batch_items_submitted"] += execution.submitted_items
            batch_metrics["anthropic_batch_cache_prefiltered_items"] += execution.cache_prefiltered_items
            batch_metrics["anthropic_batch_fallback_single_call_items"] += execution.fallback_single_call_items
            batch_metrics["anthropic_batch_completed_items"] += execution.completed_items
            batch_metrics["anthropic_batch_failed_items"] += execution.failed_items

            fallback_entries: list[dict[str, Any]] = []
            for entry in prepared_entries:
                custom_id = _profile_batch_custom_id(str(entry["raw_vendor"]))
                outcome = execution.results_by_custom_id.get(custom_id)
                parsed = _parse_profile_synthesis_text(
                    outcome.response_text if outcome is not None else None,
                    pain_heuristic=entry["pain_heuristic"],
                )
                if parsed is None:
                    fallback_entries.append(entry)
                    if outcome is not None:
                        await mark_batch_fallback_result(
                            batch_id=execution.local_batch_id,
                            custom_id=custom_id,
                            succeeded=False,
                            error_text=outcome.error_text or "profile_batch_parse_failed",
                            pool=pool,
                        )
                    continue

                summary, pain_addressed = parsed
                if not (outcome is not None and outcome.cached):
                    cleaned = _clean_profile_synthesis_text(outcome.response_text if outcome is not None else None)
                    if cleaned:
                        await store_b2b_exact_stage_text(
                            entry["request"],
                            response_text=cleaned,
                            metadata={
                                "vendor_name": entry["vendor_name"],
                                "product_category": entry["metrics"].get("product_category"),
                                "backend": "anthropic_batch",
                            },
                        )
                        _log_product_profile_cache_trace(
                            cfg=cfg,
                            phase="store",
                            request=entry["request"],
                            vendor_name=entry["vendor_name"],
                            payload=entry["payload"],
                            response_text=cleaned,
                        )
                await _upsert_vendor_profile(entry, summary=summary, pain_addressed=pain_addressed)
                generated += 1

            if fallback_entries:
                await asyncio.gather(
                    *[_fallback_process_entry(entry) for entry in fallback_entries],
                    return_exceptions=True,
                )
        else:
            await asyncio.gather(
                *[_fallback_process_entry(entry) for entry in vendor_entries],
                return_exceptions=True,
            )
    finally:
        await _http_client.aclose()

    logger.info(
        "Product profile generation complete: %d generated, %d failed",
        generated, failed,
    )

    # 3. Capture daily snapshots
    snapshots_persisted = 0
    if generated > 0:
        snapshots_persisted = await _persist_profile_snapshots(pool, generated)

    return {
        "_skip_synthesis": "B2B product profiles complete",
        "vendors_processed": generated,
        "failed": failed,
        "total_eligible": len(aggregate_metrics),
        "snapshots_persisted": snapshots_persisted,
        **batch_metrics,
    }
