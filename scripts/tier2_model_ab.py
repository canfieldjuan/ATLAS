#!/usr/bin/env python3
"""Tier 2 Model A/B harness: Sonnet 4.5 vs Haiku 4.5.

Read-only against production data:
- Reads `b2b_reviews` and `b2b_llm_exact_cache` (Sonnet outputs).
- Calls OpenRouter Haiku 4.5 only.
- Writes a markdown results doc.
- Does NOT mutate enrichment rows or model config.
- Does NOT change routing.

Sample selection:
  Stratified by (source, vendor_name) with recency tie-breaker. Reviews must
  have non-null `enrichment` AND a Sonnet 4.5 cache entry under the
  `b2b_enrichment.tier2` namespace; rows without a cached Sonnet baseline
  are skipped.

Comparison metrics per row:
  - JSON validity (parses, dict, contains required top-level keys)
  - Enum completeness (canonical values for pain_categories, evidence_type,
    displacement_confidence, role_type, buying_stage, decision_timeline,
    contract_value_signal)
  - Attribution disagreement: per-row delta between models on
    competitors_mentioned[*].evidence_type promotions and on whether
    pain_categories is non-empty when Tier 1 found no specific_complaints.
  - Latency (ms per Haiku call)
  - Token usage and projected cost (Haiku only; Sonnet served from cache)

Cost guard:
  Hard-stop at --max-cost-usd estimated Haiku spend (default $1.00).

CLI:
  --sample-size N          target sample size (default 100)
  --limit N                run only N reviews from the sample
  --dry-run                select sample, lookup Sonnet, but do NOT call Haiku
  --review-id UUID         debug a single row
  --max-cost-usd 1.00      hard cap on estimated Haiku spend
  --output PATH            markdown results doc path

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/tier2_model_ab.py --dry-run --limit 5
  python scripts/tier2_model_ab.py --sample-size 100 --max-cost-usd 1.00
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any
from uuid import UUID

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

from atlas_brain.autonomous.tasks.b2b_enrichment import (  # type: ignore[import-not-found]
    _build_classify_payload,
    _lookup_cached_json_response,
    _tier2_system_prompt_for_content_type,
)
from atlas_brain.config import settings  # type: ignore[import-not-found]
from atlas_brain.pipelines.llm import parse_json_response  # type: ignore[import-not-found]
from atlas_brain.skills import get_skill_registry  # type: ignore[import-not-found]
from atlas_brain.storage.database import (  # type: ignore[import-not-found]
    close_database,
    get_db_pool,
    init_database,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tier2_model_ab")


# -- Pricing (Anthropic via OpenRouter, USD per 1M tokens, April 2026) -------

PRICING = {
    "anthropic/claude-sonnet-4-5": (3.0, 15.0),
    "anthropic/claude-haiku-4-5": (1.0, 5.0),
}

SONNET_MODEL = "anthropic/claude-sonnet-4-5"
HAIKU_MODEL = "anthropic/claude-haiku-4-5"


# -- Canonical enums (mirrors digest/b2b_churn_extraction_tier2.md) ----------

REQUIRED_TOP_LEVEL_KEYS = (
    "competitors_mentioned",
    "pain_categories",
    "sentiment_trajectory",
    "buyer_authority",
    "timeline",
    "contract_context",
    "positive_aspects",
    "feature_gaps",
    "recommendation_language",
    "pricing_phrases",
    "event_mentions",
    "urgency_indicators",
)

PAIN_CATEGORY_ENUM = {
    "pricing", "features", "reliability", "support", "integration",
    "performance", "security", "ux", "onboarding", "technical_debt",
    "contract_lock_in", "data_migration", "api_limitations", "outcome_gap",
    "admin_burden", "ai_hallucination", "integration_debt", "other",
}

PAIN_SEVERITY_ENUM = {"primary", "secondary", "minor"}

EVIDENCE_TYPE_ENUM = {
    "explicit_switch", "active_evaluation", "implied_preference",
    "reverse_flow", "neutral_mention",
}

DISPLACEMENT_CONFIDENCE_ENUM = {"high", "medium", "low", "none"}

REASON_CATEGORY_ENUM = {
    "pricing", "features", "reliability", "ux", "support", "integration",
}

ROLE_TYPE_ENUM = {
    "economic_buyer", "champion", "evaluator", "end_user", "unknown",
}

BUYING_STAGE_ENUM = {
    "active_purchase", "evaluation", "renewal_decision", "post_purchase",
    "unknown",
}

DECISION_TIMELINE_ENUM = {
    "immediate", "within_quarter", "within_year", "unknown",
}

CONTRACT_VALUE_ENUM = {
    "enterprise_high", "enterprise_mid", "mid_market", "smb", "unknown",
}


# -- Sample selection --------------------------------------------------------


_SAMPLE_QUERY = """
WITH ranked AS (
    SELECT
        r.id,
        r.vendor_name,
        COALESCE(r.source, '') AS source,
        r.enriched_at,
        ROW_NUMBER() OVER (
            PARTITION BY COALESCE(r.source, ''), COALESCE(r.vendor_name, '')
            ORDER BY r.enriched_at DESC NULLS LAST
        ) AS strata_rank
    FROM b2b_reviews r
    WHERE r.enrichment IS NOT NULL
      AND r.enriched_at IS NOT NULL
      AND r.review_text IS NOT NULL
)
SELECT id, vendor_name, source, enriched_at, strata_rank
FROM ranked
ORDER BY strata_rank ASC, enriched_at DESC
LIMIT $1;
"""


async def select_sample(pool, sample_size: int) -> list[dict[str, Any]]:
    """Pull sample_size review ids stratified by (source, vendor)."""
    rows = await pool.fetch(_SAMPLE_QUERY, sample_size)
    return [dict(r) for r in rows]


_REVIEW_QUERY = """
SELECT
    id, vendor_name, product_name, product_category, source, content_type,
    rating, rating_max, summary, review_text, pros, cons, reviewer_title,
    reviewer_company, company_size_raw, reviewer_industry, raw_metadata,
    enrichment
FROM b2b_reviews
WHERE id = $1;
"""


async def fetch_review(pool, review_id: UUID) -> dict[str, Any] | None:
    row = await pool.fetchrow(_REVIEW_QUERY, review_id)
    return dict(row) if row else None


def extract_tier1_from_enrichment(enrichment: Any) -> dict[str, Any]:
    """Pull the Tier 1 fields the Tier 2 payload depends on."""
    if isinstance(enrichment, str):
        try:
            enrichment = json.loads(enrichment)
        except (json.JSONDecodeError, TypeError):
            return {"specific_complaints": [], "quotable_phrases": []}
    if not isinstance(enrichment, dict):
        return {"specific_complaints": [], "quotable_phrases": []}
    return {
        "specific_complaints": enrichment.get("specific_complaints") or [],
        "quotable_phrases": enrichment.get("quotable_phrases") or [],
    }


def build_tier2_payload(row: dict[str, Any], tier1: dict[str, Any]) -> dict[str, Any]:
    truncate = int(getattr(settings.b2b_churn, "review_truncate_length", 3000))
    payload = _build_classify_payload(row, truncate)
    payload["tier1_specific_complaints"] = tier1.get("specific_complaints") or []
    payload["tier1_quotable_phrases"] = tier1.get("quotable_phrases") or []
    return payload


def tier2_system_prompt(content_type: str | None) -> str:
    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    if skill is None:
        raise RuntimeError("digest/b2b_churn_extraction_tier2 skill not found")
    return _tier2_system_prompt_for_content_type(skill.content, content_type)


# -- LLM calls ---------------------------------------------------------------


async def lookup_sonnet_cache(
    payload: dict[str, Any],
    system_prompt: str,
    cfg,
    model: str,
) -> dict[str, Any] | None:
    """Lookup the cached Sonnet output by reproducing the Tier 2 cache key."""
    cached, _envelope, _hit = await _lookup_cached_json_response(
        "b2b_enrichment.tier2",
        provider="openrouter",
        model=model,
        system_prompt=system_prompt,
        user_content=json.dumps(payload),
        max_tokens=cfg.enrichment_tier2_max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return cached


async def call_haiku(
    payload: dict[str, Any],
    system_prompt: str,
    cfg,
    api_key: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], float]:
    """Call OpenRouter Haiku 4.5 once. Returns (parsed, usage, latency_ms)."""
    payload_json = json.dumps(payload)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload_json},
    ]
    timeout = httpx.Timeout(
        cfg.enrichment_tier2_timeout_seconds, connect=10.0
    )
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
                "messages": messages,
                "max_tokens": cfg.enrichment_tier2_max_tokens,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        body = resp.json()
    elapsed_ms = (time.monotonic() - started) * 1000
    raw = body.get("choices", [{}])[0].get("message", {}).get("content") or ""
    parsed = parse_json_response(raw, recover_truncated=True)
    if isinstance(parsed, dict) and parsed.get("_parse_fallback"):
        parsed = None
    return (parsed if isinstance(parsed, dict) else None), body.get("usage", {}) or {}, elapsed_ms


def estimate_call_cost(model: str, usage: dict[str, Any]) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    in_per_m, out_per_m = rates
    in_tok = float(usage.get("prompt_tokens") or 0)
    out_tok = float(usage.get("completion_tokens") or 0)
    return (in_tok / 1_000_000.0) * in_per_m + (out_tok / 1_000_000.0) * out_per_m


# -- Comparison metrics ------------------------------------------------------


def is_valid_top_level(parsed: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if not isinstance(parsed, dict):
        return False, list(REQUIRED_TOP_LEVEL_KEYS)
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in parsed]
    return (len(missing) == 0), missing


def enum_violations(parsed: dict[str, Any] | None) -> list[str]:
    """Return human-readable strings describing canonical-enum violations."""
    violations: list[str] = []
    if not isinstance(parsed, dict):
        return ["non-dict output"]

    pain = parsed.get("pain_categories") or []
    if isinstance(pain, list):
        for i, item in enumerate(pain):
            if not isinstance(item, dict):
                continue
            cat = item.get("category")
            if cat is not None and cat not in PAIN_CATEGORY_ENUM:
                violations.append(f"pain_categories[{i}].category={cat!r}")
            sev = item.get("severity")
            if sev is not None and sev not in PAIN_SEVERITY_ENUM:
                violations.append(f"pain_categories[{i}].severity={sev!r}")

    comps = parsed.get("competitors_mentioned") or []
    if isinstance(comps, list):
        for i, item in enumerate(comps):
            if not isinstance(item, dict):
                continue
            ev = item.get("evidence_type")
            if ev is not None and ev not in EVIDENCE_TYPE_ENUM:
                violations.append(f"competitors_mentioned[{i}].evidence_type={ev!r}")
            dc = item.get("displacement_confidence")
            if dc is not None and dc not in DISPLACEMENT_CONFIDENCE_ENUM:
                violations.append(f"competitors_mentioned[{i}].displacement_confidence={dc!r}")
            rc = item.get("reason_category")
            if rc is not None and rc not in REASON_CATEGORY_ENUM:
                violations.append(f"competitors_mentioned[{i}].reason_category={rc!r}")

    ba = parsed.get("buyer_authority")
    if isinstance(ba, dict):
        rt = ba.get("role_type")
        if rt is not None and rt not in ROLE_TYPE_ENUM:
            violations.append(f"buyer_authority.role_type={rt!r}")
        bs = ba.get("buying_stage")
        if bs is not None and bs not in BUYING_STAGE_ENUM:
            violations.append(f"buyer_authority.buying_stage={bs!r}")

    tl = parsed.get("timeline")
    if isinstance(tl, dict):
        dt = tl.get("decision_timeline")
        if dt is not None and dt not in DECISION_TIMELINE_ENUM:
            violations.append(f"timeline.decision_timeline={dt!r}")

    cc = parsed.get("contract_context")
    if isinstance(cc, dict):
        cv = cc.get("contract_value_signal")
        if cv is not None and cv not in CONTRACT_VALUE_ENUM:
            violations.append(f"contract_context.contract_value_signal={cv!r}")

    return violations


def count_explicit_promotions(parsed: dict[str, Any] | None) -> int:
    """Count competitor mentions promoted to explicit_switch or active_evaluation."""
    if not isinstance(parsed, dict):
        return 0
    comps = parsed.get("competitors_mentioned") or []
    if not isinstance(comps, list):
        return 0
    return sum(
        1
        for c in comps
        if isinstance(c, dict)
        and c.get("evidence_type") in {"explicit_switch", "active_evaluation"}
    )


def has_pain_without_tier1_complaints(
    parsed: dict[str, Any] | None,
    tier1_specific_complaints: list[Any],
) -> bool:
    """Skill rule: pain_categories should derive from tier1 complaints/quotes.

    A non-empty pain_categories with zero Tier 1 complaints is a leading
    indicator of self-transition or competitor language being misattributed
    as subject-vendor pain.
    """
    if not isinstance(parsed, dict):
        return False
    pain = parsed.get("pain_categories") or []
    return bool(pain) and not (tier1_specific_complaints or [])


def pain_category_set(parsed: dict[str, Any] | None) -> set[str]:
    if not isinstance(parsed, dict):
        return set()
    pain = parsed.get("pain_categories") or []
    if not isinstance(pain, list):
        return set()
    return {
        c.get("category")
        for c in pain
        if isinstance(c, dict) and isinstance(c.get("category"), str)
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# -- Markdown rendering ------------------------------------------------------


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def write_markdown(
    rollup: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# Tier 2 Model A/B — Sonnet 4.5 vs Haiku 4.5")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC._")
    lines.append("")
    lines.append("## Run config")
    lines.append("")
    lines.append(f"- Sample size requested: {rollup['sample_size_requested']}")
    lines.append(f"- Sample selected (cache hits on Sonnet): {rollup['sample_with_sonnet_cache']}")
    lines.append(f"- Haiku calls executed: {rollup['haiku_calls']}")
    lines.append(f"- Dry run: {rollup['dry_run']}")
    lines.append(f"- Cost cap (USD): ${rollup['max_cost_usd']:.2f}")
    lines.append(f"- Cost capped: {rollup['cost_capped']}")
    lines.append(f"- Sonnet model: `{SONNET_MODEL}` (cached, free)")
    lines.append(f"- Haiku model:  `{HAIKU_MODEL}`")
    if rollup["dry_run"]:
        lines.append("")
        lines.append(
            "> **Dry run notice.** No Haiku calls were made. The Haiku columns "
            "below reflect absence of data, not model quality. Use this output "
            "to verify sample selection and Sonnet cache coverage only — "
            "model-quality comparison requires a non-dry run."
        )
    lines.append("")
    lines.append("## Stratification coverage")
    lines.append("")
    lines.append(f"- Distinct (source, vendor) pairs: {rollup['distinct_pairs']}")
    lines.append(f"- Distinct sources: {rollup['distinct_sources']}")
    lines.append(f"- Distinct vendors: {rollup['distinct_vendors']}")
    lines.append("")
    lines.append("## Rollup")
    lines.append("")
    lines.append("| Metric | Sonnet 4.5 | Haiku 4.5 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| JSON valid + required keys | "
        f"{_format_pct(rollup['sonnet_valid'], rollup['n'])} ({rollup['sonnet_valid']}/{rollup['n']}) | "
        f"{_format_pct(rollup['haiku_valid'], rollup['n'])} ({rollup['haiku_valid']}/{rollup['n']}) |"
    )
    lines.append(
        f"| Enum-violation rate | "
        f"{_format_pct(rollup['sonnet_enum_violations'], rollup['n'])} | "
        f"{_format_pct(rollup['haiku_enum_violations'], rollup['n'])} |"
    )
    lines.append(
        f"| Explicit-displacement promotions (per row, mean) | "
        f"{rollup['sonnet_promotions_mean']:.2f} | "
        f"{rollup['haiku_promotions_mean']:.2f} |"
    )
    lines.append(
        f"| Pain-without-tier1-complaints (suspect attribution) | "
        f"{_format_pct(rollup['sonnet_pain_without_tier1'], rollup['n'])} | "
        f"{_format_pct(rollup['haiku_pain_without_tier1'], rollup['n'])} |"
    )
    lines.append(
        f"| Mean latency (ms) | n/a (cache) | {rollup['haiku_latency_ms_mean']:.0f} |"
    )
    lines.append(
        f"| Median latency (ms) | n/a (cache) | {rollup['haiku_latency_ms_median']:.0f} |"
    )
    lines.append(
        f"| Total Haiku spend (USD, projected) | n/a (cache) | "
        f"${rollup['haiku_total_cost_usd']:.4f} |"
    )
    lines.append("")
    lines.append("## Cross-model agreement")
    lines.append("")
    lines.append(f"- Pain-category Jaccard (mean across rows): {rollup['pain_jaccard_mean']:.3f}")
    lines.append(
        f"- Rows where Haiku promotes more displacement evidence than Sonnet: "
        f"{rollup['haiku_more_promotions']}/{rollup['n']}"
    )
    lines.append(
        f"- Rows where Sonnet promotes more displacement evidence than Haiku: "
        f"{rollup['sonnet_more_promotions']}/{rollup['n']}"
    )
    lines.append("")
    lines.append("## Per-row appendix")
    lines.append("")
    lines.append(
        "| review_id | vendor | source | sonnet_valid | haiku_valid | "
        "sonnet_promo | haiku_promo | pain_jaccard | haiku_ms | haiku_cost |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        lines.append(
            f"| `{r['review_id']}` "
            f"| {r['vendor_name']} "
            f"| {r['source']} "
            f"| {r['sonnet_valid']} "
            f"| {r['haiku_valid']} "
            f"| {r['sonnet_promotions']} "
            f"| {r['haiku_promotions']} "
            f"| {r['pain_jaccard']:.2f} "
            f"| {r['haiku_latency_ms']:.0f} "
            f"| ${r['haiku_cost_usd']:.4f} |"
        )
    lines.append("")
    lines.append("## Decision guidance")
    lines.append("")
    lines.append(
        "Routing change is intentionally deferred. Read the rollup above and "
        "the per-row appendix; if Haiku matches Sonnet on JSON validity, enum "
        "completeness, and promotion conservatism while running materially "
        "cheaper, a follow-up patch can flip the default. If Haiku promotes "
        "more displacement evidence or shows lower pain-category agreement, "
        "stay on Sonnet."
    )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


# -- Main --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--review-id", type=str, default=None)
    p.add_argument("--max-cost-usd", type=float, default=1.00)
    p.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / f"docs/progress/tier2_model_ab_{date.today().isoformat()}.md"),
    )
    return p.parse_args()


async def run(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY") or getattr(
        settings.b2b_churn, "openrouter_api_key", ""
    )
    if not api_key and not args.dry_run:
        logger.error("OPENROUTER_API_KEY missing; pass --dry-run to skip Haiku calls")
        return 2

    cfg = settings.b2b_churn

    await init_database()
    pool = get_db_pool()

    try:
        if args.review_id:
            sample_ids = [UUID(args.review_id)]
            sample_meta = [{"id": sample_ids[0], "vendor_name": "?", "source": "?"}]
        else:
            sample_meta = await select_sample(pool, args.sample_size)
            sample_ids = [m["id"] for m in sample_meta]
            if args.limit is not None:
                sample_ids = sample_ids[: args.limit]
                sample_meta = sample_meta[: args.limit]

        logger.info("Selected %d candidate review(s) for replay", len(sample_ids))

        rows_out: list[dict[str, Any]] = []
        haiku_calls = 0
        haiku_total_cost = 0.0
        haiku_latencies: list[float] = []
        sample_with_sonnet_cache = 0
        cost_capped = False

        for meta in sample_meta:
            review_id = meta["id"]
            row = await fetch_review(pool, review_id)
            if not row:
                logger.warning("Review %s not found, skipping", review_id)
                continue

            tier1 = extract_tier1_from_enrichment(row.get("enrichment"))
            payload = build_tier2_payload(row, tier1)
            system_prompt = tier2_system_prompt(row.get("content_type"))

            sonnet = await lookup_sonnet_cache(payload, system_prompt, cfg, SONNET_MODEL)
            if sonnet is None:
                logger.debug("No Sonnet cache for review %s, skipping", review_id)
                continue
            sample_with_sonnet_cache += 1

            haiku: dict[str, Any] | None = None
            haiku_usage: dict[str, Any] = {}
            haiku_latency_ms = 0.0
            haiku_cost = 0.0

            if not args.dry_run:
                # Cost guard: estimate the next call before executing.
                projected_next_cost = estimate_call_cost(
                    HAIKU_MODEL,
                    {"prompt_tokens": 600, "completion_tokens": 400},
                )
                if haiku_total_cost + projected_next_cost > args.max_cost_usd:
                    logger.warning(
                        "Cost cap reached: ${:.4f} + ${:.4f} > ${:.2f}, stopping",
                        haiku_total_cost,
                        projected_next_cost,
                        args.max_cost_usd,
                    )
                    cost_capped = True
                    break

                try:
                    haiku, haiku_usage, haiku_latency_ms = await call_haiku(
                        payload, system_prompt, cfg, api_key
                    )
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Haiku call failed for review %s: %s", review_id, exc
                    )
                    haiku = None
                haiku_calls += 1
                haiku_cost = estimate_call_cost(HAIKU_MODEL, haiku_usage)
                haiku_total_cost += haiku_cost
                haiku_latencies.append(haiku_latency_ms)

            sonnet_valid, _ = is_valid_top_level(sonnet)
            haiku_valid, _ = is_valid_top_level(haiku) if haiku is not None else (False, [])
            sonnet_violations = enum_violations(sonnet)
            haiku_violations = enum_violations(haiku) if haiku is not None else []
            sonnet_promo = count_explicit_promotions(sonnet)
            haiku_promo = count_explicit_promotions(haiku)
            sonnet_pain_orphan = has_pain_without_tier1_complaints(
                sonnet, tier1.get("specific_complaints") or []
            )
            haiku_pain_orphan = (
                has_pain_without_tier1_complaints(
                    haiku, tier1.get("specific_complaints") or []
                )
                if haiku is not None
                else False
            )
            pain_jacc = jaccard(pain_category_set(sonnet), pain_category_set(haiku))

            rows_out.append({
                "review_id": str(review_id),
                "vendor_name": row.get("vendor_name") or meta.get("vendor_name") or "?",
                "source": row.get("source") or meta.get("source") or "?",
                "sonnet_valid": sonnet_valid,
                "haiku_valid": haiku_valid,
                "sonnet_enum_violations": len(sonnet_violations),
                "haiku_enum_violations": len(haiku_violations),
                "sonnet_promotions": sonnet_promo,
                "haiku_promotions": haiku_promo,
                "sonnet_pain_orphan": sonnet_pain_orphan,
                "haiku_pain_orphan": haiku_pain_orphan,
                "pain_jaccard": pain_jacc,
                "haiku_latency_ms": haiku_latency_ms,
                "haiku_cost_usd": haiku_cost,
            })

        n = len(rows_out)
        if n == 0:
            logger.warning("No comparable rows produced; markdown will reflect zero data")

        rollup = {
            "sample_size_requested": args.sample_size,
            "sample_with_sonnet_cache": sample_with_sonnet_cache,
            "haiku_calls": haiku_calls,
            "dry_run": args.dry_run,
            "max_cost_usd": args.max_cost_usd,
            "cost_capped": cost_capped,
            "n": n,
            "distinct_pairs": len({(r["source"], r["vendor_name"]) for r in rows_out}),
            "distinct_sources": len({r["source"] for r in rows_out}),
            "distinct_vendors": len({r["vendor_name"] for r in rows_out}),
            "sonnet_valid": sum(1 for r in rows_out if r["sonnet_valid"]),
            "haiku_valid": sum(1 for r in rows_out if r["haiku_valid"]),
            "sonnet_enum_violations": sum(
                1 for r in rows_out if r["sonnet_enum_violations"] > 0
            ),
            "haiku_enum_violations": sum(
                1 for r in rows_out if r["haiku_enum_violations"] > 0
            ),
            "sonnet_promotions_mean": (
                mean(r["sonnet_promotions"] for r in rows_out) if n else 0.0
            ),
            "haiku_promotions_mean": (
                mean(r["haiku_promotions"] for r in rows_out) if n else 0.0
            ),
            "sonnet_pain_without_tier1": sum(
                1 for r in rows_out if r["sonnet_pain_orphan"]
            ),
            "haiku_pain_without_tier1": sum(
                1 for r in rows_out if r["haiku_pain_orphan"]
            ),
            "pain_jaccard_mean": (
                mean(r["pain_jaccard"] for r in rows_out) if n else 0.0
            ),
            "haiku_more_promotions": sum(
                1 for r in rows_out if r["haiku_promotions"] > r["sonnet_promotions"]
            ),
            "sonnet_more_promotions": sum(
                1 for r in rows_out if r["sonnet_promotions"] > r["haiku_promotions"]
            ),
            "haiku_latency_ms_mean": mean(haiku_latencies) if haiku_latencies else 0.0,
            "haiku_latency_ms_median": median(haiku_latencies) if haiku_latencies else 0.0,
            "haiku_total_cost_usd": haiku_total_cost,
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rollup, rows_out, out_path)
        logger.info("Wrote results to %s", out_path)

        return 0

    finally:
        await close_database()


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
