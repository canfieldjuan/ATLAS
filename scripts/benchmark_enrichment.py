#!/usr/bin/env python3
"""Benchmark B2B enrichment models against the production end-to-end pipeline.

This script mirrors the deployed enrichment flow as closely as possible:
  1. Skip too-short reviews (`not_applicable`)
  2. Run triage for non-verified sources
  3. Skip full extraction when triage returns `signal=false` (`no_signal`)
  4. Build the production extraction payload with smart truncation
  5. Clean, parse, and validate/coerce extraction output with production logic

It is therefore a benchmark of pipeline outcomes, not just raw prompt responses.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/benchmark_enrichment.py [--per-bucket 10] [--output /tmp/enrichment_benchmark.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

from atlas_brain.autonomous.tasks.b2b_enrichment import (
    _MIN_REVIEW_TEXT_LENGTH,
    _build_classify_payload,
    _validate_enrichment,
)
from atlas_brain.config import settings
from atlas_brain.pipelines.llm import clean_llm_output
from atlas_brain.services.scraping.sources import VERIFIED_SOURCES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_enrichment")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("atlas.llm").setLevel(logging.WARNING)
logging.getLogger("atlas.registry").setLevel(logging.WARNING)


BASELINE = {
    "id": "openrouter/hunter-alpha",
    "label": "Hunter Alpha",
    "input_cost": 0.00,
    "output_cost": 0.00,
}

CHALLENGERS = [
    {
        "id": "openai/gpt-4.1",
        "label": "GPT-4.1",
        "input_cost": 2.00,
        "output_cost": 8.00,
    },
]

ALL_MODELS = [BASELINE] + CHALLENGERS


async def fetch_sample(pool, per_bucket: int = 10):
    """Pull a stratified sample from raw review attributes, not prior enrichment labels."""
    buckets = {}
    verified_sources = sorted(VERIFIED_SOURCES)

    buckets["verified_reviews"] = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category, source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment, enrichment_status
        FROM b2b_reviews
        WHERE source = ANY($1::text[])
          AND length(COALESCE(review_text, '')) >= $2
        ORDER BY random()
        LIMIT $3
        """,
        verified_sources,
        _MIN_REVIEW_TEXT_LENGTH,
        per_bucket,
    )

    buckets["competitive_language"] = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category, source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment, enrichment_status
        FROM b2b_reviews
        WHERE length(COALESCE(review_text, '')) >= $1
          AND (
            COALESCE(summary, '') || ' ' || COALESCE(review_text, '')
          ) ~* $2
        ORDER BY random()
        LIMIT $3
        """,
        _MIN_REVIEW_TEXT_LENGTH,
        '(switch(?:ed|ing)?|migrat(?:e|ed|ing)|replac(?:e|ed|ing)|alternative|alternatives|\bvs\b|versus|renewal|shortlist|POC|proof of concept|evaluat(?:e|ing|ion))',
        per_bucket,
    )

    buckets["job_noise"] = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category, source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment, enrichment_status
        FROM b2b_reviews
        WHERE source <> ALL($1::text[])
          AND length(COALESCE(review_text, '')) >= $2
          AND (
            COALESCE(summary, '') || ' ' || COALESCE(review_text, '')
          ) ~* $3
        ORDER BY random()
        LIMIT $4
        """,
        verified_sources,
        _MIN_REVIEW_TEXT_LENGTH,
        '(career advice|interview|interviewing|interviewed|recruiter|resume|salary|job hunt|job search|hiring|offer|leetcode)',
        per_bucket,
    )

    buckets["insider"] = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category, source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment, enrichment_status
        FROM b2b_reviews
        WHERE content_type = 'insider_account'
          AND length(COALESCE(review_text, '')) >= $1
        ORDER BY random()
        LIMIT $2
        """,
        _MIN_REVIEW_TEXT_LENGTH,
        per_bucket,
    )

    buckets["reddit"] = await pool.fetch(
        """
        SELECT id, vendor_name, product_name, product_category, source, raw_metadata,
               rating, rating_max, summary, review_text, pros, cons,
               reviewer_title, reviewer_company, company_size_raw,
               reviewer_industry, content_type, enrichment, enrichment_status
        FROM b2b_reviews
        WHERE source = 'reddit'
          AND content_type = 'community_discussion'
          AND length(COALESCE(review_text, '')) >= $1
        ORDER BY random()
        LIMIT $2
        """,
        _MIN_REVIEW_TEXT_LENGTH,
        per_bucket,
    )

    logger.info("Raw sample sizes: %s", {k: len(v) for k, v in buckets.items()})
    return buckets


def build_triage_payload(row) -> str:
    """Build the same compact payload production triage uses."""
    review_text = (row.get("review_text") or "")[:1500]
    payload = {
        "vendor_name": row["vendor_name"],
        "source": row.get("source") or "",
        "content_type": row.get("content_type") or "review",
        "rating": float(row["rating"]) if row.get("rating") is not None else None,
        "summary": row.get("summary") or "",
        "review_text": review_text,
        "pros": (row.get("pros") or "")[:300],
        "cons": (row.get("cons") or "")[:300],
    }
    return json.dumps(payload)


async def call_openrouter(
    client: httpx.AsyncClient,
    model_id: str,
    system_prompt: str,
    user_content: str,
    api_key: str,
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Call a model via OpenRouter."""
    t0 = time.monotonic()
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body["response_format"] = {"type": "json_object"}

    try:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://atlas-brain.local",
                "X-Title": "Atlas Enrichment Benchmark",
            },
            json=body,
            timeout=120,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        usage = data.get("usage", {})
        return {
            "success": True,
            "content": content,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "latency_ms": round(elapsed_ms),
        }
    except Exception as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "success": False,
            "content": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round(elapsed_ms),
            "error": str(exc),
        }


def parse_json_response(raw_text: str) -> dict[str, Any] | None:
    """Clean and parse model output like production does."""
    try:
        text = clean_llm_output(raw_text or "")
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def score_extraction(parsed: dict[str, Any] | None, review_text: str) -> dict[str, Any]:
    """Quality heuristics on a validated extraction result."""
    if not parsed or not isinstance(parsed, dict):
        return {"valid_json": False}

    scores: dict[str, Any] = {"valid_json": True}
    urg = parsed.get("urgency_score")
    scores["has_urgency"] = isinstance(urg, (int, float)) and 0 <= urg <= 10

    valid_pains = {
        "pricing", "features", "reliability", "support", "integration",
        "performance", "security", "ux", "onboarding", "other",
    }
    scores["valid_pain_category"] = parsed.get("pain_category") in valid_pains

    comps = parsed.get("competitors_mentioned") or []
    scores["competitor_count"] = len(comps)
    text_lower = (review_text or "").lower()
    if comps:
        found = sum(
            1 for comp in comps
            if isinstance(comp, dict) and comp.get("name", "").lower() in text_lower
        )
        scores["competitors_grounded"] = found
        scores["competitors_hallucinated"] = len(comps) - found
    else:
        scores["competitors_grounded"] = 0
        scores["competitors_hallucinated"] = 0

    valid_evidence = {
        "explicit_switch", "active_evaluation", "implied_preference",
        "reverse_flow", "neutral_mention",
    }
    if comps:
        valid_et = sum(
            1 for comp in comps
            if isinstance(comp, dict) and comp.get("evidence_type") in valid_evidence
        )
        scores["valid_evidence_types"] = valid_et
        scores["invalid_evidence_types"] = len(comps) - valid_et
    else:
        scores["valid_evidence_types"] = 0
        scores["invalid_evidence_types"] = 0

    quotes = parsed.get("quotable_phrases") or []
    scores["quote_count"] = len(quotes)
    if quotes:
        grounded = sum(
            1 for quote in quotes
            if isinstance(quote, str) and quote.lower()[:30] in text_lower
        )
        scores["quotes_grounded"] = grounded
    else:
        scores["quotes_grounded"] = 0

    ctx = parsed.get("reviewer_context") or {}
    role_level = ctx.get("role_level")
    decision_maker = ctx.get("decision_maker")
    if role_level in ("executive", "director"):
        scores["dm_consistent"] = decision_maker is True
    elif role_level == "ic":
        scores["dm_consistent"] = decision_maker is False
    else:
        scores["dm_consistent"] = True

    expected_keys = {
        "churn_signals", "urgency_score", "reviewer_context", "pain_category",
        "specific_complaints", "feature_gaps", "competitors_mentioned",
        "contract_context", "quotable_phrases", "positive_aspects",
        "would_recommend", "pain_categories", "budget_signals", "use_case",
        "sentiment_trajectory", "buyer_authority", "timeline",
        "content_classification", "insider_signals",
    }
    scores["schema_keys_present"] = len(expected_keys & set(parsed.keys()))
    scores["schema_keys_expected"] = len(expected_keys)
    return scores


def aggregate_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pipeline and extraction scores across all reviews."""
    if not results:
        return {}

    enriched = [r for r in results if r.get("pipeline_outcome") == "enriched"]
    agg: dict[str, Any] = {
        "total": len(results),
        "enriched": len(enriched),
        "no_signal": sum(1 for r in results if r.get("pipeline_outcome") == "no_signal"),
        "not_applicable": sum(1 for r in results if r.get("pipeline_outcome") == "not_applicable"),
        "triage_failed": sum(1 for r in results if r.get("pipeline_outcome") == "triage_failed"),
        "invalid_extraction": sum(1 for r in results if r.get("pipeline_outcome") == "invalid_extraction"),
        "api_error": sum(1 for r in results if r.get("pipeline_outcome") == "api_error"),
        "valid_json": sum(1 for r in results if r["scores"].get("valid_json")),
        "valid_json_pct": 0.0,
        "enriched_pct": round(len(enriched) * 100 / len(results), 1),
        "avg_latency_ms": 0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
    }

    agg["valid_json_pct"] = round(agg["valid_json"] * 100 / agg["total"], 1)

    if enriched:
        agg["has_urgency"] = sum(1 for r in enriched if r["scores"].get("has_urgency"))
        agg["valid_pain_category"] = sum(1 for r in enriched if r["scores"].get("valid_pain_category"))
        agg["total_competitors"] = sum(r["scores"].get("competitor_count", 0) for r in enriched)
        agg["competitors_grounded"] = sum(r["scores"].get("competitors_grounded", 0) for r in enriched)
        agg["competitors_hallucinated"] = sum(r["scores"].get("competitors_hallucinated", 0) for r in enriched)
        agg["valid_evidence_types"] = sum(r["scores"].get("valid_evidence_types", 0) for r in enriched)
        agg["invalid_evidence_types"] = sum(r["scores"].get("invalid_evidence_types", 0) for r in enriched)
        agg["total_quotes"] = sum(r["scores"].get("quote_count", 0) for r in enriched)
        agg["quotes_grounded"] = sum(r["scores"].get("quotes_grounded", 0) for r in enriched)
        agg["dm_consistent"] = sum(1 for r in enriched if r["scores"].get("dm_consistent"))

        schema_scores = [r["scores"].get("schema_keys_present", 0) for r in enriched]
        expected = enriched[0]["scores"].get("schema_keys_expected", 19)
        agg["schema_completeness_avg"] = round(
            sum(schema_scores) * 100 / (len(schema_scores) * expected), 1
        )

        if agg["total_competitors"] > 0:
            agg["competitor_grounding_pct"] = round(
                agg["competitors_grounded"] * 100 / agg["total_competitors"], 1
            )
            agg["competitor_hallucination_pct"] = round(
                agg["competitors_hallucinated"] * 100 / agg["total_competitors"], 1
            )

        if agg["total_quotes"] > 0:
            agg["quote_grounding_pct"] = round(
                agg["quotes_grounded"] * 100 / agg["total_quotes"], 1
            )

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    if latencies:
        agg["avg_latency_ms"] = int(sum(latencies) / len(latencies))
        agg["p95_latency_ms"] = int(sorted(latencies)[int(len(latencies) * 0.95)])

    return agg


def _cost_usd(result: dict[str, Any], model: dict[str, Any]) -> float:
    return round(
        (result["input_tokens"] / 1_000_000) * model["input_cost"]
        + (result["output_tokens"] / 1_000_000) * model["output_cost"],
        6,
    )


async def run_pipeline_for_model(
    client: httpx.AsyncClient,
    model: dict[str, Any],
    entry: dict[str, Any],
    api_key: str,
    extraction_prompt: str,
    triage_prompt: str,
) -> dict[str, Any]:
    """Run the production-like enrichment pipeline for one model and one review."""
    row = entry["row"]
    review_text = row.get("review_text") or ""
    source = (row.get("source") or "").lower().strip()
    is_verified = source in VERIFIED_SOURCES

    output: dict[str, Any] = {
        "review_id": entry["review_id"],
        "pipeline_outcome": "api_error",
        "triage": None,
        "parsed": None,
        "scores": {"valid_json": False},
        "latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "error": None,
    }

    if len(review_text) < _MIN_REVIEW_TEXT_LENGTH:
        output["pipeline_outcome"] = "not_applicable"
        return output

    total_latency_ms = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    if not is_verified:
        triage_result = await call_openrouter(
            client,
            model["id"],
            triage_prompt,
            build_triage_payload(row),
            api_key,
            max_tokens=256,
            temperature=0.0,
        )
        total_latency_ms += triage_result["latency_ms"]
        total_input_tokens += triage_result["input_tokens"]
        total_output_tokens += triage_result["output_tokens"]
        total_cost_usd += _cost_usd(triage_result, model)

        if not triage_result["success"]:
            output["pipeline_outcome"] = "api_error"
            output["error"] = triage_result.get("error")
            output["latency_ms"] = total_latency_ms
            output["input_tokens"] = total_input_tokens
            output["output_tokens"] = total_output_tokens
            output["cost_usd"] = round(total_cost_usd, 6)
            return output

        triage = parse_json_response(triage_result["content"])
        output["triage"] = triage
        if triage is None or "signal" not in triage:
            output["pipeline_outcome"] = "triage_failed"
            output["latency_ms"] = total_latency_ms
            output["input_tokens"] = total_input_tokens
            output["output_tokens"] = total_output_tokens
            output["cost_usd"] = round(total_cost_usd, 6)
            return output

        if not triage.get("signal", True):
            output["pipeline_outcome"] = "no_signal"
            output["latency_ms"] = total_latency_ms
            output["input_tokens"] = total_input_tokens
            output["output_tokens"] = total_output_tokens
            output["cost_usd"] = round(total_cost_usd, 6)
            return output

    extraction_result = await call_openrouter(
        client,
        model["id"],
        extraction_prompt,
        json.dumps(_build_classify_payload(row, settings.b2b_churn.review_truncate_length)),
        api_key,
        max_tokens=settings.b2b_churn.enrichment_max_tokens,
        temperature=0.1,
    )
    total_latency_ms += extraction_result["latency_ms"]
    total_input_tokens += extraction_result["input_tokens"]
    total_output_tokens += extraction_result["output_tokens"]
    total_cost_usd += _cost_usd(extraction_result, model)

    output["latency_ms"] = total_latency_ms
    output["input_tokens"] = total_input_tokens
    output["output_tokens"] = total_output_tokens
    output["cost_usd"] = round(total_cost_usd, 6)

    if not extraction_result["success"]:
        output["pipeline_outcome"] = "api_error"
        output["error"] = extraction_result.get("error")
        return output

    parsed = parse_json_response(extraction_result["content"])
    if parsed and _validate_enrichment(parsed, row):
        output["pipeline_outcome"] = "enriched"
        output["parsed"] = parsed
        output["scores"] = score_extraction(parsed, review_text)
        return output

    output["pipeline_outcome"] = "invalid_extraction"
    return output


async def main():
    ap = argparse.ArgumentParser(description="Benchmark B2B enrichment models end-to-end")
    ap.add_argument("--per-bucket", type=int, default=10,
                    help="Reviews per sample bucket")
    ap.add_argument("--output", default="/tmp/enrichment_benchmark.json",
                    help="Output file path")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="Concurrent reviews per model")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        return

    from atlas_brain.skills import get_skill_registry
    from atlas_brain.storage.database import close_database, get_db_pool, init_database

    await init_database()
    pool = get_db_pool()
    logger.info("DB ready")

    buckets = await fetch_sample(pool, per_bucket=args.per_bucket)
    deduped_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_count = 0
    for bucket_name, rows in buckets.items():
        for row in rows:
            review_id = str(row["id"])
            if review_id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(review_id)
            deduped_entries.append({"bucket": bucket_name, "review_id": review_id, "row": row})

    if duplicate_count:
        logger.info("Dropped %d duplicate reviews across buckets", duplicate_count)
    if not deduped_entries:
        logger.error("No reviews found in sample")
        await close_database()
        return

    extraction_skill = get_skill_registry().get("digest/b2b_churn_extraction")
    triage_skill = get_skill_registry().get("digest/b2b_churn_triage")
    if not extraction_skill or not triage_skill:
        logger.error("Required skills missing: extraction=%s triage=%s", bool(extraction_skill), bool(triage_skill))
        await close_database()
        return

    entries: list[dict[str, Any]] = []
    for item in deduped_entries:
        row = item["row"]
        existing_enrichment = row.get("enrichment")
        if isinstance(existing_enrichment, str):
            try:
                existing_enrichment = json.loads(existing_enrichment)
            except (json.JSONDecodeError, TypeError):
                existing_enrichment = {}

        entries.append({
            "review_id": item["review_id"],
            "bucket": item["bucket"],
            "vendor": row["vendor_name"],
            "source": row.get("source"),
            "content_type": row.get("content_type"),
            "existing_status": row.get("enrichment_status"),
            "existing_urgency": existing_enrichment.get("urgency_score") if isinstance(existing_enrichment, dict) else None,
            "existing_pain": existing_enrichment.get("pain_category") if isinstance(existing_enrichment, dict) else None,
            "row": row,
        })

    logger.info("Benchmark sample: %d unique reviews", len(entries))

    model_results: dict[str, list[dict[str, Any]]] = {m["label"]: [] for m in ALL_MODELS}

    async with httpx.AsyncClient() as client:

        async def _run_model(model: dict[str, Any]) -> list[dict[str, Any]]:
            """Run full pipeline for one model across all entries (concurrently)."""
            label = model["label"]
            sem = asyncio.Semaphore(args.concurrency)
            done_count = 0

            async def _call_one(entry):
                nonlocal done_count
                async with sem:
                    result = await run_pipeline_for_model(
                        client,
                        model,
                        entry,
                        api_key,
                        extraction_skill.content,
                        triage_skill.content,
                    )
                done_count += 1
                if done_count % 10 == 0:
                    logger.info("  %s: %d/%d done", label, done_count, len(entries))
                return result

            logger.info("Running %s (%d reviews, concurrency=%d)...", label, len(entries), args.concurrency)
            batch_results = await asyncio.gather(
                *[_call_one(entry) for entry in entries], return_exceptions=True,
            )

            results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning("  %s review %d failed: %s", label, i, result)
                    results.append({
                        "review_id": entries[i]["review_id"],
                        "pipeline_outcome": "api_error",
                        "triage": None,
                        "parsed": None,
                        "latency_ms": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "scores": {"valid_json": False},
                        "error": str(result),
                    })
                else:
                    results.append(result)

            agg = aggregate_scores(results)
            logger.info(
                "  %s complete: enriched=%d no_signal=%d invalid=%d api_error=%d cost=$%.4f",
                label,
                agg.get("enriched", 0),
                agg.get("no_signal", 0),
                agg.get("invalid_extraction", 0),
                agg.get("api_error", 0),
                agg.get("total_cost_usd", 0.0),
            )
            return results

        # Run ALL models in parallel
        logger.info("Launching %d models in parallel...", len(ALL_MODELS))
        all_results = await asyncio.gather(
            *[_run_model(m) for m in ALL_MODELS], return_exceptions=True,
        )

        for model, result in zip(ALL_MODELS, all_results):
            if isinstance(result, Exception):
                logger.error("Model %s failed entirely: %s", model["label"], result)
                model_results[model["label"]] = [{
                    "review_id": e["review_id"],
                    "pipeline_outcome": "api_error",
                    "triage": None, "parsed": None,
                    "latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "scores": {"valid_json": False},
                    "error": str(result),
                } for e in entries]
            else:
                model_results[model["label"]] = result

    aggregates = {model["label"]: aggregate_scores(model_results[model["label"]]) for model in ALL_MODELS}

    baseline_label = BASELINE["label"]
    agreement: dict[str, Any] = {}
    for model in CHALLENGERS:
        label = model["label"]
        outcome_match = 0
        urgency_deltas = []
        pain_agree = 0
        pain_total = 0

        for baseline_result, challenger_result in zip(model_results[baseline_label], model_results[label]):
            if baseline_result.get("pipeline_outcome") == challenger_result.get("pipeline_outcome"):
                outcome_match += 1

            bp = baseline_result.get("parsed")
            cp = challenger_result.get("parsed")
            if isinstance(bp, dict) and isinstance(cp, dict):
                bu = bp.get("urgency_score")
                cu = cp.get("urgency_score")
                if isinstance(bu, (int, float)) and isinstance(cu, (int, float)):
                    urgency_deltas.append(abs(bu - cu))

                bpc = bp.get("pain_category")
                cpc = cp.get("pain_category")
                if bpc and cpc:
                    pain_total += 1
                    if bpc == cpc:
                        pain_agree += 1

        model_agreement: dict[str, Any] = {
            "pipeline_outcome": {
                "exact_match": outcome_match,
                "total": len(entries),
                "pct": round(outcome_match * 100 / len(entries), 1),
            }
        }
        if urgency_deltas:
            model_agreement["urgency"] = {
                "avg_delta": round(sum(urgency_deltas) / len(urgency_deltas), 2),
                "exact_match": sum(1 for delta in urgency_deltas if delta == 0),
                "within_1": sum(1 for delta in urgency_deltas if delta <= 1),
                "within_2": sum(1 for delta in urgency_deltas if delta <= 2),
                "total": len(urgency_deltas),
            }
        if pain_total:
            model_agreement["pain_category"] = {
                "exact_match": pain_agree,
                "total": pain_total,
                "pct": round(pain_agree * 100 / pain_total, 1),
            }
        agreement[label] = model_agreement

    review_comparisons = []
    for i, entry in enumerate(entries):
        comparison = {
            "review_id": entry["review_id"],
            "vendor": entry["vendor"],
            "bucket": entry["bucket"],
            "source": entry["source"],
            "content_type": entry["content_type"],
            "existing_status": entry["existing_status"],
            "existing_urgency": entry["existing_urgency"],
            "existing_pain": entry["existing_pain"],
        }
        for model in ALL_MODELS:
            label = model["label"]
            result = model_results[label][i]
            parsed = result.get("parsed") or {}
            comparison[f"{label}_outcome"] = result.get("pipeline_outcome")
            comparison[f"{label}_urgency"] = parsed.get("urgency_score") if isinstance(parsed, dict) else None
            comparison[f"{label}_pain"] = parsed.get("pain_category") if isinstance(parsed, dict) else None
            comparison[f"{label}_latency"] = result["latency_ms"]
            comparison[f"{label}_valid"] = result["scores"].get("valid_json", False)
        review_comparisons.append(comparison)

    output = {
        "sample_size": len(entries),
        "raw_bucket_sizes": {k: len(v) for k, v in buckets.items()},
        "deduped_bucket_sizes": {
            bucket: sum(1 for entry in entries if entry["bucket"] == bucket)
            for bucket in buckets
        },
        "models": {m["label"]: m["id"] for m in ALL_MODELS},
        "aggregates": aggregates,
        "agreement_vs_baseline": agreement,
        "reviews": review_comparisons,
    }

    Path(args.output).write_text(json.dumps(output, indent=2, default=str))
    logger.info("Full results: %s", args.output)

    print("\n" + "=" * 110)
    print("END-TO-END ENRICHMENT BENCHMARK")
    print("=" * 110)
    print(f"Sample: {len(entries)} unique reviews across {len(buckets)} buckets")
    print(f"Baseline: {BASELINE['label']} ({BASELINE['id']})")
    print()

    labels = [m["label"] for m in ALL_MODELS]
    col_w = 14
    header = f"  {'Metric':<35s}" + "".join(f"{label:>{col_w}s}" for label in labels)
    print(header)
    print("  " + "-" * (35 + col_w * len(labels)))

    def _fmt(agg: dict[str, Any], key: str, suffix: str = "") -> str:
        value = agg.get(key, "-")
        if value == "-":
            return "-"
        return f"{value}{suffix}"

    for key, suffix, label in [
        ("enriched_pct", "%", "Enriched"),
        ("no_signal", "", "No-signal"),
        ("invalid_extraction", "", "Invalid extraction"),
        ("triage_failed", "", "Triage failed"),
        ("api_error", "", "API errors"),
        ("valid_json_pct", "%", "Validated extraction"),
        ("schema_completeness_avg", "%", "Schema completeness"),
        ("has_urgency", "", "Has valid urgency"),
        ("valid_pain_category", "", "Valid pain_category"),
        ("competitor_grounding_pct", "%", "Competitors grounded"),
        ("quote_grounding_pct", "%", "Quotes grounded"),
        ("avg_latency_ms", "ms", "Avg latency"),
        ("p95_latency_ms", "ms", "P95 latency"),
        ("total_cost_usd", "", "Total cost ($)"),
    ]:
        row = f"  {label:<35s}"
        for model_label in labels:
            agg = aggregates.get(model_label, {})
            value = _fmt(agg, key, suffix)
            if key == "total_cost_usd":
                value = f"${agg.get(key, 0):.4f}"
            row += f"{value:>{col_w}s}"
        print(row)

    print()
    print(f"  AGREEMENT vs {BASELINE['label']} baseline:")
    for model in CHALLENGERS:
        label = model["label"]
        ag = agreement.get(label, {})
        po = ag.get("pipeline_outcome", {})
        ua = ag.get("urgency", {})
        pa = ag.get("pain_category", {})
        print(f"    {label}:")
        if po:
            print(f"      Outcome: {po['exact_match']}/{po['total']} ({po['pct']:.1f}%)")
        if ua:
            print(
                f"      Urgency: avg delta={ua['avg_delta']:.1f}, "
                f"exact={ua['exact_match']}/{ua['total']}, "
                f"within+/-1={ua['within_1']}/{ua['total']}, "
                f"within+/-2={ua['within_2']}/{ua['total']}"
            )
        if pa:
            print(f"      Pain category: {pa['exact_match']}/{pa['total']} ({pa['pct']:.1f}%)")

    print("=" * 110)
    print(f"Full results: {args.output}")

    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
