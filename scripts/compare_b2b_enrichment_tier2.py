#!/usr/bin/env python3
"""Compare B2B hybrid enrichment Tier 2 models on the same stored review rows.

This script keeps Tier 1 fixed on the production local vLLM path and reruns only
the hybrid Tier 2 interpretation leg against a candidate OpenRouter model. It
compares the candidate output against the stored production baseline rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.config import settings
from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _baseline_model_id() -> str:
    tier2 = (
        str(settings.b2b_churn.enrichment_tier2_model or "").strip()
        or str(settings.b2b_churn.enrichment_openrouter_model or "").strip()
    )
    return f"hybrid:{settings.b2b_churn.enrichment_tier1_model}+{tier2}"


def _coerce_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _bool_count(payload: dict[str, Any], key: str) -> int:
    section = payload.get(key) or {}
    if not isinstance(section, dict):
        return 0
    return sum(1 for value in section.values() if value is True)


def _role_rank(role_type: str) -> int:
    ranks = {
        "unknown": 0,
        "end_user": 1,
        "champion": 2,
        "evaluator": 3,
        "economic_buyer": 4,
    }
    return ranks.get(str(role_type or "unknown").strip().lower(), 0)


def _specificity_rank(value: str) -> int:
    text = str(value or "").strip().lower()
    return 0 if not text or text == "unknown" else 1


def _extract_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    churn_signals = payload.get("churn_signals") or {}
    buyer_authority = payload.get("buyer_authority") or {}
    timeline = payload.get("timeline") or {}
    contract = payload.get("contract_context") or {}
    competitors = payload.get("competitors_mentioned") or []
    feature_gaps = payload.get("feature_gaps") or []
    positive_aspects = payload.get("positive_aspects") or []

    competitor_names = sorted(
        comp.get("name", "").strip()
        for comp in competitors
        if isinstance(comp, dict) and comp.get("name")
    )

    return {
        "urgency_score": payload.get("urgency_score"),
        "pain_category": payload.get("pain_category"),
        "pain_category_count": len(payload.get("pain_categories") or []),
        "signal_true_count": _bool_count(payload, "churn_signals"),
        "intent_to_leave": bool(churn_signals.get("intent_to_leave")),
        "actively_evaluating": bool(churn_signals.get("actively_evaluating")),
        "competitor_count": len(competitor_names),
        "competitor_names": competitor_names,
        "feature_gap_count": len(feature_gaps),
        "positive_aspect_count": len(positive_aspects),
        "buyer_role_type": buyer_authority.get("role_type", "unknown"),
        "buyer_buying_stage": buyer_authority.get("buying_stage", "unknown"),
        "has_budget_authority": buyer_authority.get("has_budget_authority"),
        "decision_timeline": timeline.get("decision_timeline", "unknown"),
        "contract_value_signal": contract.get("contract_value_signal", "unknown"),
        "sentiment_direction": (payload.get("sentiment_trajectory") or {}).get("direction", "unknown"),
    }


def _row_context(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": str(row.get("id") or ""),
        "vendor_name": row.get("vendor_name"),
        "product_name": row.get("product_name"),
        "product_category": row.get("product_category"),
        "source": row.get("source"),
        "summary": row.get("summary") or "",
        "review_text": row.get("review_text") or "",
        "pros": row.get("pros") or "",
        "cons": row.get("cons") or "",
        "reviewer_title": row.get("reviewer_title") or "",
        "reviewer_company": row.get("reviewer_company") or "",
        "company_size_raw": row.get("company_size_raw") or "",
        "reviewer_industry": row.get("reviewer_industry") or "",
        "content_type": row.get("content_type") or "",
    }


def _compare_metrics(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    changes = []
    for key in (
        "pain_category",
        "buyer_role_type",
        "buyer_buying_stage",
        "decision_timeline",
        "contract_value_signal",
        "sentiment_direction",
        "competitor_names",
    ):
        if baseline.get(key) != candidate.get(key):
            changes.append(key)

    for key in ("urgency_score", "competitor_count", "feature_gap_count", "signal_true_count"):
        if baseline.get(key) != candidate.get(key):
            changes.append(key)

    role_gain = _role_rank(candidate.get("buyer_role_type", "unknown")) - _role_rank(
        baseline.get("buyer_role_type", "unknown")
    )
    timeline_gain = _specificity_rank(candidate.get("decision_timeline", "unknown")) - _specificity_rank(
        baseline.get("decision_timeline", "unknown")
    )
    contract_gain = _specificity_rank(candidate.get("contract_value_signal", "unknown")) - _specificity_rank(
        baseline.get("contract_value_signal", "unknown")
    )

    return {
        "changed_fields": changes,
        "urgency_delta": (candidate.get("urgency_score") or 0) - (baseline.get("urgency_score") or 0),
        "signal_true_delta": candidate.get("signal_true_count", 0) - baseline.get("signal_true_count", 0),
        "competitor_delta": candidate.get("competitor_count", 0) - baseline.get("competitor_count", 0),
        "feature_gap_delta": candidate.get("feature_gap_count", 0) - baseline.get("feature_gap_count", 0),
        "role_specificity_delta": role_gain,
        "timeline_specificity_delta": timeline_gain,
        "contract_specificity_delta": contract_gain,
        "stronger_signal": (
            (candidate.get("signal_true_count", 0), role_gain, timeline_gain, contract_gain, candidate.get("urgency_score") or 0)
            >
            (baseline.get("signal_true_count", 0), 0, 0, 0, baseline.get("urgency_score") or 0)
        ),
    }


def _vendor_rollup(entries: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(entries)
    if not count:
        return {}

    def _avg(key: str, side: str) -> float:
        values = [entry[side][key] for entry in entries if isinstance(entry[side].get(key), (int, float))]
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "sample_size": count,
        "baseline_avg_urgency": _avg("urgency_score", "baseline"),
        "candidate_avg_urgency": _avg("urgency_score", "candidate"),
        "rows_changed": sum(1 for entry in entries if entry["diff"]["changed_fields"]),
        "stronger_signal_rows": sum(1 for entry in entries if entry["diff"]["stronger_signal"]),
        "role_specificity_wins": sum(1 for entry in entries if entry["diff"]["role_specificity_delta"] > 0),
        "timeline_specificity_wins": sum(1 for entry in entries if entry["diff"]["timeline_specificity_delta"] > 0),
        "contract_specificity_wins": sum(1 for entry in entries if entry["diff"]["contract_specificity_delta"] > 0),
        "competitor_delta_total": sum(entry["diff"]["competitor_delta"] for entry in entries),
        "feature_gap_delta_total": sum(entry["diff"]["feature_gap_delta"] for entry in entries),
    }


async def _pick_vendors(pool, baseline_model: str, vendor_limit: int, per_vendor: int) -> list[str]:
    rows = await pool.fetch(
        """
        SELECT vendor_name, count(*) AS review_count
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND enrichment_model = $1
        GROUP BY 1
        HAVING count(*) >= $2
        ORDER BY review_count DESC, vendor_name
        LIMIT $3
        """,
        baseline_model,
        per_vendor,
        vendor_limit,
    )
    return [row["vendor_name"] for row in rows]


async def _fetch_rows(pool, baseline_model: str, vendors: list[str], per_vendor: int) -> list[dict[str, Any]]:
    return await pool.fetch(
        """
        WITH ranked AS (
            SELECT id, vendor_name, product_name, product_category, source, raw_metadata,
                   rating, rating_max, summary, review_text, pros, cons,
                   reviewer_title, reviewer_company, company_size_raw, reviewer_industry,
                   content_type, enrichment, enrichment_model, enriched_at,
                   row_number() OVER (
                       PARTITION BY vendor_name
                       ORDER BY COALESCE((enrichment->>'urgency_score')::numeric, 0) DESC,
                                enriched_at DESC,
                                id
                   ) AS rn
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND enrichment_model = $1
              AND vendor_name = ANY($2::text[])
        )
        SELECT *
        FROM ranked
        WHERE rn <= $3
        ORDER BY vendor_name, rn
        """,
        baseline_model,
        vendors,
        per_vendor,
    )


async def _rerun_candidate(row: dict[str, Any], candidate_model: str) -> dict[str, Any]:
    cfg = SimpleNamespace(
        enrichment_tier1_vllm_url=settings.b2b_churn.enrichment_tier1_vllm_url,
        enrichment_tier1_model=settings.b2b_churn.enrichment_tier1_model,
        enrichment_tier1_max_tokens=settings.b2b_churn.enrichment_tier1_max_tokens,
        enrichment_tier2_model=candidate_model,
        enrichment_max_tokens=settings.b2b_churn.enrichment_max_tokens,
    )

    started = time.monotonic()
    tier1, tier2, tier2_model_id = await b2b_enrichment._enrich_hybrid(
        row,
        cfg,
        settings.b2b_churn.review_truncate_length,
    )
    latency_ms = round((time.monotonic() - started) * 1000)

    if tier1 is None:
        return {
            "valid": False,
            "latency_ms": latency_ms,
            "candidate_model": tier2_model_id or candidate_model,
            "error": "tier1_failed",
            "result": None,
        }

    merged = b2b_enrichment._merge_tier1_tier2(tier1, tier2)
    valid = b2b_enrichment._validate_enrichment(merged, row)
    return {
        "valid": bool(valid),
        "latency_ms": latency_ms,
        "candidate_model": tier2_model_id or candidate_model,
        "error": None if valid else "validation_failed",
        "result": merged,
    }


async def _run_rows(
    rows: list[dict[str, Any]],
    candidate_model: str,
    concurrency: int,
    *,
    include_context: bool = False,
    include_results: bool = False,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            candidate = await _rerun_candidate(row, candidate_model)
            baseline = _coerce_json(row.get("enrichment"))
            baseline_metrics = _extract_metrics(baseline)
            candidate_metrics = _extract_metrics(candidate["result"] or {})
            entry = {
                "review_id": str(row["id"]),
                "vendor_name": row["vendor_name"],
                "summary": row.get("summary") or "",
                "baseline_model": row.get("enrichment_model"),
                "candidate_model": candidate["candidate_model"],
                "valid": candidate["valid"],
                "latency_ms": candidate["latency_ms"],
                "error": candidate["error"],
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "diff": _compare_metrics(baseline_metrics, candidate_metrics),
            }
            if include_context:
                entry["context"] = _row_context(row)
            if include_results:
                entry["baseline_result"] = baseline
                entry["candidate_result"] = candidate["result"] or {}
            return entry

    return await asyncio.gather(*[_bounded(dict(row)) for row in rows])


def _build_report(
    entries: list[dict[str, Any]],
    baseline_model: str,
    candidate_model: str,
    vendors: list[str],
    *,
    include_rows: bool = False,
    top_diff_limit: int = 3,
) -> dict[str, Any]:
    valid_entries = [entry for entry in entries if entry["valid"]]
    vendor_names = sorted({entry["vendor_name"] for entry in entries})

    vendor_results = []
    for vendor_name in vendor_names:
        vendor_entries = [entry for entry in valid_entries if entry["vendor_name"] == vendor_name]
        vendor_entries.sort(key=lambda entry: len(entry["diff"]["changed_fields"]), reverse=True)
        vendor_results.append({
            "vendor_name": vendor_name,
            "rollup": _vendor_rollup(vendor_entries),
            "top_diffs": vendor_entries[:top_diff_limit],
        })

    overall = _vendor_rollup(valid_entries)
    overall["total_rows"] = len(entries)
    overall["valid_rows"] = len(valid_entries)
    overall["invalid_rows"] = len(entries) - len(valid_entries)
    overall["avg_latency_ms"] = round(
        sum(entry["latency_ms"] for entry in entries) / len(entries),
        1,
    ) if entries else 0.0

    report = {
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "vendors": vendors,
        "overall": overall,
        "vendor_results": vendor_results,
    }
    if include_rows:
        report["entries"] = entries
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("\nOverall")
    print(json.dumps(report["overall"], indent=2))
    print("\nPer vendor")
    for vendor in report["vendor_results"]:
        print(f"- {vendor['vendor_name']}: {json.dumps(vendor['rollup'], sort_keys=True)}")


async def main_async(args: argparse.Namespace) -> int:
    baseline_model = args.baseline_model or _baseline_model_id()
    await init_database()
    pool = get_db_pool()
    try:
        vendors = args.vendors
        if not vendors:
            vendors = await _pick_vendors(pool, baseline_model, args.vendor_limit, args.per_vendor)
        rows = await _fetch_rows(pool, baseline_model, vendors, args.per_vendor)
        if not rows:
            print("No eligible rows found for comparison.")
            return 1

        entries = await _run_rows(
            rows,
            args.model,
            args.concurrency,
            include_context=args.include_context,
            include_results=args.include_results,
        )
        report = _build_report(
            entries,
            baseline_model,
            args.model,
            vendors,
            include_rows=args.include_rows,
            top_diff_limit=args.top_diff_limit,
        )
        _print_summary(report)

        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {output_path}")
        return 0
    finally:
        await close_database()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Candidate Tier 2 OpenRouter model id")
    parser.add_argument(
        "--vendors",
        nargs="*",
        default=[],
        help="Explicit vendor names. If omitted, picks the top baseline vendors automatically.",
    )
    parser.add_argument("--vendor-limit", type=int, default=8, help="Auto-pick top N vendors")
    parser.add_argument("--per-vendor", type=int, default=10, help="Rows per vendor to compare")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent candidate reruns")
    parser.add_argument("--baseline-model", default="", help="Override the stored baseline enrichment_model filter")
    parser.add_argument(
        "--include-rows",
        action="store_true",
        help="Include all row-level comparison entries in the saved report",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include raw review context in each saved row entry",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include full baseline and candidate enrichment payloads in each saved row entry",
    )
    parser.add_argument(
        "--top-diff-limit",
        type=int,
        default=3,
        help="How many top diffs to keep per vendor in the summary section",
    )
    parser.add_argument(
        "--output",
        default="/tmp/b2b_enrichment_tier2_comparison.json",
        help="Where to write the JSON report",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
