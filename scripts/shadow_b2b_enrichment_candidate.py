#!/usr/bin/env python3
"""Run a non-persisting candidate enrichment shadow analysis for selected vendors.

This script reruns hybrid enrichment Tier 2 for a candidate model against recent
enriched B2B review rows, compares the candidate output to the stored baseline,
and computes downstream lift proxies without writing to production tables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.autonomous.tasks._b2b_shared import (
    _build_company_signal_blocked_names_by_vendor,
    _extract_alternative_names,
    _company_signal_name_is_eligible,
    build_account_intelligence,
)
from atlas_brain.config import settings
from atlas_brain.services.company_normalization import normalize_company_name
from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _coerce_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _known_text(value: Any) -> bool:
    text = _text(value).lower()
    return bool(text) and text not in {"unknown", "none", "null", "n/a", "na"}


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_cfg(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        enrichment_tier1_vllm_url=settings.b2b_churn.enrichment_tier1_vllm_url,
        enrichment_tier1_model=settings.b2b_churn.enrichment_tier1_model,
        enrichment_tier1_max_tokens=settings.b2b_churn.enrichment_tier1_max_tokens,
        enrichment_tier2_model=model,
        enrichment_max_tokens=settings.b2b_churn.enrichment_max_tokens,
    )


async def _fetch_rows(
    pool,
    vendors: list[str],
    window_days: int,
    per_vendor: int,
) -> list[dict[str, Any]]:
    return await pool.fetch(
        """
        WITH ranked AS (
            SELECT
                id AS review_id,
                vendor_name,
                product_name,
                product_category,
                source,
                summary,
                review_text,
                pros,
                cons,
                reviewer_title,
                reviewer_company,
                company_size_raw,
                COALESCE(
                    reviewer_industry,
                    enrichment->'reviewer_context'->>'industry'
                ) AS reviewer_industry,
                content_type,
                rating,
                rating_max,
                raw_metadata,
                relevance_score,
                author_churn_score,
                enriched_at,
                enrichment,
                enrichment_model,
                row_number() OVER (
                    PARTITION BY vendor_name
                    ORDER BY
                        COALESCE((enrichment->>'urgency_score')::numeric, 0) DESC,
                        COALESCE(relevance_score, 0.5) DESC,
                        enriched_at DESC,
                        id
                ) AS rn
            FROM b2b_reviews
            WHERE enrichment_status = 'enriched'
              AND vendor_name = ANY($1::text[])
              AND enriched_at >= NOW() - ($2::int * INTERVAL '1 day')
        )
        SELECT *
        FROM ranked
        WHERE rn <= $3
        ORDER BY vendor_name, rn
        """,
        vendors,
        window_days,
        per_vendor,
    )


async def _rerun_candidate(row: dict[str, Any], model: str) -> dict[str, Any]:
    cfg = _candidate_cfg(model)
    tier1, tier2, resolved_model = await b2b_enrichment._enrich_hybrid(
        row,
        cfg,
        settings.b2b_churn.review_truncate_length,
    )
    if tier1 is None:
        return {"valid": False, "error": "tier1_failed", "model": resolved_model or model}
    merged = b2b_enrichment._merge_tier1_tier2(tier1, tier2)
    valid = b2b_enrichment._validate_enrichment(merged, row)
    return {
        "valid": bool(valid),
        "error": None if valid else "validation_failed",
        "model": resolved_model or model,
        "result": merged if valid else None,
    }


async def _shadow_candidate_rows(
    rows: list[dict[str, Any]],
    model: str,
    concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sem = asyncio.Semaphore(concurrency)
    candidate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    async def _one(row: dict[str, Any]) -> None:
        async with sem:
            result = await _rerun_candidate(row, model)
        if not result.get("valid"):
            failures.append(
                {
                    "review_id": str(row.get("review_id") or ""),
                    "vendor_name": row.get("vendor_name"),
                    "error": result.get("error"),
                    "model": result.get("model"),
                }
            )
            return
        candidate_rows.append({**dict(row), "enrichment": result["result"]})

    await asyncio.gather(*[_one(dict(row)) for row in rows])
    return candidate_rows, failures


def _summary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    urgencies: list[float] = []
    signal_true_count = 0
    known_role_reviews = 0
    known_timeline_reviews = 0
    known_contract_reviews = 0
    decision_maker_reviews = 0
    active_eval_reviews = 0
    intent_to_leave_reviews = 0
    feature_gap_mentions = 0
    unique_competitors: set[str] = set()
    competitor_mentions = 0
    top_pain_counter: Counter[str] = Counter()

    for row in rows:
        enrichment = _coerce_json(row.get("enrichment"))
        if not enrichment:
            continue
        urgency = enrichment.get("urgency_score")
        if isinstance(urgency, (int, float)):
            urgencies.append(float(urgency))
        churn = enrichment.get("churn_signals") or {}
        buyer = enrichment.get("buyer_authority") or {}
        reviewer_ctx = enrichment.get("reviewer_context") or {}
        timeline = enrichment.get("timeline") or {}
        contract = enrichment.get("contract_context") or {}

        signal_true_count += sum(1 for value in churn.values() if value is True)
        active_eval_reviews += 1 if churn.get("actively_evaluating") is True else 0
        intent_to_leave_reviews += 1 if churn.get("intent_to_leave") is True else 0

        role = buyer.get("role_type") or reviewer_ctx.get("role_level")
        if _known_text(role):
            known_role_reviews += 1
        if buyer.get("has_budget_authority") is True or reviewer_ctx.get("decision_maker") is True:
            decision_maker_reviews += 1
        if _known_text(timeline.get("decision_timeline")):
            known_timeline_reviews += 1
        if _known_text(contract.get("contract_value_signal")):
            known_contract_reviews += 1

        pain = _text(enrichment.get("pain_category")).lower()
        if pain:
            top_pain_counter[pain] += 1
        feature_gaps = _safe_list(enrichment.get("feature_gaps"))
        feature_gap_mentions += len(feature_gaps)

        competitors = _safe_list(enrichment.get("competitors_mentioned"))
        competitor_mentions += len(competitors)
        for item in competitors:
            if not isinstance(item, dict):
                continue
            name = _text(item.get("name"))
            if name:
                unique_competitors.add(name)

    avg_urgency = round(sum(urgencies) / len(urgencies), 2) if urgencies else 0.0
    return {
        "review_count": len(rows),
        "avg_urgency": avg_urgency,
        "signal_true_count": signal_true_count,
        "active_eval_reviews": active_eval_reviews,
        "intent_to_leave_reviews": intent_to_leave_reviews,
        "decision_maker_reviews": decision_maker_reviews,
        "known_role_reviews": known_role_reviews,
        "known_timeline_reviews": known_timeline_reviews,
        "known_contract_reviews": known_contract_reviews,
        "feature_gap_mentions": feature_gap_mentions,
        "competitor_mentions": competitor_mentions,
        "unique_competitor_count": len(unique_competitors),
        "top_pains": [
            {"pain_category": key, "count": count}
            for key, count in top_pain_counter.most_common(5)
        ],
    }


def _extract_high_intent_entries(rows: list[dict[str, Any]], urgency_threshold: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        enrichment = _coerce_json(row.get("enrichment"))
        if not enrichment:
            continue
        urgency = _float(enrichment.get("urgency_score"))
        if urgency < float(urgency_threshold):
            continue
        company_name = _text(row.get("reviewer_company"))
        if not company_name:
            continue
        competitors = _safe_list(enrichment.get("competitors_mentioned"))
        blocked_names = {
            normalize_company_name(name)
            for name in _extract_alternative_names(competitors)
            if normalize_company_name(name)
        }
        if not _company_signal_name_is_eligible(
            company_name,
            current_vendor=_text(row.get("vendor_name")),
            blocked_names=blocked_names,
        ):
            continue

        reviewer_ctx = enrichment.get("reviewer_context") or {}
        buyer = enrichment.get("buyer_authority") or {}
        budget = enrichment.get("budget_signals") or {}
        timeline = enrichment.get("timeline") or {}
        quotes = _safe_list(enrichment.get("quotable_phrases"))

        results.append(
            {
                "company": company_name,
                "vendor": _text(row.get("vendor_name")),
                "category": _text(row.get("product_category")),
                "title": _text(row.get("reviewer_title")),
                "company_size": _text(row.get("company_size_raw")),
                "industry": _text(row.get("reviewer_industry")) or _text(reviewer_ctx.get("industry")),
                "role_level": _text(reviewer_ctx.get("role_level")) or _text(buyer.get("role_type")),
                "decision_maker": bool(
                    reviewer_ctx.get("decision_maker") is True
                    or buyer.get("has_budget_authority") is True
                ),
                "urgency": urgency,
                "pain": _text(enrichment.get("pain_category")),
                "alternatives": competitors,
                "quotes": quotes,
                "review_id": str(row.get("review_id") or "") or None,
                "source": _text(row.get("source")),
                "seat_count": _int(budget.get("seat_count")),
                "contract_end": _text(timeline.get("contract_end")) or None,
                "buying_stage": _text(buyer.get("buying_stage")) or None,
                "relevance_score": _float(row.get("relevance_score")),
                "author_churn_score": _float(row.get("author_churn_score")),
            }
        )
    return results


def _account_proxy(vendor: str, rows: list[dict[str, Any]], urgency_threshold: int) -> dict[str, Any]:
    entries = _extract_high_intent_entries(rows, urgency_threshold)
    blocked = _build_company_signal_blocked_names_by_vendor(
        [vendor],
        high_intent_entries=entries,
        integration_lookup={},
    )
    account = build_account_intelligence(
        vendor_name=vendor,
        high_intent_entries=entries,
        blocked_names=blocked.get(vendor),
        analysis_window_days=int(settings.b2b_churn.intelligence_window_days),
    )
    accounts = account.get("accounts") or []
    return {
        "summary": account.get("summary") or {},
        "top_accounts": [
            {
                "company_name": acct.get("company_name"),
                "urgency_score": acct.get("urgency_score"),
                "buying_stage": acct.get("buying_stage"),
                "buyer_role": acct.get("buyer_role"),
            }
            for acct in accounts[:5]
        ],
    }


def _delta(before: dict[str, Any], after: dict[str, Any], key: str) -> Any:
    a = after.get(key)
    b = before.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(a - b, 2)
    return None


def _vendor_report(
    vendor: str,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    urgency_threshold: int,
) -> dict[str, Any]:
    baseline_summary = _summary_metrics(baseline_rows)
    candidate_summary = _summary_metrics(candidate_rows)
    baseline_account = _account_proxy(vendor, baseline_rows, urgency_threshold)
    candidate_account = _account_proxy(vendor, candidate_rows, urgency_threshold)
    return {
        "vendor_name": vendor,
        "baseline": {
            "summary": baseline_summary,
            "account_proxy": baseline_account,
        },
        "candidate": {
            "summary": candidate_summary,
            "account_proxy": candidate_account,
        },
        "lift": {
            "avg_urgency_delta": _delta(baseline_summary, candidate_summary, "avg_urgency"),
            "active_eval_review_delta": _delta(baseline_summary, candidate_summary, "active_eval_reviews"),
            "intent_to_leave_review_delta": _delta(baseline_summary, candidate_summary, "intent_to_leave_reviews"),
            "known_role_review_delta": _delta(baseline_summary, candidate_summary, "known_role_reviews"),
            "known_timeline_review_delta": _delta(baseline_summary, candidate_summary, "known_timeline_reviews"),
            "known_contract_review_delta": _delta(baseline_summary, candidate_summary, "known_contract_reviews"),
            "feature_gap_delta": _delta(baseline_summary, candidate_summary, "feature_gap_mentions"),
            "competitor_delta": _delta(baseline_summary, candidate_summary, "unique_competitor_count"),
            "named_account_delta": _delta(
                baseline_account["summary"],
                candidate_account["summary"],
                "total_accounts",
            ),
            "high_intent_account_delta": _delta(
                baseline_account["summary"],
                candidate_account["summary"],
                "high_intent_count",
            ),
            "active_eval_account_delta": _delta(
                baseline_account["summary"],
                candidate_account["summary"],
                "active_eval_signal_count",
            ),
        },
    }


async def _run(args: argparse.Namespace) -> int:
    await init_database()
    try:
        pool = get_db_pool()
        rows = [
            dict(row)
            for row in await _fetch_rows(
                pool,
                args.vendors,
                args.window_days,
                args.per_vendor,
            )
        ]
        rows_by_vendor: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            rows_by_vendor.setdefault(_text(row.get("vendor_name")), []).append(row)

        candidate_rows, failures = await _shadow_candidate_rows(
            rows,
            args.model,
            args.concurrency,
        )
        candidate_by_review = {
            str(row.get("review_id") or ""): row
            for row in candidate_rows
        }

        vendor_reports: list[dict[str, Any]] = []
        for vendor in args.vendors:
            baseline_vendor_rows = rows_by_vendor.get(vendor, [])
            candidate_vendor_rows = [
                candidate_by_review[str(row.get("review_id") or "")]
                for row in baseline_vendor_rows
                if str(row.get("review_id") or "") in candidate_by_review
            ]
            vendor_reports.append(
                _vendor_report(
                    vendor,
                    baseline_vendor_rows,
                    candidate_vendor_rows,
                    args.urgency_threshold,
                )
            )

        report = {
            "candidate_model": args.model,
            "vendors": args.vendors,
            "window_days": args.window_days,
            "per_vendor": args.per_vendor,
            "urgency_threshold": args.urgency_threshold,
            "row_counts": {
                "baseline_rows": len(rows),
                "candidate_rows": len(candidate_rows),
                "failed_rows": len(failures),
            },
            "vendor_reports": vendor_reports,
            "failures": failures[:50],
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["row_counts"], indent=2))
        print(f"Saved report to {args.output}")
        return 0
    finally:
        await close_database()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Candidate Tier 2 OpenRouter model id")
    parser.add_argument("--vendors", nargs="+", required=True, help="Vendors to shadow")
    parser.add_argument(
        "--window-days",
        type=int,
        default=int(settings.b2b_churn.intelligence_window_days),
        help="How many recent days of enriched reviews to shadow",
    )
    parser.add_argument(
        "--per-vendor",
        type=int,
        default=30,
        help="How many recent/high-urgency rows to sample per vendor",
    )
    parser.add_argument(
        "--urgency-threshold",
        type=int,
        default=int(settings.b2b_churn.high_churn_urgency_threshold),
        help="Urgency threshold for named-account pressure proxies",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent candidate reruns")
    parser.add_argument(
        "--output",
        default="/tmp/b2b_enrichment_shadow_report.json",
        help="Where to write the shadow analysis JSON report",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
