#!/usr/bin/env python3
"""Build a row-level adjudication pack from multiple Tier 2 comparison reports."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

from atlas_brain.storage.database import close_database, get_db_pool, init_database


def _text(value: Any) -> str:
    return str(value or "").strip()


def _known(value: Any) -> str:
    text = _text(value)
    return text if text and text.lower() not in {"unknown", "none", "null", "n/a", "na"} else ""


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    entries = report.get("entries")
    if isinstance(entries, list) and entries:
        return entries
    rows: list[dict[str, Any]] = []
    for vendor in report.get("vendor_results") or []:
        for item in vendor.get("top_diffs") or []:
            if isinstance(item, dict):
                rows.append(item)
    return rows


async def _fetch_missing_context(review_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not review_ids:
        return {}
    pool = get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            id,
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
            reviewer_industry,
            content_type
        FROM b2b_reviews
        WHERE id = ANY($1::uuid[])
        """,
        review_ids,
    )
    context: dict[str, dict[str, Any]] = {}
    for row in rows:
        context[str(row["id"])] = {
            "review_id": str(row["id"]),
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
    return context


def _competitor_similarity(metrics_by_model: dict[str, dict[str, Any]]) -> tuple[float, list[str]]:
    sets: list[set[str]] = []
    for metrics in metrics_by_model.values():
        names = metrics.get("competitor_names") or []
        sets.append({str(name).strip() for name in names if str(name).strip()})
    if len(sets) < 2:
        return 1.0, []
    similarities: list[float] = []
    reasons: list[str] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            left = sets[i]
            right = sets[j]
            union = left | right
            if not union:
                similarities.append(1.0)
                continue
            score = len(left & right) / len(union)
            similarities.append(score)
            if score < 1.0:
                reasons.append(
                    f"competitor overlap {score:.2f}: {sorted(left)} vs {sorted(right)}"
                )
    return (statistics.fmean(similarities) if similarities else 1.0), reasons[:3]


def _dispute_score(case: dict[str, Any]) -> dict[str, Any]:
    metrics_by_model = dict(case["models"])
    score = 0.0
    reasons: list[str] = []

    def _values(field: str) -> list[Any]:
        return [metrics.get(field) for metrics in metrics_by_model.values()]

    def _distinct(field: str, *, known_only: bool = False) -> list[str]:
        values: set[str] = set()
        for metrics in metrics_by_model.values():
            raw = metrics.get(field)
            if known_only:
                text = _known(raw)
                if text:
                    values.add(text)
            else:
                values.add(_text(raw))
        return sorted(v for v in values if v or not known_only)

    urgency_values = [value for value in (_float(v) for v in _values("urgency_score")) if value is not None]
    if len(urgency_values) >= 2:
        spread = max(urgency_values) - min(urgency_values)
        if spread >= 2:
            score += spread * 1.5
            reasons.append(f"urgency spread {spread:.1f}")

    signal_values = [int(v or 0) for v in _values("signal_true_count")]
    signal_spread = max(signal_values) - min(signal_values) if signal_values else 0
    if signal_spread > 0:
        score += signal_spread * 2.0
        reasons.append(f"signal count spread {signal_spread}")

    for field, weight, label in (
        ("intent_to_leave", 5.0, "intent_to_leave disagreement"),
        ("actively_evaluating", 5.0, "actively_evaluating disagreement"),
    ):
        values = {bool(v) for v in _values(field)}
        if len(values) > 1:
            score += weight
            reasons.append(label)

    for field, weight, label in (
        ("buyer_role_type", 3.0, "role disagreement"),
        ("buyer_buying_stage", 2.0, "buying stage disagreement"),
        ("decision_timeline", 3.0, "timeline disagreement"),
        ("contract_value_signal", 3.0, "contract disagreement"),
        ("pain_category", 2.0, "pain category disagreement"),
    ):
        values = _distinct(field, known_only=field in {"decision_timeline", "contract_value_signal"})
        if len(values) > 1:
            score += weight
            reasons.append(f"{label}: {values}")

    competitor_similarity, competitor_reasons = _competitor_similarity(metrics_by_model)
    if competitor_similarity < 1.0:
        score += (1.0 - competitor_similarity) * 4.0
        reasons.extend(competitor_reasons)

    feature_values = [int(v or 0) for v in _values("feature_gap_count")]
    feature_spread = max(feature_values) - min(feature_values) if feature_values else 0
    if feature_spread >= 2:
        score += feature_spread * 0.75
        reasons.append(f"feature gap spread {feature_spread}")

    return {
        "score": round(score, 2),
        "reasons": reasons[:10],
    }


async def _run(args: argparse.Namespace) -> int:
    reports = []
    for report_path in args.reports:
        data = json.loads(Path(report_path).read_text())
        reports.append((Path(report_path), data))

    cases: dict[str, dict[str, Any]] = {}
    for report_path, report in reports:
        candidate_model = report.get("candidate_model") or report_path.stem
        baseline_model = report.get("baseline_model") or ""
        for entry in _entry_rows(report):
            if not isinstance(entry, dict):
                continue
            review_id = _text(entry.get("review_id"))
            if not review_id:
                continue
            case = cases.setdefault(
                review_id,
                {
                    "review_id": review_id,
                    "vendor_name": entry.get("vendor_name"),
                    "baseline_model": baseline_model,
                    "baseline": entry.get("baseline") or {},
                    "baseline_result": entry.get("baseline_result") or {},
                    "context": entry.get("context"),
                    "models": {},
                },
            )
            if not case.get("context") and entry.get("context"):
                case["context"] = entry["context"]
            if not case.get("baseline_result") and entry.get("baseline_result"):
                case["baseline_result"] = entry["baseline_result"]
            if not case.get("baseline"):
                case["baseline"] = entry.get("baseline") or {}
            case["models"][candidate_model] = {
                **(entry.get("candidate") or {}),
                "_candidate_model": entry.get("candidate_model") or candidate_model,
                "_diff": entry.get("diff") or {},
                "_candidate_result": entry.get("candidate_result") or {},
                "_report": str(report_path),
            }

    missing_context_ids = [
        review_id
        for review_id, case in cases.items()
        if not isinstance(case.get("context"), dict)
    ]

    await init_database()
    try:
        fetched_context = await _fetch_missing_context(missing_context_ids)
    finally:
        await close_database()

    dispute_cases: list[dict[str, Any]] = []
    for review_id, case in cases.items():
        if len(case["models"]) < max(2, args.min_models):
            continue
        if not isinstance(case.get("context"), dict):
            case["context"] = fetched_context.get(review_id) or {}
        combined_models = {case["baseline_model"]: case["baseline"], **case["models"]}
        disagreement = _dispute_score({"models": combined_models})
        if disagreement["score"] < args.min_score:
            continue
        dispute_cases.append(
            {
                "review_id": review_id,
                "vendor_name": case.get("vendor_name") or case.get("context", {}).get("vendor_name"),
                "baseline_model": case.get("baseline_model"),
                "disagreement": disagreement,
                "context": case.get("context") or {},
                "baseline": case.get("baseline") or {},
                "baseline_result": case.get("baseline_result") or {},
                "candidates": case["models"],
            }
        )

    dispute_cases.sort(
        key=lambda item: (
            -float(item["disagreement"]["score"]),
            _text(item.get("vendor_name")),
            _text(item.get("review_id")),
        )
    )

    report = {
        "source_reports": [str(path) for path, _ in reports],
        "case_count": len(dispute_cases),
        "selected_count": min(args.top, len(dispute_cases)),
        "cases": dispute_cases[: args.top],
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"case_count": report["case_count"], "selected_count": report["selected_count"]}, indent=2))
    print(f"Saved dispute pack to {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, help="Comparison report JSON files")
    parser.add_argument("--top", type=int, default=25, help="How many disputed cases to keep")
    parser.add_argument("--min-score", type=float, default=6.0, help="Minimum disagreement score to keep")
    parser.add_argument("--min-models", type=int, default=2, help="Minimum candidate models that must cover a row")
    parser.add_argument(
        "--output",
        default="/tmp/b2b_enrichment_dispute_pack.json",
        help="Where to write the adjudication pack JSON",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
