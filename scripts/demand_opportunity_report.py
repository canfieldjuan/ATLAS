#!/usr/bin/env python3
"""On-demand market/product demand-opportunity report from enriched reviews.

Branches off the churn pipeline right after enrichment (stage 2): reads the
existing enriched `b2b_reviews` for one product_category, joins each to its
primary vendor, and ranks product-opportunity themes from the structured
enrichment multi-fields -- NOT the coarse single `pain_category` (which is
dominated by the non-actionable "overall_dissatisfaction" catch-all).

Opportunity heuristic per pain theme:
    score = review_count * mean_urgency * cross_vendor_breadth
where cross_vendor_breadth = distinct vendors showing the theme / distinct
vendors in the category. A widespread, intense, every-vendor-unsolved pain is
a market gap; a pain isolated to one weak vendor is just that vendor's problem.

Read-only, on-demand, human-in-the-loop -- not a cron task, no DB writes.

Usage:
    python demand_opportunity_report.py --category=CRM
    python demand_opportunity_report.py --category="Project Management" --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.storage.database import close_database, get_db_pool, init_database

# "overall_dissatisfaction" is a catch-all, not a buildable product theme; it is
# reported as a baseline but excluded from the ranked opportunity table.
_NON_ACTIONABLE = {"overall_dissatisfaction"}

# Relevance filter for common-word-vendor contamination (see memory:
# common-word-vendor-contamination). Vendors whose names are also common words
# -- Copper, Close -- keyword-match unrelated reviews in the corpus
# (copper-the-metal / crafting games, account "closing", audio gear). These
# high-confidence markers never appear in genuine B2B-SaaS reviews, so a review
# carrying any of them is treated as off-topic and excluded from the report
# (it does not count toward themes or appear as evidence). Deliberately
# specific to avoid false-dropping legitimate reviews (e.g. a "Silver plan"
# tier or a "durable workflow" survive; "30 stone axes" and "150 durability"
# do not). The real fix is upstream vendor-name disambiguation; this keeps the
# report honest until then.
_OFFTOPIC_RE = re.compile(
    "|".join((
        r"\bstone axes?\b", r"\bcopper axe", r"\bpickaxe", r"\brespawn", r"\bsmelt",
        r"\d+\s+silver\b", r"\bsilver per day", r"\d+\s+durability", r"durability compared",
        r"\bcraft(?:ing|ed)?\s+\d", r"\bore\s+(?:vein|deposit|node)", r"\bblacksmith",
        r"\bsoldering\b", r"\bgauge wire", r"\bspeaker cable", r"\bsilver[- ]plated",
        r"\bheadphones?\b", r"\bamplifier", r"\baudio equipment", r"\baluminum\b",
    )),
    re.IGNORECASE,
)


def _review_is_offtopic(review: dict[str, Any]) -> bool:
    """True if any of the review's evidence text trips a high-confidence
    off-topic marker -- i.e. it is keyword-match contamination, not a real
    review of the product category."""
    for field in ("quotable_phrases", "pricing_phrases", "feature_gaps", "specific_complaints"):
        for s in _as_list(review.get(field)):
            if isinstance(s, str) and _OFFTOPIC_RE.search(s):
                return True
    return False


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


# ---------------------------------------------------------------------------
# Pure aggregation (no DB) -- unit-testable
# ---------------------------------------------------------------------------


def aggregate(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate enriched review dicts into demand-opportunity report data.

    Each review dict: vendor (str|None), urgency (float|None), pain_categories
    (list[{category,severity}]), feature_gaps (list[str]), specific_complaints
    (list[str]), competitors (list[{name,context}]), pricing_phrases
    (list[str]), positive_aspects (list[str]), quotable_phrases (list[str]),
    price_increase (bool|None).
    """
    # Relevance filter: drop common-word-vendor contamination before any
    # counting, so off-topic reviews neither inflate themes nor surface as
    # evidence. Track what was dropped for transparency.
    filtered_vendors: dict[str, int] = defaultdict(int)
    clean: list[dict[str, Any]] = []
    for r in reviews:
        if _review_is_offtopic(r):
            filtered_vendors[r.get("vendor") or "unknown"] += 1
        else:
            clean.append(r)
    filtered_offtopic = len(reviews) - len(clean)
    reviews = clean

    all_vendors = {r["vendor"] for r in reviews if r.get("vendor")}
    total_vendors = len(all_vendors) or 1

    # Pain theme -> {reviews, urgency_sum, urgency_n, vendors, quotes}
    pain: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"reviews": 0, "urgency_sum": 0.0, "urgency_n": 0, "vendors": set(), "quotes": []}
    )
    baseline_count = 0
    feature_gaps: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "verbatim": None})
    positive_aspects: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "verbatim": None})
    # Keyed by normalized name (case-folded, trailing " CRM" stripped) so
    # variants like "HubSpot" / "Hubspot" / "HubSpot CRM" tally as one; the
    # most frequent raw spelling is used for display.
    competitors: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "contexts": defaultdict(int), "names": defaultdict(int)}
    )
    pricing_phrases: list[str] = []
    price_increase_count = 0

    for r in reviews:
        vendor = r.get("vendor")
        urg = r.get("urgency")
        cats = {str(pc.get("category")) for pc in _as_list(r.get("pain_categories")) if isinstance(pc, dict) and pc.get("category")}
        for cat in cats:
            if cat in _NON_ACTIONABLE:
                baseline_count += 1
                continue
            slot = pain[cat]
            slot["reviews"] += 1
            if urg is not None:
                slot["urgency_sum"] += float(urg)
                slot["urgency_n"] += 1
            if vendor:
                slot["vendors"].add(vendor)
            if len(slot["quotes"]) < 3:
                q = next((str(x) for x in _as_list(r.get("quotable_phrases")) if x), None)
                if q:
                    slot["quotes"].append({"text": q[:280], "vendor": vendor, "source": r.get("source")})

        for gap in _as_list(r.get("feature_gaps")):
            key = _norm(gap)
            if not key:
                continue
            feature_gaps[key]["count"] += 1
            if feature_gaps[key]["verbatim"] is None:
                feature_gaps[key]["verbatim"] = str(gap).strip()

        for pos in _as_list(r.get("positive_aspects")):
            key = _norm(pos)
            if not key:
                continue
            positive_aspects[key]["count"] += 1
            if positive_aspects[key]["verbatim"] is None:
                positive_aspects[key]["verbatim"] = str(pos).strip()

        for comp in _as_list(r.get("competitors")):
            if isinstance(comp, dict) and comp.get("name"):
                raw = str(comp["name"]).strip()
                key = re.sub(r"\s+crm$", "", raw.lower()).strip()
                if not key:
                    continue
                slot = competitors[key]
                slot["count"] += 1
                slot["names"][raw] += 1
                slot["contexts"][str(comp.get("context") or "unspecified")] += 1

        for phrase in _as_list(r.get("pricing_phrases")):
            if phrase:
                pricing_phrases.append(str(phrase).strip())
        if r.get("price_increase"):
            price_increase_count += 1

    themes = []
    for cat, s in pain.items():
        mean_urg = (s["urgency_sum"] / s["urgency_n"]) if s["urgency_n"] else 0.0
        breadth = len(s["vendors"]) / total_vendors
        score = s["reviews"] * mean_urg * breadth
        themes.append({
            "theme": cat,
            "reviews": s["reviews"],
            "mean_urgency": round(mean_urg, 2),
            "vendor_breadth": f"{len(s['vendors'])}/{total_vendors}",
            "breadth_pct": round(breadth, 2),
            "opportunity_score": round(score, 1),
            "quotes": s["quotes"],
        })
    themes.sort(key=lambda t: t["opportunity_score"], reverse=True)

    gaps = sorted(
        ({"gap": v["verbatim"], "count": v["count"]} for v in feature_gaps.values()),
        key=lambda g: g["count"], reverse=True,
    )
    works = sorted(
        ({"aspect": v["verbatim"], "count": v["count"]} for v in positive_aspects.values()),
        key=lambda w: w["count"], reverse=True,
    )
    comps = sorted(
        (
            {
                "name": max(v["names"].items(), key=lambda x: x[1])[0],
                "count": v["count"],
                "contexts": dict(v["contexts"]),
            }
            for v in competitors.values()
        ),
        key=lambda c: c["count"], reverse=True,
    )

    return {
        "total_reviews": len(reviews),
        "total_vendors": total_vendors,
        "vendors": sorted(all_vendors),
        "filtered_offtopic": filtered_offtopic,
        "filtered_vendors": dict(sorted(filtered_vendors.items(), key=lambda x: -x[1])),
        "baseline_overall_dissatisfaction": baseline_count,
        "themes": themes,
        "feature_gaps": gaps,
        "works_well": works,
        "competitors": comps,
        "pricing": {
            "price_increase_mentions": price_increase_count,
            "sample_phrases": pricing_phrases[:8],
        },
    }


def render_markdown(category: str, data: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"# Demand-Opportunity Report: {category}")
    out.append("")
    out.append(
        f"{data['total_reviews']} enriched reviews across {data['total_vendors']} "
        f"primary vendors ({', '.join(data['vendors'])})."
    )
    if data.get("filtered_offtopic"):
        fv = ", ".join(f"{v}:{n}" for v, n in data["filtered_vendors"].items())
        out.append(
            f"Relevance filter dropped {data['filtered_offtopic']} off-topic "
            f"(common-word-vendor contamination) reviews before analysis [{fv}]."
        )
    out.append(
        f"Baseline: {data['baseline_overall_dissatisfaction']} reviews carry only "
        f"non-specific \"overall dissatisfaction\" (excluded from ranked themes below)."
    )
    out.append("")
    out.append("## Ranked opportunity themes")
    out.append("")
    out.append("| Theme | Reviews | Mean urgency | Vendor breadth | Opportunity |")
    out.append("|---|--:|--:|:--:|--:|")
    for t in data["themes"]:
        out.append(
            f"| {t['theme']} | {t['reviews']} | {t['mean_urgency']} | "
            f"{t['vendor_breadth']} | {t['opportunity_score']} |"
        )
    out.append("")
    out.append("## Top unmet needs / feature gaps (verbatim)")
    out.append("")
    if data["feature_gaps"]:
        for g in data["feature_gaps"][:20]:
            out.append(f"- ({g['count']}x) {g['gap']}")
    else:
        out.append("_None extracted._")
    out.append("")
    out.append("## What already works well (do NOT compete here)")
    out.append("")
    if data.get("works_well"):
        for w in data["works_well"][:12]:
            out.append(f"- ({w['count']}x) {w['aspect']}")
    else:
        out.append("_None extracted._")
    out.append("")
    out.append("## Pricing pain")
    out.append("")
    out.append(f"- Price-increase mentions: {data['pricing']['price_increase_mentions']}")
    for p in data["pricing"]["sample_phrases"]:
        out.append(f"- \"{p}\"")
    out.append("")
    out.append("## Competitors cited (alternatives buyers mention)")
    out.append("")
    if data["competitors"]:
        for c in data["competitors"][:15]:
            ctx = ", ".join(f"{k}:{v}" for k, v in sorted(c["contexts"].items(), key=lambda x: -x[1]))
            out.append(f"- {c['name']} ({c['count']}x) -- {ctx}")
    else:
        out.append("_None._")
    out.append("")
    out.append("## Evidence (sample quotes per top theme)")
    out.append("")
    for t in data["themes"][:6]:
        if not t["quotes"]:
            continue
        out.append(f"**{t['theme']}**")
        for q in t["quotes"]:
            out.append(f"> {q['text']}  \n> -- {q['vendor'] or 'unknown'} ({q['source'] or 'n/a'})")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------


async def _fetch(pool, category: str) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT r.id, r.source,
               vm.vendor_name AS vendor,
               CASE WHEN r.enrichment->>'urgency_score' ~ '^[0-9.]+$'
                    THEN (r.enrichment->>'urgency_score')::numeric END AS urgency,
               r.enrichment->'pain_categories'      AS pain_categories,
               r.enrichment->'feature_gaps'         AS feature_gaps,
               r.enrichment->'specific_complaints'  AS specific_complaints,
               r.enrichment->'competitors_mentioned' AS competitors,
               r.enrichment->'pricing_phrases'      AS pricing_phrases,
               r.enrichment->'positive_aspects'     AS positive_aspects,
               r.enrichment->'quotable_phrases'     AS quotable_phrases,
               (r.enrichment->'budget_signals'->>'price_increase_mentioned')::boolean AS price_increase
        FROM b2b_reviews r
        LEFT JOIN b2b_review_vendor_mentions vm
            ON vm.review_id = r.id AND vm.is_primary
        WHERE r.enrichment_status = 'enriched'
          AND r.product_category = $1
        """,
        category,
    )
    return [
        {
            "vendor": r["vendor"],
            "urgency": float(r["urgency"]) if r["urgency"] is not None else None,
            "pain_categories": r["pain_categories"],
            "feature_gaps": r["feature_gaps"],
            "specific_complaints": r["specific_complaints"],
            "competitors": r["competitors"],
            "pricing_phrases": r["pricing_phrases"],
            "positive_aspects": r["positive_aspects"],
            "quotable_phrases": r["quotable_phrases"],
            "price_increase": r["price_increase"],
            "source": r["source"],
        }
        for r in rows
    ]


async def _run(category: str) -> dict[str, Any]:
    await init_database()
    pool = get_db_pool()
    reviews = await _fetch(pool, category)
    return aggregate(reviews)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="product_category, e.g. CRM")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args()

    async def _main() -> int:
        try:
            data = await _run(args.category)
        finally:
            await close_database()
        if not data["total_reviews"]:
            print(f"No enriched reviews found for category: {args.category}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True, default=str))
        else:
            print(render_markdown(args.category, data))
        return 0

    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
