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
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.b2b_blog_post_generation")


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


# -- entry point --------------------------------------------------

async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate a B2B data-backed blog post."""
    cfg = settings.b2b_churn
    if not cfg.blog_post_enabled:
        return {"_skip_synthesis": "B2B blog post generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Stage 1: topic selection
    topic = await _select_topic(pool, cfg.blog_post_max_per_run)
    if topic is None:
        return {"_skip_synthesis": "No viable B2B blog topic found"}

    topic_type, topic_ctx = topic

    # Stage 2: data gathering
    data = await _gather_data(pool, topic_type, topic_ctx)

    # Stage 3: blueprint construction (deterministic, no LLM)
    blueprint = _build_blueprint(topic_type, topic_ctx, data)

    # Stage 4: content generation (LLM)
    from ...pipelines.llm import get_pipeline_llm

    llm = get_pipeline_llm(
        prefer_cloud=True,
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model=cfg.blog_post_openrouter_model,
    )
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"_skip_synthesis": "No LLM available for B2B blog post generation"}

    content = _generate_content(llm, blueprint, cfg.blog_post_max_tokens)
    if content is None:
        return {"_skip_synthesis": "LLM content generation failed"}

    # Inject affiliate links: replace {{affiliate:slug}} with actual URLs
    affiliate_url = blueprint.data_context.get("affiliate_url", "")
    affiliate_slug = blueprint.data_context.get("affiliate_slug", "")
    if affiliate_slug and affiliate_url and content.get("content"):
        content["content"] = content["content"].replace(
            f"{{{{affiliate:{affiliate_slug}}}}}",
            affiliate_url,
        )

    # Stage 5: assembly & storage
    post_id = await _assemble_and_store(pool, blueprint, content, llm)

    if not post_id:
        return {
            "_skip_synthesis": f"Skipped: slug {blueprint.slug} is already published",
        }

    # Notify
    from ...pipelines.notify import send_pipeline_notification

    n_charts = len(blueprint.charts)
    msg = (
        f"B2B Blog draft: '{content['title']}' ({blueprint.topic_type}) "
        f"with {n_charts} chart{'s' if n_charts != 1 else ''}. "
        f"Review at /admin/blog"
    )
    await send_pipeline_notification(
        msg, task, title="Atlas: B2B Blog Post Draft",
        default_tags="brain,newspaper",
    )

    return {
        "_skip_synthesis": msg,
        "post_id": str(post_id),
        "topic_type": blueprint.topic_type,
        "slug": blueprint.slug,
        "charts": n_charts,
    }


# -- Stage 1: Topic Selection -------------------------------------

async def _select_topic(pool, max_per_run: int = 1) -> tuple[str, dict[str, Any]] | None:
    """Score candidates and pick the best unwritten B2B topic."""
    today = date.today()
    month_suffix = today.strftime("%Y-%m")

    alternatives, showdowns, churn_reports, migrations = await asyncio.gather(
        _find_vendor_alternative_candidates(pool),
        _find_vendor_showdown_candidates(pool),
        _find_churn_report_candidates(pool),
        _find_migration_guide_candidates(pool),
        return_exceptions=True,
    )
    alternatives = alternatives if not isinstance(alternatives, Exception) else []
    showdowns = showdowns if not isinstance(showdowns, Exception) else []
    churn_reports = churn_reports if not isinstance(churn_reports, Exception) else []
    migrations = migrations if not isinstance(migrations, Exception) else []

    raw_candidates: list[tuple[str, float, str, dict[str, Any]]] = []

    for alt in alternatives:
        slug = f"{_slugify(alt['vendor'])}-alternatives-{month_suffix}"
        score = alt["urgency"] * alt["review_count"] * (2.0 if alt.get("has_affiliate") else 1.0)
        raw_candidates.append((slug, score, "vendor_alternative", {**alt, "slug": slug}))

    for pair in showdowns:
        slug = f"{_slugify(pair['vendor_a'])}-vs-{_slugify(pair['vendor_b'])}-{month_suffix}"
        score = pair["pain_diff"] * pair["total_reviews"]
        raw_candidates.append((slug, score, "vendor_showdown", {**pair, "slug": slug}))

    for cr in churn_reports:
        slug = f"{_slugify(cr['vendor'])}-churn-report-{month_suffix}"
        score = cr["negative_reviews"] * cr["avg_urgency"]
        raw_candidates.append((slug, score, "churn_report", {**cr, "slug": slug}))

    for mig in migrations:
        slug = f"migration-from-{_slugify(mig['vendor'])}-{month_suffix}"
        score = mig["switch_count"] * mig["review_total"]
        raw_candidates.append((slug, score, "migration_guide", {**mig, "slug": slug}))

    if not raw_candidates:
        return None

    # --- Dedup layer 1: exact slug match (same topic+vendor+month) ---
    all_slugs = list({c[0] for c in raw_candidates})
    existing_slugs = await _batch_slug_check(pool, all_slugs)

    # --- Dedup layer 2: vendor-level cooldown (any topic type, 90 days) ---
    covered_vendors = await _recently_covered_vendors(pool, days=90)

    def _vendor_keys(ctx: dict) -> set[str]:
        """Return all vendor names from a candidate (normalized for dedup).

        Showdowns have vendor_a + vendor_b; others have vendor.
        """
        keys = set()
        for k in ("vendor", "vendor_a", "vendor_b"):
            v = ctx.get(k, "")
            if v:
                keys.add(v.lower().strip())
        return keys

    candidates = [
        (score, topic_type, ctx)
        for slug, score, topic_type, ctx in raw_candidates
        if slug not in existing_slugs
        and not _vendor_keys(ctx) & covered_vendors  # no overlap with recent posts
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Dedup layer 3: one vendor per run (pick highest-scoring topic) ---
    seen_vendors: set[str] = set()
    best = None
    for c in candidates:
        vks = _vendor_keys(c[2])
        if vks & seen_vendors:
            continue
        seen_vendors |= vks
        if best is None:
            best = c

    if best is None:
        return None
    logger.info(
        "Selected B2B topic: %s (score=%.1f, slug=%s)",
        best[1], best[0], best[2].get("slug"),
    )
    return best[1], best[2]


async def _find_vendor_alternative_candidates(pool) -> list[dict[str, Any]]:
    """Vendors with high churn + affiliate partner covering the category."""
    rows = await pool.fetch(
        """
        SELECT
            cs.vendor_name AS vendor,
            cs.product_category AS category,
            cs.avg_urgency_score AS urgency,
            cs.total_reviews AS review_count,
            ap.id AS affiliate_id,
            ap.name AS affiliate_name,
            ap.product_name AS affiliate_product,
            ap.affiliate_url
        FROM b2b_churn_signals cs
        LEFT JOIN affiliate_partners ap
            ON LOWER(ap.category) = LOWER(cs.product_category)
            AND ap.enabled = true
        WHERE cs.avg_urgency_score >= 6
          AND cs.total_reviews >= 5
        ORDER BY cs.avg_urgency_score * cs.total_reviews DESC
        LIMIT 15
        """
    )
    return [
        {
            "vendor": r["vendor"],
            "category": r["category"],
            "urgency": float(r["urgency"]),
            "review_count": r["review_count"],
            "has_affiliate": r["affiliate_id"] is not None,
            "affiliate_id": str(r["affiliate_id"]) if r["affiliate_id"] else None,
            "affiliate_name": r["affiliate_name"],
            "affiliate_product": r["affiliate_product"],
            "affiliate_url": r["affiliate_url"],
        }
        for r in rows
    ]


async def _find_vendor_showdown_candidates(pool) -> list[dict[str, Any]]:
    """Pairs of vendors in the same category with contrasting pain profiles."""
    rows = await pool.fetch(
        """
        SELECT
            a.vendor_name AS vendor_a, b.vendor_name AS vendor_b,
            a.product_category AS category,
            a.total_reviews AS reviews_a, b.total_reviews AS reviews_b,
            (a.total_reviews + b.total_reviews) AS total_reviews,
            a.avg_urgency_score AS urgency_a, b.avg_urgency_score AS urgency_b,
            ABS(a.avg_urgency_score - b.avg_urgency_score) AS pain_diff
        FROM b2b_churn_signals a
        JOIN b2b_churn_signals b
            ON a.product_category = b.product_category
            AND a.vendor_name < b.vendor_name
        WHERE a.total_reviews >= 5 AND b.total_reviews >= 5
          AND ABS(a.avg_urgency_score - b.avg_urgency_score) > 1.0
        ORDER BY (a.total_reviews + b.total_reviews) DESC
        LIMIT 10
        """
    )
    return [
        {
            "vendor_a": r["vendor_a"],
            "vendor_b": r["vendor_b"],
            "category": r["category"],
            "reviews_a": r["reviews_a"],
            "reviews_b": r["reviews_b"],
            "total_reviews": r["total_reviews"],
            "urgency_a": round(float(r["urgency_a"]), 1),
            "urgency_b": round(float(r["urgency_b"]), 1),
            "pain_diff": round(float(r["pain_diff"]), 1),
        }
        for r in rows
    ]


async def _find_churn_report_candidates(pool) -> list[dict[str, Any]]:
    """Single vendor with high urgency + many negative reviews."""
    rows = await pool.fetch(
        """
        SELECT
            vendor_name AS vendor,
            product_category AS category,
            negative_reviews,
            avg_urgency_score AS avg_urgency,
            total_reviews
        FROM b2b_churn_signals
        WHERE negative_reviews >= 8
          AND avg_urgency_score >= 6
        ORDER BY negative_reviews * avg_urgency_score DESC
        LIMIT 10
        """
    )
    return [
        {
            "vendor": r["vendor"],
            "category": r["category"],
            "negative_reviews": r["negative_reviews"],
            "avg_urgency": round(float(r["avg_urgency"]), 1),
            "total_reviews": r["total_reviews"],
        }
        for r in rows
    ]


async def _find_migration_guide_candidates(pool) -> list[dict[str, Any]]:
    """Vendors with high switched_from counts in product profiles."""
    rows = await pool.fetch(
        """
        SELECT
            pp.vendor_name AS vendor,
            pp.product_category AS category,
            COALESCE(jsonb_array_length(pp.commonly_switched_from), 0) AS switch_count,
            (SELECT COUNT(*) FROM b2b_reviews br WHERE br.vendor_name = pp.vendor_name) AS review_total
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


async def _batch_slug_check(pool, slugs: list[str]) -> set[str]:
    """Check which slugs already exist (all time). Single query."""
    if not slugs:
        return set()
    rows = await pool.fetch(
        "SELECT slug FROM blog_posts WHERE slug = ANY($1)",
        slugs,
    )
    return {r["slug"] for r in rows}


async def _recently_covered_vendors(pool, days: int = 90) -> set[str]:
    """Return vendor names that already have a B2B blog post in the last N days.

    Prevents the same vendor from dominating the blog across topic types
    (e.g. Freshdesk getting a churn_report, migration_guide, AND vendor_showdown).
    Includes vendor_b from showdown posts so both sides of a comparison are covered.
    """
    rows = await pool.fetch(
        """
        SELECT DISTINCT LOWER(vendor) AS vendor FROM (
            SELECT data_context->>'vendor' AS vendor
            FROM blog_posts
            WHERE topic_type IN ('vendor_alternative','vendor_showdown','churn_report','migration_guide')
              AND created_at > NOW() - make_interval(days => $1)
              AND data_context->>'vendor' IS NOT NULL
            UNION ALL
            SELECT data_context->>'vendor_a' AS vendor
            FROM blog_posts
            WHERE topic_type IN ('vendor_alternative','vendor_showdown','churn_report','migration_guide')
              AND created_at > NOW() - make_interval(days => $1)
              AND data_context->>'vendor_a' IS NOT NULL
            UNION ALL
            SELECT data_context->>'vendor_b' AS vendor
            FROM blog_posts
            WHERE topic_type IN ('vendor_alternative','vendor_showdown','churn_report','migration_guide')
              AND created_at > NOW() - make_interval(days => $1)
              AND data_context->>'vendor_b' IS NOT NULL
        ) sub
        WHERE vendor != ''
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


# -- Stage 2: Data Gathering --------------------------------------

async def _gather_data(
    pool, topic_type: str, topic_ctx: dict[str, Any]
) -> dict[str, Any]:
    """Fetch data needed for the blueprint from B2B tables."""
    data: dict[str, Any] = {}

    if topic_type == "vendor_alternative":
        vendor = topic_ctx["vendor"]
        category = topic_ctx["category"]
        profile, signals, reviews, partner = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_churn_signals(pool, vendor),
            _fetch_quotable_reviews(pool, vendor_name=vendor),
            _fetch_affiliate_partner(pool, topic_ctx.get("affiliate_id")),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = signals if not isinstance(signals, Exception) else []
        data["quotes"] = reviews if not isinstance(reviews, Exception) else []
        data["partner"] = partner if not isinstance(partner, Exception) else None

    elif topic_type == "vendor_showdown":
        vendor_a, vendor_b = topic_ctx["vendor_a"], topic_ctx["vendor_b"]
        prof_a, prof_b, sigs_a, sigs_b, quotes = await asyncio.gather(
            _fetch_product_profile(pool, vendor_a),
            _fetch_product_profile(pool, vendor_b),
            _fetch_churn_signals(pool, vendor_a),
            _fetch_churn_signals(pool, vendor_b),
            _fetch_quotable_reviews(pool, category=topic_ctx["category"]),
            return_exceptions=True,
        )
        data["profile_a"] = prof_a if not isinstance(prof_a, Exception) else {}
        data["profile_b"] = prof_b if not isinstance(prof_b, Exception) else {}
        data["signals_a"] = sigs_a if not isinstance(sigs_a, Exception) else []
        data["signals_b"] = sigs_b if not isinstance(sigs_b, Exception) else []
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    elif topic_type == "churn_report":
        vendor = topic_ctx["vendor"]
        profile, signals, quotes = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_churn_signals(pool, vendor),
            _fetch_quotable_reviews(pool, vendor_name=vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = signals if not isinstance(signals, Exception) else []
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    elif topic_type == "migration_guide":
        vendor = topic_ctx["vendor"]
        profile, signals, quotes = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_churn_signals(pool, vendor),
            _fetch_quotable_reviews(pool, vendor_name=vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = signals if not isinstance(signals, Exception) else []
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    # Data context metadata
    ctx_row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_reviews,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched,
            MIN(imported_at)::date AS earliest,
            MAX(imported_at)::date AS latest
        FROM b2b_reviews
        """
    )
    data["data_context"] = {
        "total_reviews_analyzed": ctx_row["total_reviews"] if ctx_row else 0,
        "enriched_count": ctx_row["enriched"] if ctx_row else 0,
        "review_period": (
            f"{ctx_row['earliest']} to {ctx_row['latest']}"
            if ctx_row and ctx_row["earliest"]
            else "dates unavailable"
        ),
        "report_date": str(date.today()),
    }

    # Embed vendor name(s) so vendor-level dedup can query data_context later
    if topic_ctx.get("vendor"):
        data["data_context"]["vendor"] = topic_ctx["vendor"]
    if topic_ctx.get("vendor_a"):
        data["data_context"]["vendor_a"] = topic_ctx["vendor_a"]
    if topic_ctx.get("vendor_b"):
        data["data_context"]["vendor_b"] = topic_ctx["vendor_b"]

    # Attach affiliate info to data_context if available
    partner = data.get("partner")
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


async def _fetch_churn_signals(pool, vendor_name: str) -> list[dict[str, Any]]:
    """Fetch churn signal data for a vendor from the aggregate table."""
    row = await pool.fetchrow(
        """
        SELECT
            avg_urgency_score, total_reviews, negative_reviews,
            top_pain_categories, top_feature_gaps, top_competitors,
            quotable_evidence, product_category
        FROM b2b_churn_signals
        WHERE vendor_name = $1
        LIMIT 1
        """,
        vendor_name,
    )
    if not row:
        return []

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
        count = pc.get("count", 1) if isinstance(pc, dict) else 1
        results.append({
            "pain_category": cat_name,
            "signal_count": count,
            "avg_urgency": round(float(row["avg_urgency_score"]), 1),
            "feature_gaps": [
                fg.get("feature", fg) if isinstance(fg, dict) else str(fg)
                for fg in feature_gaps[:5]
            ],
        })
    return results


async def _fetch_quotable_reviews(
    pool, vendor_name: str | None = None, category: str | None = None
) -> list[dict[str, Any]]:
    """Pull impactful review excerpts relevant to the topic."""
    if vendor_name:
        rows = await pool.fetch(
            """
            SELECT review_text, vendor_name, reviewer_title, rating,
                   enrichment->>'urgency_score' AS urgency
            FROM b2b_reviews
            WHERE vendor_name = $1
              AND enrichment_status = 'enriched'
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT 10
            """,
            vendor_name,
        )
    elif category:
        rows = await pool.fetch(
            """
            SELECT review_text, vendor_name, reviewer_title, rating,
                   enrichment->>'urgency_score' AS urgency
            FROM b2b_reviews
            WHERE product_category = $1
              AND enrichment_status = 'enriched'
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT 10
            """,
            category,
        )
    else:
        return []

    results = []
    for r in rows:
        text = r["review_text"] or ""
        # Extract the most impactful sentence
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        phrase = sentences[0] if sentences else text[:150]
        urg = 0.0
        try:
            urg = float(r["urgency"]) if r["urgency"] else 0.0
        except (ValueError, TypeError):
            pass
        results.append({
            "phrase": phrase[:200],
            "vendor": r["vendor_name"],
            "urgency": urg,
            "role": r["reviewer_title"],
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
    }[topic_type]
    return builder(topic_ctx, data)


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

    sections.append(SectionSpec(
        id="verdict",
        heading="The Verdict",
        goal="Summarize findings and recommend action",
        key_stats={"vendor": vendor, "urgency": ctx["urgency"]},
    ))

    # Build affiliate context
    data_context = {**data["data_context"]}
    if partner:
        data_context["affiliate_url"] = partner["affiliate_url"]
        data_context["affiliate_slug"] = _slugify(partner["product_name"])

    return PostBlueprint(
        topic_type="vendor_alternative",
        slug=ctx["slug"],
        suggested_title=f"{vendor} Alternatives: What {ctx['review_count']}+ Enterprise Reviews Reveal",
        tags=[category, vendor.lower(), "alternatives", "churn-analysis"],
        data_context=data_context,
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_vendor_showdown(ctx: dict, data: dict) -> PostBlueprint:
    vendor_a, vendor_b = ctx["vendor_a"], ctx["vendor_b"]
    category = ctx.get("category", "software")

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

    sections.append(SectionSpec(
        id="verdict",
        heading="The Verdict",
        goal="Declare which vendor fares better and the decisive factor",
        key_stats={
            "vendor_a": vendor_a,
            "vendor_b": vendor_b,
            "urgency_a": ctx["urgency_a"],
            "urgency_b": ctx["urgency_b"],
        },
    ))

    return PostBlueprint(
        topic_type="vendor_showdown",
        slug=ctx["slug"],
        suggested_title=f"{vendor_a} vs {vendor_b}: What {ctx['total_reviews']}+ Churn Signals Reveal",
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

    sections.append(SectionSpec(
        id="outlook",
        heading="What This Means for Teams Using " + vendor,
        goal="Provide actionable guidance for current users",
        key_stats={"vendor": vendor, "avg_urgency": ctx["avg_urgency"]},
    ))

    return PostBlueprint(
        topic_type="churn_report",
        slug=ctx["slug"],
        suggested_title=f"{vendor} Churn Report: {ctx['negative_reviews']}+ Reviews Signal Growing Frustration",
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
    ]

    charts = []
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal=f"Highlight the volume of migrations to {vendor}",
            key_stats={
                "vendor": vendor,
                "category": category,
                "switch_count": ctx["switch_count"],
                "review_total": ctx["review_total"],
            },
            data_summary=(
                f"{vendor} attracts users from {ctx['switch_count']} competitors "
                f"based on {ctx['review_total']} total reviews."
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

    sections.append(SectionSpec(
        id="practical",
        heading="Making the Switch: What to Expect",
        goal="Practical migration considerations (integrations, learning curve)",
        key_stats={
            "vendor": vendor,
            "integrations": profile.get("integrations", [])[:5] if isinstance(profile.get("integrations"), list) else [],
        },
    ))

    sections.append(SectionSpec(
        id="takeaway",
        heading="Key Takeaways",
        goal="Summary and recommendations",
        key_stats={"vendor": vendor, "switch_count": ctx["switch_count"]},
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


# -- Stage 4: Content Generation ----------------------------------

def _generate_content(
    llm, blueprint: PostBlueprint, max_tokens: int
) -> dict[str, Any] | None:
    """Single LLM call: blueprint in, {title, description, content} out."""
    from ...pipelines.llm import clean_llm_output, parse_json_response
    from ...skills.registry import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_blog_post_generation")
    if skill is None:
        logger.error("Skill digest/b2b_blog_post_generation not found")
        return None

    payload = {
        "topic_type": blueprint.topic_type,
        "suggested_title": blueprint.suggested_title,
        "data_context": blueprint.data_context,
        "sections": [
            {
                "id": s.id,
                "heading": s.heading,
                "goal": s.goal,
                "key_stats": s.key_stats,
                "chart_ids": s.chart_ids,
                "data_summary": s.data_summary,
            }
            for s in blueprint.sections
        ],
        "available_charts": [
            {
                "chart_id": c.chart_id,
                "chart_type": c.chart_type,
                "title": c.title,
            }
            for c in blueprint.charts
        ],
        "quotable_phrases": blueprint.quotable_phrases[:5],
    }

    from ...services.protocols import Message

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload, indent=2, default=str)),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=0.7)
        text = result.get("response", "") if isinstance(result, dict) else str(result)
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)

        if parsed.get("_parse_fallback"):
            logger.error("Failed to parse LLM response as JSON")
            return None

        if not all(k in parsed for k in ("title", "description", "content")):
            logger.error("LLM response missing required keys: %s", list(parsed.keys()))
            return None

        return parsed
    except Exception:
        logger.exception("LLM content generation failed")
        return None


# -- Stage 5: Assembly & Storage ----------------------------------

async def _assemble_and_store(
    pool, blueprint: PostBlueprint, content: dict[str, Any], llm
) -> str:
    """Store the assembled post as a draft in blog_posts."""
    charts_json = [asdict(c) for c in blueprint.charts]
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")

    row = await pool.fetchrow(
        """
        INSERT INTO blog_posts (
            slug, title, description, topic_type, tags,
            content, charts, data_context,
            status, llm_model, source_report_date
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'draft',$9,$10)
        ON CONFLICT (slug) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            content = EXCLUDED.content,
            charts = EXCLUDED.charts,
            data_context = EXCLUDED.data_context,
            llm_model = EXCLUDED.llm_model,
            source_report_date = EXCLUDED.source_report_date
        WHERE blog_posts.status != 'published'
        RETURNING id
        """,
        blueprint.slug,
        content["title"],
        content.get("description", ""),
        blueprint.topic_type,
        json.dumps(blueprint.tags),
        content["content"],
        json.dumps(charts_json, default=str),
        json.dumps(blueprint.data_context, default=str),
        str(model_name),
        date.today(),
    )
    if not row:
        logger.warning(
            "Skipped overwrite of published post: slug=%s", blueprint.slug
        )
        return ""
    post_id = str(row["id"])
    logger.info("Stored B2B blog draft: slug=%s, id=%s", blueprint.slug, post_id)

    # Write .ts file for the frontend if ui_path is configured
    cfg = settings.b2b_churn
    if cfg.blog_post_ui_path:
        try:
            _write_ui_post(
                cfg.blog_post_ui_path,
                blueprint,
                content,
                charts_json,
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
) -> None:
    """Write a .ts post file and register it in index.ts."""
    from pathlib import Path

    blog_dir = Path(ui_path)
    if not blog_dir.is_dir():
        logger.warning("blog_post_ui_path does not exist: %s", ui_path)
        return

    slug = blueprint.slug
    filename = slug + ".ts"
    var_name = re.sub(r"[^a-zA-Z0-9]", "_", slug).strip("_")
    parts = var_name.split("_")
    var_name = parts[0] + "".join(p.capitalize() for p in parts[1:])

    charts_str = json.dumps(charts_json, indent=2, default=str)
    escaped_content = (
        content["content"]
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )
    escaped_title = content["title"].replace("'", "\\'")
    escaped_desc = content.get("description", "").replace("'", "\\'")

    ts_content = f"""import type {{ BlogPost }} from './index'

const post: BlogPost = {{
  slug: '{slug}',
  title: '{escaped_title}',
  description: '{escaped_desc}',
  date: '{date.today().isoformat()}',
  author: 'Churn Intel Team',
  tags: {json.dumps(blueprint.tags)},
  topic_type: '{blueprint.topic_type}',
  charts: {charts_str},
  content: `{escaped_content}`,
}}

export default post
"""

    post_path = blog_dir / filename
    post_path.write_text(ts_content, encoding="utf-8")
    logger.info("Wrote B2B blog UI file: %s", post_path)

    index_path = blog_dir / "index.ts"
    if not index_path.exists():
        logger.warning("index.ts not found in %s", blog_dir)
        return

    index_text = index_path.read_text(encoding="utf-8")
    import_line = f"import {var_name} from './{slug}'"

    if slug in index_text:
        logger.debug("Post %s already in index.ts, skipping", slug)
        return

    lines = index_text.split("\n")
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("import "):
            last_import_idx = i

    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, import_line)
    else:
        lines.insert(0, import_line)

    new_text = "\n".join(lines)
    new_text = re.sub(
        r"(].sort\()",
        f"  {var_name},\n\\1",
        new_text,
        count=1,
    )

    index_path.write_text(new_text, encoding="utf-8")
    logger.info("Updated index.ts with %s", slug)
