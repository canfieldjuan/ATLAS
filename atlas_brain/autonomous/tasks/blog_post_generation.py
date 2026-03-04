"""
Blog post generation: picks the best data story each night, builds a
deterministic blueprint (sections + chart specs), generates prose via LLM,
and stores the assembled draft in blog_posts.

Runs daily after complaint_content_generation (default 11 PM).

Pipeline stages:
  1. Topic selection  -- score candidates, pick best unwritten story
  2. Data gathering   -- reuse existing SQL fetchers
  3. Blueprint build  -- deterministic section/chart layout
  4. Content gen      -- single LLM call with blueprint as input
  5. Assembly/store   -- draft in blog_posts, ntfy notification

Returns _skip_synthesis.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask

logger = logging.getLogger("atlas.autonomous.tasks.blog_post_generation")


# -- dataclasses --------------------------------------------------

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
    """Autonomous task handler: generate a data-backed blog post."""
    cfg = settings.external_data
    if not cfg.complaint_mining_enabled or not cfg.blog_post_enabled:
        return {"_skip_synthesis": "Blog post generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    # Stage 1: topic selection
    topic = await _select_topic(pool)
    if topic is None:
        return {"_skip_synthesis": "No viable blog topic found"}

    topic_type, topic_ctx = topic

    # Stage 2: data gathering
    data = await _gather_data(pool, topic_type, topic_ctx)

    # Stage 3: blueprint construction (deterministic, no LLM)
    blueprint = _build_blueprint(topic_type, topic_ctx, data)

    # Stage 4: content generation (LLM)
    from ...pipelines.llm import get_pipeline_llm, clean_llm_output, parse_json_response

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
        return {"_skip_synthesis": "No LLM available for blog post generation"}

    content = _generate_content(llm, blueprint, cfg.blog_post_max_tokens)
    if content is None:
        return {"_skip_synthesis": "LLM content generation failed"}

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
        f"Blog draft: '{content['title']}' ({blueprint.topic_type}) "
        f"with {n_charts} chart{'s' if n_charts != 1 else ''}. "
        f"Review at /admin/blog"
    )
    await send_pipeline_notification(
        msg, task, title="Atlas: Blog Post Draft",
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

async def _select_topic(pool) -> tuple[str, dict[str, Any]] | None:
    """Score candidates and pick the best unwritten topic."""
    today = date.today()
    month_suffix = today.strftime("%Y-%m")

    # Gather all candidate data in parallel
    brand_pairs, cat_stats, migrations, safety = await asyncio.gather(
        _find_brand_showdown_candidates(pool),
        _find_complaint_roundup_candidates(pool),
        _find_migration_candidates(pool),
        _find_safety_candidates(pool),
        return_exceptions=True,
    )
    brand_pairs = brand_pairs if not isinstance(brand_pairs, Exception) else []
    cat_stats = cat_stats if not isinstance(cat_stats, Exception) else []
    migrations = migrations if not isinstance(migrations, Exception) else []
    safety = safety if not isinstance(safety, Exception) else []

    # Build slug -> (score, topic_type, ctx) candidates
    raw_candidates: list[tuple[str, float, str, dict[str, Any]]] = []

    for pair in brand_pairs:
        slug = f"{_slugify(pair['brand_a'])}-vs-{_slugify(pair['brand_b'])}-{month_suffix}"
        score = pair["review_total"] * 0.3 + pair["pain_diff"] * 10
        raw_candidates.append((slug, score, "brand_showdown", {**pair, "slug": slug}))

    for cat in cat_stats:
        slug = f"top-complaints-{_slugify(cat['category'])}-{month_suffix}"
        score = cat["review_count"] * 0.2 + cat["avg_pain"] * 8
        raw_candidates.append((slug, score, "complaint_roundup", {**cat, "slug": slug}))

    for mig in migrations:
        slug = f"migration-{_slugify(mig['category'])}-{month_suffix}"
        score = mig["total_mentions"] * 5
        raw_candidates.append((slug, score, "migration_report", {**mig, "slug": slug}))

    for sf in safety:
        slug = f"safety-{_slugify(sf['category'])}-{month_suffix}"
        score = sf["safety_count"] * 15
        raw_candidates.append((slug, score, "safety_spotlight", {**sf, "slug": slug}))

    if not raw_candidates:
        return None

    # --- Dedup layer 1: exact slug match (same topic+category+month) ---
    all_slugs = list({c[0] for c in raw_candidates})
    existing_slugs = await _batch_slug_check(pool, all_slugs)

    # --- Dedup layer 2: category-level cooldown (any topic type, 90 days) ---
    covered = await _recently_covered_subjects(pool, days=90)

    def _subject_key(ctx: dict) -> str:
        """Normalize category/brand for dedup."""
        return (
            ctx.get("category") or ctx.get("brand_a") or ""
        ).lower().strip()

    candidates = [
        (score, topic_type, ctx)
        for slug, score, topic_type, ctx in raw_candidates
        if slug not in existing_slugs
        and _subject_key(ctx) not in covered
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Dedup layer 3: one subject per run (pick highest-scoring) ---
    seen: set[str] = set()
    best = None
    for c in candidates:
        sk = _subject_key(c[2])
        if sk in seen:
            continue
        seen.add(sk)
        if best is None:
            best = c

    if best is None:
        return None
    logger.info(
        "Selected topic: %s (score=%.1f, slug=%s)",
        best[1], best[0], best[2].get("slug"),
    )
    return best[1], best[2]


async def _find_brand_showdown_candidates(pool) -> list[dict[str, Any]]:
    """Find pairs of brands in the same category suitable for a showdown."""
    rows = await pool.fetch(
        """
        WITH brand_stats AS (
            SELECT
                pm.brand,
                COALESCE(
                    REPLACE(pm.categories->>2, '&amp;', '&'),
                    REPLACE(pm.categories->>1, '&amp;', '&'),
                    pr.source_category
                ) AS category,
                count(*) AS review_count,
                avg(pr.pain_score) AS avg_pain,
                avg(pr.rating) AS avg_rating
            FROM product_reviews pr
            JOIN product_metadata pm ON pm.asin = pr.asin
            WHERE pr.deep_enrichment_status = 'enriched'
              AND pm.brand IS NOT NULL AND pm.brand != ''
            GROUP BY pm.brand, category
            HAVING count(*) >= 20
        )
        SELECT
            a.brand AS brand_a, b.brand AS brand_b,
            a.category,
            a.review_count AS reviews_a, b.review_count AS reviews_b,
            (a.review_count + b.review_count) AS review_total,
            a.avg_pain AS pain_a, b.avg_pain AS pain_b,
            abs(a.avg_pain - b.avg_pain) AS pain_diff,
            a.avg_rating AS rating_a, b.avg_rating AS rating_b
        FROM brand_stats a
        JOIN brand_stats b ON a.category = b.category AND a.brand < b.brand
        WHERE abs(a.avg_pain - b.avg_pain) > 1.5
        ORDER BY (a.review_count + b.review_count) DESC
        LIMIT 10
        """
    )
    return [
        {
            "brand_a": r["brand_a"],
            "brand_b": r["brand_b"],
            "category": r["category"],
            "reviews_a": r["reviews_a"],
            "reviews_b": r["reviews_b"],
            "review_total": r["review_total"],
            "pain_a": round(float(r["pain_a"]), 1),
            "pain_b": round(float(r["pain_b"]), 1),
            "pain_diff": round(float(r["pain_diff"]), 1),
            "rating_a": round(float(r["rating_a"]), 2),
            "rating_b": round(float(r["rating_b"]), 2),
        }
        for r in rows
    ]


async def _find_complaint_roundup_candidates(pool) -> list[dict[str, Any]]:
    """Find categories with enough enriched reviews and high pain."""
    rows = await pool.fetch(
        """
        SELECT
            COALESCE(
                REPLACE(pm.categories->>2, '&amp;', '&'),
                REPLACE(pm.categories->>1, '&amp;', '&'),
                pr.source_category
            ) AS category,
            count(*) AS review_count,
            avg(pr.pain_score) AS avg_pain,
            avg(pr.rating) AS avg_rating
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
        GROUP BY category
        HAVING count(*) >= 50 AND avg(pr.pain_score) >= 5.0
        ORDER BY avg(pr.pain_score) DESC
        LIMIT 10
        """
    )
    return [
        {
            "category": r["category"],
            "review_count": r["review_count"],
            "avg_pain": round(float(r["avg_pain"]), 1),
            "avg_rating": round(float(r["avg_rating"]), 2),
        }
        for r in rows
    ]


async def _find_migration_candidates(pool) -> list[dict[str, Any]]:
    """Find categories with significant competitive migration mentions."""
    rows = await pool.fetch(
        """
        SELECT
            COALESCE(
                REPLACE(pm.categories->>2, '&amp;', '&'),
                REPLACE(pm.categories->>1, '&amp;', '&'),
                pr.source_category
            ) AS category,
            count(*) AS total_mentions
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        CROSS JOIN jsonb_array_elements(pr.deep_extraction->'product_comparisons') AS comp
        WHERE pr.deep_enrichment_status = 'enriched'
          AND comp->>'direction' = 'switched_to'
          AND jsonb_array_length(pr.deep_extraction->'product_comparisons') > 0
        GROUP BY category
        HAVING count(*) >= 10
        ORDER BY count(*) DESC
        LIMIT 10
        """
    )
    return [
        {
            "category": r["category"],
            "total_mentions": r["total_mentions"],
        }
        for r in rows
    ]


async def _find_safety_candidates(pool) -> list[dict[str, Any]]:
    """Find categories with safety-flagged reviews."""
    rows = await pool.fetch(
        """
        SELECT
            COALESCE(
                REPLACE(pm.categories->>2, '&amp;', '&'),
                REPLACE(pm.categories->>1, '&amp;', '&'),
                pr.source_category
            ) AS category,
            count(*) AS safety_count,
            avg(pr.pain_score) AS avg_pain
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.deep_enrichment_status = 'enriched'
          AND (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE
        GROUP BY category
        HAVING count(*) >= 5
        ORDER BY count(*) DESC
        LIMIT 10
        """
    )
    return [
        {
            "category": r["category"],
            "safety_count": r["safety_count"],
            "avg_pain": round(float(r["avg_pain"]), 1),
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


async def _recently_covered_subjects(pool, days: int = 90) -> set[str]:
    """Return category/brand names that already have a blog post in the last N days.

    Prevents the same category from dominating the blog across topic types.
    """
    rows = await pool.fetch(
        """
        SELECT DISTINCT
            LOWER(COALESCE(
                data_context->>'category',
                data_context->>'brand_a',
                ''
            )) AS subject
        FROM blog_posts
        WHERE topic_type IN ('brand_showdown','complaint_roundup','migration_report','safety_spotlight')
          AND created_at > NOW() - make_interval(days => $1)
          AND COALESCE(data_context->>'category', data_context->>'brand_a', '') != ''
        """,
        days,
    )
    return {r["subject"] for r in rows if r["subject"]}


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
    """Fetch data needed for the blueprint. Reuses existing SQL fetchers."""
    from .competitive_intelligence import (
        _fetch_brand_health,
        _fetch_competitive_flows,
        _fetch_feature_gaps,
        _fetch_safety_signals,
        _fetch_loyalty_churn,
        _fetch_sentiment_landscape,
    )
    from .complaint_analysis import _fetch_category_stats, _fetch_product_stats

    data: dict[str, Any] = {}

    if topic_type == "brand_showdown":
        brand_health, flows, sentiment, loyalty = await asyncio.gather(
            _fetch_brand_health(pool),
            _fetch_competitive_flows(pool),
            _fetch_sentiment_landscape(pool),
            _fetch_loyalty_churn(pool),
            return_exceptions=True,
        )
        data["brand_health"] = brand_health if not isinstance(brand_health, Exception) else []
        data["flows"] = flows if not isinstance(flows, Exception) else []
        data["sentiment"] = sentiment if not isinstance(sentiment, Exception) else []
        data["loyalty"] = loyalty if not isinstance(loyalty, Exception) else []

    elif topic_type == "complaint_roundup":
        cat_stats, prod_stats, feature_gaps = await asyncio.gather(
            _fetch_category_stats(pool),
            _fetch_product_stats(pool),
            _fetch_feature_gaps(pool),
            return_exceptions=True,
        )
        data["category_stats"] = cat_stats if not isinstance(cat_stats, Exception) else []
        data["product_stats"] = prod_stats if not isinstance(prod_stats, Exception) else []
        data["feature_gaps"] = feature_gaps if not isinstance(feature_gaps, Exception) else []

    elif topic_type == "migration_report":
        flows, brand_health, feature_gaps = await asyncio.gather(
            _fetch_competitive_flows(pool),
            _fetch_brand_health(pool),
            _fetch_feature_gaps(pool),
            return_exceptions=True,
        )
        data["flows"] = flows if not isinstance(flows, Exception) else []
        data["brand_health"] = brand_health if not isinstance(brand_health, Exception) else []
        data["feature_gaps"] = feature_gaps if not isinstance(feature_gaps, Exception) else []

    elif topic_type == "safety_spotlight":
        safety, brand_health, cat_stats = await asyncio.gather(
            _fetch_safety_signals(pool),
            _fetch_brand_health(pool),
            _fetch_category_stats(pool),
            return_exceptions=True,
        )
        data["safety"] = safety if not isinstance(safety, Exception) else []
        data["brand_health"] = brand_health if not isinstance(brand_health, Exception) else []
        data["category_stats"] = cat_stats if not isinstance(cat_stats, Exception) else []

    # Shared: quotable phrases for all topic types
    try:
        data["quotes"] = await _fetch_quotable_phrases(pool, topic_type, topic_ctx)
    except Exception:
        logger.warning("Failed to fetch quotable phrases, continuing without", exc_info=True)
        data["quotes"] = []

    # Data context metadata
    ctx_row = await pool.fetchrow(
        """
        SELECT
            count(*) AS total_reviews,
            count(*) FILTER (WHERE deep_enrichment_status = 'enriched') AS deep_enriched,
            min(reviewed_at)::date AS earliest,
            max(reviewed_at)::date AS latest
        FROM product_reviews
        WHERE enrichment_status = 'enriched'
        """
    )
    data["data_context"] = {
        "total_reviews_analyzed": ctx_row["total_reviews"] if ctx_row else 0,
        "deep_enriched_count": ctx_row["deep_enriched"] if ctx_row else 0,
        "review_period": (
            f"{ctx_row['earliest']} to {ctx_row['latest']}"
            if ctx_row and ctx_row["earliest"]
            else "dates unavailable"
        ),
        "report_date": str(date.today()),
    }

    # Embed subject keys so category/brand-level dedup can query data_context
    if topic_ctx.get("category"):
        data["data_context"]["category"] = topic_ctx["category"]
    if topic_ctx.get("brand_a"):
        data["data_context"]["brand_a"] = topic_ctx["brand_a"]
    if topic_ctx.get("brand_b"):
        data["data_context"]["brand_b"] = topic_ctx["brand_b"]

    return data


async def _fetch_quotable_phrases(
    pool, topic_type: str, topic_ctx: dict[str, Any]
) -> list[dict[str, Any]]:
    """Pull impactful review excerpts relevant to the topic."""
    # Build a topic-specific filter
    if topic_type == "brand_showdown":
        where = (
            "AND pm.brand IN ($1, $2)"
        )
        args = [topic_ctx["brand_a"], topic_ctx["brand_b"]]
    elif topic_type in ("complaint_roundup", "migration_report", "safety_spotlight"):
        where = """
            AND COALESCE(
                REPLACE(pm.categories->>2, '&amp;', '&'),
                REPLACE(pm.categories->>1, '&amp;', '&'),
                pr.source_category
            ) = $1
        """
        args = [topic_ctx["category"]]
    else:
        where = ""
        args = []

    query = f"""
        SELECT qp.value AS phrase, pr.asin, pm.brand, pr.rating
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin,
             jsonb_array_elements_text(pr.deep_extraction->'quotable_phrases') AS qp
        WHERE pr.deep_enrichment_status = 'enriched'
          AND jsonb_typeof(pr.deep_extraction->'quotable_phrases') = 'array'
          AND jsonb_array_length(pr.deep_extraction->'quotable_phrases') > 0
          {where}
        ORDER BY pr.pain_score DESC
        LIMIT 10
    """
    rows = await pool.fetch(query, *args)
    return [
        {
            "phrase": r["phrase"],
            "brand": r["brand"],
            "rating": r["rating"],
        }
        for r in rows
    ]


# -- Stage 3: Blueprint Construction ------------------------------

def _build_blueprint(
    topic_type: str, topic_ctx: dict[str, Any], data: dict[str, Any]
) -> PostBlueprint:
    """Build a structured post blueprint deterministically from data."""
    builder = {
        "brand_showdown": _blueprint_brand_showdown,
        "complaint_roundup": _blueprint_complaint_roundup,
        "migration_report": _blueprint_migration_report,
        "safety_spotlight": _blueprint_safety_spotlight,
    }[topic_type]
    return builder(topic_ctx, data)


def _blueprint_brand_showdown(ctx: dict, data: dict) -> PostBlueprint:
    brand_a, brand_b = ctx["brand_a"], ctx["brand_b"]
    category = ctx.get("category", "products")

    # Filter brand_health to our two brands
    bh = {
        b["brand"]: b for b in data.get("brand_health", [])
        if b["brand"] in (brand_a, brand_b)
    }
    bh_a = bh.get(brand_a, {})
    bh_b = bh.get(brand_b, {})

    # Head-to-head chart
    h2h_data = []
    for metric, key_a, key_b, label in [
        ("Pain Score", "avg_pain_score", "avg_pain_score", "Pain Score (0-10)"),
        ("Avg Rating", "avg_rating", "avg_rating", "Avg Rating (1-5)"),
        ("Critical Issues", "critical", "critical", "Critical Issues"),
    ]:
        val_a = bh_a.get(key_a) or (bh_a.get("severity_distribution", {}).get(key_a, 0))
        val_b = bh_b.get(key_b) or (bh_b.get("severity_distribution", {}).get(key_b, 0))
        h2h_data.append({"name": label, brand_a: val_a, brand_b: val_b})

    h2h_chart = ChartSpec(
        chart_id="head2head-bar",
        chart_type="horizontal_bar",
        title=f"Head-to-Head: {brand_a} vs {brand_b}",
        data=h2h_data,
        config={
            "x_key": "name",
            "bars": [
                {"dataKey": brand_a, "color": "#22d3ee"},
                {"dataKey": brand_b, "color": "#f472b6"},
            ],
        },
    )

    # Complaint distribution chart -- top root causes from category stats
    complaint_data = _build_complaint_comparison_data(brand_a, brand_b, data)
    complaint_chart = ChartSpec(
        chart_id="complaints-bar",
        chart_type="bar",
        title=f"Top Complaint Categories",
        data=complaint_data,
        config={
            "x_key": "name",
            "bars": [
                {"dataKey": brand_a, "color": "#22d3ee"},
                {"dataKey": brand_b, "color": "#f472b6"},
            ],
        },
    )

    # Migration flows chart
    flow_data = _build_flow_data(brand_a, brand_b, data)
    charts = [h2h_chart, complaint_chart]
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal="Hook the reader with a surprising stat or contrast between the brands",
            key_stats={
                "brand_a": brand_a,
                "brand_b": brand_b,
                "category": category,
                "reviews_a": ctx["reviews_a"],
                "reviews_b": ctx["reviews_b"],
                "pain_a": ctx["pain_a"],
                "pain_b": ctx["pain_b"],
                "pain_diff": ctx["pain_diff"],
            },
            data_summary=(
                f"{brand_a} has {ctx['reviews_a']} negative reviews (avg pain {ctx['pain_a']}) "
                f"vs {brand_b}'s {ctx['reviews_b']} (avg pain {ctx['pain_b']}). "
                f"Pain score difference: {ctx['pain_diff']}."
            ),
        ),
        SectionSpec(
            id="head2head",
            heading=f"{brand_a} vs {brand_b}: By the Numbers",
            goal="Present core metrics side by side with the chart",
            key_stats={
                "rating_a": ctx["rating_a"],
                "rating_b": ctx["rating_b"],
                "pain_a": ctx["pain_a"],
                "pain_b": ctx["pain_b"],
            },
            chart_ids=["head2head-bar"],
            data_summary=(
                f"{brand_a}: rating {ctx['rating_a']}, pain {ctx['pain_a']}. "
                f"{brand_b}: rating {ctx['rating_b']}, pain {ctx['pain_b']}."
            ),
        ),
        SectionSpec(
            id="complaints",
            heading="What Buyers Complain About Most",
            goal="Break down the top complaint categories per brand",
            chart_ids=["complaints-bar"],
            data_summary=f"Grouped comparison of top complaint types for {brand_a} and {brand_b}.",
        ),
    ]

    if flow_data:
        flow_chart = ChartSpec(
            chart_id="migration-bar",
            chart_type="horizontal_bar",
            title=f"Customer Migration: {brand_a} vs {brand_b}",
            data=flow_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "mentions", "color": "#22d3ee"}],
            },
        )
        charts.append(flow_chart)
        sections.append(SectionSpec(
            id="migration",
            heading="Where Are Customers Going?",
            goal="Show the migration direction between brands",
            chart_ids=["migration-bar"],
            data_summary=f"Net customer flow between {brand_a} and {brand_b}.",
        ))

    sections.append(SectionSpec(
        id="verdict",
        heading="The Verdict",
        goal="Summarize findings and declare which brand fares better in this data",
        key_stats={
            "brand_a": brand_a,
            "brand_b": brand_b,
            "pain_a": ctx["pain_a"],
            "pain_b": ctx["pain_b"],
        },
    ))

    return PostBlueprint(
        topic_type="brand_showdown",
        slug=ctx["slug"],
        suggested_title=f"{brand_a} vs {brand_b}: What {ctx['review_total']}+ Negative Reviews Reveal",
        tags=[category, brand_a.lower(), brand_b.lower(), "comparison", "reviews"],
        data_context=data["data_context"],
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_complaint_roundup(ctx: dict, data: dict) -> PostBlueprint:
    category = ctx["category"]

    # Filter product stats to this category
    products = [
        p for p in data.get("product_stats", [])
        if (p.get("category") or "").lower() == category.lower()
    ][:10]

    # Top products by pain chart
    # _fetch_product_stats returns "avg_pain_score" (not "avg_pain") and no "brand" key
    prod_chart_data = [
        {
            "name": p.get("asin", "?")[:20],
            "pain_score": p.get("avg_pain_score", 0),
            "reviews": p.get("complaint_count", 0),
        }
        for p in products[:8]
    ]
    prod_chart = ChartSpec(
        chart_id="products-bar",
        chart_type="bar",
        title=f"Highest Pain Products in {category}",
        data=prod_chart_data,
        config={
            "x_key": "name",
            "bars": [
                {"dataKey": "pain_score", "color": "#f87171"},
                {"dataKey": "reviews", "color": "#22d3ee"},
            ],
        },
    )

    # Feature gaps chart
    gaps = [
        g for g in data.get("feature_gaps", [])
        if (g.get("category") or "").lower() == category.lower()
    ][:6]
    gap_chart_data = [
        {"name": g["feature"][:30], "mentions": g["mentions"]}
        for g in gaps
    ]
    charts = [prod_chart]
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal="Highlight the scale of complaints in this category",
            key_stats={
                "category": category,
                "review_count": ctx["review_count"],
                "avg_pain": ctx["avg_pain"],
                "avg_rating": ctx["avg_rating"],
            },
            data_summary=(
                f"Analyzed {ctx['review_count']} negative reviews in {category}. "
                f"Average pain score: {ctx['avg_pain']}/10, average rating: {ctx['avg_rating']}/5."
            ),
        ),
        SectionSpec(
            id="worst_offenders",
            heading="The Products Generating the Most Complaints",
            goal="Rank the most-complained-about products",
            chart_ids=["products-bar"],
            data_summary=f"Top {len(prod_chart_data)} products by pain score in {category}.",
        ),
    ]

    if gap_chart_data:
        gap_chart = ChartSpec(
            chart_id="gaps-bar",
            chart_type="horizontal_bar",
            title=f"Most Requested Features in {category}",
            data=gap_chart_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "mentions", "color": "#a78bfa"}],
            },
        )
        charts.append(gap_chart)
        sections.append(SectionSpec(
            id="feature_gaps",
            heading="What Buyers Wish These Products Had",
            goal="List the most requested features",
            chart_ids=["gaps-bar"],
            data_summary=f"Top {len(gap_chart_data)} feature requests in {category}.",
        ))

    sections.append(SectionSpec(
        id="takeaway",
        heading="Key Takeaways",
        goal="Actionable summary for buyers considering this category",
        key_stats={"category": category, "review_count": ctx["review_count"]},
    ))

    return PostBlueprint(
        topic_type="complaint_roundup",
        slug=ctx["slug"],
        suggested_title=f"Top Complaints in {category}: What {ctx['review_count']}+ Reviews Reveal",
        tags=[category, "complaints", "reviews", "buyer-guide"],
        data_context=data["data_context"],
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_migration_report(ctx: dict, data: dict) -> PostBlueprint:
    category = ctx["category"]

    # Filter flows to this category
    all_flows = data.get("flows", [])
    # Get top migration destinations
    dest_counts: dict[str, int] = {}
    for f in all_flows:
        if f.get("direction") == "switched_to":
            dest = f.get("competitor", "")
            dest_counts[dest] = dest_counts.get(dest, 0) + f.get("mentions", 0)

    top_dests = sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    flow_chart_data = [
        {"name": name[:25], "mentions": count}
        for name, count in top_dests
    ]

    flow_chart = ChartSpec(
        chart_id="flow-bar",
        chart_type="horizontal_bar",
        title=f"Top Migration Destinations in {category}",
        data=flow_chart_data,
        config={
            "x_key": "name",
            "bars": [{"dataKey": "mentions", "color": "#34d399"}],
        },
    )

    # Source brands losing customers
    source_counts: dict[str, int] = {}
    for f in all_flows:
        if f.get("direction") == "switched_to":
            src = f.get("source_brand", "")
            source_counts[src] = source_counts.get(src, 0) + f.get("mentions", 0)

    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    source_chart_data = [
        {"name": name[:25], "lost_customers": count}
        for name, count in top_sources
    ]

    charts = [flow_chart]
    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal="Highlight the volume of customer migration in this category",
            key_stats={
                "category": category,
                "total_mentions": ctx["total_mentions"],
            },
            data_summary=(
                f"Found {ctx['total_mentions']} mentions of customers switching products "
                f"in the {category} category."
            ),
        ),
        SectionSpec(
            id="destinations",
            heading="Where Are Customers Migrating To?",
            goal="Show the top destinations customers are switching to",
            chart_ids=["flow-bar"],
            data_summary=f"Top {len(flow_chart_data)} products customers are switching to.",
        ),
    ]

    if source_chart_data:
        source_chart = ChartSpec(
            chart_id="sources-bar",
            chart_type="bar",
            title=f"Brands Losing Customers in {category}",
            data=source_chart_data,
            config={
                "x_key": "name",
                "bars": [{"dataKey": "lost_customers", "color": "#f87171"}],
            },
        )
        charts.append(source_chart)
        sections.append(SectionSpec(
            id="sources",
            heading="Which Brands Are Losing the Most Customers?",
            goal="Rank brands by outgoing migration",
            chart_ids=["sources-bar"],
            data_summary=f"Top {len(source_chart_data)} brands losing customers.",
        ))

    sections.append(SectionSpec(
        id="triggers",
        heading="What Triggers the Switch?",
        goal="Explain the common reasons behind migration",
        key_stats={"category": category},
    ))
    sections.append(SectionSpec(
        id="takeaway",
        heading="Key Takeaways",
        goal="Summary and recommendations for buyers",
        key_stats={"category": category, "total_mentions": ctx["total_mentions"]},
    ))

    return PostBlueprint(
        topic_type="migration_report",
        slug=ctx["slug"],
        suggested_title=f"Product Migration in {category}: Where {ctx['total_mentions']}+ Customers Are Switching",
        tags=[category, "migration", "competitive-analysis", "reviews"],
        data_context=data["data_context"],
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


def _blueprint_safety_spotlight(ctx: dict, data: dict) -> PostBlueprint:
    category = ctx["category"]

    # Safety by brand
    safety_rows = data.get("safety", [])
    brand_safety: dict[str, int] = {}
    consequence_dist: dict[str, int] = {}
    for s in safety_rows:
        brand_safety[s["brand"]] = brand_safety.get(s["brand"], 0) + s["count"]
        cons = s.get("consequence") or "unspecified"
        consequence_dist[cons] = consequence_dist.get(cons, 0) + s["count"]

    top_brands = sorted(brand_safety.items(), key=lambda x: x[1], reverse=True)[:8]
    brand_chart_data = [
        {"name": name[:25], "safety_flags": count}
        for name, count in top_brands
    ]

    brand_chart = ChartSpec(
        chart_id="safety-brands-bar",
        chart_type="bar",
        title=f"Safety Flags by Brand in {category}",
        data=brand_chart_data,
        config={
            "x_key": "name",
            "bars": [{"dataKey": "safety_flags", "color": "#f87171"}],
        },
    )

    # Consequence severity chart
    cons_chart_data = [
        {"name": k, "count": v}
        for k, v in sorted(consequence_dist.items(), key=lambda x: x[1], reverse=True)
    ]
    cons_chart = ChartSpec(
        chart_id="consequence-bar",
        chart_type="horizontal_bar",
        title="Safety Issues by Severity",
        data=cons_chart_data,
        config={
            "x_key": "name",
            "bars": [{"dataKey": "count", "color": "#fbbf24"}],
        },
    )

    sections = [
        SectionSpec(
            id="hook",
            heading="Introduction",
            goal="Lead with the most concerning safety signal",
            key_stats={
                "category": category,
                "safety_count": ctx["safety_count"],
                "avg_pain": ctx["avg_pain"],
            },
            data_summary=(
                f"Found {ctx['safety_count']} safety-flagged reviews in {category}. "
                f"Average pain score: {ctx['avg_pain']}/10."
            ),
        ),
        SectionSpec(
            id="brands",
            heading="Which Brands Have the Most Safety Concerns?",
            goal="Rank brands by safety flag count",
            chart_ids=["safety-brands-bar"],
            data_summary=f"Top {len(brand_chart_data)} brands by safety flags.",
        ),
        SectionSpec(
            id="severity",
            heading="How Serious Are These Issues?",
            goal="Break down by consequence severity",
            chart_ids=["consequence-bar"],
            data_summary=f"Distribution of consequence severity across {ctx['safety_count']} flagged reviews.",
        ),
        SectionSpec(
            id="takeaway",
            heading="What Buyers Should Know",
            goal="Actionable safety guidance for buyers",
            key_stats={"category": category, "safety_count": ctx["safety_count"]},
        ),
    ]

    return PostBlueprint(
        topic_type="safety_spotlight",
        slug=ctx["slug"],
        suggested_title=f"Safety Alert: {ctx['safety_count']} Flagged Reviews in {category}",
        tags=[category, "safety", "consumer-protection", "reviews"],
        data_context=data["data_context"],
        sections=sections,
        charts=[brand_chart, cons_chart],
        quotable_phrases=data.get("quotes", []),
    )


# -- Blueprint helpers ---------------------------------------------

def _build_complaint_comparison_data(
    brand_a: str, brand_b: str, data: dict
) -> list[dict[str, Any]]:
    """Build grouped bar data comparing sentiment aspects between brands."""
    sentiment = data.get("sentiment", [])
    aspects_a: dict[str, int] = {}
    aspects_b: dict[str, int] = {}
    for s in sentiment:
        if s.get("sentiment") == "negative":
            if s["brand"] == brand_a:
                aspects_a[s["aspect"]] = aspects_a.get(s["aspect"], 0) + s["count"]
            elif s["brand"] == brand_b:
                aspects_b[s["aspect"]] = aspects_b.get(s["aspect"], 0) + s["count"]

    # Merge and take top 6
    all_aspects = set(aspects_a.keys()) | set(aspects_b.keys())
    ranked = sorted(
        all_aspects,
        key=lambda a: (aspects_a.get(a, 0) + aspects_b.get(a, 0)),
        reverse=True,
    )[:6]

    return [
        {"name": asp, brand_a: aspects_a.get(asp, 0), brand_b: aspects_b.get(asp, 0)}
        for asp in ranked
    ]


def _build_flow_data(
    brand_a: str, brand_b: str, data: dict
) -> list[dict[str, Any]]:
    """Build migration flow data between two brands."""
    flows = data.get("flows", [])
    relevant = [
        f for f in flows
        if f.get("source_brand") in (brand_a, brand_b)
        and f.get("direction") == "switched_to"
    ]
    if not relevant:
        return []

    agg: dict[str, int] = {}
    for f in relevant:
        label = f"{f['source_brand']} -> {f['competitor']}"
        agg[label] = agg.get(label, 0) + f.get("mentions", 0)

    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:6]
    return [{"name": label, "mentions": count} for label, count in top]


# -- Stage 4: Content Generation ----------------------------------

def _generate_content(
    llm, blueprint: PostBlueprint, max_tokens: int
) -> dict[str, Any] | None:
    """Single LLM call: blueprint in, {title, description, content} out."""
    from ...pipelines.llm import clean_llm_output, parse_json_response
    from ...skills.registry import get_skill_registry

    skill = get_skill_registry().get("digest/blog_post_generation")
    if skill is None:
        logger.error("Skill digest/blog_post_generation not found")
        return None

    # Build the payload from blueprint
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

        # parse_json_response returns fallback dict on failure, not None
        if parsed.get("_parse_fallback"):
            logger.error("Failed to parse LLM response as JSON")
            return None

        # Validate required keys
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
    logger.info("Stored blog draft: slug=%s, id=%s", blueprint.slug, post_id)

    # Write .ts file for the frontend if ui_path is configured
    cfg = settings.external_data
    if cfg.blog_post_ui_path:
        try:
            _write_ui_post(
                cfg.blog_post_ui_path,
                blueprint,
                content,
                charts_json,
            )
        except Exception:
            logger.warning("Failed to write UI blog file", exc_info=True)
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
                logger.warning("Blog auto-deploy failed", exc_info=True)

    return post_id


def _write_ui_post(
    ui_path: str,
    blueprint: PostBlueprint,
    content: dict[str, Any],
    charts_json: list[dict[str, Any]],
) -> None:
    """Write a .ts post file and register it in index.ts."""
    from pathlib import Path

    # Resolve relative paths against project root (where main.py lives)
    blog_dir = Path(ui_path)
    if not blog_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        blog_dir = project_root / blog_dir
    if not blog_dir.is_dir():
        logger.warning("blog_post_ui_path does not exist: %s", blog_dir)
        return

    # Derive a stable filename from the slug
    slug = blueprint.slug
    filename = slug + ".ts"
    var_name = re.sub(r"[^a-zA-Z0-9]", "_", slug).strip("_")
    # camelCase: split on _ and capitalize each part after the first
    parts = var_name.split("_")
    var_name = parts[0] + "".join(p.capitalize() for p in parts[1:])
    # JS identifiers cannot start with a digit
    if var_name and var_name[0].isdigit():
        var_name = "post" + var_name[0].upper() + var_name[1:]

    # Build the .ts file content
    charts_str = json.dumps(charts_json, indent=2, default=str)
    # Escape backticks and ${} in content for template literal
    escaped_content = (
        content["content"]
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )
    # Escape single-quoted strings (backslash first, then quotes, then newlines)
    escaped_title = _escape_js_single(content["title"])
    escaped_desc = _escape_js_single(content.get("description", ""))

    ts_content = f"""import type {{ BlogPost }} from './index'

const post: BlogPost = {{
  slug: '{slug}',
  title: '{escaped_title}',
  description: '{escaped_desc}',
  date: '{date.today().isoformat()}',
  author: 'Atlas Intelligence Team',
  tags: {json.dumps(blueprint.tags)},
  topic_type: '{blueprint.topic_type}',
  charts: {charts_str},
  content: `{escaped_content}`,
}}

export default post
"""

    # Write the post file
    post_path = blog_dir / filename
    post_path.write_text(ts_content, encoding="utf-8")
    logger.info("Wrote blog UI file: %s", post_path)

    # Update index.ts: add import + array entry if not already present
    index_path = blog_dir / "index.ts"
    if not index_path.exists():
        logger.warning("index.ts not found in %s", blog_dir)
        return

    index_text = index_path.read_text(encoding="utf-8")
    import_line = f"import {var_name} from './{slug}'"

    if slug in index_text:
        logger.debug("Post %s already in index.ts, skipping", slug)
        return

    # Insert import after the last existing import line
    lines = index_text.split("\n")
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("import "):
            last_import_idx = i

    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, import_line)
    else:
        lines.insert(0, import_line)

    # Add to POSTS array -- find the closing ].sort line and insert before it
    new_text = "\n".join(lines)
    new_text = re.sub(
        r"(].sort\()",
        f"  {var_name},\n\\1",
        new_text,
        count=1,
    )

    index_path.write_text(new_text, encoding="utf-8")
    logger.info("Updated index.ts with %s", slug)


def _escape_js_single(text: str) -> str:
    """Escape a string for use inside JS single quotes."""
    return (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
