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
from ...services.scraping.sources import VERIFIED_SOURCES, parse_source_allowlist

logger = logging.getLogger("atlas.autonomous.tasks.b2b_blog_post_generation")


def _blog_source_allowlist() -> list[str]:
    """Return the configured source allowlist as a list for SQL ANY() binding."""
    return parse_source_allowlist(settings.b2b_churn.blog_source_allowlist)


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
    """Autonomous task handler: generate B2B data-backed blog posts.

    Loops up to ``max_per_run`` times, picking a fresh topic each iteration.
    All posts are stored as drafts.
    """
    cfg = settings.b2b_churn
    if not cfg.blog_post_enabled:
        return {"_skip_synthesis": "B2B blog post generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    from ...pipelines.llm import get_pipeline_llm
    from ...pipelines.notify import send_pipeline_notification

    llm = get_pipeline_llm(
        workload="synthesis",
        try_openrouter=True,
        auto_activate_ollama=False,
        openrouter_model=cfg.blog_post_openrouter_model,
    )
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"_skip_synthesis": "No LLM available for B2B blog post generation"}

    max_posts = max(1, cfg.blog_post_max_per_run)
    results: list[dict[str, Any]] = []

    # Regeneration mode: re-process existing drafts through fixed pipeline
    if cfg.blog_post_regenerate_mode:
        return await _regenerate_existing_posts(pool, llm, cfg, task, max_posts)

    for i in range(max_posts):
        topic = await _select_topic(pool, max_posts)
        if topic is None:
            logger.info("No more viable B2B topics after %d posts", i)
            break

        topic_type, topic_ctx = topic
        data = await _gather_data(pool, topic_type, topic_ctx)

        sufficiency = _check_data_sufficiency(topic_type, data)
        if not sufficiency["sufficient"]:
            logger.warning(
                "Data insufficiency for %s (%s): %s",
                topic_ctx.get("slug", "?"), topic_type, sufficiency["reason"],
            )
            continue

        blueprint = _build_blueprint(topic_type, topic_ctx, data)
        link_posts = await _fetch_related_for_linking(
            pool, blueprint.tags, blueprint.slug,
        )
        content = _generate_content(
            llm, blueprint, cfg.blog_post_max_tokens,
            related_posts=link_posts,
        )
        if content is None:
            logger.warning("LLM failed for B2B topic %s, skipping", blueprint.slug)
            continue

        # Inject affiliate links as proper markdown links
        affiliate_url = blueprint.data_context.get("affiliate_url", "")
        affiliate_slug = blueprint.data_context.get("affiliate_slug", "")
        partner_info = blueprint.data_context.get("affiliate_partner", {})
        partner_name = partner_info.get("name", "") or partner_info.get("product_name", "")
        if affiliate_slug and affiliate_url and content.get("content"):
            md_link = f"[{partner_name}]({affiliate_url})" if partner_name else affiliate_url
            content["content"] = content["content"].replace(
                f"{{{{affiliate:{affiliate_slug}}}}}",
                md_link,
            )
            # Also fix any raw affiliate URLs the LLM may have embedded directly
            # (not already inside a markdown link)
            raw = content["content"]
            if affiliate_url in raw:
                # Match the URL when NOT preceded by ]( (already a markdown link)
                raw = re.sub(
                    r'(?<!\]\()' + re.escape(affiliate_url) + r'(?!\))',
                    md_link,
                    raw,
                )
                content["content"] = raw

        post_id = await _assemble_and_store(pool, blueprint, content, llm)
        if not post_id:
            logger.info("Slug %s already published, skipping", blueprint.slug)
            continue

        n_charts = len(blueprint.charts)
        results.append({
            "post_id": str(post_id),
            "topic_type": blueprint.topic_type,
            "slug": blueprint.slug,
            "charts": n_charts,
        })

    if not results:
        return {"_skip_synthesis": "No B2B blog posts generated this run"}

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

    # Only regenerate the 10 known B2B topic types (skip consumer types like
    # safety_spotlight / migration_report that have different schemas)
    rows = await pool.fetch(
        """
        SELECT id, slug, topic_type, data_context, created_at
        FROM blog_posts
        WHERE status = 'draft'
          AND topic_type IN (
              'vendor_showdown', 'vendor_deep_dive', 'churn_report',
              'pricing_reality_check', 'vendor_alternative', 'switching_story',
              'migration_guide', 'market_landscape', 'pain_point_roundup',
              'best_fit_guide'
          )
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
        LIMIT $1
        """,
        max_posts,
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
            blueprint = _build_blueprint(topic_type, topic_ctx, data)
            link_posts = await _fetch_related_for_linking(
                pool, blueprint.tags, blueprint.slug,
            )
            content = _generate_content(
                llm, blueprint, cfg.blog_post_max_tokens,
                related_posts=link_posts,
            )
            if content is None:
                logger.warning("Regen: LLM failed for %s, skipping", slug)
                continue

            # Inject affiliate links
            affiliate_url = blueprint.data_context.get("affiliate_url", "")
            affiliate_slug = blueprint.data_context.get("affiliate_slug", "")
            partner_info = blueprint.data_context.get("affiliate_partner", {})
            partner_name = partner_info.get("name", "") or partner_info.get("product_name", "")
            if affiliate_slug and affiliate_url and content.get("content"):
                md_link = f"[{partner_name}]({affiliate_url})" if partner_name else affiliate_url
                content["content"] = content["content"].replace(
                    f"{{{{affiliate:{affiliate_slug}}}}}",
                    md_link,
                )
                raw = content["content"]
                if affiliate_url in raw:
                    raw = re.sub(
                        r'(?<!\]\()' + re.escape(affiliate_url) + r'(?!\))',
                        md_link,
                        raw,
                    )
                    content["content"] = raw

            post_id = await _assemble_and_store(pool, blueprint, content, llm)
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

async def _select_topic(pool, max_per_run: int = 1) -> tuple[str, dict[str, Any]] | None:
    """Score candidates and pick the best unwritten B2B topic."""
    today = date.today()
    month_suffix = today.strftime("%Y-%m")

    (alternatives, showdowns, churn_reports, migrations,
     deep_dives, landscapes,
     pricing_checks, switching_stories, pain_roundups, fit_guides,
    ) = await asyncio.gather(
        _find_vendor_alternative_candidates(pool),
        _find_vendor_showdown_candidates(pool),
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
    churn_reports = churn_reports if not isinstance(churn_reports, Exception) else []
    migrations = migrations if not isinstance(migrations, Exception) else []
    deep_dives = deep_dives if not isinstance(deep_dives, Exception) else []
    landscapes = landscapes if not isinstance(landscapes, Exception) else []
    pricing_checks = pricing_checks if not isinstance(pricing_checks, Exception) else []
    switching_stories = switching_stories if not isinstance(switching_stories, Exception) else []
    pain_roundups = pain_roundups if not isinstance(pain_roundups, Exception) else []
    fit_guides = fit_guides if not isinstance(fit_guides, Exception) else []

    raw_candidates: list[tuple[str, float, str, dict[str, Any]]] = []

    for alt in alternatives:
        slug = f"{_slugify(alt['vendor'])}-alternatives-{month_suffix}"
        score = alt["urgency"] * alt["review_count"]
        raw_candidates.append((slug, score, "vendor_alternative", {**alt, "slug": slug}))

    for pair in showdowns:
        slug = f"{_slugify(pair['vendor_a'])}-vs-{_slugify(pair['vendor_b'])}-{month_suffix}"
        # Weight reviews heavily — popular pairs are most interesting to readers.
        # pain_diff is a bonus, not the primary driver.
        score = (pair["total_reviews"] + pair["pain_diff"] * 50) * 1.5
        raw_candidates.append((slug, score, "vendor_showdown", {**pair, "slug": slug}))

    for cr in churn_reports:
        slug = f"{_slugify(cr['vendor'])}-churn-report-{month_suffix}"
        score = cr["negative_reviews"] * cr["avg_urgency"]
        raw_candidates.append((slug, score, "churn_report", {**cr, "slug": slug}))

    for mig in migrations:
        slug = f"migration-from-{_slugify(mig['vendor'])}-{month_suffix}"
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
        and not _vendor_keys(ctx) & covered_vendors  # no overlap with recent posts
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

    # --- Dedup layer 3: one vendor per run (pick highest-scoring topic) ---
    # Also enforce topic type diversity: max 2 of the same type per run.
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
        WHERE a.total_reviews >= 10 AND b.total_reviews >= 10
        ORDER BY (a.total_reviews + b.total_reviews) DESC
        LIMIT 80
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


async def _find_pricing_reality_check_candidates(pool) -> list[dict[str, Any]]:
    """Vendors where users complain about pricing -- bait-and-switch, hidden costs, etc."""
    sources = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT
            vendor_name AS vendor,
            product_category AS category,
            COUNT(*) AS total_reviews,
            COUNT(*) FILTER (
                WHERE enrichment->>'pain_categories' ILIKE '%pricing%'
            ) AS pricing_complaints,
            ROUND(AVG(
                CASE WHEN enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND source = ANY($1)
        GROUP BY vendor_name, product_category
        HAVING COUNT(*) FILTER (WHERE enrichment->>'pain_categories' ILIKE '%pricing%') >= 2
        ORDER BY COUNT(*) FILTER (WHERE enrichment->>'pain_categories' ILIKE '%pricing%') DESC
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
    sources = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT
            vendor_name AS from_vendor,
            product_category AS category,
            COUNT(*) AS total_reviews,
            COUNT(*) FILTER (
                WHERE (enrichment->>'urgency_score')::numeric >= 7
            ) AS high_urgency_count,
            COUNT(*) FILTER (
                WHERE review_text ILIKE '%switch%' OR review_text ILIKE '%migrat%'
                   OR review_text ILIKE '%moved to%' OR review_text ILIKE '%moving to%'
                   OR review_text ILIKE '%left for%' OR review_text ILIKE '%leaving for%'
            ) AS switch_mentions,
            ROUND(AVG(
                CASE WHEN enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND source = ANY($1)
        GROUP BY vendor_name, product_category
        HAVING COUNT(*) FILTER (
            WHERE review_text ILIKE '%switch%' OR review_text ILIKE '%migrat%'
               OR review_text ILIKE '%moved to%' OR review_text ILIKE '%moving to%'
               OR review_text ILIKE '%left for%' OR review_text ILIKE '%leaving for%'
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
    sources = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT
            product_category AS category,
            COUNT(DISTINCT vendor_name) AS vendor_count,
            COUNT(*) AS total_complaints,
            ROUND(AVG(
                CASE WHEN enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency
        FROM b2b_reviews
        WHERE enrichment_status = 'enriched'
          AND product_category IS NOT NULL AND product_category != ''
          AND source = ANY($1)
        GROUP BY product_category
        HAVING COUNT(DISTINCT vendor_name) >= 3
        ORDER BY COUNT(DISTINCT vendor_name) DESC
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
    rows = await pool.fetch(
        """
        SELECT
            pp.product_category AS category,
            COUNT(DISTINCT pp.vendor_name) AS vendor_count,
            (SELECT COUNT(*) FROM b2b_reviews br
             WHERE br.product_category = pp.product_category
               AND br.enrichment_status = 'enriched') AS total_reviews,
            MODE() WITHIN GROUP (
                ORDER BY COALESCE(
                    (SELECT key FROM jsonb_each_text(pp.typical_company_size) ORDER BY value::numeric DESC LIMIT 1),
                    'unknown'
                )
            ) AS dominant_size
        FROM b2b_product_profiles pp
        WHERE pp.product_category IS NOT NULL AND pp.product_category != ''
        GROUP BY pp.product_category
        HAVING COUNT(DISTINCT pp.vendor_name) >= 2
        ORDER BY COUNT(DISTINCT pp.vendor_name) DESC
        LIMIT 10
        """
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
            (SELECT COUNT(*) FROM b2b_reviews br WHERE br.vendor_name = pp.vendor_name) AS review_count,
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
    rows = await pool.fetch(
        """
        SELECT
            cs.product_category AS category,
            COUNT(DISTINCT cs.vendor_name) AS vendor_count,
            SUM(cs.total_reviews) AS total_reviews,
            ROUND(AVG(cs.avg_urgency_score)::numeric, 1) AS avg_urgency
        FROM b2b_churn_signals cs
        WHERE cs.product_category IS NOT NULL AND cs.product_category != ''
        GROUP BY cs.product_category
        HAVING COUNT(DISTINCT cs.vendor_name) >= 2
        ORDER BY COUNT(DISTINCT cs.vendor_name) DESC, SUM(cs.total_reviews) DESC
        LIMIT 10
        """
    )
    return [
        {
            "category": r["category"],
            "vendor_count": r["vendor_count"],
            "total_reviews": r["total_reviews"],
            "avg_urgency": float(r["avg_urgency"]),
        }
        for r in rows
    ]


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
        SELECT LOWER(vendor_name) AS vn, COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE vendor_name = ANY($1) AND source = ANY($2)
        GROUP BY LOWER(vendor_name)
        """,
        list(vendor_set), sources,
    )
    return {r["vn"]: r["cnt"] for r in rows}


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

    elif topic_type == "pricing_reality_check":
        vendor = topic_ctx["vendor"]
        profile, signals = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_churn_signals(pool, vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = signals if not isinstance(signals, Exception) else []
        # Pull actual pricing complaint reviews directly
        sources = _blog_source_allowlist()
        pricing_reviews = await pool.fetch(
            """
            SELECT review_text, vendor_name, reviewer_title, rating,
                   enrichment->>'urgency_score' AS urgency,
                   enrichment->>'pain_categories' AS pains
            FROM b2b_reviews
            WHERE vendor_name = $1 AND enrichment_status = 'enriched'
              AND enrichment->>'pain_categories' ILIKE '%pricing%'
              AND source = ANY($2)
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT 10
            """,
            vendor, sources,
        )
        data["pricing_reviews"] = [
            {
                "text": r["review_text"][:300],
                "vendor": r["vendor_name"],
                "role": r["reviewer_title"],
                "rating": float(r["rating"]) if r["rating"] else None,
                "urgency": float(r["urgency"]) if r["urgency"] else 0,
            }
            for r in pricing_reviews
        ]
        # Also pull positive reviews for balance
        positive_reviews = await pool.fetch(
            """
            SELECT review_text, reviewer_title, rating
            FROM b2b_reviews
            WHERE vendor_name = $1 AND enrichment_status = 'enriched'
              AND rating >= 4
              AND source = ANY($2)
            ORDER BY rating DESC
            LIMIT 5
            """,
            vendor, sources,
        )
        data["positive_reviews"] = [
            {"text": r["review_text"][:300], "role": r["reviewer_title"], "rating": float(r["rating"]) if r["rating"] else None}
            for r in positive_reviews
        ]

    elif topic_type == "switching_story":
        vendor = topic_ctx["from_vendor"]
        profile, signals = await asyncio.gather(
            _fetch_product_profile(pool, vendor),
            _fetch_churn_signals(pool, vendor),
            return_exceptions=True,
        )
        data["profile"] = profile if not isinstance(profile, Exception) else {}
        data["signals"] = signals if not isinstance(signals, Exception) else []
        # Pull actual switching reviews
        sources = _blog_source_allowlist()
        switch_reviews = await pool.fetch(
            """
            SELECT review_text, vendor_name, reviewer_title, rating,
                   enrichment->>'urgency_score' AS urgency
            FROM b2b_reviews
            WHERE vendor_name = $1 AND enrichment_status = 'enriched'
              AND (review_text ILIKE '%switch%' OR review_text ILIKE '%migrat%'
                   OR review_text ILIKE '%moved to%' OR review_text ILIKE '%moving to%'
                   OR review_text ILIKE '%left for%' OR review_text ILIKE '%leaving for%')
              AND source = ANY($2)
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT 10
            """,
            vendor, sources,
        )
        data["switch_reviews"] = [
            {
                "text": r["review_text"][:400],
                "vendor": r["vendor_name"],
                "role": r["reviewer_title"],
                "rating": float(r["rating"]) if r["rating"] else None,
                "urgency": float(r["urgency"]) if r["urgency"] else 0,
            }
            for r in switch_reviews
        ]
        data["quotes"] = data["switch_reviews"]

    elif topic_type == "pain_point_roundup":
        category = topic_ctx["category"]
        # Per-vendor pain breakdown from raw reviews
        sources = _blog_source_allowlist()
        vendor_pains = await pool.fetch(
            """
            SELECT
                vendor_name,
                COUNT(*) AS review_count,
                enrichment->>'pain_categories' AS pains,
                ROUND(AVG(
                    CASE WHEN enrichment->>'urgency_score' ~ '^[0-9]'
                         THEN (enrichment->>'urgency_score')::numeric ELSE NULL END
                )::numeric, 1) AS avg_urgency
            FROM b2b_reviews
            WHERE product_category = $1 AND enrichment_status = 'enriched'
              AND source = ANY($2)
            GROUP BY vendor_name, enrichment->>'pain_categories'
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
                    "vendor": vn, "review_count": 0, "top_pain": "other",
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
        vendor_rows = await pool.fetch(
            "SELECT DISTINCT vendor_name FROM b2b_product_profiles WHERE product_category = $1",
            category,
        )
        profiles = []
        for vr in vendor_rows[:10]:
            vn = vr["vendor_name"]
            p = await _fetch_product_profile(pool, vn)
            if p:
                # Also pull avg rating from raw reviews
                rating_row = await pool.fetchrow(
                    "SELECT ROUND(AVG(rating)::numeric, 1) AS avg_rating, COUNT(*) AS cnt FROM b2b_reviews WHERE vendor_name = $1 AND rating IS NOT NULL AND source = ANY($2)",
                    vn, sources,
                )
                profiles.append({
                    "vendor": vn,
                    "profile": p,
                    "avg_rating": float(rating_row["avg_rating"]) if rating_row and rating_row["avg_rating"] else None,
                    "review_count": rating_row["cnt"] if rating_row else 0,
                })
        data["vendor_profiles"] = profiles
        quotes = await _fetch_quotable_reviews(pool, category=category)
        data["quotes"] = quotes if not isinstance(quotes, Exception) else []

    elif topic_type == "vendor_deep_dive":
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
        # Fetch all vendors in this category
        vendor_rows = await pool.fetch(
            "SELECT DISTINCT vendor_name FROM b2b_churn_signals WHERE product_category = $1",
            category,
        )
        vendor_names = [r["vendor_name"] for r in vendor_rows]
        profiles = []
        signals_list = []
        for vn in vendor_names[:10]:
            try:
                p = await _fetch_product_profile(pool, vn)
                s = await _fetch_churn_signals(pool, vn)
                profiles.append({"vendor": vn, "profile": p})
                signals_list.append({"vendor": vn, "signals": s})
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
            "SELECT DISTINCT vendor_name FROM b2b_reviews WHERE product_category = $1 AND source = ANY($2)",
            topic_ctx["category"], ctx_sources,
        )
        vendor_names = [r["vendor_name"] for r in cat_rows]

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
            WHERE vendor_name = ANY($1) AND source = ANY($2)
            """,
            vendor_names, ctx_sources,
        )
    else:
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


async def _fetch_pain_category_urgency(pool, vendor_name: str) -> dict[str, float]:
    """Query per-category average urgency directly from b2b_reviews."""
    rows = await pool.fetch(
        """
        SELECT enrichment->>'pain_category' AS pain_cat,
               AVG((enrichment->>'urgency_score')::numeric) AS avg_urg,
               COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE vendor_name = $1 AND enrichment_status = 'enriched'
          AND enrichment->>'pain_category' IS NOT NULL
        GROUP BY enrichment->>'pain_category'
        """,
        vendor_name,
    )
    return {
        r["pain_cat"]: round(float(r["avg_urg"]), 1)
        for r in rows
        if r["pain_cat"] and r["avg_urg"] is not None
    }


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

    # Fetch per-category urgency from raw reviews (fixes broken chart data
    # where every category showed the same vendor-level average)
    per_cat_urgency = await _fetch_pain_category_urgency(pool, vendor_name)
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
    if vendor_name:
        rows = await pool.fetch(
            """
            SELECT review_text, vendor_name, reviewer_title, rating,
                   enrichment->>'urgency_score' AS urgency,
                   source
            FROM b2b_reviews
            WHERE vendor_name = $1
              AND enrichment_status = 'enriched'
              AND source = ANY($2)
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
            LIMIT $3
            """,
            vendor_name, sources, limit,
        )
    elif category:
        rows = await pool.fetch(
            """
            SELECT review_text, vendor_name, reviewer_title, rating,
                   enrichment->>'urgency_score' AS urgency,
                   source
            FROM b2b_reviews
            WHERE product_category = $1
              AND enrichment_status = 'enriched'
              AND source = ANY($2)
            ORDER BY (enrichment->>'urgency_score')::numeric DESC NULLS LAST
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
            "vendor": r["vendor_name"],
            "urgency": urg,
            "role": r["reviewer_title"],
            "source_name": r["source"],
            "sentiment": "negative",
        })
    return results


async def _fetch_positive_quotes(
    pool, vendor_name: str | None, category: str | None,
    sources: list[str], limit: int = 6,
) -> list[dict[str, Any]]:
    """High-rated reviews from raw columns (no enrichment required)."""
    if vendor_name:
        rows = await pool.fetch(
            """
            SELECT COALESCE(pros, review_text) AS text, vendor_name,
                   reviewer_title, rating, source
            FROM b2b_reviews
            WHERE vendor_name = $1
              AND rating >= 4
              AND source = ANY($2)
              AND COALESCE(pros, review_text) IS NOT NULL
              AND LENGTH(COALESCE(pros, review_text)) > 20
            ORDER BY rating DESC, imported_at DESC
            LIMIT $3
            """,
            vendor_name, sources, limit,
        )
    elif category:
        rows = await pool.fetch(
            """
            SELECT COALESCE(pros, review_text) AS text, vendor_name,
                   reviewer_title, rating, source
            FROM b2b_reviews
            WHERE product_category = $1
              AND rating >= 4
              AND source = ANY($2)
              AND COALESCE(pros, review_text) IS NOT NULL
              AND LENGTH(COALESCE(pros, review_text)) > 20
            ORDER BY rating DESC, imported_at DESC
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
            "vendor": r["vendor_name"],
            "urgency": 0.0,
            "role": r["reviewer_title"],
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



async def _fetch_source_distribution(pool, vendor_names: list[str]) -> dict[str, Any]:
    """Return review counts by source platform for the given vendors."""
    if not vendor_names:
        return {"sources": [], "verified_count": 0, "community_count": 0}
    allowed = _blog_source_allowlist()
    rows = await pool.fetch(
        """
        SELECT COALESCE(source, 'unknown') AS src, COUNT(*) AS cnt
        FROM b2b_reviews
        WHERE vendor_name = ANY($1) AND enrichment_status = 'enriched'
          AND source = ANY($2)
        GROUP BY source
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

    # Universal: at least 2 quotable reviews
    # (pricing_reality_check builds quotes from pricing_reviews in the blueprint)
    if topic_type != "pricing_reality_check" and len(quotes) < 2:
        return {"sufficient": False, "reason": f"Only {len(quotes)} quotable reviews (need 2+)"}

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
        if len(vendor_profiles) < 2:
            return {"sufficient": False, "reason": f"Only {len(vendor_profiles)} vendor profiles (need 2+)"}

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


def _blueprint_vendor_deep_dive(ctx: dict, data: dict) -> PostBlueprint:
    """In-depth profile of a single vendor -- showcase data gathering capabilities."""
    vendor = ctx["vendor"]
    category = ctx.get("category", "software")
    profile = data.get("profile", {})
    signals = data.get("signals", [])
    competitor_profiles = data.get("competitor_profiles", [])

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

    # Integrations and use cases
    integrations = profile.get("integrations", [])
    use_cases = profile.get("use_cases", [])
    if integrations or use_cases:
        sections.append(SectionSpec(
            id="ecosystem",
            heading=f"The {vendor} Ecosystem: Integrations & Use Cases",
            goal="Show the product ecosystem and typical deployment scenarios",
            key_stats={
                "integrations": [str(i)[:30] for i in integrations[:8]] if isinstance(integrations, list) else [],
                "use_cases": [str(u)[:40] for u in use_cases[:6]] if isinstance(use_cases, list) else [],
            },
            data_summary=f"{len(integrations)} integrations and {len(use_cases)} primary use cases.",
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

    sections.append(SectionSpec(
        id="verdict",
        heading=f"The Bottom Line on {vendor}",
        goal="Synthesize all data into actionable guidance for potential buyers",
        key_stats={"vendor": vendor, "review_count": ctx["review_count"]},
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
            },
            data_summary=(
                f"The {category} landscape has {vendor_count} major vendors "
                f"with {ctx['total_reviews']} total churn signals analyzed."
            ),
        ),
    ]

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

    # Per-vendor breakdowns
    for vp in vendor_profiles[:5]:
        vendor = vp["vendor"]
        profile = vp.get("profile", {})
        strengths = profile.get("strengths", [])
        weaknesses = profile.get("weaknesses", [])
        if strengths or weaknesses:
            sections.append(SectionSpec(
                id=f"vendor-{_slugify(vendor)}",
                heading=f"{vendor}: Strengths & Weaknesses",
                goal=f"Brief profile of {vendor} in the {category} space",
                key_stats={
                    "vendor": vendor,
                    "strengths": [str(s.get("area", s)) if isinstance(s, dict) else str(s) for s in strengths[:3]],
                    "weaknesses": [str(w.get("area", w)) if isinstance(w, dict) else str(w) for w in weaknesses[:3]],
                },
            ))

    sections.append(SectionSpec(
        id="takeaway",
        heading=f"Choosing the Right {category} Platform",
        goal="Synthesize the landscape and help readers pick the right tool",
        key_stats={"category": category, "vendor_count": vendor_count},
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

    sections.append(SectionSpec(
        id="bottom_line",
        heading="The Bottom Line: Is It Worth the Price?",
        goal="Honest verdict -- who should pay for it and who should look elsewhere",
        key_stats={"vendor": vendor, "pricing_complaints": ctx["pricing_complaints"]},
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

    sections.append(SectionSpec(
        id="verdict",
        heading="Should You Stay or Switch?",
        goal="Honest framework for making the decision -- not everyone should switch",
        key_stats={"vendor": vendor, "avg_urgency": ctx["avg_urgency"]},
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

    sections.append(SectionSpec(
        id="takeaway",
        heading="Every Tool Has a Flaw -- Pick the One You Can Live With",
        goal="Honest summary -- there's no perfect tool, help readers pick the right trade-off",
        key_stats={"category": category, "vendor_count": ctx["vendor_count"]},
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
        ))

    # Per-vendor recommendation sections
    for vp in vendor_profiles[:6]:
        profile = vp.get("profile", {})
        company_size = profile.get("typical_company_size", {})
        size_str = ", ".join(f"{k}" for k, v in sorted(company_size.items(), key=lambda x: x[1], reverse=True)[:2]) if isinstance(company_size, dict) and company_size else "all sizes"
        strengths = profile.get("strengths", [])
        weaknesses = profile.get("weaknesses", [])
        sections.append(SectionSpec(
            id=f"vendor-{_slugify(vp['vendor'])}",
            heading=f"{vp['vendor']}: Best For {size_str} Teams",
            goal=f"Honest recommendation -- who should use {vp['vendor']} and who shouldn't",
            key_stats={
                "vendor": vp["vendor"],
                "company_size": size_str,
                "avg_rating": vp.get("avg_rating"),
                "strengths": [str(s.get("area", s)) if isinstance(s, dict) else str(s) for s in strengths[:3]],
                "weaknesses": [str(w.get("area", w)) if isinstance(w, dict) else str(w) for w in weaknesses[:3]],
            },
        ))

    sections.append(SectionSpec(
        id="decision_framework",
        heading="How to Actually Choose",
        goal="Give a clear decision framework based on budget, team size, and must-have features",
        key_stats={"category": category, "vendor_count": ctx["vendor_count"]},
    ))

    return PostBlueprint(
        topic_type="best_fit_guide",
        slug=ctx["slug"],
        suggested_title=f"Best {category} for Your Team Size: A Guide Based on {ctx['total_reviews']} Reviews",
        tags=[category.lower(), "buyers-guide", "comparison", "honest-review", "team-size"],
        data_context={**data.get("data_context", {}), "category": category},
        sections=sections,
        charts=charts,
        quotable_phrases=data.get("quotes", []),
    )


# -- Stage 4: Content Generation ----------------------------------

def _generate_content(
    llm, blueprint: PostBlueprint, max_tokens: int,
    related_posts: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Single LLM call: blueprint in, {title, description, content} out."""
    from ...pipelines.llm import clean_llm_output, parse_json_response
    from ...skills.registry import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_blog_post_generation")
    if skill is None:
        logger.error("Skill digest/b2b_blog_post_generation not found")
        return None

    payload: dict[str, Any] = {
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
    if related_posts:
        payload["related_posts"] = related_posts

    from ...services.protocols import Message

    messages = [
        Message(role="system", content=skill.content),
        Message(role="user", content=json.dumps(payload, separators=(",", ":"), default=str)),
    ]

    try:
        result = llm.chat(messages=messages, max_tokens=max_tokens, temperature=0.7)
        _usage = result.get("usage", {}) if isinstance(result, dict) else {}
        if _usage.get("input_tokens"):
            logger.info("b2b_blog_post_generation LLM tokens: in=%d out=%d",
                         _usage["input_tokens"], _usage.get("output_tokens", 0))
            from ...pipelines.llm import trace_llm_call
            trace_llm_call("task.b2b_blog_post_generation", input_tokens=_usage["input_tokens"],
                           output_tokens=_usage.get("output_tokens", 0),
                           model=getattr(llm, "model", ""), provider=getattr(llm, "name", ""))
        text = result.get("response", "") if isinstance(result, dict) else str(result)
        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)

        if parsed.get("_parse_fallback"):
            logger.error("Failed to parse LLM response as JSON")
            return None

        if not all(k in parsed for k in ("title", "description", "content")):
            logger.error("LLM response missing required keys: %s", list(parsed.keys()))
            return None

        # Ensure SEO fields have sane defaults if LLM didn't produce them
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

        return parsed
    except Exception:
        logger.exception("LLM content generation failed")
        return None


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
            status, llm_model, source_report_date,
            seo_title, seo_description, target_keyword,
            secondary_keywords, faq
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'draft',$9,$10,$11,$12,$13,$14,$15)
        ON CONFLICT (slug) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            content = EXCLUDED.content,
            charts = EXCLUDED.charts,
            data_context = EXCLUDED.data_context,
            llm_model = EXCLUDED.llm_model,
            source_report_date = EXCLUDED.source_report_date,
            seo_title = EXCLUDED.seo_title,
            seo_description = EXCLUDED.seo_description,
            target_keyword = EXCLUDED.target_keyword,
            secondary_keywords = EXCLUDED.secondary_keywords,
            faq = EXCLUDED.faq
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
        content.get("seo_title", content["title"][:60]),
        content.get("seo_description", content.get("description", "")[:155]),
        content.get("target_keyword", ""),
        json.dumps(content.get("secondary_keywords", []), default=str),
        json.dumps(content.get("faq", []), default=str),
    )
    if not row:
        logger.warning(
            "Skipped overwrite of published post: slug=%s", blueprint.slug
        )
        return ""
    post_id = str(row["id"])
    logger.info("Stored B2B blog draft: slug=%s, id=%s", blueprint.slug, post_id)

    # Compute related posts (same category/vendor overlap)
    related: list[str] = []
    try:
        related = await _compute_related_slugs(pool, blueprint.slug, blueprint.tags)
        if related:
            await pool.execute(
                "UPDATE blog_posts SET related_slugs = $1 WHERE id = $2",
                json.dumps(related), row["id"],
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
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE enrichment_status = 'enriched') AS enriched,
            COUNT(*) FILTER (WHERE rating IS NOT NULL AND rating < 3) AS negative,
            ROUND(AVG(
                CASE WHEN enrichment->>'urgency_score' ~ '^[0-9]'
                     THEN (enrichment->>'urgency_score')::numeric ELSE NULL END
            )::numeric, 1) AS avg_urgency,
            MODE() WITHIN GROUP (ORDER BY product_category) AS category
        FROM b2b_reviews
        WHERE LOWER(vendor_name) = LOWER($1)
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


async def build_manual_topic_ctx(
    pool,
    vendor_name: str,
    topic_type: str,
    vendor_b: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """Construct topic_ctx for a manually requested blog post.

    Bypasses _select_topic() dedup -- always builds context even if a post
    for this vendor+month already exists.
    """
    month_suffix = date.today().strftime("%Y-%m")
    stats = await _fetch_vendor_stats(pool, vendor_name)
    if not category:
        category = stats.get("category", "software") or "software"

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
            "vendor_count": 0,
            "total_reviews": stats.get("total", 0),
            "avg_urgency": stats.get("avg_urgency", 0),
            "slug": slug,
        })
    elif topic_type == "pain_point_roundup":
        slug = f"top-complaint-every-{_slugify(category)}-{month_suffix}"
        ctx.update({
            "vendor_count": 0,
            "total_complaints": stats.get("negative", 0),
            "avg_urgency": stats.get("avg_urgency", 0),
            "slug": slug,
        })
    elif topic_type == "best_fit_guide":
        slug = f"best-{_slugify(category)}-for-teams-{month_suffix}"
        ctx.update({
            "vendor_count": 0,
            "total_reviews": stats.get("total", 0),
            "company_size": "small-teams",
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
        slug = f"migration-from-{_slugify(vendor_name)}-{month_suffix}"
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
