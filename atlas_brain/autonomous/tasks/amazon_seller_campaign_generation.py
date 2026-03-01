"""
Amazon Seller Intelligence campaign generation.

Aggregates category intelligence from consumer review data (product_reviews,
brand_intelligence), then generates personalized outreach content targeting
Amazon sellers using the amazon_seller_campaign_generation skill.

Campaigns are stored in b2b_campaigns with target_mode='amazon_seller' and
reuse the existing send/sequence/audit infrastructure.

Returns _skip_synthesis.
"""

import json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from .campaign_audit import log_campaign_event

logger = logging.getLogger("atlas.autonomous.tasks.amazon_seller_campaign_generation")


# ------------------------------------------------------------------
# Category intelligence aggregation
# ------------------------------------------------------------------


async def _aggregate_category_intelligence(pool, category: str) -> dict[str, Any] | None:
    """Build a category intelligence snapshot from product_reviews data.

    Returns a dict matching the skill input schema, or None if insufficient data.
    """
    cfg = settings.seller_campaign

    # Total reviews + products + brands in this category
    stats = await pool.fetchrow(
        """
        SELECT
            COUNT(*) AS total_reviews,
            COUNT(DISTINCT pr.asin) AS total_products,
            COUNT(DISTINCT pm.brand) AS total_brands
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.source_category = $1
        """,
        category,
    )
    if not stats or stats["total_reviews"] < cfg.min_reviews_per_category:
        return None

    # Brand health: compute from product_reviews + product_metadata
    # (mirrors consumer_dashboard.py brand health logic)
    brand_rows = await pool.fetch(
        """
        SELECT
            pm.brand,
            COUNT(*) AS total_reviews,
            ROUND(AVG(pr.rating), 2) AS avg_rating,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction->>'would_repurchase' = 'true'
            ) AS repurchase_yes,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction->>'would_repurchase' = 'false'
            ) AS repurchase_no,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
            ) AS safety_count,
            COUNT(*) FILTER (
                WHERE pr.deep_extraction IS NOT NULL
                  AND pr.deep_extraction != '{}'::jsonb
            ) AS deep_count
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.source_category = $1
          AND pm.brand IS NOT NULL AND pm.brand != ''
        GROUP BY pm.brand
        HAVING COUNT(*) >= 5
        ORDER BY COUNT(*) DESC
        LIMIT 20
        """,
        category,
    )

    # Top pain points: root_cause aggregation from enriched reviews
    pain_rows = await pool.fetch(
        """
        SELECT root_cause AS complaint,
               COUNT(*) AS count,
               MAX(severity) AS severity
        FROM product_reviews
        WHERE source_category = $1
          AND enrichment_status = 'enriched'
          AND root_cause IS NOT NULL AND root_cause != ''
          AND rating <= 3
        GROUP BY root_cause
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        category,
    )
    top_pain_points = [
        {
            "complaint": r["complaint"],
            "count": r["count"],
            "severity": r["severity"] or "medium",
            "affected_brands": 0,
        }
        for r in pain_rows
    ]

    # Feature gaps from deep_extraction->'feature_requests'
    # Elements can be strings or objects with a 'request' field
    feature_rows = await pool.fetch(
        """
        SELECT req AS request,
               COUNT(*) AS count,
               COUNT(DISTINCT asin) AS brand_count,
               ROUND(AVG(rating), 1) AS avg_rating
        FROM (
            SELECT asin, rating,
                   CASE jsonb_typeof(elem)
                        WHEN 'string' THEN elem #>> '{}'
                        WHEN 'object' THEN elem ->> 'request'
                        ELSE elem #>> '{}'
                   END AS req
            FROM product_reviews,
                 jsonb_array_elements(
                     CASE jsonb_typeof(deep_extraction->'feature_requests')
                          WHEN 'array' THEN deep_extraction->'feature_requests'
                          ELSE '[]'::jsonb
                     END
                 ) AS elem
            WHERE source_category = $1
              AND deep_enrichment_status = 'enriched'
              AND deep_extraction->'feature_requests' IS NOT NULL
        ) sub
        WHERE req IS NOT NULL AND req != '' AND req != 'null'
        GROUP BY req
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(*) DESC
        LIMIT 15
        """,
        category,
    )
    feature_gaps = [
        {
            "request": r["request"],
            "count": r["count"],
            "brand_count": r["brand_count"],
            "avg_rating": float(r["avg_rating"]) if r["avg_rating"] else 0,
        }
        for r in feature_rows
    ]

    # Competitive flows from deep_extraction->'product_comparisons'
    flow_rows = await pool.fetch(
        """
        SELECT
            comp->>'product_name' AS from_brand,
            comp->>'product' AS to_brand,
            COALESCE(comp->>'direction', 'compared') AS direction,
            COUNT(*) AS count
        FROM product_reviews,
             jsonb_array_elements(
                 CASE jsonb_typeof(deep_extraction->'product_comparisons')
                      WHEN 'array' THEN deep_extraction->'product_comparisons'
                      ELSE '[]'::jsonb
                 END
             ) AS comp
        WHERE source_category = $1
          AND deep_enrichment_status = 'enriched'
          AND deep_extraction->'product_comparisons' IS NOT NULL
        GROUP BY comp->>'product_name', comp->>'product', comp->>'direction'
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(*) DESC
        LIMIT 15
        """,
        category,
    )
    competitive_flows = [
        {
            "from_brand": r["from_brand"] or "",
            "to_brand": r["to_brand"] or "",
            "direction": r["direction"],
            "count": r["count"],
        }
        for r in flow_rows
        if r["from_brand"] or r["to_brand"]
    ]

    # Brand health: compute score from repurchase + safety rates
    brand_health = []
    for br in (brand_rows or [])[:10]:
        yes = br["repurchase_yes"] or 0
        no = br["repurchase_no"] or 0
        repurchase_total = yes + no
        repurchase_rate = yes / repurchase_total if repurchase_total > 0 else 0.5

        deep = br["deep_count"] or 0
        safety = br["safety_count"] or 0
        safety_rate = max(0, 1.0 - (safety / deep) * 10) if deep >= 5 else 1.0

        hs = round((repurchase_rate + safety_rate) / 2 * 100, 1)
        brand_health.append({
            "brand": br["brand"],
            "health_score": hs,
            "trend": "rising" if hs >= 70 else ("falling" if hs < 40 else "stable"),
            "review_count": br["total_reviews"] or 0,
        })

    # Safety signals
    safety_rows = await pool.fetch(
        """
        SELECT
            COALESCE(pm.brand, pr.asin) AS brand,
            pr.deep_extraction->'safety_flag'->>'category' AS category,
            pr.deep_extraction->'safety_flag'->>'description' AS description,
            COUNT(*) AS flagged_count
        FROM product_reviews pr
        LEFT JOIN product_metadata pm ON pm.asin = pr.asin
        WHERE pr.source_category = $1
          AND pr.deep_enrichment_status = 'enriched'
          AND pr.deep_extraction->'safety_flag'->>'flagged' = 'true'
        GROUP BY COALESCE(pm.brand, pr.asin),
                 pr.deep_extraction->'safety_flag'->>'category',
                 pr.deep_extraction->'safety_flag'->>'description'
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        category,
    )
    safety_signals = [
        {
            "brand": r["brand"],
            "category": r["category"] or "",
            "description": r["description"] or "",
            "flagged_count": r["flagged_count"],
        }
        for r in safety_rows
    ]

    # Manufacturing insights
    mfg_rows = await pool.fetch(
        """
        SELECT manufacturing_suggestion AS suggestion,
               COUNT(*) AS count,
               ARRAY_AGG(DISTINCT asin) AS affected_asins
        FROM product_reviews
        WHERE source_category = $1
          AND enrichment_status = 'enriched'
          AND actionable_for_manufacturing = TRUE
          AND manufacturing_suggestion IS NOT NULL
          AND manufacturing_suggestion != ''
        GROUP BY manufacturing_suggestion
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        category,
    )
    manufacturing_insights = [
        {
            "suggestion": r["suggestion"],
            "count": r["count"],
            "affected_asins": (r["affected_asins"] or [])[:5],
        }
        for r in mfg_rows
    ]

    # Top root causes
    cause_rows = await pool.fetch(
        """
        SELECT root_cause AS cause, COUNT(*) AS count
        FROM product_reviews
        WHERE source_category = $1
          AND enrichment_status = 'enriched'
          AND root_cause IS NOT NULL AND root_cause != ''
        GROUP BY root_cause
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """,
        category,
    )
    top_root_causes = [
        {"cause": r["cause"], "count": r["count"]}
        for r in cause_rows
    ]

    return {
        "category": category,
        "category_stats": {
            "total_reviews": stats["total_reviews"],
            "total_brands": len(brand_health) or stats["total_brands"],
            "total_products": stats["total_products"],
            "date_range": "all available data",
        },
        "top_pain_points": top_pain_points,
        "feature_gaps": feature_gaps,
        "competitive_flows": competitive_flows,
        "brand_health": brand_health,
        "safety_signals": safety_signals,
        "manufacturing_insights": manufacturing_insights,
        "top_root_causes": top_root_causes,
    }


async def _save_intelligence_snapshot(pool, intel: dict[str, Any]) -> None:
    """Cache category intelligence snapshot for reuse by sequence progression."""
    await pool.execute(
        """
        INSERT INTO category_intelligence_snapshots (
            category, total_reviews, total_brands, total_products,
            top_pain_points, feature_gaps, competitive_flows,
            brand_health, safety_signals, manufacturing_insights,
            top_root_causes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (category, snapshot_date) DO UPDATE SET
            total_reviews = EXCLUDED.total_reviews,
            total_brands = EXCLUDED.total_brands,
            total_products = EXCLUDED.total_products,
            top_pain_points = EXCLUDED.top_pain_points,
            feature_gaps = EXCLUDED.feature_gaps,
            competitive_flows = EXCLUDED.competitive_flows,
            brand_health = EXCLUDED.brand_health,
            safety_signals = EXCLUDED.safety_signals,
            manufacturing_insights = EXCLUDED.manufacturing_insights,
            top_root_causes = EXCLUDED.top_root_causes
        """,
        intel["category"],
        intel["category_stats"]["total_reviews"],
        intel["category_stats"]["total_brands"],
        intel["category_stats"]["total_products"],
        json.dumps(intel["top_pain_points"]),
        json.dumps(intel["feature_gaps"]),
        json.dumps(intel["competitive_flows"]),
        json.dumps(intel["brand_health"]),
        json.dumps(intel["safety_signals"]),
        json.dumps(intel["manufacturing_insights"]),
        json.dumps(intel["top_root_causes"]),
    )


# ------------------------------------------------------------------
# LLM content generation
# ------------------------------------------------------------------


async def _generate_content(
    llm,
    system_prompt: str,
    payload: dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any] | None:
    """Call LLM with campaign generation skill and parse JSON response."""
    from ...pipelines.llm import clean_llm_output
    from ...services.protocols import Message

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=json.dumps(payload, indent=2, default=str)),
    ]

    text = ""
    try:
        result = llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = result.get("response", "").strip()
        if not text:
            return None

        text = clean_llm_output(text)
        parsed = json.loads(text)

        if not isinstance(parsed, dict) or "body" not in parsed:
            logger.debug("Campaign generation missing 'body' field")
            return None

        return parsed

    except json.JSONDecodeError:
        logger.debug("Failed to parse campaign JSON: %.200s", text)
        return None
    except Exception:
        logger.exception("Campaign generation LLM call failed")
        return None


# ------------------------------------------------------------------
# Seller target management
# ------------------------------------------------------------------


async def _fetch_seller_targets(
    pool,
    category: str | None = None,
) -> list[dict]:
    """Fetch active seller targets, optionally filtered by category."""
    if category:
        rows = await pool.fetch(
            """
            SELECT id, seller_name, company_name, email, seller_type,
                   categories, storefront_url, notes
            FROM seller_targets
            WHERE status = 'active'
              AND $1 = ANY(categories)
            ORDER BY created_at ASC
            """,
            category,
        )
    else:
        rows = await pool.fetch(
            """
            SELECT id, seller_name, company_name, email, seller_type,
                   categories, storefront_url, notes
            FROM seller_targets
            WHERE status = 'active'
            ORDER BY created_at ASC
            """
        )
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Campaign generation core
# ------------------------------------------------------------------


async def generate_campaigns(
    pool,
    category_filter: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Core generation logic.

    1. Get categories with sufficient data.
    2. For each category, aggregate intelligence.
    3. For each seller target in that category, generate multi-channel content.
    4. Store as drafts in b2b_campaigns with target_mode='amazon_seller'.
    5. Create campaign sequences for cold emails.
    """
    cfg = settings.seller_campaign
    max_categories = limit or cfg.max_campaigns_per_run

    # 1. Find categories with enough reviews
    if category_filter:
        cat_rows = await pool.fetch(
            """
            SELECT source_category, COUNT(*) AS cnt
            FROM product_reviews
            WHERE source_category = $1
            GROUP BY source_category
            HAVING COUNT(*) >= $2
            """,
            category_filter, cfg.min_reviews_per_category,
        )
    else:
        cat_rows = await pool.fetch(
            """
            SELECT source_category, COUNT(*) AS cnt
            FROM product_reviews
            WHERE source_category IS NOT NULL AND source_category != ''
            GROUP BY source_category
            HAVING COUNT(*) >= $1
            ORDER BY COUNT(*) DESC
            LIMIT $2
            """,
            cfg.min_reviews_per_category, max_categories,
        )

    if not cat_rows:
        return {"generated": 0, "failed": 0, "skipped": 0,
                "categories": 0, "target_mode": "amazon_seller"}

    # 2. Get LLM + skill
    from ...services.llm_router import get_llm
    llm = get_llm("campaign")
    if llm is None:
        from ...services import llm_registry
        llm = llm_registry.get_active()
    if llm is None:
        return {"generated": 0, "failed": 0, "skipped": 0,
                "categories": 0, "target_mode": "amazon_seller",
                "error": "No LLM available"}

    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/amazon_seller_campaign_generation")
    if not skill:
        logger.warning("Skill 'digest/amazon_seller_campaign_generation' not found")
        return {"generated": 0, "failed": 0, "skipped": 0,
                "categories": 0, "target_mode": "amazon_seller",
                "error": "Skill not found"}

    llm_model_name = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
    batch_id = f"batch_seller_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    now = datetime.now(timezone.utc)

    generated = 0
    failed = 0
    skipped_dedup = 0
    skipped_no_targets = 0
    sequences_created = 0
    categories_processed = 0

    for cat_row in cat_rows:
        category = cat_row["source_category"]

        # 3. Aggregate category intelligence
        intel = await _aggregate_category_intelligence(pool, category)
        if not intel:
            logger.debug("Insufficient intelligence for category %s, skipping", category)
            continue

        # Cache the snapshot for sequence progression
        try:
            await _save_intelligence_snapshot(pool, intel)
        except Exception as exc:
            logger.debug("Failed to save intel snapshot for %s: %s", category, exc)

        categories_processed += 1

        # 4. Get seller targets in this category
        targets = await _fetch_seller_targets(pool, category)
        if not targets:
            skipped_no_targets += 1
            logger.debug("No seller targets for category %s", category)
            continue

        # 5. Generate campaigns per target
        for target in targets:
            seller_email = target.get("email")
            seller_name = target.get("seller_name") or target.get("company_name") or ""

            # Dedup: skip if recently targeted
            existing = await pool.fetchval(
                """
                SELECT COUNT(*) FROM b2b_campaigns
                WHERE LOWER(company_name) = $1
                  AND target_mode = 'amazon_seller'
                  AND product_category = $2
                  AND created_at > NOW() - make_interval(days => $3)
                """,
                seller_name.lower(), category, cfg.dedup_days,
            )
            if existing and existing > 0:
                skipped_dedup += 1
                continue

            # Build LLM payload
            selling_ctx = {
                "product_name": cfg.product_name,
                "landing_url": cfg.landing_url,
                "free_report_url": cfg.free_report_url,
                "sender_name": cfg.default_sender_name,
                "sender_title": cfg.default_sender_title,
            }

            cold_email_content: dict[str, str] | None = None

            for channel in cfg.channels:
                payload = {
                    **intel,
                    "recipient_name": target.get("seller_name"),
                    "recipient_company": target.get("company_name"),
                    "recipient_type": target.get("seller_type", "private_label"),
                    "selling": selling_ctx,
                    "channel": channel,
                }

                if channel == "email_followup" and cold_email_content:
                    payload["cold_email_context"] = cold_email_content

                content = await _generate_content(
                    llm, skill.content, payload, cfg.max_tokens, cfg.temperature,
                )

                if content:
                    if channel == "email_cold":
                        cold_email_content = {
                            "subject": content.get("subject", ""),
                            "body": content.get("body", ""),
                        }

                    try:
                        await pool.execute(
                            """
                            INSERT INTO b2b_campaigns (
                                company_name, vendor_name, product_category,
                                channel, subject, body, cta,
                                status, batch_id, llm_model,
                                target_mode, recipient_email,
                                metadata
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7,
                                $8, $9, $10, $11, $12, $13
                            )
                            """,
                            seller_name,
                            "",  # vendor_name not applicable
                            category,
                            channel,
                            content.get("subject", ""),
                            content.get("body", ""),
                            content.get("cta", ""),
                            "draft",
                            batch_id,
                            llm_model_name,
                            "amazon_seller",
                            seller_email,
                            json.dumps({
                                "seller_target_id": str(target["id"]),
                                "seller_type": target.get("seller_type"),
                                "recipient_company": target.get("company_name"),
                            }),
                        )
                        generated += 1
                    except Exception:
                        logger.exception(
                            "Failed to store seller campaign for %s/%s/%s",
                            seller_name, category, channel,
                        )
                        failed += 1
                else:
                    failed += 1

            # Create campaign sequence for cold email
            if cold_email_content and settings.campaign_sequence.enabled:
                try:
                    seq_id = await _create_seller_sequence(
                        pool,
                        seller_name=seller_name,
                        seller_email=seller_email,
                        batch_id=batch_id,
                        category=category,
                        intel=intel,
                        selling_ctx=selling_ctx,
                        cold_email_subject=cold_email_content.get("subject", ""),
                        cold_email_body=cold_email_content.get("body", ""),
                    )
                    if seq_id:
                        sequences_created += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to create sequence for %s/%s: %s",
                        seller_name, category, exc,
                    )

    logger.info(
        "Seller campaign generation: %d generated, %d failed, "
        "%d skipped (dedup), %d skipped (no targets), "
        "%d sequences from %d categories",
        generated, failed, skipped_dedup, skipped_no_targets,
        sequences_created, categories_processed,
    )

    return {
        "generated": generated,
        "failed": failed,
        "skipped_dedup": skipped_dedup,
        "skipped_no_targets": skipped_no_targets,
        "sequences_created": sequences_created,
        "categories": categories_processed,
        "batch_id": batch_id,
        "target_mode": "amazon_seller",
    }


async def _create_seller_sequence(
    pool,
    *,
    seller_name: str,
    seller_email: str | None,
    batch_id: str,
    category: str,
    intel: dict[str, Any],
    selling_ctx: dict[str, Any],
    cold_email_subject: str,
    cold_email_body: str,
) -> _uuid.UUID | None:
    """Create a campaign_sequences row for a seller cold email.

    Returns the sequence ID if created, None on conflict.
    """
    cfg = settings.campaign_sequence

    # Store category + intel as company_context for sequence progression
    company_context = {
        "seller_name": seller_name,
        "category": category,
        "recipient_type": "amazon_seller",
        "category_intelligence": {
            "category_stats": intel.get("category_stats", {}),
            "top_pain_points": intel.get("top_pain_points", [])[:5],
            "feature_gaps": intel.get("feature_gaps", [])[:5],
            "competitive_flows": intel.get("competitive_flows", [])[:5],
            "brand_health": intel.get("brand_health", [])[:5],
            "safety_signals": intel.get("safety_signals", [])[:3],
            "top_root_causes": intel.get("top_root_causes", [])[:5],
        },
    }

    seq_id = await pool.fetchval(
        """
        INSERT INTO campaign_sequences (
            company_name, batch_id,
            company_context, selling_context,
            max_steps, recipient_email
        ) VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT ((LOWER(company_name)), batch_id) DO NOTHING
        RETURNING id
        """,
        seller_name,
        batch_id,
        json.dumps(company_context, default=str),
        json.dumps(selling_ctx, default=str),
        cfg.max_steps,
        seller_email,
    )

    if not seq_id:
        logger.debug("Sequence already exists for %s / %s", seller_name, batch_id)
        return None

    # Link the cold email campaign row to this sequence
    await pool.execute(
        """
        UPDATE b2b_campaigns
        SET sequence_id = $1, step_number = 1
        WHERE company_name = $2 AND batch_id = $3
          AND channel = 'email_cold' AND target_mode = 'amazon_seller'
        """,
        seq_id,
        seller_name,
        batch_id,
    )

    # Audit log
    await log_campaign_event(
        pool,
        event_type="generated",
        sequence_id=seq_id,
        step_number=1,
        source="system",
        subject=cold_email_subject,
        body=cold_email_body,
        recipient_email=seller_email,
        metadata={"target_mode": "amazon_seller", "category": category},
    )

    logger.info(
        "Created seller campaign sequence %s for %s (category: %s, batch: %s)",
        seq_id, seller_name, category, batch_id,
    )
    return seq_id


# ------------------------------------------------------------------
# Autonomous task entry point
# ------------------------------------------------------------------


async def run(task: ScheduledTask) -> dict[str, Any]:
    """Autonomous task handler: generate Amazon seller campaigns from category intelligence."""
    cfg = settings.seller_campaign
    if not cfg.enabled:
        return {"_skip_synthesis": "Amazon seller campaign generation disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "DB not ready"}

    result = await generate_campaigns(pool=pool)

    if result.get("generated", 0) > 0:
        from ...pipelines.notify import send_pipeline_notification

        msg = (
            f"Generated {result['generated']} seller campaign(s) across "
            f"{result['categories']} categories. "
            f"{result.get('sequences_created', 0)} sequences created. "
            f"Review drafts in the dashboard."
        )
        await send_pipeline_notification(
            msg, task, title="Atlas: Seller Campaigns",
            default_tags="briefcase,campaign",
        )

    return {"_skip_synthesis": "Seller campaign generation complete", **result}
