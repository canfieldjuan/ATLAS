"""
Backfill missing SEO fields on published blog posts.

For each published post that is missing seo_title, seo_description,
target_keyword, secondary_keywords, or faq, generates them from the
existing content using a structured LLM call.

Usage:
    python scripts/backfill_blog_seo.py [--dry-run]
"""

import asyncio
import json
import logging
import sys
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill_blog_seo")

# SEO field generation prompt
SEO_PROMPT = """\
You are an SEO specialist. Given a blog post title, description, topic_type,
and the first 500 words of content, generate structured SEO metadata.

Title: {title}
Description: {description}
Topic type: {topic_type}
Content preview:
{content_preview}

Return ONLY valid JSON with these fields:
{{
  "seo_title": "Max 60 chars. Front-load the target keyword. Include year if relevant.",
  "seo_description": "Max 155 chars. Include target keyword. Lead with a data point.",
  "target_keyword": "Primary search query this post should rank for.",
  "secondary_keywords": ["2-3 related long-tail keywords"],
  "faq": [
    {{"question": "A real search query", "answer": "2-3 sentence factual answer with numbers"}}
  ]
}}

Target keyword mapping by topic_type:
- vendor_showdown -> "vendor_a vs vendor_b"
- vendor_alternative -> "vendor alternatives"
- churn_report -> "vendor churn rate"
- pricing_reality_check -> "vendor pricing"
- migration_guide -> "switch from vendor"
- switching_story -> "why teams leave vendor"
- vendor_deep_dive -> "vendor reviews"
- market_landscape -> "category software comparison"
- pain_point_roundup -> "category software complaints"
- best_fit_guide -> "best category software"

Generate 3-5 FAQ items based on the content.
"""


async def main():
    dry_run = "--dry-run" in sys.argv

    # Import inside main to avoid import side effects
    sys.path.insert(0, ".")
    from atlas_brain.storage.database import get_db_pool
    from atlas_brain.config import settings

    pool = get_db_pool()

    rows = await pool.fetch("""
        SELECT id, slug, title, description, topic_type, content,
               seo_title, seo_description, target_keyword, faq
        FROM blog_posts
        WHERE status = 'published'
          AND (seo_title IS NULL OR target_keyword IS NULL OR faq IS NULL
               OR faq = '[]'::jsonb OR faq = 'null'::jsonb)
        ORDER BY published_at DESC
    """)

    if not rows:
        logger.info("All published posts have SEO fields populated.")
        return

    logger.info("Found %d posts missing SEO fields", len(rows))

    for row in rows:
        slug = row["slug"]
        content = row["content"] or ""
        content_preview = content[:1500]

        logger.info("Processing: %s", slug)
        logger.info("  Missing: seo_title=%s target_kw=%s faq=%s",
                     "NULL" if not row["seo_title"] else "ok",
                     "NULL" if not row["target_keyword"] else "ok",
                     "NULL/empty" if not row["faq"] or row["faq"] in ([], "[]", None) else "ok")

        prompt = SEO_PROMPT.format(
            title=row["title"],
            description=row["description"] or "",
            topic_type=row["topic_type"] or "",
            content_preview=content_preview,
        )

        if dry_run:
            logger.info("  [DRY RUN] Would generate SEO fields for %s", slug)
            continue

        try:
            import httpx

            # Use OpenRouter for the backfill (cheap, fast)
            api_key = settings.b2b_churn.openrouter_api_key
            if not api_key:
                logger.warning("  No OpenRouter API key, skipping")
                continue

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content_str = data["choices"][0]["message"]["content"]
                seo = json.loads(content_str)

            # Update the post
            await pool.execute("""
                UPDATE blog_posts
                SET seo_title = COALESCE($2, seo_title),
                    seo_description = COALESCE($3, seo_description),
                    target_keyword = COALESCE($4, target_keyword),
                    secondary_keywords = COALESCE($5::jsonb, secondary_keywords),
                    faq = COALESCE($6::jsonb, faq)
                WHERE id = $1
            """,
                row["id"],
                seo.get("seo_title"),
                seo.get("seo_description"),
                seo.get("target_keyword"),
                json.dumps(seo.get("secondary_keywords", [])),
                json.dumps(seo.get("faq", [])),
            )
            logger.info("  Updated: seo_title=%s, target_kw=%s, faq=%d items",
                         seo.get("seo_title", "")[:40],
                         seo.get("target_keyword", ""),
                         len(seo.get("faq", [])))

        except Exception as e:
            logger.error("  Failed: %s", e)
            continue

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
