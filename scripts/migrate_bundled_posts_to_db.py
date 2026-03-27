"""
Migrate bundled blog posts from TypeScript files into the blog_posts DB table.

- Inserts posts not in the DB
- Publishes all bundled posts (sets status='published')
- After running, the bundled .ts files are no longer needed

Usage:
    python scripts/migrate_bundled_posts_to_db.py [--dry-run]
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate_bundled_posts")

BLOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "atlas-intel-next", "content", "blog"
)


def parse_ts_post(filepath: str) -> dict | None:
    """Extract blog post data from a TypeScript file."""
    with open(filepath) as f:
        text = f.read()

    # Extract fields using regex (TS object literal)
    def extract_str(key: str) -> str:
        # Match key: 'value' or key: "value" or key: `value`
        for pattern in [
            rf"{key}:\s*'([^']*)'",
            rf'{key}:\s*"([^"]*)"',
            rf"{key}:\s*`([^`]*)`",
        ]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1)
        return ""

    def extract_array(key: str) -> list:
        m = re.search(rf"{key}:\s*\[([^\]]*)\]", text)
        if m:
            items = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
            return items
        return []

    slug = extract_str("slug")
    if not slug:
        return None

    # Content is in backtick template literal
    content_match = re.search(r"content:\s*`(.*?)`\s*[,}]", text, re.DOTALL)
    content = content_match.group(1) if content_match else ""

    # Charts are complex — extract as raw JSON-ish
    charts_match = re.search(r"charts:\s*(\[.*?\])\s*[,}]\s*$", text, re.DOTALL | re.MULTILINE)
    charts = []
    if charts_match:
        try:
            # Clean up TS syntax to valid JSON
            raw = charts_match.group(1)
            raw = re.sub(r"(\w+):", r'"\1":', raw)  # unquoted keys
            raw = raw.replace("'", '"')
            charts = json.loads(raw)
        except Exception:
            charts = []

    return {
        "slug": slug,
        "title": extract_str("title"),
        "description": extract_str("description"),
        "date": extract_str("date"),
        "author": extract_str("author") or "Atlas Intelligence Team",
        "tags": extract_array("tags"),
        "content": content,
        "charts": charts,
        "topic_type": extract_str("topic_type") or "guide",
        "seo_title": extract_str("seo_title") or None,
        "seo_description": extract_str("seo_description") or None,
        "target_keyword": extract_str("target_keyword") or None,
        "secondary_keywords": extract_array("secondary_keywords") or [],
    }


async def main():
    dry_run = "--dry-run" in sys.argv

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from atlas_brain.storage.database import get_db_pool, init_database

    await init_database()
    pool = get_db_pool()

    # Parse all bundled posts
    posts = []
    for f in sorted(os.listdir(BLOG_DIR)):
        if f == "index.ts" or not f.endswith(".ts"):
            continue
        filepath = os.path.join(BLOG_DIR, f)
        post = parse_ts_post(filepath)
        if post:
            posts.append(post)
            logger.info("Parsed: %s", post["slug"])
        else:
            logger.warning("Failed to parse: %s", f)

    logger.info("Parsed %d bundled posts", len(posts))

    # Check which are already in DB
    existing = await pool.fetch(
        "SELECT slug, status FROM blog_posts WHERE slug = ANY($1::text[])",
        [p["slug"] for p in posts],
    )
    existing_map = {r["slug"]: r["status"] for r in existing}

    inserted = 0
    published = 0

    for post in posts:
        slug = post["slug"]

        if slug not in existing_map:
            # Insert new post
            if dry_run:
                logger.info("  [DRY RUN] Would insert: %s", slug)
            else:
                await pool.execute(
                    """
                    INSERT INTO blog_posts (
                        id, slug, title, description, topic_type, tags,
                        content, charts, status, llm_model,
                        source_report_date, published_at, created_at,
                        seo_title, seo_description, target_keyword,
                        secondary_keywords
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6::jsonb,
                        $7, $8::jsonb, 'published', 'bundled_migration',
                        $9, NOW(), NOW(),
                        $10, $11, $12, $13::jsonb
                    )
                    """,
                    str(uuid4()),
                    slug,
                    post["title"],
                    post["description"],
                    post["topic_type"],
                    json.dumps(post["tags"]),
                    post["content"],
                    json.dumps(post["charts"], default=str),
                    datetime.strptime(post["date"], "%Y-%m-%d").date() if post["date"] else None,
                    post["seo_title"],
                    post["seo_description"],
                    post["target_keyword"],
                    json.dumps(post["secondary_keywords"]),
                )
                logger.info("  Inserted + published: %s", slug)
                inserted += 1
        else:
            # Update existing draft to published
            if existing_map[slug] != "published":
                if dry_run:
                    logger.info("  [DRY RUN] Would publish: %s (was %s)", slug, existing_map[slug])
                else:
                    await pool.execute(
                        """
                        UPDATE blog_posts
                        SET status = 'published',
                            published_at = COALESCE(published_at, NOW())
                        WHERE slug = $1 AND status != 'published'
                        """,
                        slug,
                    )
                    logger.info("  Published: %s (was %s)", slug, existing_map[slug])
                    published += 1
            else:
                logger.info("  Already published: %s", slug)

    logger.info("Done. Inserted: %d, Published: %d", inserted, published)


if __name__ == "__main__":
    asyncio.run(main())
