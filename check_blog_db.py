import asyncio
import os
import json
from atlas_brain.storage.database import get_db_pool
from atlas_brain.config import settings

async def check_blog_posts():
    pool = get_db_pool()
    await pool.initialize()
    try:
        row = await pool.fetchrow(
            "SELECT slug, seo_title, seo_description, target_keyword, faq FROM blog_posts WHERE slug = 'asana-vs-mondaycom-2026-03'"
        )
        if row:
            print(json.dumps(dict(row), indent=2, default=str))
        else:
            print("Post not found in database")
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(check_blog_posts())
