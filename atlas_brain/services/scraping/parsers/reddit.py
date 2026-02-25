"""
Reddit parser for B2B review scraping.

Uses Reddit's public JSON search endpoints (no API key needed).
No proxy required -- Reddit is permissive with rate limits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.reddit")

_DOMAIN = "reddit.com"
_MIN_SELFTEXT_LEN = 100  # Skip short posts

# Default subreddits for B2B software complaints
_DEFAULT_SUBREDDITS = [
    "sysadmin", "salesforce", "aws", "ITManagers",
    "devops", "msp", "networking", "cybersecurity",
]


class RedditParser:
    """Parse Reddit posts as B2B review proxies."""

    source_name = "reddit"
    prefer_residential = False  # No proxy needed

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Reddit for posts mentioning the vendor."""
        subreddits = target.metadata.get("subreddits") or _DEFAULT_SUBREDDITS
        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(",")]

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        vendor_encoded = quote_plus(target.vendor_name)

        for sub in subreddits[:target.max_pages]:
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={vendor_encoded}&sort=new&limit=25&t=year&restrict_sr=on"
            )

            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=f"https://www.reddit.com/r/{sub}/",
                    sticky_session=False,
                    prefer_residential=False,
                )
                pages_scraped += 1

                if resp.status_code != 200:
                    errors.append(f"r/{sub}: HTTP {resp.status_code}")
                    continue

                data = resp.json()
                posts = data.get("data", {}).get("children", [])

                for post_wrapper in posts:
                    post = post_wrapper.get("data", {})
                    post_id = post.get("id", "")
                    selftext = post.get("selftext", "")

                    # Skip duplicates, removed/deleted, and short posts
                    if post_id in seen_ids:
                        continue
                    if selftext in ("[removed]", "[deleted]"):
                        continue
                    if len(selftext) < _MIN_SELFTEXT_LEN:
                        continue

                    seen_ids.add(post_id)

                    # Convert Unix timestamp
                    created_utc = post.get("created_utc", 0)
                    reviewed_at = (
                        datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
                        if created_utc
                        else None
                    )

                    reviews.append({
                        "source": "reddit",
                        "source_url": f"https://www.reddit.com{post.get('permalink', '')}",
                        "source_review_id": post_id,
                        "vendor_name": target.vendor_name,
                        "product_name": target.product_name,
                        "product_category": target.product_category,
                        "rating": None,  # Reddit has no rating system
                        "rating_max": 5,  # DB column is NOT NULL, use default
                        "summary": post.get("title", "")[:500],
                        "review_text": selftext[:10000],
                        "pros": None,
                        "cons": None,
                        "reviewer_name": post.get("author", ""),
                        "reviewer_title": None,
                        "reviewer_company": None,
                        "company_size_raw": None,
                        "reviewer_industry": None,
                        "reviewed_at": reviewed_at,
                        "raw_metadata": {
                            "subreddit": sub,
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "upvote_ratio": post.get("upvote_ratio", 0),
                        },
                    })

            except Exception as exc:
                errors.append(f"r/{sub}: {exc}")
                logger.warning("Reddit scrape failed for r/%s: %s", sub, exc)

        logger.info(
            "Reddit scrape for %s: %d reviews from %d subreddits",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(RedditParser())
