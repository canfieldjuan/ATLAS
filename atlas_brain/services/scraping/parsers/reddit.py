"""
Reddit parser for B2B review scraping.

Uses Reddit's public JSON search endpoints (no API key needed).

Reddit blocked subreddit-scoped search.json in late 2025.  The working
approach is the global search endpoint with a ``subreddit:`` qualifier:
``www.reddit.com/search.json?q=<term>+subreddit:<sub>&...``

Fallback: ``old.reddit.com/r/<sub>/search.json`` still works intermittently.

No proxy required -- Reddit rate-limits but doesn't hard-block.
"""

from __future__ import annotations

import asyncio
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
            # Primary: global search with subreddit qualifier
            # (subreddit-scoped /r/{sub}/search.json returns 403 since late 2025)
            url = (
                f"https://www.reddit.com/search.json"
                f"?q={vendor_encoded}+subreddit:{sub}&sort=new&limit=25&t=year"
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

                # Fallback to old.reddit.com on 403/429
                if resp.status_code in (403, 429):
                    logger.debug(
                        "Reddit global search %d for r/%s, trying old.reddit.com",
                        resp.status_code, sub,
                    )
                    await asyncio.sleep(3)
                    fallback_url = (
                        f"https://old.reddit.com/r/{sub}/search.json"
                        f"?q={vendor_encoded}&sort=new&limit=25&t=year&restrict_sr=on"
                    )
                    resp = await client.get(
                        fallback_url,
                        domain=_DOMAIN,
                        referer=f"https://old.reddit.com/r/{sub}/",
                        sticky_session=False,
                        prefer_residential=False,
                    )

                if resp.status_code != 200:
                    errors.append(f"r/{sub}: HTTP {resp.status_code}")
                    continue

                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    errors.append(f"r/{sub}: non-JSON response ({ct[:40]})")
                    continue

                try:
                    data = resp.json()
                except (ValueError, TypeError):
                    errors.append(f"r/{sub}: non-parseable JSON body")
                    continue
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
                            "extraction_method": "api_json",
                            "source_weight": 0.5,
                            "source_type": "community_discussion",
                            "subreddit": post.get("subreddit", sub),
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
