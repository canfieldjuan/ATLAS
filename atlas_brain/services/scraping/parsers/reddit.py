"""
Reddit parser for B2B review scraping.

Uses Reddit's official OAuth2 API (oauth.reddit.com) for authenticated access.
Falls back to public JSON endpoints if credentials are not configured.

Authenticated API benefits:
  - 600 requests/10 min (vs 10/min unauthenticated)
  - Access to all subreddits including quarantined
  - Larger result sets (up to 100 per page vs 25)
  - Reliable -- no 403/429 blocks
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus

import httpx

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.reddit")

_DOMAIN = "reddit.com"
_MIN_SELFTEXT_LEN = 100  # Skip short posts

# Default subreddits for B2B software complaints/discussions
_DEFAULT_SUBREDDITS = [
    "sysadmin", "salesforce", "aws", "ITManagers",
    "devops", "msp", "networking", "cybersecurity",
    "CRM", "projectmanagement", "SaaS", "startups",
    "smallbusiness", "marketing", "CustomerSuccess",
]

# Extra search terms to find churn signal beyond just the vendor name
_CHURN_QUALIFIERS = [
    "switching from", "alternative to", "replacing",
    "migrating from", "moved away from", "leaving",
]


def _get_reddit_credentials() -> tuple[str, str]:
    """Load Reddit API credentials from config."""
    try:
        from ....config import settings
        return settings.b2b_scrape.reddit_client_id, settings.b2b_scrape.reddit_client_secret
    except Exception:
        pass
    import os
    return (
        os.environ.get("ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID", ""),
        os.environ.get("ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET", ""),
    )


class RedditParser:
    """Parse Reddit posts as B2B review proxies."""

    source_name = "reddit"
    prefer_residential = False  # No proxy needed
    version = "reddit:1"

    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expires: float = 0

    async def _get_oauth_token(self, client_id: str, client_secret: str) -> str | None:
        """Obtain Reddit OAuth2 application-only token."""
        import time
        if self._token and time.monotonic() < self._token_expires:
            return self._token

        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=(client_id, client_secret),
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": "Atlas/1.0 B2B Intelligence"},
                )
                resp.raise_for_status()
                data = resp.json()
                self._token = data["access_token"]
                # Token valid for ~1 hour, refresh at 50 min
                self._token_expires = time.monotonic() + 3000
                logger.info("Reddit OAuth2 token acquired (expires in %ds)", data.get("expires_in", 3600))
                return self._token
        except Exception as exc:
            logger.warning("Reddit OAuth2 token request failed: %s", exc)
            return None

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Reddit for posts mentioning the vendor."""
        client_id, client_secret = _get_reddit_credentials()
        has_auth = bool(client_id and client_secret)

        if has_auth:
            token = await self._get_oauth_token(client_id, client_secret)
            if token:
                return await self._scrape_authenticated(target, token)
            logger.warning("OAuth2 token failed, falling back to public endpoints")

        return await self._scrape_public(target, client)

    async def _scrape_authenticated(self, target: ScrapeTarget, token: str) -> ScrapeResult:
        """Scrape via Reddit OAuth2 API (oauth.reddit.com). Higher limits, more results."""
        subreddits = target.metadata.get("subreddits") or _DEFAULT_SUBREDDITS
        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(",")]

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Atlas/1.0 B2B Intelligence",
        }

        async with httpx.AsyncClient(timeout=30, headers=headers) as http:
            # Strategy 1: Search each subreddit (high signal)
            for sub in subreddits:
                for query in self._build_queries(target.vendor_name):
                    url = (
                        f"https://oauth.reddit.com/r/{sub}/search"
                        f"?q={quote_plus(query)}&restrict_sr=on&sort=new&limit=100&t=all"
                    )
                    try:
                        resp = await http.get(url)
                        pages_scraped += 1

                        if resp.status_code == 429:
                            logger.debug("Reddit rate limited on r/%s, pausing", sub)
                            await asyncio.sleep(2)
                            continue
                        if resp.status_code != 200:
                            errors.append(f"r/{sub} q={query}: HTTP {resp.status_code}")
                            continue

                        data = resp.json()
                        posts = data.get("data", {}).get("children", [])

                        for post_wrapper in posts:
                            review = self._parse_post(post_wrapper, target, seen_ids)
                            if review:
                                reviews.append(review)

                    except Exception as exc:
                        errors.append(f"r/{sub} q={query}: {exc}")
                        logger.warning("Reddit auth scrape failed r/%s: %s", sub, exc)

                    # Respect rate limits even with auth
                    await asyncio.sleep(0.5)

            # Strategy 2: Global search for churn-specific phrases
            for qualifier in _CHURN_QUALIFIERS:
                query = f'"{qualifier} {target.vendor_name}"'
                url = (
                    f"https://oauth.reddit.com/search"
                    f"?q={quote_plus(query)}&sort=new&limit=100&t=all"
                )
                try:
                    resp = await http.get(url)
                    pages_scraped += 1

                    if resp.status_code == 200:
                        data = resp.json()
                        posts = data.get("data", {}).get("children", [])
                        for post_wrapper in posts:
                            review = self._parse_post(post_wrapper, target, seen_ids)
                            if review:
                                reviews.append(review)
                except Exception as exc:
                    errors.append(f"global q={qualifier}: {exc}")

                await asyncio.sleep(0.5)

        logger.info(
            "Reddit authenticated scrape for %s: %d reviews from %d requests",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    def _build_queries(self, vendor_name: str) -> list[str]:
        """Build search queries for a vendor. Returns vendor name + churn variants."""
        return [vendor_name]

    def _parse_post(self, post_wrapper: dict, target: ScrapeTarget,
                    seen_ids: set[str]) -> dict | None:
        """Parse a single Reddit post into a review dict. Returns None if skipped."""
        post = post_wrapper.get("data", {})
        post_id = post.get("id", "")
        selftext = post.get("selftext", "")

        if post_id in seen_ids:
            return None
        if selftext in ("[removed]", "[deleted]"):
            return None
        if len(selftext) < _MIN_SELFTEXT_LEN:
            return None

        seen_ids.add(post_id)

        created_utc = post.get("created_utc", 0)
        reviewed_at = (
            datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
            if created_utc
            else None
        )

        return {
            "source": "reddit",
            "source_url": f"https://www.reddit.com{post.get('permalink', '')}",
            "source_review_id": post_id,
            "vendor_name": target.vendor_name,
            "product_name": target.product_name,
            "product_category": target.product_category,
            "rating": None,
            "rating_max": 5,
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
                "subreddit": post.get("subreddit", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
            },
        }

    async def _scrape_public(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Fallback: scrape via public JSON endpoints (no auth)."""
        subreddits = target.metadata.get("subreddits") or _DEFAULT_SUBREDDITS
        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(",")]

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        vendor_encoded = quote_plus(target.vendor_name)

        for sub in subreddits[:target.max_pages]:
            sub_encoded = quote_plus(sub)
            url = (
                f"https://www.reddit.com/search.json"
                f"?q={vendor_encoded}+subreddit:{sub_encoded}&sort=new&limit=25&t=year"
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

                if resp.status_code in (403, 429):
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
                    review = self._parse_post(post_wrapper, target, seen_ids)
                    if review:
                        reviews.append(review)

            except Exception as exc:
                errors.append(f"r/{sub}: {exc}")
                logger.warning("Reddit scrape failed for r/%s: %s", sub, exc)

        logger.info(
            "Reddit public scrape for %s: %d reviews from %d subreddits",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(RedditParser())
