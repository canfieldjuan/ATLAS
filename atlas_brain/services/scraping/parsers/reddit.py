"""
Reddit parser for B2B review scraping.

Uses Reddit's official OAuth2 API (oauth.reddit.com) for authenticated access.
Falls back to public JSON endpoints if credentials are not configured.

Authenticated API benefits:
  - 600 requests/10 min (vs 10/min unauthenticated)
  - Access to all subreddits including quarantined
  - Larger result sets (up to 100 per page vs 25)
  - Reliable -- no 403/429 blocks

Search profiles (set via target.metadata["search_profile"]):
  - "churn"   (default) -- vendor name + churn qualifiers, posts only
  - "deep"    -- churn + pain/frustration + comparison queries, top comments
  - "insider" -- employee/org queries, insider subreddits, top comments
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
_MIN_SELFTEXT_LEN = 100       # Skip short posts
_MIN_COMMENT_LEN = 80         # Skip short comments
_MIN_COMMENT_SCORE = 2        # Skip low-signal comments

# ---------------------------------------------------------------------------
# Subreddit lists
# ---------------------------------------------------------------------------

# Default B2B discussion subreddits (churn / deep profiles)
_DEFAULT_SUBREDDITS = [
    "sysadmin", "salesforce", "aws", "ITManagers",
    "devops", "msp", "networking", "cybersecurity",
    "CRM", "projectmanagement", "SaaS", "startups",
    "smallbusiness", "marketing", "CustomerSuccess",
]

# Subreddits focused on employee/org insider accounts
_INSIDER_SUBREDDITS = [
    "cscareerquestions", "ExperiencedDevs", "ITCareerQuestions",
    "sysadmin", "devops", "antiwork", "jobs", "technology",
    "cscareeradvice", "softwaregore",
]

# ---------------------------------------------------------------------------
# Query templates per profile
# ---------------------------------------------------------------------------

# Churn / evaluation qualifiers appended to vendor name (churn + deep profiles)
_CHURN_QUALIFIERS = [
    "switching from", "alternative to", "replacing",
    "migrating from", "moved away from", "leaving",
]

# Additional deep-profile query templates ({vendor} is substituted)
_DEEP_QUERY_TEMPLATES = [
    '"{vendor}" pricing too expensive',
    '"{vendor}" support terrible',
    '"{vendor}" vs',
    '"{vendor}" problems',
    '"{vendor}" frustrated',
    '"{vendor}" worth it',
]

# Insider-profile query templates ({vendor} is substituted)
_INSIDER_QUERY_TEMPLATES = [
    '"{vendor}" culture toxic',
    '"{vendor}" leaving why',
    '"worked at {vendor}"',
    '"left {vendor}"',
    '"{vendor}" layoffs morale',
    '"{vendor}" product quality declining',
    '"{vendor}" engineering culture',
    '"{vendor}" management',
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
    """Parse Reddit posts (and optionally comments) as B2B review proxies."""

    source_name = "reddit"
    prefer_residential = False  # No proxy needed
    version = "reddit:2"

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
        """Scrape Reddit for posts (and comments) mentioning the vendor."""
        client_id, client_secret = _get_reddit_credentials()
        has_auth = bool(client_id and client_secret)

        if has_auth:
            token = await self._get_oauth_token(client_id, client_secret)
            if token:
                return await self._scrape_authenticated(target, token)
            logger.warning("OAuth2 token failed, falling back to public endpoints")

        return await self._scrape_public(target, client)

    # ------------------------------------------------------------------
    # Authenticated path
    # ------------------------------------------------------------------

    async def _scrape_authenticated(self, target: ScrapeTarget, token: str) -> ScrapeResult:
        """Scrape via Reddit OAuth2 API. Dispatches by search_profile."""
        profile = (target.metadata.get("search_profile") or "churn").lower()

        subreddits: list[str]
        if profile == "insider":
            subreddits = target.metadata.get("subreddits") or _INSIDER_SUBREDDITS
        else:
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
            post_items = await self._collect_posts_authenticated(
                http, target, profile, subreddits, seen_ids, errors,
            )
            pages_scraped += len(post_items["pages"])

            for post_dict in post_items["posts"]:
                reviews.append(post_dict)

            # Comment harvesting for deep / insider profiles
            if profile in ("deep", "insider"):
                comment_limit = 15 if profile == "insider" else 10
                comment_min_score = _MIN_COMMENT_SCORE

                for post_dict in post_items["posts"]:
                    post_id = post_dict["source_review_id"]
                    num_comments = (post_dict.get("raw_metadata") or {}).get("num_comments", 0)
                    if num_comments < 3:
                        continue

                    comments, cpages, cerrs = await self._fetch_comments_authenticated(
                        http, post_id, target, post_dict, comment_limit, comment_min_score, seen_ids,
                    )
                    reviews.extend(comments)
                    pages_scraped += cpages
                    errors.extend(cerrs)

        logger.info(
            "Reddit authenticated scrape [%s] for %s: %d items from %d requests",
            profile, target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    async def _collect_posts_authenticated(
        self,
        http: httpx.AsyncClient,
        target: ScrapeTarget,
        profile: str,
        subreddits: list[str],
        seen_ids: set[str],
        errors: list[str],
    ) -> dict:
        """Collect post dicts for the given profile. Returns {"posts": [...], "pages": [...]}."""
        posts: list[dict] = []
        pages: list[str] = []

        queries = self._build_queries(target.vendor_name, profile)

        # Strategy 1: Search each subreddit
        for sub in subreddits:
            for query in queries:
                url = (
                    f"https://oauth.reddit.com/r/{sub}/search"
                    f"?q={quote_plus(query)}&restrict_sr=on&sort=new&limit=100&t=all"
                )
                try:
                    resp = await http.get(url)
                    pages.append(url)

                    if resp.status_code == 429:
                        logger.debug("Reddit rate limited on r/%s, pausing", sub)
                        await asyncio.sleep(2)
                        continue
                    if resp.status_code != 200:
                        errors.append(f"r/{sub} q={query}: HTTP {resp.status_code}")
                        continue

                    data = resp.json()
                    for post_wrapper in data.get("data", {}).get("children", []):
                        post_dict = self._parse_post(
                            post_wrapper, target, seen_ids, profile=profile,
                        )
                        if post_dict:
                            posts.append(post_dict)

                except Exception as exc:
                    errors.append(f"r/{sub} q={query}: {exc}")
                    logger.warning("Reddit auth scrape failed r/%s: %s", sub, exc)

                await asyncio.sleep(0.5)

        # Strategy 2: Global search for churn-specific phrases (churn / deep profiles only)
        if profile in ("churn", "deep"):
            for qualifier in _CHURN_QUALIFIERS:
                query = f'"{qualifier} {target.vendor_name}"'
                url = (
                    f"https://oauth.reddit.com/search"
                    f"?q={quote_plus(query)}&sort=new&limit=100&t=all"
                )
                try:
                    resp = await http.get(url)
                    pages.append(url)

                    if resp.status_code == 200:
                        data = resp.json()
                        for post_wrapper in data.get("data", {}).get("children", []):
                            post_dict = self._parse_post(
                                post_wrapper, target, seen_ids, profile=profile,
                            )
                            if post_dict:
                                posts.append(post_dict)
                except Exception as exc:
                    errors.append(f"global q={qualifier}: {exc}")

                await asyncio.sleep(0.5)

        return {"posts": posts, "pages": pages}

    async def _fetch_comments_authenticated(
        self,
        http: httpx.AsyncClient,
        post_id: str,
        target: ScrapeTarget,
        parent_post: dict,
        limit: int,
        min_score: int,
        seen_ids: set[str],
    ) -> tuple[list[dict], int, list[str]]:
        """Fetch top-level comments for a post. Returns (comments, pages_count, errors)."""
        url = (
            f"https://oauth.reddit.com/comments/{post_id}"
            f"?sort=best&limit={limit}&depth=2"
        )
        comments: list[dict] = []
        errors: list[str] = []

        try:
            resp = await http.get(url)
            await asyncio.sleep(0.5)

            if resp.status_code == 429:
                await asyncio.sleep(2)
                return comments, 1, errors
            if resp.status_code != 200:
                errors.append(f"comments/{post_id}: HTTP {resp.status_code}")
                return comments, 1, errors

            data = resp.json()
            # Response is [post_listing, comment_listing]
            if not isinstance(data, list) or len(data) < 2:
                return comments, 1, errors

            comment_listing = data[1]
            for child in comment_listing.get("data", {}).get("children", []):
                comment = self._parse_comment(
                    child, target, parent_post, seen_ids, depth=0, min_score=min_score,
                )
                if comment:
                    comments.append(comment)

        except Exception as exc:
            errors.append(f"comments/{post_id}: {exc}")
            logger.warning("Reddit comment fetch failed for %s: %s", post_id, exc)

        return comments, 1, errors

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    def _build_queries(self, vendor_name: str, profile: str) -> list[str]:
        """Build search queries for a vendor based on search profile."""
        if profile == "insider":
            return [t.format(vendor=vendor_name) for t in _INSIDER_QUERY_TEMPLATES]

        queries = [vendor_name]
        if profile == "deep":
            queries += [t.format(vendor=vendor_name) for t in _DEEP_QUERY_TEMPLATES]

        return queries

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_post(
        self,
        post_wrapper: dict,
        target: ScrapeTarget,
        seen_ids: set[str],
        *,
        profile: str = "churn",
    ) -> dict | None:
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

        # Content type depends on search profile
        if profile == "insider":
            content_type = "insider_account"
        else:
            content_type = "community_discussion"

        # Reddit fullname used as thread_id (e.g. "t3_abc123")
        fullname = post.get("name", f"t3_{post_id}")

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
            # Threading fields
            "content_type": content_type,
            "parent_review_id": None,  # Posts have no parent
            "thread_id": fullname,
            "comment_depth": 0,
            "raw_metadata": {
                "extraction_method": "api_json",
                "source_weight": 0.5,
                "source_type": "community_discussion",
                "search_profile": profile,
                "subreddit": post.get("subreddit", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
            },
        }

    def _parse_comment(
        self,
        child: dict,
        target: ScrapeTarget,
        parent_post: dict,
        seen_ids: set[str],
        *,
        depth: int,
        min_score: int,
    ) -> dict | None:
        """Parse a Reddit comment into a review dict. Returns None if skipped."""
        if child.get("kind") != "t1":
            return None

        data = child.get("data", {})
        comment_id = data.get("id", "")
        body = data.get("body", "")

        if comment_id in seen_ids:
            return None
        if body in ("[removed]", "[deleted]", ""):
            return None
        if len(body) < _MIN_COMMENT_LEN:
            return None

        score = data.get("score", 0)
        if isinstance(score, (int, float)) and score < min_score:
            return None

        seen_ids.add(comment_id)

        created_utc = data.get("created_utc", 0)
        reviewed_at = (
            datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
            if created_utc
            else None
        )

        # Store the parent post's source_review_id so b2b_scrape_intake can resolve
        # it to a UUID after the parent post is inserted.
        parent_source_review_id = parent_post.get("source_review_id")
        thread_id = parent_post.get("thread_id")

        return {
            "source": "reddit",
            "source_url": (
                f"https://www.reddit.com{data.get('permalink', '')}"
                if data.get("permalink")
                else parent_post.get("source_url", "")
            ),
            "source_review_id": f"t1_{comment_id}",
            "vendor_name": target.vendor_name,
            "product_name": target.product_name,
            "product_category": target.product_category,
            "rating": None,
            "rating_max": 5,
            "summary": None,
            "review_text": body[:5000],
            "pros": None,
            "cons": None,
            "reviewer_name": data.get("author", ""),
            "reviewer_title": None,
            "reviewer_company": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "reviewed_at": reviewed_at,
            # Threading fields
            "content_type": "comment",
            "parent_review_id": None,  # resolved post-insert by b2b_scrape_intake
            "thread_id": thread_id,
            "comment_depth": depth + 1,
            "raw_metadata": {
                "extraction_method": "api_json",
                "source_weight": 0.4,
                "source_type": "comment",
                "search_profile": parent_post.get("raw_metadata", {}).get("search_profile", "churn"),
                "subreddit": data.get("subreddit", ""),
                "score": score,
                "upvote_ratio": None,
                "num_comments": None,
                # For parent_review_id resolution after insert
                "parent_source_review_id": parent_source_review_id,
            },
        }

    # ------------------------------------------------------------------
    # Public (unauthenticated) fallback
    # ------------------------------------------------------------------

    async def _scrape_public(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Fallback: scrape via public JSON endpoints (no auth). Posts only, churn profile."""
        profile = (target.metadata.get("search_profile") or "churn").lower()

        subreddits: list[str]
        if profile == "insider":
            subreddits = target.metadata.get("subreddits") or _INSIDER_SUBREDDITS
        else:
            subreddits = target.metadata.get("subreddits") or _DEFAULT_SUBREDDITS

        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(",")]

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        queries = self._build_queries(target.vendor_name, profile)
        vendor_encoded = quote_plus(target.vendor_name)

        for sub in subreddits[:target.max_pages]:
            # Use the first query for the public fallback (simpler / less noisy)
            query_encoded = quote_plus(queries[0] if queries else target.vendor_name)
            sub_encoded = quote_plus(sub)
            url = (
                f"https://www.reddit.com/search.json"
                f"?q={query_encoded}+subreddit:{sub_encoded}&sort=new&limit=25&t=year"
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

                for post_wrapper in data.get("data", {}).get("children", []):
                    post_dict = self._parse_post(
                        post_wrapper, target, seen_ids, profile=profile,
                    )
                    if post_dict:
                        reviews.append(post_dict)

            except Exception as exc:
                errors.append(f"r/{sub}: {exc}")
                logger.warning("Reddit scrape failed for r/%s: %s", sub, exc)

        logger.info(
            "Reddit public scrape [%s] for %s: %d reviews from %d subreddits",
            profile, target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(RedditParser())
