"""
Reddit parser for B2B review scraping.

Uses Reddit's official OAuth2 API (oauth.reddit.com) for authenticated access.
Falls back to public JSON endpoints if credentials are not configured.

Authenticated API benefits:
  - 600 requests/10 min (vs 10/min unauthenticated)
  - Access to all subreddits including quarantined
  - Larger result sets (up to 100 per page vs 25)
  - Reliable -- no 403/429 blocks

Enhancements (v2):
  - Comment thread harvesting (top 3 comments for signal-rich posts)
  - Post flair + award-based source_weight boosting
  - Author churn scoring (repeat migration posters flagged)
  - Edit history tracking (is_edited + edit_timestamp)
  - Cross-post detection + source_weight boost
  - Time-based trending score (low/medium/high vs 30-day baseline)
  - Expanded subreddits and churn qualifiers
  - Semaphore-based parallelism with rate-limit backoff
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import httpx

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.reddit")

_DOMAIN = "reddit.com"
_MIN_SELFTEXT_LEN = 100  # Skip short posts
_MAX_COMMENT_FETCHES = 30  # Max comment-thread fetches per batch
_COMMENT_TRIGGER_KEYWORDS = {
    "migration", "migrating", "evaluating", "replacing", "budget",
    "cto", "switch", "switching", "alternative", "leaving", "moved",
    "procurement", "pilot", "rfq",
}

# Default subreddits for B2B software complaints/discussions (expanded)
_DEFAULT_SUBREDDITS = [
    "sysadmin", "salesforce", "aws", "ITManagers",
    "devops", "msp", "networking", "cybersecurity",
    "CRM", "projectmanagement", "SaaS", "startups",
    "smallbusiness", "marketing", "CustomerSuccess",
    "EnterpriseIT", "business", "softwarearchitecture",
]

# Churn qualifiers for global search
_CHURN_QUALIFIERS = [
    "switching from", "alternative to", "replacing",
    "migrating from", "moved away from", "leaving",
    "RFQ", "procurement", "budget approval", "pilot",
]

# Flair → source_weight boost
_FLAIR_WEIGHT_BOOST: dict[str, float] = {
    "rant": 0.2,
    "help": 0.15,
    "discussion": 0.1,
    "question": 0.1,
    "advice": 0.1,
    "complaint": 0.2,
}

# Author churn score weights
_AUTHOR_CHURN_MIGRATION_WEIGHT = 3
_AUTHOR_CHURN_UPVOTE_WEIGHT = 0.1
_AUTHOR_HIGH_SCORE_THRESHOLD = 7


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


def _compute_author_churn_score(author_posts: list[dict]) -> float:
    """
    Score an author's churn signal based on their post history within this batch.

    Weights:
      - Each migration-related post: +3
      - Each upvote on any post: +0.1
      - Use of churn qualifiers in titles: +2 each (capped at 6)
    """
    migration_keywords = {"migration", "migrating", "switching", "replacing", "leaving", "alternative"}
    qualifier_keywords = set(q.lower() for q in _CHURN_QUALIFIERS)

    migration_count = sum(
        1 for p in author_posts
        if any(kw in p.get("title", "").lower() for kw in migration_keywords)
    )
    upvote_total = sum(p.get("score", 0) for p in author_posts)
    qualifier_count = min(
        sum(
            1 for p in author_posts
            if any(q in p.get("title", "").lower() for q in qualifier_keywords)
        ),
        3,  # cap at 3 occurrences → max 6 points
    )

    raw = (
        migration_count * _AUTHOR_CHURN_MIGRATION_WEIGHT
        + upvote_total * _AUTHOR_CHURN_UPVOTE_WEIGHT
        + qualifier_count * 2
    )
    return min(round(raw, 2), 10.0)


def _compute_batch_trending_score(vendor_posts: list[dict]) -> str:
    """
    Classify trending level for the current scrape batch by comparing the
    7-day daily mention rate to the 30-day daily mention rate.

    Spike ≥2× monthly daily rate → high; ≥1.25× → medium; else low.
    vendor_posts: all posts collected for this vendor in the current batch.

    Returns a single score applied uniformly to all posts in the batch —
    this is a batch-level signal, not a per-post metric.
    """
    if not vendor_posts:
        return "low"

    now = datetime.now(tz=timezone.utc)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    recent = 0
    monthly = 0
    for p in vendor_posts:
        ts = p.get("reviewed_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if dt >= cutoff_7d:
            recent += 1
        if dt >= cutoff_30d:
            monthly += 1

    if monthly == 0:
        return "low"

    # Normalise to daily rates to account for the different window sizes
    daily_recent = recent / 7
    daily_monthly = monthly / 30

    if daily_monthly == 0:
        return "low"

    ratio = daily_recent / daily_monthly
    if ratio >= 2.0:
        return "high"
    if ratio >= 1.25:
        return "medium"
    return "low"


class RedditParser:
    """Parse Reddit posts as B2B review proxies."""

    source_name = "reddit"
    prefer_residential = False  # No proxy needed
    version = "reddit:2"

    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expires: float = 0

    # ------------------------------------------------------------------
    # OAuth2
    # ------------------------------------------------------------------

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
                    headers={"User-Agent": "Atlas/2.0 B2B Intelligence"},
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Authenticated scrape
    # ------------------------------------------------------------------

    async def _scrape_authenticated(self, target: ScrapeTarget, token: str) -> ScrapeResult:
        """Scrape via Reddit OAuth2 API (oauth.reddit.com). Higher limits, more results."""
        subreddits = target.metadata.get("subreddits") or _DEFAULT_SUBREDDITS
        if isinstance(subreddits, str):
            subreddits = [s.strip() for s in subreddits.split(",")]

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        # author → list of their parsed post dicts (for scoring)
        author_index: dict[str, list[dict]] = {}

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Atlas/2.0 B2B Intelligence",
        }

        # Semaphore: max 3 concurrent requests
        sem = asyncio.Semaphore(3)

        async def _get(http: httpx.AsyncClient, url: str) -> httpx.Response | None:
            async with sem:
                try:
                    resp = await http.get(url)
                    pages_scraped_ref[0] += 1
                    if resp.status_code == 429:
                        logger.debug("Reddit rate limited, pausing 2s")
                        await asyncio.sleep(2)
                        errors.append(f"GET {url}: 429 rate limited")
                        return None
                    return resp
                except Exception as exc:
                    errors.append(f"GET {url}: {exc}")
                    return None

        pages_scraped_ref = [0]  # mutable int via list

        async with httpx.AsyncClient(timeout=30, headers=headers) as http:

            # ---- Rate limit guard: pause if nearing 80% of 600/10min ----
            async def _maybe_pause() -> None:
                if pages_scraped_ref[0] > 0 and pages_scraped_ref[0] % 480 == 0:
                    logger.info("Reddit: approaching rate limit, pausing 120s")
                    await asyncio.sleep(120)

            # ---- Strategy 1: per-subreddit search (parallel batches of 3) ----
            tasks = []
            for sub in subreddits:
                for query in self._build_queries(target.vendor_name):
                    url = (
                        f"https://oauth.reddit.com/r/{sub}/search"
                        f"?q={quote_plus(query)}&restrict_sr=on&sort=new&limit=100&t=all"
                    )
                    tasks.append((sub, query, url))

            for i in range(0, len(tasks), 3):
                batch = tasks[i:i + 3]
                resps = await asyncio.gather(*[
                    _get(http, url) for _, _, url in batch
                ])
                for (sub, query, _), resp in zip(batch, resps):
                    if resp is None:
                        # error already appended inside _get (429 or exception)
                        continue
                    if resp.status_code != 200:
                        errors.append(f"r/{sub} q={query}: HTTP {resp.status_code}")
                        continue
                    data = resp.json()
                    for post_wrapper in data.get("data", {}).get("children", []):
                        review = self._parse_post(post_wrapper, target, seen_ids)
                        if review:
                            reviews.append(review)
                            author = review.get("reviewer_name", "")
                            if author:
                                author_index.setdefault(author, []).append({
                                    "title": review.get("summary", ""),
                                    "score": review.get("raw_metadata", {}).get("score", 0),
                                    "reviewed_at": review.get("reviewed_at"),
                                })

                await _maybe_pause()
                await asyncio.sleep(0.3)

            # ---- Strategy 2: global churn-qualifier search ----
            for qualifier in _CHURN_QUALIFIERS:
                query = f'"{qualifier} {target.vendor_name}"'
                url = (
                    f"https://oauth.reddit.com/search"
                    f"?q={quote_plus(query)}&sort=new&limit=100&t=all"
                )
                resp = await _get(http, url)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    for post_wrapper in data.get("data", {}).get("children", []):
                        review = self._parse_post(post_wrapper, target, seen_ids)
                        if review:
                            reviews.append(review)
                            author = review.get("reviewer_name", "")
                            if author:
                                author_index.setdefault(author, []).append({
                                    "title": review.get("summary", ""),
                                    "score": review.get("raw_metadata", {}).get("score", 0),
                                    "reviewed_at": review.get("reviewed_at"),
                                })
                await asyncio.sleep(0.3)

            # ---- Phase 2: comment harvesting ----
            comment_fetches = 0
            for review in reviews:
                if comment_fetches >= _MAX_COMMENT_FETCHES:
                    break
                meta = review.get("raw_metadata", {})
                num_comments = meta.get("num_comments", 0)
                text_lower = (review.get("review_text", "") + review.get("summary", "")).lower()
                has_trigger = any(kw in text_lower for kw in _COMMENT_TRIGGER_KEYWORDS)

                if num_comments > 5 and has_trigger:
                    post_id = review.get("source_review_id", "")
                    subreddit = meta.get("subreddit", "")
                    if post_id and subreddit:
                        comments = await self._fetch_comments(post_id, subreddit, http, sem, pages_scraped_ref)
                        if comments:
                            review["raw_metadata"]["comment_threads"] = comments
                            review["raw_metadata"]["comment_depth"] = 2
                            comment_fetches += 1
                        await asyncio.sleep(0.3)

            # ---- Phase 2: enrich with author scores + trending ----
            # Trending is a batch-level signal; compute once, apply to all posts.
            trending_baseline = [
                {"reviewed_at": r.get("reviewed_at")} for r in reviews if r.get("reviewed_at")
            ]
            batch_trending = _compute_batch_trending_score(trending_baseline)

            for review in reviews:
                author = review.get("reviewer_name", "")
                author_posts = author_index.get(author, [])
                churn_score = _compute_author_churn_score(author_posts)

                meta = review["raw_metadata"]
                meta["author_churn_score"] = churn_score
                meta["author_post_count_in_batch"] = len(author_posts)

                # Flag high-score authors in reviewer_title
                if churn_score >= _AUTHOR_HIGH_SCORE_THRESHOLD:
                    review["reviewer_title"] = f"Repeat Churn Signal (Score: {churn_score})"

                meta["trending_score"] = batch_trending

        pages_scraped = pages_scraped_ref[0]
        logger.info(
            "Reddit authenticated scrape for %s: %d reviews, %d pages, %d comment fetches",
            target.vendor_name, len(reviews), pages_scraped, comment_fetches,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    # ------------------------------------------------------------------
    # Query builder
    # ------------------------------------------------------------------

    def _build_queries(self, vendor_name: str) -> list[str]:
        """
        Build search queries for a vendor.
        Returns vendor name + key churn variant queries.
        """
        return [
            vendor_name,
            f"{vendor_name} migrating OR replacing",
            f"{vendor_name} alternative OR switching",
            f"{vendor_name} CTO OR budget OR procurement",
        ]

    # ------------------------------------------------------------------
    # Comment harvesting
    # ------------------------------------------------------------------

    async def _fetch_comments(
        self,
        post_id: str,
        subreddit: str,
        http: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        pages_scraped_ref: list[int],
    ) -> list[dict]:
        """Fetch top comments for a post. Returns up to 3 top-level comments (depth=2)."""
        url = (
            f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}.json"
            f"?limit=3&depth=2&sort=top"
        )
        try:
            async with sem:
                resp = await http.get(url)
                pages_scraped_ref[0] += 1
            if resp.status_code != 200:
                return []
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2:
                return []
            comments = []
            for comment_wrapper in payload[1].get("data", {}).get("children", []):
                comment = comment_wrapper.get("data", {})
                body = comment.get("body", "")
                if body in ("[removed]", "[deleted]", "") or not body:
                    continue
                comments.append({
                    "comment_id": comment.get("id"),
                    "text": body[:2000],
                    "author": comment.get("author", ""),
                    "score": comment.get("score", 0),
                    "created_utc": comment.get("created_utc", 0),
                })
            return comments
        except Exception as exc:
            logger.debug("Comment fetch failed for %s: %s", post_id, exc)
            return []

    # ------------------------------------------------------------------
    # Post parser
    # ------------------------------------------------------------------

    def _parse_post(
        self, post_wrapper: dict, target: ScrapeTarget, seen_ids: set[str]
    ) -> dict | None:
        """Parse a single Reddit post into a review dict. Returns None if skipped."""
        post = post_wrapper.get("data", {})
        post_id = post.get("id", "")
        selftext = post.get("selftext", "")

        if not post_id:
            return None
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

        # ---- Dynamic source_weight ----
        source_weight = 0.5

        # Flair boost
        flair = (post.get("link_flair_text") or "").lower()
        source_weight += _FLAIR_WEIGHT_BOOST.get(flair, 0.0)

        # Award boost
        awards = post.get("all_awardings", []) or []
        source_weight += min(len(awards) * 0.02, 0.1)

        # Cross-post boost
        crossposts = post.get("crosspost_parent_list") or []
        is_crosspost = bool(crossposts)
        crosspost_subreddits: list[str] = []
        if is_crosspost:
            source_weight += 0.15
            # crosspost_parent_list items have a "subreddit" field
            crosspost_subreddits = [
                cp.get("subreddit", "") for cp in crossposts if cp.get("subreddit")
            ]

        source_weight = round(min(source_weight, 1.0), 3)

        # ---- Edit history ----
        edited = post.get("edited")
        is_edited = bool(edited and edited is not False)
        edit_timestamp: str | None = None
        if is_edited and isinstance(edited, (int, float)):
            edit_timestamp = datetime.fromtimestamp(edited, tz=timezone.utc).isoformat()

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
                "source_weight": source_weight,
                "source_type": "community_discussion",
                "subreddit": post.get("subreddit", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
                # Flair
                "post_flair": post.get("link_flair_text") or "",
                # Awards
                "award_count": len(awards),
                # Edit tracking
                "is_edited": is_edited,
                "edit_timestamp": edit_timestamp,
                # Cross-post
                "is_crosspost": is_crosspost,
                "crosspost_subreddits": crosspost_subreddits,
                # Populated in post-processing:
                "comment_threads": [],
                "comment_depth": 0,
                "author_churn_score": 0.0,
                "author_post_count_in_batch": 0,
                "trending_score": "low",
            },
        }

    # ------------------------------------------------------------------
    # Public fallback
    # ------------------------------------------------------------------

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
                        domain="old.reddit.com",
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
                    review = self._parse_post(post_wrapper, target, seen_ids)
                    if review:
                        reviews.append(review)

            except Exception as exc:
                errors.append(f"r/{sub}: {exc}")
                logger.warning("Reddit scrape failed for r/%s: %s", sub, exc)

        # Also run global churn-qualifier searches in public mode
        for qualifier in _CHURN_QUALIFIERS[:4]:  # limit to top 4 in public mode
            query = f'"{qualifier} {target.vendor_name}"'
            url = (
                f"https://www.reddit.com/search.json"
                f"?q={quote_plus(query)}&sort=new&limit=25&t=year"
            )
            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer="https://www.reddit.com/search",
                    sticky_session=False,
                    prefer_residential=False,
                )
                pages_scraped += 1
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except (ValueError, TypeError):
                        continue
                    for post_wrapper in data.get("data", {}).get("children", []):
                        review = self._parse_post(post_wrapper, target, seen_ids)
                        if review:
                            reviews.append(review)
                await asyncio.sleep(1.5)
            except Exception as exc:
                errors.append(f"public global q={qualifier}: {exc}")

        logger.info(
            "Reddit public scrape for %s: %d reviews from %d subreddits",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(RedditParser())
