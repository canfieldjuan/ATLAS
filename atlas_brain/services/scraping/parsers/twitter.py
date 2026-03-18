"""
Twitter/X parser for B2B review scraping.

Uses Bright Data Web Unlocker to bypass X's anti-bot protections.
Scrapes X search results for vendor complaints, switching signals, and pain points.

Configurable metadata fields (b2b_scrape_targets.metadata JSONB):
  - search_terms: list[str] -- custom search queries (appended to defaults)
  - min_likes: int -- minimum like count to filter noise (default 2)
  - min_retweets: int -- minimum retweet count (default 0)
  - include_replies: bool -- include reply tweets (default False)
  - max_tweets_per_query: int -- cap per search term (default 50)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.twitter")

_DOMAIN = "x.com"
_SEARCH_URL = "https://x.com/search"
_MIN_TEXT_LEN = 60

# Default churn-signal search suffixes appended to vendor name
_DEFAULT_SUFFIXES = [
    "switching from",
    "alternative to",
    "moving away from",
    "terrible",
    "frustrating",
    "looking for replacement",
    "canceling subscription",
    "worst experience",
]

# Regex to extract tweet JSON from X's embedded __NEXT_DATA__ or React state
_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)

# Fallback: extract tweet-like blocks from rendered HTML
_TWEET_TEXT_RE = re.compile(
    r'data-testid="tweetText"[^>]*>(.*?)</div>', re.DOTALL
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags."""
    import html as html_mod
    return html_mod.unescape(_HTML_TAG_RE.sub("", text)).strip()


def _stable_id(source_url: str, text: str) -> str:
    """Generate a stable dedup ID from URL + text prefix."""
    raw = f"{source_url}:{text[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _parse_iso_date(date_str: str | None) -> str | None:
    """Attempt to parse various date formats to ISO8601.

    X uses multiple date formats:
      - ISO8601: "2024-02-15T10:30:00.000Z"
      - Raw/ctime: "Wed Feb 15 10:30:00 +0000 2024"
    """
    if not date_str:
        return None
    # Try ISO8601 first (API format)
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.isoformat()
    except (ValueError, TypeError):
        pass
    # Try X's raw ctime format: "Wed Feb 15 10:30:00 +0000 2024"
    try:
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return dt.isoformat()
    except (ValueError, TypeError):
        return date_str


class TwitterParser:
    """Parse Twitter/X search results for B2B churn signals.

    Strategy:
      1. Build vendor-specific search queries (vendor name + churn signal suffixes)
      2. Fetch X search pages via Bright Data Web Unlocker
      3. Extract tweets from embedded JSON (__NEXT_DATA__) or HTML fallback
      4. Filter by engagement (min_likes) and text length
    """

    source_name = "twitter"
    prefer_residential = True  # X blocks aggressively
    version = "twitter:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        from atlas_brain.config import settings

        # Config from metadata
        try:
            min_likes = int(target.metadata.get("min_likes", 2))
        except (ValueError, TypeError):
            min_likes = 2
        try:
            min_retweets = int(target.metadata.get("min_retweets", 0))
        except (ValueError, TypeError):
            min_retweets = 0
        include_replies = target.metadata.get("include_replies", False)
        try:
            max_per_query = int(target.metadata.get("max_tweets_per_query", 50))
        except (ValueError, TypeError):
            max_per_query = 50

        # Priority 1: Scraping Browser (X blocks all non-browser requests)
        sb_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
        if sb_url:
            sb_domains = {
                d.strip().lower()
                for d in settings.b2b_scrape.scraping_browser_domains.split(",")
                if d.strip()
            }
            if _DOMAIN in sb_domains:
                try:
                    result = await self._scrape_browser(
                        target, sb_url, min_likes, min_retweets,
                        include_replies, max_per_query,
                    )
                    if result.reviews:
                        return result
                    logger.warning(
                        "Scraping Browser for X/%s returned 0 tweets, falling back",
                        target.vendor_name,
                    )
                except Exception as exc:
                    logger.warning(
                        "Scraping Browser failed for X/%s: %s -- falling back",
                        target.vendor_name, exc,
                    )

        # Priority 2: curl_cffi HTTP client (usually times out for X)
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        queries = self._build_queries(target)

        for query in queries:
            query_count = 0
            for page in range(target.max_pages):
                if query_count >= max_per_query:
                    break

                url = self._build_search_url(query, include_replies, page)

                try:
                    resp = await client.get(
                        url,
                        domain=_DOMAIN,
                        referer="https://x.com/",
                        sticky_session=True,
                        prefer_residential=True,
                        extra_headers={
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            "Accept-Language": "en-US,en;q=0.9",
                        },
                    )
                    pages_scraped += 1

                    if resp.status_code == 429:
                        errors.append(f"X rate limited on query '{query}' page {page}")
                        break
                    if resp.status_code != 200:
                        errors.append(f"X search '{query}' page {page}: HTTP {resp.status_code}")
                        break

                    body = resp.text
                    tweets = self._extract_tweets(body, target, seen_ids, min_likes, min_retweets)

                    if not tweets:
                        break

                    reviews.extend(tweets)
                    query_count += len(tweets)

                except Exception as exc:
                    errors.append(f"X search '{query}' page {page}: {exc}")
                    logger.warning("X scrape failed for '%s' page %d: %s", query, page, exc)
                    break

        logger.info(
            "Twitter/X scrape for %s: %d tweets from %d pages (%d queries)",
            target.vendor_name, len(reviews), pages_scraped, len(queries),
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    # ------------------------------------------------------------------
    # Scraping Browser path (Bright Data cloud Chromium for X)
    # ------------------------------------------------------------------

    async def _scrape_browser(
        self,
        target: ScrapeTarget,
        ws_url: str,
        min_likes: int,
        min_retweets: int,
        include_replies: bool,
        max_per_query: int,
    ) -> ScrapeResult:
        """Scrape X via Bright Data Scraping Browser (cloud Chromium)."""
        import asyncio as _aio
        import random as _rand
        from playwright.async_api import async_playwright

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        queries = self._build_queries(target)

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(ws_url, timeout=30000)
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = await context.new_page()

                for query in queries:
                    url = self._build_search_url(query, include_replies, 0)
                    try:
                        resp = await page.goto(url, wait_until="networkidle", timeout=30000)
                        pages_scraped += 1

                        if not resp or resp.status != 200:
                            errors.append(f"Browser X search '{query}': HTTP {resp.status if resp else 0}")
                            continue

                        # Scroll to load more tweets
                        for _ in range(min(target.max_pages, 5)):
                            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            await _aio.sleep(_rand.uniform(1.5, 3.0))

                        html = await page.content()
                        tweets = self._extract_tweets(html, target, seen_ids, min_likes, min_retweets)
                        reviews.extend(tweets)

                        if len(reviews) >= max_per_query:
                            break

                        await _aio.sleep(_rand.uniform(2.0, 4.0))

                    except Exception as exc:
                        errors.append(f"Browser X search '{query}': {type(exc).__name__}: {str(exc)[:80]}")

                await page.close()
                await browser.close()

        except Exception as exc:
            errors.append(f"Browser connection failed: {exc}")

        logger.info(
            "Scraping Browser X/%s: %d tweets from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            status="success" if reviews else "failed",
        )

    def _build_queries(self, target: ScrapeTarget) -> list[str]:
        """Build search queries from vendor name + configurable suffixes."""
        vendor = target.vendor_name
        queries = []

        # Custom terms from metadata take priority
        custom_terms = target.metadata.get("search_terms")
        if custom_terms and isinstance(custom_terms, list):
            for term in custom_terms[:8]:
                queries.append(str(term))

        # Default churn-signal queries
        for suffix in _DEFAULT_SUFFIXES:
            queries.append(f'"{vendor}" {suffix}')

        # Direct vendor complaints (no reply noise)
        queries.append(f'"{vendor}" -filter:replies')

        return queries

    @staticmethod
    def _build_search_url(query: str, include_replies: bool, page: int) -> str:
        """Build X search URL with filters."""
        # f=live for latest tweets (chronological, better for churn signals)
        encoded = quote_plus(query)
        url = f"{_SEARCH_URL}?q={encoded}&f=live"
        if not include_replies:
            url += "&pf=on"  # "People you follow" is off, but this suppresses some replies
        return url

    def _extract_tweets(
        self,
        html: str,
        target: ScrapeTarget,
        seen_ids: set[str],
        min_likes: int,
        min_retweets: int,
    ) -> list[dict]:
        """Extract tweets from X page HTML.

        Tries embedded JSON first (__NEXT_DATA__), falls back to HTML parsing.
        """
        tweets = self._extract_from_json(html, target, seen_ids, min_likes, min_retweets)
        if tweets:
            return tweets

        # Fallback: HTML parsing (less reliable, but works if JSON unavailable)
        return self._extract_from_html(html, target, seen_ids, min_likes)

    def _extract_from_json(
        self,
        html: str,
        target: ScrapeTarget,
        seen_ids: set[str],
        min_likes: int,
        min_retweets: int,
    ) -> list[dict]:
        """Try extracting tweets from __NEXT_DATA__ JSON blob."""
        tweets: list[dict] = []

        match = _NEXT_DATA_RE.search(html)
        if not match:
            # Try finding JSON in script tags with tweet data
            return self._extract_from_inline_json(html, target, seen_ids, min_likes, min_retweets)

        try:
            data = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            return tweets

        # Walk the JSON tree to find tweet entries
        tweet_entries = _find_tweet_entries(data)

        for entry in tweet_entries:
            tweet = self._process_tweet_entry(entry, target, seen_ids, min_likes, min_retweets)
            if tweet:
                tweets.append(tweet)

        return tweets

    def _extract_from_inline_json(
        self,
        html: str,
        target: ScrapeTarget,
        seen_ids: set[str],
        min_likes: int,
        min_retweets: int,
    ) -> list[dict]:
        """Extract tweets from inline JSON state (window.__INITIAL_STATE__ etc)."""
        tweets: list[dict] = []

        # X sometimes embeds tweet data in script tags as JSON
        json_pattern = re.compile(r'"full_text"\s*:\s*"((?:[^"\\]|\\.){60,})"')
        id_pattern = re.compile(r'"id_str"\s*:\s*"(\d+)"')
        name_pattern = re.compile(r'"screen_name"\s*:\s*"([^"]+)"')
        likes_pattern = re.compile(r'"favorite_count"\s*:\s*(\d+)')
        rt_pattern = re.compile(r'"retweet_count"\s*:\s*(\d+)')
        date_pattern = re.compile(r'"created_at"\s*:\s*"([^"]+)"')

        texts = json_pattern.findall(html)
        ids = id_pattern.findall(html)
        names = name_pattern.findall(html)
        likes_list = likes_pattern.findall(html)
        rt_list = rt_pattern.findall(html)
        dates = date_pattern.findall(html)

        for i, text in enumerate(texts):
            # Decode JSON-escaped text
            try:
                text = json.loads(f'"{text}"')
            except (json.JSONDecodeError, ValueError):
                pass

            if len(text) < _MIN_TEXT_LEN:
                continue

            likes = int(likes_list[i]) if i < len(likes_list) else 0
            rts = int(rt_list[i]) if i < len(rt_list) else 0

            if likes < min_likes:
                continue
            if rts < min_retweets:
                continue

            tweet_id = ids[i] if i < len(ids) else ""
            screen_name = names[i] if i < len(names) else ""
            created_at = dates[i] if i < len(dates) else None

            source_url = f"https://x.com/{screen_name}/status/{tweet_id}" if tweet_id and screen_name else ""
            review_id = tweet_id or _stable_id(source_url or str(i), text)

            if review_id in seen_ids:
                continue
            seen_ids.add(review_id)

            tweets.append(self._build_review(
                text=text,
                summary=text[:500],
                source_url=source_url,
                review_id=review_id,
                author=f"@{screen_name}" if screen_name else None,
                created_at=_parse_iso_date(created_at),
                likes=likes,
                retweets=rts,
                target=target,
            ))

        return tweets

    def _extract_from_html(
        self,
        html: str,
        target: ScrapeTarget,
        seen_ids: set[str],
        min_likes: int,
    ) -> list[dict]:
        """Fallback: extract tweet text from rendered HTML."""
        tweets: list[dict] = []
        matches = _TWEET_TEXT_RE.findall(html)

        for i, raw_text in enumerate(matches):
            text = _strip_html(raw_text)
            if len(text) < _MIN_TEXT_LEN:
                continue

            review_id = _stable_id(f"html_{i}", text)
            if review_id in seen_ids:
                continue
            seen_ids.add(review_id)

            tweets.append(self._build_review(
                text=text,
                summary=text[:500],
                source_url="",
                review_id=review_id,
                author=None,
                created_at=None,
                likes=0,
                retweets=0,
                target=target,
            ))

        return tweets

    def _process_tweet_entry(
        self,
        entry: dict,
        target: ScrapeTarget,
        seen_ids: set[str],
        min_likes: int,
        min_retweets: int,
    ) -> dict | None:
        """Process a single tweet entry from JSON into a review dict."""
        legacy = entry.get("legacy") or entry
        text = legacy.get("full_text", "")
        if len(text) < _MIN_TEXT_LEN:
            return None

        likes = legacy.get("favorite_count", 0) or 0
        rts = legacy.get("retweet_count", 0) or 0

        if likes < min_likes:
            return None
        if rts < min_retweets:
            return None

        tweet_id = legacy.get("id_str") or entry.get("rest_id", "")
        user = entry.get("core", {}).get("user_results", {}).get("result", {}).get("legacy", {})
        screen_name = user.get("screen_name", "")
        display_name = user.get("name", "")
        created_at = legacy.get("created_at")

        source_url = f"https://x.com/{screen_name}/status/{tweet_id}" if tweet_id and screen_name else ""
        review_id = tweet_id or _stable_id(source_url, text)

        if review_id in seen_ids:
            return None
        seen_ids.add(review_id)

        return self._build_review(
            text=text,
            summary=text[:500],
            source_url=source_url,
            review_id=review_id,
            author=f"@{screen_name}" if screen_name else display_name or None,
            created_at=_parse_iso_date(created_at),
            likes=likes,
            retweets=rts,
            target=target,
        )

    @staticmethod
    def _build_review(
        *,
        text: str,
        summary: str,
        source_url: str,
        review_id: str,
        author: str | None,
        created_at: str | None,
        likes: int,
        retweets: int,
        target: ScrapeTarget,
    ) -> dict:
        """Build a b2b_reviews-compatible dict."""
        return {
            "source": "twitter",
            "source_url": source_url,
            "source_review_id": review_id,
            "vendor_name": target.vendor_name,
            "product_name": target.product_name,
            "product_category": target.product_category,
            "rating": None,
            "rating_max": 5,
            "summary": summary,
            "review_text": text[:10000],
            "pros": None,
            "cons": None,
            "reviewer_name": author,
            "reviewer_title": None,
            "reviewer_company": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "reviewed_at": created_at,
            "raw_metadata": {
                "extraction_method": "web_unlocker",
                "source_weight": 0.6,
                "source_type": "social_media",
                "likes": likes,
                "retweets": retweets,
                "platform": "x.com",
            },
        }


def _find_tweet_entries(data: dict | list, depth: int = 0) -> list[dict]:
    """Recursively walk JSON to find tweet result objects."""
    if depth > 15:
        return []

    results: list[dict] = []

    if isinstance(data, dict):
        # X wraps tweets in "tweet_results" -> "result"
        if "tweet_results" in data:
            result = data["tweet_results"].get("result", {})
            if result.get("__typename") in ("Tweet", "TweetWithVisibilityResults"):
                inner = result.get("tweet") or result
                results.append(inner)
                return results

        # Also check "tweetResult" (alternate key)
        if "tweetResult" in data:
            result = data["tweetResult"].get("result", {})
            if "legacy" in result:
                results.append(result)
                return results

        for value in data.values():
            results.extend(_find_tweet_entries(value, depth + 1))

    elif isinstance(data, list):
        for item in data:
            results.extend(_find_tweet_entries(item, depth + 1))

    return results


# Auto-register
register_parser(TwitterParser())
