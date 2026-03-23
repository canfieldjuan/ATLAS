"""
YouTube parser for B2B review scraping.

Uses the YouTube Data API v3 (googleapis.com) to search for vendor-related
videos and extract high-engagement comments with churn/comparison intent.

Free tier: 10,000 quota units/day.  Cost per call:
  - search.list: 100 units
  - commentThreads.list: 1 unit

Signal value: genuine user complaints, "switching from X" comments, comparison
discussions in video comment sections.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import quote_plus

import httpx

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, apply_date_cutoff, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.youtube")

_SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"
_COMMENTS_ENDPOINT = "https://www.googleapis.com/youtube/v3/commentThreads"

_MIN_COMMENT_LEN = 100  # Skip short/low-signal comments
_MAX_RESULTS_SEARCH = 10  # Videos per search query (each costs 100 quota units)
_MAX_COMMENT_PAGES = 3  # Comment pages per video (each costs 1 quota unit)
_COMMENTS_PER_PAGE = 100  # Maximum allowed by API

# Search templates -- {vendor} is substituted at runtime
_SEARCH_TEMPLATES = [
    "{vendor} review",
    "{vendor} vs",
    "{vendor} alternative",
    "switching from {vendor}",
]

# Comments containing these phrases carry strong churn signal
_CHURN_QUALIFIERS = [
    "switching from",
    "alternative to",
    "moved to",
    "replaced",
    "migrating from",
    "moved away from",
    "switched to",
]


def _get_youtube_api_key() -> str:
    """Load YouTube Data API key from config or environment."""
    try:
        from ....config import settings
        return settings.b2b_scrape.youtube_api_key
    except Exception:
        pass
    import os
    return os.environ.get("ATLAS_B2B_SCRAPE_YOUTUBE_API_KEY", "")


def _has_churn_signal(text: str) -> bool:
    """Return True if comment text contains at least one churn qualifier."""
    text_lower = text.lower()
    return any(q in text_lower for q in _CHURN_QUALIFIERS)


class YouTubeParser:
    """Parse YouTube video comments as B2B churn signals."""

    source_name = "youtube"
    prefer_residential = False  # API-based, no proxy needed
    version = "youtube:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Search for vendor-related videos and scrape comment threads."""
        api_key = _get_youtube_api_key()
        if not api_key:
            logger.warning("YouTube API key not configured -- skipping %s", target.vendor_name)
            return ScrapeResult(
                reviews=[],
                pages_scraped=0,
                errors=["YouTube API key not configured (set ATLAS_B2B_SCRAPE_YOUTUBE_API_KEY)"],
            )

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        # Build search queries from templates + any custom terms in metadata
        queries = [t.format(vendor=target.vendor_name) for t in _SEARCH_TEMPLATES]
        extra_terms = target.metadata.get("search_terms")
        if extra_terms and isinstance(extra_terms, list):
            queries.extend(extra_terms[:4])

        max_videos = int(target.metadata.get("max_videos_per_query", _MAX_RESULTS_SEARCH))

        async with httpx.AsyncClient(timeout=30) as http:
            # Phase 1: Discover video IDs via search
            video_ids: list[str] = []
            video_meta: dict[str, dict] = {}  # video_id -> {title, channel, published}

            for query in queries:
                params = {
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": max_videos,
                    "order": "date" if target.date_cutoff else "relevance",
                    "key": api_key,
                }
                if target.date_cutoff:
                    params["publishedAfter"] = f"{target.date_cutoff}T00:00:00Z"
                try:
                    resp = await http.get(_SEARCH_ENDPOINT, params=params)
                    pages_scraped += 1

                    if resp.status_code == 403:
                        errors.append(f"YouTube API quota exceeded or key invalid (HTTP 403)")
                        logger.warning("YouTube API 403 for query '%s' -- quota may be exhausted", query)
                        break  # Stop all queries; quota is global
                    if resp.status_code != 200:
                        errors.append(f"search q={query}: HTTP {resp.status_code}")
                        continue

                    data = resp.json()
                    for item in data.get("items", []):
                        vid = item.get("id", {}).get("videoId")
                        if vid and vid not in video_meta:
                            video_ids.append(vid)
                            snippet = item.get("snippet", {})
                            video_meta[vid] = {
                                "title": snippet.get("title", ""),
                                "channel": snippet.get("channelTitle", ""),
                                "published": snippet.get("publishedAt", ""),
                                "search_query": query,
                            }
                except Exception as exc:
                    errors.append(f"search q={query}: {exc}")
                    logger.warning("YouTube search failed for '%s': %s", query, exc)

                # Respect API rate limits
                await asyncio.sleep(0.3)

            if not video_ids:
                logger.info("YouTube: no videos found for %s", target.vendor_name)
                return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

            # Phase 2: Fetch comment threads for each video
            for vid in video_ids:
                meta = video_meta[vid]
                page_token: str | None = None

                for page_num in range(_MAX_COMMENT_PAGES):
                    params = {
                        "part": "snippet",
                        "videoId": vid,
                        "maxResults": _COMMENTS_PER_PAGE,
                        "order": "relevance",
                        "textFormat": "plainText",
                        "key": api_key,
                    }
                    if page_token:
                        params["pageToken"] = page_token

                    try:
                        resp = await http.get(_COMMENTS_ENDPOINT, params=params)
                        pages_scraped += 1

                        if resp.status_code == 403:
                            # Comments disabled or quota hit
                            if "commentsDisabled" in resp.text:
                                errors.append(f"video {vid}: comments disabled")
                            else:
                                errors.append(f"video {vid}: API 403 (quota or permission)")
                            break
                        if resp.status_code != 200:
                            errors.append(f"video {vid} comments page {page_num}: HTTP {resp.status_code}")
                            break

                        data = resp.json()
                        items = data.get("items", [])

                        if not items:
                            break

                        for item in items:
                            review = self._parse_comment(item, target, vid, meta, seen_ids)
                            if review:
                                if target.date_cutoff:
                                    kept_reviews, _ = apply_date_cutoff([review], target.date_cutoff)
                                    if not kept_reviews:
                                        continue
                                    review = kept_reviews[0]
                                reviews.append(review)

                        page_token = data.get("nextPageToken")
                        if not page_token:
                            break

                    except Exception as exc:
                        errors.append(f"video {vid} comments page {page_num}: {exc}")
                        logger.warning("YouTube comments failed for video %s: %s", vid, exc)
                        break

                    await asyncio.sleep(0.2)

        logger.info(
            "YouTube scrape for %s: %d reviews from %d API calls (%d videos)",
            target.vendor_name, len(reviews), pages_scraped, len(video_ids),
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    def _parse_comment(
        self,
        item: dict,
        target: ScrapeTarget,
        video_id: str,
        video_meta: dict,
        seen_ids: set[str],
    ) -> dict | None:
        """Parse a single comment thread item into a review dict. Returns None if skipped."""
        snippet = item.get("snippet", {})
        top_comment = snippet.get("topLevelComment", {})
        comment_snippet = top_comment.get("snippet", {})

        comment_id = top_comment.get("id", "")
        text = comment_snippet.get("textDisplay", "")

        if comment_id in seen_ids:
            return None
        if len(text) < _MIN_COMMENT_LEN:
            return None

        # Prioritise comments with churn signal; still keep long substantive ones
        has_churn = _has_churn_signal(text)
        like_count = comment_snippet.get("likeCount", 0)

        # Skip low-engagement comments that lack churn signal
        if not has_churn and like_count < 5 and len(text) < 200:
            return None

        seen_ids.add(comment_id)

        published_at = comment_snippet.get("publishedAt")  # ISO8601 from API
        author = comment_snippet.get("authorDisplayName", "")

        return {
            "source": "youtube",
            "source_url": f"https://www.youtube.com/watch?v={video_id}&lc={comment_id}",
            "source_review_id": comment_id,
            "vendor_name": target.vendor_name,
            "product_name": target.product_name,
            "product_category": target.product_category,
            "rating": None,
            "rating_max": 5,
            "summary": video_meta.get("title", "")[:500],
            "review_text": text[:10000],
            "pros": None,
            "cons": None,
            "reviewer_name": author,
            "reviewer_title": None,
            "reviewer_company": None,
            "company_size_raw": None,
            "reviewer_industry": None,
            "reviewed_at": published_at,
            "raw_metadata": {
                "extraction_method": "api_json",
                "source_weight": 0.4,
                "source_type": "video_platform",
                "video_id": video_id,
                "video_title": video_meta.get("title", ""),
                "channel": video_meta.get("channel", ""),
                "like_count": like_count,
                "reply_count": snippet.get("totalReplyCount", 0),
                "has_churn_signal": has_churn,
                "search_query": video_meta.get("search_query", ""),
            },
        }


# Auto-register
register_parser(YouTubeParser())
