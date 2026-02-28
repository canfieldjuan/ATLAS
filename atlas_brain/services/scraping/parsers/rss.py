"""
RSS/Atom feed parser for B2B review scraping.

Fetches RSS/Atom feeds (Google News, TechCrunch, etc.) and extracts
articles as churn signals. Uses feedparser for XML parsing.

Signal value: Funding rounds, acquisitions, leadership changes, pricing controversy.
"""

from __future__ import annotations

import html
import logging
import re
from calendar import timegm
from urllib.parse import urlparse

import feedparser
from datetime import datetime, timezone

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.rss")

_DOMAIN = "news.google.com"  # Default domain for rate limiting
_MIN_CONTENT_LENGTH = 200

# Strip HTML tags
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    return html.unescape(_HTML_TAG_RE.sub("", text)).strip()


def _extract_domain(url: str) -> str:
    """Extract domain from URL for rate limiting."""
    try:
        return urlparse(url).hostname or _DOMAIN
    except Exception:
        return _DOMAIN


class RSSParser:
    """Parse RSS/Atom feeds as B2B churn signals."""

    source_name = "rss"
    prefer_residential = False

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape RSS feeds for articles mentioning the vendor."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        min_content_length = target.metadata.get("min_content_length", _MIN_CONTENT_LENGTH)
        keywords = target.metadata.get("keywords")
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]

        # Collect feed URLs: product_slug is the primary feed URL,
        # metadata.feed_urls provides additional feeds
        feed_urls = [target.product_slug]
        extra_feeds = target.metadata.get("feed_urls")
        if extra_feeds and isinstance(extra_feeds, list):
            feed_urls.extend(extra_feeds)

        # max_pages controls how many feed URLs to process
        feed_urls = feed_urls[:target.max_pages]

        for feed_url in feed_urls:
            domain = _extract_domain(feed_url)

            try:
                resp = await client.get(
                    feed_url,
                    domain=domain,
                    referer="https://www.google.com/",
                    sticky_session=False,
                    prefer_residential=False,
                )
                pages_scraped += 1

                if resp.status_code != 200:
                    errors.append(f"RSS feed {feed_url}: HTTP {resp.status_code}")
                    continue

                feed = feedparser.parse(resp.text)

                if feed.bozo and not feed.entries:
                    errors.append(f"RSS feed {feed_url}: parse error ({feed.bozo_exception})")
                    continue

                feed_title = feed.feed.get("title", "")

                for entry in feed.entries:
                    # Determine entry ID (guid) for dedup
                    entry_id = entry.get("id") or entry.get("link", "")
                    if not entry_id or entry_id in seen_ids:
                        continue

                    # Extract content: prefer full content, fall back to summary
                    content = ""
                    if hasattr(entry, "content") and entry.content:
                        content = entry.content[0].get("value", "")
                    if not content:
                        content = entry.get("description", "") or entry.get("summary", "")

                    content = _strip_html(content)
                    title = entry.get("title", "")

                    # Keyword filtering: if keywords specified, at least one must match
                    if keywords:
                        search_text = (title + " " + content).lower()
                        if not any(kw.lower() in search_text for kw in keywords):
                            continue

                    # Skip short content
                    if len(content) < min_content_length:
                        continue

                    seen_ids.add(entry_id)

                    # Parse published date
                    reviewed_at = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            reviewed_at = datetime.fromtimestamp(
                                timegm(entry.published_parsed), tz=timezone.utc
                            ).isoformat()
                        except (ValueError, OverflowError):
                            pass
                    if not reviewed_at and hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        try:
                            reviewed_at = datetime.fromtimestamp(
                                timegm(entry.updated_parsed), tz=timezone.utc
                            ).isoformat()
                        except (ValueError, OverflowError):
                            pass

                    # Reviewer: entry author or feed title
                    reviewer = entry.get("author", "") or feed_title

                    # Categories
                    categories = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]

                    reviews.append({
                        "source": "rss",
                        "source_url": entry.get("link", ""),
                        "source_review_id": entry_id,
                        "vendor_name": target.vendor_name,
                        "product_name": target.product_name,
                        "product_category": target.product_category,
                        "rating": None,
                        "rating_max": 5,
                        "summary": title[:500],
                        "review_text": content[:10000],
                        "pros": None,
                        "cons": None,
                        "reviewer_name": reviewer,
                        "reviewer_title": None,
                        "reviewer_company": None,
                        "company_size_raw": None,
                        "reviewer_industry": None,
                        "reviewed_at": reviewed_at,
                        "raw_metadata": {
                            "extraction_method": "rss_feed",
                            "source_weight": 0.6,
                            "source_type": "news_feed",
                            "feed_title": feed_title,
                            "categories": categories,
                            "feed_url": feed_url,
                        },
                    })

            except Exception as exc:
                errors.append(f"RSS feed {feed_url}: {exc}")
                logger.warning("RSS scrape failed for %s: %s", feed_url, exc)

        logger.info(
            "RSS scrape for %s: %d reviews from %d feeds",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(RSSParser())
