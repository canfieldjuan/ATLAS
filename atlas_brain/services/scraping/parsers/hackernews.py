"""
HackerNews parser for B2B review scraping.

Uses the HN Algolia API (hn.algolia.com/api/v1/) -- free, no auth required.
Rate limit: 10,000 req/hr.

Signal value: B2B tool complaints, "switching from X to Y" stories, pain points.
"""

from __future__ import annotations

import html
import logging
import re
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.hackernews")

_DOMAIN = "hn.algolia.com"
_BASE_URL = "https://hn.algolia.com/api/v1/search_by_date"
_MIN_TEXT_LEN = 100
_HITS_PER_PAGE = 50

# Strip HTML tags (HN Algolia returns HTML in story_text/comment_text)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    return html.unescape(_HTML_TAG_RE.sub("", text)).strip()


class HackerNewsParser:
    """Parse HackerNews stories and comments as B2B churn signals."""

    source_name = "hackernews"
    prefer_residential = False

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape HN for stories/comments mentioning the vendor."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        include_comments = target.metadata.get("include_comments", True)
        min_points = target.metadata.get("min_points", 5)

        # Build search terms: vendor name + any custom terms from metadata
        search_terms = [target.vendor_name]
        extra_terms = target.metadata.get("search_terms")
        if extra_terms and isinstance(extra_terms, list):
            search_terms.extend(extra_terms)

        # Determine tag passes: always stories, optionally comments
        tag_passes = ["story"]
        if include_comments:
            tag_passes.append("comment")

        for tag in tag_passes:
            for term in search_terms:
                for page in range(target.max_pages):
                    url = (
                        f"{_BASE_URL}"
                        f"?query={quote_plus(term)}"
                        f"&tags={tag}"
                        f"&hitsPerPage={_HITS_PER_PAGE}"
                        f"&page={page}"
                    )

                    try:
                        resp = await client.get(
                            url,
                            domain=_DOMAIN,
                            referer="https://news.ycombinator.com/",
                            sticky_session=False,
                            prefer_residential=False,
                        )
                        pages_scraped += 1

                        if resp.status_code != 200:
                            errors.append(f"HN {tag} search '{term}' page {page}: HTTP {resp.status_code}")
                            break  # Stop paginating on error

                        data = resp.json()
                        hits = data.get("hits", [])

                        if not hits:
                            break  # No more results

                        for hit in hits:
                            object_id = hit.get("objectID", "")
                            if object_id in seen_ids:
                                continue

                            # For stories: check min_points
                            points = hit.get("points") or 0
                            if tag == "story" and points < min_points:
                                continue

                            # Extract text (HN Algolia returns HTML)
                            if tag == "story":
                                text = _strip_html(hit.get("story_text") or "")
                                if not text:
                                    text = hit.get("title", "")
                                summary = hit.get("title", "")[:500]
                            else:
                                text = _strip_html(hit.get("comment_text", ""))
                                summary = text[:500] if text else ""

                            # Skip short content
                            if len(text) < _MIN_TEXT_LEN:
                                continue

                            seen_ids.add(object_id)

                            reviews.append({
                                "source": "hackernews",
                                "source_url": f"https://news.ycombinator.com/item?id={object_id}",
                                "source_review_id": object_id,
                                "vendor_name": target.vendor_name,
                                "product_name": target.product_name,
                                "product_category": target.product_category,
                                "rating": None,
                                "rating_max": 5,
                                "summary": summary,
                                "review_text": text[:10000],
                                "pros": None,
                                "cons": None,
                                "reviewer_name": hit.get("author", ""),
                                "reviewer_title": None,
                                "reviewer_company": None,
                                "company_size_raw": None,
                                "reviewer_industry": None,
                                "reviewed_at": hit.get("created_at"),  # ISO8601 from API
                                "raw_metadata": {
                                    "extraction_method": "api_json",
                                    "source_weight": 0.5,
                                    "source_type": "community_forum",
                                    "points": points,
                                    "num_comments": hit.get("num_comments") or 0,
                                    "tag": tag,
                                    "search_term": term,
                                },
                            })

                    except Exception as exc:
                        errors.append(f"HN {tag} search '{term}' page {page}: {exc}")
                        logger.warning("HN scrape failed for '%s' (%s) page %d: %s", term, tag, page, exc)
                        break  # Stop paginating on exception

        logger.info(
            "HackerNews scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# Auto-register
register_parser(HackerNewsParser())
