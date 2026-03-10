"""
Source parser registry for B2B review scraping.

Each parser implements the ReviewParser protocol and is registered by source name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..client import AntiDetectionClient


@dataclass
class ScrapeTarget:
    """A scrape target loaded from the database.

    ``product_slug`` format varies by source:

      URL-slug sources (slug required, embedded in URL):
      - g2: ``salesforce-crm`` -> g2.com/products/salesforce-crm/reviews
      - capterra: ``123456/salesforce`` -> capterra.com/p/123456/salesforce/reviews/
      - trustradius: ``salesforce-crm`` -> trustradius.com/products/salesforce-crm/reviews
      - gartner: ``market-slug/vendor-slug`` -> gartner.com/reviews/market/{m}/vendor/{v}/reviews
      - peerspot: ``monday-com`` -> peerspot.com/products/monday-com-reviews
      - getapp: ``category/a/product`` -> getapp.com/software/{cat}/a/{slug}/reviews/
      - producthunt: ``my-product`` -> GraphQL slug + producthunt.com/products/{slug}/reviews
      - trustpilot: ``monday.com`` -> trustpilot.com/review/monday.com (company domain)

      Search-based sources (slug informational, vendor_name is the search term):
      - reddit: vendor name (Reddit API search + churn qualifiers)
      - hackernews: vendor name (HN Algolia API)
      - github: vendor name (GitHub REST API issues/repos search)
      - youtube: vendor name (YouTube Data API v3)
      - stackoverflow: vendor name (Stack Exchange API v2.3)
      - quora: vendor name (search results + direct question pages)
      - twitter: vendor name (X search + Web Unlocker)

      Special:
      - rss: feed URL (e.g. ``https://news.google.com/rss/search?q=salesforce``)
    """

    id: str
    source: str
    vendor_name: str
    product_name: str | None
    product_slug: str
    product_category: str | None
    max_pages: int
    metadata: dict[str, Any]


@dataclass
class ScrapeResult:
    """Result from scraping a single target."""

    reviews: list[dict[str, Any]]  # b2b_reviews-compatible dicts
    pages_scraped: int
    errors: list[str]
    # CAPTCHA telemetry (populated from client stats after scrape)
    captcha_attempts: int = 0
    captcha_types: list[str] | None = None
    captcha_solve_ms: int = 0

    @property
    def status(self) -> str:
        if not self.reviews and self.errors:
            # Check if any error mentions blocking
            if any("403" in e or "blocked" in e.lower() for e in self.errors):
                return "blocked"
            return "failed"
        if self.errors:
            return "partial"
        return "success"


class ReviewParser(Protocol):
    """Protocol for source-specific review parsers."""

    source_name: str
    prefer_residential: bool
    version: str

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape reviews for the given target."""
        ...


# Parser registry
_PARSERS: dict[str, ReviewParser] = {}


def register_parser(parser: ReviewParser) -> None:
    """Register a parser by its source_name."""
    _PARSERS[parser.source_name] = parser


def get_parser(source: str) -> ReviewParser | None:
    """Get a parser by source name."""
    return _PARSERS.get(source)


def get_all_parsers() -> dict[str, ReviewParser]:
    """Get all registered parsers."""
    return dict(_PARSERS)


def get_parser_version(source: str) -> str | None:
    """Get the version string for a parser by source name."""
    parser = _PARSERS.get(source)
    return getattr(parser, 'version', None) if parser else None


# Auto-register parsers on import
from . import reddit, trustradius, capterra, g2, peerspot, getapp, gartner, hackernews, github, rss, youtube, producthunt, trustpilot, stackoverflow, quora, twitter, software_advice  # noqa: E402, F401
