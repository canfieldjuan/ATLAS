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
      - g2: ``salesforce-crm`` -> g2.com/products/salesforce-crm/reviews
      - capterra: ``123456/salesforce`` -> capterra.com/p/123456/salesforce/reviews/
      - trustradius: ``salesforce-crm`` -> trustradius.com/products/salesforce-crm/reviews
      - reddit: vendor name itself (used as search term, slug is informational)
      - hackernews: vendor name (used as search term via HN Algolia API)
      - github: vendor name (used as search term via GitHub REST API)
      - rss: feed URL (e.g. Google News RSS search URL)
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


# Auto-register parsers on import
from . import reddit, trustradius, capterra, g2, hackernews, github, rss  # noqa: E402, F401
