"""
Source parser registry for B2B review scraping.

Each parser implements the ReviewParser protocol and is registered by source name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Protocol

from ..client import AntiDetectionClient

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
)


@dataclass
class ScrapeTarget:
    """A scrape target loaded from the database.

    ``product_slug`` format varies by source:

      URL-slug sources (slug required, embedded in URL):
      - g2: ``salesforce-crm`` -> g2.com/products/salesforce-crm/reviews
      - capterra: ``123456/salesforce`` -> capterra.com/p/123456/salesforce/reviews/
      - trustradius: ``salesforce-crm`` -> trustradius.com/products/salesforce-crm/reviews
      - gartner: ``market-slug/vendor-slug`` or
        ``market-slug/vendor-slug/product/product-slug`` ->
        gartner.com/reviews/market/{m}/vendor/{v}/reviews or
        gartner.com/reviews/market/{m}/vendor/{v}/product/{p}/reviews
      - peerspot: ``monday-com`` -> peerspot.com/products/monday-com-reviews
      - getapp: ``category/a/product`` -> getapp.com/software/{cat}/a/{slug}/reviews/
      - producthunt: ``my-product`` -> GraphQL slug + producthunt.com/products/{slug}/reviews
      - trustpilot: ``monday.com`` -> trustpilot.com/review/monday.com (company domain)
      - slashdot: ``slack`` -> slashdot.org/software/p/slack/

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
    date_cutoff: str | None = None  # ISO date (YYYY-MM-DD); parsers MAY stop when oldest review < this


@dataclass
class PageLog:
    """Per-page telemetry for scraper diagnostics.

    Captures what the scraper requested, what came back, what it
    extracted, and why it continued or stopped. Populated inside
    each parser's pagination loop via ``log_page()``.
    """

    # Request details
    page: int
    url: str
    timestamp: str = ""  # ISO 8601

    # Response details
    status_code: int = 0
    final_url: str = ""         # after redirects
    response_bytes: int = 0
    duration_ms: int = 0

    # Extraction details
    review_nodes_found: int = 0     # DOM containers matching selector
    reviews_parsed: int = 0         # successfully built review dicts
    missing_date: int = 0
    missing_rating: int = 0
    missing_body: int = 0
    missing_title: int = 0
    missing_author: int = 0

    # Date details
    oldest_review: str | None = None
    newest_review: str | None = None

    # Pagination details
    next_page_found: bool = False
    next_page_url: str = ""
    content_hash: str = ""       # SHA-256 of body text (duplicate page detection)

    # Dedup
    duplicate_reviews: int = 0   # reviews matching prior pages

    # Stop reason (empty = keep going)
    stop_reason: str = ""

    # Errors on this page
    errors: list[str] = field(default_factory=list)


def log_page(
    page: int,
    url: str,
    *,
    status_code: int = 0,
    final_url: str = "",
    response_bytes: int = 0,
    duration_ms: int = 0,
    reviews: list[dict[str, Any]] | None = None,
    review_nodes_found: int = 0,
    next_page_found: bool = False,
    next_page_url: str = "",
    raw_body: bytes | str | None = None,
    prior_hashes: set[str] | None = None,
    prior_review_ids: set[str] | None = None,
    errors: list[str] | None = None,
) -> PageLog:
    """Build a PageLog entry from page-level scrape data.

    Call this at the end of each page iteration inside a parser.
    ``prior_hashes`` and ``prior_review_ids`` are mutated in-place
    to track cross-page duplicates.
    """
    reviews = reviews or []
    errors = errors or []

    # Content hash for duplicate page detection
    content_hash = ""
    if raw_body:
        body = raw_body if isinstance(raw_body, bytes) else raw_body.encode("utf-8", errors="replace")
        content_hash = hashlib.sha256(body).hexdigest()[:16]

    # Check for duplicate page (same hash as a prior page)
    is_dup_page = False
    if content_hash and prior_hashes is not None:
        is_dup_page = content_hash in prior_hashes
        prior_hashes.add(content_hash)

    # Count missing fields
    missing_date = sum(1 for r in reviews if not r.get("reviewed_at"))
    missing_rating = sum(1 for r in reviews if r.get("rating") is None)
    missing_body = sum(1 for r in reviews if not r.get("review_text") and not r.get("summary"))
    missing_title = sum(1 for r in reviews if not r.get("summary"))
    missing_author = sum(1 for r in reviews if not r.get("reviewer_name"))

    # Date range on this page
    oldest = None
    newest = None
    for r in reviews:
        rd = r.get("reviewed_at")
        if rd:
            rd_str = str(rd)[:10]
            if oldest is None or rd_str < oldest:
                oldest = rd_str
            if newest is None or rd_str > newest:
                newest = rd_str

    # Cross-page review dedup count
    dup_count = 0
    if prior_review_ids is not None:
        for r in reviews:
            rid = r.get("dedup_key") or r.get("source_review_id") or ""
            if rid and rid in prior_review_ids:
                dup_count += 1
            elif rid:
                prior_review_ids.add(rid)

    pl = PageLog(
        page=page,
        url=url,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status_code=status_code,
        final_url=final_url or url,
        response_bytes=response_bytes,
        duration_ms=duration_ms,
        review_nodes_found=review_nodes_found or len(reviews),
        reviews_parsed=len(reviews),
        missing_date=missing_date,
        missing_rating=missing_rating,
        missing_body=missing_body,
        missing_title=missing_title,
        missing_author=missing_author,
        oldest_review=oldest,
        newest_review=newest,
        next_page_found=next_page_found,
        next_page_url=next_page_url,
        content_hash=content_hash,
        duplicate_reviews=dup_count,
        errors=errors,
    )

    # Auto-classify stop reasons
    if is_dup_page:
        pl.stop_reason = "duplicate_page"
    elif status_code in (403, 429):
        pl.stop_reason = "blocked_or_throttled"
    elif status_code and status_code >= 400:
        pl.stop_reason = "http_error"
    elif review_nodes_found == 0 and len(reviews) == 0 and not errors:
        pl.stop_reason = "empty_response"

    return pl


def _parse_review_date(raw: Any) -> date | None:
    """Parse a review timestamp or date string to a date."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s[:30], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def apply_date_cutoff(
    reviews: list[dict[str, Any]],
    date_cutoff: str | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Filter reviews older than the cutoff and signal when pagination can stop."""
    cutoff = _parse_review_date(date_cutoff)
    if cutoff is None or not reviews:
        return reviews, False

    kept: list[dict[str, Any]] = []
    saw_dated = False
    saw_undated = False
    has_in_range = False

    for review in reviews:
        reviewed_at = _parse_review_date(review.get("reviewed_at"))
        if reviewed_at is None:
            kept.append(review)
            saw_undated = True
            continue
        saw_dated = True
        if reviewed_at >= cutoff:
            kept.append(review)
            has_in_range = True

    should_stop = saw_dated and not saw_undated and not has_in_range
    return kept, should_stop


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
    # Per-page telemetry (populated by parsers that call log_page())
    page_logs: list[PageLog] = field(default_factory=list)
    # Why this scrape stopped (set by parser or post-hoc by script)
    stop_reason: str = ""
    # Page number where the next transport should resume, if continuation is needed
    resume_page: int | None = None

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


def known_source_review_ids(target: ScrapeTarget | None) -> set[str]:
    """Return preloaded same-source review ids attached to the scrape target."""
    metadata = (target.metadata if target is not None else None) or {}
    raw_ids = metadata.get("known_source_review_ids")
    if not isinstance(raw_ids, (list, tuple, set, frozenset)):
        return set()
    return {
        str(review_id).strip()
        for review_id in raw_ids
        if str(review_id).strip()
    }


def page_has_only_known_source_reviews(
    reviews: list[dict[str, Any]],
    target: ScrapeTarget | None,
) -> bool:
    """Return True when every parsed review on a page is already known."""
    if not reviews:
        return False
    known_ids = known_source_review_ids(target)
    if not known_ids:
        return False
    page_ids = [str(review.get("source_review_id") or "").strip() for review in reviews]
    if not page_ids or any(not review_id for review_id in page_ids):
        return False
    return all(review_id in known_ids for review_id in page_ids)


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
from . import reddit, trustradius, capterra, g2, peerspot, getapp, gartner, hackernews, github, rss, youtube, producthunt, trustpilot, stackoverflow, quora, twitter, software_advice, sourceforge, slashdot  # noqa: E402, F401
