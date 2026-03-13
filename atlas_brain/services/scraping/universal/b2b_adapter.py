"""
Universal review adapter — drop-in ReviewParser replacement for B2B pipeline.

Implements the ReviewParser protocol so the intake task can use it
interchangeably with any legacy per-site parser. Uses the universal
scraper's extraction engine (LLM) instead of hand-coded HTML selectors.

Flow:
    b2b_scrape_targets row → adapter.scrape(target, client) → ScrapeResult
    → _insert_reviews() (unchanged downstream)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from typing import Any
from urllib.parse import urlparse

from ..client import AntiDetectionClient
from ..parsers import ScrapeResult, ScrapeTarget
from .engine import extract_from_text
from .html_cleaner import html_to_text
from .source_configs import SourceAdapterConfig, get_source_adapter_config

logger = logging.getLogger("atlas.services.scraping.universal.b2b_adapter")

# Stop pagination after this many consecutive pages with 0 extracted reviews.
_CONSECUTIVE_EMPTY_THRESHOLD = 2


class UniversalReviewAdapter:
    """Drop-in ReviewParser that uses LLM extraction via the universal scraper engine.

    Implements the same interface as legacy parsers:
        source_name: str
        prefer_residential: bool
        version: str
        async def scrape(target, client) -> ScrapeResult
    """

    def __init__(self, config: SourceAdapterConfig) -> None:
        self._config = config
        self.source_name: str = config.source
        self.prefer_residential: bool = config.prefer_residential
        self.version: str = f"universal:{config.source}:v1"

    async def scrape(
        self, target: ScrapeTarget, client: AntiDetectionClient
    ) -> ScrapeResult:
        """Fetch pages, extract reviews via LLM, normalize to b2b_reviews contract."""
        reviews: list[dict[str, Any]] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_texts: set[str] = set()
        consecutive_empty = 0
        domain = urlparse(self._build_url(target, 1)).hostname or self._config.source

        for page_num in range(1, target.max_pages + 1):
            url = self._build_url(target, page_num)

            try:
                # 1. Fetch
                html = await self._fetch(url, target, client, domain)

                # 2. Clean HTML → text
                text = html_to_text(html, max_chars=30000)
                if not text or len(text.strip()) < 50:
                    errors.append(f"page_{page_num}_insufficient_content")
                    consecutive_empty += 1
                    if consecutive_empty >= _CONSECUTIVE_EMPTY_THRESHOLD:
                        break
                    continue

                # 3. LLM extraction
                items, _raw = await extract_from_text(
                    text, self._config.extraction_schema,
                    workload="triage", max_tokens=4096,
                )

                # 4. Normalize → b2b_reviews contract
                page_reviews = self._normalize(
                    items, target, url, seen_texts,
                )
                reviews.extend(page_reviews)
                pages_scraped += 1

                if not page_reviews:
                    consecutive_empty += 1
                    if consecutive_empty >= _CONSECUTIVE_EMPTY_THRESHOLD:
                        logger.info(
                            "%d consecutive empty pages on %s/%s, stopping",
                            consecutive_empty, target.source, target.vendor_name,
                        )
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"page_{page_num}: {exc}")
                logger.warning(
                    "Universal adapter page %d failed for %s/%s: %s",
                    page_num, target.source, target.vendor_name, exc,
                )
                consecutive_empty += 1
                if consecutive_empty >= _CONSECUTIVE_EMPTY_THRESHOLD:
                    break

            # Inter-page delay (skip after last page)
            if page_num < target.max_pages:
                lo, hi = self._config.inter_page_delay
                await asyncio.sleep(random.uniform(lo, hi))

        logger.info(
            "Universal adapter scraped %s/%s: %d reviews across %d pages (%d errors)",
            target.source, target.vendor_name,
            len(reviews), pages_scraped, len(errors),
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
        )

    # ── URL building ─────────────────────────────────────────────────

    def _build_url(self, target: ScrapeTarget, page: int) -> str:
        """Build the scrape URL from config template + target slug."""
        if page > 1 and self._config.pagination_template:
            return self._config.pagination_template.format(
                slug=target.product_slug, page=page,
            )
        return self._config.url_template.format(slug=target.product_slug)

    # ── Fetching ─────────────────────────────────────────────────────

    async def _fetch(
        self,
        url: str,
        target: ScrapeTarget,
        client: AntiDetectionClient,
        domain: str,
    ) -> str:
        """Fetch page HTML using the configured client."""
        if self._config.use_browser:
            from ..browser import get_stealth_browser

            browser = get_stealth_browser()
            result = await browser.scrape_page(url)
            return result.html

        resp = await client.get(
            url,
            domain=domain,
            prefer_residential=self._config.prefer_residential,
        )
        return resp.text

    # ── Normalization ────────────────────────────────────────────────

    def _normalize(
        self,
        items: list[dict[str, Any]],
        target: ScrapeTarget,
        page_url: str,
        seen_texts: set[str],
    ) -> list[dict[str, Any]]:
        """Convert LLM-extracted dicts into b2b_reviews-compatible dicts.

        Applies dedup within the scrape session, min-length filtering,
        and field mapping to match the INSERT contract.
        """
        reviews: list[dict[str, Any]] = []

        for item in items:
            # Extract review text — try common field names the LLM might use
            review_text = _get_first(item, [
                "review_text", "review", "text", "body", "content",
                "full_review_text", "full_text", "reviewBody",
            ])
            if not review_text:
                continue

            # Combine text fields for length check
            pros = _get_first(item, ["pros", "positive", "positives", "what_is_most_valuable", "likes"])
            cons = _get_first(item, ["cons", "negative", "negatives", "what_needs_improvement", "dislikes"])
            combined_len = len(review_text) + len(pros or "") + len(cons or "")
            if combined_len < self._config.min_text_length:
                continue

            # Dedup within scrape session (by text fingerprint)
            text_fp = hashlib.sha256(review_text[:200].encode()).hexdigest()[:16]
            if text_fp in seen_texts:
                continue
            seen_texts.add(text_fp)

            # Build source_review_id from LLM output or text hash
            source_review_id = _get_first(item, [
                "source_review_id", "review_id", "id",
            ])
            if not source_review_id:
                source_review_id = text_fp

            # Rating normalization
            rating = _parse_rating(item, self._config.rating_max)

            reviews.append({
                "source": target.source,
                "source_url": page_url,
                "source_review_id": str(source_review_id),
                "vendor_name": target.vendor_name,
                "product_name": target.product_name,
                "product_category": target.product_category,
                "rating": rating,
                "rating_max": self._config.rating_max,
                "summary": _get_first(item, [
                    "summary", "title", "headline", "name",
                ]),
                "review_text": str(review_text)[:10000],
                "pros": str(pros)[:5000] if pros else None,
                "cons": str(cons)[:5000] if cons else None,
                "reviewer_name": _get_first(item, [
                    "reviewer_name", "author_name", "author", "name",
                    "reviewer", "user_name",
                ]),
                "reviewer_title": _get_first(item, [
                    "reviewer_title", "job_title", "title", "role", "position",
                ]),
                "reviewer_company": _get_first(item, [
                    "reviewer_company", "company", "company_name",
                    "employer", "organization",
                ]),
                "company_size_raw": _get_first(item, [
                    "company_size_raw", "company_size", "org_size",
                    "employee_count", "employees",
                ]),
                "reviewer_industry": _get_first(item, [
                    "reviewer_industry", "industry", "sector",
                ]),
                "reviewed_at": _get_first(item, [
                    "reviewed_at", "date_published", "date", "published_at",
                    "review_date", "datePublished", "created_at",
                ]),
                "raw_metadata": {
                    "extraction_method": "universal_llm",
                    "source_weight": self._config.source_weight,
                    "source_type": self._config.source_type,
                    "page_url": page_url,
                    "schema_version": "v1",
                },
            })

        return reviews


# ── Helpers ──────────────────────────────────────────────────────────


def _get_first(d: dict, keys: list[str]) -> Any:
    """Return the first non-None, non-empty value from a dict by trying keys in order.

    Handles the variability in field names the LLM might choose.
    """
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return None


def _parse_rating(item: dict, rating_max: int) -> float | None:
    """Extract and normalize a numeric rating from the LLM output."""
    raw = _get_first(item, ["rating", "score", "stars", "rating_value"])
    if raw is None:
        return None
    try:
        val = float(raw)
        # Clamp to valid range
        if val < 0:
            return None
        if val > rating_max:
            # Might be on a 10-point scale
            if val <= rating_max * 2:
                val = val / 2
            else:
                return None
        return round(val, 1)
    except (ValueError, TypeError):
        return None


# ── Adapter Registry ─────────────────────────────────────────────────

_ADAPTERS: dict[str, UniversalReviewAdapter] = {}


def get_universal_adapter(source: str) -> UniversalReviewAdapter | None:
    """Get or create a universal adapter for the given source.

    Returns None if no adapter config exists for this source.
    """
    if source in _ADAPTERS:
        return _ADAPTERS[source]

    config = get_source_adapter_config(source)
    if config is None:
        return None

    adapter = UniversalReviewAdapter(config)
    _ADAPTERS[source] = adapter
    return adapter
