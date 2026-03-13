"""
Per-source mode switching for the B2B review pipeline.

Controls whether each source uses its legacy parser, the universal adapter,
or both in parallel (dual-run mode for validation before cutover).

Modes:
    legacy      — existing parser only (default)
    dual_run    — run both, compare results, insert from legacy
    universal   — universal adapter only
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import Enum
from typing import Any

from ..parsers import ScrapeResult

logger = logging.getLogger("atlas.services.scraping.universal.b2b_mode")


class ScrapeMode(str, Enum):
    LEGACY = "legacy"
    DUAL_RUN = "dual_run"
    UNIVERSAL = "universal"


def get_scrape_mode(source: str) -> ScrapeMode:
    """Determine the scrape mode for a source.

    Reads from ``ATLAS_B2B_SCRAPE_UNIVERSAL_SOURCE_MODES`` env var
    (via ``settings.b2b_scrape.universal_source_modes``).

    Format: comma-separated ``source:mode`` pairs.
    Example: ``trustpilot:universal,peerspot:dual_run``
    """
    from ....config import settings

    raw = settings.b2b_scrape.universal_source_modes
    if not raw:
        return ScrapeMode.LEGACY

    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(":", 1)
        if len(parts) != 2:
            continue
        src, mode = parts[0].strip(), parts[1].strip().lower()
        if src == source:
            try:
                return ScrapeMode(mode)
            except ValueError:
                logger.warning(
                    "Invalid scrape mode '%s' for source '%s', defaulting to legacy",
                    mode, src,
                )
                return ScrapeMode.LEGACY

    return ScrapeMode.LEGACY


async def log_dual_run_comparison(
    source: str,
    vendor: str,
    legacy_result: ScrapeResult,
    universal_result: ScrapeResult,
) -> None:
    """Compare legacy vs universal scrape results and log the comparison.

    This is the validation mechanism: run both parsers on the same target,
    compare coverage and field completeness, log discrepancies. No DB table
    needed — structured log output is sufficient for grep analysis.
    """
    legacy_count = len(legacy_result.reviews)
    universal_count = len(universal_result.reviews)

    # Field completeness: count how many reviews have each field non-None
    legacy_fields = _field_completeness(legacy_result.reviews)
    universal_fields = _field_completeness(universal_result.reviews)

    # Rating distribution
    legacy_ratings = _rating_distribution(legacy_result.reviews)
    universal_ratings = _rating_distribution(universal_result.reviews)

    # Text overlap (approximate match by first 100 chars of review_text)
    legacy_texts = {
        r.get("review_text", "")[:100]
        for r in legacy_result.reviews
        if r.get("review_text")
    }
    universal_texts = {
        r.get("review_text", "")[:100]
        for r in universal_result.reviews
        if r.get("review_text")
    }
    overlap = len(legacy_texts & universal_texts)
    legacy_only = len(legacy_texts - universal_texts)
    universal_only = len(universal_texts - legacy_texts)

    comparison = {
        "source": source,
        "vendor": vendor,
        "legacy_reviews": legacy_count,
        "universal_reviews": universal_count,
        "delta": universal_count - legacy_count,
        "text_overlap": overlap,
        "legacy_only": legacy_only,
        "universal_only": universal_only,
        "legacy_pages": legacy_result.pages_scraped,
        "universal_pages": universal_result.pages_scraped,
        "legacy_errors": len(legacy_result.errors),
        "universal_errors": len(universal_result.errors),
        "legacy_field_completeness": legacy_fields,
        "universal_field_completeness": universal_fields,
        "legacy_ratings": dict(legacy_ratings),
        "universal_ratings": dict(universal_ratings),
    }

    # Log at INFO level with structured data for easy grep
    logger.info(
        "DUAL_RUN_COMPARISON %s/%s: legacy=%d universal=%d overlap=%d "
        "legacy_only=%d universal_only=%d | %s",
        source, vendor,
        legacy_count, universal_count, overlap,
        legacy_only, universal_only,
        comparison,
    )


def _field_completeness(reviews: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate percentage of reviews with each field non-None."""
    if not reviews:
        return {}

    fields = [
        "rating", "summary", "review_text", "pros", "cons",
        "reviewer_name", "reviewer_title", "reviewer_company",
        "company_size_raw", "reviewer_industry", "reviewed_at",
        "source_review_id",
    ]
    total = len(reviews)
    result = {}
    for f in fields:
        present = sum(1 for r in reviews if r.get(f) is not None and r.get(f) != "")
        result[f] = round(present / total * 100, 1)
    return result


def _rating_distribution(reviews: list[dict[str, Any]]) -> Counter:
    """Count reviews by integer rating bucket."""
    c: Counter = Counter()
    for r in reviews:
        rating = r.get("rating")
        if rating is not None:
            try:
                c[int(float(rating))] += 1
            except (ValueError, TypeError):
                pass
    return c
