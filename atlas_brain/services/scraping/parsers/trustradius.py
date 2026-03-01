"""
TrustRadius parser for B2B review scraping.

STATUS: LIMITED -- TrustRadius removed their public review API in 2025 and
switched to 100% client-side rendering.  Individual reviews cannot be
extracted without a headless browser (Playwright).

What still works:
  - Product page JSON-LD: aggregateRating, positiveNotes, negativeNotes
  - These provide a product-level summary but NOT individual reviews

The parser attempts the legacy API first, falls back to product page
JSON-LD, and returns what it can with clear error messages.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.trustradius")

_DOMAIN = "trustradius.com"
_PRODUCT_BASE = "https://www.trustradius.com/products"


class TrustRadiusParser:
    """Parse TrustRadius reviews.

    Falls back to product page JSON-LD when the review API is unavailable.
    """

    source_name = "trustradius"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape TrustRadius reviews for the given product."""
        reviews: list[dict] = []
        errors: list[str] = []

        referer = (
            f"https://www.google.com/search"
            f"?q={quote_plus(target.vendor_name)}+trustradius+reviews"
        )

        # ------------------------------------------------------------------
        # Strategy: scrape the product/reviews page HTML and extract JSON-LD
        # ------------------------------------------------------------------
        url = f"{_PRODUCT_BASE}/{target.product_slug}/reviews"

        try:
            resp = await client.get(
                url,
                domain=_DOMAIN,
                referer=referer,
                sticky_session=True,
                prefer_residential=True,
            )

            if resp.status_code == 403:
                errors.append("Blocked (403) -- CAPTCHA or anti-bot")
                return ScrapeResult(reviews=[], pages_scraped=1, errors=errors)
            if resp.status_code == 404:
                errors.append(f"Product slug not found: {target.product_slug}")
                return ScrapeResult(reviews=[], pages_scraped=1, errors=errors)
            if resp.status_code != 200:
                errors.append(f"HTTP {resp.status_code}")
                return ScrapeResult(reviews=[], pages_scraped=1, errors=errors)

            # Extract JSON-LD from the product page
            reviews = _extract_jsonld_product(resp.text, target)

            if not reviews:
                errors.append(
                    "TrustRadius reviews require client-side rendering; "
                    "only product-level summary available via JSON-LD"
                )

        except Exception as exc:
            errors.append(f"Request failed: {exc}")
            logger.warning(
                "TrustRadius scrape failed for %s: %s",
                target.product_slug, exc,
            )

        logger.info(
            "TrustRadius scrape for %s: %d entries from product page",
            target.vendor_name, len(reviews),
        )

        return ScrapeResult(reviews=reviews, pages_scraped=1, errors=errors)


def _extract_jsonld_product(
    html: str, target: ScrapeTarget
) -> list[dict]:
    """Extract product-level data from JSON-LD on the reviews page.

    TrustRadius embeds ``SoftwareApplication`` schema with:
    - ``aggregateRating`` (ratingValue, ratingCount, bestRating)
    - ``positiveNotes`` (ItemList of 2-3 word tags)
    - ``negativeNotes`` (ItemList of 2-3 word tags)

    We synthesize a single "product summary" review from these so the
    downstream enrichment pipeline has *something* to work with.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue

        items = data if isinstance(data, list) else [data]

        for item in items:
            if item.get("@type") != "SoftwareApplication":
                continue

            agg = item.get("aggregateRating", {})
            if not isinstance(agg, dict) or not agg.get("ratingValue"):
                continue

            # Build pros/cons from notes
            pros_list = _extract_notes(item.get("positiveNotes", {}))
            cons_list = _extract_notes(item.get("negativeNotes", {}))

            pros = ", ".join(pros_list) if pros_list else None
            cons = ", ".join(cons_list) if cons_list else None

            product_name = item.get("name") or target.product_name

            # Build a synthetic review text from what we have
            parts = []
            if pros_list:
                parts.append(f"Users praise: {', '.join(pros_list)}.")
            if cons_list:
                parts.append(f"Users criticize: {', '.join(cons_list)}.")
            rating_count = agg.get("ratingCount", 0)
            rating_val = agg.get("ratingValue")
            best = agg.get("bestRating", 10)
            parts.append(
                f"Aggregate rating: {rating_val}/{best} "
                f"from {rating_count:,} reviews."
            )
            review_text = " ".join(parts)

            reviews.append({
                "source": "trustradius",
                "source_url": f"https://www.trustradius.com/products/{target.product_slug}/reviews",
                "source_review_id": f"tr_aggregate_{target.product_slug}",
                "vendor_name": target.vendor_name,
                "product_name": product_name,
                "product_category": target.product_category or item.get("applicationCategory"),
                "rating": float(rating_val) if rating_val is not None else None,
                "rating_max": int(best) if best else 10,
                "summary": f"{product_name} -- TrustRadius aggregate ({rating_count:,} reviews)",
                "review_text": review_text,
                "pros": pros,
                "cons": cons,
                "reviewer_name": None,
                "reviewer_title": None,
                "reviewer_company": None,
                "company_size_raw": None,
                "reviewer_industry": None,
                "reviewed_at": None,
                "raw_metadata": {
                    "extraction_method": "jsonld_aggregate",
                    "source_weight": 0.3,
                    "source_type": "aggregate_summary",
                    "aggregate_rating": agg,
                    "positive_notes": pros_list,
                    "negative_notes": cons_list,
                },
            })

    return reviews


def _extract_notes(notes_obj: dict) -> list[str]:
    """Extract note names from a JSON-LD ItemList."""
    if not isinstance(notes_obj, dict):
        return []
    items = notes_obj.get("itemListElement", [])
    if not isinstance(items, list):
        return []
    return [
        item["name"]
        for item in items
        if isinstance(item, dict) and item.get("name")
    ]


# Auto-register
register_parser(TrustRadiusParser())
