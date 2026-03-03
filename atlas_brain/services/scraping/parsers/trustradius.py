"""
TrustRadius parser for B2B review scraping.

Uses Firecrawl for JS-rendered page scraping (TrustRadius is 100% client-side
rendered since 2025). Falls back to JSON-LD aggregate if Firecrawl is unavailable.

Strategy:
  1. Scrape the reviews listing page via Firecrawl → extract review URLs + ratings
  2. Scrape each individual review page → extract structured pros/cons/use cases
  3. Fall back to JSON-LD aggregate if Firecrawl is not configured
"""

from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import quote_plus

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.trustradius")

_DOMAIN = "trustradius.com"
_PRODUCT_BASE = "https://www.trustradius.com/products"


def _get_firecrawl_key() -> str:
    """Load Firecrawl API key from config or env."""
    try:
        from ....config import settings
        return settings.b2b_scrape.firecrawl_api_key
    except Exception:
        pass
    return os.environ.get("ATLAS_B2B_SCRAPE_FIRECRAWL_API_KEY", "")


class TrustRadiusParser:
    """Parse TrustRadius reviews via Firecrawl JS rendering."""

    source_name = "trustradius"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape TrustRadius reviews for the given product."""
        api_key = _get_firecrawl_key()

        if api_key:
            try:
                return await self._scrape_firecrawl(target, api_key)
            except Exception as exc:
                logger.warning(
                    "Firecrawl scrape failed for %s, falling back to JSON-LD: %s",
                    target.product_slug, exc,
                )

        # Fallback: JSON-LD aggregate (1 synthetic review)
        return await self._scrape_jsonld_fallback(target, client)

    async def _scrape_firecrawl(self, target: ScrapeTarget, api_key: str) -> ScrapeResult:
        """Scrape via Firecrawl JS rendering — gets individual reviews."""
        import asyncio
        from firecrawl import FirecrawlApp

        app = FirecrawlApp(api_key=api_key)
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0

        # Step 1: Get the reviews listing page
        listing_url = f"{_PRODUCT_BASE}/{target.product_slug}/reviews"
        listing_result = await asyncio.to_thread(
            app.scrape, listing_url,
            formats=["markdown"],
            wait_for=8000,
        )
        pages_scraped += 1

        md = listing_result.markdown or ""
        if "Page Not Found" in md:
            errors.append(f"Product slug not found: {target.product_slug}")
            return ScrapeResult(reviews=[], pages_scraped=pages_scraped, errors=errors)

        # Extract review URLs and ratings from listing
        review_entries = _parse_listing(md, target)
        logger.info(
            "TrustRadius listing for %s: %d review entries found",
            target.vendor_name, len(review_entries),
        )

        if not review_entries:
            errors.append("No reviews found on listing page")
            return ScrapeResult(reviews=[], pages_scraped=pages_scraped, errors=errors)

        # Step 2: Scrape individual review pages (respect max_pages limit)
        max_reviews = min(len(review_entries), target.max_pages * 5)  # ~5 reviews visible per page
        for entry in review_entries[:max_reviews]:
            try:
                review_result = await asyncio.to_thread(
                    app.scrape, entry["url"],
                    formats=["markdown"],
                    wait_for=6000,
                )
                pages_scraped += 1

                review_md = review_result.markdown or ""
                if len(review_md) < 200:
                    errors.append(f"Empty review page: {entry['url']}")
                    continue

                review = _parse_individual_review(review_md, entry, target)
                if review:
                    reviews.append(review)

            except Exception as exc:
                errors.append(f"Review page failed: {entry['url']}: {exc}")
                logger.warning("TrustRadius review page failed: %s", exc)

        logger.info(
            "TrustRadius Firecrawl scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    async def _scrape_jsonld_fallback(self, target: ScrapeTarget,
                                       client: AntiDetectionClient) -> ScrapeResult:
        """Fallback: extract product-level JSON-LD aggregate."""
        errors: list[str] = []
        url = f"{_PRODUCT_BASE}/{target.product_slug}/reviews"
        referer = (
            f"https://www.google.com/search"
            f"?q={quote_plus(target.vendor_name)}+trustradius+reviews"
        )

        try:
            resp = await client.get(
                url, domain=_DOMAIN, referer=referer,
                sticky_session=True, prefer_residential=True,
            )
            if resp.status_code != 200:
                errors.append(f"HTTP {resp.status_code}")
                return ScrapeResult(reviews=[], pages_scraped=1, errors=errors)

            reviews = _extract_jsonld_product(resp.text, target)
            if not reviews:
                errors.append("Only JSON-LD aggregate available (no Firecrawl key)")
            return ScrapeResult(reviews=reviews, pages_scraped=1, errors=errors)

        except Exception as exc:
            errors.append(f"Request failed: {exc}")
            return ScrapeResult(reviews=[], pages_scraped=1, errors=errors)


def _parse_listing(md: str, target: ScrapeTarget) -> list[dict]:
    """Parse the reviews listing page markdown to extract review URLs and ratings."""
    entries = []
    seen_urls: set[str] = set()

    # Pattern: review title with URL, followed by "Rating: X out of 10"
    # ## [Review Title](https://www.trustradius.com/reviews/slug-date)
    # Rating: 8 out of 10
    blocks = re.split(r'(?=^## \[)', md, flags=re.MULTILINE)

    for block in blocks:
        url_match = re.search(
            r'\(https://www\.trustradius\.com/reviews/([^\s\)]+)\)', block,
        )
        if not url_match:
            continue

        url = f"https://www.trustradius.com/reviews/{url_match.group(1)}"
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Extract rating
        rating_match = re.search(r'Rating:\s*(\d+)\s*out of\s*(\d+)', block)
        rating = int(rating_match.group(1)) if rating_match else None
        rating_max = int(rating_match.group(2)) if rating_match else 10

        # Extract title
        title_match = re.search(r'## \[([^\]]+)\]', block)
        title = title_match.group(1) if title_match else ""

        entries.append({
            "url": url,
            "rating": rating,
            "rating_max": rating_max,
            "title": title,
            "review_slug": url_match.group(1),
        })

    return entries


def _parse_individual_review(md: str, entry: dict, target: ScrapeTarget) -> dict | None:
    """Parse an individual review page markdown into a review dict."""
    # Extract sections
    pros = _extract_section(md, r"#+\s*Pros", r"#+\s*Cons")
    cons = _extract_section(md, r"#+\s*Cons", r"#+\s*(?:Return on|Scalability|Edit|Alternatives)")

    # Build review text from all available sections
    sections = []

    use_cases = _extract_section(md, r"Use Cases and Deployment", r"#+\s*Pros")
    if use_cases:
        sections.append(f"Use Cases: {use_cases}")
    if pros:
        sections.append(f"Pros: {pros}")
    if cons:
        sections.append(f"Cons: {cons}")

    roi = _extract_section(md, r"Return on Investment", r"#+\s*(?:Scalability|Alternatives|Features)")
    if roi:
        sections.append(f"ROI: {roi}")

    alternatives = _extract_section(md, r"Alternatives Considered", r"#+\s*(?:Features|Scalability|Edit)")
    if alternatives:
        sections.append(f"Alternatives Considered: {alternatives}")

    review_text = "\n\n".join(sections)

    # Skip if no meaningful content
    if len(review_text) < 80:
        return None

    return {
        "source": "trustradius",
        "source_url": entry["url"],
        "source_review_id": f"tr_{entry['review_slug']}",
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": entry.get("rating"),
        "rating_max": entry.get("rating_max", 10),
        "summary": entry.get("title", "")[:500],
        "review_text": review_text[:10000],
        "pros": pros[:5000] if pros else None,
        "cons": cons[:5000] if cons else None,
        "reviewer_name": None,
        "reviewer_title": None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": _extract_date_from_slug(entry.get("review_slug", "")),
        "raw_metadata": {
            "extraction_method": "firecrawl_js",
            "source_weight": 0.8,
            "source_type": "verified_review_platform",
        },
    }


def _extract_section(md: str, start_pattern: str, end_pattern: str) -> str | None:
    """Extract text between two heading patterns."""
    match = re.search(
        f'{start_pattern}.*?\n(.*?)(?={end_pattern}|\\Z)',
        md, re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None

    text = match.group(1).strip()
    # Clean up markdown artifacts
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url) → text
    text = re.sub(r'(?:^|\s)Edit\s*(?:Pros|Cons)?\s*$', '', text, flags=re.MULTILINE)  # Remove "Edit" buttons
    text = re.sub(r'^\s*-\s*', '- ', text, flags=re.MULTILINE)  # Normalize list items
    text = text.strip()

    return text if len(text) > 10 else None


def _extract_date_from_slug(slug: str) -> str | None:
    """Extract ISO date from TrustRadius review slug (e.g., 'product-2025-10-21-06-21-14')."""
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', slug)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


# ------------------------------------------------------------------
# JSON-LD fallback (legacy — kept for environments without Firecrawl)
# ------------------------------------------------------------------

def _extract_jsonld_product(html: str, target: ScrapeTarget) -> list[dict]:
    """Extract product-level data from JSON-LD (aggregate only)."""
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

            pros_list = _extract_notes(item.get("positiveNotes", {}))
            cons_list = _extract_notes(item.get("negativeNotes", {}))

            parts = []
            if pros_list:
                parts.append(f"Users praise: {', '.join(pros_list)}.")
            if cons_list:
                parts.append(f"Users criticize: {', '.join(cons_list)}.")
            try:
                rating_count = int(agg.get("ratingCount", 0))
            except (ValueError, TypeError):
                rating_count = 0
            rating_val = agg.get("ratingValue")
            best = agg.get("bestRating", 10)
            parts.append(f"Aggregate rating: {rating_val}/{best} from {rating_count:,} reviews.")

            product_name = item.get("name") or target.product_name
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
                "review_text": " ".join(parts),
                "pros": ", ".join(pros_list) if pros_list else None,
                "cons": ", ".join(cons_list) if cons_list else None,
                "reviewer_name": None, "reviewer_title": None,
                "reviewer_company": None, "company_size_raw": None,
                "reviewer_industry": None, "reviewed_at": None,
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
    return [item["name"] for item in items if isinstance(item, dict) and item.get("name")]


# Auto-register
register_parser(TrustRadiusParser())
