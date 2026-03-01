"""
G2 parser for B2B review scraping.

URL pattern: g2.com/products/{slug}/reviews?page={n}
Hardest target -- heavy Cloudflare protection.
Requires residential proxy + sticky sessions + referer chain.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.g2")

_DOMAIN = "g2.com"
_BASE_URL = "https://www.g2.com/products"


class G2Parser:
    """Parse G2 review pages with Cloudflare bypass."""

    source_name = "g2"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape G2 reviews for the given product."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        consecutive_empty = 0
        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}/reviews"
            if page > 1:
                url += f"?page={page}"

            # Referer chain: Google for first page, previous page for subsequent
            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+g2+reviews"
                if page == 1
                else f"{_BASE_URL}/{target.product_slug}/reviews?page={page - 1}"
                if page > 2
                else f"{_BASE_URL}/{target.product_slug}/reviews"
            )

            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=referer,
                    sticky_session=True,
                    prefer_residential=True,
                )
                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Page {page}: blocked (403) -- CAPTCHA challenge")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    continue

                # Guard against non-HTML responses (CDN error pages, JSON errors)
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page}: unexpected content-type ({ct[:40]})")
                    break

                before = len(reviews)
                page_reviews = _parse_page(resp.text, target, seen_ids)
                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "G2 page 1 returned 0 reviews for %s -- selectors may be stale",
                            target.product_slug,
                        )
                    break  # No more reviews

                reviews.extend(page_reviews)

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("G2: 2 consecutive pages with no new reviews, stopping")
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("G2 page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "G2 scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


def _parse_page(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse a single G2 review page."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # G2 reviews have data-review-id attributes
    review_cards = soup.select('[data-review-id]')

    # Fallback selectors
    if not review_cards:
        review_cards = soup.select(
            '.nested-ajax-loading div[id^="review-"], '
            '[class*="review-listing"], '
            '[itemprop="review"]'
        )

    for card in review_cards:
        try:
            review = _parse_review_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.debug("Failed to parse G2 review card", exc_info=True)

    return reviews


def _parse_review_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a G2 review card."""
    # Review ID
    review_id = card.get("data-review-id", "")
    if not review_id:
        review_id = card.get("id", "")
    if not review_id:
        return None

    # Star rating (1-5)
    rating = None
    stars_el = card.select_one('[class*="stars"], [class*="star-rating"], [itemprop="ratingValue"]')
    if stars_el:
        if stars_el.get("content"):
            try:
                rating = float(stars_el["content"])
            except (ValueError, TypeError):
                pass
        if rating is None:
            # Count filled stars
            filled = card.select('[class*="star--filled"], .star.fill')
            if filled:
                rating = float(len(filled))

    # G2 has structured pros/cons: "What do you like best?" / "What do you dislike?"
    pros = _extract_g2_section(card, "like", "best", "love")
    cons = _extract_g2_section(card, "dislike", "worst", "hate", "missing")

    # Overall review text / summary
    review_text = ""
    summary = None

    # Title/headline
    title_el = card.select_one('[itemprop="name"], [class*="review-title"], h3, h4')
    if title_el:
        summary = title_el.get_text(strip=True)

    # Main body
    body_el = card.select_one('[itemprop="reviewBody"], [class*="review-body"]')
    if body_el:
        review_text = body_el.get_text(strip=True)

    # If no main body, combine pros/cons
    if not review_text:
        parts = []
        if pros:
            parts.append(f"What I like: {pros}")
        if cons:
            parts.append(f"What I dislike: {cons}")
        review_text = "\n".join(parts)

    if not review_text or len(review_text) < 20:
        return None

    # Reviewer info
    reviewer_name = _get_text(card, '[itemprop="author"], [class*="reviewer-name"]')
    reviewer_title = _get_text(card, '[class*="reviewer-title"], [class*="job-title"]')
    reviewer_company = _get_text(card, '[class*="reviewer-company"], [class*="organization"]')
    company_size = _get_text(card, '[class*="company-size"], [class*="employees"]')
    reviewer_industry = _get_text(card, '[class*="industry"]')

    # Date
    reviewed_at = None
    date_el = card.select_one('time, [itemprop="datePublished"], [class*="date"]')
    if date_el:
        reviewed_at = date_el.get("datetime") or date_el.get("content") or date_el.get_text(strip=True)

    return {
        "source": "g2",
        "source_url": f"https://www.g2.com/products/{target.product_slug}/reviews#{review_id}",
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": summary,
        "review_text": review_text[:10000],
        "pros": pros,
        "cons": cons,
        "reviewer_name": reviewer_name,
        "reviewer_title": reviewer_title,
        "reviewer_company": reviewer_company,
        "company_size_raw": company_size,
        "reviewer_industry": reviewer_industry,
        "reviewed_at": reviewed_at,
        "raw_metadata": {
            "extraction_method": "html",
            "source_weight": 1.0,
            "source_type": "verified_review_platform",
        },
    }


def _extract_g2_section(card, *keywords: str) -> str | None:
    """Extract a G2 review section (like/dislike/etc.) by keyword matching.

    G2 structures reviews as Q&A pairs with headings like
    "What do you like best about X?" followed by a response paragraph.
    """
    # Look for heading + sibling pattern
    for heading in card.select("h5, h4, h3, [class*='heading'], [class*='question']"):
        text = heading.get_text(strip=True).lower()
        if any(kw in text for kw in keywords):
            # Get the next sibling with text content
            sibling = heading.find_next_sibling()
            if sibling:
                content = sibling.get_text(strip=True)
                if content and len(content) > 5:
                    return content[:5000]

    # Fallback: class-based matching
    for kw in keywords:
        el = card.select_one(f'[class*="{kw}"] p, [class*="{kw}"]')
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 5:
                return text[:5000]

    return None


def _get_text(card, selector: str) -> str | None:
    """Safely extract text from the first matching element."""
    el = card.select_one(selector)
    if el:
        text = el.get_text(strip=True)
        if text:
            return text
    return None


# Auto-register
register_parser(G2Parser())
