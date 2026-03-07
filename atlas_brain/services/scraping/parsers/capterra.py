"""
Capterra parser for B2B review scraping.

URL pattern: capterra.com/p/{id}/{slug}/reviews/
Strategy: Web Unlocker first, then JSON-LD extraction, fall back to HTML parsing.
Residential proxy required (Cloudflare protected).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.capterra")

_DOMAIN = "capterra.com"
_BASE_URL = "https://www.capterra.com/p"


class CapterraParser:
    """Parse Capterra review pages using JSON-LD or HTML fallback."""

    source_name = "capterra"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Capterra reviews -- Web Unlocker first, then HTTP client."""
        from atlas_brain.config import settings

        # Priority 1: Bright Data Web Unlocker (handles Cloudflare automatically)
        if settings.b2b_scrape.web_unlocker_url:
            unlocker_domains = {
                d.strip().lower()
                for d in settings.b2b_scrape.web_unlocker_domains.split(",")
                if d.strip()
            }
            if _DOMAIN in unlocker_domains:
                try:
                    result = await self._scrape_web_unlocker(target)
                    if result.reviews:
                        return result
                    logger.warning(
                        "Web Unlocker for %s returned 0 reviews, falling back",
                        target.vendor_name,
                    )
                except Exception as exc:
                    logger.warning(
                        "Web Unlocker failed for %s: %s -- falling back",
                        target.vendor_name, exc,
                    )

        # Priority 2: curl_cffi HTTP client with residential proxy
        return await self._scrape_http(target, client)

    # ------------------------------------------------------------------
    # Web Unlocker path (Bright Data -- handles Cloudflare internally)
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape Capterra via Bright Data Web Unlocker proxy."""
        import httpx
        from atlas_brain.config import settings

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for page in range(1, target.max_pages + 1):
            base_path = f"{_BASE_URL}/{target.product_slug}/reviews/"
            url = base_path if page == 1 else f"{base_path}?page={page}"

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }

            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url, verify=False, timeout=90
                ) as http:
                    resp = await http.get(url, headers=headers)

                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Page {page}: blocked (403) via Web Unlocker")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    continue

                page_reviews = _parse_json_ld(resp.text, target, seen_ids)
                if page_reviews:
                    _supplement_pros_cons_from_html(resp.text, page_reviews)
                else:
                    page_reviews = _parse_html(resp.text, target, seen_ids)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Capterra Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "Capterra Web Unlocker page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                break

            await asyncio.sleep(random.uniform(2.0, 5.0))

        logger.info(
            "Capterra Web Unlocker scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    # ------------------------------------------------------------------
    # HTTP client path (curl_cffi + residential proxy + CAPTCHA)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Capterra via curl_cffi HTTP client."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        consecutive_empty = 0
        for page in range(1, target.max_pages + 1):
            # Capterra URL: /p/{id}/{slug}/reviews/ or /p/{id}/{slug}/reviews/?page={n}
            base_path = f"{_BASE_URL}/{target.product_slug}/reviews/"
            url = base_path if page == 1 else f"{base_path}?page={page}"

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+capterra+reviews"
                if page == 1
                else f"{base_path}?page={page - 1}" if page > 2 else base_path
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
                if resp.status_code == 429:
                    errors.append(f"Page {page}: rate limited (429)")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    continue

                # Guard against non-HTML responses
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page}: unexpected content-type ({ct[:40]})")
                    break

                html = resp.text

                # Strategy 1: JSON-LD extraction (most reliable)
                page_reviews = _parse_json_ld(html, target, seen_ids)

                # Supplement JSON-LD reviews with pros/cons from HTML cards
                if page_reviews:
                    _supplement_pros_cons_from_html(html, page_reviews)

                # Strategy 2: HTML fallback
                if not page_reviews:
                    page_reviews = _parse_html(html, target, seen_ids)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Capterra page 1 returned 0 reviews for %s -- "
                            "JSON-LD and HTML selectors may be stale",
                            target.product_slug,
                        )
                    break  # No more reviews

                before = len(reviews)
                reviews.extend(page_reviews)

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("Capterra: 2 consecutive pages with no new reviews, stopping")
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("Capterra page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "Capterra scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


def _supplement_pros_cons_from_html(html: str, reviews: list[dict]) -> None:
    """Fill in pros/cons/reviewer_title/reviewer_company from HTML cards for JSON-LD reviews.

    JSON-LD captures reviewBody but always leaves pros/cons as None.
    The HTML cards have proper pros/cons selectors. Match by source_review_id.
    Modifies reviews in-place; gracefully no-ops if HTML parsing finds nothing.
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select(
        '[data-testid="review-card"], '
        '.review-card, '
        '[class*="ReviewCard"], '
        'div[class*="review-"][class*="card"]'
    )
    if not cards:
        cards = soup.select('div[id^="review-"], div[class*="review-content"]')
    if not cards:
        return

    # Build a lookup from review_id -> parsed HTML card data
    html_data: dict[str, dict] = {}
    for card in cards:
        card_id = card.get("id", "") or card.get("data-review-id", "")
        if not card_id:
            card_id = hashlib.sha256(card.get_text(strip=True)[:200].encode()).hexdigest()[:16]

        pros = None
        for section in card.select('[class*="pros"], [class*="Pros"], [data-testid*="pros"]'):
            text = section.get_text(strip=True)
            if text:
                pros = text[:5000]
        cons = None
        for section in card.select('[class*="cons"], [class*="Cons"], [data-testid*="cons"]'):
            text = section.get_text(strip=True)
            if text:
                cons = text[:5000]

        reviewer_title = _get_text(card, '[class*="title"], [class*="job"]')
        reviewer_company = _get_text(card, '[class*="company"], [class*="org"]')

        html_data[card_id] = {
            "pros": pros,
            "cons": cons,
            "reviewer_title": reviewer_title,
            "reviewer_company": reviewer_company,
        }

    # Match JSON-LD reviews to HTML cards and fill gaps.
    # JSON-LD @id may be a full URL like "https://www.capterra.com/reviews/12345"
    # while HTML card id may be "review-12345" or just "12345". Also try
    # content-hash fallback when IDs don't match.
    for review in reviews:
        rid = review.get("source_review_id", "")
        card_data = html_data.get(rid)
        if not card_data:
            # Try matching by trailing numeric segment (e.g. ".../12345" -> "12345")
            rid_tail = rid.rsplit("/", 1)[-1] if "/" in rid else rid
            for cid, cdata in html_data.items():
                # Match "12345" == "12345" or "review-12345" ends with "12345"
                if cid == rid_tail or cid.endswith(f"-{rid_tail}") or rid_tail.endswith(cid):
                    card_data = cdata
                    break
        if not card_data:
            continue
        if review.get("pros") is None and card_data["pros"]:
            review["pros"] = card_data["pros"]
        if review.get("cons") is None and card_data["cons"]:
            review["cons"] = card_data["cons"]
        if review.get("reviewer_title") is None and card_data["reviewer_title"]:
            review["reviewer_title"] = card_data["reviewer_title"]
        if review.get("reviewer_company") is None and card_data["reviewer_company"]:
            review["reviewer_company"] = card_data["reviewer_company"]


def _parse_json_ld(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Extract reviews from JSON-LD structured data."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Handle both single object and array
        items = data if isinstance(data, list) else [data]

        for item in items:
            # Look for Product/SoftwareApplication with reviews
            review_list = item.get("review", [])
            if not isinstance(review_list, list):
                review_list = [review_list]

            for r in review_list:
                if not isinstance(r, dict):
                    continue

                review_id = r.get("@id", "") or hashlib.sha256(
                    (r.get("reviewBody", "") or "").encode()
                ).hexdigest()[:16]
                if review_id in seen_ids:
                    continue

                review_body = r.get("reviewBody", "")
                if not review_body or len(review_body) < 20:
                    continue

                seen_ids.add(review_id)

                # Extract rating
                rating = None
                rating_obj = r.get("reviewRating", {})
                if isinstance(rating_obj, dict):
                    rating_val = rating_obj.get("ratingValue")
                    if rating_val is not None:
                        try:
                            rating = float(rating_val)
                        except (ValueError, TypeError):
                            pass

                # Extract author
                author = r.get("author", {})
                reviewer_name = author.get("name", "") if isinstance(author, dict) else ""

                # Extract date
                reviewed_at = r.get("datePublished")

                reviews.append({
                    "source": "capterra",
                    "source_url": f"https://www.capterra.com/p/{target.product_slug}/reviews/",
                    "source_review_id": review_id,
                    "vendor_name": target.vendor_name,
                    "product_name": target.product_name or item.get("name"),
                    "product_category": target.product_category,
                    "rating": rating,
                    "rating_max": 5,
                    "summary": r.get("name") or r.get("headline"),
                    "review_text": review_body[:10000],
                    "pros": None,
                    "cons": None,
                    "reviewer_name": reviewer_name or None,
                    "reviewer_title": None,
                    "reviewer_company": None,
                    "company_size_raw": None,
                    "reviewer_industry": None,
                    "reviewed_at": reviewed_at,
                    "raw_metadata": {
                        "extraction_method": "json_ld",
                        "source_weight": 0.9,
                        "source_type": "verified_review_platform",
                    },
                })

    return reviews


def _parse_html(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse Capterra review page HTML as fallback."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Look for review containers
    review_cards = soup.select(
        '[data-testid="review-card"], '
        '.review-card, '
        '[class*="ReviewCard"], '
        'div[class*="review-"][class*="card"]'
    )

    if not review_cards:
        # Broader fallback
        review_cards = soup.select('div[id^="review-"], div[class*="review-content"]')

    for card in review_cards:
        try:
            review = _parse_capterra_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse Capterra review card", exc_info=True)

    return reviews


def _parse_capterra_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a Capterra review card."""
    # Review ID
    review_id = card.get("id", "") or card.get("data-review-id", "")
    if not review_id:
        review_id = hashlib.sha256(card.get_text(strip=True)[:200].encode()).hexdigest()[:16]

    # Rating (star-based, 1-5)
    rating = None
    rating_el = card.select_one('[class*="star"], [class*="rating"], [aria-label*="star"]')
    if rating_el:
        aria = rating_el.get("aria-label", "")
        match = re.search(r"(\d+(?:\.\d+)?)", aria)
        if match:
            rating = float(match.group(1))

    # Review text -- Capterra often has Overall, Pros, Cons sections
    overall_text = ""
    pros = None
    cons = None

    for section in card.select('[class*="pros"], [class*="Pros"], [data-testid*="pros"]'):
        text = section.get_text(strip=True)
        if text:
            pros = text[:5000]

    for section in card.select('[class*="cons"], [class*="Cons"], [data-testid*="cons"]'):
        text = section.get_text(strip=True)
        if text:
            cons = text[:5000]

    # Overall review text
    for text_el in card.select('[class*="review-text"], [class*="ReviewText"], [class*="overall"]'):
        text = text_el.get_text(strip=True)
        if text and len(text) > 20:
            overall_text = text[:10000]
            break

    # Combine if no overall text
    review_text = overall_text
    if not review_text:
        parts = []
        if pros:
            parts.append(f"Pros: {pros}")
        if cons:
            parts.append(f"Cons: {cons}")
        review_text = "\n".join(parts)

    if not review_text or len(review_text) < 20:
        return None

    # Reviewer info
    reviewer_name = _get_text(card, '[class*="reviewer"], [class*="author"]')
    reviewer_title = _get_text(card, '[class*="title"], [class*="job"]')
    reviewer_company = _get_text(card, '[class*="company"], [class*="org"]')
    company_size = _get_text(card, '[class*="size"], [class*="employees"]')
    reviewer_industry = _get_text(card, '[class*="industry"]')

    # Date
    reviewed_at = None
    date_el = card.select_one("time, [class*='date']")
    if date_el:
        reviewed_at = date_el.get("datetime") or date_el.get_text(strip=True)

    return {
        "source": "capterra",
        "source_url": f"https://www.capterra.com/p/{target.product_slug}/reviews/",
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": None,
        "review_text": review_text,
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
            "source_weight": 0.9,
            "source_type": "verified_review_platform",
        },
    }


def _get_text(card, selector: str) -> str | None:
    """Safely extract text from the first matching element."""
    el = card.select_one(selector)
    if el:
        text = el.get_text(strip=True)
        if text:
            return text
    return None


# Auto-register
register_parser(CapterraParser())
