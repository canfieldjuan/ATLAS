"""
Gartner Peer Insights parser for B2B review scraping.

URL pattern: gartner.com/reviews/market/{market-slug}/vendor/{vendor-slug}/reviews
Akamai-protected -- needs Web Unlocker as primary strategy.

Primary: Bright Data Web Unlocker (handles Akamai automatically).
Fallback: curl_cffi HTTP client with residential proxy.
No Playwright tier -- Web Unlocker handles Akamai better than stealth browsers.

Pagination: ?start={offset} (not page=N). Each page shows ~10 reviews.
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

from ....config import settings
from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.gartner")

_DOMAIN = "gartner.com"
_BASE_URL = "https://www.gartner.com/reviews/market"
_REVIEWS_PER_PAGE = 10


class GartnerParser:
    """Parse Gartner Peer Insights review pages with Akamai bypass."""

    source_name = "gartner"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Gartner Peer Insights -- Web Unlocker first, then HTTP."""

        # Priority 1: Bright Data Web Unlocker (handles Akamai automatically)
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
    # Web Unlocker path (Bright Data -- handles Akamai internally)
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape Gartner Peer Insights via Bright Data Web Unlocker proxy.

        Web Unlocker is an HTTP proxy that handles Akamai Bot Manager
        challenges automatically -- no CAPTCHA solving or stealth browser
        needed.  Just send a normal GET and it returns the unblocked HTML.
        """
        import httpx

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for page in range(target.max_pages):
            offset = page * _REVIEWS_PER_PAGE
            url = _build_reviews_url(target.product_slug, offset)

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+gartner+peer+insights+reviews"
                if page == 0
                else _build_reviews_url(target.product_slug, (page - 1) * _REVIEWS_PER_PAGE)
            )

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": referer,
            }

            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url,
                    verify=False,
                    timeout=90.0,
                ) as http:
                    resp = await http.get(url, headers=headers)

                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Page {page + 1}: blocked (403) via Web Unlocker")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page + 1}: HTTP {resp.status_code}")
                    continue

                # Try JSON-LD first, then HTML fallback
                page_reviews = _parse_json_ld(resp.text, target, seen_ids)
                if not page_reviews:
                    page_reviews = _parse_html(resp.text, target, seen_ids)

                if not page_reviews:
                    if page == 0:
                        logger.warning(
                            "Gartner Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page + 1}: {exc}")
                logger.warning(
                    "Gartner Web Unlocker page %d failed for %s: %s",
                    page + 1, target.product_slug, exc,
                )
                break

            # Inter-page delay
            await asyncio.sleep(random.uniform(3.0, 6.0))

        logger.info(
            "Gartner Web Unlocker scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    # ------------------------------------------------------------------
    # HTTP path (curl_cffi -- fallback)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Gartner Peer Insights via curl_cffi HTTP client."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        consecutive_empty = 0
        for page in range(target.max_pages):
            offset = page * _REVIEWS_PER_PAGE
            url = _build_reviews_url(target.product_slug, offset)

            # Referer chain: Google for first page, previous page for subsequent
            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+gartner+peer+insights+reviews"
                if page == 0
                else _build_reviews_url(target.product_slug, (page - 1) * _REVIEWS_PER_PAGE)
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
                    errors.append(f"Page {page + 1}: blocked (403) -- Akamai challenge")
                    break
                if resp.status_code == 429:
                    errors.append(f"Page {page + 1}: rate limited (429)")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page + 1}: HTTP {resp.status_code}")
                    continue

                # Guard against non-HTML responses (CDN error pages, JSON errors)
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page + 1}: unexpected content-type ({ct[:40]})")
                    break

                before = len(reviews)

                # Try JSON-LD first, then HTML fallback
                page_reviews = _parse_json_ld(resp.text, target, seen_ids)
                if not page_reviews:
                    page_reviews = _parse_html(resp.text, target, seen_ids)

                if not page_reviews:
                    if page == 0:
                        logger.warning(
                            "Gartner page 1 returned 0 reviews for %s -- selectors may be stale",
                            target.product_slug,
                        )
                    break  # No more reviews

                reviews.extend(page_reviews)

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("Gartner: 2 consecutive pages with no new reviews, stopping")
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page + 1}: {exc}")
                logger.warning("Gartner page %d failed for %s: %s", page + 1, target.product_slug, exc)
                break

        logger.info(
            "Gartner scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# ------------------------------------------------------------------
# URL builder
# ------------------------------------------------------------------

def _build_reviews_url(product_slug: str, offset: int = 0) -> str:
    """Build the Gartner Peer Insights reviews URL.

    product_slug format: ``{market-slug}/{vendor-slug}``
    Result: ``https://www.gartner.com/reviews/market/{market-slug}/vendor/{vendor-slug}/reviews``
    """
    parts = product_slug.split("/", 1)
    if len(parts) == 2:
        market_slug, vendor_slug = parts
    else:
        # Fallback: treat entire slug as vendor under a generic market
        market_slug = parts[0]
        vendor_slug = parts[0]

    url = f"{_BASE_URL}/{market_slug}/vendor/{vendor_slug}/reviews"
    if offset > 0:
        url += f"?start={offset}"
    return url


# ------------------------------------------------------------------
# JSON-LD extraction (most reliable when present)
# ------------------------------------------------------------------

def _parse_json_ld(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Extract reviews from JSON-LD structured data.

    Gartner Peer Insights embeds Review schema in JSON-LD blocks.
    """
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

                review_body = r.get("reviewBody", "")
                if not review_body or len(review_body) < 20:
                    continue

                review_id = r.get("@id", "") or hashlib.sha256(
                    review_body.encode()
                ).hexdigest()[:16]
                if review_id in seen_ids:
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
                reviewer_name = None
                reviewer_title = None
                if isinstance(author, dict):
                    reviewer_name = author.get("name") or None
                    reviewer_title = author.get("jobTitle") or None

                # Extract date
                reviewed_at = r.get("datePublished")

                reviews.append({
                    "source": "gartner",
                    "source_url": _build_reviews_url(target.product_slug),
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
                    "reviewer_name": reviewer_name,
                    "reviewer_title": reviewer_title,
                    "reviewer_company": None,
                    "company_size_raw": None,
                    "reviewer_industry": None,
                    "reviewed_at": reviewed_at,
                    "raw_metadata": {
                        "extraction_method": "json_ld",
                        "source_weight": 1.0,
                        "source_type": "verified_review_platform",
                    },
                })

    return reviews


# ------------------------------------------------------------------
# HTML parsing (fallback when JSON-LD is absent or incomplete)
# ------------------------------------------------------------------

def _parse_html(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse a Gartner Peer Insights review page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Gartner review cards: look for common container patterns
    review_cards = soup.select(
        '[data-testid="review-card"], '
        '[class*="review-card"], '
        '[class*="ReviewCard"], '
        '[itemprop="review"]'
    )

    # Fallback selectors
    if not review_cards:
        review_cards = soup.select(
            'div[class*="review-listing"], '
            'div[class*="peer-review"], '
            'div[class*="reviewContent"]'
        )

    for card in review_cards:
        try:
            review = _parse_review_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse Gartner review card", exc_info=True)

    return reviews


def _parse_review_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a Gartner Peer Insights review card."""
    # Review ID -- try data attributes first, then generate from content
    review_id = (
        card.get("data-review-id", "")
        or card.get("data-testid", "")
        or card.get("id", "")
    )
    if not review_id:
        # Generate deterministic ID from card text content
        card_text = card.get_text(strip=True)[:300]
        if not card_text:
            return None
        review_id = f"gpi_{hashlib.sha256(card_text.encode()).hexdigest()[:16]}"

    # Star rating (1-5)
    rating = _extract_rating(card)

    # Review title / summary
    summary = _get_text(card, (
        '[class*="review-title"], '
        '[class*="ReviewTitle"], '
        '[itemprop="name"], '
        'h3, h4'
    ))

    # Pros and cons -- Gartner uses structured sections
    pros = _extract_gartner_section(card, "like", "pros", "strength", "best")
    cons = _extract_gartner_section(card, "dislike", "cons", "weakness", "improvement", "drawback")

    # Main review text
    review_text = ""
    body_el = card.select_one(
        '[itemprop="reviewBody"], '
        '[class*="review-body"], '
        '[class*="ReviewBody"], '
        '[class*="review-text"], '
        '[class*="reviewContent"]'
    )
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
    reviewer_name = _get_text(card, (
        '[itemprop="author"], '
        '[class*="reviewer-name"], '
        '[class*="ReviewerName"], '
        '[class*="author-name"]'
    ))
    reviewer_title = _get_text(card, (
        '[class*="reviewer-title"], '
        '[class*="ReviewerTitle"], '
        '[class*="job-title"], '
        '[class*="jobTitle"]'
    ))
    reviewer_company = _get_text(card, (
        '[class*="reviewer-company"], '
        '[class*="ReviewerCompany"], '
        '[class*="organization"], '
        '[class*="companyName"]'
    ))
    company_size = _get_text(card, (
        '[class*="company-size"], '
        '[class*="CompanySize"], '
        '[class*="employees"], '
        '[class*="revenue"]'
    ))
    reviewer_industry = _get_text(card, (
        '[class*="industry"], '
        '[class*="Industry"]'
    ))

    # Deployment experience (Gartner-specific field)
    deployment = _get_text(card, (
        '[class*="deployment"], '
        '[class*="Deployment"], '
        '[class*="experience"]'
    ))

    # Date
    reviewed_at = None
    date_el = card.select_one('time, [itemprop="datePublished"], [class*="date"], [class*="Date"]')
    if date_el:
        reviewed_at = (
            date_el.get("datetime")
            or date_el.get("content")
            or date_el.get_text(strip=True)
        )

    return {
        "source": "gartner",
        "source_url": f"{_build_reviews_url(target.product_slug)}#{review_id}",
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
            "deployment_experience": deployment,
        },
    }


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _extract_rating(card) -> float | None:
    """Extract star rating (1-5) from a review card.

    Gartner Peer Insights uses multiple rating patterns:
    - itemprop="ratingValue" with content attribute
    - aria-label on star elements (e.g. "4 out of 5 stars")
    - Star icon count (filled vs empty)
    - Class-based rating (e.g. "rating-4")
    """
    # Pattern 1: Schema.org itemprop
    rating_el = card.select_one('[itemprop="ratingValue"]')
    if rating_el:
        val = rating_el.get("content") or rating_el.get_text(strip=True)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Pattern 2: aria-label with rating text
    star_el = card.select_one('[aria-label*="star"], [aria-label*="rating"]')
    if star_el:
        aria = star_el.get("aria-label", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*(\d+)", aria)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

    # Pattern 3: class-based rating (e.g. "stars-4", "rating-4")
    for el in card.select('[class*="star"], [class*="rating"]'):
        classes = " ".join(el.get("class", []))
        match = re.search(r'(?:star|rating)[s-]*(\d)', classes)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

    # Pattern 4: Count filled star icons
    filled = card.select('[class*="star--filled"], [class*="starFilled"], .star.fill')
    if filled:
        return float(len(filled))

    return None


def _extract_gartner_section(card, *keywords: str) -> str | None:
    """Extract a Gartner review section (pros/cons/etc.) by keyword matching.

    Gartner Peer Insights structures reviews with labeled sections like
    "What do you like most?" / "What needs improvement?" followed by
    response content.
    """
    # Strategy 1: heading + sibling pattern
    for heading in card.select("h5, h4, h3, [class*='heading'], [class*='question'], [class*='label']"):
        text = heading.get_text(strip=True).lower()
        if any(kw in text for kw in keywords):
            # Get the next sibling with text content
            sibling = heading.find_next_sibling()
            if sibling:
                content = sibling.get_text(strip=True)
                if content and len(content) > 5:
                    return content[:5000]

    # Strategy 2: class-based matching
    for kw in keywords:
        el = card.select_one(f'[class*="{kw}"] p, [class*="{kw}"]')
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 5:
                return text[:5000]

    # Strategy 3: data-testid matching (React-based Gartner pages)
    for kw in keywords:
        el = card.select_one(f'[data-testid*="{kw}"]')
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
register_parser(GartnerParser())
