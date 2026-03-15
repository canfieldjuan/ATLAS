"""
Software Advice parser for B2B review scraping.

URL pattern: softwareadvice.com/{category}/{product-slug}/reviews/
Owned by Gartner Digital Markets (same family as Capterra, GetApp).
Strategy: Web Unlocker first (Cloudflare protected), JSON-LD extraction,
fall back to HTML parsing via curl_cffi HTTP client.
Residential proxy required.
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
from . import ScrapeResult, ScrapeTarget, log_page, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.software_advice")

_DOMAIN = "softwareadvice.com"
_BASE_URL = "https://www.softwareadvice.com"


class SoftwareAdviceParser:
    """Parse Software Advice review pages using JSON-LD or HTML fallback."""

    source_name = "software_advice"
    prefer_residential = True
    version = "software_advice:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Software Advice reviews -- Web Unlocker first, then HTTP client."""
        from atlas_brain.config import settings

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

        return await self._scrape_http(target, client)

    # ------------------------------------------------------------------
    # Web Unlocker path
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape Software Advice via Bright Data Web Unlocker proxy."""
        import httpx
        from atlas_brain.config import settings

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time

        for page in range(1, target.max_pages + 1):
            url = _build_url(target.product_slug, page)

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }

            page_start = _time.monotonic()
            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url, verify=False, timeout=90
                ) as http:
                    resp = await http.get(url, headers=headers)

                pages_scraped += 1
                elapsed_ms = int((_time.monotonic() - page_start) * 1000)

                if resp.status_code == 403:
                    errors.append(f"Page {page}: blocked (403) via Web Unlocker")
                    page_logs.append(log_page(
                        page, url, status_code=403, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["blocked (403)"],
                    ))
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=[f"HTTP {resp.status_code}"],
                    ))
                    continue

                page_reviews = _parse_json_ld(resp.text, target, seen_ids)
                if not page_reviews:
                    page_reviews = _parse_html(resp.text, target, seen_ids)

                page_logs.append(log_page(
                    page, url, status_code=200, duration_ms=elapsed_ms,
                    response_bytes=len(resp.content), reviews=page_reviews,
                    raw_body=resp.content, prior_hashes=prior_hashes,
                    prior_review_ids=prior_review_ids,
                    next_page_found=bool(page_reviews),
                ))

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Software Advice Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "Software Advice Web Unlocker page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                break

            await asyncio.sleep(random.uniform(2.0, 5.0))

        logger.info(
            "Software Advice Web Unlocker scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors, page_logs=page_logs)

    # ------------------------------------------------------------------
    # HTTP client path (curl_cffi + residential proxy)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Software Advice via curl_cffi HTTP client."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time

        consecutive_empty = 0
        for page in range(1, target.max_pages + 1):
            url = _build_url(target.product_slug, page)

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+software+advice+reviews"
                if page == 1
                else _build_url(target.product_slug, page - 1)
                if page > 2
                else _build_url(target.product_slug, 1)
            )

            page_start = _time.monotonic()
            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=referer,
                    sticky_session=True,
                    prefer_residential=True,
                )
                pages_scraped += 1
                elapsed_ms = int((_time.monotonic() - page_start) * 1000)

                if resp.status_code == 403:
                    errors.append(f"Page {page}: blocked (403) -- Cloudflare challenge")
                    page_logs.append(log_page(
                        page, url, status_code=403, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content or b""), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["blocked (403)"],
                    ))
                    break
                if resp.status_code == 429:
                    errors.append(f"Page {page}: rate limited (429)")
                    page_logs.append(log_page(
                        page, url, status_code=429, duration_ms=elapsed_ms,
                        prior_hashes=prior_hashes, errors=["rate limited (429)"],
                    ))
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        prior_hashes=prior_hashes, errors=[f"HTTP {resp.status_code}"],
                    ))
                    continue

                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page}: unexpected content-type ({ct[:40]})")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        errors=[f"unexpected content-type ({ct[:40]})"],
                    ))
                    break

                html = resp.text

                page_reviews = _parse_json_ld(html, target, seen_ids)
                if not page_reviews:
                    page_reviews = _parse_html(html, target, seen_ids)

                page_logs.append(log_page(
                    page, url, status_code=200, duration_ms=elapsed_ms,
                    response_bytes=len(resp.content or b""), reviews=page_reviews,
                    raw_body=resp.content, prior_hashes=prior_hashes,
                    prior_review_ids=prior_review_ids,
                    next_page_found=bool(page_reviews),
                ))

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Software Advice page 1 returned 0 reviews for %s -- "
                            "JSON-LD and HTML selectors may be stale",
                            target.product_slug,
                        )
                    break

                before = len(reviews)
                reviews.extend(page_reviews)

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("Software Advice: 2 consecutive pages with no new reviews, stopping")
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("Software Advice page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "Software Advice scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors, page_logs=page_logs)


# ------------------------------------------------------------------
# URL helpers
# ------------------------------------------------------------------

def _build_url(product_slug: str, page: int) -> str:
    """Build a Software Advice reviews URL.

    product_slug in DB is ``{category}/{product-slug}``
    (e.g. ``crm-software/salesforce-profile``).
    Final URL: https://www.softwareadvice.com/{category}/{product-slug}/reviews/
    """
    base = f"{_BASE_URL}/{product_slug}/reviews/"
    if page > 1:
        return f"{base}?page={page}"
    return base


# ------------------------------------------------------------------
# JSON-LD extraction (primary strategy -- Gartner Digital Markets schema)
# ------------------------------------------------------------------

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

        items = data if isinstance(data, list) else [data]

        for item in items:
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

                rating = None
                rating_obj = r.get("reviewRating", {})
                if isinstance(rating_obj, dict):
                    rating_val = rating_obj.get("ratingValue")
                    if rating_val is not None:
                        try:
                            rating = float(rating_val)
                        except (ValueError, TypeError):
                            pass

                author = r.get("author", {})
                reviewer_name = author.get("name", "") if isinstance(author, dict) else ""

                reviewed_at = r.get("datePublished")

                pros = None
                cons = None
                if isinstance(r.get("positiveNotes"), str):
                    pros = r["positiveNotes"][:5000]
                elif isinstance(r.get("positiveNotes"), dict):
                    notes = _extract_itemlist_notes(r["positiveNotes"])
                    if notes:
                        pros = "; ".join(notes)[:5000]

                if isinstance(r.get("negativeNotes"), str):
                    cons = r["negativeNotes"][:5000]
                elif isinstance(r.get("negativeNotes"), dict):
                    notes = _extract_itemlist_notes(r["negativeNotes"])
                    if notes:
                        cons = "; ".join(notes)[:5000]

                reviews.append({
                    "source": "software_advice",
                    "source_url": _build_url(target.product_slug, 1),
                    "source_review_id": review_id,
                    "vendor_name": target.vendor_name,
                    "product_name": target.product_name or item.get("name"),
                    "product_category": target.product_category,
                    "rating": rating,
                    "rating_max": 5,
                    "summary": r.get("name") or r.get("headline"),
                    "review_text": review_body[:10000],
                    "pros": pros,
                    "cons": cons,
                    "reviewer_name": reviewer_name or None,
                    "reviewer_title": None,
                    "reviewer_company": None,
                    "company_size_raw": None,
                    "reviewer_industry": None,
                    "reviewed_at": reviewed_at,
                    "raw_metadata": {
                        "extraction_method": "json_ld",
                        "source_weight": 0.85,
                        "source_type": "verified_review_platform",
                    },
                })

    return reviews


def _extract_itemlist_notes(notes_obj: dict) -> list[str]:
    """Extract note names from a JSON-LD ItemList (Gartner schema)."""
    items = notes_obj.get("itemListElement", [])
    if not isinstance(items, list):
        return []
    return [item["name"] for item in items if isinstance(item, dict) and item.get("name")]


# ------------------------------------------------------------------
# HTML fallback
# ------------------------------------------------------------------

def _parse_html(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse Software Advice review page HTML as fallback."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    review_cards = soup.select(
        '[data-testid="review-card"], '
        '[data-review-id], '
        '.review-card, '
        '[class*="ReviewCard"], '
        'div[class*="review-"][class*="card"]'
    )

    if not review_cards:
        review_cards = soup.select(
            'div[id^="review-"], '
            'div[class*="review-content"], '
            'div[class*="review-listing"], '
            '[itemtype*="Review"]'
        )

    for card in review_cards:
        try:
            review = _parse_review_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse Software Advice review card", exc_info=True)

    return reviews


def _parse_review_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a Software Advice review card."""
    review_id = (
        card.get("data-review-id", "")
        or card.get("id", "")
    )
    if not review_id:
        review_id = hashlib.sha256(
            card.get_text(strip=True)[:200].encode()
        ).hexdigest()[:16]

    rating = _extract_rating(card)

    summary = None
    title_el = card.select_one(
        '[class*="review-title"], [class*="ReviewTitle"], '
        '[itemprop="name"], h3, h4'
    )
    if title_el:
        summary = title_el.get_text(strip=True)[:500]

    pros = _extract_pros_cons(card, "pros", "like", "best", "advantage")
    cons = _extract_pros_cons(card, "cons", "dislike", "worst", "disadvantage")

    overall_text = ""
    for text_el in card.select(
        '[itemprop="reviewBody"], '
        '[class*="review-text"], '
        '[class*="ReviewText"], '
        '[class*="review-body"], '
        '[class*="overall"]'
    ):
        text = text_el.get_text(strip=True)
        if text and len(text) > 20:
            overall_text = text[:10000]
            break

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

    reviewer_name = _get_text(card, '[class*="reviewer"], [class*="author"], [itemprop="author"]')
    reviewer_title = _get_text(card, '[class*="title"], [class*="job"], [class*="role"]')
    reviewer_company = _get_text(card, '[class*="company"], [class*="org"]')
    company_size = _get_text(card, '[class*="size"], [class*="employees"]')
    reviewer_industry = _get_text(card, '[class*="industry"], [class*="sector"]')

    reviewed_at = None
    date_el = card.select_one("time, [class*='date'], [itemprop='datePublished']")
    if date_el:
        reviewed_at = (
            date_el.get("datetime")
            or date_el.get("content")
            or date_el.get_text(strip=True)
        )

    return {
        "source": "software_advice",
        "source_url": _build_url(target.product_slug, 1),
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": summary,
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
            "source_weight": 0.85,
            "source_type": "verified_review_platform",
        },
    }


# ------------------------------------------------------------------
# HTML helper functions
# ------------------------------------------------------------------

def _extract_rating(card) -> float | None:
    """Extract star rating from a review card."""
    rating_el = card.select_one('[itemprop="ratingValue"]')
    if rating_el:
        val = rating_el.get("content") or rating_el.get_text(strip=True)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    for el in card.select('[class*="star"], [class*="rating"], [aria-label*="star"]'):
        aria = el.get("aria-label", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+", aria)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass
        match = re.search(r"(\d+(?:\.\d+)?)", aria)
        if match:
            try:
                rating = float(match.group(1))
                if 1.0 <= rating <= 5.0:
                    return rating
            except (ValueError, TypeError):
                pass

    filled = card.select('[class*="star--filled"], [class*="star-full"], .star.fill')
    if filled:
        count = len(filled)
        if 1 <= count <= 5:
            return float(count)

    return None


def _extract_pros_cons(card, *keywords: str) -> str | None:
    """Extract pros or cons section from a review card."""
    for heading in card.select("h5, h4, h3, [class*='heading'], [class*='label'], dt"):
        text = heading.get_text(strip=True).lower()
        if any(kw in text for kw in keywords):
            sibling = heading.find_next_sibling()
            if sibling:
                content = sibling.get_text(strip=True)
                if content and len(content) > 5:
                    return content[:5000]

    for kw in keywords:
        for el in card.select(f'[class*="{kw}"] p, [class*="{kw}"], [data-testid*="{kw}"]'):
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
register_parser(SoftwareAdviceParser())
