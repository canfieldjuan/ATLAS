"""
Trustpilot parser for B2B review scraping.

URL pattern: trustpilot.com/review/{domain}?page={n}
Strategy: Web Unlocker first (Cloudflare protected), then curl_cffi HTTP fallback.
JSON-LD structured data extraction preferred, HTML parsing as fallback.

Trustpilot uses company domains as slugs (e.g., monday.com, salesforce.com).
Reviews are consumer-facing but high volume -- useful for sentiment at scale.
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
from . import (
    ScrapeResult,
    ScrapeTarget,
    apply_date_cutoff,
    log_page,
    page_has_only_known_source_reviews,
    register_parser,
)

logger = logging.getLogger("atlas.services.scraping.parsers.trustpilot")

_DOMAIN = "trustpilot.com"
_BASE_URL = "https://www.trustpilot.com/review"


class TrustpilotParser:
    """Parse Trustpilot review pages using JSON-LD or HTML fallback."""

    source_name = "trustpilot"
    prefer_residential = True
    version = "trustpilot:2"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Trustpilot reviews -- Web Unlocker first, then HTTP client."""
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
        """Scrape Trustpilot via Bright Data Web Unlocker proxy."""
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
        stop_reason = ""

        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}"
            if page > 1:
                url += f"?page={page}"

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
                if resp.status_code == 404:
                    errors.append(f"Page {page}: HTTP 404")
                    page_logs.append(log_page(
                        page, url, status_code=404, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["HTTP 404"],
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
                page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)

                pl = log_page(
                    page, url, status_code=200, duration_ms=elapsed_ms,
                    response_bytes=len(resp.content), reviews=page_reviews,
                    raw_body=resp.content, prior_hashes=prior_hashes,
                    prior_review_ids=prior_review_ids,
                    next_page_found=bool(page_reviews),
                )
                if cutoff_hit:
                    pl.stop_reason = "date_cutoff"
                    stop_reason = "date_cutoff"
                page_logs.append(pl)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Trustpilot Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    break
                elif page_has_only_known_source_reviews(page_reviews, target):
                    pl.stop_reason = "known_source_reviews"
                    stop_reason = "known_source_reviews"
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "Trustpilot Web Unlocker page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                break

            await asyncio.sleep(random.uniform(2.0, 5.0))

        logger.info(
            "Trustpilot Web Unlocker scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            page_logs=page_logs,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # HTTP client path (curl_cffi + residential proxy)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Trustpilot via curl_cffi HTTP client."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time
        stop_reason = ""

        consecutive_empty = 0
        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}"
            if page > 1:
                url += f"?page={page}"

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+trustpilot+reviews"
                if page == 1
                else f"{_BASE_URL}/{target.product_slug}?page={page - 1}"
                if page > 2
                else f"{_BASE_URL}/{target.product_slug}"
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
                if resp.status_code == 404:
                    errors.append(f"Page {page}: HTTP 404")
                    page_logs.append(log_page(
                        page, url, status_code=404, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content or b""), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["HTTP 404"],
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

                # Guard against non-HTML responses
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page}: unexpected content-type ({ct[:40]})")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        errors=[f"unexpected content-type ({ct[:40]})"],
                    ))
                    break

                html = resp.text

                # Strategy 1: JSON-LD extraction (most reliable)
                page_reviews = _parse_json_ld(html, target, seen_ids)

                # Strategy 2: HTML fallback
                if not page_reviews:
                    page_reviews = _parse_html(html, target, seen_ids)

                page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)

                pl = log_page(
                    page, url, status_code=200, duration_ms=elapsed_ms,
                    response_bytes=len(resp.content or b""), reviews=page_reviews,
                    raw_body=resp.content, prior_hashes=prior_hashes,
                    prior_review_ids=prior_review_ids,
                    next_page_found=bool(page_reviews),
                )
                if cutoff_hit:
                    pl.stop_reason = "date_cutoff"
                    stop_reason = "date_cutoff"
                page_logs.append(pl)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Trustpilot page 1 returned 0 reviews for %s -- "
                            "JSON-LD and HTML selectors may be stale",
                            target.product_slug,
                        )
                    break  # No more reviews
                elif page_has_only_known_source_reviews(page_reviews, target):
                    pl.stop_reason = "known_source_reviews"
                    stop_reason = "known_source_reviews"
                    break

                before = len(reviews)
                reviews.extend(page_reviews)

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("Trustpilot: 2 consecutive pages with no new reviews, stopping")
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("Trustpilot page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "Trustpilot scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            page_logs=page_logs,
            stop_reason=stop_reason,
        )


# ------------------------------------------------------------------
# JSON-LD extraction (primary strategy)
# ------------------------------------------------------------------

def _parse_json_ld(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Extract reviews from JSON-LD structured data.

    Trustpilot embeds @type: Review objects within JSON-LD scripts,
    typically nested inside a LocalBusiness or Organization entity.
    Each Review contains reviewRating, author, datePublished, and
    reviewBody fields per schema.org.
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
            # Trustpilot may nest reviews under @graph or directly on the entity
            review_list = _collect_reviews_from_item(item)

            for r in review_list:
                if not isinstance(r, dict):
                    continue
                if r.get("@type") not in ("Review", "http://schema.org/Review"):
                    continue

                review_body = r.get("reviewBody", "")
                if not review_body or len(review_body) < 20:
                    continue

                # Generate deterministic ID from content + date
                id_seed = (review_body + (r.get("datePublished", "") or "")).encode()
                review_id = r.get("@id", "") or hashlib.sha256(id_seed).hexdigest()[:16]
                if review_id in seen_ids:
                    continue
                seen_ids.add(review_id)

                # Extract rating
                rating = None
                rating_max = 5
                rating_obj = r.get("reviewRating", {})
                if isinstance(rating_obj, dict):
                    rating_val = rating_obj.get("ratingValue")
                    if rating_val is not None:
                        try:
                            rating = float(rating_val)
                        except (ValueError, TypeError):
                            pass
                    best = rating_obj.get("bestRating")
                    if best is not None:
                        try:
                            rating_max = int(best)
                        except (ValueError, TypeError):
                            pass

                # Extract author
                author = r.get("author", {})
                reviewer_name = author.get("name", "") if isinstance(author, dict) else ""

                # Extract date
                reviewed_at = r.get("datePublished")

                # Extract headline / title
                headline = r.get("name") or r.get("headline")

                # Publisher is the reviewed business or page owner, not the reviewer employer.
                publisher = r.get("publisher", {})
                publisher_name = publisher.get("name") if isinstance(publisher, dict) else None
                raw_metadata = {
                    "extraction_method": "json_ld",
                    "source_weight": 0.7,
                    "source_type": "consumer_review_platform",
                }
                if publisher_name:
                    raw_metadata["publisher_name"] = publisher_name

                reviews.append({
                    "source": "trustpilot",
                    "source_url": f"https://www.trustpilot.com/review/{target.product_slug}",
                    "source_review_id": review_id,
                    "vendor_name": target.vendor_name,
                    "product_name": target.product_name or item.get("name"),
                    "product_category": target.product_category,
                    "rating": rating,
                    "rating_max": rating_max,
                    "summary": headline,
                    "review_text": review_body[:10000],
                    "pros": None,
                    "cons": None,
                    "reviewer_name": reviewer_name or None,
                    "reviewer_title": None,
                    "reviewer_company": None,
                    "company_size_raw": None,
                    "reviewer_industry": None,
                    "reviewed_at": reviewed_at,
                    "raw_metadata": raw_metadata,
                })

    return reviews


def _collect_reviews_from_item(item: dict) -> list[dict]:
    """Recursively collect Review objects from a JSON-LD item.

    Trustpilot structures can vary:
      - Direct ``review`` array on an entity
      - ``@graph`` array containing Review objects
      - Single Review at top level
    """
    collected: list[dict] = []

    # Direct review property
    review_list = item.get("review", [])
    if isinstance(review_list, dict):
        review_list = [review_list]
    if isinstance(review_list, list):
        collected.extend(review_list)

    # @graph array (common in modern JSON-LD)
    graph = item.get("@graph", [])
    if isinstance(graph, list):
        for node in graph:
            if isinstance(node, dict):
                if node.get("@type") in ("Review", "http://schema.org/Review"):
                    collected.append(node)
                # Also recurse into graph nodes that have reviews
                nested = node.get("review", [])
                if isinstance(nested, dict):
                    nested = [nested]
                if isinstance(nested, list):
                    collected.extend(nested)

    # Top-level Review
    if item.get("@type") in ("Review", "http://schema.org/Review"):
        collected.append(item)

    return collected


# ------------------------------------------------------------------
# HTML parsing (fallback strategy)
# ------------------------------------------------------------------

def _parse_html(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse Trustpilot review page HTML as fallback.

    Trustpilot review cards are typically rendered as ``<article>`` tags
    with data-service-review-card-paper attributes, or ``<div>`` elements
    with class names containing "review-card".
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Primary selectors for Trustpilot review cards
    review_cards = soup.select(
        'article[data-service-review-card-paper], '
        'div[class*="review-card"], '
        'article[class*="review-card"], '
        'section[class*="review-card"]'
    )

    if not review_cards:
        # Broader fallback selectors
        review_cards = soup.select(
            'div[data-review-id], '
            'article[data-review-id], '
            'div[class*="styles_reviewCard"]'
        )

    for card in review_cards:
        try:
            review = _parse_trustpilot_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse Trustpilot review card", exc_info=True)

    return reviews


def _parse_trustpilot_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a Trustpilot review card element."""
    # Review ID -- check data attributes first, fall back to content hash
    review_id = (
        card.get("id", "")
        or card.get("data-review-id", "")
        or card.get("data-service-review-business-unit-review-id", "")
    )
    if not review_id:
        card_text = card.get_text(strip=True)[:200]
        if not card_text:
            return None
        review_id = hashlib.sha256(card_text.encode()).hexdigest()[:16]

    # Rating -- Trustpilot uses star images or data-rating attributes
    rating = _extract_rating(card)

    # Review title / headline
    summary = None
    title_el = card.select_one(
        'h2, '
        '[class*="title"], '
        '[data-service-review-title-typography], '
        'a[class*="title"]'
    )
    if title_el:
        summary = title_el.get_text(strip=True)
        if summary and len(summary) > 500:
            summary = summary[:500]

    # Review body text
    review_text = ""
    body_el = card.select_one(
        '[data-service-review-text-typography], '
        '[class*="reviewBody"], '
        '[class*="review-content"], '
        '[class*="styles_reviewContent"], '
        'p[class*="text"]'
    )
    if body_el:
        review_text = body_el.get_text(strip=True)

    # If body is empty, try broader text extraction (skip the header/meta)
    if not review_text:
        for p_tag in card.select("p"):
            text = p_tag.get_text(strip=True)
            if text and len(text) > 30:
                review_text = text
                break

    if not review_text or len(review_text) < 20:
        return None

    # Reviewer name
    reviewer_name = _get_text(
        card,
        '[data-consumer-name-typography], '
        '[class*="consumerName"], '
        '[class*="reviewer-name"], '
        'span[class*="name"]'
    )

    # Reviewer location (Trustpilot shows location, not company)
    reviewer_location = _get_text(
        card,
        '[data-consumer-country-typography], '
        '[class*="consumerLocation"], '
        '[class*="reviewer-location"]'
    )

    # Number of reviews by this user
    review_count_text = _get_text(
        card,
        '[data-consumer-reviews-count-typography], '
        '[class*="reviewsCount"], '
        '[class*="consumer-review-count"]'
    )

    # Date
    reviewed_at = None
    date_el = card.select_one(
        'time, '
        '[data-service-review-date-time-ago], '
        '[class*="reviewDate"], '
        '[datetime]'
    )
    if date_el:
        reviewed_at = (
            date_el.get("datetime")
            or date_el.get("title")
            or date_el.get_text(strip=True)
        )

    # Build raw_metadata with extra fields
    raw_meta: dict = {
        "extraction_method": "html",
        "source_weight": 0.7,
        "source_type": "consumer_review_platform",
    }
    if reviewer_location:
        raw_meta["reviewer_location"] = reviewer_location
    if review_count_text:
        raw_meta["reviewer_review_count"] = review_count_text

    return {
        "source": "trustpilot",
        "source_url": f"https://www.trustpilot.com/review/{target.product_slug}",
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": summary,
        "review_text": review_text[:10000],
        "pros": None,
        "cons": None,
        "reviewer_name": reviewer_name,
        "reviewer_title": None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": reviewed_at,
        "raw_metadata": raw_meta,
    }


def _extract_rating(card) -> float | None:
    """Extract star rating from a Trustpilot review card.

    Trustpilot encodes ratings in several ways:
      - ``img[alt*="Rated"]`` with alt text like "Rated 4 out of 5 stars"
      - ``div[data-rating]`` attribute
      - Star count via filled star SVGs or class names
    """
    # Method 1: img alt text (e.g., "Rated 4 out of 5 stars")
    img_el = card.select_one('img[alt*="Rated"], img[alt*="rated"]')
    if img_el:
        alt = img_el.get("alt", "")
        match = re.search(r"[Rr]ated\s+(\d+(?:\.\d+)?)", alt)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

    # Method 2: data-rating attribute
    rating_el = card.select_one("[data-rating]")
    if rating_el:
        try:
            return float(rating_el["data-rating"])
        except (ValueError, TypeError):
            pass

    # Method 3: Star count from CSS classes or SVG fills
    star_el = card.select_one('[class*="star-rating"], [class*="starRating"]')
    if star_el:
        # Trustpilot often uses class like "star-rating--medium-4" for 4 stars
        class_str = " ".join(star_el.get("class", []))
        match = re.search(r"(?:star-rating--\w+-|starRating-)(\d)", class_str)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

    # Method 4: aria-label on star container
    aria_el = card.select_one('[aria-label*="star"], [aria-label*="rating"]')
    if aria_el:
        aria = aria_el.get("aria-label", "")
        match = re.search(r"(\d+(?:\.\d+)?)", aria)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

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
register_parser(TrustpilotParser())
