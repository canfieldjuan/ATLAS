"""
ProductHunt parser for B2B review scraping.

Uses ProductHunt's public GraphQL API (api.producthunt.com/v2/api/graphql)
as the primary extraction method. Falls back to HTML scraping at
producthunt.com/products/{slug}/reviews when the API is unavailable or
requires authentication.

Signal value: Tech-savvy early adopters -- useful for detecting switching
intent, feature gaps, and competitor adoption among B2B buyers.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.producthunt")

_DOMAIN = "producthunt.com"
_GRAPHQL_URL = "https://api.producthunt.com/v2/api/graphql"
_REVIEWS_URL = "https://www.producthunt.com/products/{slug}/reviews"
_MIN_BODY_LEN = 30  # Skip very short reviews


def _get_ph_token() -> str:
    """Load ProductHunt API bearer token from config or env."""
    try:
        from ....config import settings
        return settings.b2b_scrape.producthunt_api_token
    except Exception:
        pass
    import os
    return os.environ.get("ATLAS_B2B_SCRAPE_PRODUCTHUNT_API_TOKEN", "")


def _build_reviews_query(slug: str, first: int = 20, after: str | None = None) -> dict:
    """Build a GraphQL query payload for product reviews."""
    cursor_arg = f', after: "{after}"' if after else ""
    query = (
        "query {"
        f'  post(slug: "{slug}") {{'
        "    id"
        "    name"
        "    tagline"
        f"    reviews(first: {first}{cursor_arg}) {{"
        "      edges {"
        "        node {"
        "          id"
        "          body"
        "          rating"
        "          sentiment"
        "          user {"
        "            name"
        "            headline"
        "          }"
        "          createdAt"
        "        }"
        "        cursor"
        "      }"
        "      pageInfo {"
        "        hasNextPage"
        "        endCursor"
        "      }"
        "    }"
        "  }"
        "}"
    )
    return {"query": query}


def _parse_graphql_review(
    node: dict,
    target: ScrapeTarget,
    product_name: str | None,
    seen_ids: set[str],
) -> dict | None:
    """Parse a single GraphQL review node into a review dict."""
    review_id = str(node.get("id", ""))
    if not review_id or review_id in seen_ids:
        return None

    body = (node.get("body") or "").strip()
    if len(body) < _MIN_BODY_LEN:
        return None

    seen_ids.add(review_id)

    rating = node.get("rating")
    if rating is not None:
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None

    user = node.get("user") or {}
    created_at = node.get("createdAt")

    # Normalise ISO timestamp
    reviewed_at = None
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            reviewed_at = dt.isoformat()
        except (ValueError, TypeError):
            reviewed_at = created_at

    return {
        "source": "producthunt",
        "source_url": f"https://www.producthunt.com/products/{target.product_slug}/reviews",
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": product_name or target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": body[:500],
        "review_text": body[:10000],
        "pros": None,
        "cons": None,
        "reviewer_name": user.get("name") or None,
        "reviewer_title": user.get("headline") or None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": reviewed_at,
        "raw_metadata": {
            "extraction_method": "graphql_api",
            "source_weight": 0.6,
            "source_type": "product_discovery_platform",
            "sentiment": node.get("sentiment"),
        },
    }


class ProductHuntParser:
    """Parse ProductHunt reviews via GraphQL API with HTML scrape fallback."""

    source_name = "producthunt"
    prefer_residential = False  # API-based primary path needs no proxy
    version = "producthunt:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape ProductHunt reviews -- GraphQL API first, HTML fallback."""

        # Priority 1: GraphQL API (fast, structured, no proxy needed)
        try:
            result = await self._scrape_graphql(target)
            if result.reviews:
                return result
            logger.info(
                "ProductHunt GraphQL returned 0 reviews for %s, trying HTML fallback",
                target.product_slug,
            )
        except Exception as exc:
            logger.warning(
                "ProductHunt GraphQL failed for %s: %s -- falling back to HTML",
                target.product_slug, exc,
            )

        # Priority 2: HTML scrape (handles auth-gated GraphQL, changed API, etc.)
        return await self._scrape_html(target, client)

    # ------------------------------------------------------------------
    # GraphQL API path
    # ------------------------------------------------------------------

    async def _scrape_graphql(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape via ProductHunt public GraphQL API."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        after_cursor: str | None = None
        product_name: str | None = target.product_name
        page_size = 20

        async with httpx.AsyncClient(timeout=30) as http:
            for page in range(1, target.max_pages + 1):
                payload = _build_reviews_query(
                    slug=target.product_slug,
                    first=page_size,
                    after=after_cursor,
                )

                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "Atlas/1.0 B2B Intelligence",
                    }
                    token = _get_ph_token()
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                    resp = await http.post(
                        _GRAPHQL_URL,
                        json=payload,
                        headers=headers,
                    )
                    pages_scraped += 1

                    if resp.status_code == 401 or resp.status_code == 403:
                        errors.append(f"GraphQL page {page}: auth required ({resp.status_code})")
                        break
                    if resp.status_code == 429:
                        errors.append(f"GraphQL page {page}: rate limited (429)")
                        break
                    if resp.status_code != 200:
                        errors.append(f"GraphQL page {page}: HTTP {resp.status_code}")
                        break

                    data = resp.json()

                    # Check for GraphQL-level errors
                    gql_errors = data.get("errors")
                    if gql_errors:
                        error_msg = gql_errors[0].get("message", "unknown GraphQL error")
                        errors.append(f"GraphQL page {page}: {error_msg}")
                        break

                    post = (data.get("data") or {}).get("post")
                    if not post:
                        errors.append(f"GraphQL page {page}: post not found for slug '{target.product_slug}'")
                        break

                    # Capture product name from API response on first page
                    if page == 1 and post.get("name"):
                        product_name = post["name"]

                    reviews_data = post.get("reviews") or {}
                    edges = reviews_data.get("edges") or []

                    if not edges:
                        break  # No more reviews

                    for edge in edges:
                        node = edge.get("node") or {}
                        review = _parse_graphql_review(node, target, product_name, seen_ids)
                        if review:
                            reviews.append(review)

                    # Pagination
                    page_info = reviews_data.get("pageInfo") or {}
                    if not page_info.get("hasNextPage"):
                        break
                    after_cursor = page_info.get("endCursor")
                    if not after_cursor:
                        break

                except httpx.HTTPError as exc:
                    errors.append(f"GraphQL page {page}: {exc}")
                    logger.warning("PH GraphQL request failed page %d: %s", page, exc)
                    break

        logger.info(
            "ProductHunt GraphQL scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    # ------------------------------------------------------------------
    # HTML scrape fallback
    # ------------------------------------------------------------------

    async def _scrape_html(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Fallback: scrape ProductHunt review pages via HTML."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for page in range(1, target.max_pages + 1):
            url = _REVIEWS_URL.format(slug=target.product_slug)
            if page > 1:
                url += f"?page={page}"

            referer = (
                f"https://www.producthunt.com/products/{target.product_slug}"
                if page == 1
                else _REVIEWS_URL.format(slug=target.product_slug)
                if page == 2
                else f"{_REVIEWS_URL.format(slug=target.product_slug)}?page={page - 1}"
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
                    errors.append(f"HTML page {page}: blocked (403)")
                    break
                if resp.status_code != 200:
                    errors.append(f"HTML page {page}: HTTP {resp.status_code}")
                    continue

                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"HTML page {page}: unexpected content-type ({ct[:40]})")
                    break

                page_reviews = _parse_html_page(resp.text, target, seen_ids)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "ProductHunt HTML page 1 returned 0 reviews for %s -- selectors may be stale",
                            target.product_slug,
                        )
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"HTML page {page}: {exc}")
                logger.warning("PH HTML scrape page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "ProductHunt HTML scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# ------------------------------------------------------------------
# HTML parsing helpers
# ------------------------------------------------------------------


def _parse_html_page(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse a single ProductHunt HTML review page."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # ProductHunt renders reviews in structured divs.
    # Try multiple selector strategies in case the DOM changes.
    review_cards = soup.select('[data-test="review-card"], [class*="reviewCard"]')

    if not review_cards:
        # Fallback: look for common review container patterns
        review_cards = soup.select(
            '[class*="review-item"], '
            '[class*="ReviewItem"], '
            'div[class*="review"][class*="card"]'
        )

    if not review_cards:
        # Last resort: find divs that contain star ratings + body text
        for div in soup.find_all("div", recursive=True):
            if div.find(attrs={"class": lambda c: c and "star" in c.lower()}) and div.find("p"):
                review_cards.append(div)
            if len(review_cards) >= 50:
                break

    for card in review_cards:
        try:
            review = _parse_html_review_card(card, target, seen_ids)
            if review:
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse ProductHunt HTML review card", exc_info=True)

    return reviews


def _parse_html_review_card(
    card, target: ScrapeTarget, seen_ids: set[str]
) -> dict | None:
    """Extract review data from a single HTML review card."""
    # Extract review body text
    body_el = card.select_one(
        '[data-test="review-body"], '
        '[class*="reviewBody"], '
        '[class*="review-body"], '
        "p"
    )
    body = body_el.get_text(strip=True) if body_el else ""
    if len(body) < _MIN_BODY_LEN:
        return None

    # Generate a stable ID from the body content (PH HTML may not expose IDs)
    review_id = card.get("data-review-id") or card.get("id") or ""
    if not review_id:
        review_id = hashlib.sha256(body.encode()).hexdigest()[:16]

    if review_id in seen_ids:
        return None
    seen_ids.add(review_id)

    # Star rating (PH uses 1-5)
    rating = _extract_rating(card)

    # Reviewer info
    reviewer_name = _get_text(card, '[data-test="reviewer-name"], [class*="reviewerName"], [class*="user-name"]')
    reviewer_title = _get_text(card, '[data-test="reviewer-title"], [class*="reviewerTitle"], [class*="user-headline"]')

    # Date
    reviewed_at = None
    time_el = card.select_one("time, [datetime], [class*='date']")
    if time_el:
        reviewed_at = time_el.get("datetime") or time_el.get("title") or time_el.get_text(strip=True)

    return {
        "source": "producthunt",
        "source_url": _REVIEWS_URL.format(slug=target.product_slug),
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": body[:500],
        "review_text": body[:10000],
        "pros": None,
        "cons": None,
        "reviewer_name": reviewer_name,
        "reviewer_title": reviewer_title,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": reviewed_at,
        "raw_metadata": {
            "extraction_method": "html",
            "source_weight": 0.6,
            "source_type": "product_discovery_platform",
        },
    }


def _extract_rating(card) -> float | None:
    """Extract star rating from an HTML review card."""
    # Try explicit rating value attribute
    rating_el = card.select_one('[itemprop="ratingValue"], [data-rating]')
    if rating_el:
        raw = rating_el.get("content") or rating_el.get("data-rating")
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass

    # Count filled stars
    filled = card.select('[class*="star"][class*="filled"], [class*="StarFilled"], svg[class*="filled"]')
    if filled:
        return float(len(filled))

    # Aria label pattern (e.g. "Rating: 4 out of 5")
    for el in card.select('[aria-label*="rating" i], [aria-label*="star" i]'):
        label = el.get("aria-label", "")
        for word in label.split():
            try:
                val = float(word)
                if 1 <= val <= 5:
                    return val
            except ValueError:
                continue

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
register_parser(ProductHuntParser())
