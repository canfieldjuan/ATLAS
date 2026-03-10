"""
PeerSpot parser for B2B review scraping.

URL pattern: peerspot.com/products/{slug}-reviews?page={n}
PeerSpot (formerly IT Central Station) hosts verified enterprise IT reviews
with structured Q&A sections (use case, value, improvement, advice).

Strategy: Web Unlocker first (Cloudflare-protected), curl_cffi HTTP fallback.
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

from bs4 import BeautifulSoup, Tag

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.peerspot")

_DOMAIN = "peerspot.com"
_BASE_URL = "https://www.peerspot.com/products"

# PeerSpot structured Q&A section headings mapped to semantic fields.
# "What is most valuable?" -> pros, "What needs improvement?" -> cons.
_SECTION_MAP: dict[str, str] = {
    "most valuable": "pros",
    "what is most valuable": "pros",
    "how has it helped": "benefits",
    "primary use case": "use_case",
    "needs improvement": "cons",
    "what needs improvement": "cons",
    "other advice": "advice",
    "what other advice": "advice",
}


class PeerSpotParser:
    """Parse PeerSpot review pages with Cloudflare bypass."""

    source_name = "peerspot"
    prefer_residential = True
    version = "peerspot:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape PeerSpot reviews -- Web Unlocker first, then HTTP client."""
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
        """Scrape PeerSpot via Bright Data Web Unlocker proxy."""
        import httpx
        from atlas_brain.config import settings

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}-reviews"
            if page > 1:
                url += f"?page={page}"

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+peerspot+reviews"
                if page == 1
                else f"{_BASE_URL}/{target.product_slug}-reviews?page={page - 1}"
                if page > 2
                else f"{_BASE_URL}/{target.product_slug}-reviews"
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
                    proxy=proxy_url, verify=False, timeout=90.0,
                ) as http:
                    resp = await http.get(url, headers=headers)

                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Page {page}: blocked (403) via Web Unlocker")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    continue

                # Try JSON-LD first, fall back to HTML
                page_reviews = _parse_json_ld(resp.text, target, seen_ids)
                if not page_reviews:
                    page_reviews = _parse_html(resp.text, target, seen_ids)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "PeerSpot Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "PeerSpot Web Unlocker page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                break

            # Inter-page delay
            await asyncio.sleep(random.uniform(2.0, 5.0))

        logger.info(
            "PeerSpot Web Unlocker scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    # ------------------------------------------------------------------
    # HTTP client path (curl_cffi + residential proxy)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape PeerSpot via curl_cffi HTTP client."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        consecutive_empty = 0
        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}-reviews"
            if page > 1:
                url += f"?page={page}"

            # Referer chain: Google for first page, previous page for subsequent
            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+peerspot+reviews"
                if page == 1
                else f"{_BASE_URL}/{target.product_slug}-reviews?page={page - 1}"
                if page > 2
                else f"{_BASE_URL}/{target.product_slug}-reviews"
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
                    errors.append(f"Page {page}: blocked (403) -- Cloudflare challenge")
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

                # Strategy 1: JSON-LD extraction (most reliable when present)
                page_reviews = _parse_json_ld(html, target, seen_ids)

                # Strategy 2: HTML fallback
                if not page_reviews:
                    page_reviews = _parse_html(html, target, seen_ids)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "PeerSpot page 1 returned 0 reviews for %s -- "
                            "JSON-LD and HTML selectors may be stale",
                            target.product_slug,
                        )
                    break  # No more reviews

                before = len(reviews)
                reviews.extend(page_reviews)

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("PeerSpot: 2 consecutive pages with no new reviews, stopping")
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("PeerSpot page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "PeerSpot scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


# ------------------------------------------------------------------
# JSON-LD extraction
# ------------------------------------------------------------------

def _parse_json_ld(
    html: str, target: ScrapeTarget, seen_ids: set[str],
) -> list[dict]:
    """Extract reviews from JSON-LD structured data (Product + Review aggregate)."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        for item in _iter_json_ld_items(data):
            # Look for Product/SoftwareApplication with embedded reviews
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
                reviewer_name = author.get("name", "") if isinstance(author, dict) else ""

                # Extract date
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
                    "source": "peerspot",
                    "source_url": f"https://www.peerspot.com/products/{target.product_slug}-reviews",
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
                        "source_weight": 0.9,
                        "source_type": "verified_review_platform",
                    },
                })

    return reviews


def _iter_json_ld_items(data: object) -> list[dict]:
    """Expand top-level JSON-LD items and @graph nodes into a flat dict list."""
    items = data if isinstance(data, list) else [data]
    expanded: list[dict] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        expanded.append(item)

        graph = item.get("@graph")
        if isinstance(graph, list):
            expanded.extend(node for node in graph if isinstance(node, dict))
        elif isinstance(graph, dict):
            expanded.append(graph)

    return expanded


def _extract_itemlist_notes(notes_obj: dict) -> list[str]:
    """Extract note text from schema.org ItemList note payloads."""
    items = notes_obj.get("itemListElement", [])
    if not isinstance(items, list):
        return []
    return [item["name"] for item in items if isinstance(item, dict) and item.get("name")]


# ------------------------------------------------------------------
# HTML parsing (primary extraction for full structured reviews)
# ------------------------------------------------------------------

def _parse_html(
    html: str, target: ScrapeTarget, seen_ids: set[str],
) -> list[dict]:
    """Parse PeerSpot review page HTML for structured review cards."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # PeerSpot review cards: look for common container patterns
    review_cards = soup.select(
        '[data-review-id], '
        '[class*="review-card"], '
        '[class*="ReviewCard"], '
        '[itemprop="review"]'
    )

    # Broader fallback selectors
    if not review_cards:
        review_cards = soup.select(
            'div[id^="review-"], '
            'div[class*="review-listing"], '
            'article[class*="review"]'
        )

    for card in review_cards:
        try:
            review = _parse_review_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse PeerSpot review card", exc_info=True)

    return reviews


def _parse_review_card(card: Tag, target: ScrapeTarget) -> dict | None:
    """Extract review data from a PeerSpot review card element."""
    # ------------------------------------------------------------------
    # Review ID
    # ------------------------------------------------------------------
    review_id = (
        card.get("data-review-id", "")
        or card.get("id", "")
    )
    if not review_id:
        # Generate deterministic ID from card content
        review_id = hashlib.sha256(
            card.get_text(strip=True)[:300].encode()
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Overall star rating (1-5)
    # ------------------------------------------------------------------
    rating = _extract_rating(card)

    # ------------------------------------------------------------------
    # Title / headline
    # ------------------------------------------------------------------
    summary = None
    title_el = card.select_one(
        '[itemprop="name"], '
        '[class*="review-title"], '
        '[class*="ReviewTitle"], '
        'h3, h4'
    )
    if title_el:
        summary = title_el.get_text(strip=True)[:500]

    # ------------------------------------------------------------------
    # Structured Q&A sections (PeerSpot's primary content format)
    # ------------------------------------------------------------------
    sections = _extract_qa_sections(card)

    pros = sections.get("pros")
    cons = sections.get("cons")
    use_case = sections.get("use_case")
    benefits = sections.get("benefits")
    advice = sections.get("advice")

    # Build composite review text from all available sections
    parts: list[str] = []
    if use_case:
        parts.append(f"Primary use case: {use_case}")
    if benefits:
        parts.append(f"How it helped: {benefits}")
    if pros:
        parts.append(f"Most valuable: {pros}")
    if cons:
        parts.append(f"Needs improvement: {cons}")
    if advice:
        parts.append(f"Advice: {advice}")

    review_text = "\n\n".join(parts)

    # Fallback: try the general review body if no structured sections found
    if not review_text:
        body_el = card.select_one(
            '[itemprop="reviewBody"], '
            '[class*="review-body"], '
            '[class*="ReviewBody"], '
            '[class*="review-content"]'
        )
        if body_el:
            review_text = body_el.get_text(strip=True)[:10000]

    if not review_text or len(review_text) < 20:
        return None

    # ------------------------------------------------------------------
    # Reviewer info
    # ------------------------------------------------------------------
    reviewer_name = _get_text(
        card,
        '[itemprop="author"], '
        '[class*="reviewer-name"], '
        '[class*="ReviewerName"], '
        '[class*="author-name"]',
    )
    reviewer_title = _get_text(
        card,
        '[class*="reviewer-title"], '
        '[class*="job-title"], '
        '[class*="JobTitle"], '
        '[class*="reviewer-role"]',
    )
    reviewer_company = _get_text(
        card,
        '[class*="reviewer-company"], '
        '[class*="CompanyName"], '
        '[class*="organization"]',
    )
    company_size = _get_text(
        card,
        '[class*="company-size"], '
        '[class*="CompanySize"], '
        '[class*="employees"]',
    )
    reviewer_industry = _get_text(
        card,
        '[class*="industry"], '
        '[class*="Industry"]',
    )

    # Years of experience (PeerSpot-specific metadata)
    experience = _get_text(
        card,
        '[class*="experience"], '
        '[class*="Experience"], '
        '[class*="years"]',
    )

    # ------------------------------------------------------------------
    # Date posted
    # ------------------------------------------------------------------
    reviewed_at = None
    date_el = card.select_one(
        'time, '
        '[itemprop="datePublished"], '
        '[class*="date"], '
        '[class*="Date"]'
    )
    if date_el:
        reviewed_at = (
            date_el.get("datetime")
            or date_el.get("content")
            or date_el.get_text(strip=True)
        )

    # Build raw_metadata with extra PeerSpot-specific fields
    raw_metadata: dict = {
        "extraction_method": "html",
        "source_weight": 0.9,
        "source_type": "verified_review_platform",
    }
    if experience:
        raw_metadata["years_of_experience"] = experience

    return {
        "source": "peerspot",
        "source_url": f"https://www.peerspot.com/products/{target.product_slug}-reviews#{review_id}",
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": summary,
        "review_text": review_text[:10000],
        "pros": pros[:5000] if pros else None,
        "cons": cons[:5000] if cons else None,
        "reviewer_name": reviewer_name,
        "reviewer_title": reviewer_title,
        "reviewer_company": reviewer_company,
        "company_size_raw": company_size,
        "reviewer_industry": reviewer_industry,
        "reviewed_at": reviewed_at,
        "raw_metadata": raw_metadata,
    }


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _extract_rating(card: Tag) -> float | None:
    """Extract star rating (1-5) from a review card.

    PeerSpot uses multiple rating markup patterns:
      - itemprop="ratingValue" content attribute
      - Star SVGs/icons with filled/active classes
      - aria-label on rating containers
      - Numeric text in a rating element
    """
    # Pattern 1: Schema.org ratingValue
    rating_el = card.select_one('[itemprop="ratingValue"]')
    if rating_el:
        val = rating_el.get("content") or rating_el.get_text(strip=True)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Pattern 2: aria-label with numeric rating
    for el in card.select('[class*="rating"], [class*="Rating"], [aria-label*="star"]'):
        aria = el.get("aria-label", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+", aria)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

    # Pattern 3: Count filled/active star elements
    filled_stars = card.select(
        '[class*="star--filled"], '
        '[class*="star-filled"], '
        '[class*="StarFilled"], '
        '.star.fill, '
        '.star.active'
    )
    if filled_stars:
        return float(len(filled_stars))

    # Pattern 4: Numeric text in rating container
    for el in card.select('[class*="rating"], [class*="Rating"]'):
        text = el.get_text(strip=True)
        match = re.match(r"^(\d+(?:\.\d+)?)$", text)
        if match:
            try:
                val = float(match.group(1))
                if 1.0 <= val <= 5.0:
                    return val
            except (ValueError, TypeError):
                pass

    return None


def _extract_qa_sections(card: Tag) -> dict[str, str]:
    """Extract PeerSpot structured Q&A sections from a review card.

    PeerSpot reviews contain structured headings like:
      - "What is our primary use case?"
      - "How has it helped my organization?"
      - "What is most valuable?"
      - "What needs improvement?"
      - "What other advice do I have?"

    Each heading is followed by the reviewer's detailed response.
    """
    sections: dict[str, str] = {}

    # Scan all heading-like elements for Q&A pattern
    headings = card.select(
        "h3, h4, h5, "
        "[class*='heading'], "
        "[class*='Heading'], "
        "[class*='question'], "
        "[class*='Question'], "
        "[class*='section-title'], "
        "[class*='SectionTitle']"
    )

    for heading in headings:
        heading_text = heading.get_text(strip=True).lower()

        # Match heading against known section keywords
        matched_key = None
        for keyword, section_key in _SECTION_MAP.items():
            if keyword in heading_text:
                matched_key = section_key
                break

        if not matched_key:
            continue

        # Collect text from sibling elements until the next heading
        content_parts: list[str] = []
        sibling = heading.find_next_sibling()
        while sibling:
            # Stop at the next heading-level element
            if isinstance(sibling, Tag) and sibling.name in ("h3", "h4", "h5"):
                break
            if isinstance(sibling, Tag) and any(
                cls for cls in (sibling.get("class") or [])
                if "heading" in cls.lower() or "question" in cls.lower()
                or "section-title" in cls.lower()
            ):
                break

            text = sibling.get_text(strip=True) if isinstance(sibling, Tag) else ""
            if text and len(text) > 3:
                content_parts.append(text)

            sibling = sibling.find_next_sibling()

        content = "\n".join(content_parts).strip()
        if content and len(content) > 5:
            # Keep first occurrence (don't overwrite if duplicate headings)
            if matched_key not in sections:
                sections[matched_key] = content[:5000]

    return sections


def _get_text(card: Tag, selector: str) -> str | None:
    """Safely extract text from the first matching element."""
    el = card.select_one(selector)
    if el:
        text = el.get_text(strip=True)
        if text:
            return text
    return None


# Auto-register
register_parser(PeerSpotParser())
