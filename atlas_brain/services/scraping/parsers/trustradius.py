"""
TrustRadius parser for B2B review scraping.

Strategy (in priority order):
  1. Bright Data Web Unlocker (handles JS rendering + anti-bot automatically)
  2. Firecrawl for JS-rendered page scraping (paid API fallback)
  3. JSON-LD aggregate if neither is available
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

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
    """Parse TrustRadius reviews via Web Unlocker, Firecrawl, or JSON-LD."""

    source_name = "trustradius"
    prefer_residential = True
    version = "trustradius:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape TrustRadius reviews for the given product."""
        from ....config import settings

        # Priority 1: Bright Data Web Unlocker (handles JS rendering + anti-bot)
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

        # Priority 2: Firecrawl JS rendering
        api_key = _get_firecrawl_key()
        if api_key:
            try:
                return await self._scrape_firecrawl(target, api_key)
            except Exception as exc:
                logger.warning(
                    "Firecrawl scrape failed for %s, falling back to JSON-LD: %s",
                    target.product_slug, exc,
                )

        # Priority 3: JSON-LD aggregate (1 synthetic review)
        return await self._scrape_jsonld_fallback(target, client)

    # ------------------------------------------------------------------
    # Web Unlocker path (Bright Data -- handles JS rendering + anti-bot)
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape TrustRadius via Bright Data Web Unlocker proxy."""
        import httpx
        from ....config import settings

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for page in range(1, target.max_pages + 1):
            url = f"{_PRODUCT_BASE}/{target.product_slug}/reviews"
            if page > 1:
                url += f"?p={page}"

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": (
                    f"https://www.google.com/search"
                    f"?q={quote_plus(target.vendor_name)}+trustradius+reviews"
                    if page == 1
                    else f"{_PRODUCT_BASE}/{target.product_slug}/reviews"
                ),
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
                if resp.status_code == 404:
                    errors.append(f"Product slug not found: {target.product_slug}")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    continue

                html = resp.text

                # Strategy 1: JSON-LD extraction (structured data)
                page_reviews = _parse_jsonld_reviews(html, target, seen_ids)

                # Strategy 2: HTML review card extraction
                if not page_reviews:
                    page_reviews = _parse_html_reviews(html, target, seen_ids)

                if not page_reviews:
                    if page == 1:
                        # Page 1 empty -- try JSON-LD aggregate as last resort
                        agg_reviews = _extract_jsonld_product(html, target)
                        if agg_reviews:
                            reviews.extend(agg_reviews)
                        else:
                            logger.warning(
                                "TrustRadius Web Unlocker page 1 returned 0 reviews for %s",
                                target.product_slug,
                            )
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "TrustRadius Web Unlocker page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                break

            await asyncio.sleep(random.uniform(2.0, 5.0))

        logger.info(
            "TrustRadius Web Unlocker scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

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


# ------------------------------------------------------------------
# HTML/JSON-LD parsing for Web Unlocker responses
# ------------------------------------------------------------------


def _parse_jsonld_reviews(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Extract individual reviews from JSON-LD structured data.

    TrustRadius embeds SoftwareApplication JSON-LD with a ``review`` array
    containing individual Review objects with reviewBody, pros/cons, etc.
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            # TrustRadius nests reviews under SoftwareApplication
            review_list = item.get("review", [])
            if isinstance(review_list, dict):
                review_list = [review_list]
            if not isinstance(review_list, list):
                continue

            for r in review_list:
                if not isinstance(r, dict):
                    continue
                if r.get("@type") not in ("Review", "http://schema.org/Review"):
                    continue

                review_body = r.get("reviewBody", "")
                if not review_body or len(review_body) < 20:
                    continue

                # Deterministic ID from content
                id_seed = (review_body + (r.get("datePublished", "") or "")).encode()
                review_id = hashlib.sha256(id_seed).hexdigest()[:16]
                if review_id in seen_ids:
                    continue
                seen_ids.add(review_id)

                # Rating (TrustRadius uses 1-10 scale)
                rating = None
                rating_max = 10
                rating_obj = r.get("reviewRating", {})
                if isinstance(rating_obj, dict):
                    rv = rating_obj.get("ratingValue")
                    if rv is not None:
                        try:
                            rating = float(rv)
                        except (ValueError, TypeError):
                            pass
                    best = rating_obj.get("bestRating")
                    if best is not None:
                        try:
                            rating_max = int(best)
                        except (ValueError, TypeError):
                            pass

                # Author
                author = r.get("author", {})
                reviewer_name = author.get("name") if isinstance(author, dict) else None

                # Date
                reviewed_at = r.get("datePublished")

                # Headline
                headline = r.get("name") or r.get("headline")

                # Pros/cons from positiveNotes/negativeNotes
                pros_list = _extract_notes(r.get("positiveNotes", {}))
                cons_list = _extract_notes(r.get("negativeNotes", {}))

                reviews.append({
                    "source": "trustradius",
                    "source_url": f"{_PRODUCT_BASE}/{target.product_slug}/reviews",
                    "source_review_id": review_id,
                    "vendor_name": target.vendor_name,
                    "product_name": target.product_name or item.get("name"),
                    "product_category": target.product_category,
                    "rating": rating,
                    "rating_max": rating_max,
                    "summary": headline[:500] if headline else None,
                    "review_text": review_body[:10000],
                    "pros": ", ".join(pros_list) if pros_list else None,
                    "cons": ", ".join(cons_list) if cons_list else None,
                    "reviewer_name": reviewer_name,
                    "reviewer_title": None,
                    "reviewer_company": None,
                    "company_size_raw": None,
                    "reviewer_industry": None,
                    "reviewed_at": reviewed_at,
                    "raw_metadata": {
                        "extraction_method": "json_ld",
                        "source_weight": 0.8,
                        "source_type": "verified_review_platform",
                    },
                })

    return reviews


def _parse_html_reviews(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse TrustRadius review cards from rendered HTML.

    TrustRadius review cards use various selectors including
    data-testid attributes, itemprop annotations, and class-based patterns.
    """
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Primary selectors for TrustRadius review cards
    review_cards = soup.select(
        '[data-testid="review-card"], '
        '[class*="review-card"], '
        '[class*="ReviewCard"], '
        '[itemprop="review"]'
    )

    if not review_cards:
        # Broader fallback: look for sections with review-like content
        review_cards = soup.select(
            '[data-review-id], '
            'div[class*="reviewContent"], '
            'article[class*="review"]'
        )

    for card in review_cards:
        try:
            review = _parse_trustradius_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse TrustRadius review card", exc_info=True)

    return reviews


def _parse_trustradius_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a TrustRadius review card element."""
    # Review ID
    review_id = (
        card.get("data-review-id", "")
        or card.get("data-testid", "")
        or card.get("id", "")
    )
    if not review_id:
        card_text = card.get_text(strip=True)[:200]
        if not card_text:
            return None
        review_id = f"tr_{hashlib.sha256(card_text.encode()).hexdigest()[:16]}"

    # Rating (TrustRadius uses 1-10 scale, sometimes shown as stars)
    rating = _extract_html_rating(card)

    # Review title / headline
    summary = None
    title_el = card.select_one(
        'h2, h3, '
        '[class*="title"], '
        '[class*="headline"], '
        '[itemprop="name"]'
    )
    if title_el:
        summary = title_el.get_text(strip=True)[:500]

    # Review body
    review_text = ""
    body_el = card.select_one(
        '[itemprop="reviewBody"], '
        '[class*="reviewBody"], '
        '[class*="review-body"], '
        '[class*="ReviewBody"]'
    )
    if body_el:
        review_text = body_el.get_text(strip=True)

    # Fallback: concatenate all paragraph text in the card
    if not review_text:
        paragraphs = []
        for p in card.select("p"):
            t = p.get_text(strip=True)
            if t and len(t) > 20:
                paragraphs.append(t)
        review_text = " ".join(paragraphs)

    if not review_text or len(review_text) < 30:
        return None

    # Pros/Cons sections
    pros = _extract_html_section(card, ["pros", "like", "positive", "strength", "best"])
    cons = _extract_html_section(card, ["cons", "dislike", "negative", "weakness", "worst"])

    # Reviewer info
    reviewer_name = _get_card_text(
        card,
        '[itemprop="author"], [class*="author"], [class*="reviewer"]'
    )
    reviewer_title = _get_card_text(
        card,
        '[class*="jobTitle"], [class*="role"], [class*="position"]'
    )
    reviewer_company = _get_card_text(
        card,
        '[class*="company"], [class*="organization"]'
    )
    company_size = _get_card_text(
        card,
        '[class*="companySize"], [class*="company-size"]'
    )
    industry = _get_card_text(
        card,
        '[class*="industry"], [class*="sector"]'
    )

    # Date
    reviewed_at = None
    date_el = card.select_one(
        'time, [datetime], [itemprop="datePublished"], [class*="date"]'
    )
    if date_el:
        reviewed_at = (
            date_el.get("datetime")
            or date_el.get("content")
            or date_el.get_text(strip=True)
        )

    return {
        "source": "trustradius",
        "source_url": f"{_PRODUCT_BASE}/{target.product_slug}/reviews",
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 10,
        "summary": summary,
        "review_text": review_text[:10000],
        "pros": pros[:5000] if pros else None,
        "cons": cons[:5000] if cons else None,
        "reviewer_name": reviewer_name,
        "reviewer_title": reviewer_title,
        "reviewer_company": reviewer_company,
        "company_size_raw": company_size,
        "reviewer_industry": industry,
        "reviewed_at": reviewed_at,
        "raw_metadata": {
            "extraction_method": "html",
            "source_weight": 0.8,
            "source_type": "verified_review_platform",
        },
    }


def _extract_html_rating(card) -> float | None:
    """Extract rating from a TrustRadius review card."""
    # Method 1: itemprop ratingValue
    rating_el = card.select_one('[itemprop="ratingValue"]')
    if rating_el:
        val = rating_el.get("content") or rating_el.get_text(strip=True)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Method 2: data-rating attribute
    dr = card.select_one("[data-rating]")
    if dr:
        try:
            return float(dr["data-rating"])
        except (ValueError, TypeError):
            pass

    # Method 3: text pattern "X out of 10" or "X/10"
    card_text = card.get_text()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10", card_text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass

    # Method 4: aria-label
    aria_el = card.select_one('[aria-label*="rating"], [aria-label*="star"]')
    if aria_el:
        m = re.search(r"(\d+(?:\.\d+)?)", aria_el.get("aria-label", ""))
        if m:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                pass

    return None


def _extract_html_section(card, keywords: list[str]) -> str | None:
    """Extract a labeled section (pros/cons) from a review card."""
    # Look for headings/labels containing keywords, then grab sibling content
    for heading in card.select("h3, h4, h5, strong, b, dt, [class*='heading'], [class*='label']"):
        heading_text = heading.get_text(strip=True).lower()
        if any(kw in heading_text for kw in keywords):
            # Collect sibling text until next heading
            parts = []
            for sib in heading.find_next_siblings():
                if sib.name in ("h3", "h4", "h5", "strong", "b", "dt"):
                    break
                t = sib.get_text(strip=True)
                if t:
                    parts.append(t)
            if parts:
                return " ".join(parts)

    # Fallback: class-based matching
    for kw in keywords:
        el = card.select_one(f'[class*="{kw}"]')
        if el:
            t = el.get_text(strip=True)
            if t and len(t) > 10:
                return t

    return None


def _get_card_text(card, selector: str) -> str | None:
    """Safely extract text from the first matching element."""
    el = card.select_one(selector)
    if el:
        text = el.get_text(strip=True)
        if text:
            return text
    return None


# ------------------------------------------------------------------
# Firecrawl markdown parsing (existing — for Priority 2 path)
# ------------------------------------------------------------------


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
