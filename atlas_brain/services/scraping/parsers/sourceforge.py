"""
SourceForge parser for B2B review scraping.

URL pattern: sourceforge.net/software/product/{slug}/reviews/
Pagination: ?page=N (1-indexed).

SourceForge reviews are lightly protected -- datacenter proxies work fine.
Review cards contain star ratings, reviewer info, pros/cons sections, and
structured metadata (company size, industry).
"""

from __future__ import annotations

import hashlib
import logging
import re

from bs4 import BeautifulSoup

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, apply_date_cutoff, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.sourceforge")

_DOMAIN = "sourceforge.net"
_BASE_URL = "https://sourceforge.net/software/product"
_MIN_TEXT_LEN = 80


class SourceForgeParser:
    """Parse SourceForge product review pages."""

    source_name = "sourceforge"
    prefer_residential = False
    version = "sourceforge:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape SourceForge reviews for the given product slug."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        stop_reason = ""

        consecutive_empty = 0
        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}/reviews/"
            if page > 1:
                url += f"?page={page}"

            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=(
                        f"https://sourceforge.net/software/product/{target.product_slug}/"
                        if page == 1
                        else f"{_BASE_URL}/{target.product_slug}/reviews/?page={page - 1}"
                    ),
                    sticky_session=False,
                    prefer_residential=False,
                )
                pages_scraped += 1

                if resp.status_code == 404:
                    errors.append(f"Product slug not found: {target.product_slug}")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    break

                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page}: unexpected content-type ({ct[:40]})")
                    break

                before = len(reviews)
                page_reviews = _parse_page(resp.text, target, seen_ids)
                page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)
                reviews.extend(page_reviews)
                if cutoff_hit:
                    stop_reason = "date_cutoff"
                    break

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2 or page == 1:
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "SourceForge page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                break

        logger.info(
            "SourceForge scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            stop_reason=stop_reason,
        )


def _parse_page(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse a single SourceForge review listing page."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # SourceForge review cards use itemprop="review" or class-based selectors
    review_cards = soup.select(
        '[itemprop="review"], '
        '[class*="review-card"], '
        '[class*="ReviewCard"], '
        'div.review'
    )

    # Fallback: look for containers with reviewBody or rating children
    if not review_cards:
        review_cards = soup.select(
            '[data-review-id], '
            'div[class*="reviewItem"], '
            'article[class*="review"]'
        )

    for card in review_cards:
        try:
            review = _parse_review_card(card, target)
            if review and review["source_review_id"] not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse SourceForge review card", exc_info=True)

    return reviews


def _parse_review_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a single SourceForge review card element."""
    # Review ID -- try data attributes, then element id, then content hash
    review_id = (
        card.get("data-review-id", "")
        or card.get("id", "")
    )
    if not review_id:
        card_text = card.get_text(strip=True)[:300]
        if not card_text:
            return None
        review_id = f"sf_{hashlib.sha256(card_text.encode()).hexdigest()[:16]}"

    # Star rating (1-5)
    rating = _extract_rating(card)

    # Reviewer name -- itemprop author meta, or first child div in ext-review-meta
    reviewer_name = None
    author_meta = card.select_one('[itemprop="author"] [itemprop="name"]')
    if author_meta:
        reviewer_name = (author_meta.get("content") or author_meta.get_text()).strip() or None

    # SourceForge ext-review-meta block: structured label/value pairs
    reviewer_title = None
    reviewer_company = None
    company_size = None
    reviewer_industry = None

    meta_block = card.select_one(".ext-review-meta")
    if meta_block:
        from bs4 import Tag
        children = [c for c in meta_block.children if isinstance(c, Tag)]
        for i, child in enumerate(children):
            label_el = child.select_one(".value-label")
            value_el = child.select_one(".value-value")
            # The child itself may BE the value-value element
            child_cls = " ".join(child.get("class", []))
            is_value = "value-value" in child_cls

            if label_el and value_el:
                label = label_el.get_text(strip=True).lower()
                value = value_el.get_text(strip=True)
                if not value:
                    continue
                if "company size" in label:
                    company_size = value
                elif "industry" in label:
                    reviewer_industry = value
                elif "role" in label:
                    pass  # User/Admin -- not job title
            elif is_value and not label_el and i == 1:
                # Second child that IS a value-value = job title
                reviewer_title = child.get_text(strip=True) or None
            elif not label_el and not is_value and i == 0:
                # First plain div = reviewer name (fallback)
                if not reviewer_name:
                    reviewer_name = child.get_text(strip=True) or None

    # Fallback selectors
    if not reviewer_name:
        reviewer_name = _get_text(card, '[class*="author"], [class*="reviewer-name"]')
    if not reviewer_title:
        reviewer_title = _get_text(card, '[class*="job-title"], [class*="reviewer-title"]')
    if not reviewer_company:
        reviewer_company = _get_text(card, '[class*="company-name"], [class*="organization"]')
    if not company_size:
        extra = _supplement_metadata_from_list(card)
        company_size = extra.get("company_size")
        if not reviewer_industry:
            reviewer_industry = extra.get("reviewer_industry")

    # Pros and cons -- SourceForge labels them in heading or class patterns
    pros = _extract_section(card, ["pros", "like", "best", "positive", "strength"])
    cons = _extract_section(card, ["cons", "dislike", "worst", "negative", "weakness"])

    # Review title / summary
    summary = None
    title_el = card.select_one(
        '[itemprop="name"], [class*="review-title"], [class*="headline"], h3, h4'
    )
    if title_el:
        summary = title_el.get_text(strip=True)[:500]

    # Main review body
    review_text = ""
    body_el = card.select_one(
        '[itemprop="reviewBody"], [class*="review-body"], [class*="reviewBody"]'
    )
    if body_el:
        review_text = body_el.get_text(strip=True)

    # Fallback: combine paragraph text
    if not review_text:
        paragraphs = []
        for p in card.select("p"):
            t = p.get_text(strip=True)
            if t and len(t) > 15:
                paragraphs.append(t)
        review_text = " ".join(paragraphs)

    # If still no body, combine pros/cons
    if not review_text:
        parts = []
        if pros:
            parts.append(f"Pros: {pros}")
        if cons:
            parts.append(f"Cons: {cons}")
        review_text = "\n".join(parts)

    # Skip short reviews
    if not review_text or len(review_text) < _MIN_TEXT_LEN:
        return None

    # Date
    reviewed_at = None
    date_el = card.select_one(
        'time, [itemprop="datePublished"], [datetime], [class*="date"]'
    )
    if date_el:
        reviewed_at = (
            date_el.get("datetime")
            or date_el.get("content")
            or date_el.get_text(strip=True)
        )

    return {
        "source": "sourceforge",
        "source_url": f"https://sourceforge.net/software/product/{target.product_slug}/reviews/",
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
        "raw_metadata": {
            "extraction_method": "html",
            "source_weight": 0.6,
            "source_type": "review_site",
        },
    }


def _extract_rating(card) -> float | None:
    """Extract star rating from a SourceForge review card."""
    # Method 1: itemprop ratingValue
    rating_el = card.select_one('[itemprop="ratingValue"]')
    if rating_el:
        val = rating_el.get("content") or rating_el.get_text(strip=True)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Method 2: data-rating or data-star attribute
    for attr in ("data-rating", "data-star", "data-score"):
        dr = card.select_one(f"[{attr}]")
        if dr:
            try:
                return float(dr[attr])
            except (ValueError, TypeError):
                pass

    # Method 3: aria-label like "4 out of 5 stars"
    aria_el = card.select_one('[aria-label*="star"], [aria-label*="rating"]')
    if aria_el:
        m = re.search(r"(\d+(?:\.\d+)?)", aria_el.get("aria-label", ""))
        if m:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                pass

    # Method 4: count filled star elements
    filled = card.select(
        '[class*="star-filled"], [class*="star--filled"], '
        '[class*="star-full"], .star.fill, .star.active'
    )
    if filled:
        return float(len(filled))

    # Method 5: text pattern "X/5" or "X out of 5"
    card_text = card.get_text()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*5", card_text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass

    return None


def _extract_section(card, keywords: list[str]) -> str | None:
    """Extract a labeled section (pros/cons) from a review card by keyword matching."""
    # Look for headings containing keywords, then grab sibling content
    for heading in card.select("h3, h4, h5, strong, b, dt, [class*='heading'], [class*='label']"):
        heading_text = heading.get_text(strip=True).lower()
        if any(kw in heading_text for kw in keywords):
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


def _supplement_metadata_from_list(card) -> dict[str, str]:
    """Try to extract reviewer metadata from list/definition-style blocks.

    SourceForge sometimes renders reviewer details as a <ul> or <dl> with
    labels like "Company Size:", "Industry:", "Role:" followed by the value.

    Returns a dict with keys: reviewer_title, company_size, reviewer_industry.
    """
    result: dict[str, str] = {}
    for li in card.select("li, dd, span"):
        text = li.get_text(strip=True)
        if not text:
            continue
        lower = text.lower()

        if ("company size" in lower or "employees" in lower) and "company_size" not in result:
            val = _extract_label_value(text)
            if val:
                result["company_size"] = val

        elif "industry" in lower and "reviewer_industry" not in result:
            val = _extract_label_value(text)
            if val:
                result["reviewer_industry"] = val

        elif ("role" in lower or "title" in lower) and "reviewer_title" not in result:
            val = _extract_label_value(text)
            if val:
                result["reviewer_title"] = val

    return result


def _extract_label_value(text: str) -> str | None:
    """Extract value from 'Label: Value' or 'Label - Value' patterns."""
    for sep in (":", "-", "|"):
        if sep in text:
            parts = text.split(sep, 1)
            if len(parts) == 2:
                val = parts[1].strip()
                if val:
                    return val
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
register_parser(SourceForgeParser())
