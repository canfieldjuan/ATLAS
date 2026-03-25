"""
Slashdot Software parser for B2B review scraping.

URL pattern: slashdot.org/software/p/{slug}/?page={n}
Pagination: query parameter `page` (1-indexed).

Slashdot Software product pages expose structured review cards with:
- Reviewer metadata (name, title, organization size, usage cadence)
- Review title + summary/positive/negative sections
- Overall rating and dimension ratings
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, apply_date_cutoff, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.slashdot")

_DOMAIN = "slashdot.org"
_BASE_URL = "https://slashdot.org/software/p"
_MIN_TEXT_LEN = 80
_NOT_FOUND_TITLE = re.compile(r"<title>\s*Page not found", re.IGNORECASE)
_LABEL_RE = re.compile(r"^(summary|positive|negative|pros|cons)\s*:\s*(.+)$", re.IGNORECASE)
_RETRYABLE_HTTP_STATUSES = {403, 408, 429, 500, 502, 503, 520, 521, 522, 524}
_REQUEST_STRATEGIES = (
    (True, True),
    (True, False),
    (False, True),
)


class SlashdotParser:
    """Parse Slashdot Software product review pages."""

    source_name = "slashdot"
    prefer_residential = True
    version = "slashdot:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Slashdot reviews for a product slug."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        stop_reason = ""
        consecutive_empty = 0

        for page in range(1, target.max_pages + 1):
            url = _build_url(target.product_slug, page)
            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+slashdot+software+reviews"
                if page == 1
                else _build_url(target.product_slug, page - 1)
            )

            try:
                before = len(reviews)
                resp, attempts, fetch_errors = await _fetch_page(
                    client=client,
                    url=url,
                    referer=referer,
                    page=page,
                )
                if resp is None:
                    errors.append(
                        fetch_errors[-1] if fetch_errors else f"Page {page}: request failed"
                    )
                    break
                pages_scraped += 1
                html = resp.text
                used_browser_fallback = False

                if page == 1 and int(resp.status_code or 0) in _RETRYABLE_HTTP_STATUSES:
                    browser_html, browser_status, browser_error = await _fetch_page_via_scraping_browser(
                        url=url,
                        referer=referer,
                    )
                    if browser_html and browser_status == 200:
                        html = browser_html
                        used_browser_fallback = True
                        logger.info(
                            "Slashdot page %d recovered via Scraping Browser fallback for %s",
                            page,
                            target.product_slug,
                        )
                    elif browser_error:
                        logger.info(
                            "Slashdot browser fallback failed for %s: %s",
                            target.product_slug,
                            browser_error,
                        )

                if resp.status_code == 404:
                    errors.append(f"Product slug not found: {target.product_slug}")
                    break
                if resp.status_code != 200 and not used_browser_fallback:
                    retry_suffix = f" after {attempts} attempts" if attempts > 1 else ""
                    errors.append(f"Page {page}: HTTP {resp.status_code}{retry_suffix}")
                    break

                ct = resp.headers.get("content-type", "")
                if not used_browser_fallback and "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page}: unexpected content-type ({ct[:40]})")
                    break
                if _NOT_FOUND_TITLE.search(html):
                    errors.append(f"Product slug not found: {target.product_slug}")
                    break

                page_reviews = _parse_page(html, target, seen_ids, url)
                page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)
                reviews.extend(page_reviews)
                if cutoff_hit:
                    stop_reason = "date_cutoff"
                    break

                if len(reviews) == before:
                    consecutive_empty += 1
                    if page == 1 or consecutive_empty >= 2:
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "Slashdot page %d failed for %s: %s",
                    page,
                    target.product_slug,
                    exc,
                )
                break

        logger.info(
            "Slashdot scrape for %s: %d reviews from %d pages",
            target.vendor_name,
            len(reviews),
            pages_scraped,
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            stop_reason=stop_reason,
        )


async def _fetch_page(
    *,
    client: AntiDetectionClient,
    url: str,
    referer: str,
    page: int,
) -> tuple[object | None, int, list[str]]:
    """Fetch one Slashdot page with small strategy variation to reduce 403/502 failures."""
    fetch_errors: list[str] = []
    last_response = None
    attempts = 0
    for sticky_session, prefer_residential in _REQUEST_STRATEGIES:
        attempts += 1
        try:
            resp = await client.get(
                url,
                domain=_DOMAIN,
                referer=referer,
                sticky_session=sticky_session,
                prefer_residential=prefer_residential,
            )
        except Exception as exc:
            fetch_errors.append(f"Page {page} attempt {attempts}: {exc}")
            continue

        last_response = resp
        status = int(resp.status_code or 0)
        if status in (200, 404):
            return resp, attempts, fetch_errors
        if status in _RETRYABLE_HTTP_STATUSES:
            fetch_errors.append(f"Page {page} attempt {attempts}: HTTP {status}")
            continue
        return resp, attempts, fetch_errors

    return last_response, attempts, fetch_errors


async def _fetch_page_via_scraping_browser(
    *,
    url: str,
    referer: str,
) -> tuple[str | None, int, str | None]:
    """Attempt a single-page fetch via Bright Data Scraping Browser (optional)."""
    from atlas_brain.config import settings

    ws_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
    allowed_domains = {
        d.strip().lower()
        for d in settings.b2b_scrape.scraping_browser_domains.split(",")
        if d.strip()
    }
    if not ws_url or _DOMAIN not in allowed_domains:
        return None, 0, None

    timeout_ms = settings.b2b_scrape.playwright_timeout_ms
    try:
        from playwright.async_api import async_playwright
        from ..browser import solve_browser_challenge
        from ..captcha import is_captcha_enabled_for_domain

        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(ws_url, timeout=timeout_ms)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms, referer=referer)
            status = int(resp.status or 0) if resp else 0
            html = await page.content()
            if status >= 400 and is_captcha_enabled_for_domain(_DOMAIN):
                solved = await solve_browser_challenge(
                    page,
                    domain=_DOMAIN,
                    html=html,
                    status_code=status,
                )
                if solved:
                    resp = await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=timeout_ms,
                        referer=referer,
                    )
                    status = int(resp.status or 0) if resp else 0
                    html = await page.content()
            await page.close()
            await browser.close()
            return html, status, None
    except Exception as exc:
        return None, 0, f"{type(exc).__name__}: {exc}"


def _build_url(product_slug: str, page: int) -> str:
    base = f"{_BASE_URL}/{product_slug}/"
    if page <= 1:
        return base
    return f"{base}?page={page}"


def _parse_page(
    html: str,
    target: ScrapeTarget,
    seen_ids: set[str],
    page_url: str,
) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.m-review[itemprop='review']")
    reviews: list[dict] = []

    for card in cards:
        review = _parse_review_card(card, target, page_url)
        if not review:
            continue
        review_id = review["source_review_id"]
        if review_id in seen_ids:
            continue
        seen_ids.add(review_id)
        reviews.append(review)

    return reviews


def _parse_review_card(card, target: ScrapeTarget, page_url: str) -> dict | None:
    meta_fields = _extract_meta_fields(card)
    dims = _extract_dimension_ratings(card)

    rating = _extract_rating(card)
    if rating is None and dims:
        rating = round(sum(dims.values()) / max(len(dims), 1), 2)

    reviewed_at = _extract_review_date(card)
    summary, pros, cons, extra_lines = _extract_content_sections(card)
    title = _get_text(card, ".ext-review-title .review-title, .review-title")
    reviewer_name = (
        meta_fields.get("name")
        or _author_name_from_itemprop(card)
        or _get_text(card, "[itemprop='author'] [itemprop='name']")
    )

    review_lines: list[str] = []
    if summary:
        review_lines.append(f"Summary: {summary}")
    if pros:
        review_lines.append(f"Positive: {pros}")
    if cons:
        review_lines.append(f"Negative: {cons}")
    review_lines.extend(extra_lines)
    review_text = "\n".join(line for line in review_lines if line).strip()
    if not review_text and title:
        review_text = title.strip()
    if len(review_text) < _MIN_TEXT_LEN:
        return None

    review_id = (card.get("id") or "").strip()
    if not review_id:
        dedup_seed = f"{target.product_slug}|{title}|{review_text[:240]}"
        review_id = f"slashdot_{hashlib.sha256(dedup_seed.encode()).hexdigest()[:16]}"

    recommend_likelihood = _extract_recommend_likelihood(card)

    raw_metadata = {
        "extraction_method": "html_scrape",
        "source_weight": 0.8,
        "dimension_ratings": dims,
        "role": meta_fields.get("role"),
        "organization_size": meta_fields.get("organization size"),
        "used_how_often": meta_fields.get("used how often?"),
        "length_of_product_use": meta_fields.get("length of product use"),
        "recommend_likelihood": recommend_likelihood,
    }

    return {
        "source": "slashdot",
        "source_url": f"{page_url}#{review_id}",
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 5,
        "summary": (summary or title or "")[:500] or None,
        "review_text": review_text[:10000],
        "pros": pros[:4000] if pros else None,
        "cons": cons[:4000] if cons else None,
        "reviewer_name": reviewer_name,
        "reviewer_title": meta_fields.get("job title"),
        "reviewer_company": None,
        "company_size_raw": meta_fields.get("organization size"),
        "reviewer_industry": None,
        "reviewed_at": reviewed_at,
        "raw_metadata": raw_metadata,
    }


def _extract_meta_fields(card) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in card.select(".ext-review-meta > div"):
        label_el = item.select_one(".value-label")
        value_el = item.select_one(".value-value")
        if label_el is None or value_el is None:
            continue
        label = label_el.get_text(" ", strip=True).lower().rstrip(":")
        value = value_el.get_text(" ", strip=True)
        if label and value:
            fields[label] = value
    return fields


def _extract_dimension_ratings(card) -> dict[str, float]:
    dims: dict[str, float] = {}
    for row in card.select(".ext-review-dimensions .dim-rating"):
        label_el = row.select_one("span")
        if label_el is None:
            continue
        label = label_el.get_text(" ", strip=True).lower()
        if not label:
            continue
        full = len(row.select(".star.yellow"))
        half = len(row.select(".star.half"))
        score = min(5.0, float(full) + (0.5 * float(half)))
        if score > 0:
            dims[label] = score
    return dims


def _extract_rating(card) -> float | None:
    rating_el = card.select_one("[itemprop='reviewRating'] [itemprop='ratingValue']")
    if rating_el is None:
        return None
    raw = (rating_el.get("content") or rating_el.get_text(" ", strip=True) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _extract_review_date(card) -> str | None:
    raw = None
    meta = card.select_one("meta[itemprop='datePublished']")
    if meta is not None:
        raw = (meta.get("content") or "").strip()
    if not raw:
        created = _get_text(card, ".created-date")
        if created:
            raw = created.replace("Date:", "").strip()
    if not raw:
        return None
    for fmt in ("%m/%d/%Y", "%b %d %Y", "%B %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw


def _extract_content_sections(card) -> tuple[str | None, str | None, str | None, list[str]]:
    summary = None
    pros = None
    cons = None
    extras: list[str] = []

    for p in card.select(".ext-review-content p"):
        line = p.get_text(" ", strip=True)
        if not line:
            continue
        match = _LABEL_RE.match(line)
        if not match:
            extras.append(line)
            continue
        label = match.group(1).lower()
        value = match.group(2).strip()
        if not value:
            continue
        if label == "summary":
            summary = value
        elif label in {"positive", "pros"}:
            pros = value
        elif label in {"negative", "cons"}:
            cons = value
        else:
            extras.append(line)

    return summary, pros, cons, extras


def _extract_recommend_likelihood(card) -> int | None:
    active = card.select_one(".ext-review-reco-likelihood .m-reco-rate span.active")
    if active is None:
        return None
    text = active.get_text(" ", strip=True)
    try:
        value = int(text)
    except ValueError:
        return None
    if 0 <= value <= 10:
        return value
    return None


def _author_name_from_itemprop(card) -> str | None:
    author = card.select_one("[itemprop='author'] [itemprop='name']")
    if author is None:
        return None
    value = (author.get("content") or author.get_text(" ", strip=True)).strip()
    return value or None


def _get_text(node, selector: str) -> str | None:
    element = node.select_one(selector)
    if element is None:
        return None
    text = element.get_text(" ", strip=True)
    return text or None


register_parser(SlashdotParser())
