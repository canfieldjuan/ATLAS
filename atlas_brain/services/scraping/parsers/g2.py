"""
G2 parser for B2B review scraping.

URL pattern: g2.com/products/{slug}/reviews?page={n}
Hardest target -- heavy DataDome protection.

Primary: Playwright stealth browser (when playwright_enabled).
Fallback: curl_cffi HTTP client with residential proxy + sticky sessions.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ....config import settings
from ....services.company_normalization import normalize_company_name
from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, apply_date_cutoff, log_page, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.g2")

_DOMAIN = "g2.com"
_BASE_URL = "https://www.g2.com/products"
_VERIFIED_USER_RE = re.compile(
    r"^Verified User in\s+(?P<industry>.+?)\s*This reviewer's identity",
    re.I | re.S,
)
_COMPANY_SIZE_RE = re.compile(
    r"(?:\bsmall-business\b|\bmid-market\b|\benterprise\b|\b\d[\d,\- ]*\s*employees?\b|\bemp\.)",
    re.I,
)
_INDUSTRY_HINT_RE = re.compile(
    r"\b("
    r"software|technology|services|marketing|advertising|insurance|health|healthcare|"
    r"financial|finance|banking|retail|education|government|manufacturing|media|"
    r"telecommunications|telecom|hospital|hospitality|construction|legal|accounting|"
    r"real estate|internet|computer|human resources|hr|pharma|pharmaceutical|"
    r"logistics|transportation|consumer goods|biotech|automotive|newspapers?|publishing"
    r")\b",
    re.I,
)


class G2Parser:
    """Parse G2 review pages with DataDome bypass."""

    source_name = "g2"
    prefer_residential = True
    version = "g2:3"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape G2 reviews -- Web Unlocker first, then browser, then HTTP."""

        # Priority 1: Bright Data Web Unlocker (handles DataDome automatically)
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

        # Priority 2: Playwright stealth browser
        if settings.b2b_scrape.playwright_enabled:
            try:
                result = await self._scrape_browser(target)
                if result.reviews:
                    return result
                logger.warning(
                    "Browser scrape for %s returned 0 reviews, falling back to HTTP",
                    target.vendor_name,
                )
            except Exception as exc:
                logger.warning(
                    "Browser scrape failed for %s: %s -- falling back to HTTP",
                    target.vendor_name, exc,
                )

        # Priority 3: curl_cffi HTTP client
        return await self._scrape_http(target, client)

    # ------------------------------------------------------------------
    # Web Unlocker path (Bright Data -- handles DataDome internally)
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape G2 via Bright Data Web Unlocker proxy.

        Web Unlocker is an HTTP proxy that handles DataDome/Cloudflare
        challenges automatically -- no CAPTCHA solving or stealth browser
        needed.  Just send a normal GET and it returns the unblocked HTML.
        """
        import httpx

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
        consecutive_empty = 0

        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}/reviews"
            if page > 1:
                url += f"?page={page}"

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+g2+reviews"
                if page == 1
                else f"{_BASE_URL}/{target.product_slug}/reviews?page={page - 1}"
                if page > 2
                else f"{_BASE_URL}/{target.product_slug}/reviews"
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

            page_errs: list[str] = []
            page_start = _time.monotonic()
            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url,
                    verify=False,
                    timeout=60.0,
                ) as http:
                    resp = await http.get(url, headers=headers)

                pages_scraped += 1
                elapsed_ms = int((_time.monotonic() - page_start) * 1000)

                if resp.status_code == 403:
                    page_errs.append(f"blocked (403) via Web Unlocker")
                    errors.append(f"Page {page}: blocked (403) via Web Unlocker")
                    page_logs.append(log_page(
                        page, url, status_code=403, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=page_errs,
                    ))
                    break
                if resp.status_code != 200:
                    page_errs.append(f"HTTP {resp.status_code}")
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=page_errs,
                    ))
                    continue

                page_reviews = _parse_page(resp.text, target, seen_ids)

                # Retry once if 200 OK but 0 reviews (Cloudflare soft-block)
                if not page_reviews and resp.status_code == 200:
                    await asyncio.sleep(random.uniform(3.0, 6.0))
                    retry_start = _time.monotonic()
                    async with httpx.AsyncClient(
                        proxy=proxy_url, verify=False, timeout=60.0,
                    ) as http_retry:
                        resp = await http_retry.get(url, headers=headers)
                    elapsed_ms += int((_time.monotonic() - retry_start) * 1000)
                    if resp.status_code == 200:
                        page_reviews = _parse_page(resp.text, target, seen_ids)

                page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)

                pl = log_page(
                    page, url, status_code=200, duration_ms=elapsed_ms,
                    response_bytes=len(resp.content), reviews=page_reviews,
                    raw_body=resp.content, prior_hashes=prior_hashes,
                    prior_review_ids=prior_review_ids,
                    next_page_found=page < target.max_pages,
                    next_page_url=f"{_BASE_URL}/{target.product_slug}/reviews?page={page + 1}" if page_reviews else "",
                )
                if cutoff_hit:
                    pl.stop_reason = "date_cutoff"
                    stop_reason = "date_cutoff"
                page_logs.append(pl)

                if not page_reviews:
                    consecutive_empty += 1
                    if page == 1:
                        logger.warning(
                            "G2 Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    if consecutive_empty >= 2:
                        break
                    continue

                consecutive_empty = 0

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "G2 Web Unlocker page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                break

            # Inter-page delay
            await asyncio.sleep(random.uniform(2.0, 5.0))

        logger.info(
            "G2 Web Unlocker scrape for %s: %d reviews from %d pages",
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
    # Browser path (Playwright stealth)
    # ------------------------------------------------------------------

    async def _scrape_browser(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape G2 via Playwright stealth browser."""
        from ..browser import get_stealth_browser
        from ..proxy import ProxyManager

        browser = get_stealth_browser()
        proxy_mgr = ProxyManager.from_config(settings.b2b_scrape)

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time
        stop_reason = ""

        # Get a residential proxy URL if available
        proxy_config = proxy_mgr.get_proxy(domain=_DOMAIN, sticky=True, prefer_residential=True)
        proxy_url = proxy_config.url if proxy_config else None

        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}/reviews"
            if page > 1:
                url += f"?page={page}"

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+g2+reviews"
                if page == 1
                else f"{_BASE_URL}/{target.product_slug}/reviews?page={page - 1}"
                if page > 2
                else f"{_BASE_URL}/{target.product_slug}/reviews"
            )

            page_start = _time.monotonic()
            try:
                result = await browser.scrape_page(
                    url,
                    proxy_url=proxy_url,
                    wait_for_selector="[data-review-id]",
                    referer=referer,
                )
                pages_scraped += 1
                elapsed_ms = int((_time.monotonic() - page_start) * 1000)
                html_body = result.html or ""

                if result.status_code == 403:
                    errors.append(f"Page {page}: blocked (403) via browser")
                    page_logs.append(log_page(
                        page, url, status_code=403, duration_ms=elapsed_ms,
                        response_bytes=len(html_body), raw_body=html_body,
                        prior_hashes=prior_hashes, errors=["blocked (403) via browser"],
                    ))
                    break
                if result.status_code != 200:
                    errors.append(f"Page {page}: HTTP {result.status_code}")
                    page_logs.append(log_page(
                        page, url, status_code=result.status_code, duration_ms=elapsed_ms,
                        response_bytes=len(html_body), raw_body=html_body,
                        prior_hashes=prior_hashes, errors=[f"HTTP {result.status_code}"],
                    ))
                    continue

                page_reviews = _parse_page(result.html, target, seen_ids)
                page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)

                pl = log_page(
                    page, url, status_code=200, duration_ms=elapsed_ms,
                    response_bytes=len(html_body), reviews=page_reviews,
                    raw_body=html_body, prior_hashes=prior_hashes,
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
                            "G2 browser page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("G2 browser page %d failed for %s: %s", page, target.product_slug, exc)
                break

            # Human-like inter-page delay
            await asyncio.sleep(random.uniform(3.0, 7.0))

        logger.info(
            "G2 browser scrape for %s: %d reviews from %d pages",
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
    # HTTP path (curl_cffi -- original)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape G2 reviews via curl_cffi HTTP client."""
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
                    errors.append(f"Page {page}: blocked (403) -- CAPTCHA challenge")
                    page_logs.append(log_page(
                        page, url, status_code=403, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content or b""), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["blocked (403) -- CAPTCHA"],
                    ))
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content or b""), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=[f"HTTP {resp.status_code}"],
                    ))
                    continue

                # Guard against non-HTML responses (CDN error pages, JSON errors)
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Page {page}: unexpected content-type ({ct[:40]})")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        errors=[f"unexpected content-type ({ct[:40]})"],
                    ))
                    break

                before = len(reviews)
                page_reviews = _parse_page(resp.text, target, seen_ids)
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

        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            page_logs=page_logs,
            stop_reason=stop_reason,
        )


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
            logger.warning("Failed to parse G2 review card", exc_info=True)

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

    # Reviewer info -- try semantic selectors first (legacy G2 markup)
    reviewer_name = _get_text(card, '[itemprop="author"], [class*="reviewer-name"]')
    reviewer_title = _get_text(card, '[class*="reviewer-title"], [class*="job-title"]')
    reviewer_company = _get_text(card, '[class*="reviewer-company"], [class*="organization"]')
    company_size = _get_text(card, '[class*="company-size"], [class*="employees"]')
    reviewer_industry = _get_text(card, '[class*="industry"]')

    # G2 2025+ uses Tailwind utility classes (elv-*) with no semantic names.
    # Reviewer info is: bold div = name, then sibling subtle divs that may
    # include title, company, industry, and size. Parse them even if one field
    # was already captured semantically so we do not lose employer identity.
    name_el = card.select_one(
        'div[class*="elv-font-bold"]:not([class*="elv-text-lg"])'
    )
    if name_el:
        if not reviewer_name:
            reviewer_name = name_el.get_text(strip=True) or None
        info_parent = name_el.parent.parent if name_el.parent else None
        if info_parent:
            subtle = [
                d.get_text(strip=True)
                for d in info_parent.select('div[class*="elv-text-subtle"]')
                if d.get_text(strip=True)
            ]
            reviewer_title, reviewer_company, reviewer_industry, company_size = _merge_tailwind_reviewer_info(
                subtle,
                reviewer_title=reviewer_title,
                reviewer_company=reviewer_company,
                reviewer_industry=reviewer_industry,
                company_size=company_size,
            )

    if _looks_like_company_size_label(reviewer_company):
        company_size = company_size or reviewer_company
        reviewer_company = None
    if _looks_like_company_size_label(reviewer_industry):
        company_size = company_size or reviewer_industry
        reviewer_industry = None

    # Date
    reviewed_at = None
    date_el = card.select_one('time, [itemprop="datePublished"], [class*="date"]')
    if date_el:
        reviewed_at = date_el.get("datetime") or date_el.get("content") or date_el.get_text(strip=True)

    reviewer_name, reviewer_industry = _sanitize_reviewer_identity(
        reviewer_name,
        reviewer_industry,
    )
    if reviewer_company:
        normalized_company = normalize_company_name(reviewer_company)
        normalized_industry = normalize_company_name(reviewer_industry or "")
        if _looks_like_industry_label(reviewer_company) and (
            not normalized_industry or normalized_company == normalized_industry
        ):
            reviewer_industry = reviewer_industry or reviewer_company
            reviewer_company = None

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
            "parser_version": G2Parser.version,
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


def _sanitize_reviewer_identity(
    reviewer_name: str | None,
    reviewer_industry: str | None,
) -> tuple[str | None, str | None]:
    """Normalize G2 verified-user strings into stable reviewer metadata."""
    name = str(reviewer_name or "").strip()
    if not name:
        return reviewer_name, reviewer_industry
    match = _VERIFIED_USER_RE.match(name)
    if not match:
        return reviewer_name, reviewer_industry
    industry = reviewer_industry or match.group("industry").strip()
    return "Verified User", industry or None


def _looks_like_company_size_label(text: str | None) -> bool:
    return bool(text and _COMPANY_SIZE_RE.search(text))


def _looks_like_industry_label(text: str | None) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return bool(_INDUSTRY_HINT_RE.search(candidate))


def _merge_tailwind_reviewer_info(
    subtle: list[str],
    *,
    reviewer_title: str | None,
    reviewer_company: str | None,
    reviewer_industry: str | None,
    company_size: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    remaining: list[str] = []
    title = reviewer_title
    company = reviewer_company
    industry = reviewer_industry
    size = company_size

    if _looks_like_company_size_label(company):
        size = size or company
        company = None
    if _looks_like_company_size_label(industry):
        size = size or industry
        industry = None

    for idx, text in enumerate(subtle):
        if _looks_like_company_size_label(text):
            size = size or text
            continue
        if idx == 0 and not title:
            title = text
            continue
        remaining.append(text)

    for text in remaining:
        if _looks_like_company_size_label(text):
            size = size or text
        elif _looks_like_industry_label(text):
            industry = industry or text
        elif not company:
            company = text
        elif not industry:
            industry = text

    return title, company, industry, size


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
