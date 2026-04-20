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
from . import (
    ScrapeResult,
    ScrapeTarget,
    apply_date_cutoff,
    log_page,
    page_has_only_known_source_reviews,
    register_parser,
)

logger = logging.getLogger("atlas.services.scraping.parsers.capterra")

_DOMAIN = "capterra.com"
_BASE_URL = "https://www.capterra.com/p"


class CapterraParser:
    """Parse Capterra review pages using JSON-LD or HTML fallback."""

    source_name = "capterra"
    prefer_residential = True
    version = "capterra:2"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Capterra reviews -- Web Unlocker first, then browser, then HTTP client."""
        from atlas_brain.config import settings
        sb_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
        sb_domains = {
            d.strip().lower()
            for d in settings.b2b_scrape.scraping_browser_domains.split(",")
            if d.strip()
        }
        browser_enabled = bool(sb_url and _DOMAIN in sb_domains)

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

        # Priority 2: Bright Data Scraping Browser
        if browser_enabled:
            try:
                result = await self._scrape_browser(target, sb_url)
                if result.reviews:
                    return result
                logger.warning(
                    "Scraping Browser for %s returned 0 reviews, falling back to HTTP",
                    target.vendor_name,
                )
            except Exception as exc:
                logger.warning(
                    "Scraping Browser failed for %s: %s -- falling back to HTTP",
                    target.vendor_name, exc,
                )

        # Priority 3: curl_cffi HTTP client with residential proxy
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
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time
        stop_reason = ""

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
                if page_reviews:
                    _supplement_pros_cons_from_html(resp.text, page_reviews)
                else:
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
                elif page_has_only_known_source_reviews(page_reviews, target):
                    pl.stop_reason = "known_reviews_page"
                    stop_reason = "known_reviews_page"
                page_logs.append(pl)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Capterra Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    break

                if stop_reason == "known_reviews_page":
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
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            page_logs=page_logs,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Browser path (Playwright stealth browser)
    # ------------------------------------------------------------------

    async def _scrape_browser(self, target: ScrapeTarget, ws_url: str) -> ScrapeResult:
        """Scrape Capterra via Bright Data Scraping Browser (cloud Chromium)."""
        from atlas_brain.config import settings
        from playwright.async_api import async_playwright
        from ..browser import solve_browser_challenge

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time
        stop_reason = ""
        timeout_ms = settings.b2b_scrape.playwright_timeout_ms

        try:
            async with async_playwright() as pw:
                logger.info(
                    "Connecting Capterra Scraping Browser for vendor=%s slug=%s ws_host=%s",
                    target.vendor_name,
                    target.product_slug,
                    ws_url.split("@")[-1],
                )
                browser = await pw.chromium.connect_over_cdp(ws_url, timeout=timeout_ms)
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page_obj = await context.new_page()

                for page_num in range(1, target.max_pages + 1):
                    base_path = f"{_BASE_URL}/{target.product_slug}/reviews/"
                    url = base_path if page_num == 1 else f"{base_path}?page={page_num}"
                    referer = (
                        f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+capterra+reviews"
                        if page_num == 1
                        else f"{base_path}?page={page_num - 1}" if page_num > 2 else base_path
                    )

                    try:
                        logger.info(
                            "Capterra Scraping Browser navigating vendor=%s page=%d url=%s",
                            target.vendor_name,
                            page_num,
                            url,
                        )
                        page_start = _time.monotonic()
                        resp = await page_obj.goto(
                            url,
                            wait_until="commit",
                            timeout=timeout_ms,
                            referer=referer,
                        )
                        status = resp.status if resp else 0
                        pages_scraped += 1
                        elapsed_ms = int((_time.monotonic() - page_start) * 1000)
                        logger.info(
                            "Capterra Scraping Browser response vendor=%s page=%d status=%s final_url=%s duration_ms=%d",
                            target.vendor_name,
                            page_num,
                            status,
                            page_obj.url,
                            elapsed_ms,
                        )
                        try:
                            await page_obj.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 15000))
                        except Exception:
                            pass
                        html_body = await page_obj.content()

                        if await solve_browser_challenge(
                            page_obj,
                            domain=_DOMAIN,
                            html=html_body,
                            status_code=status,
                        ):
                            retry_start = _time.monotonic()
                            resp = await page_obj.goto(
                                url,
                                wait_until="commit",
                                timeout=timeout_ms,
                                referer=referer,
                            )
                            status = resp.status if resp else 0
                            elapsed_ms += int((_time.monotonic() - retry_start) * 1000)
                            try:
                                await page_obj.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 15000))
                            except Exception:
                                pass
                            html_body = await page_obj.content()

                        if status == 403:
                            errors.append(f"Page {page_num}: blocked (403) via scraping_browser")
                            page_logs.append(log_page(
                                page_num, url, status_code=403, duration_ms=elapsed_ms,
                                response_bytes=len(html_body), raw_body=html_body,
                                prior_hashes=prior_hashes, errors=["blocked (403) via scraping_browser"],
                            ))
                            break
                        if status != 200:
                            errors.append(f"Page {page_num}: HTTP {status}")
                            page_logs.append(log_page(
                                page_num, url, status_code=status, duration_ms=elapsed_ms,
                                response_bytes=len(html_body), raw_body=html_body,
                                prior_hashes=prior_hashes, errors=[f"HTTP {status}"],
                            ))
                            continue

                        try:
                            await page_obj.wait_for_selector(
                                '[data-testid="review-card"], [class*="review-card"], [itemprop="review"]',
                                timeout=min(timeout_ms, 15000),
                            )
                        except Exception:
                            pass
                        html_body = await page_obj.content()

                        page_reviews = _parse_json_ld(html_body, target, seen_ids)
                        if page_reviews:
                            _supplement_pros_cons_from_html(html_body, page_reviews)
                        else:
                            page_reviews = _parse_html(html_body, target, seen_ids)
                        page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)

                        pl = log_page(
                            page_num, url, status_code=200, duration_ms=elapsed_ms,
                            response_bytes=len(html_body), reviews=page_reviews,
                            raw_body=html_body, prior_hashes=prior_hashes,
                            prior_review_ids=prior_review_ids,
                            next_page_found=bool(page_reviews),
                        )
                        if cutoff_hit:
                            pl.stop_reason = "date_cutoff"
                            stop_reason = "date_cutoff"
                        elif page_has_only_known_source_reviews(page_reviews, target):
                            pl.stop_reason = "known_reviews_page"
                            stop_reason = "known_reviews_page"
                        page_logs.append(pl)

                        if not page_reviews:
                            if page_num == 1:
                                logger.warning(
                                    "Capterra scraping browser page 1 returned 0 reviews for %s",
                                    target.product_slug,
                                )
                            break

                        if stop_reason == "known_reviews_page":
                            break

                        reviews.extend(page_reviews)

                    except Exception as exc:
                        errors.append(f"Page {page_num}: {exc}")
                        logger.warning(
                            "Capterra scraping browser page %d failed for %s: %s",
                            page_num, target.product_slug, exc,
                        )
                        break

                    await asyncio.sleep(random.uniform(2.0, 5.0))
        except Exception as exc:
            errors.append(f"Browser setup: {exc}")
            logger.warning(
                "Capterra scraping browser setup failed for vendor=%s slug=%s: %s",
                target.vendor_name, exc,
            )

        logger.info(
            "Capterra scraping browser scrape for %s: %d reviews from %d pages",
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
    # HTTP client path (curl_cffi + residential proxy + CAPTCHA)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Capterra via curl_cffi HTTP client."""
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
            # Capterra URL: /p/{id}/{slug}/reviews/ or /p/{id}/{slug}/reviews/?page={n}
            base_path = f"{_BASE_URL}/{target.product_slug}/reviews/"
            url = base_path if page == 1 else f"{base_path}?page={page}"

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+capterra+reviews"
                if page == 1
                else f"{base_path}?page={page - 1}" if page > 2 else base_path
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

                # Supplement JSON-LD reviews with pros/cons from HTML cards
                if page_reviews:
                    _supplement_pros_cons_from_html(html, page_reviews)

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
                elif page_has_only_known_source_reviews(page_reviews, target):
                    pl.stop_reason = "known_reviews_page"
                    stop_reason = "known_reviews_page"
                page_logs.append(pl)

                if not page_reviews:
                    if page == 1:
                        logger.warning(
                            "Capterra page 1 returned 0 reviews for %s -- "
                            "JSON-LD and HTML selectors may be stale",
                            target.product_slug,
                        )
                    break  # No more reviews

                if stop_reason == "known_reviews_page":
                    break

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

        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            page_logs=page_logs,
            stop_reason=stop_reason,
        )


def _normalize_review_match_value(value: str | None) -> str:
    """Normalize review text fragments for cross-path matching."""
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text:
        return ""
    return hashlib.sha256(text[:500].encode()).hexdigest()[:16]


def _build_review_match_keys(review: dict[str, object]) -> set[str]:
    """Build match keys that survive JSON-LD and HTML ID differences."""
    keys: set[str] = set()
    review_id = str(review.get("source_review_id") or "").strip()
    if review_id:
        keys.add(review_id)
        review_tail = review_id.rsplit("/", 1)[-1]
        if review_tail:
            keys.add(review_tail)
            keys.add(f"review-{review_tail}")
    for field in ("review_text", "summary"):
        value = _normalize_review_match_value(str(review.get(field) or ""))
        if value:
            keys.add(value)
    reviewer_name = str(review.get("reviewer_name") or "").strip().lower()
    if reviewer_name:
        keys.add(f"name:{reviewer_name}")
        reviewed_at = str(review.get("reviewed_at") or "").strip().lower()
        if reviewed_at:
            keys.add(f"name_date:{reviewer_name}|{reviewed_at}")
    return keys


def _get_reviewer_title(card) -> str | None:
    """Extract reviewer title without confusing it with the review headline."""
    return _get_text(
        card,
        '[class*="job"], [class*="role"], [class*="title"]:not([class*="review"])',
    )


def _select_review_cards(soup: BeautifulSoup) -> list:
    """Return review card containers from legacy or current live DOM shapes."""
    cards = soup.select(
        '[data-testid="review-card"], '
        '.review-card, '
        '[class*="ReviewCard"], '
        'div[class*="review-"][class*="card"]'
    )
    if cards:
        return cards
    cards = soup.select('div[id^="review-"], div[class*="review-content"]')
    if cards:
        return cards

    current_cards = []
    seen: set[int] = set()
    for img in soup.select('img[data-testid="reviewer-profile-pic"]'):
        card = img
        while card and "Overall Rating" not in card.get_text(" ", strip=True):
            card = card.parent
        if not card or id(card) in seen:
            continue
        seen.add(id(card))
        current_cards.append(card)
    return current_cards


def _extract_live_profile_metadata(card) -> dict[str, str | None]:
    """Extract reviewer metadata from Capterra's current header block."""
    img = card.select_one('img[data-testid="reviewer-profile-pic"]')
    if not img or not getattr(img.parent, "parent", None):
        return {"reviewer_title": None, "reviewer_company": None, "company_size_raw": None, "reviewer_industry": None}

    lines = [text.strip() for text in img.parent.parent.stripped_strings if text and text.strip() != "Link Copied!"]
    meta_lines: list[str] = []
    for line in lines[1:]:
        if line.startswith("Used the software for:"):
            break
        if line.lower().startswith("verified "):
            continue
        meta_lines.append(line)

    reviewer_title = meta_lines[0] if meta_lines else None
    reviewer_industry = None
    company_size = None
    if len(meta_lines) > 1:
        industry_line = meta_lines[1]
        if "employees" in industry_line.lower():
            match = re.search(
                r"((?:[\d,]+(?:\+)?(?:\s*-\s*[\d,]+(?:\+)?)?\s*employees|self-employed))$",
                industry_line,
                re.IGNORECASE,
            )
            if match:
                company_size = match.group(1).strip()
                reviewer_industry = industry_line[:match.start()].strip(" ,") or None
            else:
                company_size = industry_line
        else:
            reviewer_industry = industry_line

    return {
        "reviewer_title": reviewer_title,
        "reviewer_company": None,
        "company_size_raw": company_size,
        "reviewer_industry": reviewer_industry,
    }


def _supplement_pros_cons_from_html(html: str, reviews: list[dict]) -> None:
    """Fill in JSON-LD review gaps from HTML cards when available.

    JSON-LD captures reviewBody but always leaves pros/cons as None.
    The HTML cards may also have reviewer firmographics that the JSON-LD omits.
    Match by source_review_id first, then normalized text fingerprints.
    Modifies reviews in-place; gracefully no-ops if HTML parsing finds nothing.
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = _select_review_cards(soup)
    if not cards:
        return

    html_lookup: dict[str, dict[str, str | None]] = {}
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

        profile_meta = _extract_live_profile_metadata(card)
        reviewer_title = _get_reviewer_title(card) or profile_meta["reviewer_title"]
        reviewer_company = _get_text(card, '[class*="company"], [class*="org"]') or profile_meta["reviewer_company"]
        company_size = _get_text(card, '[class*="size"], [class*="employees"]') or profile_meta["company_size_raw"]
        reviewer_industry = _get_text(card, '[class*="industry"]') or profile_meta["reviewer_industry"]
        summary = _get_text(card, '[class*="review-title"], [class*="ReviewTitle"], [itemprop="name"], h3, h4')
        review_text = _get_text(
            card,
            '[itemprop="reviewBody"], [class*="review-text"], [class*="ReviewText"], [class*="overall"]',
        )
        if not review_text:
            paragraphs = [p.get_text(" ", strip=True) for p in card.select("p") if p.get_text(" ", strip=True)]
            if paragraphs:
                review_text = paragraphs[0]
                if pros is None and len(paragraphs) > 1:
                    pros = paragraphs[1][:5000]
                if cons is None and len(paragraphs) > 2:
                    cons = paragraphs[2][:5000]
            if not review_text:
                parts = [part for part in (pros, cons) if part]
                review_text = "\n".join(parts) if parts else None
        reviewer_name = None
        if review_text:
            lines = [text.strip() for text in card.stripped_strings if text and text.strip() != "Link Copied!"]
            if lines:
                reviewer_name = lines[0]
        reviewed_at = None
        date_el = card.select_one("time, [class*='date']")
        if date_el:
            reviewed_at = date_el.get("datetime") or date_el.get_text(strip=True)

        card_data = {
            "pros": pros,
            "cons": cons,
            "reviewer_name": reviewer_name,
            "reviewer_title": reviewer_title,
            "reviewer_company": reviewer_company,
            "company_size_raw": company_size,
            "reviewer_industry": reviewer_industry,
            "reviewed_at": reviewed_at,
            "summary": summary,
            "review_text": review_text,
        }
        for key in _build_review_match_keys(
            {
                "source_review_id": card_id,
                "reviewer_name": reviewer_name,
                "reviewed_at": reviewed_at,
                "review_text": review_text,
                "summary": summary,
            }
        ):
            html_lookup.setdefault(key, card_data)

    for review in reviews:
        card_data = None
        for key in _build_review_match_keys(review):
            card_data = html_lookup.get(key)
            if card_data:
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
        if review.get("company_size_raw") is None and card_data["company_size_raw"]:
            review["company_size_raw"] = card_data["company_size_raw"]
        if review.get("reviewer_industry") is None and card_data["reviewer_industry"]:
            review["reviewer_industry"] = card_data["reviewer_industry"]


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
    review_cards = _select_review_cards(soup)

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
    if rating is None:
        match = re.search(r"(\d+(?:\.\d+)?)\s+Overall Rating", card.get_text(" ", strip=True))
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
    if not overall_text:
        paragraphs = [p.get_text(" ", strip=True) for p in card.select("p") if p.get_text(" ", strip=True)]
        if paragraphs:
            overall_text = paragraphs[0][:10000]
            if pros is None and len(paragraphs) > 1:
                pros = paragraphs[1][:5000]
            if cons is None and len(paragraphs) > 2:
                cons = paragraphs[2][:5000]

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
    profile_meta = _extract_live_profile_metadata(card)
    reviewer_name = _get_text(card, '[class*="reviewer"], [class*="author"]')
    reviewer_title = _get_reviewer_title(card) or profile_meta["reviewer_title"]
    reviewer_company = _get_text(card, '[class*="company"], [class*="org"]') or profile_meta["reviewer_company"]
    company_size = _get_text(card, '[class*="size"], [class*="employees"]') or profile_meta["company_size_raw"]
    reviewer_industry = _get_text(card, '[class*="industry"]') or profile_meta["reviewer_industry"]

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
