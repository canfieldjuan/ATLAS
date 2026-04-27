"""
GetApp parser for B2B review scraping.

URL pattern: getapp.com/software/{category-slug}/a/{product-slug}/reviews/?page={n}
Owned by Gartner (same family as Capterra) -- similar HTML structure.
Strategy: Web Unlocker first (Cloudflare protected), then JSON-LD extraction,
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
from collections.abc import Mapping

from bs4 import BeautifulSoup

from ..captcha import CaptchaType, detect_captcha
from ..client import AntiDetectionClient
from ..web_unlocker_http import get_with_web_unlocker
from . import (
    ScrapeResult,
    ScrapeTarget,
    apply_date_cutoff,
    log_page,
    page_has_only_known_source_reviews,
    register_parser,
)

logger = logging.getLogger("atlas.services.scraping.parsers.getapp")

_DOMAIN = "getapp.com"
_BASE_URL = "https://www.getapp.com"
_DEFAULT_PROTECTION_PAGE_STOP_THRESHOLD = 2
_BLOCKED_STOP_REASON = "blocked_or_throttled"


def _review_key(review: dict) -> str:
    """Return a stable key for merging reviews across transports."""
    return (
        str(review.get("dedup_key") or "")
        or str(review.get("source_review_id") or "")
        or hashlib.sha256(
            f"{review.get('summary','')}|{review.get('review_text','')}".encode()
        ).hexdigest()[:16]
    )


def _merge_results(primary: ScrapeResult, continuation: ScrapeResult) -> ScrapeResult:
    """Merge two scrape results, preserving errors, logs, and unique reviews."""
    merged_reviews: list[dict] = []
    seen: set[str] = set()
    for review in primary.reviews + continuation.reviews:
        key = _review_key(review)
        if key in seen:
            continue
        seen.add(key)
        merged_reviews.append(review)
    return ScrapeResult(
        reviews=merged_reviews,
        pages_scraped=primary.pages_scraped + continuation.pages_scraped,
        errors=primary.errors + continuation.errors,
        captcha_attempts=primary.captcha_attempts + continuation.captcha_attempts,
        captcha_types=(primary.captcha_types or []) + (continuation.captcha_types or []),
        captcha_solve_ms=primary.captcha_solve_ms + continuation.captcha_solve_ms,
        page_logs=primary.page_logs + continuation.page_logs,
        stop_reason=continuation.stop_reason or primary.stop_reason,
        resume_page=continuation.resume_page,
    )


def _unlocker_header_detail(headers: Mapping[str, Any] | None) -> str | None:
    if not isinstance(headers, Mapping):
        return None
    error_code = str(headers.get("x-brd-error-code") or headers.get("x-luminati-error-code") or "").strip()
    error_text = str(headers.get("x-brd-error") or headers.get("x-luminati-error") or "").strip()
    if error_code and error_text:
        return f"{error_code}: {error_text}"
    if error_code:
        return error_code
    if error_text:
        return error_text
    return None


class GetAppParser:
    """Parse GetApp review pages using JSON-LD or HTML fallback."""

    source_name = "getapp"
    prefer_residential = True
    version = "getapp:1"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape GetApp reviews -- Web Unlocker first, then HTTP client."""
        from atlas_brain.config import settings

        # Priority 0: SERP-based review page discovery (bypasses Cloudflare)
        if settings.b2b_scrape.serp_api_token:
            try:
                result = await self._scrape_via_serp(target)
                if result.reviews:
                    return result
                logger.info(
                    "SERP discovery for GetApp/%s returned 0 reviews, trying Web Unlocker",
                    target.vendor_name,
                )
            except Exception as exc:
                logger.warning(
                    "SERP discovery failed for GetApp/%s: %s -- falling back",
                    target.vendor_name, exc,
                )

        unlocker_errors: list[str] = []
        result = ScrapeResult(reviews=[], pages_scraped=0, errors=[])
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
                    if result.reviews and result.resume_page is None:
                        return result
                    unlocker_errors.extend(result.errors)
                    logger.warning(
                        "Web Unlocker for %s requires fallback%s",
                        target.vendor_name,
                        f" ({result.errors[0]})" if result.errors else "",
                    )
                except Exception as exc:
                    unlocker_errors.append(f"Web Unlocker exception: {exc}")
                    logger.warning(
                        "Web Unlocker failed for %s: %s -- falling back",
                        target.vendor_name, exc,
                    )

        # Priority 1.5: continue or retry via Bright Data Scraping Browser
        if browser_enabled and (result.resume_page is not None or not result.reviews):
            browser_start = result.resume_page or 1
            seed_ids = {_review_key(review) for review in result.reviews}
            try:
                browser_result = await self._scrape_browser(
                    target,
                    sb_url,
                    start_page=browser_start,
                    seed_seen_ids=seed_ids,
                )
                result = _merge_results(result, browser_result)
                if result.reviews and result.resume_page is None:
                    return result
                unlocker_errors = result.errors.copy()
                logger.warning(
                    "Scraping Browser for %s requires HTTP fallback%s",
                    target.vendor_name,
                    f" ({browser_result.errors[0]})" if browser_result.errors else "",
                )
            except Exception as exc:
                unlocker_errors.append(f"Scraping Browser exception: {exc}")
                logger.warning(
                    "Scraping Browser failed for %s: %s -- falling back",
                    target.vendor_name, exc,
                )

        # Priority 2: curl_cffi HTTP client with residential proxy
        http_start = result.resume_page or 1
        seed_ids = {_review_key(review) for review in result.reviews}
        http_result = await self._scrape_http(
            target,
            client,
            start_page=http_start,
            seed_seen_ids=seed_ids,
        )
        result = _merge_results(result, http_result)
        if unlocker_errors and not result.errors[: len(unlocker_errors)] == unlocker_errors:
            result.errors = unlocker_errors + result.errors
        return result

    # ------------------------------------------------------------------
    # SERP discovery path (bypasses Cloudflare via Google cached pages)
    # ------------------------------------------------------------------

    async def _scrape_via_serp(self, target: ScrapeTarget) -> ScrapeResult:
        """Discover GetApp review pages via Google SERP, then parse."""
        import httpx
        from atlas_brain.config import settings
        from ..serp_discovery import discover_urls

        _GETAPP_SUFFIXES = ["reviews", "alternatives", "pricing"]
        question_urls = await discover_urls(
            site_domain=_DOMAIN,
            vendor_name=target.vendor_name,
            query_suffixes=_GETAPP_SUFFIXES,
            max_results_per_query=5,
        )

        if not question_urls:
            return ScrapeResult(reviews=[], pages_scraped=0, errors=[])

        # Filter to actual review page URLs (not category/compare pages)
        review_urls = [u for u in question_urls if "/reviews" in u or "/a/" in u]
        if not review_urls:
            return ScrapeResult(reviews=[], pages_scraped=0, errors=[])

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for url in review_urls[:5]:
            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url, verify=False, timeout=90.0,
                ) as http:
                    resp = await http.get(url, headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        ),
                    })

                pages_scraped += 1
                if resp.status_code != 200:
                    errors.append(f"SERP URL {url}: HTTP {resp.status_code}")
                    continue

                page_reviews = _parse_json_ld(resp.text, target, seen_ids)
                if not page_reviews:
                    page_reviews = _parse_html(resp.text, target, seen_ids)
                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"SERP URL {url}: {exc}")

            import asyncio
            import random
            await asyncio.sleep(random.uniform(2.0, 4.0))

        logger.info(
            "GetApp SERP scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Web Unlocker path (Bright Data -- handles Cloudflare internally)
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape GetApp via Bright Data Web Unlocker proxy."""
        from atlas_brain.config import settings

        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time
        resume_page: int | None = None
        stop_reason = ""

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
                resp = await get_with_web_unlocker(url, headers=headers, domain=_DOMAIN)

                pages_scraped += 1
                elapsed_ms = int((_time.monotonic() - page_start) * 1000)
                unlocker_detail = _unlocker_header_detail(resp.headers)

                if resp.status_code == 403:
                    detail = _describe_getapp_response(resp.text, resp.status_code, resp.headers)
                    if detail:
                        errors.append(f"Page {page}: blocked (403) via Web Unlocker ({detail})")
                    else:
                        errors.append(f"Page {page}: blocked (403) via Web Unlocker")
                    page_logs.append(log_page(
                        page, url, status_code=403, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["blocked (403) via Web Unlocker"],
                    ))
                    stop_reason = _BLOCKED_STOP_REASON
                    resume_page = page
                    break
                if resp.status_code != 200:
                    detail = _describe_getapp_response(resp.text, resp.status_code, resp.headers)
                    if not detail:
                        detail = unlocker_detail
                    if detail:
                        errors.append(f"Page {page}: HTTP {resp.status_code} ({detail})")
                    else:
                        errors.append(f"Page {page}: HTTP {resp.status_code}")
                    page_logs.append(log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=[f"HTTP {resp.status_code}"],
                    ))
                    if _is_protection_like_response(resp.status_code, detail):
                        stop_reason = _BLOCKED_STOP_REASON
                    resume_page = page
                    break

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
                    detail = _describe_getapp_response(resp.text, resp.status_code, resp.headers)
                    if not detail:
                        detail = unlocker_detail
                    if detail:
                        errors.append(f"Page {page}: 0 reviews after Web Unlocker ({detail})")
                        stop_reason = _BLOCKED_STOP_REASON
                    if page == 1:
                        logger.warning(
                            "GetApp Web Unlocker page 1 returned 0 reviews for %s",
                            target.product_slug,
                        )
                    if reviews:
                        resume_page = page
                    break
                elif page_has_only_known_source_reviews(page_reviews, target):
                    pl.stop_reason = "known_source_reviews"
                    stop_reason = "known_source_reviews"
                    break

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning(
                    "GetApp Web Unlocker page %d failed for %s: %s",
                    page, target.product_slug, exc,
                )
                if reviews:
                    resume_page = page
                break

            await asyncio.sleep(random.uniform(2.0, 5.0))

        logger.info(
            "GetApp Web Unlocker scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            page_logs=page_logs,
            stop_reason=stop_reason,
            resume_page=resume_page,
        )

    # ------------------------------------------------------------------
    # Scraping Browser path (Bright Data cloud Chromium)
    # ------------------------------------------------------------------

    async def _scrape_browser(
        self,
        target: ScrapeTarget,
        ws_url: str,
        *,
        start_page: int = 1,
        seed_seen_ids: set[str] | None = None,
    ) -> ScrapeResult:
        """Scrape GetApp via Bright Data Scraping Browser (cloud Chromium)."""
        from atlas_brain.config import settings
        from playwright.async_api import async_playwright
        from ..browser import solve_browser_challenge

        max_pages = target.max_pages or 15
        timeout_ms = settings.b2b_scrape.playwright_timeout_ms
        protection_stop_threshold = _get_protection_page_stop_threshold()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set(seed_seen_ids or ())
        has_upstream_reviews = start_page > 1 or bool(seed_seen_ids)
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time
        resume_page: int | None = None
        stop_reason = ""
        consecutive_protection_pages = 0

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(ws_url, timeout=timeout_ms)
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = await context.new_page()

                for page_num in range(start_page, max_pages + 1):
                    url = _build_url(target.product_slug, page_num)

                    try:
                        page_start = _time.monotonic()
                        resp = await page.goto(url, wait_until="commit", timeout=timeout_ms)
                        status = resp.status if resp else 0
                        pages_scraped += 1
                        elapsed_ms = int((_time.monotonic() - page_start) * 1000)
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=min(timeout_ms, 15000))
                        except Exception:
                            pass
                        try:
                            await page.wait_for_selector(
                                '[data-review-id], [data-testid="review-card"], [itemtype*="Review"]',
                                timeout=min(timeout_ms, 15000),
                            )
                        except Exception:
                            pass
                        html = await page.content()

                        if await solve_browser_challenge(
                            page,
                            domain=_DOMAIN,
                            html=html,
                            status_code=status,
                        ):
                            retry_start = _time.monotonic()
                            resp = await page.goto(
                                url,
                                wait_until="commit",
                                timeout=timeout_ms,
                            )
                            status = resp.status if resp else 0
                            elapsed_ms += int((_time.monotonic() - retry_start) * 1000)
                            try:
                                await page.wait_for_load_state(
                                    "domcontentloaded",
                                    timeout=min(timeout_ms, 15000),
                                )
                            except Exception:
                                pass
                            html = await page.content()

                        if status == 404:
                            errors.append(f"Page {page_num}: HTTP {status}")
                            pl = log_page(
                                page_num, url, status_code=status, duration_ms=elapsed_ms,
                                response_bytes=len(html), raw_body=html,
                                prior_hashes=prior_hashes, errors=errors[-1:],
                            )
                            page_logs.append(pl)
                            break

                        if status >= 400:
                            detail = _describe_getapp_response(html, status)
                            if detail:
                                errors.append(f"Page {page_num}: HTTP {status} ({detail})")
                            else:
                                errors.append(f"Page {page_num}: HTTP {status}")
                            pl = log_page(
                                page_num, url, status_code=status, duration_ms=elapsed_ms,
                                response_bytes=len(html), raw_body=html,
                                prior_hashes=prior_hashes, errors=errors[-1:],
                            )
                            if _is_protection_like_response(status, detail):
                                consecutive_protection_pages += 1
                                if (
                                    reviews
                                    or has_upstream_reviews
                                    or consecutive_protection_pages >= protection_stop_threshold
                                ):
                                    pl.stop_reason = _BLOCKED_STOP_REASON
                                    stop_reason = _BLOCKED_STOP_REASON
                                    if reviews or has_upstream_reviews:
                                        resume_page = page_num
                                    page_logs.append(pl)
                                    break
                            else:
                                consecutive_protection_pages = 0
                            page_logs.append(pl)
                            if reviews or has_upstream_reviews:
                                resume_page = page_num
                                break
                            continue

                        page_reviews = _parse_json_ld(html, target, seen_ids)
                        if not page_reviews:
                            page_reviews = _parse_html(html, target, seen_ids)
                        page_reviews, cutoff_hit = apply_date_cutoff(page_reviews, target.date_cutoff)
                        reviews.extend(page_reviews)
                        consecutive_protection_pages = 0
                        pl = log_page(
                            page_num, url, status_code=status, duration_ms=elapsed_ms,
                            response_bytes=len(html), reviews=page_reviews,
                            raw_body=html, prior_hashes=prior_hashes,
                            prior_review_ids=prior_review_ids,
                            next_page_found=bool(page_reviews),
                        )
                        if cutoff_hit:
                            pl.stop_reason = "date_cutoff"
                            stop_reason = "date_cutoff"
                        page_logs.append(pl)

                        if not page_reviews:
                            detail = _describe_getapp_response(html, status)
                            if detail:
                                pl.stop_reason = _BLOCKED_STOP_REASON
                                stop_reason = _BLOCKED_STOP_REASON
                            break
                        elif page_has_only_known_source_reviews(page_reviews, target):
                            pl.stop_reason = "known_source_reviews"
                            stop_reason = "known_source_reviews"
                            break

                        await asyncio.sleep(random.uniform(1.5, 3.0))

                    except Exception as exc:
                        errors.append(f"Page {page_num}: {type(exc).__name__}: {str(exc)[:80]}")
                        page_logs.append(log_page(page_num, url, errors=errors[-1:]))
                        if reviews:
                            resume_page = page_num
                        break

                await page.close()
                await browser.close()

        except Exception as exc:
            errors.append(f"Browser connection failed: {exc}")

        logger.info(
            "Scraping Browser for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
            page_logs=page_logs,
            stop_reason=stop_reason,
            resume_page=resume_page,
        )

    # ------------------------------------------------------------------
    # HTTP client path (curl_cffi + residential proxy)
    # ------------------------------------------------------------------

    async def _scrape_http(
        self,
        target: ScrapeTarget,
        client: AntiDetectionClient,
        *,
        start_page: int = 1,
        seed_seen_ids: set[str] | None = None,
    ) -> ScrapeResult:
        """Scrape GetApp via curl_cffi HTTP client."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set(seed_seen_ids or ())
        protection_stop_threshold = _get_protection_page_stop_threshold()
        has_upstream_reviews = start_page > 1 or bool(seed_seen_ids)
        page_logs = []
        prior_hashes: set[str] = set()
        prior_review_ids: set[str] = set()
        import time as _time
        stop_reason = ""

        consecutive_empty = 0
        consecutive_protection_pages = 0
        for page in range(start_page, target.max_pages + 1):
            url = _build_url(target.product_slug, page)

            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+getapp+reviews"
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
                    timeout_seconds=90,
                )
                pages_scraped += 1
                elapsed_ms = int((_time.monotonic() - page_start) * 1000)

                if resp.status_code == 403:
                    detail = _describe_getapp_response(resp.text, resp.status_code)
                    if detail:
                        errors.append(f"Page {page}: blocked (403) -- {detail}")
                    else:
                        errors.append(f"Page {page}: blocked (403) -- Cloudflare challenge")
                    page_logs.append(log_page(
                        page, url, status_code=403, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content or b""), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["blocked (403)"],
                    ))
                    stop_reason = _BLOCKED_STOP_REASON
                    break
                if resp.status_code == 429:
                    errors.append(f"Page {page}: rate limited (429)")
                    page_logs.append(log_page(
                        page, url, status_code=429, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content or b""), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=["rate limited (429)"],
                    ))
                    stop_reason = _BLOCKED_STOP_REASON
                    break
                if resp.status_code != 200:
                    detail = _describe_getapp_response(resp.text, resp.status_code)
                    if detail:
                        errors.append(f"Page {page}: HTTP {resp.status_code} ({detail})")
                    else:
                        errors.append(f"Page {page}: HTTP {resp.status_code}")
                    pl = log_page(
                        page, url, status_code=resp.status_code, duration_ms=elapsed_ms,
                        response_bytes=len(resp.content or b""), raw_body=resp.content,
                        prior_hashes=prior_hashes, errors=[f"HTTP {resp.status_code}"],
                    )
                    if _is_protection_like_response(resp.status_code, detail):
                        consecutive_protection_pages += 1
                        if has_upstream_reviews or consecutive_protection_pages >= protection_stop_threshold:
                            pl.stop_reason = _BLOCKED_STOP_REASON
                            stop_reason = _BLOCKED_STOP_REASON
                            page_logs.append(pl)
                            break
                    else:
                        consecutive_protection_pages = 0
                    page_logs.append(pl)
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
                    detail = _describe_getapp_response(html, resp.status_code)
                    if detail:
                        errors.append(f"Page {page}: 0 reviews ({detail})")
                        pl.stop_reason = _BLOCKED_STOP_REASON
                        stop_reason = _BLOCKED_STOP_REASON
                    if page == 1:
                        logger.warning(
                            "GetApp page 1 returned 0 reviews for %s -- "
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
                consecutive_protection_pages = 0

                if len(reviews) == before:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        logger.info("GetApp: 2 consecutive pages with no new reviews, stopping")
                        break
                else:
                    consecutive_empty = 0

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("GetApp page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "GetApp scrape for %s: %d reviews from %d pages",
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
# URL helpers
# ------------------------------------------------------------------

def _build_url(product_slug: str, page: int) -> str:
    """Build a GetApp reviews URL.

    product_slug in DB is ``{category-slug}/a/{product-slug}``
    (e.g. ``marketing-software/a/mailchimp``).
    Final URL: https://www.getapp.com/software/{category-slug}/a/{product-slug}/reviews/
    """
    base = f"{_BASE_URL}/software/{product_slug}/reviews/"
    if page > 1:
        return f"{base}?page={page}"
    return base


def _get_protection_page_stop_threshold() -> int:
    """Return the configured GetApp protection cutoff with a safe fallback."""
    from atlas_brain.config import settings

    raw = getattr(settings.b2b_scrape, "getapp_protection_page_stop_threshold", None)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return _DEFAULT_PROTECTION_PAGE_STOP_THRESHOLD
    return max(1, raw)


def _is_protection_like_response(status_code: int, detail: str | None) -> bool:
    """Return True when the response looks like a challenge or proxy block page."""
    if status_code in (403, 429, 463):
        return True
    if not detail:
        return False
    return any(
        token in detail
        for token in (
            "challenge",
            "cloudflare",
            "access denied",
            "reject_block",
            "captcha or protection page found",
            "gateway/proxy error page",
        )
    )


def _describe_getapp_response(
    html: str,
    status_code: int,
    headers: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a concise reason string for blocked or empty GetApp responses."""
    header_detail = _unlocker_header_detail(headers)
    if header_detail:
        return header_detail
    captcha_type = detect_captcha(html, status_code)
    if captcha_type != CaptchaType.NONE:
        return f"{captcha_type.value} challenge"

    body = html.lower()
    if "challenge-platform" in body or "_cf_chl_opt" in body or "cloudflare" in body:
        return "possible cloudflare challenge"
    if "access denied" in body:
        return "access denied page"
    if status_code >= 500 and body:
        return "gateway/proxy error page"
    return None


# ------------------------------------------------------------------
# JSON-LD extraction (primary strategy -- same Gartner schema as Capterra)
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
                reviewer_name = author.get("name", "") if isinstance(author, dict) else ""

                # Extract date
                reviewed_at = r.get("datePublished")

                # GetApp JSON-LD may include pros/cons in named sub-properties
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
                    "source": "getapp",
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
# HTML fallback (similar selectors to Capterra -- same Gartner UI kit)
# ------------------------------------------------------------------

def _parse_html(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse GetApp review page HTML as fallback."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # GetApp review containers -- try specific selectors first, then broader
    review_cards = soup.select(
        '[data-testid="review-card"], '
        '[data-review-id], '
        '.review-card, '
        '[class*="ReviewCard"], '
        'div[class*="review-"][class*="card"]'
    )

    if not review_cards:
        # Broader fallback -- GetApp uses similar patterns to Capterra
        review_cards = soup.select(
            'div[id^="review-"], '
            'div[class*="review-content"], '
            'div[class*="review-listing"], '
            '[itemtype*="Review"]'
        )

    for card in review_cards:
        try:
            review = _parse_getapp_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.warning("Failed to parse GetApp review card", exc_info=True)

    return reviews


def _parse_getapp_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a GetApp review card."""
    # Review ID
    review_id = (
        card.get("data-review-id", "")
        or card.get("id", "")
    )
    if not review_id:
        review_id = hashlib.sha256(
            card.get_text(strip=True)[:200].encode()
        ).hexdigest()[:16]

    # Rating (star-based, 1-5)
    rating = _extract_rating(card)

    # Review title / summary
    summary = None
    title_el = card.select_one(
        '[class*="review-title"], [class*="ReviewTitle"], '
        '[itemprop="name"], h3, h4'
    )
    if title_el:
        summary = title_el.get_text(strip=True)[:500]

    # Pros / Cons -- GetApp separates these like Capterra
    pros = _extract_pros_cons(card, "pros", "like", "best", "advantage")
    cons = _extract_pros_cons(card, "cons", "dislike", "worst", "disadvantage")

    # Overall review text
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

    # Combine if no overall text found
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
    reviewer_name = _get_text(card, '[class*="reviewer"], [class*="author"], [itemprop="author"]')
    reviewer_title = _get_text(card, '[class*="title"], [class*="job"], [class*="role"]')
    reviewer_company = _get_text(card, '[class*="company"], [class*="org"]')
    company_size = _get_text(card, '[class*="size"], [class*="employees"]')
    reviewer_industry = _get_text(card, '[class*="industry"], [class*="sector"]')

    # Date
    reviewed_at = None
    date_el = card.select_one("time, [class*='date'], [itemprop='datePublished']")
    if date_el:
        reviewed_at = (
            date_el.get("datetime")
            or date_el.get("content")
            or date_el.get_text(strip=True)
        )

    return {
        "source": "getapp",
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
    """Extract star rating from a review card.

    GetApp uses several rating patterns: aria-label with star count,
    itemprop ratingValue, or class-based filled-star counting.
    """
    # Try itemprop first (most structured)
    rating_el = card.select_one('[itemprop="ratingValue"]')
    if rating_el:
        val = rating_el.get("content") or rating_el.get_text(strip=True)
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Try aria-label with numeric rating
    for el in card.select('[class*="star"], [class*="rating"], [aria-label*="star"]'):
        aria = el.get("aria-label", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+", aria)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass
        # Simpler pattern: just a number in the aria label
        match = re.search(r"(\d+(?:\.\d+)?)", aria)
        if match:
            try:
                rating = float(match.group(1))
                if 1.0 <= rating <= 5.0:
                    return rating
            except (ValueError, TypeError):
                pass

    # Count filled stars as last resort
    filled = card.select('[class*="star--filled"], [class*="star-full"], .star.fill')
    if filled:
        count = len(filled)
        if 1 <= count <= 5:
            return float(count)

    return None


def _extract_pros_cons(card, *keywords: str) -> str | None:
    """Extract pros or cons section from a GetApp review card.

    GetApp structures reviews with labeled sections similar to Capterra.
    Searches by heading + sibling pattern first, then class-based matching.
    """
    # Strategy 1: Heading + sibling pattern
    for heading in card.select("h5, h4, h3, [class*='heading'], [class*='label'], dt"):
        text = heading.get_text(strip=True).lower()
        if any(kw in text for kw in keywords):
            sibling = heading.find_next_sibling()
            if sibling:
                content = sibling.get_text(strip=True)
                if content and len(content) > 5:
                    return content[:5000]

    # Strategy 2: Class-based matching
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
register_parser(GetAppParser())
