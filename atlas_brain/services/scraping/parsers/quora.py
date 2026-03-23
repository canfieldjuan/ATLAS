"""
Quora parser for B2B review scraping.

Scrapes Quora search results and question pages for vendor-related opinions.
Quora has NO public API -- must scrape HTML.

Primary: Bright Data Web Unlocker (Quora blocks scrapers heavily).
Fallback: curl_cffi HTTP client with residential proxy.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
from datetime import date, timedelta
from urllib.parse import quote_plus, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, Tag

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, apply_date_cutoff, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.quora")

_DOMAIN = "quora.com"
_BASE_URL = "https://www.quora.com"
_MIN_TEXT_LEN = 100

# Multiple search queries per vendor to maximise coverage
_SEARCH_SUFFIXES = [
    "alternative",
    "alternatives",
    "migrate from",
    "migration",
    "switching from",
    "vs",
    "review",
    "reviews",
    "problems",
    "complaints",
    "pricing",
    "competitors",
]
_INTENT_HINTS = (
    "alternative",
    "alternatives",
    "switch",
    "switching",
    "vs",
    "versus",
    "review",
    "reviews",
    "problem",
    "problems",
    "complaint",
    "complaints",
    "competitor",
    "competitors",
    "cheaper",
    "pricing",
    "worth",
    "migrate",
    "migration",
    "replace",
    "replacement",
)
_PROMO_HINTS = (
    "sponsored by",
    "promoted by",
    "learn more",
    "shop now",
)
_GENERIC_PRODUCT_URL_TOKENS = {
    "analytics",
    "automation",
    "business",
    "cloud",
    "commerce",
    "crm",
    "customer",
    "data",
    "email",
    "helpdesk",
    "hub",
    "infrastructure",
    "management",
    "marketing",
    "platform",
    "product",
    "project",
    "sales",
    "service",
    "services",
    "software",
    "support",
    "tool",
    "tools",
}
_SERP_MIN_LOOKBACK_DAYS = 30
_SERP_DISCOVERY_TIMEOUT_SECONDS = 12.0
_UNLOCKER_TIMEOUT_SECONDS = 30.0
_REFRESH_TIMEOUT_SECONDS = 25.0

_DATE_TEXT_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)

# Headers mimicking a real browser session on Quora
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


class QuoraParser:
    """Parse Quora search results and answers for B2B vendor opinions."""

    source_name = "quora"
    prefer_residential = True
    version = "quora:2"

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Quora -- SERP discovery first, then Web Unlocker, then HTTP."""
        from atlas_brain.config import settings

        # Priority 0: SERP-based question discovery (bypasses login wall)
        if settings.b2b_scrape.serp_api_token:
            try:
                result = await self._scrape_via_serp(target, client)
                if result.reviews:
                    return result
                if not _should_try_http_fallback(target):
                    return result
                logger.info(
                    "SERP discovery for %s returned 0 reviews, trying Web Unlocker",
                    target.vendor_name,
                )
            except Exception as exc:
                logger.warning(
                    "SERP discovery failed for %s: %s -- falling back",
                    target.vendor_name, exc,
                )

        # Priority 1: Bright Data Web Unlocker (handles Quora anti-bot automatically)
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
                    if not _should_try_http_fallback(target):
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
                    if not _should_try_http_fallback(target):
                        return ScrapeResult(reviews=[], pages_scraped=0, errors=[str(exc)])

        # Priority 2: curl_cffi HTTP client with residential proxy
        return await self._scrape_http(target, client)

    # ------------------------------------------------------------------
    # SERP discovery path (bypasses Quora login wall via Google)
    # ------------------------------------------------------------------

    async def _scrape_via_serp(
        self,
        target: ScrapeTarget,
        client: AntiDetectionClient,
    ) -> ScrapeResult:
        """Discover Quora question URLs via Google SERP, then scrape each."""
        import httpx
        from atlas_brain.config import settings
        from ..serp_discovery import discover_urls_with_snippets

        search_suffixes = _build_serp_query_suffixes(target)
        discovered_results = await discover_urls_with_snippets(
            site_domain=_DOMAIN,
            vendor_name=target.vendor_name,
            query_suffixes=search_suffixes,
            max_results_per_query=8,
            timeout=_SERP_DISCOVERY_TIMEOUT_SECONDS,
        )
        question_urls = _select_serp_question_urls(discovered_results, target)
        if not question_urls and target.date_cutoff:
            question_urls = await _load_historical_question_urls(
                target,
                limit=_max_question_urls(target),
            )

        if not question_urls:
            return ScrapeResult(reviews=[], pages_scraped=0, errors=[])

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        browser_ws_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        browser_candidates: list[str] = []
        for url in question_urls[:_max_question_urls(target)]:
            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url, verify=False, timeout=_REFRESH_TIMEOUT_SECONDS,
                ) as http:
                    resp = await http.get(url, headers=_BROWSER_HEADERS)

                if resp.status_code != 200:
                    browser_candidates.append(url)
                    errors.append(f"Question {url}: HTTP {resp.status_code}")
                    continue

                pages_scraped += 1
                page_reviews = _parse_question_page(
                    resp.text, target, seen_ids, page_url=url,
                )
                page_reviews, _ = apply_date_cutoff(page_reviews, target.date_cutoff)
                reviews.extend(page_reviews)

            except Exception as exc:
                client_reviews = await self._fetch_single_question_http(
                    url,
                    target,
                    client,
                    seen_ids,
                    errors,
                )
                if client_reviews:
                    pages_scraped += 1
                    reviews.extend(client_reviews)
                else:
                    browser_candidates.append(url)
                    errors.append(f"Question {url}: {_describe_error(exc)}")
                    logger.debug(
                        "Quora SERP question page failed: %s", exc,
                    )

            await asyncio.sleep(random.uniform(2.0, 4.0))

        if not reviews and browser_ws_url and browser_candidates:
            browser_reviews, browser_pages = await self._fetch_question_urls_browser(
                browser_candidates,
                target,
                seen_ids,
                errors,
                browser_ws_url,
            )
            reviews.extend(browser_reviews)
            pages_scraped += browser_pages

        logger.info(
            "Quora SERP scrape for %s: %d reviews from %d question pages",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(
            reviews=reviews,
            pages_scraped=pages_scraped,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Web Unlocker path (Bright Data -- handles anti-bot internally)
    # ------------------------------------------------------------------

    async def _scrape_web_unlocker(self, target: ScrapeTarget) -> ScrapeResult:
        """Scrape Quora via Bright Data Web Unlocker proxy."""
        import httpx
        from atlas_brain.config import settings

        proxy_url = settings.b2b_scrape.web_unlocker_url.strip()
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        discovered_question_urls: list[str] = []
        seen_question_urls: set[str] = set()

        queries = _build_search_queries(target)

        for query in queries[:target.max_pages]:
            url = f"{_BASE_URL}/search?q={quote_plus(query)}&type=answer"

            headers = {
                **_BROWSER_HEADERS,
                "Referer": f"https://www.google.com/search?q={quote_plus(query)}+site:quora.com",
            }

            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url, verify=False, timeout=_UNLOCKER_TIMEOUT_SECONDS,
                ) as http:
                    resp = await http.get(url, headers=headers)

                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Query '{query}': blocked (403) via Web Unlocker")
                    continue
                if resp.status_code != 200:
                    errors.append(f"Query '{query}': HTTP {resp.status_code}")
                    continue

                for question_url in _extract_question_urls_from_search(resp.text):
                    if not _is_historical_vendor_question_url(question_url, "", target):
                        continue
                    if question_url in seen_question_urls:
                        continue
                    seen_question_urls.add(question_url)
                    discovered_question_urls.append(question_url)

            except Exception as exc:
                errors.append(f"Query '{query}': {exc}")
                logger.warning(
                    "Quora Web Unlocker query '%s' failed for %s: %s",
                    query, target.vendor_name, exc,
                )

            # Inter-query delay to avoid rate limits
            await asyncio.sleep(random.uniform(3.0, 6.0))

        # Fetch discovered question URLs instead of persisting search-result snippets.
        if not discovered_question_urls and target.date_cutoff:
            discovered_question_urls = await _load_historical_question_urls(
                target,
                limit=_max_question_urls(target),
            )
        if discovered_question_urls:
            fetched_reviews = await self._fetch_question_urls_unlocker(
                discovered_question_urls[:_max_question_urls(target)],
                proxy_url,
                target,
                seen_ids,
                errors,
            )
            reviews.extend(fetched_reviews)
            pages_scraped += len(discovered_question_urls[:_max_question_urls(target)])

        # Direct-question fallback is evergreen and expensive. Skip it on
        # incremental runs where we only want newly surfaced pages.
        if _should_try_direct_question(target):
            direct_reviews = await self._fetch_direct_question_unlocker(
                target, proxy_url, seen_ids, errors,
            )
            reviews.extend(direct_reviews)
            if direct_reviews:
                pages_scraped += 1

        logger.info(
            "Quora Web Unlocker scrape for %s: %d reviews from %d requests",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    async def _fetch_direct_question_unlocker(
        self,
        target: ScrapeTarget,
        proxy_url: str,
        seen_ids: set[str],
        errors: list[str],
    ) -> list[dict]:
        """Try fetching a direct question URL via Web Unlocker."""
        import httpx

        # Normalise vendor name for URL: "Salesforce CRM" -> "Salesforce-CRM"
        slug = target.vendor_name.replace(" ", "-")
        url = f"{_BASE_URL}/What-are-the-best-alternatives-to-{slug}"

        headers = {
            **_BROWSER_HEADERS,
            "Referer": f"https://www.google.com/search?q=best+alternatives+to+{quote_plus(target.vendor_name)}+quora",
        }

        try:
            async with httpx.AsyncClient(
                proxy=proxy_url, verify=False, timeout=_REFRESH_TIMEOUT_SECONDS,
            ) as http:
                resp = await http.get(url, headers=headers)

            if resp.status_code != 200:
                return []

            page_reviews = _parse_question_page(resp.text, target, seen_ids, page_url=url)
            page_reviews, _ = apply_date_cutoff(page_reviews, target.date_cutoff)
            return page_reviews
        except Exception as exc:
            errors.append(f"Direct question URL: {exc}")
            logger.warning(
                "Quora direct question failed for %s: %s",
                target.vendor_name, exc,
            )
            return []

    async def _fetch_question_urls_unlocker(
        self,
        question_urls: list[str],
        proxy_url: str,
        target: ScrapeTarget,
        seen_ids: set[str],
        errors: list[str],
    ) -> list[dict]:
        """Fetch concrete question pages discovered from search HTML."""
        import httpx
        from atlas_brain.config import settings

        reviews: list[dict] = []
        browser_candidates: list[str] = []
        for url in question_urls:
            if not _is_scrapable_quora_url(url):
                continue
            try:
                async with httpx.AsyncClient(
                    proxy=proxy_url, verify=False, timeout=_REFRESH_TIMEOUT_SECONDS,
                ) as http:
                    resp = await http.get(url, headers=_BROWSER_HEADERS)
                if resp.status_code != 200:
                    browser_candidates.append(url)
                    errors.append(f"Question {url}: HTTP {resp.status_code}")
                    continue
                page_reviews = _parse_question_page(resp.text, target, seen_ids, page_url=url)
                page_reviews, _ = apply_date_cutoff(page_reviews, target.date_cutoff)
                reviews.extend(page_reviews)
            except Exception as exc:
                browser_candidates.append(url)
                errors.append(f"Question {url}: {_describe_error(exc)}")
                logger.debug("Quora discovered question failed: %s", exc, exc_info=True)
            await asyncio.sleep(random.uniform(2.0, 4.0))
        browser_ws_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
        if not reviews and browser_ws_url and browser_candidates:
            browser_reviews, _ = await self._fetch_question_urls_browser(
                browser_candidates,
                target,
                seen_ids,
                errors,
                browser_ws_url,
            )
            reviews.extend(browser_reviews)
        return reviews

    async def _fetch_single_question_http(
        self,
        url: str,
        target: ScrapeTarget,
        client: AntiDetectionClient,
        seen_ids: set[str],
        errors: list[str],
    ) -> list[dict]:
        """Try a tightly-scoped residential fallback for a single known-relevant question URL."""
        if not _is_scrapable_quora_url(url):
            return []
        try:
            resp = await client.get(
                url,
                domain=_DOMAIN,
                referer=f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+site:quora.com",
                sticky_session=True,
                prefer_residential=True,
            )
            if resp.status_code != 200:
                errors.append(f"Question {url}: HTTP {resp.status_code}")
                return []
            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "text" not in ct:
                errors.append(f"Question {url}: unexpected content-type ({ct[:40]})")
                return []
            page_reviews = _parse_question_page(resp.text, target, seen_ids, page_url=url)
            page_reviews, _ = apply_date_cutoff(page_reviews, target.date_cutoff)
            return page_reviews
        except Exception as exc:
            errors.append(f"Question {url}: {_describe_error(exc)}")
            logger.debug("Quora single-question HTTP fallback failed: %s", exc, exc_info=True)
            return []

    async def _fetch_question_urls_browser(
        self,
        question_urls: list[str],
        target: ScrapeTarget,
        seen_ids: set[str],
        errors: list[str],
        ws_url: str,
    ) -> tuple[list[dict], int]:
        """Fetch a few known-relevant Quora question pages via scraping browser."""
        from atlas_brain.config import settings
        from playwright.async_api import async_playwright

        reviews: list[dict] = []
        pages_scraped = 0
        timeout_ms = settings.b2b_scrape.playwright_timeout_ms
        referer = f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+site:quora.com"

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.connect_over_cdp(ws_url, timeout=timeout_ms)
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                try:
                    for url in question_urls[:_max_question_urls(target)]:
                        page = await context.new_page()
                        try:
                            resp = await page.goto(
                                url,
                                wait_until="commit",
                                timeout=timeout_ms,
                                referer=referer,
                            )
                            status = resp.status if resp else 0
                            try:
                                await page.wait_for_load_state("domcontentloaded", timeout=15000)
                            except Exception:
                                pass
                            if status not in (200, 0):
                                errors.append(f"Question {url}: browser HTTP {status}")
                                continue
                            try:
                                await page.wait_for_selector("div.q-box, time[datetime]", timeout=15000)
                            except Exception:
                                pass
                            html = await page.content()
                            page_reviews = _parse_question_page(html, target, seen_ids, page_url=url)
                            page_reviews, _ = apply_date_cutoff(page_reviews, target.date_cutoff)
                            reviews.extend(page_reviews)
                            pages_scraped += 1
                            await asyncio.sleep(random.uniform(1.5, 3.0))
                        except Exception as exc:
                            errors.append(f"Question {url}: browser {_describe_error(exc)}")
                        finally:
                            try:
                                await page.close()
                            except Exception:
                                pass
                finally:
                    await browser.close()
        except Exception as exc:
            errors.append(f"Browser connection failed: {_describe_error(exc)}")

        return reviews, pages_scraped

    # ------------------------------------------------------------------
    # HTTP client path (curl_cffi + residential proxy)
    # ------------------------------------------------------------------

    async def _scrape_http(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape Quora via curl_cffi HTTP client with residential proxy."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()
        discovered_question_urls: list[str] = []
        seen_question_urls: set[str] = set()

        queries = _build_search_queries(target)

        for query in queries[:target.max_pages]:
            url = f"{_BASE_URL}/search?q={quote_plus(query)}&type=answer"
            referer = f"https://www.google.com/search?q={quote_plus(query)}+site:quora.com"

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
                    errors.append(f"Query '{query}': blocked (403)")
                    continue
                if resp.status_code == 429:
                    errors.append(f"Query '{query}': rate limited (429)")
                    break
                if resp.status_code != 200:
                    errors.append(f"Query '{query}': HTTP {resp.status_code}")
                    continue

                # Guard against non-HTML responses
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Query '{query}': unexpected content-type ({ct[:40]})")
                    continue

                for question_url in _extract_question_urls_from_search(resp.text):
                    if not _is_historical_vendor_question_url(question_url, "", target):
                        continue
                    if question_url in seen_question_urls:
                        continue
                    seen_question_urls.add(question_url)
                    discovered_question_urls.append(question_url)

            except Exception as exc:
                errors.append(f"Query '{query}': {exc}")
                logger.warning(
                    "Quora HTTP query '%s' failed for %s: %s",
                    query, target.vendor_name, exc,
                )

        if discovered_question_urls:
            fetched_reviews = await self._fetch_question_urls_http(
                discovered_question_urls[:_max_question_urls(target)],
                target,
                client,
                seen_ids,
                errors,
            )
            reviews.extend(fetched_reviews)
            pages_scraped += len(discovered_question_urls[:_max_question_urls(target)])
        elif target.date_cutoff:
            historical_urls = await _load_historical_question_urls(
                target,
                limit=_max_question_urls(target),
            )
            if historical_urls:
                fetched_reviews = await self._fetch_question_urls_http(
                    historical_urls,
                    target,
                    client,
                    seen_ids,
                    errors,
                )
                reviews.extend(fetched_reviews)
                pages_scraped += len(historical_urls)

        if _should_try_direct_question(target):
            direct_reviews = await self._fetch_direct_question_http(
                target, client, seen_ids, errors,
            )
            reviews.extend(direct_reviews)
            if direct_reviews:
                pages_scraped += 1

        logger.info(
            "Quora HTTP scrape for %s: %d reviews from %d requests",
            target.vendor_name, len(reviews), pages_scraped,
        )
        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)

    async def _fetch_direct_question_http(
        self,
        target: ScrapeTarget,
        client: AntiDetectionClient,
        seen_ids: set[str],
        errors: list[str],
    ) -> list[dict]:
        """Try fetching a direct question URL via HTTP client."""
        slug = target.vendor_name.replace(" ", "-")
        url = f"{_BASE_URL}/What-are-the-best-alternatives-to-{slug}"
        referer = f"https://www.google.com/search?q=best+alternatives+to+{quote_plus(target.vendor_name)}+quora"

        try:
            resp = await client.get(
                url,
                domain=_DOMAIN,
                referer=referer,
                sticky_session=True,
                prefer_residential=True,
            )
            if resp.status_code != 200:
                return []

            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "text" not in ct:
                return []

            page_reviews = _parse_question_page(resp.text, target, seen_ids, page_url=url)
            page_reviews, _ = apply_date_cutoff(page_reviews, target.date_cutoff)
            return page_reviews
        except Exception as exc:
            errors.append(f"Direct question URL: {exc}")
            logger.warning(
                "Quora direct question HTTP failed for %s: %s",
                target.vendor_name, exc,
            )
            return []

    async def _fetch_question_urls_http(
        self,
        question_urls: list[str],
        target: ScrapeTarget,
        client: AntiDetectionClient,
        seen_ids: set[str],
        errors: list[str],
    ) -> list[dict]:
        """Fetch concrete question pages discovered from search HTML."""
        from atlas_brain.config import settings

        reviews: list[dict] = []
        browser_candidates: list[str] = []
        for url in question_urls:
            if not _is_scrapable_quora_url(url):
                continue
            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+site:quora.com",
                    sticky_session=True,
                    prefer_residential=True,
                )
                if resp.status_code != 200:
                    browser_candidates.append(url)
                    errors.append(f"Question {url}: HTTP {resp.status_code}")
                    continue
                ct = resp.headers.get("content-type", "")
                if "html" not in ct and "text" not in ct:
                    errors.append(f"Question {url}: unexpected content-type ({ct[:40]})")
                    continue
                page_reviews = _parse_question_page(resp.text, target, seen_ids, page_url=url)
                page_reviews, _ = apply_date_cutoff(page_reviews, target.date_cutoff)
                reviews.extend(page_reviews)
            except Exception as exc:
                browser_candidates.append(url)
                errors.append(f"Question {url}: {_describe_error(exc)}")
                logger.debug("Quora discovered question HTTP failed: %s", exc, exc_info=True)
        browser_ws_url = settings.b2b_scrape.scraping_browser_ws_url.strip()
        if not reviews and browser_ws_url and browser_candidates:
            browser_reviews, _ = await self._fetch_question_urls_browser(
                browser_candidates,
                target,
                seen_ids,
                errors,
                browser_ws_url,
            )
            reviews.extend(browser_reviews)
        return reviews


# ------------------------------------------------------------------
# Query building
# ------------------------------------------------------------------

def _build_search_queries(target: ScrapeTarget) -> list[str]:
    """Build multiple search queries for a vendor."""
    vendor_name = target.vendor_name
    queries = [f"{vendor_name} {suffix}" for suffix in _SEARCH_SUFFIXES]
    if target.date_cutoff:
        # Quora's own search is not date-aware; narrow incremental runs so we
        # do not keep walking the same evergreen result set.
        return queries[:4]
    return queries


def _build_serp_query_suffixes(target: ScrapeTarget) -> list[str]:
    """Build Google SERP discovery suffixes with optional recency filter."""
    discovery_cutoff = _serp_discovery_cutoff(target)
    if not discovery_cutoff:
        return list(_SEARCH_SUFFIXES)
    return [f"{suffix} after:{discovery_cutoff}" for suffix in _SEARCH_SUFFIXES[:4]]


def _max_question_urls(target: ScrapeTarget) -> int:
    """Cap expensive question-page fetches on incremental runs."""
    if target.date_cutoff:
        return min(5, target.max_pages)
    return min(10, target.max_pages)


def _should_try_direct_question(target: ScrapeTarget) -> bool:
    """Only hit the evergreen direct-question fallback for initial scrapes."""
    return not bool(target.date_cutoff)


def _should_try_http_fallback(target: ScrapeTarget) -> bool:
    """Incremental Quora runs should not fall back to the raw HTTP transport."""
    return not bool(target.date_cutoff)


def _describe_error(exc: Exception) -> str:
    """Produce a non-empty error description for logs."""
    text = str(exc).strip()
    if text:
        return text
    return exc.__class__.__name__


def _serp_discovery_cutoff(target: ScrapeTarget) -> str | None:
    """Use a wider recency window for Google discovery than the hard scrape cutoff."""
    if not target.date_cutoff:
        return None
    try:
        cutoff = date.fromisoformat(target.date_cutoff)
    except ValueError:
        return target.date_cutoff
    floor_cutoff = date.today() - timedelta(days=_SERP_MIN_LOOKBACK_DAYS)
    return str(min(cutoff, floor_cutoff))


def _vendor_url_tokens(target: ScrapeTarget) -> set[str]:
    """Build vendor/product tokens we expect to see in relevant Quora URLs."""
    tokens: set[str] = set()

    def _add_tokens(value: str, *, allow_parts: bool) -> None:
        text = value.strip().lower()
        if not text:
            return
        compact = re.sub(r"[^a-z0-9]+", "", text)
        dashed = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
        if compact:
            tokens.add(compact)
        if dashed:
            tokens.add(dashed)
        if not allow_parts:
            return
        parts = [part for part in re.split(r"[^a-z0-9]+", text) if part]
        acronym = "".join(part[0] for part in parts if part)
        if len(acronym) >= 2:
            tokens.add(acronym)
        for part in parts:
            if len(part) >= 4 and part not in _GENERIC_PRODUCT_URL_TOKENS:
                tokens.add(part)

    _add_tokens(str(target.vendor_name or ""), allow_parts=True)
    _add_tokens(str(target.product_name or ""), allow_parts=False)
    _add_tokens(str(target.product_slug or ""), allow_parts=False)
    return {token for token in tokens if token}


def _url_matches_vendor(url: str, target: ScrapeTarget) -> bool:
    """Require discovered Quora URLs to at least reference the vendor/product."""
    path = (urlparse(url).path or "").lower()
    compact_path = re.sub(r"[^a-z0-9]+", "", path)
    dashed_path = re.sub(r"[^a-z0-9]+", "-", path).strip("-")
    for token in _vendor_url_tokens(target):
        if token in compact_path or token in dashed_path:
            return True
    return False


def _text_matches_vendor(text: str, target: ScrapeTarget) -> bool:
    """Check vendor/product presence in a snippet-like text blob."""
    if not text:
        return False
    lowered = text.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    for token in _vendor_url_tokens(target):
        if token in compact:
            return True
        if len(token) >= 4 and re.search(rf"\b{re.escape(token)}\b", lowered):
            return True
    return False


def _primary_vendor_phrases(target: ScrapeTarget) -> list[str]:
    """Build the strongest vendor/product phrases for snippet-level matching."""
    phrases: list[str] = []
    for value in (target.vendor_name, target.product_name, target.product_slug):
        text = str(value or "").strip().lower()
        if not text:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
        compact = re.sub(r"[^a-z0-9]+", "", text).strip()
        for candidate in (normalized, compact):
            if candidate and candidate not in phrases:
                phrases.append(candidate)
    return phrases


def _has_intent_signal(text: str) -> bool:
    """Look for review/comparison language that makes a Quora page seller-relevant."""
    lowered = text.lower()
    return any(hint in lowered for hint in _INTENT_HINTS)


def _snippet_has_vendor_intent(snippet: str, target: ScrapeTarget) -> bool:
    """Require vendor/product and review intent to appear near each other in snippets."""
    lowered = re.sub(r"\s+", " ", str(snippet or "").lower()).strip()
    if not lowered:
        return False
    for phrase in _primary_vendor_phrases(target):
        pattern = rf"(?:{re.escape(phrase)}.{{0,80}}(?:{'|'.join(map(re.escape, _INTENT_HINTS))})|(?:{'|'.join(map(re.escape, _INTENT_HINTS))}).{{0,80}}{re.escape(phrase)})"
        if re.search(pattern, lowered):
            return True
    return False


def _score_serp_candidate(url: str, snippet: str, target: ScrapeTarget) -> int:
    """Score discovered Quora candidates so we fetch the most promising ones first."""
    score = 0
    if _url_matches_vendor(url, target):
        score += 4
    if _text_matches_vendor(snippet, target):
        score += 3
    if _has_intent_signal(url):
        score += 2
    if _has_intent_signal(snippet):
        score += 2
    if _classify_quora_url(url) == "question_page":
        score += 1
    return score


def _select_serp_question_urls(
    discovered_results: list[dict[str, str]],
    target: ScrapeTarget,
) -> list[str]:
    """Pick vendor-relevant Quora URLs from SERP results using URL + snippet relevance."""
    best_scores: dict[str, int] = {}
    for item in discovered_results:
        url = _normalize_quora_url(item.get("url"))
        snippet = str(item.get("snippet") or "")
        if not url or not _is_scrapable_quora_url(url):
            continue
        url_vendor_match = _url_matches_vendor(url, target)
        snippet_vendor_intent = _snippet_has_vendor_intent(snippet, target)
        if target.date_cutoff and not (url_vendor_match and _has_intent_signal(url)):
            continue
        if not url_vendor_match and not snippet_vendor_intent:
            continue
        score = _score_serp_candidate(url, snippet, target)
        if score < 4:
            continue
        if score < 6 and not (_has_intent_signal(url) or snippet_vendor_intent):
            continue
        best_scores[url] = max(best_scores.get(url, 0), score)
    ranked = sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
    return [url for url, _ in ranked]


def _is_historical_vendor_question_url(
    url: str | None,
    summary: str,
    target: ScrapeTarget,
) -> bool:
    """Only refresh vendor-specific historical Quora questions with seller intent."""
    normalized = _normalize_quora_url(url)
    if not normalized or _classify_quora_url(normalized) != "question_page":
        return False
    if not _url_matches_vendor(normalized, target):
        return False
    return _has_intent_signal(normalized)


def _question_root_url(value: str | None) -> str | None:
    """Convert Quora answer permalinks to their parent question page."""
    url = _normalize_quora_url(value)
    if not url:
        return None
    parsed = urlparse(url)
    path = parsed.path or "/"
    if "/answer/" in path:
        path = path.split("/answer/", 1)[0]
    elif "/answers/" in path:
        path = path.split("/answers/", 1)[0]
    normalized = parsed._replace(path=path, query="", fragment="")
    root = urlunparse(normalized)
    return root if _classify_quora_url(root) == "question_page" else None


async def _load_historical_question_urls(
    target: ScrapeTarget,
    *,
    limit: int,
) -> list[str]:
    """Refresh a small set of historically productive question pages for this vendor."""
    try:
        from atlas_brain.storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return []
        rows = await pool.fetch(
            """
            SELECT source_url, summary, COUNT(*) AS row_count, MAX(imported_at) AS last_seen
            FROM b2b_reviews
            WHERE source = 'quora'
              AND vendor_name = $1
            GROUP BY source_url, summary
            ORDER BY COUNT(*) DESC, MAX(imported_at) DESC
            LIMIT 200
            """,
            target.vendor_name,
        )
    except Exception:
        logger.debug("Failed to load historical Quora question URLs", exc_info=True)
        return []

    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        question_url = _question_root_url(row["source_url"])
        if not question_url:
            question_url = _normalize_quora_url(row["source_url"])
        summary = str(row["summary"] or "")
        if not _is_historical_vendor_question_url(question_url, summary, target):
            continue
        entry = grouped.setdefault(
            question_url,
            {"count": 0, "summary": "", "last_seen": row["last_seen"]},
        )
        entry["count"] = int(entry["count"]) + int(row["row_count"] or 0)
        if summary and len(summary) > len(str(entry["summary"] or "")):
            entry["summary"] = summary
        if row["last_seen"] and (not entry["last_seen"] or row["last_seen"] > entry["last_seen"]):
            entry["last_seen"] = row["last_seen"]

    ranked = []
    for url, entry in grouped.items():
        count = int(entry["count"] or 0)
        summary = str(entry["summary"] or "")
        score = count * 10
        if _score_serp_candidate(url, summary, target) >= 4:
            score += 5
        if count < 2:
            continue
        ranked.append((score, url))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, url in ranked[:limit]]


# ------------------------------------------------------------------
# HTML parsing helpers
# ------------------------------------------------------------------

def _parse_search_results(
    html: str, target: ScrapeTarget, seen_ids: set[str],
) -> list[dict]:
    """Legacy stub: search pages are URL discovery pages, not review pages."""
    _ = (html, target, seen_ids)
    return []


def _parse_question_page(
    html: str, target: ScrapeTarget, seen_ids: set[str],
    *,
    page_url: str | None = None,
) -> list[dict]:
    """Parse a Quora question page for full answers."""
    if page_url and not _is_scrapable_quora_url(page_url):
        return []
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # Extract the question title for context
    question_title = _extract_question_title(soup)

    # Find answer containers on a question page
    answer_containers = _find_answer_containers(soup)

    for container in answer_containers:
        try:
            review = _parse_answer_container(
                container,
                target,
                seen_ids,
                question_title=question_title,
                fallback_url=page_url,
            )
            if review:
                reviews.append(review)
        except Exception:
            logger.debug("Failed to parse Quora question-page answer", exc_info=True)

    return reviews


def _find_answer_containers(soup: BeautifulSoup) -> list[Tag]:
    """Find answer containers using multiple selector strategies.

    Quora's HTML is heavily obfuscated with dynamic class names, so we try
    several selector patterns in priority order.
    """
    # Strategy 1: modern answer cards with stable annotate markers.
    containers = [
        c
        for c in soup.select('div[class*="dom_annotate_question_answer_item_"]')
        if _is_valid_answer_container(c)
    ]
    if containers:
        return containers

    # Strategy 2: direct answer-content wrappers seen in browser-rendered pages.
    containers = []
    seen_ids: set[int] = set()
    for content in soup.select(
        'div.puppeteer_test_answer_content, div[class*="spacing_log_answer_content"]'
    ):
        candidate = content
        parent = content.find_parent('div[class*="dom_annotate_question_answer_item_"]')
        if isinstance(parent, Tag):
            candidate = parent
        marker = id(candidate)
        if marker in seen_ids or not _is_valid_answer_container(candidate):
            continue
        seen_ids.add(marker)
        containers.append(candidate)
    if containers:
        return containers

    # Strategy 3: older Answer-related class fragments.
    containers = [
        c
        for c in soup.select('div[class*="Answer"], div[class*="AnswerBase"], div[class*="answer"]')
        if _is_valid_answer_container(c)
    ]
    if containers:
        return containers

    # Strategy 4: tightly-scoped q-box fallback for simplified/static HTML.
    containers = []
    for c in soup.select("div.q-box"):
        if not _is_valid_answer_container(c):
            continue
        text = c.get_text(" ", strip=True)
        if len(text) < _MIN_TEXT_LEN:
            continue
        if not c.select_one("time[datetime]") and not _looks_like_answer(c):
            continue
        containers.append(c)
    if containers:
        return containers

    # Strategy 5: broadest fallback -- any div with a large text block
    # that contains author credentials patterns
    fallback: list[Tag] = []
    for div in soup.find_all("div", recursive=True):
        text = div.get_text(strip=True)
        if len(text) >= _MIN_TEXT_LEN and _is_valid_answer_container(div) and _looks_like_answer(div):
            fallback.append(div)
    return fallback[:50]  # Cap to avoid processing the entire DOM


def _looks_like_answer(el: Tag) -> bool:
    """Heuristic: does this element look like a Quora answer block?"""
    # Must have some minimum depth (answers are nested, not top-level)
    if not el.parent or not el.parent.parent:
        return False
    # Should not be a navigation/header/footer element
    tag_classes = " ".join(el.get("class", []))
    if any(skip in tag_classes.lower() for skip in ("nav", "header", "footer", "sidebar", "menu")):
        return False
    # Must contain at least one <span> with text (Quora wraps answer text in spans)
    spans = el.find_all("span", limit=5)
    return any(len(s.get_text(strip=True)) > 30 for s in spans)


def _is_valid_answer_container(container: Tag) -> bool:
    """Reject promoted/navigation blocks before expensive text extraction."""
    classes = " ".join(container.get("class", [])).lower()
    if "dom_annotate_ad_promoted_answer" in classes:
        return False
    text = container.get_text(" ", strip=True)
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if not lowered:
        return False
    if any(hint in lowered[:200] for hint in _PROMO_HINTS):
        return False
    if lowered.startswith("all related"):
        return False
    return True


def _extract_question_title(soup: BeautifulSoup) -> str | None:
    """Extract the question title from a Quora question page."""
    # Question title selectors
    for selector in (
        'div.puppeteer_test_question_title',
        'div[class*="puppeteer_test_question_title"]',
        '#mainContent h1 span',
        'div[class*="question"] span[class*="q-text"]',
        'div[class*="QuestionHeader"] span',
        'h1 span',
        'title',
    ):
        el = soup.select_one(selector)
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 10:
                # Clean trailing " - Quora" from <title>
                if text.endswith(" - Quora"):
                    text = text[:-8].strip()
                return text[:500]
    return None


def _parse_answer_container(
    container: Tag,
    target: ScrapeTarget,
    seen_ids: set[str],
    *,
    question_title: str | None = None,
    fallback_url: str | None = None,
) -> dict | None:
    """Extract a review dict from a single Quora answer container."""
    # Extract answer text -- collect all spans/paragraphs within the container
    answer_text = _extract_answer_text(container)
    answer_text = _clean_answer_text(answer_text)
    if not answer_text or len(answer_text) < _MIN_TEXT_LEN:
        return None
    if _looks_like_promoted_text(answer_text):
        return None
    if _looks_like_question_list(answer_text):
        return None

    # Generate a stable review ID from the answer text content
    text_hash = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()[:16]
    review_id = f"quora-{text_hash}"

    if review_id in seen_ids:
        return None
    seen_ids.add(review_id)

    # Extract author info
    author_name, author_credentials = _extract_author_info(container)

    # Extract upvote count from the container
    upvotes = _extract_upvotes(container)

    # Build source URL -- best-effort from any link in the container
    source_url = _extract_answer_url(container) or fallback_url
    if not source_url or not _is_scrapable_quora_url(source_url):
        return None

    # Use question title as summary when available
    summary = question_title[:500] if question_title else None

    return {
        "source": "quora",
        "source_url": source_url,
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": None,
        "rating_max": 5,
        "summary": summary,
        "review_text": answer_text[:10000],
        "pros": None,
        "cons": None,
        "reviewer_name": author_name,
        "reviewer_title": author_credentials,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": _extract_answer_date(container),
        "raw_metadata": {
            "extraction_method": "html",
            "source_weight": 0.5,
            "source_type": "qa_platform",
            "upvotes": upvotes,
            "question_title": question_title,
        },
    }


def _extract_answer_text(container: Tag) -> str:
    """Extract the main answer text from a Quora answer container.

    Quora wraps answer content in deeply nested spans. We collect text
    from the largest contiguous text block within the container.
    """
    # Strategy 1: current browser-rendered answer-content wrappers
    for selector in (
        'div.puppeteer_test_answer_content',
        'div[class*="spacing_log_answer_content"]',
    ):
        el = container.select_one(selector)
        if el:
            text = _clean_answer_text(el.get_text(separator=" ", strip=True))
            if len(text) >= _MIN_TEXT_LEN:
                return text

    # Strategy 2: look for the main answer content span
    for selector in (
        'div[class*="AnswerContent"] span',
        'div[class*="answer_content"] span',
        'span[class*="q-text"]',
        'div.q-text',
    ):
        el = container.select_one(selector)
        if el:
            text = _clean_answer_text(el.get_text(separator=" ", strip=True))
            if len(text) >= _MIN_TEXT_LEN:
                return text

    # Strategy 3: collect all <p> and <span> text within the container
    parts: list[str] = []
    for tag in container.find_all(["p", "span"], recursive=True):
        text = _clean_answer_text(tag.get_text(strip=True))
        # Skip very short fragments (navigation, buttons, metadata)
        if len(text) > 20:
            # Avoid duplicates from nested elements
            if text not in parts:
                parts.append(text)

    if parts:
        # Take the longest contiguous block
        combined = " ".join(parts)
        if len(combined) >= _MIN_TEXT_LEN:
            return combined

    # Strategy 4: raw container text as last resort
    raw = _clean_answer_text(container.get_text(separator=" ", strip=True))
    return raw


def _clean_answer_text(value: str | None) -> str:
    """Normalize Quora answer text and remove obvious prompt chrome."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"\bContinue Reading\b", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\bUpvote\b(?:\s+\d+)?$", "", text, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_promoted_text(text: str) -> bool:
    """Skip sponsored/promoted copy that is not an actual answer."""
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    return any(hint in lowered[:160] for hint in _PROMO_HINTS)


def _looks_like_question_list(text: str) -> bool:
    """Skip related-question dumps masquerading as answer text."""
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if not lowered:
        return False
    question_marks = lowered.count("?")
    if question_marks >= 3:
        return True
    if question_marks >= 2 and lowered.startswith(
        ("what ", "how ", "which ", "who ", "is ", "are ", "can ")
    ):
        return True
    return False


def _extract_author_info(container: Tag) -> tuple[str | None, str | None]:
    """Extract author name and credentials from an answer container.

    Returns (author_name, credentials_text).
    """
    author_name: str | None = None
    credentials: str | None = None

    # Author name selectors
    for selector in (
        'div[class*="spacing_log_answer_header"] span[class*="q-text"]',
        'a[class*="user"] span',
        'span[class*="author"]',
        'a[class*="ProfileLink"]',
        'div[class*="AuthorInfo"] a',
        'span[class*="name"]',
    ):
        el = container.select_one(selector)
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 1 and len(text) < 100:
                author_name = text
                break

    # Credentials / bio line (e.g. "Software Engineer at Google (2015-present)")
    for selector in (
        'span[class*="credential"]',
        'div[class*="credential"]',
        'span[class*="UserCredential"]',
        'div[class*="AuthorInfo"] span:nth-of-type(2)',
    ):
        el = container.select_one(selector)
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 3 and text != author_name:
                credentials = text[:300]
                break

    return author_name, credentials


def _extract_answer_date(container: Tag) -> str | None:
    """Best-effort extraction of an answer timestamp from Quora HTML."""
    time_tag = container.select_one("time[datetime]")
    if time_tag:
        value = str(time_tag.get("datetime") or "").strip()
        if value:
            return value

    for selector in (
        'span[class*="date"]',
        'a[class*="date"]',
        'div[class*="date"]',
    ):
        el = container.select_one(selector)
        if not el:
            continue
        text = el.get_text(" ", strip=True)
        match = _DATE_TEXT_RE.search(text or "")
        if match:
            return match.group(0)
    return None


def _extract_upvotes(container: Tag) -> int | None:
    """Extract the upvote count from an answer container."""
    for selector in (
        'span[class*="upvote"]',
        'button[class*="upvote"] span',
        'div[class*="VoteButton"] span',
    ):
        el = container.select_one(selector)
        if el:
            text = el.get_text(strip=True).replace(",", "")
            # Handle "1.2K" style counts
            try:
                if text.upper().endswith("K"):
                    return int(float(text[:-1]) * 1000)
                if text.upper().endswith("M"):
                    return int(float(text[:-1]) * 1_000_000)
                return int(text)
            except (ValueError, TypeError):
                continue
    return None


def _extract_answer_url(container: Tag) -> str | None:
    """Try to extract a permalink to this specific answer."""
    # Look for a link that points to a Quora question/answer
    for a_tag in container.find_all("a", href=True, limit=10):
        href = a_tag["href"]
        url = _normalize_quora_url(href)
        if url and _is_scrapable_quora_url(url):
            return url
    return None


def _normalize_quora_url(value: str | None) -> str | None:
    """Normalize a Quora URL and strip query/fragment noise."""
    href = str(value or "").strip()
    if not href:
        return None
    if href.startswith("/"):
        href = urljoin(_BASE_URL, href)
    parsed = urlparse(href)
    if not parsed.scheme or not parsed.netloc:
        return None
    if not parsed.netloc.lower().endswith("quora.com"):
        return None
    normalized = parsed._replace(query="", fragment="")
    return urlunparse(normalized)


def _classify_quora_url(value: str | None) -> str:
    """Classify Quora URLs into scrapeable and junk page types."""
    url = _normalize_quora_url(value)
    if not url:
        return "other"
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or "/"
    if host != "www.quora.com":
        return "space_page"
    if path == "/" or not path:
        return "root_page"
    if path.startswith("/search"):
        return "search_page"
    if path.startswith("/profile/"):
        return "profile_page"
    if path.startswith("/topic/"):
        return "topic_page"
    if path.startswith("/unanswered/"):
        return "unanswered_page"
    if path.startswith("/qemail/"):
        return "tracking_page"
    if "/answer/" in path or "/answers/" in path:
        return "answer_page"
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) == 1:
        return "question_page"
    return "other"


def _is_scrapable_quora_url(value: str | None) -> bool:
    """Only question and answer URLs are safe Quora scrape targets."""
    return _classify_quora_url(value) in {"question_page", "answer_page"}


def _extract_question_urls_from_search(html: str) -> list[str]:
    """Extract concrete question/answer URLs from a Quora search page."""
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        url = _normalize_quora_url(a_tag.get("href"))
        if not url or not _is_scrapable_quora_url(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


# Auto-register
register_parser(QuoraParser())
