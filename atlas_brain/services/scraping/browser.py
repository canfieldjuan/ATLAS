"""
Playwright stealth browser for bypassing DataDome / Cloudflare WAFs.

Uses playwright-stealth to patch browser fingerprints (navigator.webdriver,
WebGL, Canvas, plugins, languages). Chromium only -- Firefox stealth patches
are less mature.

Singleton via get_stealth_browser(). Concurrency limited by asyncio.Semaphore
(default 1 -- Chromium uses ~200-400MB RAM per context).
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ...config import settings

logger = logging.getLogger("atlas.services.scraping.browser")


@dataclass
class BrowserScrapeResult:
    """Result of a stealth browser page scrape."""

    html: str
    status_code: int
    url: str
    cookies: dict[str, str] = field(default_factory=dict)


def _browser_cookie_domain(page_url: str, domain: str) -> str:
    """Return a cookie domain suitable for injecting solved challenge cookies."""
    host = (urlparse(page_url).hostname or domain or "").lstrip(".")
    return f".{host}" if host else ""


async def solve_browser_challenge(
    page,
    *,
    domain: str,
    html: str,
    status_code: int,
    proxy_url: str | None = None,
) -> bool:
    """Attempt to solve a browser-visible anti-bot challenge and inject cookies."""
    from .captcha import CaptchaType, detect_captcha, get_captcha_proxy, get_captcha_solver

    captcha_type = detect_captcha(html, status_code)
    if captcha_type in (CaptchaType.NONE, CaptchaType.CLOUDFLARE_BLOCK):
        return False
    solver = get_captcha_solver(domain)
    if not solver:
        logger.warning("No captcha solver configured for %s", domain)
        return False
    ua = await page.evaluate("navigator.userAgent")
    solve_proxy = proxy_url if captcha_type == CaptchaType.DATADOME else (get_captcha_proxy() or None)
    solution = await solver.solve(captcha_type, page.url, html, user_agent=ua, proxy_url=solve_proxy)
    cookie_domain = _browser_cookie_domain(page.url, domain)
    cookie_list = [
        {"name": name, "value": value, "domain": cookie_domain, "path": "/"}
        for name, value in (solution.cookies or {}).items()
        if cookie_domain and name != "cf_turnstile_response"
    ]
    if not cookie_list:
        logger.warning("Browser captcha solve for %s returned no injectible cookies", domain)
        return False
    await page.context.add_cookies(cookie_list)
    logger.info("Injected %d challenge cookies for %s", len(cookie_list), domain)
    return True


class StealthBrowser:
    """Chromium browser with stealth patches for anti-bot bypass."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._started = False
        cfg = settings.b2b_scrape
        self._headless = cfg.playwright_headless
        self._timeout_ms = cfg.playwright_timeout_ms
        self._semaphore = asyncio.Semaphore(cfg.playwright_max_concurrent)

    async def start(self) -> None:
        """Launch Chromium with stealth configuration."""
        if self._started:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        self._started = True
        logger.info("StealthBrowser started (headless=%s)", self._headless)

    async def stop(self) -> None:
        """Shut down browser and Playwright."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        self._started = False
        logger.info("StealthBrowser stopped")

    async def scrape_page(
        self,
        url: str,
        *,
        proxy_url: str | None = None,
        wait_for_selector: str | None = None,
        referer: str | None = None,
    ) -> BrowserScrapeResult:
        """Navigate to URL and return rendered HTML.

        Creates a new browser context per call so proxy is scoped per-request.
        Applies stealth patches before navigation.
        """
        if not self._started:
            await self.start()

        async with self._semaphore:
            return await self._do_scrape(url, proxy_url, wait_for_selector, referer)

    async def _do_scrape(
        self,
        url: str,
        proxy_url: str | None,
        wait_for_selector: str | None,
        referer: str | None,
    ) -> BrowserScrapeResult:
        """Internal scrape with context lifecycle."""
        from playwright_stealth import Stealth

        _stealth = Stealth()

        # Viewport jitter for fingerprint diversity
        vw = 1920 + random.randint(-20, 20)
        vh = 1080 + random.randint(-20, 20)

        ctx_kwargs: dict = {
            "viewport": {"width": vw, "height": vh},
            "locale": "en-US",
            "timezone_id": "America/Chicago",
            "color_scheme": "light",
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        }
        if proxy_url:
            ctx_kwargs["proxy"] = _parse_proxy_for_playwright(proxy_url)
            # Bright Data (and some other proxy providers) use self-signed certs
            ctx_kwargs["ignore_https_errors"] = True

        context = await self._browser.new_context(**ctx_kwargs)
        try:
            page = await context.new_page()
            await _stealth.apply_stealth_async(page)

            goto_kwargs: dict = {"timeout": self._timeout_ms, "wait_until": "domcontentloaded"}
            if referer:
                goto_kwargs["referer"] = referer

            resp = await page.goto(url, **goto_kwargs)
            status = resp.status if resp else 0

            html = await page.content()
            if await solve_browser_challenge(
                page,
                domain=urlparse(url).hostname or "",
                html=html,
                status_code=status,
                proxy_url=proxy_url,
            ):
                resp = await page.goto(url, **goto_kwargs)
                status = resp.status if resp else 0

            # Wait for review content if selector provided
            if wait_for_selector and status == 200:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=10000)
                except Exception:
                    logger.debug("Selector %s not found within timeout", wait_for_selector)

            # Simulate human behavior
            await self._simulate_human(page)

            html = await page.content()

            # Extract cookies
            cookies = {}
            for c in await context.cookies():
                cookies[c["name"]] = c["value"]

            return BrowserScrapeResult(
                html=html, status_code=status, url=page.url, cookies=cookies
            )
        finally:
            await context.close()

    async def _simulate_human(self, page) -> None:
        """Minimal human simulation: scroll and short pause."""
        try:
            # Random scroll
            scroll_y = random.randint(300, 800)
            await page.evaluate(f"window.scrollBy(0, {scroll_y})")
            await asyncio.sleep(random.uniform(0.5, 1.5))

            # Mouse move to random position
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.3, 0.8))
        except Exception:
            pass  # Non-fatal

def _parse_proxy_for_playwright(proxy_url: str) -> dict:
    """Parse a proxy URL into Playwright's proxy dict format.

    Playwright requires server/username/password as separate keys, unlike
    curl_cffi which accepts credentials embedded in the URL.
    """
    from urllib.parse import urlparse

    parsed = urlparse(proxy_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    proxy_dict: dict = {"server": f"{parsed.scheme}://{parsed.hostname}:{port}"}
    if parsed.username:
        proxy_dict["username"] = parsed.username
    if parsed.password:
        proxy_dict["password"] = parsed.password
    return proxy_dict


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: StealthBrowser | None = None


def get_stealth_browser() -> StealthBrowser:
    """Return the singleton StealthBrowser instance (lazy init)."""
    global _instance
    if _instance is None:
        _instance = StealthBrowser()
    return _instance


async def shutdown_stealth_browser() -> None:
    """Shut down the singleton browser if it was started."""
    global _instance
    if _instance is not None:
        await _instance.stop()
        _instance = None
