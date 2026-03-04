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

from ...config import settings

logger = logging.getLogger("atlas.services.scraping.browser")


@dataclass
class BrowserScrapeResult:
    """Result of a stealth browser page scrape."""

    html: str
    status_code: int
    url: str
    cookies: dict[str, str] = field(default_factory=dict)


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

            # Check for DataDome challenge and attempt solve
            if status == 403:
                html = await page.content()
                if await self._handle_datadome_challenge(page, html):
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

    async def _handle_datadome_challenge(self, page, html: str) -> bool:
        """Detect and attempt to solve DataDome challenge via captcha solver.

        Returns True if challenge was solved and cookies injected.
        """
        if "captcha-delivery.com" not in html.lower():
            return False

        logger.info("DataDome challenge detected, attempting solver")

        try:
            from .captcha import CaptchaType, get_captcha_solver

            solver = get_captcha_solver("g2.com")
            if not solver:
                logger.warning("No captcha solver configured for g2.com")
                return False

            # Get the UA from the page's browser context
            ua = await page.evaluate("navigator.userAgent")
            solution = await solver.solve(
                CaptchaType.DATADOME, page.url, html, user_agent=ua
            )
            if not solution or not solution.cookies:
                logger.warning("Captcha solver returned no cookies")
                return False

            # Inject solution cookies into browser context
            cookie_list = []
            for name, value in solution.cookies.items():
                cookie_list.append({
                    "name": name,
                    "value": value,
                    "domain": ".g2.com",
                    "path": "/",
                })
            await page.context.add_cookies(cookie_list)
            logger.info("DataDome cookies injected, reloading")
            return True

        except Exception:
            logger.warning("DataDome challenge solve failed", exc_info=True)
            return False


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
