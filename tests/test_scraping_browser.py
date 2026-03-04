"""Tests for Playwright stealth browser wrapper and proxy parsing."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pre-mock heavy deps
# ---------------------------------------------------------------------------
for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "starlette", "starlette.requests",
    "asyncpg",
    "curl_cffi", "curl_cffi.requests",
):
    sys.modules.setdefault(_mod, MagicMock())

# Mock playwright + playwright_stealth at module level
_mock_pw = MagicMock()
_mock_pw_stealth = MagicMock()
sys.modules.setdefault("playwright", _mock_pw)
sys.modules.setdefault("playwright.async_api", _mock_pw)
sys.modules.setdefault("playwright_stealth", _mock_pw_stealth)


# ---------------------------------------------------------------------------
# Test proxy parsing
# ---------------------------------------------------------------------------

class TestParseProxyForPlaywright:
    """Tests for _parse_proxy_for_playwright URL splitting."""

    def test_http_with_auth(self):
        from atlas_brain.services.scraping.browser import _parse_proxy_for_playwright
        result = _parse_proxy_for_playwright("http://user:pass@proxy.example.com:8080")
        assert result["server"] == "http://proxy.example.com:8080"
        assert result["username"] == "user"
        assert result["password"] == "pass"

    def test_https_with_auth(self):
        from atlas_brain.services.scraping.browser import _parse_proxy_for_playwright
        result = _parse_proxy_for_playwright("https://admin:secret@proxy.example.com:443")
        assert result["server"] == "https://proxy.example.com:443"
        assert result["username"] == "admin"
        assert result["password"] == "secret"

    def test_no_auth(self):
        from atlas_brain.services.scraping.browser import _parse_proxy_for_playwright
        result = _parse_proxy_for_playwright("http://proxy.example.com:3128")
        assert result["server"] == "http://proxy.example.com:3128"
        assert "username" not in result
        assert "password" not in result

    def test_default_http_port(self):
        from atlas_brain.services.scraping.browser import _parse_proxy_for_playwright
        result = _parse_proxy_for_playwright("http://proxy.example.com")
        assert result["server"] == "http://proxy.example.com:8080"

    def test_default_https_port(self):
        from atlas_brain.services.scraping.browser import _parse_proxy_for_playwright
        result = _parse_proxy_for_playwright("https://proxy.example.com")
        assert result["server"] == "https://proxy.example.com:443"

    def test_special_chars_in_password(self):
        """URL-encoded special chars in password are preserved by urlparse."""
        from atlas_brain.services.scraping.browser import _parse_proxy_for_playwright
        result = _parse_proxy_for_playwright("http://user:p%40ss%3Aword@proxy.example.com:8080")
        assert result["username"] == "user"
        # urlparse preserves percent-encoding in password
        assert result["password"] == "p%40ss%3Aword"

    def test_username_only_no_password(self):
        from atlas_brain.services.scraping.browser import _parse_proxy_for_playwright
        result = _parse_proxy_for_playwright("http://user@proxy.example.com:8080")
        assert result["username"] == "user"
        assert "password" not in result


# ---------------------------------------------------------------------------
# Test BrowserScrapeResult dataclass
# ---------------------------------------------------------------------------

class TestBrowserScrapeResult:
    def test_default_cookies(self):
        from atlas_brain.services.scraping.browser import BrowserScrapeResult
        result = BrowserScrapeResult(html="<html></html>", status_code=200, url="https://example.com")
        assert result.cookies == {}

    def test_with_cookies(self):
        from atlas_brain.services.scraping.browser import BrowserScrapeResult
        result = BrowserScrapeResult(
            html="<html></html>", status_code=200,
            url="https://example.com", cookies={"dd": "abc123"},
        )
        assert result.cookies == {"dd": "abc123"}


# ---------------------------------------------------------------------------
# Test singleton lifecycle
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_stealth_browser_returns_same_instance(self):
        from atlas_brain.services.scraping import browser as browser_mod
        browser_mod._instance = None  # Reset
        with patch.object(browser_mod, "settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            b1 = browser_mod.get_stealth_browser()
            b2 = browser_mod.get_stealth_browser()
            assert b1 is b2
            browser_mod._instance = None  # Cleanup

    @pytest.mark.asyncio
    async def test_shutdown_clears_singleton(self):
        from atlas_brain.services.scraping import browser as browser_mod
        mock_browser = MagicMock()
        mock_browser.stop = AsyncMock()
        browser_mod._instance = mock_browser

        await browser_mod.shutdown_stealth_browser()
        mock_browser.stop.assert_awaited_once()
        assert browser_mod._instance is None

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_none(self):
        from atlas_brain.services.scraping import browser as browser_mod
        browser_mod._instance = None
        await browser_mod.shutdown_stealth_browser()  # Should not raise
        assert browser_mod._instance is None


# ---------------------------------------------------------------------------
# Test StealthBrowser methods
# ---------------------------------------------------------------------------

class TestStealthBrowser:
    @pytest.mark.asyncio
    async def test_start_sets_started_flag(self):
        from atlas_brain.services.scraping.browser import StealthBrowser

        mock_pw_ctx = MagicMock()
        mock_browser = AsyncMock()
        mock_pw_ctx.chromium.launch = AsyncMock(return_value=mock_browser)

        mock_pw_start = AsyncMock(return_value=mock_pw_ctx)
        mock_async_pw = MagicMock(return_value=MagicMock(start=mock_pw_start))

        with (
            patch("atlas_brain.services.scraping.browser.settings") as mock_settings,
            patch("playwright.async_api.async_playwright", mock_async_pw),
        ):
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            assert browser._started is False
            await browser.start()
            assert browser._started is True

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Calling start() twice does not re-launch."""
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            browser._started = True
            browser._playwright = MagicMock()
            browser._browser = MagicMock()
            await browser.start()  # Should return immediately
            # _playwright.chromium.launch should NOT have been called
            browser._browser.close = AsyncMock()

    @pytest.mark.asyncio
    async def test_stop_clears_state(self):
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            browser._started = True
            browser._browser = AsyncMock()
            browser._browser.close = AsyncMock()
            browser._playwright = AsyncMock()
            browser._playwright.stop = AsyncMock()

            await browser.stop()
            assert browser._started is False
            assert browser._browser is None
            assert browser._playwright is None

    @pytest.mark.asyncio
    async def test_scrape_page_auto_starts(self):
        """scrape_page should auto-start if not already started."""
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            browser.start = AsyncMock()
            browser._started = False

            # Mock _do_scrape to return a result
            from atlas_brain.services.scraping.browser import BrowserScrapeResult
            mock_result = BrowserScrapeResult(html="<html/>", status_code=200, url="https://example.com")
            browser._do_scrape = AsyncMock(return_value=mock_result)

            result = await browser.scrape_page("https://example.com")
            browser.start.assert_awaited_once()
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_do_scrape_uses_proxy(self):
        """When proxy_url is provided, proxy dict is added to context kwargs."""
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()

            # Set up mocks for browser internals
            mock_page = AsyncMock()
            mock_page.url = "https://example.com"
            mock_page.content = AsyncMock(return_value="<html></html>")
            mock_page.evaluate = AsyncMock(return_value=None)
            mock_page.mouse = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_page.goto = AsyncMock(return_value=mock_resp)

            mock_context = AsyncMock()
            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_context.cookies = AsyncMock(return_value=[])
            mock_context.close = AsyncMock()

            mock_browser_obj = AsyncMock()
            mock_browser_obj.new_context = AsyncMock(return_value=mock_context)
            browser._browser = mock_browser_obj

            with patch("playwright_stealth.stealth_async", new_callable=AsyncMock):
                result = await browser._do_scrape(
                    "https://example.com",
                    proxy_url="http://user:pass@proxy.example.com:8080",
                    wait_for_selector=None,
                    referer=None,
                )

            # Verify proxy was passed to new_context
            ctx_call = mock_browser_obj.new_context.call_args
            proxy_arg = ctx_call.kwargs.get("proxy") or ctx_call[1].get("proxy")
            assert proxy_arg is not None
            assert proxy_arg["server"] == "http://proxy.example.com:8080"
            assert proxy_arg["username"] == "user"
            assert result.status_code == 200


# ---------------------------------------------------------------------------
# Test DataDome challenge detection
# ---------------------------------------------------------------------------

class TestDataDomeChallenge:
    @pytest.mark.asyncio
    async def test_no_datadome_returns_false(self):
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            page = AsyncMock()
            result = await browser._handle_datadome_challenge(page, "<html>Normal page</html>")
            assert result is False

    @pytest.mark.asyncio
    async def test_datadome_detected_no_solver(self):
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            page = AsyncMock()
            html = '<html><iframe src="https://captcha-delivery.com/challenge"></iframe></html>'

            with patch(
                "atlas_brain.services.scraping.captcha.get_captcha_solver",
                return_value=None,
            ):
                result = await browser._handle_datadome_challenge(page, html)
            assert result is False

    @pytest.mark.asyncio
    async def test_datadome_solved_injects_cookies(self):
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            page = AsyncMock()
            page.url = "https://www.g2.com/products/test/reviews"
            page.evaluate = AsyncMock(return_value="Mozilla/5.0 ...")
            page.context = AsyncMock()
            page.context.add_cookies = AsyncMock()

            html = '<html>captcha-delivery.com challenge</html>'

            mock_solution = MagicMock()
            mock_solution.cookies = {"datadome": "solved_token"}
            mock_solver = AsyncMock()
            mock_solver.solve = AsyncMock(return_value=mock_solution)

            mock_captcha_type = MagicMock()

            with (
                patch(
                    "atlas_brain.services.scraping.captcha.get_captcha_solver",
                    return_value=mock_solver,
                ),
                patch(
                    "atlas_brain.services.scraping.captcha.CaptchaType",
                    mock_captcha_type,
                ),
            ):
                result = await browser._handle_datadome_challenge(page, html)

            assert result is True
            page.context.add_cookies.assert_awaited_once()
            cookies_arg = page.context.add_cookies.call_args[0][0]
            assert any(c["name"] == "datadome" for c in cookies_arg)

    @pytest.mark.asyncio
    async def test_datadome_solver_exception_returns_false(self):
        from atlas_brain.services.scraping.browser import StealthBrowser

        with patch("atlas_brain.services.scraping.browser.settings") as mock_settings:
            cfg = MagicMock()
            cfg.playwright_headless = True
            cfg.playwright_timeout_ms = 30000
            cfg.playwright_max_concurrent = 1
            mock_settings.b2b_scrape = cfg

            browser = StealthBrowser()
            page = AsyncMock()
            page.url = "https://www.g2.com/products/test/reviews"
            page.evaluate = AsyncMock(return_value="Mozilla/5.0")

            html = '<html>captcha-delivery.com</html>'

            with patch(
                "atlas_brain.services.scraping.captcha.get_captcha_solver",
                side_effect=ImportError("captcha module not configured"),
            ):
                result = await browser._handle_datadome_challenge(page, html)

            assert result is False
