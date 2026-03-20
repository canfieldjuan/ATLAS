"""Tests for GetApp parser browser and fallback behavior."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.models",
    "asyncpg",
    "playwright", "playwright.async_api", "curl_cffi", "curl_cffi.requests",
):
    sys.modules.setdefault(_mod, MagicMock())


def _make_target(**overrides) -> MagicMock:
    defaults = {
        "id": "target-1",
        "source": "getapp",
        "vendor_name": "Acme",
        "product_name": "Acme",
        "product_slug": "project-management-software/a/acme",
        "product_category": "Project Management",
        "max_pages": 2,
        "metadata": {},
        "date_cutoff": None,
    }
    defaults.update(overrides)
    target = MagicMock()
    for key, value in defaults.items():
        setattr(target, key, value)
    return target


def _make_browser_html() -> str:
    return """
    <html><body>
    <script type="application/ld+json">
    {
      "@type": "SoftwareApplication",
      "name": "Acme",
      "review": [{
        "@id": "ga-1",
        "name": "Great fit",
        "reviewBody": "This product handles our workflow well and has enough detail for parsing.",
        "datePublished": "2026-03-01",
        "reviewRating": {"ratingValue": "5"},
        "author": {"name": "Jane Doe"}
      }]
    }
    </script>
    </body></html>
    """


class TestGetAppHelpers:
    def test_build_url_uses_stored_slug_shape(self):
        from atlas_brain.services.scraping.parsers.getapp import _build_url

        url = _build_url("project-management-software/a/acme", 2)
        assert url == (
            "https://www.getapp.com/software/"
            "project-management-software/a/acme/reviews/?page=2"
        )


class TestGetAppBrowser:
    @pytest.mark.asyncio
    async def test_scrape_browser_parses_reviews(self):
        from atlas_brain.services.scraping.parsers.getapp import GetAppParser

        parser = GetAppParser()
        target = _make_target(max_pages=1)
        html = _make_browser_html()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.content = AsyncMock(return_value=html)
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.contexts = []
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_pw = MagicMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_pw
        mock_cm.__aexit__.return_value = None

        with patch("playwright.async_api.async_playwright", return_value=mock_cm):
            result = await parser._scrape_browser(target, "wss://browser")

        assert result.pages_scraped == 1
        assert len(result.reviews) == 1
        assert result.reviews[0]["source_review_id"] == "ga-1"
        assert len(result.page_logs) == 1
        mock_page.goto.assert_awaited_once_with(
            "https://www.getapp.com/software/project-management-software/a/acme/reviews/",
            wait_until="domcontentloaded",
            timeout=30000,
        )

    @pytest.mark.asyncio
    async def test_scrape_browser_stops_fast_on_protected_resume_page(self):
        from atlas_brain.services.scraping.parsers.getapp import GetAppParser

        parser = GetAppParser()
        target = _make_target(max_pages=12)
        blocked_html = "<html>captcha or protection page found</html>"

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=502))
        mock_page.content = AsyncMock(return_value=blocked_html)
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.contexts = []
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_pw = MagicMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_pw
        mock_cm.__aexit__.return_value = None

        with patch("atlas_brain.config.settings") as mock_settings, \
             patch("playwright.async_api.async_playwright", return_value=mock_cm):
            mock_settings.b2b_scrape.playwright_timeout_ms = 30000
            mock_settings.b2b_scrape.getapp_protection_page_stop_threshold = 2
            result = await parser._scrape_browser(
                target,
                "wss://browser",
                start_page=9,
                seed_seen_ids={"ga-1"},
            )

        assert result.pages_scraped == 1
        assert result.resume_page == 9
        assert result.stop_reason == "blocked_or_throttled"
        assert result.errors == ["Page 9: HTTP 502 (gateway/proxy error page)"]

    @pytest.mark.asyncio
    async def test_scrape_browser_retries_after_solved_cloudflare_challenge(self):
        from atlas_brain.services.scraping.parsers.getapp import GetAppParser

        parser = GetAppParser()
        target = _make_target(max_pages=1)
        challenge_html = "<html>cloudflare verify you are human</html>"
        html = _make_browser_html()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(
            side_effect=[MagicMock(status=403), MagicMock(status=200)]
        )
        mock_page.content = AsyncMock(side_effect=[challenge_html, html])
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.contexts = []
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_pw = MagicMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_pw
        mock_cm.__aexit__.return_value = None

        with patch("atlas_brain.config.settings") as mock_settings, \
             patch("playwright.async_api.async_playwright", return_value=mock_cm), \
             patch("atlas_brain.services.scraping.browser.solve_browser_challenge", new=AsyncMock(return_value=True)) as solve:
            mock_settings.b2b_scrape.playwright_timeout_ms = 30000
            mock_settings.b2b_scrape.getapp_protection_page_stop_threshold = 2
            result = await parser._scrape_browser(target, "wss://browser")

        solve.assert_awaited_once()
        assert mock_page.goto.await_count == 2
        assert result.pages_scraped == 1
        assert len(result.reviews) == 1
        assert result.errors == []


class TestGetAppRouting:
    @pytest.mark.asyncio
    async def test_browser_errors_flow_into_http_fallback(self):
        from atlas_brain.services.scraping.parsers import ScrapeResult
        from atlas_brain.services.scraping.parsers.getapp import GetAppParser

        parser = GetAppParser()
        target = _make_target()
        client = AsyncMock()
        browser_result = ScrapeResult(reviews=[], pages_scraped=1, errors=["browser empty"])
        http_result = ScrapeResult(reviews=[], pages_scraped=1, errors=["http empty"])

        with patch("atlas_brain.config.settings") as mock_settings:
            mock_settings.b2b_scrape.web_unlocker_url = ""
            mock_settings.b2b_scrape.scraping_browser_ws_url = "wss://browser"
            mock_settings.b2b_scrape.scraping_browser_domains = "getapp.com,x.com"
            parser._scrape_browser = AsyncMock(return_value=browser_result)
            parser._scrape_http = AsyncMock(return_value=http_result)

            result = await parser.scrape(target, client)

        assert result.errors == ["browser empty", "http empty"]

    @pytest.mark.asyncio
    async def test_partial_unlocker_result_continues_in_browser(self):
        from atlas_brain.services.scraping.parsers import ScrapeResult
        from atlas_brain.services.scraping.parsers.getapp import GetAppParser

        parser = GetAppParser()
        target = _make_target(max_pages=12)
        client = AsyncMock()
        unlocker_result = ScrapeResult(
            reviews=[{"source_review_id": "ga-1", "review_text": "one"}],
            pages_scraped=8,
            errors=["Page 9: HTTP 502 (gateway/proxy error page)"],
            resume_page=9,
        )
        browser_result = ScrapeResult(
            reviews=[{"source_review_id": "ga-2", "review_text": "two"}],
            pages_scraped=2,
            errors=[],
        )

        with patch("atlas_brain.config.settings") as mock_settings:
            mock_settings.b2b_scrape.web_unlocker_url = "http://unlocker"
            mock_settings.b2b_scrape.web_unlocker_domains = "getapp.com"
            mock_settings.b2b_scrape.scraping_browser_ws_url = "wss://browser"
            mock_settings.b2b_scrape.scraping_browser_domains = "getapp.com,x.com"
            parser._scrape_web_unlocker = AsyncMock(return_value=unlocker_result)
            parser._scrape_browser = AsyncMock(return_value=browser_result)
            parser._scrape_http = AsyncMock()

            result = await parser.scrape(target, client)

        parser._scrape_browser.assert_awaited_once_with(
            target,
            "wss://browser",
            start_page=9,
            seed_seen_ids={"ga-1"},
        )
        parser._scrape_http.assert_not_called()
        assert [review["source_review_id"] for review in result.reviews] == ["ga-1", "ga-2"]
        assert result.pages_scraped == 10

    @pytest.mark.asyncio
    async def test_web_unlocker_stops_on_midstream_protection_page(self):
        from atlas_brain.services.scraping.parsers.getapp import GetAppParser

        parser = GetAppParser()
        target = _make_target(max_pages=4)
        review_html = _make_browser_html()

        page_one = MagicMock(status_code=200, text=review_html, content=review_html.encode())
        page_two = MagicMock(
            status_code=502,
            text="<html>captcha or protection page found</html>",
            content=b"blocked",
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client.get = AsyncMock(side_effect=[page_one, page_two])

        with patch("atlas_brain.config.settings") as mock_settings, \
             patch("httpx.AsyncClient", return_value=client), \
             patch("atlas_brain.services.scraping.parsers.getapp.asyncio.sleep", new=AsyncMock()):
            mock_settings.b2b_scrape.web_unlocker_url = "http://unlocker"
            result = await parser._scrape_web_unlocker(target)

        assert result.pages_scraped == 2
        assert len(result.reviews) == 1
        assert result.resume_page == 2
        assert "Page 2: HTTP 502" in result.errors[0]

    @pytest.mark.asyncio
    async def test_http_stops_after_repeated_protection_pages(self):
        from atlas_brain.services.scraping.parsers.getapp import GetAppParser

        parser = GetAppParser()
        target = _make_target(max_pages=5)
        blocked_page = MagicMock(
            status_code=502,
            text="<html>captcha or protection page found</html>",
            content=b"blocked",
            headers={"content-type": "text/html"},
        )
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[blocked_page, blocked_page, blocked_page])

        with patch("atlas_brain.config.settings") as mock_settings:
            mock_settings.b2b_scrape.getapp_protection_page_stop_threshold = 2
            result = await parser._scrape_http(target, client)

        assert result.pages_scraped == 2
        assert result.stop_reason == "blocked_or_throttled"
        assert result.errors == [
            "Page 1: HTTP 502 (gateway/proxy error page)",
            "Page 2: HTTP 502 (gateway/proxy error page)",
        ]
