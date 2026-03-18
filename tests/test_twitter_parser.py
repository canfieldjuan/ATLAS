"""Tests for Twitter/X parser browser and fallback behavior."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.models",
    "starlette", "starlette.requests", "asyncpg",
    "playwright", "playwright.async_api", "curl_cffi", "curl_cffi.requests",
):
    sys.modules.setdefault(_mod, MagicMock())


def _make_target(**overrides) -> MagicMock:
    defaults = {
        "id": "target-1",
        "source": "twitter",
        "vendor_name": "Acme",
        "product_name": "Acme",
        "product_slug": "Acme",
        "product_category": "Project Management",
        "max_pages": 3,
        "metadata": {},
    }
    defaults.update(overrides)
    target = MagicMock()
    for key, value in defaults.items():
        setattr(target, key, value)
    return target


def _tweet_html(*texts: str) -> str:
    blocks = "".join(
        f'<div data-testid="tweetText">{text}</div>'
        for text in texts
    )
    return f"<html><body>{blocks}</body></html>"


class TestTwitterFallback:
    @pytest.mark.asyncio
    async def test_http_fallback_requests_one_page_per_query(self):
        from atlas_brain.services.scraping.parsers.twitter import TwitterParser

        parser = TwitterParser()
        target = _make_target()
        client = AsyncMock()
        parser._build_queries = MagicMock(return_value=['"Acme" terrible'])

        text = "This vendor has become frustrating to use and we are actively evaluating alternatives now."
        resp = MagicMock()
        resp.status_code = 200
        resp.text = _tweet_html(text)
        client.get = AsyncMock(return_value=resp)

        with patch("atlas_brain.config.settings") as mock_settings:
            mock_settings.b2b_scrape.scraping_browser_ws_url = ""
            result = await parser.scrape(target, client)

        assert len(result.reviews) == 1
        assert result.reviews[0]["raw_metadata"]["extraction_method"] == "http_html"
        assert client.get.await_count == 1
        assert len(result.page_logs) == 1

    @pytest.mark.asyncio
    async def test_browser_errors_are_preserved_in_fallback_result(self):
        from atlas_brain.services.scraping.parsers import ScrapeResult
        from atlas_brain.services.scraping.parsers.twitter import TwitterParser

        parser = TwitterParser()
        target = _make_target()
        client = AsyncMock()
        parser._build_queries = MagicMock(return_value=['"Acme" terrible'])
        parser._scrape_browser = AsyncMock(
            return_value=ScrapeResult(reviews=[], pages_scraped=1, errors=["browser empty"])
        )

        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body></body></html>"
        client.get = AsyncMock(return_value=resp)

        with patch("atlas_brain.config.settings") as mock_settings:
            mock_settings.b2b_scrape.scraping_browser_ws_url = "wss://browser"
            mock_settings.b2b_scrape.scraping_browser_domains = "x.com"
            result = await parser.scrape(target, client)

        assert result.errors == ["browser empty"]


class TestTwitterBrowser:
    @pytest.mark.asyncio
    async def test_scrape_browser_respects_per_query_limit_per_query(self):
        from atlas_brain.services.scraping.parsers.twitter import TwitterParser

        parser = TwitterParser()
        target = _make_target(max_pages=2)
        parser._build_queries = MagicMock(return_value=['"Acme" terrible', '"Acme" switching from'])
        html_one = _tweet_html(
            "This vendor has become frustrating to use and our team is planning a switch this quarter.",
            "We are moving away from this product because support has been slow and billing is painful.",
        )
        html_two = _tweet_html(
            "Billing keeps getting worse and our operations team is evaluating a replacement right now.",
            "Support quality has fallen sharply and our leadership team is discussing alternatives this month.",
        )

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
        mock_page.wait_for_selector = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=None)
        mock_page.content = AsyncMock(side_effect=[html_one, html_two])
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
            result = await parser._scrape_browser(target, "wss://browser", 0, 0, False, 1)

        assert result.pages_scraped == 2
        assert len(result.reviews) == 2
        assert result.reviews[0]["source"] == "twitter"
        assert result.reviews[0]["raw_metadata"]["extraction_method"] == "scraping_browser"
        assert len(result.page_logs) == 2
