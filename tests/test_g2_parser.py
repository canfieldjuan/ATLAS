"""Tests for G2 review parser: browser-first + HTTP fallback, HTML parsing."""

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
    "asyncpg",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
):
    sys.modules.setdefault(_mod, MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(**overrides) -> MagicMock:
    """Build a mock ScrapeTarget."""
    defaults = {
        "id": "test-target-1",
        "source": "g2",
        "vendor_name": "TestVendor",
        "product_name": "TestProduct",
        "product_slug": "test-product",
        "product_category": "CRM",
        "max_pages": 3,
        "metadata": {},
    }
    defaults.update(overrides)
    target = MagicMock()
    for k, v in defaults.items():
        setattr(target, k, v)
    return target


def _make_client() -> AsyncMock:
    """Build a mock AntiDetectionClient."""
    client = AsyncMock()
    return client


def _g2_review_html(review_id: str, title: str, body: str, rating: str = "4") -> str:
    """Generate a minimal G2 review card HTML."""
    return f"""
    <div data-review-id="{review_id}">
        <div itemprop="ratingValue" content="{rating}"></div>
        <h3 itemprop="name">{title}</h3>
        <div itemprop="reviewBody">{body}</div>
        <span itemprop="author">Jane Doe</span>
        <span class="reviewer-title">CTO</span>
        <span class="reviewer-company">Acme Corp</span>
        <span class="company-size">51-200 employees</span>
        <span class="industry">Technology</span>
        <time datetime="2026-02-15">Feb 15, 2026</time>
        <h5>What do you like best?</h5>
        <p>Great product with excellent support</p>
        <h5>What do you dislike?</h5>
        <p>Sometimes slow during peak hours</p>
    </div>
    """


def _g2_page_html(*review_htmls: str) -> str:
    """Wrap review cards in a page-level HTML shell."""
    body = "\n".join(review_htmls)
    return f"<html><body>{body}</body></html>"


# ---------------------------------------------------------------------------
# Test _parse_page and _parse_review_card
# ---------------------------------------------------------------------------

class TestParseReviewCard:
    """Test HTML parsing of individual G2 review cards."""

    def test_parse_full_review(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = _g2_page_html(
            _g2_review_html("r123", "Excellent CRM", "This CRM has transformed our workflow completely and saved us hours.", "5")
        )
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 1
        r = reviews[0]
        assert r["source"] == "g2"
        assert r["source_review_id"] == "r123"
        assert r["vendor_name"] == "TestVendor"
        assert r["rating"] == 5.0
        assert r["summary"] == "Excellent CRM"
        assert "transformed our workflow" in r["review_text"]
        assert r["reviewer_name"] == "Jane Doe"
        assert r["reviewer_title"] == "CTO"
        assert r["reviewer_company"] == "Acme Corp"
        assert r["company_size_raw"] == "51-200 employees"
        assert r["reviewed_at"] == "2026-02-15"

    def test_parse_pros_cons(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = _g2_page_html(
            _g2_review_html("r124", "Good tool", "A decent product with some room for improvement in many areas.")
        )
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 1
        r = reviews[0]
        assert r["pros"] == "Great product with excellent support"
        assert r["cons"] == "Sometimes slow during peak hours"

    def test_dedup_seen_ids(self):
        """Duplicate review IDs should be skipped."""
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = _g2_page_html(
            _g2_review_html("dup-1", "Review A", "A lengthy review body that is definitely more than twenty characters"),
            _g2_review_html("dup-1", "Review B", "Another lengthy review body that is more than twenty characters long"),
        )
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 1
        assert reviews[0]["summary"] == "Review A"

    def test_skip_short_review_text(self):
        """Reviews with < 20 chars body are skipped."""
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        # Short body, no pros/cons headings
        html = """
        <html><body>
        <div data-review-id="short-1">
            <h3>Short</h3>
            <div itemprop="reviewBody">Too short</div>
        </div>
        </body></html>
        """
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 0

    def test_no_review_id_skipped(self):
        """Cards without data-review-id or id attr are skipped."""
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = """
        <html><body>
        <div itemprop="review">
            <div itemprop="reviewBody">This review has no ID and should be skipped entirely.</div>
        </div>
        </body></html>
        """
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 0

    def test_fallback_selector_itemprop(self):
        """Fallback selectors: [itemprop=review] when no data-review-id."""
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = """
        <html><body>
        <div itemprop="review" id="review-abc123">
            <div itemprop="ratingValue" content="3"></div>
            <div itemprop="reviewBody">This product is okay but could use some major improvements in usability.</div>
        </div>
        </body></html>
        """
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 1
        assert reviews[0]["source_review_id"] == "review-abc123"
        assert reviews[0]["rating"] == 3.0

    def test_multiple_reviews_on_page(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = _g2_page_html(
            _g2_review_html("r1", "Review One", "This is the first review with enough body text for parsing."),
            _g2_review_html("r2", "Review Two", "This is the second review with enough body text for parsing."),
            _g2_review_html("r3", "Review Three", "This is the third review with enough body text for parsing."),
        )
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 3
        ids = {r["source_review_id"] for r in reviews}
        assert ids == {"r1", "r2", "r3"}

    def test_source_url_format(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target(product_slug="my-product")
        html = _g2_page_html(
            _g2_review_html("rev99", "Good", "A really solid product that does everything I need it to do and more.")
        )
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert reviews[0]["source_url"] == "https://www.g2.com/products/my-product/reviews#rev99"

    def test_verified_user_name_is_normalized_and_industry_recovered(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = """
        <html><body>
        <div data-review-id="verified-1">
            <div itemprop="reviewBody">This product is reliable enough for a real review body and long enough to parse correctly.</div>
            <span itemprop="author">Verified User in Human ResourcesThis reviewer's identity has been verified by our review moderation team. They have asked not to show their name, job title, or picture.</span>
            <span class="company-size">Mid-Market (51-1000 emp.)</span>
        </div>
        </body></html>
        """
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 1
        assert reviews[0]["reviewer_name"] == "Verified User"
        assert reviews[0]["reviewer_industry"] == "Human Resources"

    def test_review_text_truncated_at_10000(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        long_body = "A" * 15000
        html = f"""
        <html><body>
        <div data-review-id="long-1">
            <div itemprop="reviewBody">{long_body}</div>
        </div>
        </body></html>
        """
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert len(reviews) == 1
        assert len(reviews[0]["review_text"]) == 10000

    def test_raw_metadata_always_present(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = _g2_page_html(
            _g2_review_html("r-meta", "Meta", "A review with enough text to pass the twenty char minimum check.")
        )
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        meta = reviews[0]["raw_metadata"]
        assert meta["extraction_method"] == "html"
        assert meta["source_weight"] == 1.0
        assert meta["source_type"] == "verified_review_platform"

    def test_empty_page(self):
        from atlas_brain.services.scraping.parsers.g2 import _parse_page
        target = _make_target()
        html = "<html><body><p>No reviews here</p></body></html>"
        seen: set[str] = set()
        reviews = _parse_page(html, target, seen)
        assert reviews == []


# ---------------------------------------------------------------------------
# Test G2Parser.scrape() routing (browser vs HTTP)
# ---------------------------------------------------------------------------

class TestG2ParserRouting:
    """Test browser-first / HTTP-fallback routing in G2Parser.scrape()."""

    @pytest.mark.asyncio
    async def test_browser_primary_when_enabled(self):
        """When playwright_enabled=True and browser returns reviews, use them."""
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target()
        client = _make_client()

        browser_result = MagicMock()
        browser_result.reviews = [{"source_review_id": "b1"}]
        browser_result.pages_scraped = 1
        browser_result.errors = []

        with (
            patch("atlas_brain.services.scraping.parsers.g2.settings") as mock_settings,
        ):
            mock_settings.b2b_scrape.playwright_enabled = True
            parser._scrape_browser = AsyncMock(return_value=browser_result)
            parser._scrape_http = AsyncMock()

            result = await parser.scrape(target, client)

        assert result.reviews == [{"source_review_id": "b1"}]
        parser._scrape_browser.assert_awaited_once_with(target)
        parser._scrape_http.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_to_http_when_browser_empty(self):
        """When browser returns 0 reviews, fall back to HTTP."""
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target()
        client = _make_client()

        browser_result = MagicMock()
        browser_result.reviews = []  # Empty
        browser_result.pages_scraped = 1
        browser_result.errors = []

        http_result = MagicMock()
        http_result.reviews = [{"source_review_id": "h1"}]

        with patch("atlas_brain.services.scraping.parsers.g2.settings") as mock_settings:
            mock_settings.b2b_scrape.playwright_enabled = True
            parser._scrape_browser = AsyncMock(return_value=browser_result)
            parser._scrape_http = AsyncMock(return_value=http_result)

            result = await parser.scrape(target, client)

        assert result.reviews == [{"source_review_id": "h1"}]
        parser._scrape_browser.assert_awaited_once()
        parser._scrape_http.assert_awaited_once_with(target, client)

    @pytest.mark.asyncio
    async def test_fallback_to_http_on_browser_exception(self):
        """When browser scrape raises, fall back to HTTP."""
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target()
        client = _make_client()

        http_result = MagicMock()
        http_result.reviews = [{"source_review_id": "h2"}]

        with patch("atlas_brain.services.scraping.parsers.g2.settings") as mock_settings:
            mock_settings.b2b_scrape.playwright_enabled = True
            parser._scrape_browser = AsyncMock(side_effect=RuntimeError("Browser crashed"))
            parser._scrape_http = AsyncMock(return_value=http_result)

            result = await parser.scrape(target, client)

        assert result.reviews == [{"source_review_id": "h2"}]

    @pytest.mark.asyncio
    async def test_http_only_when_playwright_disabled(self):
        """When playwright_enabled=False, skip browser entirely."""
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target()
        client = _make_client()

        http_result = MagicMock()
        http_result.reviews = [{"source_review_id": "h3"}]

        with patch("atlas_brain.services.scraping.parsers.g2.settings") as mock_settings:
            mock_settings.b2b_scrape.playwright_enabled = False
            parser._scrape_browser = AsyncMock()
            parser._scrape_http = AsyncMock(return_value=http_result)

            result = await parser.scrape(target, client)

        assert result.reviews == [{"source_review_id": "h3"}]
        parser._scrape_browser.assert_not_awaited()


# ---------------------------------------------------------------------------
# Test _scrape_browser internals
# ---------------------------------------------------------------------------

class TestG2BrowserScrape:
    """Test _scrape_browser page loop behavior."""

    @pytest.mark.asyncio
    async def test_stops_on_403(self):
        """403 from browser should stop pagination."""
        from atlas_brain.services.scraping.parsers.g2 import G2Parser
        from atlas_brain.services.scraping.browser import BrowserScrapeResult

        parser = G2Parser()
        target = _make_target(max_pages=3)

        mock_browser = AsyncMock()
        mock_browser.scrape_page = AsyncMock(
            return_value=BrowserScrapeResult(html="blocked", status_code=403, url="https://g2.com")
        )

        mock_proxy_mgr = MagicMock()
        mock_proxy_mgr.get_proxy = MagicMock(return_value=None)

        with (
            patch("atlas_brain.services.scraping.parsers.g2.settings") as mock_settings,
            patch("atlas_brain.services.scraping.browser.get_stealth_browser", return_value=mock_browser),
            patch("atlas_brain.services.scraping.proxy.ProxyManager.from_config", return_value=mock_proxy_mgr),
        ):
            mock_settings.b2b_scrape.playwright_enabled = True
            result = await parser._scrape_browser(target)

        assert result.pages_scraped == 1
        assert any("403" in e for e in result.errors)
        assert result.reviews == []

    @pytest.mark.asyncio
    async def test_stops_on_empty_page_1(self):
        """If page 1 returns no reviews, stop immediately."""
        from atlas_brain.services.scraping.parsers.g2 import G2Parser
        from atlas_brain.services.scraping.browser import BrowserScrapeResult

        parser = G2Parser()
        target = _make_target(max_pages=3)

        # Return valid HTML but no review cards
        mock_browser = AsyncMock()
        mock_browser.scrape_page = AsyncMock(
            return_value=BrowserScrapeResult(
                html="<html><body>No reviews</body></html>",
                status_code=200,
                url="https://g2.com",
            )
        )

        mock_proxy_mgr = MagicMock()
        mock_proxy_mgr.get_proxy = MagicMock(return_value=None)

        with (
            patch("atlas_brain.services.scraping.parsers.g2.settings") as mock_settings,
            patch("atlas_brain.services.scraping.browser.get_stealth_browser", return_value=mock_browser),
            patch("atlas_brain.services.scraping.proxy.ProxyManager.from_config", return_value=mock_proxy_mgr),
        ):
            mock_settings.b2b_scrape.playwright_enabled = True
            result = await parser._scrape_browser(target)

        assert result.pages_scraped == 1
        assert result.reviews == []

    @pytest.mark.asyncio
    async def test_browser_scrape_with_reviews(self):
        """Browser returns HTML with reviews, they get parsed correctly."""
        from atlas_brain.services.scraping.parsers.g2 import G2Parser
        from atlas_brain.services.scraping.browser import BrowserScrapeResult

        parser = G2Parser()
        target = _make_target(max_pages=1)

        review_html = _g2_page_html(
            _g2_review_html("br-1", "Browser Review", "This review was fetched via the stealth browser path successfully."),
        )

        mock_browser = AsyncMock()
        mock_browser.scrape_page = AsyncMock(
            return_value=BrowserScrapeResult(html=review_html, status_code=200, url="https://g2.com")
        )

        mock_proxy_mgr = MagicMock()
        mock_proxy_mgr.get_proxy = MagicMock(return_value=None)

        with (
            patch("atlas_brain.services.scraping.parsers.g2.settings") as mock_settings,
            patch("atlas_brain.services.scraping.browser.get_stealth_browser", return_value=mock_browser),
            patch("atlas_brain.services.scraping.proxy.ProxyManager.from_config", return_value=mock_proxy_mgr),
        ):
            mock_settings.b2b_scrape.playwright_enabled = True
            result = await parser._scrape_browser(target)

        assert len(result.reviews) == 1
        assert result.reviews[0]["source_review_id"] == "br-1"
        assert result.pages_scraped == 1


# ---------------------------------------------------------------------------
# Test _scrape_http internals
# ---------------------------------------------------------------------------

class TestG2HttpScrape:
    """Test _scrape_http page loop behavior."""

    @pytest.mark.asyncio
    async def test_http_403_stops(self):
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target(max_pages=3)
        client = _make_client()

        resp = MagicMock()
        resp.status_code = 403
        client.get = AsyncMock(return_value=resp)

        result = await parser._scrape_http(target, client)
        assert result.pages_scraped == 1
        assert any("403" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_http_non_html_content_type_stops(self):
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target(max_pages=3)
        client = _make_client()

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        client.get = AsyncMock(return_value=resp)

        result = await parser._scrape_http(target, client)
        assert result.pages_scraped == 1
        assert any("unexpected content-type" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_http_successful_single_page(self):
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target(max_pages=1)
        client = _make_client()

        review_html = _g2_page_html(
            _g2_review_html("http-1", "HTTP Review", "This review was fetched via the HTTP fallback path successfully."),
        )

        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.text = review_html
        resp.content = review_html.encode()
        client.get = AsyncMock(return_value=resp)

        result = await parser._scrape_http(target, client)
        assert len(result.reviews) == 1
        assert result.reviews[0]["source_review_id"] == "http-1"
        assert result.pages_scraped == 1

    @pytest.mark.asyncio
    async def test_http_exception_stops(self):
        from atlas_brain.services.scraping.parsers.g2 import G2Parser

        parser = G2Parser()
        target = _make_target(max_pages=3)
        client = _make_client()

        client.get = AsyncMock(side_effect=ConnectionError("Network down"))

        result = await parser._scrape_http(target, client)
        assert result.pages_scraped == 0
        assert any("Network down" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Test _extract_g2_section
# ---------------------------------------------------------------------------

class TestExtractG2Section:
    def test_heading_sibling_pattern(self):
        from atlas_brain.services.scraping.parsers.g2 import _extract_g2_section
        from bs4 import BeautifulSoup

        html = """
        <div>
            <h5>What do you like best about TestProduct?</h5>
            <p>The integration capabilities are outstanding and seamless.</p>
        </div>
        """
        card = BeautifulSoup(html, "html.parser").div
        result = _extract_g2_section(card, "like", "best")
        assert result == "The integration capabilities are outstanding and seamless."

    def test_no_match_returns_none(self):
        from atlas_brain.services.scraping.parsers.g2 import _extract_g2_section
        from bs4 import BeautifulSoup

        html = "<div><h5>Unrelated heading</h5><p>Some text here.</p></div>"
        card = BeautifulSoup(html, "html.parser").div
        result = _extract_g2_section(card, "like", "best")
        assert result is None

    def test_short_content_skipped(self):
        from atlas_brain.services.scraping.parsers.g2 import _extract_g2_section
        from bs4 import BeautifulSoup

        html = """
        <div>
            <h5>What do you like best?</h5>
            <p>OK</p>
        </div>
        """
        card = BeautifulSoup(html, "html.parser").div
        result = _extract_g2_section(card, "like", "best")
        assert result is None  # len("OK") <= 5

    def test_class_based_fallback(self):
        from atlas_brain.services.scraping.parsers.g2 import _extract_g2_section
        from bs4 import BeautifulSoup

        html = """
        <div>
            <div class="review-like-section">
                <p>Really love this product for daily workflows</p>
            </div>
        </div>
        """
        card = BeautifulSoup(html, "html.parser").div
        result = _extract_g2_section(card, "like")
        assert result == "Really love this product for daily workflows"

    def test_content_truncated_at_5000(self):
        from atlas_brain.services.scraping.parsers.g2 import _extract_g2_section
        from bs4 import BeautifulSoup

        long_text = "X" * 6000
        html = f"""
        <div>
            <h5>What do you like best?</h5>
            <p>{long_text}</p>
        </div>
        """
        card = BeautifulSoup(html, "html.parser").div
        result = _extract_g2_section(card, "like", "best")
        assert len(result) == 5000


class TestG2ParserAttributes:
    """Verify parser class attributes."""

    def test_source_name(self):
        from atlas_brain.services.scraping.parsers.g2 import G2Parser
        assert G2Parser.source_name == "g2"

    def test_prefer_residential(self):
        from atlas_brain.services.scraping.parsers.g2 import G2Parser
        assert G2Parser.prefer_residential is True
