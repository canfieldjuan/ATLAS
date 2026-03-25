from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.services.scraping.parsers import ScrapeTarget
from atlas_brain.services.scraping.parsers.slashdot import SlashdotParser, _parse_page


def _target(**overrides) -> ScrapeTarget:
    data = {
        "id": "slashdot-1",
        "source": "slashdot",
        "vendor_name": "Slack",
        "product_name": "Slack",
        "product_slug": "slack",
        "product_category": "Collaboration",
        "max_pages": 3,
        "metadata": {},
        "date_cutoff": None,
    }
    data.update(overrides)
    return ScrapeTarget(**data)


def _review_html(
    review_id: str,
    date_published: str,
    *,
    include_org_meta: bool = False,
) -> str:
    extra_meta = ""
    if include_org_meta:
        extra_meta = """
          <div><span class="value-label">Organization Size:</span><span class="value-value">201-500</span></div>
          <div><span class="value-label">Role:</span><span class="value-value">Decision Maker</span></div>
        """
    return f"""
    <html><body>
      <div class="m-review" itemprop="review" id="{review_id}">
        <div class="ext-review-title"><span class="review-title">Reliable collaboration upgrade</span></div>
        <meta itemprop="datePublished" content="{date_published}" />
        <div itemprop="reviewRating"><meta itemprop="ratingValue" content="4.0" /></div>
        <div class="ext-review-content">
          <p>Summary: The platform reduced coordination friction across support and engineering while keeping notifications manageable throughout weekly operations.</p>
          <p>Positive: Fast onboarding, stable channels, and clear controls for external collaboration with customers.</p>
          <p>Negative: Search and history exports can lag during larger migrations and require manual retries.</p>
        </div>
        <div class="ext-review-meta">
          <div><span class="value-label">Name:</span><span class="value-value">Alex Doe</span></div>
          <div><span class="value-label">Job Title:</span><span class="value-value">IT Manager</span></div>
          {extra_meta}
        </div>
      </div>
    </body></html>
    """


def test_parse_page_extracts_structured_review_fields() -> None:
    html = _review_html("rev-100", "2026-03-20", include_org_meta=True)
    reviews = _parse_page(html, _target(), set(), "https://slashdot.org/software/p/slack/")
    assert len(reviews) == 1
    review = reviews[0]
    assert review["source"] == "slashdot"
    assert review["source_review_id"] == "rev-100"
    assert review["rating"] == 4.0
    assert review["reviewed_at"] == "2026-03-20"
    assert review["reviewer_name"] == "Alex Doe"
    assert review["reviewer_title"] == "IT Manager"
    assert review["company_size_raw"] == "201-500"
    assert review["pros"]
    assert review["cons"]


@pytest.mark.asyncio
async def test_scrape_stops_when_page_is_older_than_cutoff() -> None:
    html = _review_html("old-1", "2025-12-01")
    resp = MagicMock(
        status_code=200,
        text=html,
        headers={"content-type": "text/html"},
    )
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    result = await SlashdotParser().scrape(_target(date_cutoff="2026-03-01"), client)
    assert result.stop_reason == "date_cutoff"
    assert result.pages_scraped == 1
    assert result.reviews == []


@pytest.mark.asyncio
async def test_scrape_handles_not_found_title() -> None:
    html = "<html><head><title>Page not found - Slashdot</title></head><body></body></html>"
    resp = MagicMock(
        status_code=200,
        text=html,
        headers={"content-type": "text/html"},
    )
    client = AsyncMock()
    client.get = AsyncMock(return_value=resp)
    result = await SlashdotParser().scrape(_target(), client)
    assert result.reviews == []
    assert result.pages_scraped == 1
    assert any("Product slug not found" in err for err in result.errors)


@pytest.mark.asyncio
async def test_scrape_retries_on_403_and_recovers() -> None:
    blocked = MagicMock(
        status_code=403,
        text="<html><body>blocked</body></html>",
        headers={"content-type": "text/html"},
    )
    ok = MagicMock(
        status_code=200,
        text=_review_html("retry-ok", "2026-03-22"),
        headers={"content-type": "text/html"},
    )
    client = AsyncMock()
    client.get = AsyncMock(side_effect=[blocked, ok])
    result = await SlashdotParser().scrape(_target(max_pages=1), client)
    assert len(result.reviews) == 1
    assert result.errors == []
    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_scrape_reports_attempt_count_when_retry_statuses_persist() -> None:
    blocked = MagicMock(
        status_code=403,
        text="<html><body>blocked</body></html>",
        headers={"content-type": "text/html"},
    )
    client = AsyncMock()
    client.get = AsyncMock(side_effect=[blocked, blocked, blocked])
    result = await SlashdotParser().scrape(_target(max_pages=1), client)
    assert result.reviews == []
    assert result.pages_scraped == 1
    assert any("HTTP 403 after 3 attempts" in err for err in result.errors)


@pytest.mark.asyncio
async def test_scrape_uses_browser_fallback_for_blocked_first_page() -> None:
    blocked = MagicMock(
        status_code=403,
        text="<html><body>blocked</body></html>",
        headers={"content-type": "text/html"},
    )
    client = AsyncMock()
    client.get = AsyncMock(side_effect=[blocked, blocked, blocked])

    fallback_html = _review_html("browser-ok", "2026-03-23")
    with patch(
        "atlas_brain.services.scraping.parsers.slashdot._fetch_page_via_scraping_browser",
        new=AsyncMock(return_value=(fallback_html, 200, None)),
    ):
        result = await SlashdotParser().scrape(_target(max_pages=1), client)

    assert len(result.reviews) == 1
    assert result.errors == []
