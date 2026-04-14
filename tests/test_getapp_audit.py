"""Tests for GetApp raw-page audit helpers."""

import pytest

from atlas_brain.services.scraping.getapp_audit import (
    analyze_getapp_jsonld_fields,
    analyze_getapp_raw_html,
    analyze_getapp_review_cards,
    capture_getapp_page,
    fetch_getapp_page_web_unlocker,
)
from atlas_brain.services.scraping.parsers import ScrapeTarget


def test_analyze_getapp_jsonld_fields_detects_author_identity_metadata():
    html = """
    <html><body>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Product",
        "review": [{
          "@type": "Review",
          "author": {
            "@type": "Person",
            "name": "Jane Reviewer",
            "jobTitle": "VP Operations",
            "worksFor": {"@type": "Organization", "name": "Acme Corp"}
          }
        }]
      }
      </script>
    </body></html>
    """

    report = analyze_getapp_jsonld_fields(html)

    assert report["review_object_count"] == 1
    assert report["author_job_title_count"] == 1
    assert report["author_company_count"] == 1


def test_analyze_getapp_review_cards_detects_live_profile_metadata():
    html = """
    <html><body>
      <div data-review-id="rev-1">
        <div class="reviewer-title">Engineering Manager</div>
        <div class="reviewer-company">Acme Corp</div>
        <div class="reviewer-industry">Computer Software</div>
        <div class="reviewer-size">51-200 employees</div>
      </div>
    </body></html>
    """

    report = analyze_getapp_review_cards(html)

    assert report["review_card_count"] >= 1
    assert report["reviewer_title_count"] == 1
    assert report["reviewer_company_count"] == 1
    assert report["reviewer_industry_count"] == 1
    assert report["company_size_count"] == 1


def test_analyze_getapp_raw_html_combines_jsonld_and_review_cards():
    html = """
    <html><body>
      <script type="application/ld+json">
      {"@type":"Review","author":{"name":"Sam","jobTitle":"Director","worksFor":{"name":"Acme"}}}
      </script>
      <div data-review-id="rev-1">
        <div class="reviewer-title">Director</div>
        <div class="reviewer-size">201-500 employees</div>
      </div>
    </body></html>
    """

    report = analyze_getapp_raw_html(html)

    assert report["employer_fields_present"] is True
    assert report["title_fields_present"] is True


@pytest.mark.asyncio
async def test_fetch_getapp_page_web_unlocker_surfaces_timeout(monkeypatch):
    target = ScrapeTarget(
        id="target-1",
        source="getapp",
        vendor_name="ClickUp",
        product_name="ClickUp",
        product_slug="project-management-software/a/clickup",
        product_category="project-management",
        max_pages=1,
        metadata={},
        date_cutoff=None,
    )

    async def _fake_get_with_web_unlocker(url: str, *, headers: dict[str, str], domain: str):
        raise RuntimeError("Web Unlocker timed out for getapp.com after 45s")

    monkeypatch.setattr(
        "atlas_brain.services.scraping.getapp_audit.get_with_web_unlocker",
        _fake_get_with_web_unlocker,
    )

    with pytest.raises(RuntimeError, match="timed out for getapp.com"):
        await fetch_getapp_page_web_unlocker(target, page=1)


@pytest.mark.asyncio
async def test_capture_getapp_page_dispatches_browser(monkeypatch):
    target = ScrapeTarget(
        id="target-1",
        source="getapp",
        vendor_name="ClickUp",
        product_name="ClickUp",
        product_slug="project-management-software/a/clickup",
        product_category="project-management",
        max_pages=1,
        metadata={},
        date_cutoff=None,
    )

    async def _fake_fetch_getapp_page_browser(target: ScrapeTarget, *, page: int):
        return {
            "method": "browser",
            "url": "https://example.com",
            "status_code": 200,
            "final_url": "https://example.com",
            "headers": {},
            "body": "<html></html>",
        }

    monkeypatch.setattr(
        "atlas_brain.services.scraping.getapp_audit.fetch_getapp_page_browser",
        _fake_fetch_getapp_page_browser,
    )

    capture = await capture_getapp_page(target, page=1, method="browser")

    assert capture["method"] == "browser"
    assert capture["analysis"]["employer_fields_present"] is False
