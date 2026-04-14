"""Tests for Capterra raw-page audit helpers."""

import pytest

from atlas_brain.services.scraping.capterra_audit import (
    analyze_capterra_jsonld_fields,
    analyze_capterra_raw_html,
    analyze_capterra_review_cards,
    capture_capterra_page,
    fetch_capterra_page_web_unlocker,
)
from atlas_brain.services.scraping.parsers import ScrapeTarget


def test_analyze_capterra_jsonld_fields_detects_author_identity_metadata():
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

    report = analyze_capterra_jsonld_fields(html)

    assert report["review_object_count"] == 1
    assert report["author_job_title_count"] == 1
    assert report["author_company_count"] == 1


def test_analyze_capterra_review_cards_detects_live_profile_metadata():
    html = """
    <html><body>
      <div data-testid="review-card">
        <div>
          <div>
            <img data-testid="reviewer-profile-pic" src="x.jpg" />
          </div>
          <div>
            <span>Verified Reviewer</span>
            <span>Engineering Manager</span>
            <span>Computer Software, 51-200 employees</span>
            <span>Used the software for: 2+ years</span>
          </div>
        </div>
        <div>Overall Rating 4</div>
      </div>
    </body></html>
    """

    report = analyze_capterra_review_cards(html)

    assert report["review_card_count"] >= 1
    assert report["reviewer_title_count"] == 1
    assert report["reviewer_company_count"] == 0
    assert report["reviewer_industry_count"] == 1
    assert report["company_size_count"] == 1
    assert report["samples"][0]["reviewer_title"] == "Engineering Manager"


def test_analyze_capterra_raw_html_combines_jsonld_and_review_cards():
    html = """
    <html><body>
      <script type="application/ld+json">
      {"@type":"Review","author":{"name":"Sam","jobTitle":"Director","worksFor":{"name":"Acme"}}}
      </script>
      <div data-testid="review-card">
        <div>
          <div><img data-testid="reviewer-profile-pic" src="x.jpg" /></div>
          <div>
            <span>Verified Reviewer</span>
            <span>Director</span>
            <span>Financial Services, 201-500 employees</span>
          </div>
        </div>
        <div>Overall Rating 5</div>
      </div>
    </body></html>
    """

    report = analyze_capterra_raw_html(html)

    assert report["employer_fields_present"] is True
    assert report["title_fields_present"] is True


@pytest.mark.asyncio
async def test_fetch_capterra_page_web_unlocker_surfaces_timeout(monkeypatch):
    target = ScrapeTarget(
        id="target-1",
        source="capterra",
        vendor_name="ClickUp",
        product_name="ClickUp",
        product_slug="clickup",
        product_category="project-management",
        max_pages=1,
        metadata={},
        date_cutoff=None,
    )

    async def _fake_get_with_web_unlocker(url: str, *, headers: dict[str, str], domain: str):
        raise RuntimeError("Web Unlocker timed out for capterra.com after 45s")

    monkeypatch.setattr(
        "atlas_brain.services.scraping.capterra_audit.get_with_web_unlocker",
        _fake_get_with_web_unlocker,
    )

    with pytest.raises(RuntimeError, match="timed out for capterra.com"):
        await fetch_capterra_page_web_unlocker(target, page=1)


@pytest.mark.asyncio
async def test_capture_capterra_page_dispatches_browser(monkeypatch):
    target = ScrapeTarget(
        id="target-1",
        source="capterra",
        vendor_name="ClickUp",
        product_name="ClickUp",
        product_slug="clickup",
        product_category="project-management",
        max_pages=1,
        metadata={},
        date_cutoff=None,
    )

    async def _fake_fetch_capterra_page_browser(target: ScrapeTarget, *, page: int):
        return {
            "method": "browser",
            "url": "https://example.com",
            "status_code": 200,
            "final_url": "https://example.com",
            "headers": {},
            "body": "<html></html>",
        }

    monkeypatch.setattr(
        "atlas_brain.services.scraping.capterra_audit.fetch_capterra_page_browser",
        _fake_fetch_capterra_page_browser,
    )

    capture = await capture_capterra_page(target, page=1, method="browser")

    assert capture["method"] == "browser"
    assert capture["analysis"]["employer_fields_present"] is False
