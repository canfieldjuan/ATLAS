"""Tests for Trustpilot parser pagination behavior."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.models",
    "starlette", "starlette.requests", "asyncpg",
):
    sys.modules.setdefault(_mod, MagicMock())


def _make_target(**overrides) -> MagicMock:
    defaults = {
        "id": "target-1",
        "source": "trustpilot",
        "vendor_name": "Jira",
        "product_name": "Jira",
        "product_slug": "jira.com",
        "product_category": "Project Management",
        "max_pages": 10,
        "metadata": {},
        "date_cutoff": None,
    }
    defaults.update(overrides)
    target = MagicMock()
    for key, value in defaults.items():
        setattr(target, key, value)
    return target


@pytest.mark.asyncio
async def test_http_stops_on_first_404_page():
    from atlas_brain.services.scraping.parsers.trustpilot import TrustpilotParser

    parser = TrustpilotParser()
    target = _make_target(max_pages=5)
    ok_page = MagicMock(
        status_code=200,
        text="""
        <html><body>
        <script type="application/ld+json">
        {"@type":"Organization","review":[{"@type":"Review","@id":"tp-1","reviewBody":"This review has enough detail to be parsed correctly by the Trustpilot test.","datePublished":"2026-03-01","reviewRating":{"ratingValue":"4"},"author":{"name":"Jane"}}]}
        </script>
        </body></html>
        """,
        content=b"""
        <html><body>
        <script type="application/ld+json">
        {"@type":"Organization","review":[{"@type":"Review","@id":"tp-1","reviewBody":"This review has enough detail to be parsed correctly by the Trustpilot test.","datePublished":"2026-03-01","reviewRating":{"ratingValue":"4"},"author":{"name":"Jane"}}]}
        </script>
        </body></html>
        """,
        headers={"content-type": "text/html"},
    )
    not_found = MagicMock(
        status_code=404,
        text="<html>Not Found</html>",
        content=b"<html>Not Found</html>",
        headers={"content-type": "text/html"},
    )
    client = AsyncMock()
    client.get = AsyncMock(side_effect=[ok_page, not_found, not_found])

    result = await parser._scrape_http(target, client)

    assert result.pages_scraped == 2
    assert result.errors == ["Page 2: HTTP 404"]
    assert result.page_logs[-1].status_code == 404
    assert result.page_logs[-1].stop_reason == "http_error"


def test_parse_json_ld_does_not_map_publisher_name_to_reviewer_company():
    from atlas_brain.services.scraping.parsers.trustpilot import _parse_json_ld

    target = _make_target(product_slug="jira.com")
    html = """
    <html><body>
    <script type="application/ld+json">
    {
      "@type": "Organization",
      "name": "Jira",
      "review": [{
        "@type": "Review",
        "@id": "tp-jsonld-1",
        "name": "Detailed writeup",
        "reviewBody": "This review has enough detail to be parsed correctly and exceeds the minimum length requirement.",
        "datePublished": "2026-03-01",
        "reviewRating": {"ratingValue": "4", "bestRating": "5"},
        "author": {"name": "Jane"},
        "publisher": {"name": "Publisher Org"}
      }]
    }
    </script>
    </body></html>
    """

    reviews = _parse_json_ld(html, target, set())

    assert len(reviews) == 1
    assert reviews[0]["reviewer_company"] is None
    assert reviews[0]["reviewer_name"] == "Jane"
    assert reviews[0]["raw_metadata"]["extraction_method"] == "json_ld"
    assert reviews[0]["raw_metadata"]["publisher_name"] == "Publisher Org"


def test_trustpilot_parser_version_bumped():
    from atlas_brain.services.scraping.parsers.trustpilot import TrustpilotParser

    assert TrustpilotParser.version == "trustpilot:2"
