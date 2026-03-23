from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_github_issue_search_query_includes_date_cutoff():
    from atlas_brain.services.scraping.parsers.github import _issue_search_query

    query = _issue_search_query("HubSpot", "migration", "2026-03-01")

    assert "HubSpot" in query
    assert "label:migration" in query
    assert "created:>=2026-03-01" in query


def test_github_repo_search_query_includes_date_cutoff():
    from atlas_brain.services.scraping.parsers.github import _repo_search_query

    query = _repo_search_query("HubSpot", 10, "2026-03-01")

    assert "HubSpot" in query
    assert "stars:>=10" in query
    assert "created:>=2026-03-01" in query


def test_twitter_build_queries_append_since_operator():
    from atlas_brain.services.scraping.parsers import ScrapeTarget
    from atlas_brain.services.scraping.parsers.twitter import TwitterParser

    parser = TwitterParser()
    target = ScrapeTarget(
        id="target-1",
        source="twitter",
        vendor_name="HubSpot",
        product_name="HubSpot",
        product_slug="hubspot",
        product_category="CRM",
        max_pages=3,
        metadata={},
        date_cutoff="2026-03-01",
    )

    queries = parser._build_queries(target)

    assert queries
    assert all("since:2026-03-01" in query for query in queries)


def test_stackoverflow_cutoff_epoch_uses_utc_start_of_day():
    from atlas_brain.services.scraping.parsers.stackoverflow import _cutoff_epoch

    assert _cutoff_epoch("2026-03-01") == 1772323200


class _Resp:
    def __init__(self, text: str):
        self.status_code = 200
        self.text = text


@pytest.mark.asyncio
async def test_rss_parser_filters_entries_older_than_cutoff():
    from atlas_brain.services.scraping.parsers import ScrapeTarget
    from atlas_brain.services.scraping.parsers.rss import RSSParser

    xml = """<?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0">
      <channel>
        <title>Example Feed</title>
        <item>
          <title>Fresh HubSpot churn signal</title>
          <link>https://example.com/new</link>
          <guid>new-1</guid>
          <pubDate>Sat, 20 Mar 2026 12:00:00 GMT</pubDate>
          <description>%s</description>
        </item>
        <item>
          <title>Old HubSpot news</title>
          <link>https://example.com/old</link>
          <guid>old-1</guid>
          <pubDate>Mon, 20 Jan 2026 12:00:00 GMT</pubDate>
          <description>%s</description>
        </item>
      </channel>
    </rss>
    """ % ("A" * 240, "B" * 240)

    client = SimpleNamespace(get=AsyncMock(return_value=_Resp(xml)))
    parser = RSSParser()
    target = ScrapeTarget(
        id="target-1",
        source="rss",
        vendor_name="HubSpot",
        product_name="HubSpot",
        product_slug="https://example.com/feed.xml",
        product_category="CRM",
        max_pages=1,
        metadata={},
        date_cutoff="2026-03-01",
    )

    result = await parser.scrape(target, client)

    assert [review["source_review_id"] for review in result.reviews] == ["new-1"]
