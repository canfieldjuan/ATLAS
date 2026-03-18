"""Tests for scraper wiring outside parser-local behavior."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

for _mod in ("starlette", "starlette.requests", "starlette.exceptions", "asyncpg"):
    sys.modules.setdefault(_mod, MagicMock())


def test_rate_limiter_includes_x_domain_from_config():
    from atlas_brain.services.scraping.rate_limiter import DomainRateLimiter

    cfg = MagicMock()
    cfg.g2_rpm = 6
    cfg.capterra_rpm = 8
    cfg.trustradius_rpm = 10
    cfg.reddit_rpm = 30
    cfg.hackernews_rpm = 100
    cfg.github_rpm = 25
    cfg.rss_rpm = 10
    cfg.gartner_rpm = 4
    cfg.trustpilot_rpm = 6
    cfg.getapp_rpm = 8
    cfg.twitter_rpm = 10
    cfg.producthunt_rpm = 20
    cfg.youtube_rpm = 50
    cfg.quora_rpm = 4
    cfg.stackoverflow_rpm = 25
    cfg.peerspot_rpm = 4
    cfg.software_advice_rpm = 8

    limiter = DomainRateLimiter.from_config(cfg)
    assert limiter._rpm_map["x.com"] == 10


def test_capabilities_match_getapp_and_twitter_parser_paths():
    from atlas_brain.services.scraping.capabilities import AccessPattern, get_capability

    getapp = get_capability("getapp")
    twitter = get_capability("twitter")

    assert getapp is not None
    assert twitter is not None
    assert AccessPattern.js_rendered in getapp.access_patterns
    assert getapp.fallback_chain == ("web_unlocker", "js_rendered", "html_scrape")
    assert twitter.fallback_chain == ("js_rendered", "html_scrape")


@pytest.mark.asyncio
async def test_manual_scrape_log_persists_page_logs():
    from atlas_brain.api.b2b_scrape import _write_scrape_log
    from atlas_brain.services.scraping.parsers import log_page

    pool = AsyncMock()
    pool.fetchval = AsyncMock(return_value="run-123")
    parser = MagicMock(prefer_residential=True, version="getapp:test")
    page_logs = [log_page(1, "https://example.com/reviews", status_code=200)]

    with patch("atlas_brain.autonomous.tasks.b2b_scrape_intake._persist_page_logs", new=AsyncMock()) as persist:
        run_id = await _write_scrape_log(
            pool,
            "00000000-0000-0000-0000-000000000001",
            "getapp",
            "failed",
            0,
            0,
            1,
            ["blocked"],
            123,
            parser,
            page_logs=page_logs,
        )

    assert run_id == "run-123"
    persist.assert_awaited_once_with(pool, "run-123", page_logs)
