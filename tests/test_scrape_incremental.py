"""Tests for incremental scrape checkpoint wiring and cutoff behavior."""

from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from unittest.mock import AsyncMock

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


def _row(**overrides):
    row = {
        "id": "target-1",
        "source": "reddit",
        "vendor_name": "HubSpot",
        "product_name": "HubSpot CRM",
        "product_slug": "hubspot",
        "product_category": "CRM",
        "max_pages": 5,
        "metadata": {},
        "scrape_mode": "incremental",
    }
    row.update(overrides)
    return row


def test_prepare_scrape_target_maps_exhaustive_to_initial_runtime_mode():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _prepare_scrape_target

    cfg = SimpleNamespace(exhaustive_lookback_days=365, exhaustive_max_pages_default=40)
    target, scrape_mode, metadata = _prepare_scrape_target(
        _row(scrape_mode="exhaustive", max_pages=5),
        cfg,
    )

    assert scrape_mode == "exhaustive"
    assert target.metadata["scrape_mode"] == "initial"
    assert "scrape_mode" not in metadata
    assert target.date_cutoff is not None
    assert target.max_pages == 40


def test_prepare_scrape_target_uses_incremental_checkpoint_for_cutoff():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _prepare_scrape_target

    cfg = SimpleNamespace(exhaustive_lookback_days=365, exhaustive_max_pages_default=40)
    target, scrape_mode, metadata = _prepare_scrape_target(
        _row(
            metadata={
                "scrape_state": {
                    "newest_review": "2026-03-10",
                }
            }
        ),
        cfg,
    )

    assert scrape_mode == "incremental"
    assert target.metadata["scrape_mode"] == "incremental"
    assert "scrape_mode" not in metadata
    assert target.date_cutoff == "2026-03-10"


def test_prepare_scrape_target_prefers_explicit_checkpoint_columns():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _prepare_scrape_target

    cfg = SimpleNamespace(exhaustive_lookback_days=365, exhaustive_max_pages_default=40)
    target, _, _ = _prepare_scrape_target(
        _row(
            metadata={"scrape_state": {"newest_review": "2026-03-05"}},
            last_scrape_newest_review=date(2026, 3, 19),
        ),
        cfg,
    )

    assert target.date_cutoff == "2026-03-19"


def test_prepare_scrape_target_ignores_legacy_metadata_scrape_mode_override():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _prepare_scrape_target

    cfg = SimpleNamespace(exhaustive_lookback_days=365, exhaustive_max_pages_default=40)
    target, scrape_mode, metadata = _prepare_scrape_target(
        _row(metadata={"scrape_mode": "initial"}),
        cfg,
    )

    assert scrape_mode == "incremental"
    assert target.metadata["scrape_mode"] == "incremental"
    assert "scrape_mode" not in metadata


def test_build_scrape_state_preserves_previous_boundaries_on_empty_run():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _build_scrape_state
    from atlas_brain.services.scraping.parsers import ScrapeResult, ScrapeTarget

    target = ScrapeTarget(
        id="target-1",
        source="reddit",
        vendor_name="HubSpot",
        product_name="HubSpot CRM",
        product_slug="hubspot",
        product_category="CRM",
        max_pages=5,
        metadata={"scrape_mode": "incremental"},
        date_cutoff="2026-03-10",
    )
    result = ScrapeResult(reviews=[], pages_scraped=1, errors=[])
    state = _build_scrape_state(
        {
            "scrape_state": {
                "oldest_review": "2026-03-01",
                "newest_review": "2026-03-10",
            }
        },
        target,
        "incremental",
        result,
        inserted=0,
        filtered_count=0,
        date_dropped=0,
        duration_ms=125,
    )

    assert state["oldest_review"] == "2026-03-01"
    assert state["newest_review"] == "2026-03-10"
    assert state["date_cutoff_used"] == "2026-03-10"
    assert state["runtime_mode"] == "incremental"


@pytest.mark.asyncio
async def test_update_target_after_scrape_writes_explicit_checkpoint_columns():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _update_target_after_scrape

    pool = AsyncMock()
    await _update_target_after_scrape(
        pool,
        "target-1",
        "success",
        1,
        metadata={"scrape_mode": "incremental"},
        scrape_state={
            "runtime_mode": "incremental",
            "stop_reason": "page_cap",
            "oldest_review": "2026-02-22",
            "newest_review": "2026-03-19",
            "date_cutoff_used": "2026-03-19",
            "pages_scraped": 29,
            "reviews_found": 66,
            "reviews_filtered": 56,
            "date_dropped": 0,
            "duration_ms": 23491,
            "resume_page": 12,
        },
    )

    sql_args = pool.execute.await_args.args
    assert "last_scrape_runtime_mode" in sql_args[0]
    assert "\"scrape_state\"" not in sql_args[4]
    assert sql_args[5] == "incremental"
    assert sql_args[6] == "page_cap"
    assert str(sql_args[7]) == "2026-02-22"
    assert str(sql_args[8]) == "2026-03-19"
    assert str(sql_args[9]) == "2026-03-19"
    assert sql_args[10] == 29
    assert sql_args[11] == 66
    assert sql_args[12] == 56
    assert sql_args[13] == 0
    assert sql_args[14] == 23491
    assert sql_args[15] == 12


class _Resp:
    def __init__(self, html: str):
        self.status_code = 200
        self.headers = {"content-type": "text/html"}
        self.text = html


def _sourceforge_html(review_id: str, reviewed_at: str) -> str:
    body = (
        "This is a long enough review body for parser testing and cutoff checks. "
        "It should stay above the minimum text threshold for SourceForge reviews."
    )
    return f"""
    <html><body>
    <div data-review-id="{review_id}">
        <div itemprop="reviewBody">{body}</div>
        <time datetime="{reviewed_at}">{reviewed_at}</time>
    </div>
    </body></html>
    """


@pytest.mark.asyncio
async def test_sourceforge_parser_stops_at_date_cutoff():
    from atlas_brain.services.scraping.parsers import ScrapeTarget
    from atlas_brain.services.scraping.parsers.sourceforge import SourceForgeParser

    parser = SourceForgeParser()
    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            _Resp(_sourceforge_html("new-1", "2026-03-10")),
            _Resp(_sourceforge_html("old-1", "2026-02-01")),
            _Resp(_sourceforge_html("old-2", "2026-01-01")),
        ]
    )

    target = ScrapeTarget(
        id="target-1",
        source="sourceforge",
        vendor_name="Datadog",
        product_name="Datadog",
        product_slug="datadog",
        product_category="Cloud Infrastructure",
        max_pages=3,
        metadata={},
        date_cutoff="2026-03-01",
    )

    result = await parser.scrape(target, client)

    assert result.stop_reason == "date_cutoff"
    assert result.pages_scraped == 2
    assert [review["source_review_id"] for review in result.reviews] == ["new-1"]
