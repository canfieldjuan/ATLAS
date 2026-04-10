"""Tests for incremental scrape checkpoint wiring and cutoff behavior."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
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


def test_prepare_scrape_target_uses_last_scraped_at_for_date_sparse_sources():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _prepare_scrape_target

    cfg = SimpleNamespace(exhaustive_lookback_days=365, exhaustive_max_pages_default=40)
    target, _, _ = _prepare_scrape_target(
        _row(
            source="quora",
            last_scraped_at=datetime(2026, 3, 21, 6, 30, tzinfo=timezone.utc),
        ),
        cfg,
    )

    assert target.date_cutoff == "2026-03-21"
    assert target.max_pages == 3


def test_prepare_scrape_target_uses_last_scraped_at_for_twitter_and_caps_queries():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _prepare_scrape_target

    cfg = SimpleNamespace(exhaustive_lookback_days=365, exhaustive_max_pages_default=40)
    target, _, _ = _prepare_scrape_target(
        _row(
            source="twitter",
            last_scraped_at=datetime(2026, 3, 20, 8, 15, tzinfo=timezone.utc),
            max_pages=50,
        ),
        cfg,
    )

    assert target.date_cutoff == "2026-03-20"
    assert target.max_pages == 4


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


class _FakeInsertConn:
    def __init__(self, pool):
        self.pool = pool

    async def executemany(self, _sql, rows):
        self.pool.inserted_rows = list(rows)


class _FakeTransaction:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return _FakeInsertConn(self.pool)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeInsertPool:
    def __init__(self, fetch_rows=None):
        self.inserted_rows = []
        self.fetch_rows = list(fetch_rows or [])

    def transaction(self):
        return _FakeTransaction(self)

    async def fetch(self, _sql, *_args):
        return list(self.fetch_rows)

    async def fetchrow(self, _sql, _batch_id):
        return {
            "cnt": len(self.inserted_rows),
            "named_company_reviews": 0,
        }

    async def execute(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_insert_reviews_dedupes_duplicate_reviews_within_same_batch(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    reviews = [
        {
            "source": "g2",
            "vendor_name": "HubSpot",
            "source_review_id": "rev-1",
            "review_text": "A" * 120,
            "reviewed_at": "2026-03-20",
        },
        {
            "source": "g2",
            "vendor_name": "HubSpot",
            "source_review_id": "rev-1",
            "review_text": "A" * 120,
            "reviewed_at": "2026-03-20",
        },
    ]

    stats = await _insert_reviews(pool, reviews, "batch-1", parser_version="g2:1")

    assert len(pool.inserted_rows) == 1
    assert stats["inserted"] == 1
    assert stats["eligible_rows"] == 1
    assert stats["duplicate_or_existing"] == 1
    assert stats["duplicate_same_batch"] == 1
    assert stats["duplicate_existing"] == 0
    assert stats["duplicate_db_conflict"] == 0


@pytest.mark.asyncio
async def test_insert_reviews_dedupes_by_text_hash_when_ids_differ(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    reviews = [
        {
            "source": "reddit",
            "vendor_name": "HubSpot",
            "source_review_id": "post-1",
            "review_text": "same body text " * 12,
            "reviewed_at": "2026-03-20",
        },
        {
            "source": "reddit",
            "vendor_name": "HubSpot",
            "source_review_id": "post-2",
            "review_text": "same body text " * 12,
            "reviewed_at": "2026-03-21",
        },
    ]

    stats = await _insert_reviews(pool, reviews, "batch-hash", parser_version="reddit:1")

    assert len(pool.inserted_rows) == 1
    assert stats["inserted"] == 1
    assert stats["duplicate_or_existing"] == 1
    assert stats["duplicate_same_batch"] == 1
    assert stats["skipped_quality_gate"] == 0


@pytest.mark.asyncio
async def test_insert_reviews_marks_cross_source_duplicate_rows(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    canonical_id = "11111111-1111-1111-1111-111111111111"
    pool = _FakeInsertPool(fetch_rows=[
        {
            "id": canonical_id,
            "source": "g2",
            "source_review_id": "g2-existing",
            "reviewer_name": "Alex P",
            "reviewed_at": "2026-03-20T00:00:00+00:00",
            "rating": 2.0,
            "imported_at": "2026-03-20T01:00:00+00:00",
            "enrichment_status": "enriched",
            "source_weight": 1.0,
            "cross_source_content_hash": None,
            "cross_source_identity_key": "hubspot|alexp|2026-03-20|2.0",
            "summary": None,
            "review_text": "same syndicated body " * 12,
            "pros": None,
            "cons": None,
        }
    ])
    reviews = [
        {
            "source": "capterra",
            "vendor_name": "HubSpot",
            "source_review_id": "cap-1",
            "review_text": "same syndicated body " * 12,
            "reviewed_at": "2026-03-20",
            "reviewer_name": "Alex P.",
            "rating": 2.0,
        },
    ]

    stats = await _insert_reviews(pool, reviews, "batch-cross", parser_version="mixed:1")

    assert len(pool.inserted_rows) == 1
    assert stats["inserted"] == 1
    assert stats["cross_source_duplicates"] == 1
    assert pool.inserted_rows[0][43] == "duplicate"
    assert str(pool.inserted_rows[0][40]) == canonical_id
    assert pool.inserted_rows[0][41] in {
        "cross_source_exact_content",
        "cross_source_identity_similarity",
    }


@pytest.mark.asyncio
async def test_insert_reviews_retains_capterra_aggregate_pages_in_raw_only_lane(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    reviews = [
        {
            "source": "capterra",
            "vendor_name": "HubSpot",
            "source_review_id": "agg-1",
            "review_text": "aggregate row body " * 10,
            "reviewed_at": "2026-03-20",
            "raw_metadata": {"extraction_method": "jsonld_aggregate"},
        }
    ]

    stats = await _insert_reviews(pool, reviews, "batch-agg", parser_version="capterra:1")

    assert len(pool.inserted_rows) == 1
    assert stats["inserted"] == 1
    assert stats["quality_gate_flagged"] == 1
    assert stats["retained_raw_only"] == 1
    assert pool.inserted_rows[0][43] == "raw_only"


@pytest.mark.asyncio
async def test_insert_reviews_retains_short_reviews_in_raw_only_lane(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    reviews = [
        {
            "source": "reddit",
            "vendor_name": "HubSpot",
            "source_review_id": "post-short",
            "review_text": "too short",
            "reviewed_at": "2026-03-20",
        }
    ]

    stats = await _insert_reviews(pool, reviews, "batch-short", parser_version="reddit:1")

    assert len(pool.inserted_rows) == 1
    assert stats["inserted"] == 1
    assert stats["short_flagged"] == 1
    assert stats["retained_raw_only"] == 1
    assert pool.inserted_rows[0][43] == "raw_only"


@pytest.mark.asyncio
async def test_insert_reviews_skips_legacy_existing_identity_even_when_hash_differs(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import (
        _insert_reviews,
        _make_review_identity_key,
    )

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    reviews = [
        {
            "source": "g2",
            "vendor_name": "HubSpot",
            "source_review_id": "rev-legacy",
            "review_text": "B" * 140,
            "reviewer_name": "Alice",
            "reviewed_at": "2026-03-21T00:00:00Z",
        },
    ]

    stats = await _insert_reviews(
        pool,
        reviews,
        "batch-2",
        parser_version="g2:1",
        known_keys=set(),
        known_identities={
            _make_review_identity_key(
                "g2",
                "HubSpot",
                "rev-legacy",
                "Alice",
                "2026-03-21T00:00:00+00:00",
            )
        },
    )

    assert pool.inserted_rows == []
    assert stats["inserted"] == 0
    assert stats["eligible_rows"] == 0
    assert stats["duplicate_or_existing"] == 1
    assert stats["duplicate_same_batch"] == 0
    assert stats["duplicate_existing"] == 1
    assert stats["duplicate_db_conflict"] == 0


@pytest.mark.asyncio
async def test_insert_reviews_repairs_missing_fields_on_existing_duplicate(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import (
        _PROMOTE_RAW_ONLY_DUPLICATE_SQL,
        _REPAIR_PARSER_FIELDS_SQL,
        _insert_reviews,
        _make_dedup_key,
    )

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    pool.execute = AsyncMock(side_effect=["UPDATE 1", "UPDATE 0"])
    dedup_key = _make_dedup_key("capterra", "Trello", "rev-1", "Namratha C.", "2026-02-23")
    reviews = [
        {
            "source": "capterra",
            "vendor_name": "Trello",
            "source_review_id": "rev-1",
            "reviewer_name": "Namratha C.",
            "review_text": "A" * 140,
            "reviewed_at": "2026-02-23",
            "reviewer_title": "Product owner",
            "company_size_raw": "1,001 - 5,000 employees",
            "reviewer_industry": "Insurance",
        },
    ]

    stats = await _insert_reviews(
        pool,
        reviews,
        "batch-3",
        parser_version="capterra:2",
        known_keys={dedup_key},
        known_identities=set(),
    )

    assert pool.inserted_rows == []
    assert stats["inserted"] == 0
    assert stats["eligible_rows"] == 0
    assert stats["duplicate_existing"] == 1
    assert stats["repaired_existing"] == 1
    assert stats["promoted_existing"] == 0
    assert pool.execute.await_count == 2
    assert pool.execute.await_args_list[0].args[0] == _REPAIR_PARSER_FIELDS_SQL
    assert pool.execute.await_args_list[0].args[1:] == (
        dedup_key,
        "Product owner",
        None,
        None,
        "1,001 - 5,000 employees",
        "Insurance",
        "capterra:2",
    )
    assert pool.execute.await_args_list[1].args[0] == _PROMOTE_RAW_ONLY_DUPLICATE_SQL


@pytest.mark.asyncio
async def test_insert_reviews_promotes_existing_raw_only_duplicate_with_fuller_payload(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import (
        _PROMOTE_RAW_ONLY_DUPLICATE_SQL,
        _insert_reviews,
        _make_dedup_key,
    )

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    pool.execute = AsyncMock(return_value="UPDATE 1")
    dedup_key = _make_dedup_key("reddit", "HubSpot", "post-1", "Alex", "2026-03-20")
    reviews = [
        {
            "source": "reddit",
            "vendor_name": "HubSpot",
            "source_review_id": "post-1",
            "reviewer_name": "Alex",
            "review_text": "A" * 160,
            "pros": "B" * 40,
            "cons": "C" * 30,
            "reviewed_at": "2026-03-20",
            "raw_metadata": {"source_weight": 0.4},
        },
    ]

    stats = await _insert_reviews(
        pool,
        reviews,
        "batch-promote",
        parser_version="reddit:2",
        known_keys={dedup_key},
        known_identities=set(),
    )

    assert pool.inserted_rows == []
    assert stats["inserted"] == 0
    assert stats["duplicate_existing"] == 1
    assert stats["promoted_existing"] == 1
    pool.execute.assert_awaited_once()
    assert pool.execute.await_args.args[0] == _PROMOTE_RAW_ONLY_DUPLICATE_SQL
    assert pool.execute.await_args.args[1:4] == ("HubSpot", "reddit", dedup_key)


@pytest.mark.asyncio
async def test_insert_reviews_sanitizes_synthetic_reviewer_title(monkeypatch):
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _insert_reviews

    async def _resolve(vendor_name):
        return vendor_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake.resolve_vendor_name",
        _resolve,
    )

    pool = _FakeInsertPool()
    reviews = [
        {
            "source": "reddit",
            "vendor_name": "HubSpot",
            "source_review_id": "post-3",
            "review_text": "A" * 160,
            "reviewed_at": "2026-03-25",
            "reviewer_title": "Repeat Churn Signal (Score: 10.0)",
        },
    ]

    stats = await _insert_reviews(pool, reviews, "batch-sanitize", parser_version="reddit:1")

    assert stats["inserted"] == 1
    assert len(pool.inserted_rows) == 1
    assert pool.inserted_rows[0][14] is None


def test_repair_parser_fields_sql_casts_nullable_params():
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _REPAIR_PARSER_FIELDS_SQL

    for token in ("$2::text", "$3::text", "$4::text", "$5::text", "$6::text", "$7::text"):
        assert token in _REPAIR_PARSER_FIELDS_SQL
