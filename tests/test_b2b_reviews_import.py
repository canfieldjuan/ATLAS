from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class _FakeInsertConn:
    def __init__(self, pool):
        self.pool = pool

    async def executemany(self, sql, rows):
        sql_text = str(sql)
        if "INSERT INTO b2b_review_vendor_mentions" in sql_text:
            self.pool.inserted_vendor_mentions = list(rows)
            return
        self.pool.inserted_rows = list(rows)


class _FakeTransaction:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return _FakeInsertConn(self.pool)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, fetch_rows=None):
        self.is_initialized = True
        self.inserted_rows = []
        self.inserted_vendor_mentions = []
        self.fetch_rows = list(fetch_rows or [])

    def transaction(self):
        return _FakeTransaction(self)

    async def fetch(self, _sql, *_args):
        return list(self.fetch_rows)

    async def fetchrow(self, _sql, _batch_id):
        return {"cnt": len(self.inserted_rows)}


@pytest.mark.asyncio
async def test_import_b2b_reviews_dedupes_same_request_semantic_duplicates(monkeypatch):
    from atlas_brain.api.b2b_reviews import B2BReviewInput, import_b2b_reviews

    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.b2b_reviews.get_db_pool", lambda: pool)
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews.resolve_vendor_name",
        AsyncMock(side_effect=lambda vendor: vendor),
    )
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews._load_existing_review_fingerprints",
        AsyncMock(return_value=(set(), set(), set())),
    )

    reviews = [
        B2BReviewInput(source="g2", vendor_name="HubSpot", review_text="A" * 120, source_review_id="rev-1"),
        B2BReviewInput(source="g2", vendor_name="HubSpot", review_text="A" * 120, source_review_id="rev-1"),
    ]

    result = await import_b2b_reviews(reviews)

    assert len(pool.inserted_rows) == 1
    assert result["imported"] == 1
    assert result["duplicates"] == 1


@pytest.mark.asyncio
async def test_import_b2b_reviews_skips_existing_semantic_identity(monkeypatch):
    from atlas_brain.api.b2b_reviews import B2BReviewInput, import_b2b_reviews
    from atlas_brain.autonomous.tasks.b2b_scrape_intake import _make_review_identity_key

    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.b2b_reviews.get_db_pool", lambda: pool)
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews.resolve_vendor_name",
        AsyncMock(side_effect=lambda vendor: vendor),
    )
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews._load_existing_review_fingerprints",
        AsyncMock(return_value=(
            set(),
            {_make_review_identity_key("g2", "HubSpot", "rev-legacy", "Alice", "2026-03-21T00:00:00+00:00")},
            set(),
        )),
    )

    reviews = [
        B2BReviewInput(
            source="g2",
            vendor_name="HubSpot",
            review_text="B" * 120,
            source_review_id="rev-legacy",
            reviewer_name="Alice",
            reviewed_at="2026-03-21T00:00:00Z",
        ),
    ]

    result = await import_b2b_reviews(reviews)

    assert pool.inserted_rows == []
    assert result["imported"] == 0
    assert result["duplicates"] == 1


@pytest.mark.asyncio
async def test_import_b2b_reviews_dedupes_same_text_with_different_ids(monkeypatch):
    from atlas_brain.api.b2b_reviews import B2BReviewInput, import_b2b_reviews

    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.b2b_reviews.get_db_pool", lambda: pool)
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews.resolve_vendor_name",
        AsyncMock(side_effect=lambda vendor: vendor),
    )
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews._load_existing_review_fingerprints",
        AsyncMock(return_value=(set(), set(), set())),
    )

    reviews = [
        B2BReviewInput(
            source="reddit",
            vendor_name="HubSpot",
            review_text="same review body " * 10,
            source_review_id="post-1",
        ),
        B2BReviewInput(
            source="reddit",
            vendor_name="HubSpot",
            review_text="same review body " * 10,
            source_review_id="post-2",
        ),
    ]

    result = await import_b2b_reviews(reviews)

    assert len(pool.inserted_rows) == 1
    assert result["imported"] == 1
    assert result["duplicates"] == 1


@pytest.mark.asyncio
async def test_import_b2b_reviews_canonicalizes_same_source_item_across_vendors(monkeypatch):
    from atlas_brain.api.b2b_reviews import B2BReviewInput, import_b2b_reviews

    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.b2b_reviews.get_db_pool", lambda: pool)
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews.resolve_vendor_name",
        AsyncMock(side_effect=lambda vendor: vendor),
    )
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews._load_existing_review_fingerprints",
        AsyncMock(return_value=(set(), set(), set())),
    )

    reviews = [
        B2BReviewInput(
            source="reddit",
            vendor_name="HubSpot",
            review_text="same review body " * 10,
            source_review_id="post-1",
        ),
        B2BReviewInput(
            source="reddit",
            vendor_name="Salesforce",
            review_text="same review body " * 10,
            source_review_id="post-1",
        ),
    ]

    result = await import_b2b_reviews(reviews)

    assert len(pool.inserted_rows) == 1
    assert len(pool.inserted_vendor_mentions) == 2
    assert result["imported"] == 1
    assert result["duplicates"] == 1


@pytest.mark.asyncio
async def test_import_b2b_reviews_marks_cross_source_duplicates(monkeypatch):
    from atlas_brain.api.b2b_reviews import B2BReviewInput, import_b2b_reviews

    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.b2b_reviews.get_db_pool", lambda: pool)
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews.resolve_vendor_name",
        AsyncMock(side_effect=lambda vendor: vendor),
    )
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews._load_existing_review_fingerprints",
        AsyncMock(return_value=(set(), set(), set())),
    )

    reviews = [
        B2BReviewInput(
            source="g2",
            vendor_name="HubSpot",
            review_text="same syndicated review " * 10,
            source_review_id="g2-1",
            reviewer_name="Alex P",
            reviewed_at="2026-03-20T00:00:00Z",
            rating=2.0,
        ),
        B2BReviewInput(
            source="capterra",
            vendor_name="HubSpot",
            review_text="same syndicated review " * 10,
            source_review_id="cap-1",
            reviewer_name="Alex P.",
            reviewed_at="2026-03-20T00:00:00Z",
            rating=2.0,
        ),
    ]

    result = await import_b2b_reviews(reviews)

    assert len(pool.inserted_rows) == 2
    assert result["imported"] == 2
    assert result["duplicates"] == 1
    assert pool.inserted_rows[0][28] == "pending"
    assert pool.inserted_rows[1][28] == "duplicate"
    assert pool.inserted_rows[1][25] == pool.inserted_rows[0][29]
    assert pool.inserted_rows[1][26] == "cross_source_exact_content"


@pytest.mark.asyncio
async def test_import_b2b_reviews_sanitizes_synthetic_reviewer_title(monkeypatch):
    from atlas_brain.api.b2b_reviews import B2BReviewInput, import_b2b_reviews

    pool = _FakePool()
    monkeypatch.setattr("atlas_brain.api.b2b_reviews.get_db_pool", lambda: pool)
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews.resolve_vendor_name",
        AsyncMock(side_effect=lambda vendor: vendor),
    )
    monkeypatch.setattr(
        "atlas_brain.api.b2b_reviews._load_existing_review_fingerprints",
        AsyncMock(return_value=(set(), set(), set())),
    )

    reviews = [
        B2BReviewInput(
            source="reddit",
            vendor_name="HubSpot",
            review_text="same review body " * 10,
            source_review_id="post-9",
            reviewer_title="Repeat Churn Signal (Score: 10.0)",
        ),
    ]

    result = await import_b2b_reviews(reviews)

    assert result["imported"] == 1
    assert len(pool.inserted_rows) == 1
    assert pool.inserted_rows[0][14] is None
