from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


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


class _FakePool:
    def __init__(self):
        self.is_initialized = True
        self.inserted_rows = []

    def transaction(self):
        return _FakeTransaction(self)

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
        "atlas_brain.api.b2b_reviews._load_existing_review_identity_sets",
        AsyncMock(return_value=(set(), set())),
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
        "atlas_brain.api.b2b_reviews._load_existing_review_identity_sets",
        AsyncMock(return_value=(
            set(),
            {_make_review_identity_key("g2", "HubSpot", "rev-legacy", "Alice", "2026-03-21T00:00:00+00:00")},
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
