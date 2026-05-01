"""Tests for atlas_brain.services.b2b.evidence_gate (Gap 3 shadow audit).

Pure-function tests (rank floor mapping, edge cases) plus integration
tests that exercise the SQL against a real Postgres with seeded
b2b_evidence_claims rows.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from atlas_brain.services.b2b.evidence_gate import (
    _rank_floor,
    audit_witness_evidence_coverage,
)


def test_rank_floor_strong_maps_to_zero():
    assert _rank_floor("strong") == 0


def test_rank_floor_weak_maps_to_one():
    assert _rank_floor("weak") == 1


def test_rank_floor_unknown_falls_back_to_strong():
    assert _rank_floor("unknown") == 0
    assert _rank_floor("") == 0


def test_rank_floor_case_insensitive():
    assert _rank_floor("STRONG") == 0
    assert _rank_floor("Weak") == 1


@pytest.mark.asyncio
async def test_audit_returns_zero_for_empty_review_ids(db_pool):
    result = await audit_witness_evidence_coverage(
        db_pool,
        vendor_name="acme",
        source_review_ids=[],
    )
    assert result["total_review_ids"] == 0
    assert result["covered_count"] == 0
    assert result["uncovered_count"] == 0
    assert result["coverage_ratio"] == 1.0


@pytest.mark.asyncio
async def test_audit_returns_zero_for_empty_vendor(db_pool):
    result = await audit_witness_evidence_coverage(
        db_pool,
        vendor_name="",
        source_review_ids=[uuid4()],
    )
    assert result["total_review_ids"] == 1
    assert result["covered_count"] == 0
    assert result["uncovered_count"] == 1


@pytest.mark.asyncio
async def test_audit_dedups_review_ids():
    # Pure-function test; doesn't need pool dispatch.
    class _NullPool:
        async def fetch(self, *_args, **_kwargs):
            return []

    rid = uuid4()
    result = await audit_witness_evidence_coverage(
        _NullPool(),
        vendor_name="acme",
        source_review_ids=[rid, rid, str(rid)],
    )
    assert result["total_review_ids"] == 1


@pytest.mark.asyncio
async def test_audit_skips_invalid_uuid_strings():
    class _NullPool:
        async def fetch(self, *_args, **_kwargs):
            return []

    result = await audit_witness_evidence_coverage(
        _NullPool(),
        vendor_name="acme",
        source_review_ids=["not-a-uuid", None, ""],
    )
    assert result["total_review_ids"] == 0


@pytest_asyncio.fixture
async def cleanup_claims(db_pool):
    inserted_review_ids: list[UUID] = []
    yield inserted_review_ids
    if inserted_review_ids:
        await db_pool.execute(
            "DELETE FROM b2b_evidence_claims WHERE source_review_id = ANY($1::uuid[])",
            inserted_review_ids,
        )


async def _seed_claim(
    pool,
    *,
    vendor_name: str,
    source_review_id: UUID,
    pain_confidence: str = "strong",
    status: str = "valid",
) -> None:
    await pool.execute(
        """
        INSERT INTO b2b_evidence_claims (
            artifact_type, artifact_id, vendor_name,
            claim_type, target_entity, status,
            source_review_id, pain_confidence,
            source_excerpt_fingerprint
        ) VALUES (
            'synthesis', gen_random_uuid(), $1,
            'pain_claim_about_vendor', $1, $2,
            $3, $4,
            $5
        )
        """,
        vendor_name,
        status,
        source_review_id,
        pain_confidence,
        f"fp-{source_review_id}",
    )


@pytest.mark.asyncio
async def test_audit_counts_strong_backed_review_ids(db_pool, cleanup_claims):
    vendor = f"vendor-{uuid4().hex[:8]}"
    strong_rid = uuid4()
    weak_rid = uuid4()
    unbacked_rid = uuid4()

    await _seed_claim(db_pool, vendor_name=vendor, source_review_id=strong_rid, pain_confidence="strong")
    await _seed_claim(db_pool, vendor_name=vendor, source_review_id=weak_rid, pain_confidence="weak")
    cleanup_claims.extend([strong_rid, weak_rid])

    result = await audit_witness_evidence_coverage(
        db_pool,
        vendor_name=vendor,
        source_review_ids=[strong_rid, weak_rid, unbacked_rid],
        min_pain_confidence="strong",
    )

    assert result["total_review_ids"] == 3
    assert str(strong_rid) in result["covered_review_ids"]
    assert str(weak_rid) not in result["covered_review_ids"]
    assert str(unbacked_rid) not in result["covered_review_ids"]
    assert result["covered_count"] == 1
    assert result["uncovered_count"] == 2
    assert result["coverage_ratio"] == round(1 / 3, 3)


@pytest.mark.asyncio
async def test_audit_loosened_to_weak_includes_weak_claims(db_pool, cleanup_claims):
    vendor = f"vendor-{uuid4().hex[:8]}"
    strong_rid = uuid4()
    weak_rid = uuid4()
    unbacked_rid = uuid4()

    await _seed_claim(db_pool, vendor_name=vendor, source_review_id=strong_rid, pain_confidence="strong")
    await _seed_claim(db_pool, vendor_name=vendor, source_review_id=weak_rid, pain_confidence="weak")
    cleanup_claims.extend([strong_rid, weak_rid])

    result = await audit_witness_evidence_coverage(
        db_pool,
        vendor_name=vendor,
        source_review_ids=[strong_rid, weak_rid, unbacked_rid],
        min_pain_confidence="weak",
    )

    assert result["covered_count"] == 2
    assert str(strong_rid) in result["covered_review_ids"]
    assert str(weak_rid) in result["covered_review_ids"]
    assert str(unbacked_rid) not in result["covered_review_ids"]


@pytest.mark.asyncio
async def test_audit_ignores_invalid_status_rows(db_pool, cleanup_claims):
    vendor = f"vendor-{uuid4().hex[:8]}"
    rid = uuid4()

    await _seed_claim(
        db_pool,
        vendor_name=vendor,
        source_review_id=rid,
        pain_confidence="strong",
        status="rejected",
    )
    cleanup_claims.append(rid)

    result = await audit_witness_evidence_coverage(
        db_pool,
        vendor_name=vendor,
        source_review_ids=[rid],
        min_pain_confidence="strong",
    )

    assert result["covered_count"] == 0
    assert result["uncovered_count"] == 1
