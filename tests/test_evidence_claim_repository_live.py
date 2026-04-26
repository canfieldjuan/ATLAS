"""Live integration tests for the EvidenceClaim repository.

Hits a real Postgres against migration 305_b2b_evidence_claims.sql. No
mocks. Each test scopes its writes to a unique artifact_id so cleanup is
safe and tests can run concurrently.

Run:
    python -m pytest tests/test_evidence_claim_repository_live.py -v -s --tb=short

Requires:
    - Postgres running (ATLAS_DB_* env vars)
    - Migration 305_b2b_evidence_claims.sql applied
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

from atlas_brain.services.b2b.evidence_claim import (
    ClaimType,
    ClaimValidationStatus,
    source_excerpt_fingerprint as _compute_fingerprint,
)
from atlas_brain.services.b2b.evidence_claim_repository import (
    ClaimSelection,
    PersistedClaim,
    dedup_selections,
    select_best_claim,
    upsert_claim,
)


@pytest.fixture
async def pool():
    from atlas_brain.storage.config import db_settings

    p = await asyncpg.create_pool(
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.database,
        user=db_settings.user,
        password=db_settings.password,
        min_size=1,
        max_size=3,
    )
    yield p
    await p.close()


@pytest.fixture
async def artifact_id(pool) -> UUID:
    """Yield a unique artifact_id and clean up any rows referencing it
    before and after the test."""
    aid = uuid4()
    await pool.execute(
        "DELETE FROM b2b_evidence_claims WHERE artifact_id = $1", aid
    )
    yield aid
    await pool.execute(
        "DELETE FROM b2b_evidence_claims WHERE artifact_id = $1", aid
    )


def _make_claim(
    *,
    artifact_id: UUID,
    vendor: str = "TestVendor",
    claim_type: ClaimType = ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
    witness_id: str | None = "witness:test:0",
    secondary_target: str | None = None,
    status: ClaimValidationStatus = ClaimValidationStatus.VALID,
    rejection_reason: str | None = None,
    salience_score: float = 5.0,
    grounding_status: str | None = "grounded",
    pain_confidence: str | None = "strong",
    excerpt_text: str | None = "test excerpt",
    source_review_id: UUID | None = None,
    payload: dict | None = None,
    target_entity: str | None = None,
) -> PersistedClaim:
    """Build a PersistedClaim with excerpt_text + source_review_id and let
    upsert_claim() compute the fingerprint at write time. This proves the
    production write path -- not the test helper -- owns the computation."""
    review_id = source_review_id or uuid4()
    return PersistedClaim(
        artifact_type="synthesis",
        artifact_id=artifact_id,
        vendor_name=vendor,
        claim_type=claim_type,
        target_entity=target_entity or vendor,
        secondary_target=secondary_target,
        status=status,
        rejection_reason=rejection_reason,
        synthesis_id=artifact_id,
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        witness_id=witness_id,
        source_review_id=review_id,
        salience_score=salience_score,
        grounding_status=grounding_status,
        pain_confidence=pain_confidence,
        excerpt_text=excerpt_text,
        supporting_fields=("phrase_subject", "phrase_polarity"),
        claim_payload=payload or {"excerpt_text": excerpt_text},
    )


@pytest.mark.asyncio
async def test_upsert_inserts_then_replay_updates(pool, artifact_id):
    """First call inserts; second call with same replay key updates
    validated_at and the mutable columns, but NOT created_at."""
    claim_v1 = _make_claim(
        artifact_id=artifact_id, salience_score=3.0, payload={"v": 1}
    )
    await upsert_claim(pool, claim_v1)

    row1 = await pool.fetchrow(
        "SELECT id, created_at, validated_at, salience_score, claim_payload "
        "FROM b2b_evidence_claims WHERE artifact_id = $1",
        artifact_id,
    )
    assert row1 is not None
    first_id = row1["id"]
    first_created_at = row1["created_at"]
    first_validated_at = row1["validated_at"]
    assert float(row1["salience_score"]) == 3.0

    # Replay with same replay key but mutated salience and payload.
    claim_v2 = _make_claim(
        artifact_id=artifact_id, salience_score=7.5, payload={"v": 2}
    )
    await upsert_claim(pool, claim_v2)

    rows = await pool.fetch(
        "SELECT id, created_at, validated_at, salience_score, claim_payload "
        "FROM b2b_evidence_claims WHERE artifact_id = $1",
        artifact_id,
    )
    assert len(rows) == 1, "replay must produce 1 row, not 2"
    assert rows[0]["id"] == first_id, "id must be preserved on replay"
    assert rows[0]["created_at"] == first_created_at, "created_at preserved"
    assert rows[0]["validated_at"] >= first_validated_at, "validated_at advances"
    assert float(rows[0]["salience_score"]) == 7.5
    payload = rows[0]["claim_payload"]
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    assert payload == {"v": 2}


@pytest.mark.asyncio
async def test_upsert_with_null_secondary_target_idempotent(pool, artifact_id):
    """COALESCE in the unique index must collapse two NULLs to a single
    row on replay (default Postgres NULL semantics treat NULLs as
    distinct in unique indexes)."""
    c1 = _make_claim(artifact_id=artifact_id, secondary_target=None, salience_score=4.0)
    c2 = _make_claim(artifact_id=artifact_id, secondary_target=None, salience_score=8.0)
    await upsert_claim(pool, c1)
    await upsert_claim(pool, c2)

    count = await pool.fetchval(
        "SELECT count(*) FROM b2b_evidence_claims WHERE artifact_id = $1",
        artifact_id,
    )
    assert count == 1, f"two NULL secondary_target replays produced {count} rows"


@pytest.mark.asyncio
async def test_upsert_distinguishes_distinct_secondary_targets(pool, artifact_id):
    """Same replay key except for secondary_target = different rows."""
    base = dict(
        artifact_id=artifact_id,
        claim_type=ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
    )
    await upsert_claim(pool, _make_claim(**base, secondary_target="HubSpot"))
    await upsert_claim(pool, _make_claim(**base, secondary_target="Salesforce"))
    await upsert_claim(pool, _make_claim(**base, secondary_target=None))

    count = await pool.fetchval(
        "SELECT count(*) FROM b2b_evidence_claims WHERE artifact_id = $1",
        artifact_id,
    )
    assert count == 3, f"three distinct secondary_targets should produce 3 rows, got {count}"


@pytest.mark.asyncio
async def test_select_best_claim_filters_status_valid(pool, artifact_id):
    """Invalid and cannot_validate rows must not surface in selection."""
    valid = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:valid",
        excerpt_text="valid pain quote",
        salience_score=6.0,
    )
    invalid = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:invalid",
        excerpt_text="positive quote misused",
        status=ClaimValidationStatus.INVALID,
        rejection_reason="polarity_not_negative_or_mixed",
        salience_score=9.0,  # higher salience but should not surface
    )
    cannot = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:cannot",
        excerpt_text="v3 backed",
        status=ClaimValidationStatus.CANNOT_VALIDATE,
        rejection_reason="phrase_subject_unavailable",
        salience_score=8.0,
    )
    await upsert_claim(pool, valid)
    await upsert_claim(pool, invalid)
    await upsert_claim(pool, cannot)

    results = await select_best_claim(
        pool,
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        target_entity="TestVendor",
        vendor_name="TestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        limit=10,
    )
    assert len(results) == 1
    assert results[0].witness_id == "witness:test:valid"


@pytest.mark.asyncio
async def test_select_best_claim_ranks_correctly(pool, artifact_id):
    """Order: salience DESC, grounding_rank ASC, pain_confidence_rank ASC,
    witness_id ASC. Build a small set with deliberate ranking conflicts
    and confirm the order matches the spec."""
    # Higher salience wins regardless of grounding/confidence.
    high_salience = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:a_high_salience",
        excerpt_text="rank: high salience",
        salience_score=9.0,
        grounding_status=None,  # not grounded -> rank 1
        pain_confidence="weak",  # rank 1
    )
    # Same salience as below, but grounded -> wins on tie-break #2.
    grounded = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:b_grounded",
        excerpt_text="rank: grounded",
        salience_score=5.0,
        grounding_status="grounded",  # rank 0
        pain_confidence="weak",  # rank 1
    )
    not_grounded = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:c_not_grounded",
        excerpt_text="rank: not grounded",
        salience_score=5.0,
        grounding_status=None,  # rank 1
        pain_confidence="strong",  # rank 0
    )
    # Same salience + grounding; pain_confidence breaks tie.
    grounded_strong = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:d_grounded_strong",
        excerpt_text="rank: grounded strong",
        salience_score=4.0,
        grounding_status="grounded",
        pain_confidence="strong",
    )
    grounded_weak = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:e_grounded_weak",
        excerpt_text="rank: grounded weak",
        salience_score=4.0,
        grounding_status="grounded",
        pain_confidence="weak",
    )
    for c in (high_salience, grounded, not_grounded, grounded_strong, grounded_weak):
        await upsert_claim(pool, c)

    results = await select_best_claim(
        pool,
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        target_entity="TestVendor",
        vendor_name="TestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        limit=10,
    )
    order = [r.witness_id for r in results]
    assert order == [
        "witness:test:a_high_salience",  # salience=9
        "witness:test:b_grounded",        # salience=5, grounded beats not-grounded
        "witness:test:c_not_grounded",    # salience=5, not grounded
        "witness:test:d_grounded_strong",  # salience=4, strong beats weak
        "witness:test:e_grounded_weak",    # salience=4, weak last
    ], f"unexpected ordering: {order}"


@pytest.mark.asyncio
async def test_select_best_claim_dedups_by_fingerprint(pool, artifact_id):
    """When two valid claims share a source_excerpt_fingerprint, only the
    higher-ranked one surfaces. Use the same review_id + excerpt_text on
    both; differentiate by claim_type so they pass the unique index."""
    review_id = uuid4()
    a = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:dup_a",
        excerpt_text="same phrase",
        source_review_id=review_id,
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        salience_score=6.0,
    )
    b = _make_claim(
        artifact_id=artifact_id,
        witness_id="witness:test:dup_b",
        excerpt_text="same phrase",
        source_review_id=review_id,
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        salience_score=4.0,
    )
    await upsert_claim(pool, a)
    await upsert_claim(pool, b)
    # Sanity: writer must have computed identical fingerprints from the
    # shared (review_id, excerpt_text). If the writer didn't compute,
    # both rows would have NULL fingerprints and dedup would no-op.
    fps = await pool.fetch(
        "SELECT source_excerpt_fingerprint FROM b2b_evidence_claims "
        "WHERE artifact_id = $1",
        artifact_id,
    )
    assert {row["source_excerpt_fingerprint"] for row in fps} == {
        _compute_fingerprint(source_review_id=review_id, excerpt_text="same phrase")
    }

    results = await select_best_claim(
        pool,
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        target_entity="TestVendor",
        vendor_name="TestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        limit=10,
    )
    assert len(results) == 1
    assert results[0].witness_id == "witness:test:dup_a", (
        "higher-salience row must win the dedup"
    )


@pytest.mark.asyncio
async def test_select_best_claim_secondary_target_filter(pool, artifact_id):
    """Passing secondary_target must narrow results; passing None matches all."""
    base = dict(
        artifact_id=artifact_id,
        claim_type=ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
        excerpt_text="displacement",
    )
    a = _make_claim(
        **base,
        witness_id="witness:test:to_hubspot",
        secondary_target="HubSpot",
        salience_score=5.0,
    )
    b = _make_claim(
        **base,
        witness_id="witness:test:to_salesforce",
        secondary_target="Salesforce",
        salience_score=6.0,
    )
    await upsert_claim(pool, a)
    await upsert_claim(pool, b)

    only_hubspot = await select_best_claim(
        pool,
        claim_type=ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
        target_entity="TestVendor",
        vendor_name="TestVendor",
        secondary_target="HubSpot",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        limit=10,
    )
    assert [r.secondary_target for r in only_hubspot] == ["HubSpot"]

    both = await select_best_claim(
        pool,
        claim_type=ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
        target_entity="TestVendor",
        vendor_name="TestVendor",
        secondary_target=None,
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        limit=10,
    )
    assert sorted(r.secondary_target for r in both) == ["HubSpot", "Salesforce"]


@pytest.mark.asyncio
async def test_upsert_computes_fingerprint_from_excerpt(pool, artifact_id):
    """The writer (not the caller) must compute source_excerpt_fingerprint
    from (source_review_id, excerpt_text). This is the contract guard
    against production callers forgetting to set the field."""
    review_id = uuid4()
    excerpt = "specific phrase to fingerprint"
    claim = PersistedClaim(
        artifact_type="synthesis",
        artifact_id=artifact_id,
        vendor_name="TestVendor",
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        target_entity="TestVendor",
        status=ClaimValidationStatus.VALID,
        synthesis_id=artifact_id,
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        witness_id="witness:test:fp",
        source_review_id=review_id,
        excerpt_text=excerpt,
        salience_score=5.0,
        grounding_status="grounded",
        pain_confidence="strong",
        # Note: source_excerpt_fingerprint NOT pre-set on the claim.
    )
    await upsert_claim(pool, claim)
    stored = await pool.fetchval(
        "SELECT source_excerpt_fingerprint FROM b2b_evidence_claims "
        "WHERE artifact_id = $1",
        artifact_id,
    )
    expected = _compute_fingerprint(source_review_id=review_id, excerpt_text=excerpt)
    assert stored is not None
    assert stored == expected


@pytest.mark.asyncio
async def test_upsert_rejects_valid_without_fingerprint_inputs(pool, artifact_id):
    """A valid row with no excerpt_text AND no precomputed fingerprint
    must raise ValueError. Otherwise dedup would silently break for any
    consumer that calls select_best_claim with limit > 1."""
    bad = PersistedClaim(
        artifact_type="synthesis",
        artifact_id=artifact_id,
        vendor_name="TestVendor",
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        target_entity="TestVendor",
        status=ClaimValidationStatus.VALID,
        synthesis_id=artifact_id,
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        witness_id="witness:test:no_fp",
        source_review_id=uuid4(),  # have review_id but no excerpt_text
        salience_score=5.0,
        grounding_status="grounded",
        pain_confidence="strong",
    )
    with pytest.raises(ValueError, match="source_excerpt_fingerprint"):
        await upsert_claim(pool, bad)
    # And: nothing was written.
    count = await pool.fetchval(
        "SELECT count(*) FROM b2b_evidence_claims WHERE artifact_id = $1",
        artifact_id,
    )
    assert count == 0


@pytest.mark.asyncio
async def test_upsert_allows_invalid_without_fingerprint(pool, artifact_id):
    """Invalid / cannot_validate rows do not need a fingerprint -- they
    never surface in select_best_claim, so dedup does not apply. This
    keeps the contract guard from blocking the audit-row write path."""
    claim = PersistedClaim(
        artifact_type="synthesis",
        artifact_id=artifact_id,
        vendor_name="TestVendor",
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        target_entity="TestVendor",
        status=ClaimValidationStatus.INVALID,
        rejection_reason="polarity_not_negative_or_mixed",
        synthesis_id=artifact_id,
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        witness_id="witness:test:invalid_no_fp",
        salience_score=0.0,
    )
    await upsert_claim(pool, claim)
    row = await pool.fetchrow(
        "SELECT status, source_excerpt_fingerprint FROM b2b_evidence_claims "
        "WHERE artifact_id = $1",
        artifact_id,
    )
    assert row["status"] == "invalid"
    assert row["source_excerpt_fingerprint"] is None


@pytest.mark.asyncio
async def test_select_best_claim_no_underfill_under_fingerprint_saturation(pool, artifact_id):
    """Pathological case: the leading rows by salience all share one
    fingerprint, then a second fingerprint, then a third. The old
    over-fetch heuristic could under-fill if the share band exceeded
    limit*3. The SQL ROW_NUMBER() dedup must always return up to N
    unique fingerprint groups."""
    # 10 rows sharing fingerprint A (high salience), then 1 row with
    # fingerprint B, then 1 row with fingerprint C. limit=3 must return
    # exactly 3 (one per fingerprint group), not under-fill.
    review_a = uuid4()
    review_b = uuid4()
    review_c = uuid4()

    for i in range(10):
        await upsert_claim(
            pool,
            _make_claim(
                artifact_id=artifact_id,
                witness_id=f"witness:test:saturate_a_{i:02d}",
                excerpt_text="phrase A",
                source_review_id=review_a,
                salience_score=9.0 - i * 0.01,  # all higher than B/C
            ),
        )
    await upsert_claim(
        pool,
        _make_claim(
            artifact_id=artifact_id,
            witness_id="witness:test:saturate_b",
            excerpt_text="phrase B",
            source_review_id=review_b,
            salience_score=4.0,
        ),
    )
    await upsert_claim(
        pool,
        _make_claim(
            artifact_id=artifact_id,
            witness_id="witness:test:saturate_c",
            excerpt_text="phrase C",
            source_review_id=review_c,
            salience_score=3.0,
        ),
    )

    results = await select_best_claim(
        pool,
        claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
        target_entity="TestVendor",
        vendor_name="TestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        limit=3,
    )
    assert len(results) == 3, (
        f"expected 3 unique fingerprint groups; got {len(results)} "
        f"(witnesses: {[r.witness_id for r in results]})"
    )
    # Highest salience per group wins; group A's top row is saturate_a_00.
    assert results[0].witness_id == "witness:test:saturate_a_00"
    assert results[1].witness_id == "witness:test:saturate_b"
    assert results[2].witness_id == "witness:test:saturate_c"


def test_dedup_selections_helper_preserves_first():
    """Pure helper, no DB. First-seen fingerprint wins; un-fingerprinted
    rows pass through unchanged."""
    def _sel(witness_id: str, fp: str | None) -> ClaimSelection:
        return ClaimSelection(
            id=uuid4(),
            artifact_type="synthesis",
            artifact_id=uuid4(),
            vendor_name="V",
            claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
            target_entity="V",
            secondary_target=None,
            witness_id=witness_id,
            source_review_id=None,
            source_span_id=None,
            salience_score=0.0,
            grounding_status=None,
            pain_confidence=None,
            source_excerpt_fingerprint=fp,
            supporting_fields=(),
            claim_payload={},
        )

    sels = [
        _sel("a", "fp1"),
        _sel("b", "fp2"),
        _sel("c", "fp1"),  # duplicate of a
        _sel("d", None),   # passes through
        _sel("e", None),   # passes through
        _sel("f", "fp2"),  # duplicate of b
    ]
    deduped = dedup_selections(sels)
    assert [s.witness_id for s in deduped] == ["a", "b", "d", "e"]
