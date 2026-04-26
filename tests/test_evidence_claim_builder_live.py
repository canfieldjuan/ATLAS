"""Live integration tests for the EvidenceClaim shadow builder.

Exercises the synthesis-side wiring against a real Postgres + the
b2b_evidence_claims shadow table from migration 305. Verifies:

  - Deterministic synthesis_artifact_id (same composite key -> same UUID).
  - Eligibility selector emits the universal frames + only typed claims
    matching the witness's signal data.
  - Orchestrator writes one row per (witness x eligible claim_type).
  - Idempotent on replay: second call produces the same row set, no
    duplicates, no row_count drift.
  - Antecedent_trap is detected when source_reviews + known_vendor_names
    are plumbed (this is the contract gap Step 5 is supposed to close).
  - Source-provenance invariant: a witness without excerpt_text never
    produces a 'valid' claim, so the writer's contract guard never fires.

Run:
    python -m pytest tests/test_evidence_claim_builder_live.py -v -s --tb=short
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

from atlas_brain.services.b2b.evidence_claim import ClaimType
from atlas_brain.services.b2b.evidence_claim_builder import (
    eligible_claim_types_for,
    synthesis_artifact_id,
    write_evidence_claims_for_synthesis,
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
async def clean_artifact(pool):
    """Compute the deterministic artifact_id that the test will write
    under, and clean any rows referencing it before and after."""
    aid = synthesis_artifact_id(
        vendor_name="ShadowTestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
    )
    await pool.execute(
        "DELETE FROM b2b_evidence_claims WHERE artifact_id = $1", aid
    )
    yield aid
    await pool.execute(
        "DELETE FROM b2b_evidence_claims WHERE artifact_id = $1", aid
    )


def _witness(
    *,
    review_id: UUID,
    excerpt: str,
    pain_category: str = "pricing",
    phrase_subject: str | None = "subject_vendor",
    phrase_polarity: str | None = "negative",
    phrase_role: str | None = "primary_driver",
    pain_confidence: str | None = "strong",
    competitor: str | None = None,
    reviewer_company: str | None = None,
    salience: float = 5.0,
    grounding: str = "grounded",
    vendor_name: str = "ShadowTestVendor",
    witness_id_suffix: str = "0",
) -> dict:
    return {
        "witness_id": f"witness:{review_id}:{witness_id_suffix}",
        "review_id": str(review_id),
        "vendor_name": vendor_name,
        "excerpt_text": excerpt,
        "pain_category": pain_category,
        "phrase_subject": phrase_subject,
        "phrase_polarity": phrase_polarity,
        "phrase_role": phrase_role,
        "phrase_verbatim": True,
        "pain_confidence": pain_confidence,
        "grounding_status": grounding,
        "salience_score": salience,
        "source_span_id": f"review:{review_id}:span:0",
        "competitor": competitor,
        "reviewer_company": reviewer_company,
    }


def test_synthesis_artifact_id_is_deterministic():
    a = synthesis_artifact_id(
        vendor_name="Asana",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
    )
    b = synthesis_artifact_id(
        vendor_name="Asana",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
    )
    assert a == b
    # Different composite -> different UUID.
    c = synthesis_artifact_id(
        vendor_name="Asana",
        as_of_date=date(2026, 4, 26),
        analysis_window_days=90,
        schema_version="v2",
    )
    assert a != c


def test_eligibility_includes_universal_and_filters_typed():
    # Default _witness() is phrase_polarity='negative' -- counterevidence
    # is structurally ineligible and must NOT be emitted (would always
    # reject as polarity_not_positive, audit noise).
    base = _witness(review_id=uuid4(), excerpt="x", pain_category="pricing")
    types = eligible_claim_types_for(base)
    assert ClaimType.PAIN_CLAIM_ABOUT_VENDOR in types
    assert ClaimType.COUNTEREVIDENCE_ABOUT_VENDOR not in types  # negative polarity
    assert ClaimType.PRICING_URGENCY_CLAIM in types
    # Off-category: not present.
    assert ClaimType.SUPPORT_FAILURE_CLAIM not in types
    assert ClaimType.FEATURE_GAP_CLAIM not in types
    # No competitor or reviewer_company on this witness.
    assert ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR not in types
    assert ClaimType.NAMED_ACCOUNT_ANCHOR not in types

    # Positive-polarity witness DOES emit counterevidence.
    positive = _witness(
        review_id=uuid4(), excerpt="great", pain_category="pricing",
        phrase_polarity="positive",
    )
    types = eligible_claim_types_for(positive)
    assert ClaimType.COUNTEREVIDENCE_ABOUT_VENDOR in types

    # Tagless witness (None polarity) still skips counterevidence.
    tagless = _witness(
        review_id=uuid4(), excerpt="x", pain_category="pricing",
        phrase_polarity=None,
    )
    types = eligible_claim_types_for(tagless)
    assert ClaimType.COUNTEREVIDENCE_ABOUT_VENDOR not in types

    with_competitor = _witness(
        review_id=uuid4(), excerpt="x", competitor="HubSpot"
    )
    types = eligible_claim_types_for(with_competitor)
    assert ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR in types
    assert ClaimType.DISPLACEMENT_PROOF_FROM_COMPETITOR in types

    with_company = _witness(
        review_id=uuid4(), excerpt="x", reviewer_company="Globex"
    )
    types = eligible_claim_types_for(with_company)
    assert ClaimType.NAMED_ACCOUNT_ANCHOR in types


@pytest.mark.asyncio
async def test_orchestrator_writes_one_row_per_eligible_claim_type(pool, clean_artifact):
    """Negative-polarity pain witness with pain_category=pricing emits
    pain_claim_about_vendor + pricing_urgency_claim only. Counterevidence
    is no longer attempted because phrase_polarity != 'positive' makes it
    structurally ineligible -- the rejection would always be
    polarity_not_positive and is audit noise."""
    rid = uuid4()
    witness = _witness(
        review_id=rid,
        excerpt="pricing keeps creeping up at every renewal",
        pain_category="pricing",
    )
    source_reviews = {
        str(rid): {
            "id": str(rid),
            "vendor_name": "ShadowTestVendor",
            "review_text": "pricing keeps creeping up at every renewal.",
            "summary": "",
        }
    }
    counts = await write_evidence_claims_for_synthesis(
        pool,
        vendor_name="ShadowTestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
        witnesses=[witness],
        source_reviews=source_reviews,
    )
    rows = await pool.fetch(
        "SELECT claim_type, status, rejection_reason "
        "FROM b2b_evidence_claims WHERE artifact_id = $1 ORDER BY claim_type",
        clean_artifact,
    )
    assert len(rows) == 2
    by_type = {r["claim_type"]: (r["status"], r["rejection_reason"]) for r in rows}
    assert by_type["pain_claim_about_vendor"] == ("valid", None)
    assert by_type["pricing_urgency_claim"] == ("valid", None)
    assert "counterevidence_about_vendor" not in by_type, (
        "negative-polarity witness must not emit a counterevidence row"
    )

    assert counts["valid"] == 2
    assert counts["invalid"] == 0
    assert counts["cannot_validate"] == 0


@pytest.mark.asyncio
async def test_orchestrator_purges_stale_rows_when_eligibility_shrinks(pool, clean_artifact):
    """Per-artifact rewrite invariant: tightening eligibility (or any
    other change that REMOVES an attempt) must not leave orphan rows
    behind. The replay-key upsert preserves rows the current run still
    emits; the per-artifact purge removes rows the current run no
    longer emits.

    Simulate by manually inserting a 'stale' row under the canary
    artifact_id with a claim_type the current eligibility selector
    won't emit, then run the orchestrator. The stale row must be gone.
    """
    rid = uuid4()
    witness = _witness(
        review_id=rid,
        excerpt="pricing keeps creeping up",
        pain_category="pricing",
    )
    # Plant a stale row under the canary artifact_id. Use a claim_type
    # that the orchestrator would NOT emit for this witness (e.g. a
    # support_failure_claim on a pricing pain witness).
    await pool.execute(
        """
        INSERT INTO b2b_evidence_claims (
            artifact_type, artifact_id, vendor_name, claim_type,
            target_entity, status, rejection_reason, salience_score,
            grounding_status, witness_id
        ) VALUES (
            'synthesis', $1, 'ShadowTestVendor', 'support_failure_claim',
            'ShadowTestVendor', 'invalid', 'pain_category_mismatch', 0,
            'grounded', 'witness:test:stale'
        )
        """,
        clean_artifact,
    )
    pre_count = await pool.fetchval(
        "SELECT count(*) FROM b2b_evidence_claims WHERE artifact_id = $1",
        clean_artifact,
    )
    assert pre_count == 1

    counts = await write_evidence_claims_for_synthesis(
        pool,
        vendor_name="ShadowTestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
        witnesses=[witness],
        source_reviews={},
    )
    assert counts["purged_stale"] == 1, (
        f"orchestrator must report 1 purged row; got {counts['purged_stale']}"
    )

    rows = await pool.fetch(
        "SELECT claim_type FROM b2b_evidence_claims WHERE artifact_id = $1",
        clean_artifact,
    )
    by_type = {r["claim_type"] for r in rows}
    assert "support_failure_claim" not in by_type, (
        "stale row must be purged before re-emit"
    )
    # The current run still wrote its own rows.
    assert "pain_claim_about_vendor" in by_type
    assert "pricing_urgency_claim" in by_type


@pytest.mark.asyncio
async def test_orchestrator_idempotent_on_replay(pool, clean_artifact):
    """Set-level idempotency: same synthesis composite + same witnesses
    produces the same set of (claim_type, status, rejection_reason)
    tuples on replay. Row count holds.

    Row IDENTITY does NOT hold across replays under the purge model:
    the per-artifact pre-write DELETE means the second run inserts
    fresh rows. created_at and validated_at both advance, and id is
    a new UUID. This is the right semantics for shadow capture --
    each synthesis cycle is a fresh decision snapshot, not a
    accreting first-seen log."""
    rid = uuid4()
    witness = _witness(
        review_id=rid,
        excerpt="pricing keeps creeping up",
        pain_category="pricing",
    )
    args = dict(
        vendor_name="ShadowTestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
        witnesses=[witness],
        source_reviews={},
    )
    await write_evidence_claims_for_synthesis(pool, **args)
    first = await pool.fetch(
        "SELECT claim_type, status, rejection_reason "
        "FROM b2b_evidence_claims WHERE artifact_id = $1 ORDER BY claim_type",
        clean_artifact,
    )
    first_set = {(r["claim_type"], r["status"], r["rejection_reason"]) for r in first}

    counts = await write_evidence_claims_for_synthesis(pool, **args)
    second = await pool.fetch(
        "SELECT claim_type, status, rejection_reason "
        "FROM b2b_evidence_claims WHERE artifact_id = $1 ORDER BY claim_type",
        clean_artifact,
    )
    second_set = {(r["claim_type"], r["status"], r["rejection_reason"]) for r in second}

    assert len(first) == len(second), (
        f"replay changed row count: {len(first)} -> {len(second)}"
    )
    assert first_set == second_set, (
        "replay must produce the same (claim_type, status, rejection_reason) set"
    )
    # The replay should have purged the first run's rows.
    assert counts["purged_stale"] == len(first), (
        f"replay should purge {len(first)} prior rows; got {counts['purged_stale']}"
    )


@pytest.mark.asyncio
async def test_antecedent_trap_detected_with_aliases_and_review_text(pool, clean_artifact):
    """When the source_window matches a competitor-named antecedent
    pattern AND known_vendor_names contains the competitor, the trap
    fires and pain_claim_about_vendor returns invalid(antecedent_trap)
    instead of valid. This is the Step 5 wiring contract: aliases +
    review text plumbed -> trap caught at write time."""
    rid = uuid4()
    # Mistagged subject (LLM said subject_vendor) but the source proves
    # the negative trait belongs to HubSpot.
    witness = _witness(
        review_id=rid,
        excerpt="did not provide a good UI for non-sales users",
        pain_category="ux",
        phrase_subject="subject_vendor",  # <-- mistagged
    )
    source_reviews = {
        str(rid): {
            "id": str(rid),
            "vendor_name": "ShadowTestVendor",
            "review_text": (
                "We migrated from HubSpot to ShadowTestVendor last year. "
                "The previous tool did not provide a good UI for non-sales "
                "users and our tutors could not pick it up."
            ),
            "summary": "",
        }
    }
    counts = await write_evidence_claims_for_synthesis(
        pool,
        vendor_name="ShadowTestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
        witnesses=[witness],
        source_reviews=source_reviews,
        known_vendor_names=frozenset({"ShadowTestVendor", "HubSpot"}),
    )
    row = await pool.fetchrow(
        "SELECT status, rejection_reason FROM b2b_evidence_claims "
        "WHERE artifact_id = $1 AND claim_type = 'pain_claim_about_vendor'",
        clean_artifact,
    )
    assert row is not None
    assert row["status"] == "invalid"
    assert row["rejection_reason"] == "antecedent_trap"
    assert counts["invalid"] >= 1


@pytest.mark.asyncio
async def test_witness_without_excerpt_never_yields_valid(pool, clean_artifact):
    """The source-provenance invariant. A witness with no excerpt_text
    that would otherwise validate must produce cannot_validate, never
    valid -- otherwise the repository's contract guard would raise on
    upsert. The write must succeed because invalid/cannot_validate rows
    do not require fingerprint inputs."""
    witness = _witness(
        review_id=uuid4(),
        excerpt="",  # blank excerpt
        pain_category="pricing",
    )
    counts = await write_evidence_claims_for_synthesis(
        pool,
        vendor_name="ShadowTestVendor",
        as_of_date=date(2026, 4, 25),
        analysis_window_days=90,
        schema_version="v2",
        witnesses=[witness],
        source_reviews={},
    )
    valid_rows = await pool.fetch(
        "SELECT claim_type FROM b2b_evidence_claims "
        "WHERE artifact_id = $1 AND status = 'valid'",
        clean_artifact,
    )
    assert valid_rows == [], (
        f"witness without excerpt_text leaked a valid row: {valid_rows}"
    )
    assert counts["valid"] == 0
    assert counts["errors"] == 0  # writer guard never fires
