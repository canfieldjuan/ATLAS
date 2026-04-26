"""Live integration tests for the EvidenceClaim audit summary.

Seeds b2b_evidence_claims with a known mix of statuses + rejection
reasons across multiple vendors / claim_types and verifies that
summarize_claim_validation() returns correct aggregations.

Each test scopes its writes to a unique date so concurrent runs do not
collide and cleanup is safe (DELETE WHERE as_of_date = ...).

Run:
    python -m pytest tests/test_evidence_claim_audit_live.py -v -s --tb=short
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
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
)
from atlas_brain.services.b2b.evidence_claim_repository import (
    PersistedClaim,
    upsert_claim,
)
from atlas_brain.services.reasoning_delivery_audit import (
    summarize_claim_validation,
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
async def audit_date(pool) -> date:
    """Pick a deterministic but unused date (deep in the future) and
    clean up any rows referencing it before and after the test."""
    # Deep-future date so we don't collide with real synthesis output.
    aod = date(2099, 1, 1) + timedelta(days=hash(uuid4()) % 365)
    await pool.execute(
        "DELETE FROM b2b_evidence_claims WHERE as_of_date = $1", aod
    )
    yield aod
    await pool.execute(
        "DELETE FROM b2b_evidence_claims WHERE as_of_date = $1", aod
    )


def _make(
    *,
    audit_date: date,
    artifact_id: UUID,
    vendor: str,
    claim_type: ClaimType,
    status: ClaimValidationStatus,
    rejection_reason: str | None,
    witness_id: str,
    excerpt: str = "test excerpt",
    pain_category: str | None = "pricing",
) -> PersistedClaim:
    return PersistedClaim(
        artifact_type="synthesis",
        artifact_id=artifact_id,
        vendor_name=vendor,
        claim_type=claim_type,
        target_entity=vendor,
        status=status,
        rejection_reason=rejection_reason,
        synthesis_id=artifact_id,
        as_of_date=audit_date,
        analysis_window_days=90,
        witness_id=witness_id,
        source_review_id=uuid4() if status == ClaimValidationStatus.VALID else None,
        excerpt_text=excerpt if status == ClaimValidationStatus.VALID else excerpt,
        salience_score=5.0,
        grounding_status="grounded",
        pain_confidence="strong",
        claim_payload={
            "excerpt_text": excerpt,
            "pain_category": pain_category,
        },
    )


@pytest.mark.asyncio
async def test_audit_summary_basic_counts(pool, audit_date):
    aid = uuid4()
    rows = [
        # 3 valid pain claims (Asana)
        _make(audit_date=audit_date, artifact_id=aid, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.VALID, rejection_reason=None,
              witness_id="w:asana:1", excerpt="phrase 1"),
        _make(audit_date=audit_date, artifact_id=aid, vendor="Asana",
              claim_type=ClaimType.PRICING_URGENCY_CLAIM,
              status=ClaimValidationStatus.VALID, rejection_reason=None,
              witness_id="w:asana:2", excerpt="phrase 2"),
        _make(audit_date=audit_date, artifact_id=aid, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.VALID, rejection_reason=None,
              witness_id="w:asana:3", excerpt="phrase 3"),
        # 2 invalid (Asana, polarity gate)
        _make(audit_date=audit_date, artifact_id=aid, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.INVALID,
              rejection_reason="polarity_not_negative_or_mixed",
              witness_id="w:asana:4"),
        _make(audit_date=audit_date, artifact_id=aid, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.INVALID,
              rejection_reason="role_passing_mention",
              witness_id="w:asana:5"),
        # 1 cannot_validate (synthesized span)
        _make(audit_date=audit_date, artifact_id=aid, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.CANNOT_VALIDATE,
              rejection_reason="phrase_subject_unavailable",
              witness_id="w:asana:6"),
    ]
    for r in rows:
        await upsert_claim(pool, r)

    summary = await summarize_claim_validation(pool, as_of_date=audit_date)

    assert summary["as_of_date"] == str(audit_date)
    assert summary["scope"]["total_rows"] == 6
    assert summary["scope"]["distinct_vendors"] == 1
    assert summary["scope"]["distinct_artifacts"] == 1
    assert summary["totals"] == {"valid": 3, "invalid": 2, "cannot_validate": 1}

    by_type = summary["by_claim_type"]
    assert by_type["pain_claim_about_vendor"] == {
        "total": 5, "valid": 2, "invalid": 2, "cannot_validate": 1
    }
    assert by_type["pricing_urgency_claim"] == {
        "total": 1, "valid": 1, "invalid": 0, "cannot_validate": 0
    }


@pytest.mark.asyncio
async def test_audit_top_rejection_reasons_sorted_by_count(pool, audit_date):
    aid = uuid4()
    # 3 polarity rejections, 1 role rejection -- polarity should rank first.
    fixtures = [
        ("polarity_not_negative_or_mixed", 3),
        ("role_passing_mention", 1),
    ]
    counter = 0
    for reason, n in fixtures:
        for i in range(n):
            counter += 1
            await upsert_claim(
                pool,
                _make(
                    audit_date=audit_date, artifact_id=aid, vendor="V",
                    claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
                    status=ClaimValidationStatus.INVALID,
                    rejection_reason=reason,
                    witness_id=f"w:v:{counter}",
                ),
            )

    summary = await summarize_claim_validation(pool, as_of_date=audit_date)
    reasons = summary["rejection_reasons_by_claim_type"]["pain_claim_about_vendor"]
    assert len(reasons) == 2
    assert reasons[0]["rejection_reason"] == "polarity_not_negative_or_mixed"
    assert reasons[0]["count"] == 3
    assert reasons[1]["rejection_reason"] == "role_passing_mention"
    assert reasons[1]["count"] == 1


@pytest.mark.asyncio
async def test_audit_per_vendor_counts(pool, audit_date):
    aid_a, aid_b = uuid4(), uuid4()
    # Asana: 2 valid, 1 invalid. Pipedrive: 1 valid, 0 invalid.
    rows = [
        _make(audit_date=audit_date, artifact_id=aid_a, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.VALID, rejection_reason=None,
              witness_id="w:a:1", excerpt="a1"),
        _make(audit_date=audit_date, artifact_id=aid_a, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.VALID, rejection_reason=None,
              witness_id="w:a:2", excerpt="a2"),
        _make(audit_date=audit_date, artifact_id=aid_a, vendor="Asana",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.INVALID,
              rejection_reason="role_passing_mention",
              witness_id="w:a:3"),
        _make(audit_date=audit_date, artifact_id=aid_b, vendor="Pipedrive",
              claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
              status=ClaimValidationStatus.VALID, rejection_reason=None,
              witness_id="w:p:1", excerpt="p1"),
    ]
    for r in rows:
        await upsert_claim(pool, r)

    summary = await summarize_claim_validation(pool, as_of_date=audit_date)
    by_vendor = {v["vendor_name"]: v for v in summary["by_vendor"]}
    assert set(by_vendor.keys()) == {"Asana", "Pipedrive"}
    assert by_vendor["Asana"]["valid"] == 2
    assert by_vendor["Asana"]["invalid"] == 1
    assert by_vendor["Asana"]["total"] == 3
    assert by_vendor["Pipedrive"]["valid"] == 1
    assert by_vendor["Pipedrive"]["total"] == 1
    # Vendor list is ordered by total DESC -> Asana first.
    assert summary["by_vendor"][0]["vendor_name"] == "Asana"


@pytest.mark.asyncio
async def test_audit_invalid_examples_capped_per_reason(pool, audit_date):
    aid = uuid4()
    # 5 rows with the same (claim_type, rejection_reason). The example
    # cap should clamp to invalid_examples_per_reason=2.
    for i in range(5):
        await upsert_claim(
            pool,
            _make(
                audit_date=audit_date, artifact_id=aid, vendor="V",
                claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
                status=ClaimValidationStatus.INVALID,
                rejection_reason="role_passing_mention",
                witness_id=f"w:v:{i}",
                excerpt=f"example excerpt {i}",
            ),
        )

    summary = await summarize_claim_validation(
        pool, as_of_date=audit_date, invalid_examples_per_reason=2
    )
    examples = summary["invalid_examples"]
    assert len(examples) == 2
    for ex in examples:
        assert ex["claim_type"] == "pain_claim_about_vendor"
        assert ex["rejection_reason"] == "role_passing_mention"
        assert ex["excerpt_preview"].startswith("example excerpt")


@pytest.mark.asyncio
async def test_audit_pain_category_breakdown_from_payload(pool, audit_date):
    aid = uuid4()
    cats = ("pricing", "pricing", "support", "ux", None)
    for i, cat in enumerate(cats):
        await upsert_claim(
            pool,
            _make(
                audit_date=audit_date, artifact_id=aid, vendor="V",
                claim_type=ClaimType.PAIN_CLAIM_ABOUT_VENDOR,
                status=ClaimValidationStatus.VALID, rejection_reason=None,
                witness_id=f"w:cat:{i}",
                excerpt=f"e{i}",
                pain_category=cat,
            ),
        )

    summary = await summarize_claim_validation(pool, as_of_date=audit_date)
    bucket = summary["by_pain_category"]
    assert bucket["pricing"]["total"] == 2
    assert bucket["support"]["total"] == 1
    assert bucket["ux"]["total"] == 1
    # None pain_category collapses to 'unknown'
    assert bucket["unknown"]["total"] == 1
