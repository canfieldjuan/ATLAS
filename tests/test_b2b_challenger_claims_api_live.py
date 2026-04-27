"""Live integration tests for /b2b/challenger-claims/{challenger}.

Exercises the full HTTP stack: FastAPI router -> challenger ProductClaim
aggregator -> real Postgres + b2b_evidence_claims + b2b_reviews. Verifies
that the API preserves every render gate, the witness-by-reviewer
semantics from Patch 5a1, and the challenger-centric envelope flip
(target_entity=challenger, secondary_target=incumbent).

Tests insert their own b2b_evidence_claims + b2b_reviews rows scoped to
unique artifact_ids and review_ids so they're cleanup-safe and parallel-
safe. Uses httpx.AsyncClient with ASGITransport so the asyncpg pool and
the request loop share an event loop (TestClient creates a separate loop
per request, which collides with asyncpg's per-connection binding).
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
from fastapi import FastAPI

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

from atlas_brain.api import b2b_challenger_claims as api_module
from atlas_brain.auth.dependencies import AuthUser, require_auth


def _auth_user() -> AuthUser:
    return AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )


async def _new_pool() -> asyncpg.Pool:
    from atlas_brain.storage.config import db_settings

    return await asyncpg.create_pool(
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.database,
        user=db_settings.user,
        password=db_settings.password,
        min_size=1,
        max_size=3,
    )


def _make_app(pool: asyncpg.Pool, monkeypatch, *, with_auth_override: bool = True) -> FastAPI:
    def _live_pool_stub():
        return pool

    monkeypatch.setattr(api_module, "_pool_or_503", _live_pool_stub)
    monkeypatch.setattr(api_module, "get_db_pool", _live_pool_stub)

    app = FastAPI()
    app.include_router(api_module.router)
    if with_auth_override:
        app.dependency_overrides[require_auth] = _auth_user
    return app


def _async_client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


def _make_mounted_app(pool: asyncpg.Pool, monkeypatch) -> FastAPI:
    """Mirror main.py:718 wiring -- aggregate /api router under /api/v1.
    Catches both router de-registration and prefix drift."""

    def _live_pool_stub():
        return pool

    monkeypatch.setattr(api_module, "_pool_or_503", _live_pool_stub)
    monkeypatch.setattr(api_module, "get_db_pool", _live_pool_stub)

    from atlas_brain.api import router as api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _auth_user
    return app


# ---------------------------------------------------------------------------
# Test data setup helpers.
# ---------------------------------------------------------------------------


async def _insert_review(pool, review_id: UUID, *, vendor_name: str, reviewer_name: str) -> None:
    """Insert a minimal b2b_reviews row so the JOIN on reviewer_name resolves.

    Only the NOT-NULL-without-default columns need explicit values
    (dedup_key, review_text, source, vendor_name). The rest take their
    schema defaults."""
    await pool.execute(
        """
        INSERT INTO b2b_reviews (
            id, vendor_name, reviewer_name, review_text,
            source, enrichment_status, dedup_key
        )
        VALUES ($1, $2, $3, $4, 'g2', 'enriched', $5)
        ON CONFLICT (id) DO UPDATE SET reviewer_name = EXCLUDED.reviewer_name
        """,
        review_id,
        vendor_name,
        reviewer_name,
        f"Test review for {vendor_name} (reviewer: {reviewer_name})",
        f"test-dedup-{review_id}",
    )


async def _insert_displacement_claim(
    pool,
    *,
    artifact_id: UUID,
    incumbent: str,
    challenger: str,
    review_id: UUID,
    witness_id: str,
    excerpt_text: str = "We switched from incumbent to challenger.",
    salience_score: float = 5.0,
    as_of_date: date = date(2026, 4, 26),
    analysis_window_days: int = 30,
) -> None:
    """Insert a valid displacement_proof_to_competitor row.

    The aggregator filters by status='valid' and performs the
    incumbent->challenger query, so vendor_name=incumbent +
    secondary_target=challenger is the substrate shape we need.
    """
    from atlas_brain.services.b2b.evidence_claim import source_excerpt_fingerprint

    fingerprint = source_excerpt_fingerprint(
        source_review_id=review_id, excerpt_text=excerpt_text
    )
    await pool.execute(
        """
        INSERT INTO b2b_evidence_claims (
            artifact_type, artifact_id, synthesis_id,
            vendor_name, as_of_date, analysis_window_days,
            claim_schema_version, claim_type, target_entity,
            secondary_target, witness_id, source_review_id,
            salience_score, grounding_status, pain_confidence,
            source_excerpt_fingerprint,
            status, supporting_fields, claim_payload,
            validated_at, created_at
        ) VALUES (
            'synthesis', $1, $1,
            $2, $3, $4,
            'v1', 'displacement_proof_to_competitor', $2,
            $5, $6, $7,
            $8, 'grounded', 'strong',
            $9,
            'valid', '[]'::jsonb,
            jsonb_build_object('excerpt_text', $10::text),
            now(), now()
        )
        """,
        artifact_id,
        incumbent,
        as_of_date,
        analysis_window_days,
        challenger,
        witness_id,
        review_id,
        salience_score,
        fingerprint,
        excerpt_text,
    )


@pytest.fixture
async def cleanup_artifacts():
    """Tracks artifact_ids and review_ids created during a test, cleans
    them up afterwards even on failure."""
    artifacts: list[UUID] = []
    reviews: list[UUID] = []
    pool = await _new_pool()
    try:
        yield artifacts, reviews, pool
    finally:
        if artifacts:
            await pool.execute(
                "DELETE FROM b2b_evidence_claims WHERE artifact_id = ANY($1::uuid[])",
                artifacts,
            )
        if reviews:
            await pool.execute(
                "DELETE FROM b2b_reviews WHERE id = ANY($1::uuid[])",
                reviews,
            )
        await pool.close()


# ---------------------------------------------------------------------------
# Auth.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_requires_auth(monkeypatch):
    """No auth override -> 401."""
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.saas_auth, "enabled", True, raising=False)

    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch, with_auth_override=False)
        async with _async_client(app) as client:
            response = await client.get(
                "/b2b/challenger-claims/Microsoft-365?analysis_window_days=30"
            )
        assert response.status_code == 401
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Mount path: production-effective /api/v1/b2b/challenger-claims/{challenger}.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_resolves_under_production_api_v1_prefix(monkeypatch):
    """The frontend will hit /api/v1/b2b/challenger-claims/{challenger}.
    Pin both the prefixed (200) and unprefixed (404) paths so a future
    registration drop or prefix change gets caught here."""
    pool = await _new_pool()
    try:
        app = _make_mounted_app(pool, monkeypatch)
        async with _async_client(app) as client:
            mounted = await client.get(
                "/api/v1/b2b/challenger-claims/UnknownChallenger?analysis_window_days=30"
            )
            unmounted = await client.get(
                "/b2b/challenger-claims/UnknownChallenger?analysis_window_days=30"
            )
        assert mounted.status_code == 200, mounted.text
        body = mounted.json()
        assert body["challenger"] == "UnknownChallenger"
        assert body["rows"] == []
        assert unmounted.status_code == 404
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Empty challenger: 200 with rows=[] (NOT 404).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_challenger_returns_empty_rows(monkeypatch):
    """A challenger with no displacement evidence must return 200 with
    rows=[] so the React side handles 'no data' with one rendering
    path. 404 would conflate 'unknown vendor' with 'real challenger
    without evidence yet'."""
    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch)
        async with _async_client(app) as client:
            response = await client.get(
                "/b2b/challenger-claims/NonexistentChallenger_xyz?analysis_window_days=30"
            )
        assert response.status_code == 200
        body = response.json()
        assert body["challenger"] == "NonexistentChallenger_xyz"
        assert body["rows"] == []
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Hero case: challenger with seeded displacement evidence renders correctly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seeded_challenger_returns_displacement_row_with_envelope_flip(monkeypatch, cleanup_artifacts):
    """End-to-end through the API: insert a valid
    displacement_proof_to_competitor row (vendor_name=Google Workspace,
    secondary_target=Microsoft 365), then GET /Microsoft 365 and assert
    the envelope flips correctly: target_entity=Microsoft 365,
    secondary_target=Google Workspace, claim_key='incumbent:Google Workspace'."""
    artifacts, reviews, pool = cleanup_artifacts
    artifact_id = uuid4()
    review_id = uuid4()
    artifacts.append(artifact_id)
    reviews.append(review_id)
    challenger_id = f"ChallengerTest_{uuid4().hex[:8]}"
    incumbent_id = f"IncumbentTest_{uuid4().hex[:8]}"

    await _insert_review(
        pool, review_id, vendor_name=incumbent_id, reviewer_name="Jane Smith"
    )
    await _insert_displacement_claim(
        pool,
        artifact_id=artifact_id,
        incumbent=incumbent_id,
        challenger=challenger_id,
        review_id=review_id,
        witness_id=f"witness:{uuid4()}",
    )

    app = _make_app(pool, monkeypatch)
    async with _async_client(app) as client:
        response = await client.get(
            f"/b2b/challenger-claims/{challenger_id}?analysis_window_days=30"
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["challenger"] == challenger_id
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["challenger"] == challenger_id
    assert row["incumbent"] == incumbent_id

    claim = row["claim"]
    # Envelope flip (the key Patch 5a1 invariant).
    assert claim["target_entity"] == challenger_id
    assert claim["secondary_target"] == incumbent_id
    assert claim["claim_key"] == f"incumbent:{incumbent_id}"
    assert claim["claim_scope"] == "competitor_pair"
    assert claim["claim_type"] == "direct_displacement"

    # Single witness, single distinct reviewer (Jane Smith).
    assert claim["supporting_count"] == 1
    assert claim["direct_evidence_count"] == 1
    assert claim["witness_count"] == 1

    # One witness -> LOW confidence -> renders but does not publish.
    assert claim["render_allowed"] is True
    assert claim["report_allowed"] is False
    assert claim["confidence"] == "low"
    assert claim["suppression_reason"] == "low_confidence"


# ---------------------------------------------------------------------------
# Microsoft-365-style failure mode: same person posting multiple displacement
# reviews must NOT trip MEDIUM confidence. Reviewer-name dedup blocks it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeat_reviewer_does_not_inflate_witness_count(monkeypatch, cleanup_artifacts):
    """The exact failure mode Patch 5a1 was meant to prevent: one
    reviewer posting 3 separate "switching" reviews must NOT trip
    MEDIUM confidence. Reviewer-name dedup at the SQL layer collapses
    them to 1 distinct witness -> LOW confidence -> report_allowed=False."""
    artifacts, reviews, pool = cleanup_artifacts
    artifact_ids = [uuid4(), uuid4(), uuid4()]
    review_ids = [uuid4(), uuid4(), uuid4()]
    artifacts.extend(artifact_ids)
    reviews.extend(review_ids)
    challenger_id = f"ChallengerTest_{uuid4().hex[:8]}"
    incumbent_id = f"IncumbentTest_{uuid4().hex[:8]}"
    repeat_reviewer = "Repeat Reviewer"

    for aid, rid in zip(artifact_ids, review_ids):
        await _insert_review(
            pool, rid, vendor_name=incumbent_id, reviewer_name=repeat_reviewer
        )
        await _insert_displacement_claim(
            pool,
            artifact_id=aid,
            incumbent=incumbent_id,
            challenger=challenger_id,
            review_id=rid,
            witness_id=f"witness:{uuid4()}",
            excerpt_text=f"Switching pass {aid}",
        )

    app = _make_app(pool, monkeypatch)
    async with _async_client(app) as client:
        response = await client.get(
            f"/b2b/challenger-claims/{challenger_id}?analysis_window_days=30"
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["rows"]) == 1
    claim = body["rows"][0]["claim"]

    # 3 distinct review_ids, but ONE distinct reviewer_name.
    assert claim["supporting_count"] == 3
    assert claim["witness_count"] == 1, (
        "single reviewer's repeat posts must collapse to 1 witness"
    )
    # 1 witness < medium_confidence_min_witnesses=2 -> LOW.
    assert claim["confidence"] == "low"
    assert claim["report_allowed"] is False, (
        "single-reviewer testimony cannot trip publish, regardless of repeat count"
    )


# ---------------------------------------------------------------------------
# Inverse-direction contradiction: bidirectional evidence cancels publish.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inverse_direction_evidence_blocks_publish(monkeypatch, cleanup_artifacts):
    """Pair with both X->Y and Y->X displacement evidence: the
    contradiction-ratio gate suppresses report_allowed even when each
    side individually has corroboration."""
    artifacts, reviews, pool = cleanup_artifacts
    challenger_id = f"ChallengerTest_{uuid4().hex[:8]}"
    incumbent_id = f"IncumbentTest_{uuid4().hex[:8]}"

    # 5 valid direct rows (incumbent->challenger): 5 distinct reviewers
    # so confidence math thinks corroboration exists.
    for i in range(5):
        aid = uuid4()
        rid = uuid4()
        artifacts.append(aid)
        reviews.append(rid)
        await _insert_review(
            pool, rid, vendor_name=incumbent_id, reviewer_name=f"Forward Reviewer {i}"
        )
        await _insert_displacement_claim(
            pool,
            artifact_id=aid,
            incumbent=incumbent_id,
            challenger=challenger_id,
            review_id=rid,
            witness_id=f"witness:fwd:{i}",
            excerpt_text=f"Switched from incumbent to challenger pass {i}",
        )

    # 3 valid INVERSE rows (challenger->incumbent): contradiction.
    for i in range(3):
        aid = uuid4()
        rid = uuid4()
        artifacts.append(aid)
        reviews.append(rid)
        await _insert_review(
            pool, rid, vendor_name=challenger_id, reviewer_name=f"Reverse Reviewer {i}"
        )
        await _insert_displacement_claim(
            pool,
            artifact_id=aid,
            incumbent=challenger_id,  # flipped: challenger is the subject losing share
            challenger=incumbent_id,  # incumbent is the named winner in the inverse row
            review_id=rid,
            witness_id=f"witness:inv:{i}",
            excerpt_text=f"Switched from challenger back to incumbent pass {i}",
        )

    app = _make_app(pool, monkeypatch)
    async with _async_client(app) as client:
        response = await client.get(
            f"/b2b/challenger-claims/{challenger_id}?analysis_window_days=30"
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["rows"]) == 1
    claim = body["rows"][0]["claim"]

    assert claim["supporting_count"] == 5
    assert claim["contradiction_count"] == 3
    # 3/5 = 0.6 above default contradiction_ratio_threshold=0.4 -> CONTRADICTORY.
    assert claim["evidence_posture"] == "contradictory"
    assert claim["render_allowed"] is True
    assert claim["report_allowed"] is False
    assert claim["suppression_reason"] == "contradictory_evidence"


# ---------------------------------------------------------------------------
# Response shape: every ProductClaim field round-trips.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_preserves_every_product_claim_field(monkeypatch, cleanup_artifacts):
    """Per-row claim envelope must mirror ProductClaim 1:1. If a future
    refactor drops a gate field from VendorClaimResponse, the React
    side could silently lose its render gate."""
    expected_fields = {
        "claim_id",
        "claim_key",
        "claim_scope",
        "claim_type",
        "claim_text",
        "target_entity",
        "secondary_target",
        "supporting_count",
        "direct_evidence_count",
        "witness_count",
        "contradiction_count",
        "denominator",
        "sample_size",
        "has_grounded_evidence",
        "confidence",
        "evidence_posture",
        "render_allowed",
        "report_allowed",
        "suppression_reason",
        "evidence_links",
        "contradicting_links",
        "as_of_date",
        "analysis_window_days",
        "schema_version",
    }
    artifacts, reviews, pool = cleanup_artifacts
    artifact_id = uuid4()
    review_id = uuid4()
    artifacts.append(artifact_id)
    reviews.append(review_id)
    challenger_id = f"ChallengerTest_{uuid4().hex[:8]}"
    incumbent_id = f"IncumbentTest_{uuid4().hex[:8]}"

    await _insert_review(
        pool, review_id, vendor_name=incumbent_id, reviewer_name="Field Test Reviewer"
    )
    await _insert_displacement_claim(
        pool,
        artifact_id=artifact_id,
        incumbent=incumbent_id,
        challenger=challenger_id,
        review_id=review_id,
        witness_id=f"witness:{uuid4()}",
    )

    app = _make_app(pool, monkeypatch)
    async with _async_client(app) as client:
        response = await client.get(
            f"/b2b/challenger-claims/{challenger_id}?analysis_window_days=30"
        )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"], "test setup should produce at least one row"
    claim_keys = set(body["rows"][0]["claim"].keys())
    missing = expected_fields - claim_keys
    extra = claim_keys - expected_fields
    assert not missing, f"response missing ProductClaim fields: {missing}"
    assert not extra, f"response has unexpected fields beyond ProductClaim: {extra}"


# ---------------------------------------------------------------------------
# Window param validation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_window_returns_422(monkeypatch):
    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch)
        async with _async_client(app) as client:
            too_small = await client.get(
                "/b2b/challenger-claims/Foo?analysis_window_days=1"
            )
            too_large = await client.get(
                "/b2b/challenger-claims/Foo?analysis_window_days=9999"
            )
        assert too_small.status_code == 422
        assert too_large.status_code == 422
    finally:
        await pool.close()
