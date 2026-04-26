"""Live integration tests for /b2b/vendor-claims/{vendor_name}.

Exercises the full HTTP stack: FastAPI router -> ProductClaim aggregator
-> real Postgres. Verifies that the API preserves every render gate,
suppression reason, denominator, and evidence-link field exactly from
the underlying ProductClaim, so the React side cannot drift from the
contract.

The ClickUp suppressed-claim case (Phase 9 v3 dataset -> UNVERIFIED ->
render_allowed=False) is the hero pin: it proves the contract works
end to end through the API, not just at the aggregator boundary.

Uses httpx.AsyncClient with ASGITransport so the asyncpg pool and the
HTTP calls share a single event loop. TestClient creates a separate
loop per request, which collides with asyncpg's per-connection loop
binding.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

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

from atlas_brain.api import b2b_vendor_claims as api_module
from atlas_brain.auth.dependencies import AuthUser, require_auth


def _auth_user() -> AuthUser:
    """Authenticated B2B-retention user that satisfies require_b2b_plan('b2b_trial')."""
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
    """Build a FastAPI app with the vendor-claims router, the real
    asyncpg pool wired through _pool_or_503, and (optionally)
    require_auth overridden so plan-tier checks see a B2B-retention
    user."""

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
    """Build a FastAPI app that mounts the aggregate /api router under
    /api/v1 exactly like atlas_brain/main.py does in production.

    The other helper (_make_app) mounts the vendor-claims router
    directly at /b2b/vendor-claims, which validates router-level
    behavior but misses two production drift cases:
      1. The router is no longer registered in api/__init__.py.
      2. The aggregate /api/v1 prefix moves or changes.
    The mounted-path test below covers both."""

    def _live_pool_stub():
        return pool

    monkeypatch.setattr(api_module, "_pool_or_503", _live_pool_stub)
    monkeypatch.setattr(api_module, "get_db_pool", _live_pool_stub)

    from atlas_brain.api import router as api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _auth_user
    return app


# ----------------------------------------------------------------------------
# Auth.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_requires_auth(monkeypatch):
    """No auth override -> 401. The route must not bypass authentication
    just because the data layer is happy."""
    from atlas_brain.config import settings

    monkeypatch.setattr(settings.saas_auth, "enabled", True, raising=False)

    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch, with_auth_override=False)
        async with _async_client(app) as client:
            response = await client.get(
                "/b2b/vendor-claims/ClickUp?analysis_window_days=365"
            )
        assert response.status_code == 401
    finally:
        await pool.close()


# ----------------------------------------------------------------------------
# Hero case: ClickUp returns the suppressed claim shape.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clickup_returns_suppressed_claim_shape(monkeypatch):
    """End-to-end proof of the contract through the API: ClickUp's
    DM-churn rate is rendered with render_allowed=False and
    suppression_reason='unverified_evidence' because the underlying
    rows are v3-backed. denominator + supporting_count carry through
    so the dashboard can show 'Insufficient evidence (X of Y DMs)'
    instead of just hiding the card."""
    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch)
        async with _async_client(app) as client:
            response = await client.get(
                "/b2b/vendor-claims/ClickUp?analysis_window_days=365"
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["vendor_name"] == "ClickUp"
        assert body["analysis_window_days"] == 365
        assert isinstance(body["claims"], list)

        dm_claims = [
            c for c in body["claims"]
            if c["claim_type"] == "decision_maker_churn_rate"
        ]
        assert len(dm_claims) == 1
        claim = dm_claims[0]

        # The render gate -- the operating-rule proof.
        assert claim["render_allowed"] is False
        assert claim["report_allowed"] is False
        assert claim["suppression_reason"] == "unverified_evidence"
        assert claim["evidence_posture"] == "unverified"
        assert claim["has_grounded_evidence"] is False

        # Denominator + numerator carry through with real numbers so the
        # dashboard can label the suppression honestly.
        assert claim["denominator"] is not None and claim["denominator"] > 0
        assert claim["supporting_count"] >= 0
        assert claim["supporting_count"] <= claim["denominator"]
        assert claim["sample_size"] == claim["denominator"]

        # Rate-claim contradiction semantics: zero by design.
        assert claim["contradiction_count"] == 0
    finally:
        await pool.close()


# ----------------------------------------------------------------------------
# Unknown vendor: empty claims list, NOT 404.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_vendor_returns_empty_claims(monkeypatch):
    """A vendor with no DM data must return 200 with claims=[] so the
    dashboard renders 'no data' cleanly. Returning 404 would conflate
    'unknown vendor' with 'real vendor without DM coverage', and the
    UI surface needs a stable JSON shape either way."""
    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch)
        async with _async_client(app) as client:
            response = await client.get(
                "/b2b/vendor-claims/NonexistentVendor_xyz_123?analysis_window_days=365"
            )
        assert response.status_code == 200
        body = response.json()
        assert body["vendor_name"] == "NonexistentVendor_xyz_123"
        assert body["claims"] == []
    finally:
        await pool.close()


# ----------------------------------------------------------------------------
# Response shape: every ProductClaim field round-trips.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_preserves_every_product_claim_field(monkeypatch):
    """The API response must mirror ProductClaim 1:1 (minus `policy`,
    which is internal). If a future change drops a gate field from
    the response, the React side would silently lose its render gate
    and reports could publish suppressed claims."""
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
    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch)
        async with _async_client(app) as client:
            response = await client.get(
                "/b2b/vendor-claims/ClickUp?analysis_window_days=365"
            )
        assert response.status_code == 200
        body = response.json()
        assert body["claims"], "ClickUp must return at least one claim for this pin"
        claim_keys = set(body["claims"][0].keys())
        missing = expected_fields - claim_keys
        extra = claim_keys - expected_fields
        assert not missing, f"response missing ProductClaim fields: {missing}"
        assert not extra, f"response has unexpected fields beyond ProductClaim: {extra}"
    finally:
        await pool.close()


# ----------------------------------------------------------------------------
# Window param validation.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_resolves_under_production_api_v1_prefix(monkeypatch):
    """The frontend will hit /api/v1/b2b/vendor-claims/{vendor_name}
    because main.py mounts the aggregate api_router with prefix
    '/api/v1'. The other tests mount the vendor-claims router
    directly, which catches router-level regressions but not the
    case where the registration in api/__init__.py is removed or
    the aggregate prefix moves. This test mirrors main.py's mount
    path exactly so production drift gets caught here."""
    pool = await _new_pool()
    try:
        app = _make_mounted_app(pool, monkeypatch)
        async with _async_client(app) as client:
            # Hitting the production-effective path.
            mounted = await client.get(
                "/api/v1/b2b/vendor-claims/ClickUp?analysis_window_days=365"
            )
            # Same path WITHOUT the prefix should NOT resolve, so the
            # test pins both directions of the wiring.
            unmounted = await client.get(
                "/b2b/vendor-claims/ClickUp?analysis_window_days=365"
            )
        assert mounted.status_code == 200, mounted.text
        body = mounted.json()
        assert body["vendor_name"] == "ClickUp"
        assert body["claims"], "expected at least one claim for ClickUp"
        assert unmounted.status_code == 404, (
            "router must NOT resolve at the unprefixed path under main.py's mount"
        )
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_clickup_returns_both_dm_churn_and_price_complaint_claims(monkeypatch):
    """Both rate cards (DM churn + price complaint) come through the
    same /b2b/vendor-claims endpoint. The API loops aggregators and
    appends each non-None ProductClaim; the React side filters by
    claim_type. ClickUp returns both claims, both UNVERIFIED in the
    current v3-heavy dataset."""
    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch)
        async with _async_client(app) as client:
            response = await client.get(
                "/b2b/vendor-claims/ClickUp?analysis_window_days=365"
            )
        assert response.status_code == 200, response.text
        body = response.json()
        types_present = {c["claim_type"] for c in body["claims"]}
        assert "decision_maker_churn_rate" in types_present
        assert "price_complaint_rate" in types_present
        for claim_type in ("decision_maker_churn_rate", "price_complaint_rate"):
            claim = next(c for c in body["claims"] if c["claim_type"] == claim_type)
            # Same hero pin for both: tagless evidence -> suppression.
            assert claim["render_allowed"] is False, (
                f"{claim_type}: render_allowed should be False on v3 dataset"
            )
            assert claim["suppression_reason"] == "unverified_evidence"
            assert claim["denominator"] is not None and claim["denominator"] > 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_invalid_window_returns_422(monkeypatch):
    pool = await _new_pool()
    try:
        app = _make_app(pool, monkeypatch)
        async with _async_client(app) as client:
            too_small = await client.get(
                "/b2b/vendor-claims/ClickUp?analysis_window_days=1"
            )
            too_large = await client.get(
                "/b2b/vendor-claims/ClickUp?analysis_window_days=9999"
            )
        assert too_small.status_code == 422
        assert too_large.status_code == 422
    finally:
        await pool.close()
