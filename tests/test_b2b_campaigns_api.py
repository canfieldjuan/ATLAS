import pytest

from atlas_brain.api import b2b_campaigns as mod


@pytest.mark.asyncio
async def test_review_candidates_endpoint_uses_helper(monkeypatch):
    captured = {}

    async def _fake_list_candidates(
        pool,
        *,
        min_score,
        max_score,
        limit,
        vendor_filter=None,
        company_filter=None,
        qualified_only=True,
        ignore_recent_dedup=False,
    ):
        captured.update({
            "pool": pool,
            "min_score": min_score,
            "max_score": max_score,
            "limit": limit,
            "vendor_filter": vendor_filter,
            "company_filter": company_filter,
            "qualified_only": qualified_only,
            "ignore_recent_dedup": ignore_recent_dedup,
        })
        return {"count": 1, "candidates": [{"company_name": "Acme Co"}], "summary": {"qualified": 1}}

    sentinel_pool = object()
    monkeypatch.setattr(mod, "_pool_or_503", lambda: sentinel_pool)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_campaign_generation.list_churning_company_review_candidates",
        _fake_list_candidates,
    )

    result = await mod.review_candidates(
        limit=25,
        min_score=56,
        max_score=68,
        vendor="CrowdStrike",
        company="Pax8",
        qualified_only=False,
        ignore_recent_dedup=True,
    )

    assert result["count"] == 1
    assert captured == {
        "pool": sentinel_pool,
        "min_score": 56,
        "max_score": 68,
        "limit": 25,
        "vendor_filter": "CrowdStrike",
        "company_filter": "Pax8",
        "qualified_only": False,
        "ignore_recent_dedup": True,
    }


@pytest.mark.asyncio
async def test_review_candidates_summary_returns_summary_only(monkeypatch):
    async def _fake_list_candidates(
        pool,
        *,
        min_score,
        max_score,
        limit,
        vendor_filter=None,
        company_filter=None,
        qualified_only=True,
        ignore_recent_dedup=False,
    ):
        return {
            "count": 3,
            "candidates": [{"company_name": "Acme Co"}],
            "summary": {"qualified": 2, "unqualified": 1},
        }

    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_campaign_generation.list_churning_company_review_candidates",
        _fake_list_candidates,
    )

    result = await mod.review_candidates_summary(
        min_score=55,
        max_score=69,
        vendor=None,
        company=None,
        ignore_recent_dedup=False,
    )

    assert result == {
        "count": 3,
        "summary": {"qualified": 2, "unqualified": 1},
    }
