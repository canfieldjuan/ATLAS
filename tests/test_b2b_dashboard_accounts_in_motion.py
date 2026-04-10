"""Tests for the dashboard accounts-in-motion endpoint."""

import importlib
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "PIL",
    "PIL.Image",
    "numpy",
    "cv2",
    "sounddevice",
    "soundfile",
    "playwright",
    "playwright.async_api",
    "playwright_stealth",
    "curl_cffi",
    "curl_cffi.requests",
    "pytrends",
    "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

b2b_dashboard = importlib.import_module("atlas_brain.api.b2b_dashboard")


@pytest.mark.asyncio
async def test_list_accounts_in_motion_uses_persisted_report_only():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with patch.object(
            b2b_dashboard,
            "_validate_accounts_in_motion_window",
        ):
            with patch.object(
                b2b_dashboard,
                "_list_accounts_in_motion_from_report",
                new=AsyncMock(return_value={"data_source": "persisted_report", "accounts": [], "count": 0}),
            ) as report_mock:
                with patch.object(
                    b2b_dashboard,
                    "_list_accounts_in_motion_from_reviews",
                    new=AsyncMock(return_value={"data_source": "live_reviews"}),
                ) as reviews_mock:
                    result = await b2b_dashboard.list_accounts_in_motion("Zendesk", 5, 30, 25, None)
    assert result["data_source"] == "persisted_report"
    report_mock.assert_awaited_once()
    reviews_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_accounts_in_motion_raises_when_persisted_missing():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with patch.object(
            b2b_dashboard,
            "_validate_accounts_in_motion_window",
        ):
            with patch.object(
                b2b_dashboard,
                "_list_accounts_in_motion_from_report",
                new=AsyncMock(return_value=None),
            ):
                with patch.object(
                    b2b_dashboard,
                    "has_complete_core_run_marker",
                    new=AsyncMock(return_value=True),
                ):
                    with pytest.raises(b2b_dashboard.HTTPException) as exc:
                        await b2b_dashboard.list_accounts_in_motion("Zendesk", 5, 30, 25, None)
    assert exc.value.status_code == 404
    assert exc.value.detail == "No persisted accounts-in-motion report found for that vendor"


@pytest.mark.asyncio
async def test_list_accounts_in_motion_explains_incomplete_core_when_report_missing():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with patch.object(
            b2b_dashboard,
            "_validate_accounts_in_motion_window",
        ):
            with patch.object(
                b2b_dashboard,
                "_list_accounts_in_motion_from_report",
                new=AsyncMock(return_value=None),
            ):
                with patch.object(
                    b2b_dashboard,
                    "has_complete_core_run_marker",
                    new=AsyncMock(return_value=False),
                ):
                    with pytest.raises(b2b_dashboard.HTTPException) as exc:
                        await b2b_dashboard.list_accounts_in_motion("Zendesk", 5, 30, 25, None)
    assert exc.value.status_code == 404
    assert "core churn materialization is incomplete" in exc.value.detail


@pytest.mark.asyncio
async def test_list_accounts_in_motion_live_uses_raw_review_path():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with patch.object(
            b2b_dashboard,
            "_list_accounts_in_motion_from_reviews",
            new=AsyncMock(return_value={"data_source": "live_reviews", "count": 1}),
        ) as reviews_mock:
            result = await b2b_dashboard.list_accounts_in_motion_live("Zendesk", 5, 90, 25, None)
    assert result["data_source"] == "live_reviews"
    reviews_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_report_handles_null_battle_card_quality():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "2ea3fd03-7fd9-4b72-8f24-117667f723e9",
            "report_date": "2026-03-22",
            "report_type": "battle_card",
            "vendor_filter": "Zendesk",
            "category_filter": "Help Desk",
            "executive_summary": "summary",
            "intelligence_data": {
                "quality_status": None,
                "battle_card_quality": None,
            },
            "data_density": {"status": "ok"},
            "status": "completed",
            "llm_model": "pipeline_deterministic",
            "created_at": "2026-03-22T18:00:00",
            "latest_failure_step": None,
            "latest_error_code": None,
            "latest_error_summary": None,
            "blocker_count": 0,
            "warning_count": 0,
            "account_id": None,
        }
    )
    pool.fetchval = AsyncMock(return_value=0)

    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        result = await b2b_dashboard.get_report("2ea3fd03-7fd9-4b72-8f24-117667f723e9", None)

    assert result["quality_status"] is None
    assert result["quality_score"] is None
    assert result["report_type"] == "battle_card"
    assert result["artifact_state"] == "ready"
    assert result["artifact_label"] == "Ready"
    assert result["freshness_state"] == "stale"
    assert result["review_state"] == "clean"
    assert result["trust"]["artifact_state"] == "ready"


@pytest.mark.asyncio
async def test_list_reports_exposes_normalized_trust_fields():
    created_at = datetime.now(timezone.utc) - timedelta(hours=8)
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": "2ea3fd03-7fd9-4b72-8f24-117667f723e9",
                "report_date": created_at.date(),
                "report_type": "vendor_scorecard",
                "executive_summary": "summary",
                "vendor_filter": "Zendesk",
                "category_filter": None,
                "status": "published",
                "created_at": created_at,
                "data_stale": False,
                "latest_failure_step": None,
                "latest_error_code": None,
                "latest_error_summary": None,
                "blocker_count": 0,
                "warning_count": 1,
                "unresolved_issue_count": 0,
                "quality_status": None,
                "quality_score": None,
            }
        ]
    )

    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        result = await b2b_dashboard.list_reports(
            report_type=None,
            vendor_filter=None,
            include_stale=False,
            limit=10,
            user=None,
        )

    assert result["count"] == 1
    report = result["reports"][0]
    assert report["artifact_state"] == "ready"
    assert report["artifact_label"] == "Ready"
    assert report["freshness_state"] == "fresh"
    assert report["review_state"] == "warnings"
    assert report["trust"] == {
        "artifact_state": "ready",
        "artifact_label": "Ready",
        "freshness_state": "fresh",
        "freshness_label": "Fresh",
        "review_state": "warnings",
        "review_label": "Warnings",
    }


def test_validate_accounts_in_motion_window_rejects_custom_window():
    configured = b2b_dashboard.settings.b2b_churn.intelligence_window_days
    with pytest.raises(b2b_dashboard.HTTPException) as exc:
        b2b_dashboard._validate_accounts_in_motion_window(configured + 1)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_list_accounts_in_motion_from_report_shapes_and_enriches():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "report_date": "2026-03-18",
            "vendor_filter": "Zendesk",
            "intelligence_data": {
                "vendor": "Zendesk",
                "reference_ids": {
                    "metric_ids": ["metric:zendesk:1"],
                    "witness_ids": ["witness:zendesk:1"],
                },
                "accounts": [
                    {
                        "company": "Acme Corp",
                        "vendor": "Zendesk",
                        "category": "Helpdesk",
                        "urgency": 8.5,
                        "role_level": "executive",
                        "decision_maker": True,
                        "buying_stage": "evaluation",
                        "pain_category": "pricing",
                        "top_quote": "We need to move fast.",
                        "contract_end": "Q3 2026",
                        "title": "VP Support",
                        "company_size": "500",
                        "quality_flags": [],
                        "opportunity_score": 82,
                        "quote_match_type": "company_match",
                        "confidence": 7.5,
                        "source_distribution": {"reddit": 2},
                        "last_seen": "2026-03-17T10:00:00",
                        "alternatives_considering": ["Freshdesk"],
                        "source_reviews": ["c6581fe1-32a9-4dbc-96cb-f7ef55707001"],
                        "evidence_count": 2,
                    }
                ],
            },
        }
    )
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "company_name_norm": "acme corp",
                    "employee_count": 500,
                    "industry": "SaaS",
                    "annual_revenue_range": "$10M-$50M",
                    "domain": "acme.com",
                }
            ],
            [
                {
                    "company_key": "acme corp",
                    "name": "Taylor Smith",
                    "title": "VP Support",
                    "seniority": "vp",
                    "email": "taylor@acme.com",
                    "linkedin_url": "https://linkedin.com/in/taylor",
                }
            ],
            [
                {
                    "id": "c6581fe1-32a9-4dbc-96cb-f7ef55707001",
                    "source": "reddit",
                    "source_url": "https://reddit.example/acme",
                    "vendor_name": "Zendesk",
                    "rating": 2.0,
                    "summary": "Support is slipping",
                    "review_excerpt": "We need to move fast before renewal.",
                    "reviewer_name": "Taylor",
                    "reviewer_title": "VP Support",
                    "reviewer_company": "Acme Corp",
                    "reviewed_at": datetime(2026, 3, 16, 0, 0, 0),
                }
            ],
        ]
    )

    result = await b2b_dashboard._list_accounts_in_motion_from_report(pool, "Zendesk", 5, 25, None)

    assert result["data_source"] == "persisted_report"
    assert result["vendor"] == "Zendesk"
    assert result["count"] == 1
    assert "stale_days" in result
    assert "is_stale" in result
    assert isinstance(result["is_stale"], bool)
    account = result["accounts"][0]
    assert account["company"] == "Acme Corp"
    assert account["pain_categories"] == [{"category": "pricing", "severity": ""}]
    assert account["evidence"] == [
        "We need to move fast before renewal.",
        "We need to move fast.",
    ]
    assert account["alternatives_considering"] == [{"name": "Freshdesk", "reason": ""}]
    assert account["domain"] == "acme.com"
    assert account["industry"] == "SaaS"
    assert account["contact_count"] == 1
    assert account["contacts"][0]["email"] == "taylor@acme.com"
    assert account["source_review_ids"] == ["c6581fe1-32a9-4dbc-96cb-f7ef55707001"]
    assert account["evidence_count"] == 2
    assert account["source_reviews"][0]["source"] == "reddit"
    assert account["source_reviews"][0]["review_excerpt"] == "We need to move fast before renewal."
    assert account["reasoning_reference_ids"]["witness_ids"] == ["witness:zendesk:1"]


def test_accounts_in_motion_report_freshness_treats_published_as_healthy():
    status, reason, timestamp = b2b_dashboard._accounts_in_motion_report_freshness(
        {
            "status": "published",
            "latest_failure_step": None,
            "latest_error_summary": None,
            "created_at": "2026-04-07T10:00:00Z",
        },
        report_date="2026-04-07",
        stale_days=0,
    )

    assert status == "fresh"
    assert reason is None
    assert timestamp == "2026-04-07"


@pytest.mark.asyncio
async def test_list_accounts_in_motion_from_reviews_enriches_in_bulk():
    pool = MagicMock()
    pool.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "reviewer_company": "Acme Corp",
                    "vendor_name": "Zendesk",
                    "product_category": "Helpdesk",
                    "urgency": 8.0,
                    "role_type": "executive",
                    "buying_stage": "evaluation",
                    "budget_authority": True,
                    "pain_categories": [{"category": "pricing", "severity": "high"}],
                    "quotable_phrases": ["Switching soon"],
                    "competitors": [{"name": "Freshdesk", "reason": "pricing"}],
                    "contract_signal": "renewal",
                    "reviewer_title": "VP Support",
                    "company_size_raw": "500",
                    "industry": "SaaS",
                    "enriched_at": "2026-03-17T00:00:00",
                }
            ],
            [
                {
                    "company_name_norm": "acme corp",
                    "employee_count": 500,
                    "industry": "SaaS",
                    "annual_revenue_range": "$10M-$50M",
                    "domain": "acme.com",
                }
            ],
            [
                {
                    "company_key": "acme corp",
                    "name": "Taylor Smith",
                    "title": "VP Support",
                    "seniority": "vp",
                    "email": "taylor@acme.com",
                    "linkedin_url": "https://linkedin.com/in/taylor",
                }
            ],
        ]
    )

    result = await b2b_dashboard._list_accounts_in_motion_from_reviews(
        pool,
        "Zendesk",
        min_urgency=5,
        window_days=90,
        limit=25,
        user=None,
    )

    assert result["data_source"] == "live_reviews"
    assert result["count"] == 1
    account = result["accounts"][0]
    assert account["contacts"][0]["email"] == "taylor@acme.com"
    assert account["domain"] == "acme.com"
    assert account["alternatives_considering"] == [{"name": "Freshdesk", "reason": "pricing"}]
