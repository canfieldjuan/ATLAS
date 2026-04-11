"""Tests for the dashboard accounts-in-motion endpoint."""

import importlib
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, patch

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


def _transaction_context(conn):
    @asynccontextmanager
    async def _transaction():
        yield conn

    return _transaction()


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
async def test_list_company_signal_candidates_uses_analyst_review_bucket_by_default():
    pool = MagicMock()
    returned = [{"company": "Acme Corp", "candidate_bucket": "analyst_review"}]
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ) as scope_mock:
            with patch.object(
                b2b_dashboard,
                "read_company_signal_candidates",
                new=AsyncMock(return_value=returned),
            ) as read_mock:
                result = await b2b_dashboard.list_company_signal_candidates(
                    vendor_name=None,
                    company_name=None,
                    candidate_bucket="analyst_review",
                    review_status="pending",
                    canonical_gap_reason=None,
                    min_urgency=0,
                    min_confidence=None,
                    decision_makers_only=False,
                    signal_evidence_present=None,
                    window_days=90,
                    limit=50,
                    user=None,
                )

    assert result == {
        "candidates": returned,
        "count": 1,
        "candidate_bucket": "analyst_review",
        "review_status": "pending",
    }
    scope_mock.assert_awaited_once_with(pool, None)
    read_mock.assert_awaited_once_with(
        pool,
        window_days=90,
        vendor_name=None,
        company_name=None,
        scoped_vendors=None,
        candidate_bucket="analyst_review",
        review_status="pending",
        canonical_gap_reason=None,
        min_urgency=0,
        min_confidence=None,
        decision_makers_only=False,
        signal_evidence_present=None,
        limit=50,
    )


@pytest.mark.asyncio
async def test_list_company_signal_candidate_groups_uses_group_reader_by_default():
    pool = MagicMock()
    returned = [{"group_id": "group-1", "display_company": "Acme Corp"}]
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ) as scope_mock:
            with patch.object(
                b2b_dashboard,
                "read_company_signal_candidate_groups",
                new=AsyncMock(return_value=returned),
            ) as read_mock:
                result = await b2b_dashboard.list_company_signal_candidate_groups(
                    vendor_name=None,
                    company_name=None,
                    candidate_bucket="analyst_review",
                    review_status="pending",
                    canonical_gap_reason=None,
                    min_urgency=0,
                    min_confidence=None,
                    min_reviews=1,
                    decision_makers_only=False,
                    signal_evidence_present=None,
                    window_days=90,
                    limit=50,
                    user=None,
                )

    assert result == {
        "groups": returned,
        "count": 1,
        "candidate_bucket": "analyst_review",
        "review_status": "pending",
    }
    scope_mock.assert_awaited_once_with(pool, None)
    read_mock.assert_awaited_once_with(
        pool,
        window_days=90,
        vendor_name=None,
        company_name=None,
        scoped_vendors=None,
        candidate_bucket="analyst_review",
        review_status="pending",
        canonical_gap_reason=None,
        min_urgency=0,
        min_confidence=None,
        min_reviews=1,
        decision_makers_only=False,
        signal_evidence_present=None,
        limit=50,
    )


@pytest.mark.asyncio
async def test_get_company_signal_candidate_group_summary_uses_summary_reader():
    pool = MagicMock()
    returned = {
        "totals": {"total_groups": 3, "pending_groups": 2},
        "gap_reasons": [{"gap_reason": "low_confidence_low_trust_source", "group_count": 2, "review_count": 5}],
        "top_vendors": [{"vendor_name": "Zendesk", "group_count": 3, "review_count": 7, "pending_groups": 2, "canonical_ready_groups": 1}],
        "confidence_tiers": [{"confidence_tier": "low", "group_count": 3}],
    }
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ) as scope_mock:
            with patch.object(
                b2b_dashboard,
                "read_company_signal_candidate_group_summary",
                new=AsyncMock(return_value=returned),
            ) as read_mock:
                result = await b2b_dashboard.get_company_signal_candidate_group_summary(
                    vendor_name=None,
                    company_name=None,
                    candidate_bucket=None,
                    review_status=None,
                    canonical_gap_reason=None,
                    min_urgency=0,
                    min_confidence=None,
                    min_reviews=1,
                    decision_makers_only=False,
                    signal_evidence_present=None,
                    window_days=90,
                    top_n=10,
                    user=None,
                )

    assert result == {
        **returned,
        "candidate_bucket": None,
        "review_status": None,
    }
    scope_mock.assert_awaited_once_with(pool, None)
    read_mock.assert_awaited_once_with(
        pool,
        window_days=90,
        vendor_name=None,
        company_name=None,
        scoped_vendors=None,
        candidate_bucket=None,
        review_status=None,
        canonical_gap_reason=None,
        min_urgency=0,
        min_confidence=None,
        min_reviews=1,
        decision_makers_only=False,
        signal_evidence_present=None,
        top_n=10,
    )


@pytest.mark.asyncio
async def test_get_company_signal_candidate_group_summary_rejects_invalid_bucket():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with pytest.raises(b2b_dashboard.HTTPException) as exc:
            await b2b_dashboard.get_company_signal_candidate_group_summary(
                vendor_name=None,
                company_name=None,
                candidate_bucket="invalid",
                review_status=None,
                canonical_gap_reason=None,
                min_urgency=0,
                min_confidence=None,
                min_reviews=1,
                decision_makers_only=False,
                signal_evidence_present=None,
                window_days=90,
                top_n=10,
                user=None,
            )

    assert exc.value.status_code == 400
    assert "candidate_bucket" in exc.value.detail


@pytest.mark.asyncio
async def test_get_company_signal_review_impact_summary_uses_shared_reader():
    pool = MagicMock()
    returned = {
        "totals": {"total_actions": 4, "approvals": 3},
        "scopes": [{"review_scope": "group", "action_count": 4}],
        "priority_bands": [{"review_priority_band": "high", "action_count": 3}],
        "top_vendors": [{"vendor_name": "Zendesk", "action_count": 4}],
    }
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=["Zendesk"]),
        ) as scope_mock:
            with patch.object(
                b2b_dashboard,
                "read_company_signal_review_impact_summary",
                new=AsyncMock(return_value=returned),
            ) as read_mock:
                result = await b2b_dashboard.get_company_signal_review_impact_summary(
                    vendor_name="Zen",
                    review_action="approved",
                    window_days=14,
                    top_n=5,
                    user=MagicMock(),
                )

    assert result == {
        **returned,
        "review_action": "approved",
    }
    scope_mock.assert_awaited_once_with(pool, ANY)
    read_mock.assert_awaited_once_with(
        pool,
        window_days=14,
        vendor_name="Zen",
        scoped_vendors=["Zendesk"],
        review_action="approved",
        top_n=5,
    )


@pytest.mark.asyncio
async def test_get_company_signal_review_impact_summary_rejects_invalid_action():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with pytest.raises(b2b_dashboard.HTTPException) as exc:
            await b2b_dashboard.get_company_signal_review_impact_summary(
                vendor_name=None,
                review_action="invalid",
                window_days=30,
                top_n=10,
                user=None,
            )

    assert exc.value.status_code == 400
    assert "review_action" in exc.value.detail


@pytest.mark.asyncio
async def test_record_company_signal_review_event_persists_rebuild_outcome():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)

    await b2b_dashboard._record_company_signal_review_event(
        pool,
        review_batch_id="77777777-7777-7777-7777-777777777777",
        review_scope="group",
        review_action="approved",
        candidate_id=None,
        candidate_group_id="33333333-3333-3333-3333-333333333333",
        company_name="acme",
        vendor_name="Zendesk",
        reviewer="api:user-1",
        review_notes="approved cluster",
        rebuild_requested=True,
        rebuild={
            "triggered": True,
            "as_of": "2026-04-11",
            "persisted": 1,
            "total_accounts": 4,
            "vendors": 1,
        },
        review_priority_band="high",
        review_priority_reason="has_signal_evidence_and_decision_maker",
        company_signal_id="22222222-2222-2222-2222-222222222222",
        company_signal_action="created",
    )

    sql = pool.execute.call_args[0][0]
    args = pool.execute.call_args[0][1:]
    assert "INSERT INTO b2b_company_signal_review_events" in sql
    assert args[0] == "77777777-7777-7777-7777-777777777777"
    assert args[1] == "group"
    assert args[2] == "approved"
    assert args[4] == "33333333-3333-3333-3333-333333333333"
    assert args[9] == "high"
    assert args[10] == "has_signal_evidence_and_decision_maker"
    assert args[12] == "created"
    assert args[13] is True
    assert args[14] is True
    assert args[16] == "2026-04-11"
    assert args[17] == 1
    assert args[18] == 4
    assert args[19] == 1


@pytest.mark.asyncio
async def test_list_company_signal_candidates_passes_scope_and_filters():
    pool = MagicMock()
    user = MagicMock()
    scoped = ["Zendesk", "Freshdesk"]
    returned = [{"company": "Acme Corp", "candidate_bucket": "canonical_ready"}]
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=scoped),
        ):
            with patch.object(
                b2b_dashboard,
                "read_company_signal_candidates",
                new=AsyncMock(return_value=returned),
            ) as read_mock:
                result = await b2b_dashboard.list_company_signal_candidates(
                    vendor_name="Zendesk",
                    company_name="Acme",
                    candidate_bucket="canonical_ready",
                    review_status="approved",
                    canonical_gap_reason="below_high_intent_threshold",
                    min_urgency=6.5,
                    min_confidence=0.4,
                    decision_makers_only=True,
                    signal_evidence_present=False,
                    window_days=30,
                    limit=25,
                    user=user,
                )

    assert result["count"] == 1
    read_mock.assert_awaited_once_with(
        pool,
        window_days=30,
        vendor_name="Zendesk",
        company_name="Acme",
        scoped_vendors=scoped,
        candidate_bucket="canonical_ready",
        review_status="approved",
        canonical_gap_reason="below_high_intent_threshold",
        min_urgency=6.5,
        min_confidence=0.4,
        decision_makers_only=True,
        signal_evidence_present=False,
        limit=25,
    )


@pytest.mark.asyncio
async def test_list_company_signal_candidates_rejects_invalid_bucket():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with pytest.raises(b2b_dashboard.HTTPException) as exc:
            await b2b_dashboard.list_company_signal_candidates(
                vendor_name=None,
                company_name=None,
                candidate_bucket="invalid",
                review_status="pending",
                canonical_gap_reason=None,
                min_urgency=0,
                min_confidence=None,
                decision_makers_only=False,
                signal_evidence_present=None,
                window_days=90,
                limit=50,
                user=None,
            )

    assert exc.value.status_code == 400
    assert "candidate_bucket" in exc.value.detail


@pytest.mark.asyncio
async def test_list_company_signal_candidates_rejects_invalid_review_status():
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=MagicMock()):
        with pytest.raises(b2b_dashboard.HTTPException) as exc:
            await b2b_dashboard.list_company_signal_candidates(
                vendor_name=None,
                company_name=None,
                candidate_bucket="analyst_review",
                review_status="invalid",
                canonical_gap_reason=None,
                min_urgency=0,
                min_confidence=None,
                decision_makers_only=False,
                signal_evidence_present=None,
                window_days=90,
                limit=50,
                user=None,
            )

    assert exc.value.status_code == 400
    assert "review_status" in exc.value.detail


@pytest.mark.asyncio
async def test_approve_company_signal_candidate_promotes_and_triggers_rebuild():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=None)
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "review_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "company_name": "Acme Corp",
                "company_name_raw": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "source": "reddit",
                "urgency_score": 8.2,
                "pain_category": "pricing",
                "buyer_role": "vp",
                "decision_maker": True,
                "seat_count": 120,
                "contract_end": "2026-07-01",
                "buying_stage": "evaluation",
                "confidence_score": 0.26,
                "review_status": "pending",
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "company_name": "acme corp",
                "vendor_name": "Zendesk",
            },
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "review_status": "approved",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 0, tzinfo=timezone.utc),
                "reviewed_by": "api:user-1",
                "review_notes": "looks good",
            },
        ]
    )
    body = b2b_dashboard.CompanySignalCandidateReviewBody(notes="looks good", trigger_rebuild=True)
    user = MagicMock(user_id="user-1")
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ):
            with patch.object(
                b2b_dashboard,
                "_trigger_accounts_in_motion_rebuild",
                new=AsyncMock(return_value={"triggered": True, "vendor_name": "Zendesk"}),
            ) as rebuild_mock:
                result = await b2b_dashboard.approve_company_signal_candidate(
                    "11111111-1111-1111-1111-111111111111",
                    body,
                    user,
                )

    assert result["review_status"] == "approved"
    assert result["review_status_updated_at"] == "2026-04-11T12:00:00+00:00"
    assert result["company_signal_id"] == "22222222-2222-2222-2222-222222222222"
    assert result["company_signal_action"] == "created"
    assert result["rebuild"]["triggered"] is True
    rebuild_mock.assert_awaited_once_with(pool, "Zendesk")


@pytest.mark.asyncio
async def test_approve_company_signal_candidate_group_promotes_and_triggers_rebuild():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=None)
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "company_name": "acme",
                "display_company_name": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "review_count": 4,
                "max_urgency_score": 8.4,
                "corroborated_confidence_score": 0.66,
                "representative_review_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "representative_source": "reddit",
                "representative_pain_category": "pricing",
                "representative_buyer_role": "vp",
                "representative_decision_maker": True,
                "representative_seat_count": 120,
                "representative_contract_end": "2026-07-01",
                "representative_buying_stage": "evaluation",
                "review_status": "pending",
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "company_name": "acme",
                "vendor_name": "Zendesk",
            },
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "review_status": "approved",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 10, tzinfo=timezone.utc),
                "reviewed_by": "api:user-1",
                "review_notes": "cluster is real",
            },
        ]
    )
    body = b2b_dashboard.CompanySignalCandidateReviewBody(notes="cluster is real", trigger_rebuild=True)
    user = MagicMock(user_id="user-1")
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ):
            with patch.object(
                b2b_dashboard,
                "_trigger_accounts_in_motion_rebuild",
                new=AsyncMock(return_value={"triggered": True, "vendor_name": "Zendesk"}),
            ) as rebuild_mock:
                result = await b2b_dashboard.approve_company_signal_candidate_group(
                    "33333333-3333-3333-3333-333333333333",
                    body,
                    user,
                )

    assert result["review_status"] == "approved"
    assert result["company_signal_id"] == "22222222-2222-2222-2222-222222222222"
    assert result["company_signal_action"] == "created"
    assert result["review_count"] == 4
    assert result["rebuild"]["triggered"] is True
    rebuild_mock.assert_awaited_once_with(pool, "Zendesk")
    assert pool.execute.await_count == 2


@pytest.mark.asyncio
async def test_suppress_company_signal_candidate_marks_suppressed_without_rebuild():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "review_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "company_name": "Acme Corp",
                "company_name_raw": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "source": "reddit",
                "urgency_score": 8.2,
                "pain_category": "pricing",
                "buyer_role": "vp",
                "decision_maker": True,
                "seat_count": 120,
                "contract_end": "2026-07-01",
                "buying_stage": "evaluation",
                "confidence_score": 0.26,
                "review_status": "pending",
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "company_name": "Acme Corp",
                "vendor_name": "Zendesk",
            },
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "review_status": "suppressed",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 5, tzinfo=timezone.utc),
                "reviewed_by": "analyst",
                "review_notes": "not a real target",
            },
        ]
    )
    body = b2b_dashboard.CompanySignalCandidateReviewBody(notes="not a real target", trigger_rebuild=False)
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ):
            result = await b2b_dashboard.suppress_company_signal_candidate(
                "11111111-1111-1111-1111-111111111111",
                body,
                None,
            )

    assert result["review_status"] == "suppressed"
    assert result["review_status_updated_at"] == "2026-04-11T12:05:00+00:00"
    assert result["review_notes"] == "not a real target"
    assert result["retracted_company_signal_id"] == "22222222-2222-2222-2222-222222222222"
    assert result["company_signal_action"] == "deleted"
    assert result["rebuild"]["reason"] == "disabled"


@pytest.mark.asyncio
async def test_suppress_company_signal_candidate_group_marks_members_suppressed():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "company_name": "acme",
                "display_company_name": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "review_count": 4,
                "max_urgency_score": 8.4,
                "corroborated_confidence_score": 0.66,
                "representative_review_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "representative_source": "reddit",
                "representative_pain_category": "pricing",
                "representative_buyer_role": "vp",
                "representative_decision_maker": True,
                "representative_seat_count": 120,
                "representative_contract_end": "2026-07-01",
                "representative_buying_stage": "evaluation",
                "review_status": "pending",
            },
            {
                "id": "44444444-4444-4444-4444-444444444444",
                "company_name": "acme",
                "vendor_name": "Zendesk",
            },
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "review_status": "suppressed",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 15, tzinfo=timezone.utc),
                "reviewed_by": "analyst",
                "review_notes": "noise cluster",
            },
        ]
    )
    body = b2b_dashboard.CompanySignalCandidateReviewBody(notes="noise cluster", trigger_rebuild=False)
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ):
            result = await b2b_dashboard.suppress_company_signal_candidate_group(
                "33333333-3333-3333-3333-333333333333",
                body,
                None,
            )

    assert result["review_status"] == "suppressed"
    assert result["review_status_updated_at"] == "2026-04-11T12:15:00+00:00"
    assert result["review_count"] == 4
    assert result["retracted_company_signal_id"] == "44444444-4444-4444-4444-444444444444"
    assert result["company_signal_action"] == "deleted"
    assert result["rebuild"]["reason"] == "disabled"
    assert pool.execute.await_count == 2


@pytest.mark.asyncio
async def test_bulk_approve_company_signal_candidate_groups_promotes_and_rebuilds_per_vendor():
    pool = MagicMock()
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(side_effect=[None, None])
    conn.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "company_name": "acme",
                "display_company_name": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "review_count": 4,
                "max_urgency_score": 8.4,
                "corroborated_confidence_score": 0.66,
                "representative_review_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "representative_source": "reddit",
                "representative_pain_category": "pricing",
                "representative_buyer_role": "vp",
                "representative_decision_maker": True,
                "representative_seat_count": 120,
                "representative_contract_end": "2026-07-01",
                "representative_buying_stage": "evaluation",
                "review_status": "pending",
            },
            {
                "id": "44444444-4444-4444-4444-444444444444",
                "company_name": "beta",
                "display_company_name": "Beta Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "review_count": 3,
                "max_urgency_score": 7.8,
                "corroborated_confidence_score": 0.62,
                "representative_review_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "representative_source": "g2",
                "representative_pain_category": "support",
                "representative_buyer_role": "director",
                "representative_decision_maker": False,
                "representative_seat_count": 85,
                "representative_contract_end": "2026-10-01",
                "representative_buying_stage": "consideration",
                "review_status": "pending",
            },
            {
                "id": "55555555-5555-5555-5555-555555555555",
                "company_name": "acme",
                "vendor_name": "Zendesk",
            },
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "review_status": "approved",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 20, tzinfo=timezone.utc),
                "reviewed_by": "api:user-1",
                "review_notes": "bulk approve",
            },
            {
                "id": "66666666-6666-6666-6666-666666666666",
                "company_name": "beta",
                "vendor_name": "Zendesk",
            },
            {
                "id": "44444444-4444-4444-4444-444444444444",
                "review_status": "approved",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 21, tzinfo=timezone.utc),
                "reviewed_by": "api:user-1",
                "review_notes": "bulk approve",
            },
        ]
    )
    pool.transaction = MagicMock(return_value=_transaction_context(conn))
    body = b2b_dashboard.BulkCompanySignalCandidateGroupReviewBody(
        group_ids=[
            "33333333-3333-3333-3333-333333333333",
            "44444444-4444-4444-4444-444444444444",
        ],
        notes="bulk approve",
        trigger_rebuild=True,
    )
    user = MagicMock(user_id="user-1")
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ):
            with patch.object(
                b2b_dashboard,
                "_trigger_accounts_in_motion_rebuild",
                new=AsyncMock(return_value={"triggered": True, "vendor_name": "Zendesk"}),
            ) as rebuild_mock:
                result = await b2b_dashboard.approve_company_signal_candidate_groups(body, user)

    assert result["count"] == 2
    assert result["review_batch_id"]
    assert result["groups"][0]["company_signal_id"] == "55555555-5555-5555-5555-555555555555"
    assert result["groups"][1]["company_signal_id"] == "66666666-6666-6666-6666-666666666666"
    assert result["groups"][0]["company_signal_action"] == "created"
    assert result["groups"][1]["company_signal_action"] == "created"
    assert result["rebuilds"] == [{"vendor_name": "Zendesk", "triggered": True}]
    rebuild_mock.assert_awaited_once_with(pool, "Zendesk")
    assert conn.execute.await_count == 2


@pytest.mark.asyncio
async def test_bulk_suppress_company_signal_candidate_groups_retracts_and_rebuilds_per_vendor():
    pool = MagicMock()
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "company_name": "acme",
                "display_company_name": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "review_count": 4,
                "max_urgency_score": 8.4,
                "corroborated_confidence_score": 0.66,
                "representative_review_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "representative_source": "reddit",
                "representative_pain_category": "pricing",
                "representative_buyer_role": "vp",
                "representative_decision_maker": True,
                "representative_seat_count": 120,
                "representative_contract_end": "2026-07-01",
                "representative_buying_stage": "evaluation",
                "review_status": "approved",
            },
            {
                "id": "44444444-4444-4444-4444-444444444444",
                "company_name": "beta",
                "display_company_name": "Beta Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "review_count": 3,
                "max_urgency_score": 7.8,
                "corroborated_confidence_score": 0.62,
                "representative_review_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "representative_source": "g2",
                "representative_pain_category": "support",
                "representative_buyer_role": "director",
                "representative_decision_maker": False,
                "representative_seat_count": 85,
                "representative_contract_end": "2026-10-01",
                "representative_buying_stage": "consideration",
                "review_status": "approved",
            },
            {
                "id": "55555555-5555-5555-5555-555555555555",
                "company_name": "acme",
                "vendor_name": "Zendesk",
            },
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "review_status": "suppressed",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 22, tzinfo=timezone.utc),
                "reviewed_by": "api:user-1",
                "review_notes": "bulk suppress",
            },
            {
                "id": "66666666-6666-6666-6666-666666666666",
                "company_name": "beta",
                "vendor_name": "Zendesk",
            },
            {
                "id": "44444444-4444-4444-4444-444444444444",
                "review_status": "suppressed",
                "review_status_updated_at": datetime(2026, 4, 11, 12, 23, tzinfo=timezone.utc),
                "reviewed_by": "api:user-1",
                "review_notes": "bulk suppress",
            },
        ]
    )
    pool.transaction = MagicMock(return_value=_transaction_context(conn))
    body = b2b_dashboard.BulkCompanySignalCandidateGroupReviewBody(
        group_ids=[
            "33333333-3333-3333-3333-333333333333",
            "44444444-4444-4444-4444-444444444444",
        ],
        notes="bulk suppress",
        trigger_rebuild=True,
    )
    user = MagicMock(user_id="user-1")
    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with patch.object(
            b2b_dashboard,
            "_get_scoped_vendors",
            new=AsyncMock(return_value=None),
        ):
            with patch.object(
                b2b_dashboard,
                "_trigger_accounts_in_motion_rebuild",
                new=AsyncMock(return_value={"triggered": True, "vendor_name": "Zendesk"}),
            ) as rebuild_mock:
                result = await b2b_dashboard.suppress_company_signal_candidate_groups(body, user)

    assert result["count"] == 2
    assert result["review_batch_id"]
    assert result["groups"][0]["retracted_company_signal_id"] == "55555555-5555-5555-5555-555555555555"
    assert result["groups"][1]["retracted_company_signal_id"] == "66666666-6666-6666-6666-666666666666"
    assert result["groups"][0]["company_signal_action"] == "deleted"
    assert result["groups"][1]["company_signal_action"] == "deleted"
    assert result["rebuilds"] == [{"vendor_name": "Zendesk", "triggered": True}]
    rebuild_mock.assert_awaited_once_with(pool, "Zendesk")
    assert conn.execute.await_count == 2


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
                "key_insights": [{"label": "Pricing", "summary": "Pricing churn risk"}],
                "key_insights_reference_ids": {"witness_ids": ["w1"]},
                "recommended_plays": {"summary": "Lead with migration support"},
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
    assert result["has_pdf_export"] is True
    assert result["artifact_state"] == "ready"
    assert result["artifact_label"] == "Ready"
    assert result["freshness_state"] == "stale"
    assert result["review_state"] == "clean"
    assert result["section_evidence"] == {
        "key_insights": {
            "state": "witness_backed",
            "label": "Witness-backed",
            "detail": "1 linked witness citation",
            "witness_count": 1,
            "metric_count": 0,
        },
        "recommended_plays": {
            "state": "thin",
            "label": "Thin evidence",
            "detail": "No linked witness citations for this section yet.",
            "witness_count": 0,
            "metric_count": 0,
        },
    }
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
    assert report["has_pdf_export"] is True
    assert report["artifact_state"] == "ready"
    assert report["artifact_label"] == "Ready"
    assert report["freshness_state"] == "monitor"
    assert report["review_state"] == "warnings"
    assert report["trust"] == {
        "artifact_state": "ready",
        "artifact_label": "Ready",
        "freshness_state": "monitor",
        "freshness_label": "Monitor",
        "review_state": "warnings",
        "review_label": "Warnings",
    }


@pytest.mark.asyncio
async def test_list_webhooks_exposes_latest_failure_summary():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    failed_at = datetime.now(timezone.utc) - timedelta(hours=2)
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "id": "2ea3fd03-7fd9-4b72-8f24-117667f723e9",
                "url": "https://hooks.example.com/churn",
                "event_types": ["churn_alert", "signal_update"],
                "channel": "generic",
                "enabled": True,
                "description": "PagerDuty bridge",
                "created_at": created_at,
                "updated_at": created_at,
                "recent_deliveries": 12,
                "recent_successes": 11,
                "latest_failure_event_type": "signal_update",
                "latest_failure_status_code": 500,
                "latest_failure_error": "downstream timeout",
                "latest_failure_at": failed_at,
            }
        ]
    )
    user = MagicMock(account_id="account-1")

    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        result = await b2b_dashboard.list_webhooks(user=user)

    assert result["count"] == 1
    webhook = result["webhooks"][0]
    assert webhook["recent_success_rate_7d"] == 0.917
    assert webhook["latest_failure_event_type"] == "signal_update"
    assert webhook["latest_failure_status_code"] == 500
    assert webhook["latest_failure_error"] == "downstream timeout"
    assert webhook["latest_failure_at"] == failed_at.isoformat()


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
