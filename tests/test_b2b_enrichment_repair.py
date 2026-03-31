import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import b2b_enrichment_repair as repair_mod
from atlas_brain.storage.models import ScheduledTask


def _task() -> ScheduledTask:
    return ScheduledTask(
        id=uuid4(),
        name="b2b_enrichment_repair",
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=900,
        enabled=True,
        metadata={"builtin_handler": "b2b_enrichment_repair"},
    )


@pytest.mark.asyncio
async def test_run_skips_when_repair_disabled(monkeypatch):
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_enabled",
        False,
        raising=False,
    )

    result = await repair_mod.run(_task())

    assert result["_skip_synthesis"] == "B2B enrichment repair disabled"


@pytest.mark.asyncio
async def test_repair_single_promotes_structural_fields(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "vendor_name": "Example",
        "review_text": "Renewal pricing is forcing us to reevaluate." * 4,
        "summary": "Renewal pricing issue",
        "source": "reddit",
        "rating": 2.0,
        "rating_max": 5,
        "product_name": "Example",
        "product_category": "CRM",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "enrichment": {
            "churn_signals": {"intent_to_leave": True, "actively_evaluating": False},
            "urgency_score": 7,
            "buyer_authority": {"role_type": "unknown", "buying_stage": "unknown"},
            "timeline": {"decision_timeline": "unknown"},
            "contract_context": {"contract_value_signal": "unknown"},
            "reviewer_context": {"role_level": "unknown", "decision_maker": False},
            "sentiment_trajectory": {"direction": "declining"},
        },
    }
    cfg = SimpleNamespace(
        enrichment_repair_model="google/gemini-3.1-flash-lite-preview",
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        enrichment_full_extraction_timeout_seconds=30,
        enrichment_low_fidelity_enabled=False,
    )

    monkeypatch.setattr(
        repair_mod,
        "_call_repair_extractor",
        AsyncMock(return_value=({
            "buyer_authority": {
                "role_type": "economic_buyer",
                "buying_stage": "renewal_decision",
            },
            "timeline": {"decision_timeline": "within_quarter"},
            "contract_context": {"contract_value_signal": "mid_market"},
        }, "google/gemini-3.1-flash-lite-preview")),
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_validate_enrichment",
        lambda result, source_row=None: True,
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_apply_field_repair",
        lambda baseline, repair_result: (
            {
                **baseline,
                "competitors_mentioned": ["HubSpot"],
                "specific_complaints": ["Renewal pricing increased"],
            },
            ["competitors_mentioned", "specific_complaints"],
        ),
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_needs_field_repair",
        lambda baseline, row=None: True,
    )

    result = await repair_mod._repair_single(pool, row, cfg, max_attempts=2)

    assert result == "promoted"
    query = pool.execute.await_args.args[0]
    assert "enrichment_baseline = COALESCE(enrichment_baseline, enrichment)" in query
    assert "enrichment_repair_status = 'promoted'" in query


@pytest.mark.asyncio
async def test_repair_single_shadows_when_nothing_promotable(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "vendor_name": "Example",
        "review_text": "We may consider options next quarter." * 4,
        "summary": "Potential evaluation",
        "source": "reddit",
        "rating": 3.0,
        "rating_max": 5,
        "product_name": "Example",
        "product_category": "CRM",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "enrichment": {
            "churn_signals": {"intent_to_leave": True, "actively_evaluating": True},
            "urgency_score": 7,
            "buyer_authority": {"role_type": "economic_buyer", "buying_stage": "evaluation"},
            "timeline": {"decision_timeline": "within_quarter"},
            "contract_context": {"contract_value_signal": "mid_market"},
            "reviewer_context": {"role_level": "executive", "decision_maker": True},
            "sentiment_trajectory": {"direction": "declining"},
        },
    }
    cfg = SimpleNamespace(
        enrichment_repair_model="google/gemini-3.1-flash-lite-preview",
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        enrichment_full_extraction_timeout_seconds=30,
    )

    result = await repair_mod._repair_single(pool, row, cfg, max_attempts=2)

    assert result == "shadowed"
    query = pool.execute.await_args.args[0]
    assert "enrichment_repair_status = 'shadowed'" in query


@pytest.mark.asyncio
async def test_repair_single_quarantines_shadowed_technical_source(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "vendor_name": "Intercom",
        "review_text": "How can I integrate Intercom in Xamarin.Forms?" * 4,
        "summary": "How can I integrate Intercom in Xamarin.Forms?",
        "source": "stackoverflow",
        "rating": 2.0,
        "rating_max": 5,
        "product_name": "Intercom",
        "product_category": "Support",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "enrichment": {
            "churn_signals": {"intent_to_leave": False, "actively_evaluating": False},
            "urgency_score": 2,
            "buyer_authority": {"role_type": "end_user", "buying_stage": "unknown"},
            "timeline": {"decision_timeline": "unknown"},
            "contract_context": {"contract_value_signal": "unknown"},
            "reviewer_context": {"role_level": "unknown", "decision_maker": False},
            "sentiment_trajectory": {"direction": "declining"},
        },
    }
    cfg = SimpleNamespace(
        enrichment_repair_model="google/gemini-3.1-flash-lite-preview",
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        enrichment_full_extraction_timeout_seconds=30,
    )

    monkeypatch.setattr(
        repair_mod,
        "_call_repair_extractor",
        AsyncMock(return_value=({
            "buyer_authority": {"role_type": "end_user", "buying_stage": "unknown"},
            "timeline": {"decision_timeline": "unknown"},
            "contract_context": {"contract_value_signal": "unknown"},
        }, "google/gemini-3.1-flash-lite-preview")),
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_needs_field_repair",
        lambda baseline, row=None: True,
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_apply_field_repair",
        lambda baseline, repair_result: (baseline, []),
    )

    result = await repair_mod._repair_single(pool, row, cfg, max_attempts=2)

    assert result == "shadowed"
    query = pool.execute.await_args.args[0]
    args = pool.execute.await_args.args
    assert "enrichment_status = $6" in query
    assert args[6] == "quarantined"
    assert args[7] is True
    assert json.loads(args[8]) == ["repair_shadowed_technical_source"]


@pytest.mark.asyncio
async def test_run_claim_query_requires_pressure_signal(monkeypatch):
    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(return_value="UPDATE 0"),
        fetch=AsyncMock(return_value=[]),
    )

    monkeypatch.setattr(repair_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_model",
        "google/gemini-3.1-flash-lite-preview",
        raising=False,
    )
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_min_urgency",
        3,
        raising=False,
    )

    result = await repair_mod.run(_task())

    assert result["_skip_synthesis"] == "No enriched reviews need repair"
    query = pool.fetch.await_args.args[0]
    assert "COALESCE(enrichment->>'pain_category', 'overall_dissatisfaction')" in query
    assert "review_text ~* '(cancel|cancellation|billing dispute|refund denied" in query
    assert "review_text ~* '(switched to|moved to|replaced with|evaluating|looking at|considering" in query


@pytest.mark.asyncio
async def test_demote_stale_no_signal_rows_marks_generic_empty_rows(monkeypatch):
    review_id = uuid4()
    row = {
        "id": review_id,
        "vendor_name": "Salesforce",
        "product_name": "Salesforce",
        "product_category": "CRM",
        "source": "reddit",
        "raw_metadata": {},
        "rating": None,
        "rating_max": None,
        "summary": "Career advice thread",
        "review_text": "How long should a Salesforce admin resume be?",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "community_discussion",
        "enrichment": {
            "pain_category": "overall_dissatisfaction",
            "competitors_mentioned": [],
            "specific_complaints": [],
            "quotable_phrases": [],
            "pricing_phrases": [],
            "recommendation_language": [],
            "feature_gaps": [],
            "event_mentions": [],
        },
    }
    pool = SimpleNamespace(
        fetch=AsyncMock(return_value=[row]),
        execute=AsyncMock(return_value="UPDATE 1"),
    )

    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_is_no_signal_result",
        lambda enrichment, source_row=None: True,
    )

    demoted = await repair_mod._demote_stale_no_signal_rows(pool, limit=25)

    assert demoted == 1
    query = pool.execute.await_args.args[0]
    assert "enrichment_status = 'no_signal'" in query
    assert pool.execute.await_args.args[1] is not None


@pytest.mark.asyncio
async def test_run_returns_demotion_result_when_no_repair_rows(monkeypatch):
    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(return_value="UPDATE 0"),
        fetch=AsyncMock(return_value=[]),
    )

    monkeypatch.setattr(repair_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_enabled",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_model",
        "google/gemini-3.1-flash-lite-preview",
        raising=False,
    )
    monkeypatch.setattr(
        repair_mod,
        "_demote_stale_no_signal_rows",
        AsyncMock(return_value=3),
    )

    result = await repair_mod.run(_task())

    assert result["_skip_synthesis"] == "B2B enrichment repair complete"
    assert result["stale_no_signal_demoted"] == 3
    assert result["rounds"] == 0
