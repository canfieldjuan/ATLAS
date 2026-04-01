import json
from datetime import datetime, timezone
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
        AsyncMock(return_value=(
            {
                "buyer_authority": {
                    "role_type": "economic_buyer",
                    "buying_stage": "renewal_decision",
                },
                "timeline": {"decision_timeline": "within_quarter"},
                "contract_context": {"contract_value_signal": "mid_market"},
            },
            "google/gemini-3.1-flash-lite-preview",
            False,
        )),
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
async def test_call_repair_extractor_uses_exact_cache_hit(monkeypatch):
    cached_response = {
        "response_text": json.dumps({"competitors_mentioned": ["HubSpot"]}),
        "model": "google/gemini-3.1-flash-lite-preview",
    }

    monkeypatch.setattr(
        repair_mod,
        "get_pipeline_llm",
        lambda **_kwargs: SimpleNamespace(
            name="openrouter",
            model="google/gemini-3.1-flash-lite-preview",
        ),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.llm_exact_cache.build_skill_request_envelope",
        lambda **_kwargs: ("cache-key", {"messages": [{"role": "user", "content": "cached"}]}, []),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.llm_exact_cache.lookup_cached_text",
        AsyncMock(return_value=cached_response),
    )

    def _unexpected_llm_call(*_args, **_kwargs):
        raise AssertionError("repair extractor should not run on exact-cache hit")

    monkeypatch.setattr(repair_mod, "call_llm_with_skill", _unexpected_llm_call)

    parsed, model, cache_hit = await repair_mod._call_repair_extractor(
        {"vendor_name": "Zendesk"},
        "google/gemini-3.1-flash-lite-preview",
        SimpleNamespace(enrichment_repair_max_tokens=512),
        include_cache_hit=True,
    )

    assert parsed == {"competitors_mentioned": ["HubSpot"]}
    assert model == "google/gemini-3.1-flash-lite-preview"
    assert cache_hit is True


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

    monkeypatch.setattr(
        repair_mod,
        "_call_repair_extractor",
        AsyncMock(return_value=({"event_mentions": []}, "google/gemini-3.1-flash-lite-preview", False)),
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_apply_field_repair",
        lambda baseline, repair_result: (baseline, []),
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
        AsyncMock(return_value=(
            {
                "buyer_authority": {"role_type": "end_user", "buying_stage": "unknown"},
                "timeline": {"decision_timeline": "unknown"},
                "contract_context": {"contract_value_signal": "unknown"},
            },
            "google/gemini-3.1-flash-lite-preview",
            False,
        )),
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
    assert args[7] is False
    assert json.loads(args[8]) == ["repair_shadowed_technical_source"]


@pytest.mark.asyncio
async def test_persist_shadow_result_quarantines_hard_gap_payload_without_explicit_shadow_reason():
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    review_id = uuid4()
    row = {
        "id": review_id,
        "source": "reddit",
        "enrichment_status": "enriched",
        "enrichment": {
            "replacement_mode": "",
            "operating_model_shift": "",
            "productivity_delta_claim": "",
            "org_pressure_type": "",
            "evidence_map_hash": "ab1aab98d2178a6d",
            "evidence_spans": None,
        },
    }

    result = await repair_mod._persist_shadow_result(
        pool,
        review_id=review_id,
        row=row,
        repair_result={"competitors_mentioned": []},
        model_id="google/gemini-3.1-flash-lite-preview",
        applied_fields=[],
        repaired_at=datetime.now(timezone.utc),
    )

    assert result == "shadowed"
    query = pool.execute.await_args.args[0]
    args = pool.execute.await_args.args
    assert "enrichment_repair_status = 'shadowed'" in query
    assert args[6] == "quarantined"
    assert args[7] is False
    assert json.loads(args[8]) == []


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
    assert "review_text ~* '(switched to|moved to|replaced with|migrating to|migration to|evaluating|looking at|considering" in query


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
async def test_quarantine_shadowed_hard_gap_rows_marks_shadowed_enriched_rows():
    pool = SimpleNamespace(
        execute=AsyncMock(return_value="UPDATE 5"),
    )

    quarantined = await repair_mod._quarantine_shadowed_hard_gap_rows(pool, limit=25)

    assert quarantined == 5
    query = pool.execute.await_args.args[0]
    assert "enrichment_status = 'quarantined'" in query
    assert "enrichment_repair_status = 'shadowed'" in query
    assert "status:quarantined_hard_gap_shadow" in query
    assert pool.execute.await_args.args[1] == 25
    assert pool.execute.await_args.args[2] is not None


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
    monkeypatch.setattr(
        repair_mod,
        "_quarantine_shadowed_hard_gap_rows",
        AsyncMock(return_value=4),
    )

    result = await repair_mod.run(_task())

    assert result["_skip_synthesis"] == "B2B enrichment repair complete"
    assert result["stale_no_signal_demoted"] == 3
    assert result["shadowed_hard_gap_quarantined"] == 4
    assert result["rounds"] == 0


@pytest.mark.asyncio
async def test_run_records_attempt_and_summary_event_for_non_clean_repair_run(monkeypatch):
    execution_id = str(uuid4())
    task = _task()
    task.metadata = {
        "builtin_handler": "b2b_enrichment_repair",
        "_execution_id": execution_id,
    }
    rows = [{
        "id": uuid4(),
        "vendor_name": "Smartsheet",
        "enrichment_repair_attempts": 0,
    }]
    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(return_value="UPDATE 0"),
        fetch=AsyncMock(side_effect=[rows, []]),
    )

    monkeypatch.setattr(repair_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_enabled", True, raising=False)
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_model",
        "google/gemini-3.1-flash-lite-preview",
        raising=False,
    )
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_attempts", 3, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_per_batch", 5, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_rounds_per_run", 1, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_concurrency", 1, raising=False)
    monkeypatch.setattr(repair_mod, "_demote_stale_no_signal_rows", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_quarantine_shadowed_hard_gap_rows", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_recover_orphaned_repairing", AsyncMock(return_value=0))
    monkeypatch.setattr(
        repair_mod,
        "_repair_rows",
        AsyncMock(return_value={
            "promoted": 0,
            "shadowed": 1,
            "failed": 0,
            "exact_cache_hits": 0,
            "generated": 1,
        }),
    )
    record_attempt = AsyncMock()
    emit_event = AsyncMock()
    monkeypatch.setattr("atlas_brain.autonomous.visibility.record_attempt", record_attempt)
    monkeypatch.setattr("atlas_brain.autonomous.visibility.emit_event", emit_event)

    result = await repair_mod.run(task)

    assert result["shadowed"] == 1
    assert record_attempt.await_count == 1
    assert emit_event.await_count == 1
    assert record_attempt.await_args.kwargs["run_id"] == execution_id
    assert record_attempt.await_args.kwargs["artifact_type"] == "enrichment_repair"
    assert record_attempt.await_args.kwargs["warning_count"] == 1
    assert emit_event.await_args.kwargs["run_id"] == execution_id
    assert emit_event.await_args.kwargs["event_type"] == "repair_run_summary"
    assert emit_event.await_args.kwargs["reason_code"] == "enrichment_repair_shadowed"
    assert emit_event.await_args.kwargs["update_review_state"] is False


def test_strategic_adjudication_reasons_detects_missing_witness_interfaces():
    row = {
        "summary": "Renewal is coming up and we're considering HubSpot for our team docs workflow.",
        "review_text": (
            "Acme Corp was quoted $29 per seat at renewal. We're considering HubSpot "
            "next quarter because we want docs instead of chat."
        ),
        "pros": "",
        "cons": "",
        "reviewer_company": "Acme Corp",
    }
    result = {
        "salience_flags": ["explicit_dollar"],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"contract_renewal_mentioned": True},
        "reviewer_context": {"company_name": "Acme Corp"},
        "competitors_mentioned": [{"name": "HubSpot"}],
        "evidence_spans": [
            {
                "signal_type": "complaint",
                "text": "quoted $29 per seat at renewal",
                "flags": [],
                "company_name": "",
                "time_anchor": "",
            }
        ],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "money_without_pricing_span" in reasons
    assert "competitor_without_displacement_framing" in reasons
    assert "named_company_without_named_account_evidence" in reasons
    assert "timeline_language_without_timing_anchor" in reasons
    assert "workflow_language_without_replacement_mode" in reasons


def test_build_repair_payload_includes_strategic_targets():
    row = {
        "vendor_name": "Slack",
        "summary": "Renewal pricing issue",
        "review_text": "Acme was quoted $29 and is evaluating Teams next quarter.",
        "pros": "",
        "cons": "",
        "reviewer_company": "Acme",
    }
    baseline = {
        "salience_flags": ["explicit_dollar"],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"contract_renewal_mentioned": True},
        "competitors_mentioned": [{"name": "Teams"}],
        "evidence_spans": [],
    }

    payload = repair_mod._build_repair_payload(row, baseline, 500)

    assert "pricing_phrases" in payload["target_fields"]
    assert "competitors_mentioned" in payload["target_fields"]
    assert "event_mentions" in payload["target_fields"]
    assert "money_without_pricing_span" in payload["strategic_adjudication_reasons"]


def test_strategic_adjudication_skips_neutral_competitor_comparisons_without_pressure():
    row = {
        "summary": "Should I use Azure DevOps or stick to GitHub?",
        "review_text": "Azure DevOps has higher limits than GitHub, but I do not want to waste the free Azure credits.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
        },
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [{"name": "GitHub", "evidence_type": "neutral_mention"}],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_roundup_style_comparisons_without_named_displacement():
    row = {
        "summary": "I went through hundreds of user reviews of project management tools, here's what actually matters.",
        "review_text": (
            "Asana is strong for structured teams and workflows. Monday.com has great UI. "
            "Notion is loved for docs plus project hybrid use. Trello falls short for growing teams."
        ),
        "pros": "",
        "cons": "",
        "reviewer_company": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "workflow_substitution",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
        },
        "specific_complaints": ["limited customization compared to others"],
        "pricing_phrases": [],
        "feature_gaps": ["limited customization compared to others"],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


@pytest.mark.asyncio
async def test_repair_single_shadows_with_adjudication_markers_when_no_llm_targets(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "vendor_name": "Notion",
        "review_text": "Acme Corp wants docs instead of chat, but the extraction still missed that shift.",
        "summary": "Acme Corp wants docs instead of chat",
        "source": "reddit",
        "rating": 3.0,
        "rating_max": 5,
        "product_name": "Notion",
        "product_category": "Knowledge Base",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "Acme Corp",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "enrichment": {
            "salience_flags": ["named_account"],
            "replacement_mode": "none",
            "timeline": {"decision_timeline": "unknown"},
            "churn_signals": {"contract_renewal_mentioned": False},
            "competitors_mentioned": [],
            "evidence_spans": [{"signal_type": "review_context", "company_name": "Acme Corp", "flags": ["named_org"]}],
        },
    }
    cfg = SimpleNamespace(
        enrichment_repair_model="google/gemini-3.1-flash-lite-preview",
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        enrichment_full_extraction_timeout_seconds=30,
    )

    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_repair_target_fields",
        lambda baseline, row=None: [],
    )

    result = await repair_mod._repair_single(pool, row, cfg, max_attempts=2)

    assert result == "shadowed"
    query = pool.execute.await_args.args[0]
    args = pool.execute.await_args.args
    assert "enrichment_repair_applied_fields = $5::jsonb" in query
    assert json.loads(args[5]) == ["adjudication:workflow_language_without_replacement_mode"]


@pytest.mark.asyncio
async def test_run_query_includes_strategic_adjudication_conditions(monkeypatch):
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

    result = await repair_mod.run(_task())

    assert result["_skip_synthesis"] == "No enriched reviews need repair"
    query = pool.fetch.await_args.args[0]
    assert "jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.signal_type == \"pricing_backlash\")')" in query
    assert "COALESCE(NULLIF(reviewer_company, ''), NULLIF(enrichment->'reviewer_context'->>'company_name', '')) IS NOT NULL" in query
    assert "lower(COALESCE(enrichment->>'replacement_mode', 'none')) = 'none'" in query
    assert "enrichment_repair_status = 'shadowed'" in query
    assert "enrichment_repair_status = 'promoted'" in query
    assert 'like_regex "^adjudication:"' in query
    assert "WHEN enrichment_repair_status IS NULL THEN 0" in query
    assert "WHEN enrichment_status = 'enriched' THEN 0" in query


@pytest.mark.asyncio
async def test_repair_single_persists_shadowed_enrichment_when_strategic_gap_remains(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "vendor_name": "Copper",
        "review_text": "We are considering HubSpot because Copper is not worth the money.",
        "summary": "Considering HubSpot",
        "source": "reddit",
        "rating": 2.0,
        "rating_max": 5,
        "product_name": "Copper",
        "product_category": "CRM",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "enrichment_status": "enriched",
        "enrichment": {
            "churn_signals": {"intent_to_leave": True},
            "urgency_score": 8,
            "competitors_mentioned": [{"name": "HubSpot"}],
            "evidence_spans": [],
            "replacement_mode": "none",
            "salience_flags": [],
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
        AsyncMock(return_value=(
            {"competitors_mentioned": [{"name": "HubSpot"}]},
            "google/gemini-3.1-flash-lite-preview",
            False,
        )),
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_apply_field_repair",
        lambda baseline, repair_result: ({**baseline, "specific_complaints": ["not worth the money"]}, ["specific_complaints"]),
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_compute_derived_fields",
        lambda result, source_row: dict(result),
    )
    monkeypatch.setattr(
        repair_mod.base_enrichment,
        "_validate_enrichment",
        lambda result, source_row=None: True,
    )

    calls = {"n": 0}

    def _strategic(result, source_row):
        calls["n"] += 1
        if calls["n"] == 1:
            return ["competitor_without_displacement_framing"]
        return ["competitor_without_displacement_framing"]

    monkeypatch.setattr(repair_mod, "_strategic_adjudication_reasons", _strategic)

    result = await repair_mod._repair_single(pool, row, cfg, max_attempts=2)

    assert result == "shadowed"
    query = pool.execute.await_args.args[0]
    args = pool.execute.await_args.args
    assert "enrichment = $2::jsonb" in query
    assert "enrichment_repair_status = 'shadowed'" in query
    assert args[7] == "quarantined"
    assert json.loads(args[9]) == ["competitor_without_displacement_framing"]


@pytest.mark.asyncio
async def test_run_scopes_rows_to_test_vendors(monkeypatch):
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

    task = type("Task", (), {"metadata": {"test_vendors": ["Zendesk", "Freshdesk"]}})()
    result = await repair_mod.run(task)

    assert result["_skip_synthesis"] == "No enriched reviews need repair"
    args = pool.fetch.await_args.args
    assert args[4] == ["zendesk", "freshdesk"]
