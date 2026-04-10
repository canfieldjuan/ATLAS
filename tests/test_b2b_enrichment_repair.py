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


def test_repair_batch_custom_id_is_anthropic_safe():
    assert repair_mod._repair_batch_custom_id("1234-5678") == "repair_1234-5678"


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
    assert args[7] is True
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
    assert "review_text ~* '(switched to|moved to|replaced with|migrating to|migration to)'" in query
    assert "review_text ~* '(evaluating|looking at|considering|shortlisting|shortlisted|poc with|proof of concept with)'" in query
    assert "jsonb_array_elements(COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb))" in query


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
        "source": "reddit",
        "content_type": "community_discussion",
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
    assert result["strict_discussion_candidates_kept"] == 1
    assert result["strict_discussion_candidates_dropped"] == 0
    assert record_attempt.await_count == 1
    assert emit_event.await_count == 1
    assert record_attempt.await_args.kwargs["run_id"] == execution_id
    assert record_attempt.await_args.kwargs["artifact_type"] == "enrichment_repair"
    assert record_attempt.await_args.kwargs["warning_count"] == 1
    assert emit_event.await_args.kwargs["run_id"] == execution_id
    assert emit_event.await_args.kwargs["event_type"] == "repair_run_summary"
    assert emit_event.await_args.kwargs["reason_code"] == "enrichment_repair_shadowed"
    assert emit_event.await_args.kwargs["update_review_state"] is False
    assert emit_event.await_args.kwargs["detail"]["strict_discussion_candidates_kept"] == 1
    assert emit_event.await_args.kwargs["detail"]["low_signal_discussion_skipped"] == 0


@pytest.mark.asyncio
async def test_skip_low_signal_strict_discussion_rows_marks_terminal_shadow(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 3"))

    result = await repair_mod._skip_low_signal_strict_discussion_rows(
        pool,
        strict_sources=["reddit"],
        strict_content_types=["comment", "community_discussion"],
        scoped_vendors=["slack"],
        max_attempts=2,
        limit=25,
    )

    assert result == 3
    query = pool.execute.await_args.args[0]
    args = pool.execute.await_args.args
    assert "enrichment_repair_status = 'shadowed'" in query
    assert args[4] == ["slack"]
    assert "repair_skipped_low_signal_discussion" in str(args[5])
    assert "NOT (" in query


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


def test_strategic_adjudication_skips_low_signal_money_discussion_noise():
    row = {
        "summary": "General reddit thread",
        "review_text": "I found a cheaper airport close to downtown and the ticket was $100 less.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "community_discussion",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": ["explicit_dollar"],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {},
        "pricing_phrases": [],
        "specific_complaints": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "money_without_pricing_span" not in reasons


def test_strategic_adjudication_keeps_real_pricing_gap():
    row = {
        "summary": "Renewal pricing issue",
        "review_text": "At renewal we were quoted $29 per seat and started evaluating alternatives.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "community_discussion",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": ["explicit_dollar"],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"actively_evaluating": True, "contract_renewal_mentioned": True},
        "pricing_phrases": [],
        "specific_complaints": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "money_without_pricing_span" in reasons


def test_strategic_adjudication_skips_low_signal_vendor_money_discussion_noise():
    row = {
        "summary": "How Amazon Really Makes Its Money",
        "review_text": "Amazon Web Services is investing $15 billion in a new data center campus in Indiana.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "community_discussion",
        "reviewer_title": "Repeat Churn Signal (Score: 8.8)",
        "vendor_name": "Amazon Web Services",
        "product_name": "Amazon Web Services",
        "source": "reddit",
    }
    result = {
        "salience_flags": ["explicit_dollar"],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
        },
        "reviewer_context": {"company_name": ""},
        "pricing_phrases": [],
        "specific_complaints": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "money_without_pricing_span" not in reasons


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


def test_strategic_adjudication_skips_soft_evaluation_without_displacement_context():
    row = {
        "summary": "Storage design question",
        "review_text": "I am re-evaluating my storage design philosophy for my next build.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "community_discussion",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"actively_evaluating": True},
        "competitors_mentioned": [],
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_keeps_soft_evaluation_with_displacement_context():
    row = {
        "summary": "CRM evaluation at renewal",
        "review_text": "We are evaluating alternative CRM platforms at renewal because Salesforce is too expensive.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "review",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"actively_evaluating": True, "contract_renewal_mentioned": True},
        "competitors_mentioned": [
            {"name": "HubSpot", "evidence_type": "neutral_mention", "displacement_confidence": "low"}
        ],
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" in reasons


def test_strategic_adjudication_skips_soft_evaluation_without_named_alternative():
    row = {
        "summary": "CRM evaluation at renewal",
        "review_text": "We are evaluating alternative CRM platforms at renewal because Salesforce is too expensive.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "review",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"actively_evaluating": True, "contract_renewal_mentioned": True},
        "competitors_mentioned": [],
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_positive_switch_without_named_competitor():
    row = {
        "summary": "Very good onboarding experience",
        "review_text": "We switched to Klaviyo a few weeks ago and already notice amazing results for a price that is only a portion of what we used to pay with our previous solution.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "review",
        "reviewer_title": "",
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
        "pricing_phrases": ["price that is only a portion of what we used to pay"],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_keeps_negative_switch_without_named_competitor():
    row = {
        "summary": "Migration due to failure",
        "review_text": "We switched to another provider because the platform kept failing and support was a nightmare.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "review",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "migration_in_progress": True,
            "contract_renewal_mentioned": False,
        },
        "specific_complaints": ["platform kept failing", "support was a nightmare"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" in reasons


def test_strategic_adjudication_skips_generic_switch_phrase_without_alternative_context():
    row = {
        "summary": "Assignment workflow issue",
        "review_text": "Anyway what Ginger wants is when you simply view the Case, the ownership of the Case is automatically switched to you. This is an impending disaster.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "review",
        "reviewer_title": "",
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
        "specific_complaints": ["This is an impending disaster"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_comment_switch_noise():
    row = {
        "vendor_name": "Fortinet",
        "product_name": "Fortinet",
        "summary": "Happy install base note",
        "review_text": "We have used fortinet firewalls forever. Two years ago we switched to their switches and this year started using some of their cloud software.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "comment",
        "source": "reddit",
        "reviewer_title": "",
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
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_reddit_vendor_ambiguity_noise():
    row = {
        "vendor_name": "Copper",
        "product_name": "Copper CRM",
        "summary": "General discussion",
        "review_text": "I am considering switching to a hormonal one because the copper one is causing issues.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "community_discussion",
        "source": "reddit",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": True},
        "competitors_mentioned": [],
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_self_referential_competitor_objects():
    row = {
        "vendor_name": "Salesforce",
        "product_name": "Salesforce",
        "summary": "Admin workflow complaint",
        "review_text": "We are evaluating options because Salesforce setup is too heavy.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "review",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"actively_evaluating": True},
        "specific_complaints": ["setup is too heavy"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [
            {"name": "Salesforce", "evidence_type": "implied_preference", "reason_category": ""}
        ],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_integration_artifact_competitor_objects():
    row = {
        "vendor_name": "Salesforce",
        "product_name": "Salesforce",
        "summary": "Outlook sync issues",
        "review_text": "We are considering alternatives because the Outlook sync is painful.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "review",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"actively_evaluating": True},
        "specific_complaints": ["Outlook sync is painful"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [
            {"name": "Outlook Integration", "evidence_type": "implied_preference", "reason_category": "integration"}
        ],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_raw_reviewer_company_without_trusted_company_context():
    row = {
        "vendor_name": "ActiveCampaign",
        "product_name": "ActiveCampaign",
        "summary": "Support complaint",
        "review_text": "Slow and frustrating support.",
        "pros": "",
        "cons": "",
        "reviewer_company": "Univera",
        "content_type": "review",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {},
        "specific_complaints": ["Slow and frustrating support"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "named_company_without_named_account_evidence" not in reasons


def test_strategic_adjudication_skips_vendor_employment_noise():
    row = {
        "vendor_name": "Salesforce",
        "product_name": "Salesforce",
        "summary": "Career discussion",
        "review_text": "I currently work at Salesforce and I am thinking about leaving my role for a smaller company.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "content_type": "community_discussion",
        "source": "reddit",
        "reviewer_title": "",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "churn_signals": {"intent_to_leave": True},
        "competitors_mentioned": [],
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_low_signal_community_discussion_competitor_noise():
    row = {
        "summary": "HubSpot vs. Salesforce vs. Pipedrive: Which CRM is Best for Small Teams?",
        "review_text": "I have been engaging in the CRM community for a while and am comparing options for small teams.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "community_discussion",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
        },
        "specific_complaints": ["can get expensive as you scale"],
        "pricing_phrases": ["pricey plans"],
        "feature_gaps": [],
        "competitors_mentioned": [
            {"name": "HubSpot", "evidence_type": "implied_preference", "displacement_confidence": "low"},
            {"name": "Pipedrive", "evidence_type": "neutral_mention", "displacement_confidence": "low"},
        ],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_keeps_real_churn_community_discussion_competitor_gap():
    row = {
        "summary": "We are considering HubSpot because Copper is not worth the money.",
        "review_text": "We are considering HubSpot because Copper is not worth the money and renewal is coming up.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "community_discussion",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "contract_renewal_mentioned": True,
        },
        "specific_complaints": ["not worth the money"],
        "pricing_phrases": ["not worth the money"],
        "feature_gaps": [],
        "competitors_mentioned": [
            {"name": "HubSpot", "evidence_type": "neutral_mention", "displacement_confidence": "low"}
        ],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" in reasons


def test_strategic_adjudication_skips_low_signal_insider_account_competitor_noise():
    row = {
        "summary": "Need some advice, 5 YOE and went through a layoff",
        "review_text": "I have software development experience and am considering career options after a layoff.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "insider_account",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
        },
        "specific_complaints": [],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "competitor_without_displacement_framing" not in reasons


def test_strategic_adjudication_skips_generic_timeline_discussion_noise():
    row = {
        "summary": "I want to send a push to customers who bought in the last 30 days.",
        "review_text": "I want to send a push to customers who bought in the last 30 days but have not opened the app in 2 weeks.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "community_discussion",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
        },
        "specific_complaints": ["retention tools are built for engineers"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "timeline_language_without_timing_anchor" not in reasons


def test_strategic_adjudication_keeps_real_timeline_churn_gap():
    row = {
        "summary": "We are switching next quarter if renewal pricing does not change.",
        "review_text": "We are evaluating alternatives and switching next quarter if renewal pricing does not change.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "community_discussion",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "contract_renewal_mentioned": True,
        },
        "specific_complaints": ["renewal pricing"],
        "pricing_phrases": ["renewal pricing"],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "timeline_language_without_timing_anchor" in reasons


def test_strategic_adjudication_keeps_notice_period_timeline_gap():
    row = {
        "summary": "Cancel before renewal",
        "review_text": "We have to give 30 days notice before renewal or the contract auto renews.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "review",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": True,
            "renewal_timing": "",
        },
        "specific_complaints": ["contract auto renews"],
        "pricing_phrases": [],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "timeline_language_without_timing_anchor" in reasons


def test_strategic_adjudication_skips_contract_renewal_without_explicit_timing_cue():
    row = {
        "summary": "Auto-renew complaint",
        "review_text": "They automatically renewed the contract and refused to refund us.",
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "review",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": True,
            "renewal_timing": "",
        },
        "specific_complaints": ["automatically renewed the contract"],
        "pricing_phrases": ["refused to refund us"],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "timeline_language_without_timing_anchor" not in reasons


def test_strategic_adjudication_skips_generic_task_deadline_reviews():
    row = {
        "summary": "Project tracking with better deadlines",
        "review_text": (
            "It helps us manage project deadlines and plugin costs, but this is still just "
            "our day-to-day work tracking system."
        ),
        "pros": "",
        "cons": "",
        "reviewer_company": "",
        "reviewer_title": "",
        "content_type": "review",
    }
    result = {
        "salience_flags": [],
        "replacement_mode": "none",
        "timeline": {"decision_timeline": "unknown"},
        "reviewer_context": {"company_name": ""},
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": "",
        },
        "specific_complaints": ["plugin costs keep going up"],
        "pricing_phrases": ["plugin costs keep going up"],
        "feature_gaps": [],
        "competitors_mentioned": [],
        "evidence_spans": [],
    }

    reasons = repair_mod._strategic_adjudication_reasons(result, row)

    assert "timeline_language_without_timing_anchor" not in reasons


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
    (
        query,
        max_attempts,
        max_batch,
        trusted_sources,
        scoped_vendors,
        excluded_sources,
        strict_discussion_sources,
        strict_discussion_content_types,
    ) = pool.fetch.await_args.args
    assert "jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.signal_type == \"pricing_backlash\")')" in query
    assert "COALESCE(NULLIF(reviewer_company, ''), NULLIF(enrichment->'reviewer_context'->>'company_name', '')) IS NOT NULL" in query
    assert "lower(COALESCE(enrichment->>'replacement_mode', 'none')) = 'none'" in query
    assert "enrichment_repair_status = 'shadowed'" in query
    assert "enrichment_repair_status = 'promoted'" in query
    assert 'like_regex "^adjudication:"' in query
    assert "WHEN enrichment_repair_status IS NULL THEN 0" in query
    assert "WHEN enrichment_status = 'enriched' THEN 0" in query
    assert "lower(source) <> ALL($5::text[])" in query
    assert "lower(source) <> ALL($6::text[])" in query
    assert "lower(COALESCE(content_type, '')) <> ALL($7::text[])" in query
    assert "review_text ILIKE ('%' || vendor_name || '%')" in query
    assert "trustradius" not in trusted_sources
    assert "trustpilot" in excluded_sources
    assert strict_discussion_sources == ["reddit"]
    assert strict_discussion_content_types == ["comment", "community_discussion", "insider_account"]


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


@pytest.mark.asyncio
async def test_run_scopes_strict_discussion_gate_from_config(monkeypatch):
    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(return_value="UPDATE 0"),
        fetch=AsyncMock(return_value=[]),
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
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_strict_discussion_sources",
        "reddit,hackernews",
        raising=False,
    )
    monkeypatch.setattr(
        repair_mod.settings.b2b_churn,
        "enrichment_repair_strict_discussion_content_types",
        ["community_discussion", "comment"],
        raising=False,
    )

    result = await repair_mod.run(_task())

    assert result["_skip_synthesis"] == "No enriched reviews need repair"
    args = pool.fetch.await_args.args
    assert args[6] == ["hackernews", "reddit"]
    assert args[7] == ["comment", "community_discussion"]


@pytest.mark.asyncio
async def test_repair_rows_uses_anthropic_batch_when_enabled(monkeypatch):
    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    row_id = uuid4()
    row = {
        "id": row_id,
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Help Desk",
        "source": "reddit",
        "summary": "Pricing issue",
        "review_text": "Pricing increased and we are considering alternatives.",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "company_size_raw": "",
        "reviewer_industry": "",
        "content_type": "review",
        "raw_metadata": {},
        "enrichment": {
            "specific_complaints": [],
            "competitors_mentioned": [],
            "pricing_phrases": [],
            "recommendation_language": [],
            "feature_gaps": [],
            "event_mentions": [],
        },
        "enrichment_repair_attempts": 0,
    }
    cfg = SimpleNamespace(
        enrichment_repair_max_attempts=2,
        enrichment_repair_concurrency=2,
        enrichment_repair_model="anthropic/claude-haiku-4-5",
        enrichment_repair_max_tokens=256,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
        enrichment_low_fidelity_enabled=False,
        enrichment_repair_anthropic_batch_enabled=True,
    )
    task = _task()
    task.metadata["anthropic_batch_enabled"] = True
    task.metadata["enrichment_repair_anthropic_batch_enabled"] = True
    pool = SimpleNamespace()
    persist = AsyncMock(return_value="promoted")

    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: FakeAnthropicLLM(),
    )
    monkeypatch.setattr(repair_mod, "_repair_target_fields", lambda *_args, **_kwargs: ["specific_complaints"])
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_skill_stage_request",
        lambda *args, **_kwargs: (
            SimpleNamespace(namespace="ns", request_envelope={}, provider="openrouter", model="anthropic/claude-haiku-4-5"),
            [
                {"role": "system", "content": "repair prompt"},
                {"role": "user", "content": "{}"},
            ],
        ),
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        AsyncMock(
            return_value=SimpleNamespace(
                local_batch_id="batch-1",
                provider_batch_id="provider-batch-1",
                submitted_items=1,
                cache_prefiltered_items=0,
                fallback_single_call_items=0,
                completed_items=1,
                failed_items=0,
                results_by_custom_id={
                    repair_mod._repair_batch_custom_id(row_id): SimpleNamespace(
                        response_text=json.dumps({"specific_complaints": ["pricing pressure"]}),
                        cached=False,
                        usage={},
                        error_text=None,
                    )
                },
            )
        ),
    )
    monkeypatch.setattr(repair_mod, "_persist_repair_result", persist)

    result = await repair_mod._repair_rows([row], cfg, pool, run_id="run-1", task=task)

    assert result["promoted"] == 1
    assert result["anthropic_batch_jobs"] == 1
    assert result["anthropic_batch_items_submitted"] == 1
    assert persist.await_count == 1


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_high_failure_rate(monkeypatch):
    """Loop should break immediately when >50% of a round fails."""
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_enabled", True, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_model", "test-model", raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_per_batch", 10, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_rounds_per_run", 5, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_attempts", 3, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_concurrency", 2, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_strict_discussion_skip_limit", 100, raising=False)

    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(return_value="UPDATE 0"),
        fetch=AsyncMock(side_effect=[
            # Round 1: return rows
            [{"id": uuid4(), "vendor_name": "V", "source": "g2", "content_type": "review",
              "enrichment": {}, "enrichment_repair_attempts": 0}],
            # Round 2: would return rows, but should not be reached
            [],
        ]),
    )
    monkeypatch.setattr(repair_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(repair_mod, "_recover_orphaned_repairing", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_demote_stale_no_signal_rows", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_quarantine_shadowed_hard_gap_rows", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_skip_low_signal_strict_discussion_rows", AsyncMock(return_value=0))

    # _repair_rows returns mostly failures
    monkeypatch.setattr(
        repair_mod, "_repair_rows",
        AsyncMock(return_value={"promoted": 0, "shadowed": 1, "failed": 8,
                                "exact_cache_hits": 0, "generated": 9,
                                "witness_rows": 0, "witness_count": 0}),
    )

    result = await repair_mod.run(_task())

    assert result["rounds"] == 1
    assert result["circuit_breaker_reason"] is not None
    assert "high failure rate" in result["circuit_breaker_reason"]


@pytest.mark.asyncio
async def test_circuit_breaker_consecutive_no_progress(monkeypatch):
    """Loop should break after 2 consecutive rounds with zero promotions."""
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_enabled", True, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_model", "test-model", raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_per_batch", 10, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_rounds_per_run", 10, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_max_attempts", 3, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_concurrency", 2, raising=False)
    monkeypatch.setattr(repair_mod.settings.b2b_churn, "enrichment_repair_strict_discussion_skip_limit", 100, raising=False)

    call_count = 0

    async def _mock_fetch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return [{"id": uuid4(), "vendor_name": "V", "source": "g2", "content_type": "review",
                      "enrichment": {}, "enrichment_repair_attempts": 0}]
        return []

    pool = SimpleNamespace(
        is_initialized=True,
        execute=AsyncMock(return_value="UPDATE 0"),
        fetch=AsyncMock(side_effect=_mock_fetch),
    )
    monkeypatch.setattr(repair_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(repair_mod, "_recover_orphaned_repairing", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_demote_stale_no_signal_rows", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_quarantine_shadowed_hard_gap_rows", AsyncMock(return_value=0))
    monkeypatch.setattr(repair_mod, "_skip_low_signal_strict_discussion_rows", AsyncMock(return_value=0))

    # All rounds: shadowed only, zero promoted
    monkeypatch.setattr(
        repair_mod, "_repair_rows",
        AsyncMock(return_value={"promoted": 0, "shadowed": 1, "failed": 0,
                                "exact_cache_hits": 0, "generated": 1,
                                "witness_rows": 0, "witness_count": 0}),
    )

    result = await repair_mod.run(_task())

    # Should break after round 2 (2 consecutive no-progress), not go to 10
    assert result["rounds"] == 2
    assert result["circuit_breaker_reason"] is not None
    assert "no promotions" in result["circuit_breaker_reason"]
