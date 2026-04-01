import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import b2b_enrichment
from atlas_brain.reasoning import evidence_engine
from atlas_brain.storage.models import ScheduledTask


def _task() -> ScheduledTask:
    return ScheduledTask(
        id=uuid4(),
        name="b2b_enrichment",
        task_type="builtin",
        schedule_type="interval",
        interval_seconds=300,
        enabled=True,
        metadata={"builtin_handler": "b2b_enrichment"},
    )


class _Pool:
    def __init__(self, batches):
        self.is_initialized = True
        self.fetch = AsyncMock(side_effect=batches)
        self.fetchval = AsyncMock(return_value=0)
        self.execute = AsyncMock(return_value="UPDATE 0")


def test_normalize_pain_category_maps_legacy_other_to_overall_dissatisfaction():
    assert b2b_enrichment._normalize_pain_category("other") == "overall_dissatisfaction"
    assert b2b_enrichment._normalize_pain_category("general_dissatisfaction") == "overall_dissatisfaction"


def test_normalize_pain_category_accepts_new_specific_buckets():
    assert b2b_enrichment._normalize_pain_category("admin_burden") == "admin_burden"
    assert b2b_enrichment._normalize_pain_category("integration_debt") == "integration_debt"


def test_derive_competitor_annotations_prunes_generic_provider_labels():
    row = {
        "vendor_name": "Amazon Web Services",
        "summary": "AWS Failed Me",
        "review_text": "We may need a competing provider after this outage.",
        "pros": "",
        "cons": "",
    }
    result = {
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": False},
        "competitors_mentioned": [{"name": "competing provider"}],
    }

    derived = b2b_enrichment._derive_competitor_annotations(result, row)

    assert derived == []


def test_derive_competitor_annotations_prunes_weak_neutral_mentions_without_named_context():
    row = {
        "vendor_name": "ActiveCampaign",
        "summary": "Pricing and support issues",
        "review_text": "Their pricing is outrageous and support is nonexistent. We need a better answer fast.",
        "pros": "",
        "cons": "",
    }
    result = {
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": False},
        "competitors_mentioned": [{"name": "HubSpot"}],
    }

    derived = b2b_enrichment._derive_competitor_annotations(result, row)

    assert derived == []


def test_repair_target_fields_skips_roundup_style_competitor_repair_without_named_displacement():
    row = {
        "source": "reddit",
        "enrichment_status": "enriched",
        "summary": "I went through hundreds of user reviews of project management tools, here's what actually matters.",
        "review_text": (
            "Asana is strong for structured teams and workflows. Monday.com has great UI. "
            "Notion is loved for docs plus project hybrid use. Trello falls short for growing teams."
        ),
        "pros": "",
        "cons": "",
    }
    result = {
        "pain_category": "overall_dissatisfaction",
        "salience_flags": [],
        "competitors_mentioned": [],
        "specific_complaints": ["limited customization compared to others"],
        "pricing_phrases": [],
        "recommendation_language": [],
        "feature_gaps": ["limited customization compared to others"],
        "event_mentions": [],
        "timeline": {"decision_timeline": "unknown"},
    }

    targets = b2b_enrichment._repair_target_fields(result, row)

    assert "competitors_mentioned" not in targets


def test_is_no_signal_result_accepts_empty_community_discussion_without_rating():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "support_escalation": False,
            "contract_renewal_mentioned": False,
        },
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "event_mentions": [],
        "feature_gaps": [],
    }
    row = {
        "content_type": "community_discussion",
        "rating": None,
    }

    assert b2b_enrichment._is_no_signal_result(result, row) is True


@pytest.mark.asyncio
async def test_run_limits_rounds_and_reports_orphan_recovery(monkeypatch):
    rows = [{"id": uuid4(), "enrichment_attempts": 0}]
    pool = _Pool([rows, rows])

    monkeypatch.setattr(b2b_enrichment, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        b2b_enrichment,
        "_recover_orphaned_enriching",
        AsyncMock(return_value=5),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_queue_version_upgrades",
        AsyncMock(return_value=2),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_enrich_rows",
        AsyncMock(return_value={"enriched": 3, "failed": 1, "no_signal": 2}),
    )

    cfg = b2b_enrichment.settings.b2b_churn
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "enrichment_max_per_batch", 10)
    monkeypatch.setattr(cfg, "enrichment_max_attempts", 3)
    monkeypatch.setattr(cfg, "enrichment_max_rounds_per_run", 1)
    monkeypatch.setattr(cfg, "enrichment_priority_sources", "stackoverflow,github")

    result = await b2b_enrichment.run(_task())

    assert result["rounds"] == 1
    assert result["orphaned_requeued"] == 5
    assert result["version_upgrade_requeued"] == 2
    assert result["enriched"] == 3
    assert pool.fetch.await_count == 1
    fetch_query, max_attempts, max_batch, priority_sources = pool.fetch.await_args.args
    assert "source = ANY($3::text[])" in fetch_query
    assert "imported_at DESC" in fetch_query
    assert max_attempts == 3
    assert max_batch == 10
    assert priority_sources == ["stackoverflow", "github"]


@pytest.mark.asyncio
async def test_run_applies_manual_metadata_overrides(monkeypatch):
    rows = [{"id": uuid4(), "enrichment_attempts": 0}]
    pool = _Pool([rows, []])
    task = _task()
    task.metadata = {
        "builtin_handler": "b2b_enrichment",
        "enrichment_max_per_batch": 25,
        "enrichment_max_rounds_per_run": 3,
        "enrichment_concurrency": 17,
    }

    monkeypatch.setattr(b2b_enrichment, "get_db_pool", lambda: pool)
    monkeypatch.setattr(
        b2b_enrichment,
        "_recover_orphaned_enriching",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_queue_version_upgrades",
        AsyncMock(return_value=0),
    )
    enrich_rows = AsyncMock(return_value={"enriched": 1, "failed": 0, "no_signal": 0})
    monkeypatch.setattr(b2b_enrichment, "_enrich_rows", enrich_rows)

    cfg = b2b_enrichment.settings.b2b_churn
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "enrichment_max_per_batch", 10)
    monkeypatch.setattr(cfg, "enrichment_max_attempts", 3)
    monkeypatch.setattr(cfg, "enrichment_max_rounds_per_run", 1)
    monkeypatch.setattr(cfg, "enrichment_concurrency", 10)
    monkeypatch.setattr(cfg, "enrichment_priority_sources", "stackoverflow,github")

    result = await b2b_enrichment.run(task)

    assert result["rounds"] == 1
    fetch_query, max_attempts, max_batch, priority_sources = pool.fetch.await_args_list[0].args
    assert "source = ANY($3::text[])" in fetch_query
    assert max_attempts == 3
    assert max_batch == 25
    assert priority_sources == ["stackoverflow", "github"]
    assert enrich_rows.await_args.kwargs["concurrency_override"] == 17


@pytest.mark.asyncio
async def test_recover_orphaned_enriching_parses_update_count():
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 7"))

    count = await b2b_enrichment._recover_orphaned_enriching(pool, 3)

    assert count == 7
    query = pool.execute.await_args.args[0]
    assert "WHERE enrichment_status = 'enriching'" in query


@pytest.mark.asyncio
async def test_enrich_rows_uses_configured_concurrency(monkeypatch):
    active = 0
    max_seen = 0

    async def _fake_enrich_single(pool, row, max_attempts, local_only, max_tokens, truncate_length):
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.01)
        active -= 1
        return True

    monkeypatch.setattr(b2b_enrichment, "_enrich_single", _fake_enrich_single)

    rows = [{"id": uuid4(), "enrichment_attempts": 0} for _ in range(5)]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
    )
    pool = SimpleNamespace(
        fetchval=AsyncMock(return_value=0),
        fetch=AsyncMock(return_value=[{"enrichment_status": "enriched", "ct": 5}]),
    )

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool)

    assert result["enriched"] == 5
    assert max_seen == 2


def test_get_base_enrichment_llm_uses_vllm_first(monkeypatch):
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return object()

    with patch("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm):
        llm = b2b_enrichment._get_base_enrichment_llm(local_only=False)

    assert llm is not None
    assert calls == [{
        "workload": "vllm",
        "try_openrouter": False,
        "auto_activate_ollama": False,
    }]


def test_get_base_enrichment_llm_respects_local_only(monkeypatch):
    calls = []

    def _fake_get_pipeline_llm(**kwargs):
        calls.append(kwargs)
        return None

    with patch("atlas_brain.pipelines.llm.get_pipeline_llm", _fake_get_pipeline_llm):
        llm = b2b_enrichment._get_base_enrichment_llm(local_only=True)

    assert llm is None
    assert calls == [{
        "workload": "vllm",
        "try_openrouter": False,
        "auto_activate_ollama": False,
    }]


@pytest.mark.asyncio
async def test_call_vllm_tier1_uses_exact_cache_hit(monkeypatch):
    class _Registry:
        def get(self, name):
            if name == "digest/b2b_churn_extraction_tier1":
                return SimpleNamespace(content="tier1")
            return None

    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _Registry())
    monkeypatch.setattr(
        b2b_enrichment,
        "_lookup_cached_json_response",
        AsyncMock(
            return_value=(
                {"specific_complaints": ["support delays"], "churn_signals": {"actively_evaluating": True}},
                {"messages": [{"role": "user", "content": "cached"}]},
            )
        ),
    )
    client = SimpleNamespace(
        post=AsyncMock(side_effect=AssertionError("tier1 HTTP should not run on exact-cache hit"))
    )
    cfg = SimpleNamespace(
        enrichment_tier1_model="qwen3-30b",
        enrichment_tier1_max_tokens=512,
    )

    parsed, model = await b2b_enrichment._call_vllm_tier1(
        json.dumps({"vendor_name": "Zendesk"}),
        cfg,
        client,
    )

    assert parsed["specific_complaints"] == ["support delays"]
    assert parsed["churn_signals"]["actively_evaluating"] is True
    assert model == "qwen3-30b"
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_openrouter_tier2_uses_exact_cache_hit(monkeypatch):
    class _Registry:
        def get(self, name):
            if name == "digest/b2b_churn_extraction_tier2":
                return SimpleNamespace(content="tier2")
            return None

    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _Registry())
    monkeypatch.setattr(
        b2b_enrichment,
        "_lookup_cached_json_response",
        AsyncMock(
            return_value=(
                {"pain_categories": [{"category": "pricing", "severity": "primary"}]},
                {"messages": [{"role": "user", "content": "cached"}]},
            )
        ),
    )
    cfg = SimpleNamespace(
        openrouter_api_key="test-key",
        enrichment_tier2_openrouter_model="anthropic/claude-haiku-4-5",
        enrichment_openrouter_model="openai/gpt-oss-120b",
        enrichment_tier2_max_tokens=512,
        enrichment_tier2_timeout_seconds=30.0,
    )
    row = {
        "vendor_name": "Zendesk",
        "product_name": "Zendesk",
        "product_category": "Helpdesk",
        "source": "g2",
        "raw_metadata": {},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
        "summary": "Pricing issues",
        "review_text": "We are evaluating alternatives because pricing keeps rising.",
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Support",
        "reviewer_company": "Acme",
        "company_size_raw": "201-500",
        "reviewer_industry": "SaaS",
    }
    tier1_result = {
        "specific_complaints": ["pricing keeps rising"],
        "quotable_phrases": ["evaluating alternatives"],
    }

    parsed, model = await b2b_enrichment._call_openrouter_tier2(
        tier1_result,
        row,
        cfg,
        truncate_length=3000,
    )

    assert parsed["pain_categories"] == [{"category": "pricing", "severity": "primary"}]
    assert model == "anthropic/claude-haiku-4-5"


@pytest.mark.asyncio
async def test_enrich_single_uses_single_pass_tier1_only(monkeypatch):
    pool = SimpleNamespace(execute=AsyncMock(return_value="UPDATE 1"))
    row = {
        "id": uuid4(),
        "source": "reddit",
        "enrichment_attempts": 0,
        "vendor_name": "Example",
        "product_name": "Example Product",
        "product_category": "CRM",
        "raw_metadata": {},
        "rating_max": 5,
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Sales",
        "reviewer_company": "Acme",
        "company_size_raw": "1001-5000",
        "reviewer_industry": "Technology",
        "content_type": "review",
        "summary": "Switching evaluation",
        "review_text": "We are actively evaluating alternatives after support issues." * 4,
        "rating": 2.0
    }
    tier1_result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {"role_level": "director", "decision_maker": True},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["support issues"],
        "quotable_phrases": ["actively evaluating alternatives"],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "urgency_indicators": {},
        "sentiment_trajectory": {},
        "buyer_authority": {},
        "timeline": {},
        "contract_context": {},
        "insider_signals": None,
    }

    tier1_call = AsyncMock(return_value=(tier1_result, "vllm-model"))
    monkeypatch.setattr(
        b2b_enrichment,
        "_get_tier1_client",
        lambda cfg: object(),
    )
    monkeypatch.setattr(
        b2b_enrichment,
        "_call_vllm_tier1",
        tier1_call,
    )
    monkeypatch.setattr(b2b_enrichment, "_validate_enrichment", lambda result, source_row=None: True)
    monkeypatch.setattr(b2b_enrichment, "_notify_high_urgency", AsyncMock(return_value=None))

    ok = await b2b_enrichment._enrich_single(
        pool,
        row,
        max_attempts=3,
        local_only=True,
        max_tokens=512,
    )

    assert ok is True
    tier1_call.assert_awaited_once()


def test_detect_low_fidelity_reasons_flags_vendor_absent_noisy_source(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter",
        raising=False,
    )
    row = {
        "source": "hackernews",
        "vendor_name": "Shopify",
        "product_name": "Shopify",
        "summary": "I switched from Spotify back to Apple Music",
        "review_text": "Spotify playback is better. Apple Music failed in the car.",
        "pros": "",
        "cons": "",
    }
    result = {
        "competitors_mentioned": [{"name": "Spotify"}],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "vendor_absent_noisy_source" in reasons
    assert "competitor_only_context" in reasons


def test_detect_low_fidelity_reasons_keeps_vendor_present_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter",
        raising=False,
    )
    row = {
        "source": "hackernews",
        "vendor_name": "Intercom",
        "product_name": "Intercom",
        "summary": "We use Intercom for support and need secure file uploads",
        "review_text": "Intercom file limits are a long-term issue for our support workflow.",
        "pros": "",
        "cons": "",
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, {"competitors_mentioned": []})

    assert reasons == []


def test_detect_low_fidelity_reasons_flags_technical_stackoverflow_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "stackoverflow,github",
        raising=False,
    )
    row = {
        "source": "stackoverflow",
        "vendor_name": "Intercom",
        "product_name": "Intercom",
        "summary": "How can I integrate Intercom in Xamarin.Forms?",
        "review_text": "I am trying to add Intercom in Xamarin.Forms for iOS and things are not going well.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 2,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "technical_question_context" in reasons


def test_detect_low_fidelity_reasons_keeps_commercial_stackoverflow_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "stackoverflow,github",
        raising=False,
    )
    row = {
        "source": "stackoverflow",
        "vendor_name": "Jira",
        "product_name": "Jira",
        "summary": "Looking for Jira alternatives before renewal",
        "review_text": "We are evaluating alternatives to Jira because renewal pricing doubled for our team.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 7,
        "competitors_mentioned": [{"name": "Linear"}],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "technical_question_context" not in reasons


def test_detect_low_fidelity_reasons_flags_consumer_trustpilot_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter,github,stackoverflow",
        raising=False,
    )
    row = {
        "source": "trustpilot",
        "vendor_name": "Microsoft Teams",
        "product_name": "Microsoft Teams",
        "summary": "Emailed copilot via app ghosting Email address on Google play",
        "review_text": "This app downloaded Microsoft 365 was a free version. Contacted app support via app and got a ghosting email from Google Play.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 3,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "consumer_support_context" in reasons


def test_detect_low_fidelity_reasons_keeps_commercial_trustpilot_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter,github,stackoverflow",
        raising=False,
    )
    row = {
        "source": "trustpilot",
        "vendor_name": "HubSpot",
        "product_name": "HubSpot",
        "summary": "Great product, but every wiggle cost money",
        "review_text": "HubSpot has powerful tools but the costs climb steeply and support wants to charge more for every change.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 4,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "consumer_support_context" not in reasons


def test_detect_low_fidelity_reasons_keeps_trustpilot_vendor_absent_false_positive(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "hackernews,quora,twitter,github,stackoverflow",
        raising=False,
    )
    row = {
        "source": "trustpilot",
        "vendor_name": "RingCentral",
        "product_name": "RingCentral",
        "summary": "Ring Central's servicing is very poor",
        "review_text": "Ring Central's servicing is very poor. They commit to having a supervisor call to assist with unclear contract terms with no follow through.",
        "pros": "",
        "cons": "",
    }
    result = {
        "urgency_score": 4,
        "competitors_mentioned": [],
    }

    reasons = b2b_enrichment._detect_low_fidelity_reasons(row, result)

    assert "vendor_absent_noisy_source" not in reasons
    assert "consumer_support_context" not in reasons


def test_text_mentions_name_matches_compact_vendor_with_spaced_text():
    haystack = b2b_enrichment._normalize_compare_text(
        "Ring Central's servicing is very poor and support did not clarify the contract."
    )

    assert b2b_enrichment._text_mentions_name(haystack, "RingCentral") is True


def test_apply_structural_repair_promotes_only_unknown_fields():
    baseline = {
        "urgency_score": 8,
        "churn_signals": {"intent_to_leave": True, "actively_evaluating": True},
        "buyer_authority": {"role_type": "unknown", "buying_stage": "unknown"},
        "timeline": {"decision_timeline": "unknown"},
        "contract_context": {"contract_value_signal": "unknown", "usage_duration": None},
    }
    repair = {
        "urgency_score": 2,
        "churn_signals": {"intent_to_leave": False, "actively_evaluating": False},
        "buyer_authority": {"role_type": "economic_buyer", "buying_stage": "renewal_decision"},
        "timeline": {"decision_timeline": "within_quarter"},
        "contract_context": {"contract_value_signal": "enterprise_mid", "usage_duration": "2_years"},
    }

    merged, applied = b2b_enrichment._apply_structural_repair(baseline, repair)

    assert merged["urgency_score"] == 8
    assert merged["churn_signals"]["intent_to_leave"] is True
    assert merged["buyer_authority"]["role_type"] == "economic_buyer"
    assert merged["timeline"]["decision_timeline"] == "within_quarter"
    assert merged["contract_context"]["contract_value_signal"] == "enterprise_mid"
    assert "buyer_authority.role_type" in applied
    assert "timeline.decision_timeline" in applied
    assert "contract_context.contract_value_signal" in applied


def test_derive_decision_timeline_uses_raw_text_when_commercial_context_exists():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": ["pricing pressure"],
        "specific_complaints": ["We need to decide next quarter"],
        "competitors_mentioned": [],
    }
    row = {
        "summary": "Renewal decision coming soon",
        "review_text": "Pricing pressure means we need to decide next quarter whether to switch.",
        "pros": "",
        "cons": "",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "within_quarter"


def test_derive_decision_timeline_ignores_generic_deadline_without_commercial_context():
    result = {
        "churn_signals": {},
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": [],
        "specific_complaints": [],
        "competitors_mentioned": [],
    }
    row = {
        "summary": "Helpful for daily work",
        "review_text": "This helps me track tasks and deadlines every week for my team.",
        "pros": "",
        "cons": "",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "unknown"


def test_derive_decision_timeline_ignores_soft_deadline_complaint_without_strong_commercial_signal():
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": [],
        "specific_complaints": ["The reminders can get annoying."],
        "competitors_mentioned": [],
    }
    row = {
        "summary": "Helpful but noisy reminders",
        "review_text": "It reminds us about deadlines and action items all the time, which can get annoying.",
        "pros": "",
        "cons": "",
        "source": "software_advice",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "unknown"


def test_derive_decision_timeline_ignores_ambiguous_noisy_source_without_vendor_context(monkeypatch):
    monkeypatch.setattr(
        b2b_enrichment.settings.b2b_churn,
        "enrichment_low_fidelity_noisy_sources",
        "reddit,quora,twitter",
    )
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "contract_renewal_mentioned": False,
            "renewal_timing": None,
        },
        "timeline": {"contract_end": None, "evaluation_deadline": None},
        "event_mentions": [],
        "pricing_phrases": [],
        "specific_complaints": ["The zipper design is weak."],
        "competitors_mentioned": [],
    }
    row = {
        "vendor_name": "Copper",
        "product_name": "Copper",
        "summary": "Thinking about replacing my Copper Spur tent next quarter",
        "review_text": "I am considering other tent options next quarter because the zipper design is weak.",
        "pros": "",
        "cons": "",
        "source": "reddit",
        "content_type": "community_discussion",
    }

    assert b2b_enrichment._derive_decision_timeline(result, row) == "unknown"


def test_compute_derived_fields_promotes_contract_notice_into_evaluation_deadline(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 7.5

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return True

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "summary": "Cancel before renewal",
        "review_text": (
            "We need to give 30 days notice before renewal or they auto renew the contract. "
            "Support refused to help us cancel."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "Operations Manager",
        "reviewer_company": "Acme Corp",
        "raw_metadata": {"source_weight": 0.8},
        "content_type": "review",
        "rating": 1.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "contract_renewal_mentioned": True,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "manager",
            "decision_maker": False,
            "company_name": "Acme Corp",
        },
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["Support refused to help us cancel."],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["timeline"]["evaluation_deadline"] == "30 days"
    assert derived["timeline"]["decision_timeline"] == "within_quarter"
    assert any(span["time_anchor"] == "30 days" for span in derived["evidence_spans"])


def test_compute_derived_fields_promotes_event_timeframe_into_contract_end(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 7.1

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return False

        def derive_price_complaint(self, result):
            return False

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "summary": "Renewal planning",
        "review_text": "We are evaluating alternatives ahead of renewal.",
        "pros": "",
        "cons": "",
        "reviewer_title": "",
        "reviewer_company": "",
        "raw_metadata": {"source_weight": 0.8},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": False,
            "actively_evaluating": True,
            "contract_renewal_mentioned": True,
            "renewal_timing": None,
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {},
        "budget_signals": {},
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": [],
        "quotable_phrases": [],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": [],
        "pricing_phrases": [],
        "event_mentions": [{"event": "renewal", "timeframe": "next quarter"}],
        "timeline": {},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["timeline"]["contract_end"] == "next quarter"
    assert derived["timeline"]["decision_timeline"] == "within_quarter"


@pytest.mark.asyncio
async def test_enrich_rows_counts_quarantined(monkeypatch):
    async def _fake_enrich_single(pool, row, max_attempts, local_only, max_tokens, truncate_length):
        return "quarantined"

    monkeypatch.setattr(b2b_enrichment, "_enrich_single", _fake_enrich_single)

    rows = [{"id": uuid4(), "enrichment_attempts": 0} for _ in range(3)]
    cfg = SimpleNamespace(
        enrichment_max_attempts=3,
        enrichment_concurrency=2,
        enrichment_local_only=False,
        enrichment_max_tokens=2048,
        review_truncate_length=3000,
    )
    pool = SimpleNamespace(
        fetchval=AsyncMock(return_value=0),
        fetch=AsyncMock(return_value=[{"enrichment_status": "quarantined", "ct": 3}]),
    )

    result = await b2b_enrichment._enrich_rows(rows, cfg, pool)

    assert result["enriched"] == 0
    assert result["quarantined"] == 3


def test_compute_derived_fields_adds_witness_primitives(monkeypatch):
    class _Engine:
        map_hash = "test-hash"

        def compute_urgency(self, indicators, rating, rating_max, content_type, source_weight):
            return 8.4

        def override_pain(self, primary_pain, complaints, quotable, pricing_phrases, feature_gaps, recommendation_language):
            return primary_pain

        def derive_recommend(self, rec_lang, rating, rating_max):
            return False

        def derive_budget_authority(self, result):
            return True

        def derive_price_complaint(self, result):
            return True

    monkeypatch.setattr(evidence_engine, "get_evidence_engine", lambda: _Engine())

    row = {
        "id": uuid4(),
        "summary": "Renewal pricing pushed us toward async docs",
        "review_text": (
            "Slack wanted $200k/year at renewal. "
            "We became more productive using docs and async updates instead."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "CTO",
        "reviewer_company": "Hack Club",
        "raw_metadata": {"source_weight": 0.9},
        "content_type": "review",
        "rating": 2.0,
        "rating_max": 5,
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "contract_renewal_mentioned": True,
            "renewal_timing": "next quarter",
            "migration_in_progress": False,
            "support_escalation": False,
        },
        "reviewer_context": {
            "role_level": "executive",
            "decision_maker": True,
            "company_name": "Hack Club",
        },
        "budget_signals": {
            "annual_spend_estimate": 200000,
            "price_per_seat": None,
            "seat_count": None,
            "price_increase_mentioned": True,
        },
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["Slack wanted $200k/year at renewal"],
        "quotable_phrases": ["We became more productive using docs and async updates instead"],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": ["I would not recommend this"],
        "pricing_phrases": ["$200k/year at renewal"],
        "event_mentions": [{"event": "renewal", "timeframe": "next quarter"}],
        "timeline": {"contract_end": "next quarter", "evaluation_deadline": None},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }

    derived = b2b_enrichment._compute_derived_fields(result, row)

    assert derived["replacement_mode"] == "workflow_substitution"
    assert derived["operating_model_shift"] == "sync_to_async"
    assert derived["productivity_delta_claim"] == "more_productive"
    assert derived["org_pressure_type"] == "none"
    assert {"explicit_dollar", "named_account", "decision_maker"}.issubset(set(derived["salience_flags"]))
    assert derived["evidence_spans"]
    assert any(span["signal_type"] == "pricing_backlash" for span in derived["evidence_spans"])
    assert any(span["productivity_delta_claim"] == "more_productive" for span in derived["evidence_spans"])


def test_repair_target_fields_flags_semantic_pricing_and_timeline_gaps():
    row = {
        "summary": "Renewal confusion",
        "review_text": "We got a $50k renewal quote and need to decide next quarter.",
        "pros": "",
        "cons": "",
        "source": "reddit",
        "enrichment_status": "enriched",
    }
    result = {
        "pain_category": "ux",
        "specific_complaints": [],
        "pricing_phrases": [],
        "recommendation_language": [],
        "feature_gaps": [],
        "event_mentions": [],
        "competitors_mentioned": [],
        "salience_flags": ["explicit_dollar"],
        "timeline": {"decision_timeline": "unknown"},
    }

    targets = b2b_enrichment._repair_target_fields(result, row)

    assert "specific_complaints" in targets
    assert "pricing_phrases" in targets
    assert "event_mentions" in targets


def _witness_ready_row_and_result():
    row = {
        "id": "review-1",
        "vendor_name": "Slack",
        "source": "g2",
        "content_type": "review",
        "summary": "Renewal pushed us away from Slack",
        "review_text": (
            "Slack wanted $200k/year at renewal. We became more productive using docs "
            "and async updates instead."
        ),
        "pros": "",
        "cons": "",
        "reviewer_title": "VP Operations",
        "reviewer_company": "Hack Club",
        "rating": 2,
        "rating_max": 5,
        "raw_metadata": {"source_weight": 0.9},
    }
    result = {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": True,
            "migration_in_progress": False,
            "support_escalation": False,
            "contract_renewal_mentioned": True,
        },
        "reviewer_context": {
            "role_level": "executive",
            "decision_maker": True,
            "company_name": "Hack Club",
        },
        "budget_signals": {
            "annual_spend_estimate": 200000,
            "price_per_seat": None,
            "seat_count": None,
            "price_increase_mentioned": True,
        },
        "use_case": {"modules_mentioned": [], "integration_stack": [], "lock_in_level": "low"},
        "content_classification": "review",
        "competitors_mentioned": [],
        "specific_complaints": ["Slack wanted $200k/year at renewal"],
        "quotable_phrases": ["We became more productive using docs and async updates instead"],
        "positive_aspects": [],
        "feature_gaps": [],
        "recommendation_language": ["I would not recommend this"],
        "pricing_phrases": ["$200k/year at renewal"],
        "event_mentions": [{"event": "renewal", "timeframe": "next quarter"}],
        "timeline": {"contract_end": "next quarter", "evaluation_deadline": None},
        "contract_context": {},
        "buyer_authority": {},
        "sentiment_trajectory": {},
    }
    return row, result


def test_finalize_enrichment_for_persist_populates_witness_primitives():
    row, result = _witness_ready_row_and_result()

    finalized, error = b2b_enrichment._finalize_enrichment_for_persist(result, row)

    assert error is None
    assert finalized is not None
    assert finalized["replacement_mode"] == "workflow_substitution"
    assert finalized["operating_model_shift"] == "sync_to_async"
    assert finalized["productivity_delta_claim"] == "more_productive"
    assert finalized["org_pressure_type"] == "none"
    assert finalized["salience_flags"]
    assert finalized["evidence_spans"]
    assert finalized["evidence_map_hash"]


def test_validate_enrichment_schema_v3_recomputes_missing_witness_primitives():
    row, result = _witness_ready_row_and_result()
    derived = b2b_enrichment._compute_derived_fields(result, row)
    for key in (
        "replacement_mode",
        "operating_model_shift",
        "productivity_delta_claim",
        "org_pressure_type",
        "salience_flags",
        "evidence_spans",
        "evidence_map_hash",
    ):
        derived.pop(key, None)

    assert b2b_enrichment._validate_enrichment(derived, row)
    assert derived["replacement_mode"] == "workflow_substitution"
    assert derived["evidence_spans"]
    assert derived["evidence_map_hash"]


def test_validate_enrichment_schema_v3_rejects_missing_witness_primitives_without_source_row():
    row, result = _witness_ready_row_and_result()
    derived = b2b_enrichment._compute_derived_fields(result, row)
    derived.pop("evidence_spans", None)
    derived.pop("evidence_map_hash", None)

    assert not b2b_enrichment._validate_enrichment(derived)
