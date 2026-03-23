"""Focused tests for evidence-vault overlays in B2B blog generation."""

from unittest.mock import AsyncMock

import pytest

import atlas_brain.autonomous.tasks.b2b_blog_post_generation as blog_mod
from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
    _candidate_overlaps_gap_pain,
    _blueprint_churn_report,
    _blueprint_migration_guide,
    _blueprint_vendor_alternative,
    _build_specialized_blog_review_rows_from_evidence_vault,
    _detect_campaign_content_gaps,
    _gather_data,
    _load_pool_layers_for_blog,
    _merge_blog_quotes_with_evidence_vault,
    _merge_blog_signals_with_evidence_vault,
)


def test_merge_blog_signals_with_evidence_vault_prefers_canonical_rows():
    raw = [
        {"pain_category": "pricing", "signal_count": 3, "avg_urgency": 4.0, "feature_gaps": ["Legacy exports"]},
        {"pain_category": "support", "signal_count": 2, "avg_urgency": 6.2, "feature_gaps": []},
    ]
    vault = {
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "mention_count_total": 12,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.1},
            },
            {
                "key": "custom_roles",
                "label": "Custom roles",
                "evidence_type": "feature_gap",
                "mention_count_total": 5,
            },
        ],
    }
    merged = _merge_blog_signals_with_evidence_vault(raw, vault)
    assert merged[0]["pain_category"] == "Pricing opacity"
    assert merged[0]["signal_count"] == 12
    assert merged[0]["avg_urgency"] == 7.1
    assert merged[0]["feature_gaps"] == ["Custom roles"]
    assert any(item["pain_category"] == "support" for item in merged)


def test_merge_blog_quotes_with_evidence_vault_prefers_canonical_quotes():
    raw = [
        {
            "phrase": "Pricing opacity kept surprising us",
            "vendor": "Zendesk",
            "urgency": 5.0,
            "role": "Ops Manager",
            "company": "Acme",
            "company_size": "Mid-Market",
            "industry": "SaaS",
            "source_name": "g2",
            "sentiment": "negative",
        },
        {
            "phrase": "The integrations save hours every week",
            "vendor": "Zendesk",
            "urgency": 2.0,
            "role": "RevOps",
            "company": "Acme",
            "company_size": "Mid-Market",
            "industry": "SaaS",
            "source_name": "g2",
            "sentiment": "positive",
        },
    ]
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "best_quote": "Pricing opacity kept surprising us",
                "mention_count_total": 11,
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "company": "Acme"},
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.4},
            },
        ],
        "strength_evidence": [
            {
                "best_quote": "The integrations save hours every week",
                "mention_count_total": 6,
                "quote_source": {"source": "capterra", "reviewer_title": "RevOps", "company": "Acme"},
                "supporting_metrics": {},
            },
        ],
    }
    merged = _merge_blog_quotes_with_evidence_vault(raw, vault)
    assert merged[0]["phrase"] == "Pricing opacity kept surprising us"
    assert merged[0]["source_name"] == "reddit"
    assert merged[0]["sentiment"] == "negative"
    assert any(item["phrase"] == "The integrations save hours every week" and item["source_name"] == "capterra" for item in merged)


def test_build_specialized_blog_review_rows_from_evidence_vault_filters_pricing():
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "best_quote": "The contract cost kept climbing after the add-ons",
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "rating": 2.0},
                "supporting_metrics": {"avg_urgency_when_mentioned": 8.1},
                "mention_count_total": 9,
            },
            {
                "key": "support",
                "label": "Support",
                "evidence_type": "pain_category",
                "best_quote": "Support vanished during onboarding",
                "quote_source": {"source": "g2", "reviewer_title": "Director"},
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.0},
                "mention_count_total": 12,
            },
        ],
    }
    rows = _build_specialized_blog_review_rows_from_evidence_vault(
        vault,
        mode="pricing",
        limit=5,
    )
    assert len(rows) == 1
    assert rows[0]["text"] == "The contract cost kept climbing after the add-ons"
    assert rows[0]["source_name"] == "reddit"


def _section_by_id(blueprint, section_id: str):
    for section in blueprint.sections:
        if section.id == section_id:
            return section
    raise AssertionError(f"missing section {section_id}")


def test_blueprint_vendor_alternative_uses_reasoning_contracts_for_market_context():
    blueprint = _blueprint_vendor_alternative(
        {"vendor": "Zendesk", "category": "Helpdesk", "urgency": 7.2, "review_count": 42, "slug": "zendesk-alternatives"},
        {
            "profile": {"strengths": []},
            "signals": [{"pain_category": "pricing", "avg_urgency": 7.1, "signal_count": 12, "feature_gaps": []}],
            "partner": None,
            "pool_displacement": [],
            "pool_temporal": {},
            "pool_segment": {},
            "data_context": {},
            "synthesis_contracts": {
                "vendor_core_reasoning": {
                    "segment_playbook": {
                        "priority_segments": [{"segment": "mid-market finance teams", "estimated_reach": {"value": 18}}],
                    },
                    "timing_intelligence": {
                        "best_timing_window": "Before renewal review",
                        "active_eval_signals": {"value": 11},
                        "immediate_triggers": [{"trigger": "Renewal review", "type": "deadline"}],
                    },
                    "causal_narrative": {"trigger": "Price hike", "why_now": "Budget pressure"},
                },
                "displacement_reasoning": {
                    "migration_proof": {
                        "switching_is_real": True,
                        "evidence_type": "explicit_switch",
                        "switch_volume": {"value": 6},
                    },
                },
                "account_reasoning": {
                    "market_summary": "Three strategic accounts are actively evaluating alternatives.",
                    "top_accounts": [{"name": "Acme Corp"}, {"name": "Globex"}],
                },
                "category_reasoning": {"market_regime": "consolidating", "winner": "Freshdesk"},
            },
            "synthesis_wedge": "price_squeeze",
            "synthesis_wedge_label": "Price Squeeze",
        },
    )

    market_context = _section_by_id(blueprint, "market_context")
    verdict = _section_by_id(blueprint, "verdict")

    assert market_context.key_stats["switching_is_real"] is True
    assert market_context.key_stats["timing_summary"]
    assert market_context.key_stats["segment_targeting_summary"]
    assert market_context.key_stats["account_pressure_summary"] == (
        "Three strategic accounts are actively evaluating alternatives."
    )
    assert market_context.key_stats["category_market_regime"] == "consolidating"
    assert verdict.key_stats["market_regime"] == "consolidating"


def test_blueprint_churn_report_uses_contract_sections_when_pool_slices_are_thin():
    blueprint = _blueprint_churn_report(
        {
            "vendor": "Zendesk",
            "category": "Helpdesk",
            "negative_reviews": 14,
            "total_reviews": 42,
            "avg_urgency": 7.2,
            "slug": "zendesk-churn-report",
        },
        {
            "signals": [{"pain_category": "pricing", "signal_count": 12, "avg_urgency": 7.1, "feature_gaps": []}],
            "profile": {},
            "pool_displacement": [],
            "pool_segment": {},
            "pool_temporal": {},
            "category_overview": {},
            "data_context": {},
            "synthesis_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"trigger": "Price hike", "why_now": "Budget pressure"},
                    "segment_playbook": {
                        "priority_segments": [{"segment": "finance teams", "estimated_reach": {"value": 9}}],
                    },
                    "timing_intelligence": {
                        "best_timing_window": "Before renewal review",
                        "active_eval_signals": {"value": 8},
                        "immediate_triggers": [{"trigger": "Renewal review", "type": "deadline"}],
                    },
                },
                "displacement_reasoning": {
                    "migration_proof": {"switching_is_real": True, "switch_volume": {"value": 4}},
                },
                "account_reasoning": {"market_summary": "Two named accounts are in active evaluation."},
                "category_reasoning": {
                    "market_regime": "fragmented",
                    "narrative": "Buyers are actively re-evaluating vendor fit.",
                },
            },
            "synthesis_wedge": "price_squeeze",
        },
    )

    hook = _section_by_id(blueprint, "hook")
    displacement = _section_by_id(blueprint, "displacement")
    buyer_segments = _section_by_id(blueprint, "buyer_segments")
    timing = _section_by_id(blueprint, "timing")
    outlook = _section_by_id(blueprint, "outlook")

    assert hook.key_stats["market_regime"] == "fragmented"
    assert displacement.key_stats["switching_is_real"] is True
    assert buyer_segments.key_stats["segment_targeting_summary"]
    assert timing.key_stats["timing_summary"]
    assert outlook.key_stats["account_pressure_summary"] == (
        "Two named accounts are in active evaluation."
    )
    assert outlook.key_stats["category_narrative"] == (
        "Buyers are actively re-evaluating vendor fit."
    )


def test_blueprint_migration_guide_promotes_migration_proof_and_account_reasoning():
    blueprint = _blueprint_migration_guide(
        {
            "vendor": "Zendesk",
            "category": "Helpdesk",
            "switch_count": 9,
            "review_total": 42,
            "slug": "zendesk-migration-guide",
        },
        {
            "profile": {},
            "signals": [],
            "pool_displacement": [],
            "data_context": {},
            "synthesis_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {"trigger": "Price hike"},
                    "timing_intelligence": {
                        "best_timing_window": "Before renewal review",
                        "active_eval_signals": {"value": 8},
                        "immediate_triggers": [{"trigger": "Renewal review", "type": "deadline"}],
                    },
                },
                "displacement_reasoning": {
                    "migration_proof": {
                        "switching_is_real": True,
                        "evidence_type": "explicit_switch",
                        "switch_volume": {"value": 4},
                        "top_destination": {"value": "Freshdesk"},
                    },
                },
                "account_reasoning": {
                    "market_summary": "Two named accounts are in active evaluation.",
                },
                "category_reasoning": {"market_regime": "consolidating"},
            },
            "synthesis_wedge": "price_squeeze",
        },
    )

    takeaway = _section_by_id(blueprint, "takeaway")

    assert takeaway.key_stats["switching_is_real"] is True
    assert takeaway.key_stats["top_destination"] == "Freshdesk"
    assert takeaway.key_stats["account_pressure_summary"] == (
        "Two named accounts are in active evaluation."
    )
    assert takeaway.key_stats["timing_summary"]
    assert takeaway.key_stats["market_regime"] == "consolidating"


@pytest.mark.asyncio
async def test_gather_data_vendor_alternative_uses_evidence_vault_overlay(monkeypatch):
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "mention_count_total": 14,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.8},
                "best_quote": "Pricing opacity kept surprising us",
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "company": "Acme"},
            },
        ],
    }
    pool = type("Pool", (), {"fetchrow": AsyncMock(return_value={
        "total_reviews": 25,
        "enriched": 20,
        "churn_intent": 8,
        "earliest": "2026-01-01",
        "latest": "2026-03-18",
    })})()

    monkeypatch.setattr(blog_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={"Zendesk": vault}))
    monkeypatch.setattr(blog_mod, "_fetch_product_profile", AsyncMock(return_value={"strengths": []}))
    monkeypatch.setattr(blog_mod, "_fetch_churn_signals", AsyncMock(return_value=[
        {"pain_category": "pricing", "signal_count": 3, "avg_urgency": 4.0, "feature_gaps": []},
    ]))
    monkeypatch.setattr(blog_mod, "_fetch_quotable_reviews", AsyncMock(return_value=[
        {
            "phrase": "Pricing opacity kept surprising us",
            "vendor": "Zendesk",
            "urgency": 5.0,
            "role": "Ops Manager",
            "company": "Acme",
            "company_size": "Mid-Market",
            "industry": "SaaS",
            "source_name": "g2",
            "sentiment": "negative",
        },
    ]))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_source_distribution", AsyncMock(return_value={"g2": 10}))
    monkeypatch.setattr(blog_mod, "_fetch_category_overview_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner_by_category", AsyncMock(return_value=None))

    data = await _gather_data(
        pool,
        "vendor_alternative",
        {"vendor": "Zendesk", "category": "Helpdesk", "review_count": 42, "urgency": 7.2, "slug": "zendesk-alternatives"},
    )

    assert data["signals"][0]["pain_category"] == "Pricing opacity"
    assert data["signals"][0]["signal_count"] == 14
    assert data["quotes"][0]["phrase"] == "Pricing opacity kept surprising us"
    assert data["quotes"][0]["source_name"] == "reddit"
    assert data["data_context"]["evidence_vault_used"] is True
    assert data["data_context"]["evidence_vault_vendors"] == ["Zendesk"]


@pytest.mark.asyncio
async def test_gather_data_pricing_reality_check_uses_evidence_vault_review_fallback(monkeypatch):
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "mention_count_total": 14,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.8},
                "best_quote": "Pricing kept increasing after the initial contract",
                "quote_source": {"source": "reddit", "reviewer_title": "VP Ops", "rating": 2.0},
            },
        ],
        "strength_evidence": [
            {
                "key": "integrations",
                "label": "Integrations",
                "best_quote": "The integrations still save us a lot of time",
                "quote_source": {"source": "capterra", "reviewer_title": "RevOps", "rating": 4.0},
                "mention_count_total": 6,
            },
        ],
    }
    pool = type(
        "Pool",
        (),
        {
            "fetch": AsyncMock(return_value=[]),
            "fetchrow": AsyncMock(return_value={
                "total_reviews": 25,
                "enriched": 20,
                "churn_intent": 8,
                "earliest": "2026-01-01",
                "latest": "2026-03-18",
            }),
        },
    )()

    monkeypatch.setattr(blog_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={"Zendesk": vault}))
    monkeypatch.setattr(blog_mod, "_fetch_product_profile", AsyncMock(return_value={"strengths": []}))
    monkeypatch.setattr(blog_mod, "_fetch_churn_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_fetch_source_distribution", AsyncMock(return_value={"g2": 10}))
    monkeypatch.setattr(blog_mod, "_fetch_category_overview_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner_by_category", AsyncMock(return_value=None))

    data = await _gather_data(
        pool,
        "pricing_reality_check",
        {"vendor": "Zendesk", "category": "Helpdesk", "pricing_complaints": 8, "total_reviews": 42, "avg_urgency": 7.2, "slug": "zendesk-pricing"},
    )

    assert data["pricing_reviews"][0]["text"] == "Pricing kept increasing after the initial contract"
    assert data["pricing_reviews"][0]["source_name"] == "reddit"
    assert data["positive_reviews"][0]["text"] == "The integrations still save us a lot of time"


@pytest.mark.asyncio
async def test_gather_data_switching_story_uses_evidence_vault_review_fallback(monkeypatch):
    vault = {
        "vendor_name": "Zendesk",
        "weakness_evidence": [
            {
                "key": "support",
                "label": "Support",
                "evidence_type": "pain_category",
                "mention_count_total": 10,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.5},
                "best_quote": "We moved away after support stopped responding during renewal",
                "quote_source": {"source": "g2", "reviewer_title": "Director", "rating": 2.0},
            },
        ],
    }
    pool = type(
        "Pool",
        (),
        {
            "fetch": AsyncMock(return_value=[]),
            "fetchrow": AsyncMock(return_value={
                "total_reviews": 25,
                "enriched": 20,
                "churn_intent": 8,
                "earliest": "2026-01-01",
                "latest": "2026-03-18",
            }),
        },
    )()

    monkeypatch.setattr(blog_mod, "_fetch_latest_evidence_vault", AsyncMock(return_value={"Zendesk": vault}))
    monkeypatch.setattr(blog_mod, "_fetch_product_profile", AsyncMock(return_value={"strengths": []}))
    monkeypatch.setattr(blog_mod, "_fetch_churn_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(blog_mod, "_fetch_source_distribution", AsyncMock(return_value={"g2": 10}))
    monkeypatch.setattr(blog_mod, "_fetch_category_overview_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(blog_mod, "_fetch_affiliate_partner_by_category", AsyncMock(return_value=None))

    data = await _gather_data(
        pool,
        "switching_story",
        {"from_vendor": "Zendesk", "category": "Helpdesk", "switch_mentions": 6, "total_reviews": 42, "avg_urgency": 7.2, "slug": "why-teams-leave-zendesk"},
    )

    assert data["switch_reviews"][0]["text"] == "We moved away after support stopped responding during renewal"
    assert data["switch_reviews"][0]["source_name"] == "g2"
    assert data["quotes"] == data["switch_reviews"]


def test_candidate_overlaps_gap_pain_ignores_vendor_substring_false_positives():
    ctx = {"vendor": "Pricingly CRM", "category": "CRM"}
    assert _candidate_overlaps_gap_pain("vendor_deep_dive", ctx, {"pricing"}) is False
    assert _candidate_overlaps_gap_pain("pricing_reality_check", ctx, {"pricing"}) is True


@pytest.mark.asyncio
async def test_load_pool_layers_for_blog_category_topic_sets_pool_category_without_vendor_query(monkeypatch):
    monkeypatch.setattr(
        blog_mod,
        "fetch_all_pool_layers",
        AsyncMock(return_value={
            "Zendesk": {
                "evidence_vault": {"product_category": "Helpdesk"},
                "category": {"category": "Helpdesk", "market_regime": {"regime_type": "high_churn"}},
            },
        }),
    )
    pool = type("Pool", (), {"fetch": AsyncMock(return_value=[])})()
    data: dict = {}

    await _load_pool_layers_for_blog(
        pool,
        "market_landscape",
        {"category": "Helpdesk"},
        data,
    )

    assert data["pool_category"]["category"] == "Helpdesk"
    pool.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_pool_layers_for_blog_scopes_synthesis_query_to_requested_vendor(monkeypatch):
    monkeypatch.setattr(
        blog_mod,
        "fetch_all_pool_layers",
        AsyncMock(return_value={"Zendesk": {"segment": {}, "temporal": {}, "accounts": {}, "category": {}, "displacement": []}}),
    )
    pool = type("Pool", (), {"fetch": AsyncMock(return_value=[])})()
    data: dict = {}

    await _load_pool_layers_for_blog(
        pool,
        "vendor_alternative",
        {"vendor": "Zendesk", "category": "Helpdesk"},
        data,
    )

    query = pool.fetch.await_args.args[0]
    vendor_filter = pool.fetch.await_args.args[3]
    assert "LOWER(vendor_name) = ANY($3::text[])" in query
    assert vendor_filter == ["zendesk"]


@pytest.mark.asyncio
async def test_detect_campaign_content_gaps_marks_showdown_drafts_as_coverage():
    class _GapPool:
        async def fetch(self, query, *args):
            normalized = " ".join(str(query).split())
            if "FROM b2b_campaigns" in normalized:
                return [{"vendor": "salesforce", "pain": "pricing"}]
            if "FROM blog_posts" in normalized:
                assert "status = ANY($1::text[])" in normalized
                assert "topic_type = ANY($2::text[])" in normalized
                return [{
                    "title": "Salesforce vs HubSpot",
                    "slug": "salesforce-vs-hubspot-2026-03",
                    "topic_type": "vendor_showdown",
                    "tags": ["crm", "comparison"],
                    "data_context": {
                        "vendor_a": "Salesforce",
                        "vendor_b": "HubSpot",
                        "pain_distribution": "pricing",
                    },
                }]
            raise AssertionError(f"Unexpected query: {normalized}")

    gaps = await _detect_campaign_content_gaps(_GapPool())
    assert gaps == {}
