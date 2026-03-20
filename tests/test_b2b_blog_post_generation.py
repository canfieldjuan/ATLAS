"""Focused tests for evidence-vault overlays in B2B blog generation."""

from unittest.mock import AsyncMock

import pytest

import atlas_brain.autonomous.tasks.b2b_blog_post_generation as blog_mod
from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
    _build_specialized_blog_review_rows_from_evidence_vault,
    _gather_data,
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
