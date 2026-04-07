import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import _campaign_sequence_context as ctx_mod
from atlas_brain.autonomous.tasks import amazon_seller_campaign_generation as seller_mod
from atlas_brain.autonomous.tasks import b2b_campaign_generation as gen_mod
from atlas_brain.autonomous.tasks import campaign_sequence_progression as mod


class _FakeSkillRegistry:
    def __init__(self, content: str):
        self._content = content

    def get(self, name: str):
        assert name
        return SimpleNamespace(content=self._content)


class _FakeLLM:
    def __init__(self, response: str):
        self._response = response
        self.calls = []

    def chat(self, messages, max_tokens=256, temperature=0.7, **kwargs):
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "kwargs": kwargs,
        })
        return {
            "response": self._response,
            "usage": {},
        }


class _CapturePool:
    def __init__(self):
        self.fetchval_calls = []
        self.execute_calls = []

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return uuid4()

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "OK"


class _ProgressionPool:
    is_initialized = True

    def __init__(self, sequence_rows, previous_rows):
        self.sequence_rows = sequence_rows
        self.previous_rows = previous_rows
        self.fetch_calls = []
        self.fetchval_calls = []
        self.execute_calls = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        if "FROM campaign_sequences cs" in query:
            return self.sequence_rows
        if "FROM b2b_campaigns" in query:
            return self.previous_rows
        raise AssertionError(f"Unexpected fetch query: {query}")

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return uuid4()

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "OK"


def test_prepare_prompt_contexts_removes_duplicate_and_heavy_fields():
    seq = {
        "company_context": json.dumps({
            "target_persona": "executive",
            "key_quotes": ["q1", "q2", "q3", "q4"],
            "pain_categories": [
                {"category": "pricing", "severity": "high"},
                {"category": "support", "severity": "medium"},
            ],
            "feature_gaps": ["automation", "reporting", "search", "alerts", "exports", "audit"],
            "comparison_asset": {
                "alternative_vendor": "ClickUp",
                "primary_blog_post": {"title": "ignore"},
            },
            "reasoning_witness_highlights": [{"excerpt_text": "ignore me"}],
            "selling": {
                "sender_name": "Atlas Intel",
                "booking_url": "https://atlas.test/book",
                "blog_posts": [{
                    "title": "Pricing pressure",
                    "url": "https://atlas.test/blog/pricing",
                    "topic_type": "pricing_reality_check",
                    "slug": "pricing-pressure",
                }],
            },
        }),
        "selling_context": "",
    }

    company_context, selling_context = ctx_mod.prepare_sequence_prompt_contexts(seq)

    assert "selling" not in company_context
    assert "comparison_asset" not in company_context
    assert "reasoning_witness_highlights" not in company_context
    assert company_context["key_quotes"] == ["q1", "q2", "q3"]
    assert company_context["feature_gaps"] == [
        "automation", "reporting", "search", "alerts", "exports",
    ]
    assert selling_context == {
        "sender_name": "Atlas Intel",
        "booking_url": "https://atlas.test/book",
        "blog_posts": [{
            "title": "Pricing pressure",
            "url": "https://atlas.test/blog/pricing",
            "topic_type": "pricing_reality_check",
        }],
    }


def test_prepare_prompt_contexts_compacts_vendor_reasoning_payload():
    seq = {
        "company_context": json.dumps({
            "recipient_type": "vendor_retention",
            "signal_summary": {
                "total_signals": 14,
                "competitor_distribution": [
                    {"name": f"Competitor {idx}", "count": idx, "reason": "extra"}
                    for idx in range(1, 8)
                ],
                "feature_gaps": [f"gap-{idx}" for idx in range(1, 8)],
            },
            "reasoning_context": {
                "summary": "Retention pressure is clustering around renewal windows.",
                "key_signals": ["Renewal pressure", "Support frustration", "Migration risk", "Extra"],
                "why_they_stay": {
                    "summary": "Implementation depth still matters.",
                    "strengths": [{"area": "workflow", "evidence": "long setup"}],
                },
                "switch_triggers": [
                    {"type": "renewal", "description": "Q2 renewal"},
                    {"type": "support", "description": "ticket backlog"},
                    {"type": "pricing", "description": "price jump"},
                    {"type": "extra", "description": "should trim"},
                ],
            },
            "reasoning_contracts": {"raw": "drop"},
        }),
        "selling_context": "{}",
    }

    company_context, selling_context = ctx_mod.prepare_sequence_prompt_contexts(seq)

    assert selling_context == {}
    assert "reasoning_contracts" not in company_context
    assert company_context["signal_summary"]["total_signals"] == 14
    assert len(company_context["signal_summary"]["competitor_distribution"]) == 5
    assert len(company_context["signal_summary"]["feature_gaps"]) == 5
    assert company_context["reasoning_context"]["key_signals"] == [
        "Renewal pressure", "Support frustration", "Migration risk",
    ]
    assert company_context["reasoning_context"]["why_they_stay"] == {
        "summary": "Implementation depth still matters.",
    }
    assert len(company_context["reasoning_context"]["switch_triggers"]) == 3


def test_prepare_storage_contexts_preserves_specificity_fields():
    company_context, selling_context = ctx_mod.prepare_sequence_storage_contexts(
        {
            "target_persona": "executive",
            "selling": {
                "sender_name": "Atlas Intel",
                "booking_url": "https://atlas.test/book",
            },
            "comparison_asset": {"alternative_vendor": "ClickUp"},
            "reasoning_anchor_examples": {"outlier_or_named_account": [{"witness_id": "w1"}]},
            "reasoning_witness_highlights": [{"witness_id": "w1", "excerpt_text": "signal"}],
            "reasoning_reference_ids": {"witness_ids": ["w1"]},
            "reasoning_contracts": {"vendor_core_reasoning": {"summary": "drop"}},
        },
        {},
    )

    assert "selling" not in company_context
    assert "comparison_asset" not in company_context
    assert "reasoning_contracts" not in company_context
    assert company_context["reasoning_anchor_examples"]["outlier_or_named_account"][0]["witness_id"] == "w1"
    assert company_context["reasoning_witness_highlights"][0]["witness_id"] == "w1"
    assert company_context["reasoning_reference_ids"]["witness_ids"] == ["w1"]
    assert selling_context["sender_name"] == "Atlas Intel"


def test_build_previous_emails_uses_plain_text_preview_without_timestamps():
    timestamp = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    rendered = mod._build_previous_emails([
        {
            "step_number": 2,
            "subject": "Pricing signal",
            "body": "<p>Hello <strong>team</strong>.</p><p>We tracked a sharp pricing shift.</p>",
            "status": "sent",
            "sent_at": timestamp,
            "opened_at": timestamp,
            "clicked_at": None,
        },
    ])

    assert "<p>" not in rendered
    assert "<strong>" not in rendered
    assert timestamp.isoformat() not in rendered
    assert "sent:" not in rendered
    assert "Preview: Hello team. We tracked a sharp pricing shift." in rendered
    assert "Engagement: Opened" in rendered


def test_build_engagement_summary_omits_exact_timestamps():
    opened_at = datetime(2026, 4, 5, 15, 0, tzinfo=timezone.utc)
    clicked_at = datetime(2026, 4, 5, 16, 0, tzinfo=timezone.utc)
    rendered = mod._build_engagement_summary(
        {
            "open_count": 2,
            "last_opened_at": opened_at,
            "click_count": 1,
            "last_clicked_at": clicked_at,
        },
        [{"step_number": 1, "opened_at": opened_at, "clicked_at": None}],
    )

    assert "Opened 2 time(s)" in rendered
    assert "Clicked 1 time(s)" in rendered
    assert "Last opened" not in rendered
    assert "Last clicked" not in rendered
    assert opened_at.isoformat() not in rendered
    assert clicked_at.isoformat() not in rendered


def test_plain_text_preview_decodes_entities_and_normalizes_spacing():
    preview = ctx_mod.plain_text_preview(
        "<p>AT&amp;T&nbsp;teams need faster onboarding &quot;now&quot;.</p>"
    )

    assert preview == 'AT&T teams need faster onboarding "now".'


@pytest.mark.asyncio
async def test_generate_next_step_uses_compact_prompt_and_token_budget(monkeypatch):
    llm = _FakeLLM(json.dumps({
        "subject": "Follow up",
        "body": "<p>Body</p>",
        "cta": "Reply",
        "angle_reasoning": "Signals were strong.",
    }))
    skill_content = (
        "Company Context:\n{company_context}\n"
        "Selling Context:\n{selling_context}\n"
        "Previous Emails:\n{previous_emails}\n"
    )

    monkeypatch.setattr("atlas_brain.skills.get_skill_registry", lambda: _FakeSkillRegistry(skill_content))
    monkeypatch.setattr("atlas_brain.services.llm_router.get_triage_llm", lambda: llm)
    monkeypatch.setattr("atlas_brain.services.llm_registry.get_active", lambda: None)

    seq = {
        "company_name": "Acme Co",
        "current_step": 1,
        "max_steps": 4,
        "company_context": json.dumps({
            "target_persona": "executive",
            "comparison_asset": {"alternative_vendor": "ClickUp"},
            "selling": {
                "sender_name": "Atlas Intel",
                "booking_url": "https://atlas.test/book",
            },
            "key_quotes": ["quote-1", "quote-2", "quote-3", "quote-4"],
        }),
        "selling_context": "{}",
    }
    previous_campaigns = [{
        "step_number": 1,
        "subject": "Old subject",
        "body": "<p>Old <strong>body</strong> content.</p>",
        "status": "sent",
        "opened_at": datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
        "clicked_at": None,
    }]

    result = await mod._generate_next_step(seq, previous_campaigns)

    assert result["subject"] == "Follow up"
    assert len(llm.calls) == 1
    assert llm.calls[0]["max_tokens"] == ctx_mod.prompt_max_tokens()

    prompt = llm.calls[0]["messages"][0].content
    assert "comparison_asset" not in prompt
    assert prompt.count("sender_name") == 1
    assert "<strong>" not in prompt
    assert "sent:" not in prompt


@pytest.mark.asyncio
async def test_generate_next_step_replaces_onboarding_product_name(monkeypatch):
    llm = _FakeLLM(json.dumps({
        "subject": "Welcome",
        "body": "<p>Body</p>",
        "cta": "Start",
        "angle_reasoning": "Welcome step.",
    }))

    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry("Product: {product_name}"),
    )
    monkeypatch.setattr("atlas_brain.services.llm_router.get_triage_llm", lambda: llm)
    monkeypatch.setattr("atlas_brain.services.llm_registry.get_active", lambda: None)
    monkeypatch.setattr(
        mod.settings.campaign_sequence,
        "onboarding_product_name",
        "Atlas Onboarding",
    )

    result = await mod._generate_next_step(
        {
            "company_name": "Acme Co",
            "current_step": 1,
            "company_context": json.dumps({
                "recipient_type": "onboarding",
                "product_name": "Configured Product",
            }),
            "selling_context": "{}",
        },
        [],
    )

    assert result["subject"] == "Welcome"
    assert llm.calls[0]["messages"][0].content == "Product: Configured Product"


@pytest.mark.asyncio
async def test_progression_run_uses_sender_type_specific_from_email(monkeypatch):
    pool = _ProgressionPool(
        sequence_rows=[{
            "id": uuid4(),
            "company_name": "Acme Co",
            "batch_id": "batch_1",
            "partner_id": None,
            "recipient_email": "owner@example.com",
            "company_context": json.dumps({"recipient_type": "vendor_retention"}),
            "selling_context": json.dumps({"sender_name": "Atlas Intel"}),
            "status": "active",
            "current_step": 1,
            "max_steps": 4,
            "open_count": 0,
            "click_count": 0,
            "next_step_after": datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            "last_sent_at": datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc),
        }],
        previous_rows=[{
            "step_number": 1,
            "subject": "Initial note",
            "body": "<p>Old body</p>",
            "status": "sent",
            "opened_at": None,
            "clicked_at": None,
        }],
    )

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", True)
    monkeypatch.setattr(mod.settings.campaign_sequence, "sender_type", "ses")
    monkeypatch.setattr(mod.settings.campaign_sequence, "ses_from_email", "ses@example.com")
    monkeypatch.setattr(mod.settings.campaign_sequence, "resend_from_email", "resend@example.com")
    monkeypatch.setattr(mod.settings.campaign_sequence, "progression_batch_limit", 7)
    monkeypatch.setattr(mod, "_generate_next_step", AsyncMock(return_value={
        "subject": "Follow up",
        "body": "<p>Body</p>",
        "cta": "Reply",
        "angle_reasoning": "No opens yet.",
        "_recipient_type": "vendor_retention",
        "_category": None,
    }))
    monkeypatch.setattr(mod, "_send_ntfy_notification", AsyncMock())
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())

    result = await mod.run(SimpleNamespace())

    assert result["progressed"] == 1
    insert_args = pool.fetchval_calls[0][1]
    assert pool.fetch_calls[0][1][1] == 7
    assert insert_args[9] == "ses@example.com"


@pytest.mark.asyncio
async def test_create_sequence_for_cold_email_persists_compact_context(monkeypatch):
    pool = _CapturePool()
    monkeypatch.setattr(gen_mod, "log_campaign_event", AsyncMock())

    await gen_mod._create_sequence_for_cold_email(
        pool,
        company_name="Acme Co",
        batch_id="batch_1",
        partner_id=None,
        context={
            "target_persona": "executive",
            "key_quotes": ["q1", "q2", "q3", "q4"],
            "selling": {
                "sender_name": "Atlas Intel",
                "booking_url": "https://atlas.test/book",
                "blog_posts": [{
                    "title": "Pricing pressure",
                    "url": "https://atlas.test/blog/pricing",
                    "topic_type": "pricing_reality_check",
                    "slug": "drop-me",
                }],
            },
            "comparison_asset": {"alternative_vendor": "ClickUp"},
            "reasoning_anchor_examples": {"outlier_or_named_account": [{"witness_id": "w1"}]},
            "reasoning_witness_highlights": [{"witness_id": "w1", "excerpt_text": "proof"}],
            "reasoning_reference_ids": {"witness_ids": ["w1"]},
            "reasoning_contracts": {"vendor_core_reasoning": {"summary": "drop"}},
        },
        cold_email_subject="Subject",
        cold_email_body="<p>Body</p>",
    )

    insert_args = pool.fetchval_calls[0][1]
    stored_company_context = json.loads(insert_args[3])
    stored_selling_context = json.loads(insert_args[4])

    assert "selling" not in stored_company_context
    assert "comparison_asset" not in stored_company_context
    assert "reasoning_contracts" not in stored_company_context
    assert stored_company_context["reasoning_anchor_examples"]["outlier_or_named_account"][0]["witness_id"] == "w1"
    assert stored_company_context["reasoning_witness_highlights"][0]["witness_id"] == "w1"
    assert stored_company_context["reasoning_reference_ids"]["witness_ids"] == ["w1"]
    assert stored_company_context["key_quotes"] == ["q1", "q2", "q3"]
    assert stored_selling_context["blog_posts"] == [{
        "title": "Pricing pressure",
        "url": "https://atlas.test/blog/pricing",
        "topic_type": "pricing_reality_check",
    }]


@pytest.mark.asyncio
async def test_create_seller_sequence_persists_compact_context(monkeypatch):
    pool = _CapturePool()
    monkeypatch.setattr(seller_mod, "log_campaign_event", AsyncMock())

    await seller_mod._create_seller_sequence(
        pool,
        seller_name="Acme Seller",
        seller_email="owner@example.com",
        batch_id="seller_batch",
        category="pet supplies",
        intel={
            "category_stats": {
                "total_reviews": 1000,
                "total_brands": 20,
                "total_products": 50,
                "date_range": "all available data",
            },
            "top_pain_points": [{"pain": f"pain-{idx}", "count": idx} for idx in range(1, 8)],
            "feature_gaps": [{"feature": f"gap-{idx}", "count": idx} for idx in range(1, 8)],
            "competitive_flows": [{"brand": f"brand-{idx}", "count": idx} for idx in range(1, 8)],
            "brand_health": [{"brand": f"brand-{idx}", "score": idx} for idx in range(1, 8)],
            "safety_signals": [{"signal": f"signal-{idx}", "count": idx} for idx in range(1, 8)],
            "top_root_causes": [{"cause": f"cause-{idx}", "count": idx} for idx in range(1, 8)],
        },
        selling_ctx={
            "sender_name": "Atlas Intel",
            "sender_title": "Founder",
            "free_report_url": "https://atlas.test/report",
            "blog_posts": [{
                "title": "Market breakdown",
                "url": "https://atlas.test/blog/market-breakdown",
                "topic_type": "migration_report",
                "slug": "drop-me",
            }],
        },
        cold_email_subject="Subject",
        cold_email_body="<p>Body</p>",
    )

    insert_args = pool.fetchval_calls[0][1]
    stored_company_context = json.loads(insert_args[2])
    stored_selling_context = json.loads(insert_args[3])

    assert stored_company_context["recipient_type"] == "amazon_seller"
    assert len(stored_company_context["category_intelligence"]["feature_gaps"]) == 5
    assert len(stored_company_context["category_intelligence"]["safety_signals"]) == 3
    assert stored_selling_context["blog_posts"] == [{
        "title": "Market breakdown",
        "url": "https://atlas.test/blog/market-breakdown",
        "topic_type": "migration_report",
    }]
