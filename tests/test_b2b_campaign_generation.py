import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from atlas_brain.autonomous.tasks import b2b_campaign_generation as mod


class _FakeSkillRegistry:
    def get(self, name):
        return SimpleNamespace(content=f"skill:{name}")


class _FakePool:
    def __init__(
        self,
        briefing_data,
        fetch_rows=None,
        briefing_exists: bool = True,
        briefing_age_days: int = 8,
        dedup_count: int = 0,
    ):
        self.briefing_data = briefing_data
        self.fetch_rows = fetch_rows or []
        self.briefing_exists = briefing_exists
        self.briefing_age_days = briefing_age_days
        self.dedup_count = dedup_count
        self.fetchrow_calls = []
        self.execute_calls = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if "FROM b2b_vendor_briefings" in query:
            if not self.briefing_exists:
                return None
            return {
                "id": 1,
                "created_at": datetime.now(timezone.utc) - timedelta(days=self.briefing_age_days),
                "briefing_data": self.briefing_data,
            }
        if "FROM b2b_churn_signals" in query:
            return None
        return None

    async def fetchval(self, query, *args):
        if "FROM b2b_campaigns" in query:
            return self.dedup_count
        return 0

    async def fetch(self, query, *args):
        return self.fetch_rows

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "OK"


def test_briefing_context_from_data_parses_string_payload():
    briefing_data = json.dumps({
        "headline": "Retention pressure is visible this cycle.",
        "named_accounts": [
            {"account_name": "Acme"},
            {"company_name": "Beta"},
            {"name": "Acme"},
        ],
        "top_displacement_targets": [{"name": "HubSpot"}, "Salesforce"],
        "top_feature_gaps": [{"feature": "automation"}, "reporting"],
        "pain_breakdown": [{"category": "support"}, {"category": "support"}],
    })

    context = mod._briefing_context_from_data(briefing_data)

    assert context["executive_summary"] == "Retention pressure is visible this cycle."
    assert context["priority_account_names"] == ["Acme", "Beta"]
    assert context["top_displacement_targets"] == ["HubSpot", "Salesforce"]
    assert context["top_feature_gaps"] == ["automation", "reporting"]
    assert context["pain_labels"] == ["support"]


def test_build_comparison_asset_prefers_recommended_alternative_and_matching_blog():
    context = {
        "company": "Acme Co",
        "churning_from": "Monday.com",
        "role_type": "decision_maker",
        "company_size": "51-200 employees",
        "buying_stage": "evaluation",
        "pain_categories": [{"category": "pricing", "severity": "primary"}],
        "recommended_alternatives": [{
            "vendor_name": "ClickUp",
            "reasoning": "Addresses pain: pricing; 6 companies switched from Monday.com",
        }],
        "competitors_considering": [{"name": "Asana", "reason": "lower cost"}],
    }
    blog_posts = [
        {
            "title": "Monday.com vs ClickUp for pricing pressure",
            "url": "https://atlas.test/blog/monday-clickup-pricing",
            "topic_type": "vendor_showdown",
        },
        {
            "title": "Project management pricing reality check",
            "url": "https://atlas.test/blog/pricing-reality",
            "topic_type": "pricing_reality_check",
        },
    ]

    asset = mod._build_comparison_asset(context, blog_posts)

    assert asset["qualified"] is True
    assert asset["alternative_vendor"] == "ClickUp"
    assert asset["selection_source"] == "recommended_alternative"
    assert asset["primary_blog_post"]["url"] == "https://atlas.test/blog/monday-clickup-pricing"
    assert asset["supporting_blog_posts"][0]["topic_type"] == "vendor_showdown"


def test_build_comparison_asset_ignores_unmatched_blog_posts():
    context = {
        "company": "Acme Co",
        "churning_from": "Monday.com",
        "role_type": "decision_maker",
        "company_size": "51-200 employees",
        "buying_stage": "evaluation",
        "pain_categories": [{"category": "pricing", "severity": "primary"}],
        "competitors_considering": [{"name": "ClickUp", "reason": "lower cost"}],
    }
    blog_posts = [{
        "title": "SentinelOne Deep Dive",
        "url": "https://atlas.test/blog/sentinelone-deep-dive",
        "topic_type": "vendor_deep_dive",
    }]

    asset = mod._build_comparison_asset(context, blog_posts)

    assert asset["primary_blog_post"] is None
    assert asset["qualified"] is False


def test_evaluate_outbound_qualification_respects_blog_requirement():
    asset = {
        "company_safe": True,
        "pain_categories": ["pricing"],
        "alternative_vendor": "ClickUp",
        "primary_blog_post": None,
    }

    strict = mod._evaluate_outbound_qualification(
        asset,
        require_display_safe_company=True,
        require_primary_blog_post=True,
        min_pain_categories=1,
    )
    relaxed = mod._evaluate_outbound_qualification(
        asset,
        require_display_safe_company=True,
        require_primary_blog_post=False,
        min_pain_categories=1,
    )

    assert strict["qualified"] is False
    assert "primary_blog_post" in strict["missing_checks"]
    assert relaxed["qualified"] is True


def test_match_partner_prefers_comparison_vendor():
    context = {
        "category": "Project Management",
        "recommended_alternatives": [{"vendor_name": "ClickUp"}],
        "competitors_considering": [{"name": "Asana"}],
        "comparison_asset": {"alternative_vendor": "ClickUp"},
    }
    partner_index = {
        "by_product": {
            "clickup": {"id": "partner-clickup", "product_name": "ClickUp"},
            "asana": {"id": "partner-asana", "product_name": "Asana"},
        },
        "by_category": {},
    }

    partner = mod._match_partner(context, partner_index)

    assert partner["id"] == "partner-clickup"


@pytest.mark.asyncio
async def test_generate_vendor_campaigns_uses_mode_aware_briefing_context(monkeypatch):
    captured_payloads = []

    async def _fake_fetch_vendor_targets(pool, vendor_filter):
        return [{
            "company_name": "Asana",
            "contact_name": "Alex",
            "contact_role": "VP Customer Success",
            "contact_email": "alex@asana.test",
            "tier": "report",
            "products_tracked": ["Asana"],
        }]

    async def _fake_fetch_opportunities(pool, min_score, limit, company_filter=None, dm_only=False):
        return [{
            "vendor_name": "Asana",
            "review_id": "rev-1",
            "product_category": "Project Management",
            "opportunity_score": 91,
            "urgency": 9,
            "pain_json": [{"category": "support"}],
            "competitors": [{"name": "ClickUp", "reason": "automation"}],
            "feature_gaps": [{"feature": "reporting"}],
            "quotable_phrases": [{"text": "support has slowed down"}],
            "contract_end": None,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "seat_count": 120,
            "industry": "Software",
            "score_components": {"base": 1},
        }]

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        captured_payloads.append(payload)
        return {"subject": "test", "body": "<p>body</p>", "cta": "Book time"}

    async def _fake_trend(pool, vendor_name, products=None):
        return "increasing"

    async def _fake_blog_posts(*args, **kwargs):
        return []

    monkeypatch.setattr(mod, "_fetch_vendor_targets", _fake_fetch_vendor_targets)
    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_compute_vendor_trend", _fake_trend)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)

    pool = _FakePool({
        "account_pressure_summary": "Nine accounts are under active churn pressure.",
        "timing_summary": "Pressure is clustering around renewal windows.",
        "segment_targeting_summary": "Best tested on CS leaders in mid-market accounts.",
        "named_accounts": [{"account_name": "Acme"}, {"company_name": "Beta"}],
        "top_displacement_targets": [{"name": "ClickUp"}],
        "top_feature_gaps": [{"feature": "reporting"}],
        "pain_labels": ["support"],
    })

    result = await mod._generate_vendor_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
    )

    briefing_query = next(
        call for call in pool.fetchrow_calls if "FROM b2b_vendor_briefings" in call[0]
    )
    payload = captured_payloads[0]

    assert briefing_query[1] == ("Asana", "vendor_retention")
    assert payload["briefing_context"]["account_pressure_summary"] == (
        "Nine accounts are under active churn pressure."
    )
    assert payload["briefing_context"]["priority_account_names"] == ["Acme", "Beta"]
    assert payload["briefing_context"]["top_displacement_targets"] == ["ClickUp"]
    assert result["generated"] == 2


@pytest.mark.asyncio
async def test_generate_challenger_campaigns_uses_mode_aware_briefing_context(monkeypatch):
    captured_payloads = []

    async def _fake_fetch_challenger_targets(pool, vendor_filter):
        return [{
            "company_name": "HubSpot",
            "contact_name": "Taylor",
            "contact_role": "VP Sales",
            "contact_email": "taylor@hubspot.test",
            "tier": "report",
            "competitors_tracked": ["IncumbentCRM"],
        }]

    async def _fake_fetch_opportunities(pool, min_score, limit, company_filter=None, dm_only=False):
        return [{
            "vendor_name": "IncumbentCRM",
            "review_id": "rev-2",
            "product_category": "CRM",
            "opportunity_score": 88,
            "urgency": 8,
            "pain_json": [{"category": "pricing"}],
            "competitors": [{"name": "HubSpot", "reason": "ease of use"}],
            "quotable_phrases": [{"text": "we are evaluating alternatives"}],
            "contract_end": None,
            "decision_timeline": "this quarter",
            "buying_stage": "evaluation",
            "seat_count": 80,
            "industry": "Technology",
            "score_components": {"base": 1},
        }]

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        captured_payloads.append(payload)
        return {"subject": "test", "body": "<p>body</p>", "cta": "Book time"}

    async def _fake_blog_posts(*args, **kwargs):
        return []

    monkeypatch.setattr(mod, "_fetch_challenger_targets", _fake_fetch_challenger_targets)
    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)

    pool = _FakePool(
        {
            "executive_summary": "Competitive pull is strongest during active evaluations.",
            "named_accounts": [{"name": "Gamma Co"}],
            "top_displacement_targets": [{"name": "IncumbentCRM"}],
            "top_feature_gaps": ["implementation speed"],
            "trend": "increasing",
            "category": "CRM",
        },
        fetch_rows=[{"vendor_name": "IncumbentCRM", "archetype": "pricing_shock"}],
    )

    result = await mod._generate_challenger_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
    )

    briefing_query = next(
        call for call in pool.fetchrow_calls if "FROM b2b_vendor_briefings" in call[0]
    )
    payload = captured_payloads[0]

    assert briefing_query[1] == ("HubSpot", "challenger_intel")
    assert payload["briefing_context"]["executive_summary"] == (
        "Competitive pull is strongest during active evaluations."
    )
    assert payload["briefing_context"]["priority_account_names"] == ["Gamma Co"]
    assert payload["briefing_context"]["top_displacement_targets"] == ["IncumbentCRM"]
    assert payload["briefing_context"]["trend"] == "increasing"
    # pricing_shock legacy archetype maps to price_squeeze wedge
    assert payload["incumbent_archetypes"]["price_squeeze"] == ["IncumbentCRM"]
    assert result["generated"] == 2


@pytest.mark.asyncio
async def test_generate_churning_company_campaigns_generates_without_partner(monkeypatch):
    async def _fake_fetch_opportunities(
        pool,
        min_score,
        limit,
        vendor_filter=None,
        company_filter=None,
        dm_only=False,
    ):
        return [{
            "review_id": "rev-1",
            "reviewer_company": "Acme Co",
            "vendor_name": "Monday.com",
            "product_category": "Project Management",
            "opportunity_score": 88,
            "urgency": 8,
            "pain_json": [{"category": "pricing", "severity": "primary"}],
            "competitors": [{"name": "Asana", "reason": "lower cost"}],
            "quotable_phrases": [{"text": "pricing keeps going up"}],
            "feature_gaps": [{"feature": "automation"}],
            "integration_stack": ["Slack"],
            "seat_count": 120,
            "contract_end": None,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "role_type": "decision_maker",
            "industry": "Software",
            "primary_workflow": "project management",
            "sentiment_direction": "down",
            "score_components": {"base": 1},
        }]

    async def _fake_fetch_affiliate_partners(pool):
        return {"by_product": {}, "by_category": {}}

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        return {"subject": "test", "body": "<p>body</p>", "cta": "Book time"}

    async def _fake_blog_posts(*args, **kwargs):
        return [{
            "title": "Monday.com vs Asana for pricing pressure",
            "url": "https://atlas.test/blog/monday-asana-pricing",
            "topic_type": "vendor_showdown",
        }]

    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(mod, "_fetch_affiliate_partners", _fake_fetch_affiliate_partners)
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)

    pool = _FakePool({})

    result = await mod._generate_churning_company_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
        company_filter=None,
    )

    assert result["generated"] > 0
    assert result["skipped_no_partner"] == 0
    assert result["generated_without_partner"] == 1


@pytest.mark.asyncio
async def test_generate_churning_company_campaigns_attaches_comparison_asset(monkeypatch):
    captured_payloads = []

    async def _fake_fetch_opportunities(
        pool,
        min_score,
        limit,
        vendor_filter=None,
        company_filter=None,
        dm_only=False,
    ):
        return [{
            "review_id": "rev-1",
            "reviewer_company": "Acme Co",
            "vendor_name": "Monday.com",
            "product_category": "Project Management",
            "opportunity_score": 88,
            "urgency": 8,
            "pain_json": [{"category": "pricing", "severity": "primary"}],
            "competitors": [{"name": "Asana", "reason": "lower cost"}],
            "quotable_phrases": [{"text": "pricing keeps going up"}],
            "feature_gaps": [{"feature": "automation"}],
            "integration_stack": ["Slack"],
            "seat_count": 120,
            "contract_end": None,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "role_type": "decision_maker",
            "industry": "Software",
            "primary_workflow": "project management",
            "sentiment_direction": "down",
            "score_components": {"base": 1},
        }]

    async def _fake_match_products(**kwargs):
        return [{
            "vendor_name": "ClickUp",
            "score": 91.0,
            "reasoning": "Addresses pain: pricing; 4 companies switched from Monday.com",
            "profile_summary": "Lower-cost project management alternative.",
        }]

    async def _fake_fetch_affiliate_partners(pool):
        return {
            "by_product": {
                "clickup": {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "product_name": "ClickUp",
                    "affiliate_url": "https://partners.test/clickup",
                },
            },
            "by_category": {},
        }

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        captured_payloads.append(payload)
        return {"subject": "test", "body": "<p>body</p>", "cta": "Book time"}

    async def _fake_blog_posts(*args, **kwargs):
        return [
            {
                "title": "Monday.com vs ClickUp for pricing pressure",
                "url": "https://atlas.test/blog/monday-clickup-pricing",
                "topic_type": "vendor_showdown",
            },
            {
                "title": "Project management pricing reality check",
                "url": "https://atlas.test/blog/pricing-reality",
                "topic_type": "pricing_reality_check",
            },
        ]

    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.product_matching.match_products",
        _fake_match_products,
    )
    monkeypatch.setattr(mod, "_fetch_affiliate_partners", _fake_fetch_affiliate_partners)
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)

    pool = _FakePool({})

    result = await mod._generate_churning_company_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
        company_filter=None,
    )

    payload = captured_payloads[0]

    assert payload["comparison_asset"]["qualified"] is True
    assert payload["comparison_asset"]["alternative_vendor"] == "ClickUp"
    assert payload["reasoning_anchor_examples"]["outlier_or_named_account"][0]["reviewer_company"] == "Acme Co"
    assert payload["reasoning_witness_highlights"][0]["witness_id"] == "campaign_witness:rev-1:0"
    assert payload["reasoning_reference_ids"]["witness_ids"] == ["campaign_witness:rev-1:0"]
    assert payload["selling"]["primary_blog_post"]["url"] == (
        "https://atlas.test/blog/monday-clickup-pricing"
    )
    assert payload["selling"]["affiliate_url"] == "https://partners.test/clickup"
    assert payload["selling"]["blog_posts"][0]["topic_type"] == "vendor_showdown"
    assert result["generated"] > 0


@pytest.mark.asyncio
async def test_generate_churning_company_campaigns_skips_unqualified_without_blog(monkeypatch):
    generated_calls = {"count": 0}

    async def _fake_fetch_opportunities(
        pool,
        min_score,
        limit,
        vendor_filter=None,
        company_filter=None,
        dm_only=False,
    ):
        return [{
            "review_id": "rev-1",
            "reviewer_company": "Acme Co",
            "vendor_name": "Monday.com",
            "product_category": "Project Management",
            "opportunity_score": 88,
            "urgency": 8,
            "pain_json": [{"category": "pricing", "severity": "primary"}],
            "competitors": [{"name": "Asana", "reason": "lower cost"}],
            "quotable_phrases": [{"text": "pricing keeps going up"}],
            "feature_gaps": [{"feature": "automation"}],
            "integration_stack": ["Slack"],
            "seat_count": 120,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "role_type": "decision_maker",
            "industry": "Software",
            "score_components": {"base": 1},
        }]

    async def _fake_match_products(**kwargs):
        return [{"vendor_name": "ClickUp", "score": 91.0, "reasoning": "pricing fit"}]

    async def _fake_fetch_affiliate_partners(pool):
        return {"by_product": {}, "by_category": {}}

    async def _fake_blog_posts(*args, **kwargs):
        return []

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        generated_calls["count"] += 1
        return {"subject": "test", "body": "<p>body</p>", "cta": "Book time"}

    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.product_matching.match_products",
        _fake_match_products,
    )
    monkeypatch.setattr(mod, "_fetch_affiliate_partners", _fake_fetch_affiliate_partners)
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "require_primary_blog_post", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "require_display_safe_company", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "min_pain_categories", 1, raising=False)

    pool = _FakePool({})

    result = await mod._generate_churning_company_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
        company_filter=None,
    )

    assert result["generated"] == 0
    assert result["failed"] == 0
    assert result["companies"] == 0
    assert result["candidate_companies"] == 1
    assert result["skipped_unqualified"] == 1
    assert result["qualification_summary"]["primary_blog_post"] == 1
    assert generated_calls["count"] == 0


@pytest.mark.asyncio
async def test_generate_churning_company_campaigns_prefers_accounts_in_motion(monkeypatch):
    captured_payloads = []

    async def _fake_fetch_opportunities(*args, **kwargs):
        raise AssertionError("review fallback should not be used when accounts_in_motion is available")

    async def _fake_fetch_affiliate_partners(pool):
        return {"by_product": {}, "by_category": {}}

    async def _fake_match_products(**kwargs):
        return [{"vendor_name": "Basecamp", "score": 91.0, "reasoning": "generic match"}]

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        captured_payloads.append(payload)
        return {"subject": "test", "body": "<p>body</p>", "cta": "Book time"}

    async def _fake_blog_posts(*args, **kwargs):
        return [{
            "title": "Monday.com vs ClickUp for pricing pressure",
            "url": "https://atlas.test/blog/monday-clickup-pricing",
            "topic_type": "vendor_showdown",
        }]

    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(mod, "_fetch_affiliate_partners", _fake_fetch_affiliate_partners)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.product_matching.match_products",
        _fake_match_products,
    )
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)

    pool = _FakePool(
        {},
        fetch_rows=[{
            "vendor_filter": "Monday.com",
            "intelligence_data": {
                "vendor": "Monday.com",
                "category": "Project Management",
                "feature_gaps": [{"feature": "automation", "mentions": 3}],
                "accounts": [{
                    "company": "Acme Co",
                    "opportunity_score": 88,
                    "urgency": 8,
                    "pain_category": "pricing",
                    "alternatives_considering": ["ClickUp"],
                    "top_quote": "pricing keeps going up",
                    "seat_count": 120,
                    "buying_stage": "evaluation",
                    "role_level": "economic_buyer",
                    "title": "VP Operations",
                    "industry": "Software",
                    "company_size": "51-200 employees",
                    "score_components": {"urgency": 18},
                }],
            },
        }],
    )

    result = await mod._generate_churning_company_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
        company_filter=None,
    )

    assert result["generated"] > 0
    assert result["opportunity_source"] == "accounts_in_motion"
    assert result["companies"] == 1
    assert captured_payloads[0]["comparison_asset"]["alternative_vendor"] == "ClickUp"


@pytest.mark.asyncio
async def test_list_churning_company_review_candidates_returns_mid_band_named_accounts(monkeypatch):
    async def _fake_fetch_accounts_in_motion(*args, **kwargs):
        return [{
            "review_id": "rev-1",
            "vendor_name": "Monday.com",
            "reviewer_company": "Acme Co",
            "product_category": "Project Management",
            "opportunity_score": 67,
            "urgency": 8,
            "pain_json": [{"category": "pricing", "severity": "primary"}],
            "competitors": [{"name": "ClickUp", "reason": "lower cost"}],
            "quotable_phrases": [{"text": "pricing keeps going up"}],
            "feature_gaps": [{"feature": "automation"}],
            "integration_stack": [],
            "seat_count": 120,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "role_type": "economic_buyer",
            "industry": "Software",
            "reviewer_title": "VP Operations",
            "company_size_raw": "51-200 employees",
            "score_components": {"urgency": 18},
            "opportunity_source": "accounts_in_motion",
        }]

    async def _fake_fetch_affiliate_partners(pool):
        return {"by_product": {}, "by_category": {}}

    async def _fake_match_products(**kwargs):
        return [{"vendor_name": "Basecamp", "score": 90.0, "reasoning": "generic recommendation"}]

    async def _fake_blog_posts(*args, **kwargs):
        return [{
            "title": "Monday.com vs ClickUp for pricing pressure",
            "url": "https://atlas.test/blog/monday-clickup-pricing",
            "topic_type": "vendor_showdown",
        }]

    monkeypatch.setattr(mod, "_fetch_accounts_in_motion_opportunities", _fake_fetch_accounts_in_motion)
    monkeypatch.setattr(mod, "_fetch_affiliate_partners", _fake_fetch_affiliate_partners)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.product_matching.match_products",
        _fake_match_products,
    )

    result = await mod.list_churning_company_review_candidates(
        _FakePool({}),
        min_score=55,
        max_score=69,
        limit=10,
    )

    assert result["count"] == 1
    candidate = result["candidates"][0]
    assert candidate["company_name"] == "Acme Co"
    assert candidate["comparison_asset"]["alternative_vendor"] == "ClickUp"
    assert candidate["reasoning_anchor_examples"]["outlier_or_named_account"][0]["reviewer_company"] == "Acme Co"
    assert candidate["reasoning_reference_ids"]["witness_ids"] == ["campaign_witness:rev-1:0"]
    assert candidate["qualification"]["qualified"] is True
    assert candidate["generate_request"]["min_score"] == 67
    assert result["summary"]["qualified"] == 1


@pytest.mark.asyncio
async def test_list_churning_company_review_candidates_tracks_missing_requirements(monkeypatch):
    async def _fake_fetch_accounts_in_motion(*args, **kwargs):
        return [{
            "review_id": "rev-1",
            "vendor_name": "Monday.com",
            "reviewer_company": "Acme Co",
            "product_category": "Project Management",
            "opportunity_score": 61,
            "urgency": 8,
            "pain_json": [{"category": "pricing", "severity": "primary"}],
            "competitors": [{"name": "ClickUp", "reason": "lower cost"}],
            "quotable_phrases": [{"text": "pricing keeps going up"}],
            "feature_gaps": [],
            "integration_stack": [],
            "seat_count": 120,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "role_type": "economic_buyer",
            "industry": "Software",
            "reviewer_title": "VP Operations",
            "company_size_raw": "51-200 employees",
            "score_components": {"urgency": 18},
            "opportunity_source": "accounts_in_motion",
        }]

    async def _fake_fetch_affiliate_partners(pool):
        return {"by_product": {}, "by_category": {}}

    async def _fake_blog_posts(*args, **kwargs):
        return []

    monkeypatch.setattr(mod, "_fetch_accounts_in_motion_opportunities", _fake_fetch_accounts_in_motion)
    monkeypatch.setattr(mod, "_fetch_affiliate_partners", _fake_fetch_affiliate_partners)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)

    result = await mod.list_churning_company_review_candidates(
        _FakePool({}),
        min_score=55,
        max_score=69,
        limit=10,
        qualified_only=False,
    )

    assert result["count"] == 1
    assert result["candidates"][0]["qualification"]["qualified"] is False
    assert result["summary"]["unqualified"] == 1
    assert result["summary"]["missing_requirements"]["primary_blog_post"] == 1


@pytest.mark.asyncio
async def test_generate_vendor_campaigns_bypass_briefing_gate(monkeypatch):
    async def _fake_fetch_vendor_targets(pool, vendor_filter):
        return [{
            "company_name": "Asana",
            "contact_name": "Alex",
            "contact_role": "VP Customer Success",
            "contact_email": "alex@asana.test",
            "tier": "report",
            "products_tracked": ["Asana"],
        }]

    async def _fake_fetch_opportunities(pool, min_score, limit, company_filter=None, dm_only=False):
        return [{
            "vendor_name": "Asana",
            "review_id": "rev-1",
            "product_category": "Project Management",
            "opportunity_score": 91,
            "urgency": 9,
            "pain_json": [{"category": "support"}],
            "competitors": [{"name": "ClickUp", "reason": "automation"}],
            "feature_gaps": [{"feature": "reporting"}],
            "quotable_phrases": [{"text": "support has slowed down"}],
            "contract_end": None,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "seat_count": 120,
            "industry": "Software",
            "score_components": {"base": 1},
        }]

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        return {"subject": "test", "body": "<p>body</p>", "cta": "Book time"}

    async def _fake_trend(pool, vendor_name, products=None):
        return "increasing"

    async def _fake_blog_posts(*args, **kwargs):
        return []

    monkeypatch.setattr(mod, "_fetch_vendor_targets", _fake_fetch_vendor_targets)
    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_compute_vendor_trend", _fake_trend)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)

    pool = _FakePool({}, briefing_exists=False)

    result_default = await mod._generate_vendor_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
    )
    result_bypass = await mod._generate_vendor_campaigns(
        pool,
        min_score=50,
        limit=1,
        vendor_filter=None,
        bypass_briefing_gate=True,
    )

    assert result_default["generated"] == 0
    assert result_bypass["generated"] > 0


@pytest.mark.asyncio
async def test_generate_content_appends_signoff_when_missing(monkeypatch):
    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        return json.dumps({
            "subject": "CrowdStrike vs SentinelOne",
            "body": "<p>Hi there,</p><p>Here is the comparison.</p>",
            "cta": "Read the analysis",
        })

    monkeypatch.setattr(mod, "_call_llm", _fake_call_llm)

    payload = {
        "channel": "email_cold",
        "selling": {
            "sender_name": "Juan Canfield",
            "sender_title": "Founder",
            "sender_company": "Atlas Intel",
        },
    }

    result = await mod._generate_content(
        llm=object(),
        system_prompt="skill",
        payload=payload,
        max_tokens=256,
        temperature=0.1,
    )

    assert result is not None
    assert "Best,<br>Juan Canfield<br>Founder, Atlas Intel" in result["body"]


@pytest.mark.asyncio
async def test_generate_content_retries_when_anchor_backed_context_is_ignored(monkeypatch):
    responses = [
        json.dumps({
            "subject": "Renewal pressure",
            "body": (
                "<p>Teams are seeing broad pressure right now across pricing, support, and evaluation behavior. "
                "We are seeing the pattern show up in multiple accounts and the signal mix is getting harder to dismiss.</p>"
                "<p>If you are still seeing this in active opportunities, I can share the pattern we are tracking and where it is showing up first.</p>"
            ),
            "cta": "Book time",
        }),
        json.dumps({
            "subject": "Renewal pressure",
            "body": (
                "<p>Teams are hitting a $200k/year renewal flashpoint in Q2, which makes the pricing risk concrete instead of theoretical. "
                "That timing window is where we keep seeing the signal compress into active evaluation.</p>"
                "<p>Freshdesk keeps showing up in those pricing conversations, so the competitive angle is now visible enough to act on before the renewal locks in.</p>"
            ),
            "cta": "Book time",
        }),
    ]

    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        return responses.pop(0)

    monkeypatch.setattr(mod, "_call_llm", _fake_call_llm)

    payload = {
        "channel": "email_followup",
        "briefing_context": {
            "reasoning_anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "Hack Club said Slack tried to charge $200k/year at Q2 renewal.",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    },
                ],
            },
            "reasoning_witness_highlights": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "Hack Club said Slack tried to charge $200k/year at Q2 renewal.",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Freshdesk",
                    "pain_category": "pricing",
                },
            ],
            "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
        },
        "selling": {
            "sender_name": "Juan Canfield",
            "sender_title": "Founder",
            "sender_company": "Atlas Intel",
        },
    }

    result = await mod._generate_content(
        llm=object(),
        system_prompt="skill",
        payload=payload,
        max_tokens=256,
        temperature=0.1,
    )

    assert result is not None
    assert "$200k/year" in result["body"]
    assert "Q2" in result["body"]


@pytest.mark.asyncio
async def test_generate_content_retries_when_top_level_anchor_context_is_ignored(monkeypatch):
    responses = [
        json.dumps({
            "subject": "Switch signal",
            "body": (
                "<p>Teams are seeing broad commercial pressure and the pattern is widening.</p>"
                "<p>We can share the signal map if it helps.</p>"
            ),
            "cta": "Book time",
        }),
        json.dumps({
            "subject": "Switch signal",
            "body": (
                "<p>Teams are running into a 120-seat pricing problem inside a 30-day evaluation window.</p>"
                "<p>ClickUp is showing up in that workflow-specific comparison often enough to sharpen the next step.</p>"
            ),
            "cta": "Book time",
        }),
    ]

    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        return responses.pop(0)

    monkeypatch.setattr(mod, "_call_llm", _fake_call_llm)

    payload = {
        "channel": "email_followup",
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [
                {
                    "witness_id": "campaign_witness:rev-1:0",
                    "excerpt_text": "pricing keeps going up",
                    "reviewer_company": "Acme Co",
                    "time_anchor": "30 days",
                    "numeric_literals": {"seat_count": ["120"]},
                    "competitor": "ClickUp",
                    "pain_category": "pricing",
                },
            ],
        },
        "reasoning_witness_highlights": [
            {
                "witness_id": "campaign_witness:rev-1:0",
                "excerpt_text": "pricing keeps going up",
                "reviewer_company": "Acme Co",
                "time_anchor": "30 days",
                "numeric_literals": {"seat_count": ["120"]},
                "competitor": "ClickUp",
                "pain_category": "pricing",
            },
        ],
        "reasoning_reference_ids": {"witness_ids": ["campaign_witness:rev-1:0"]},
        "selling": {
            "sender_name": "Juan Canfield",
            "sender_title": "Founder",
            "sender_company": "Atlas Intel",
        },
    }

    result = await mod._generate_content(
        llm=object(),
        system_prompt="skill",
        payload=payload,
        max_tokens=256,
        temperature=0.1,
    )

    assert result is not None
    assert "120-seat" in result["body"]
    assert "30-day" in result["body"]
    assert "ClickUp" in result["body"]


@pytest.mark.asyncio
async def test_generate_content_records_generation_audit_on_success(monkeypatch):
    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        return json.dumps({
            "subject": "Renewal pressure",
            "body": (
                "<p>The Q2 renewal now carries a $200k/year pricing issue.</p>"
                "<p>Freshdesk is showing up in those active evaluations.</p>"
            ),
            "cta": "Book time",
        })

    monkeypatch.setattr(mod, "_call_llm", _fake_call_llm)

    payload = {
        "channel": "email_followup",
        "briefing_context": {
            "reasoning_anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "Hack Club said Slack tried to charge $200k/year at Q2 renewal.",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    },
                ],
            },
            "reasoning_witness_highlights": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "Hack Club said Slack tried to charge $200k/year at Q2 renewal.",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Freshdesk",
                    "pain_category": "pricing",
                },
            ],
            "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
        },
        "selling": {
            "sender_name": "Juan Canfield",
            "sender_title": "Founder",
            "sender_company": "Atlas Intel",
        },
    }

    result = await mod._generate_content(
        llm=object(),
        system_prompt="skill",
        payload=payload,
        max_tokens=256,
        temperature=0.1,
    )

    assert result is not None
    assert payload["_generation_audit"]["status"] == "succeeded"
    assert payload["_generation_audit"]["specificity"]["status"] == "pass"
    assert payload["_campaign_specificity_context"]["reference_ids"]["witness_ids"] == ["witness:r1:0"]


def test_campaign_storage_metadata_includes_specificity_context_and_audit():
    payload = {
        "_campaign_specificity_context": {
            "anchor_examples": {
                "outlier_or_named_account": [
                    {"witness_id": "witness:r1:0", "excerpt_text": "a customer hit a $200k/year renewal"}
                ],
            },
            "witness_highlights": [
                {"witness_id": "witness:r1:0", "excerpt_text": "a customer hit a $200k/year renewal"}
            ],
            "reference_ids": {"witness_ids": ["witness:r1:0"]},
        },
        "_generation_audit": {
            "status": "succeeded",
            "specificity": {
                "status": "pass",
                "blocking_issues": [],
                "warnings": ["timing anchor exists but content does not mention the live trigger window"],
            },
        },
    }

    metadata = mod._campaign_storage_metadata(payload)

    assert metadata["reasoning_anchor_examples"]["outlier_or_named_account"][0]["witness_id"] == "witness:r1:0"
    assert metadata["generation_audit"]["status"] == "succeeded"
    assert metadata["latest_specificity_audit"]["status"] == "pass"


# ---------------------------------------------------------------------------
# _briefing_context_from_data: new fields from Sources 10-12
# ---------------------------------------------------------------------------

def test_briefing_context_extracts_account_pressure_metrics():
    import json
    briefing_data = json.dumps({
        "headline": "Churn pressure rising.",
        "account_pressure_metrics": {
            "total_accounts": 5,
            "high_intent_count": 3,
            "active_eval_signal_count": 2,
            "decision_maker_count": 1,
        },
    })

    context = mod._briefing_context_from_data(briefing_data)

    assert context["account_high_intent_count"] == 3
    assert context["account_active_eval_count"] == 2


def test_briefing_context_omits_zero_account_metrics():
    import json
    briefing_data = json.dumps({
        "account_pressure_metrics": {
            "total_accounts": 0,
            "high_intent_count": 0,
            "active_eval_signal_count": 0,
        },
    })

    context = mod._briefing_context_from_data(briefing_data)

    assert "account_high_intent_count" not in context
    assert "account_active_eval_count" not in context


def test_briefing_context_extracts_timing_triggers():
    import json
    briefing_data = json.dumps({
        "headline": "Timing window active.",
        "priority_timing_triggers": ["Q2 renewal approaching", "Price increase signal"],
    })

    context = mod._briefing_context_from_data(briefing_data)

    assert context["priority_timing_triggers"] == [
        "Q2 renewal approaching",
        "Price increase signal",
    ]


def test_briefing_context_extracts_buyer_profiles():
    import json
    briefing_data = json.dumps({
        "headline": "Buyer signals.",
        "buyer_profiles": [
            {"role_type": "economic_buyer", "buying_stage": "renewal_decision", "avg_urgency": 8.5},
            {"role_type": "evaluator", "buying_stage": "evaluation", "avg_urgency": 7.2},
            {"role_type": "end_user", "buying_stage": "post_purchase", "avg_urgency": 3.1},
        ],
    })

    context = mod._briefing_context_from_data(briefing_data)

    profiles = context["top_buyer_profiles"]
    # Only top 2 extracted
    assert len(profiles) == 2
    assert profiles[0]["role_type"] == "economic_buyer"
    assert profiles[0]["buying_stage"] == "renewal_decision"
    assert profiles[0]["avg_urgency"] == 8.5


def test_briefing_context_extracts_competitive_dynamics():
    import json
    # Use canonical dict shape for switch_reasons (as built by build_displacement_dynamics)
    briefing_data = json.dumps({
        "headline": "Competitive pressure.",
        "competitive_dynamics": {
            "pairs": [
                {
                    "challenger": "Zoho",
                    "battle_summary": "Zoho wins on price in SMB.",
                    "switch_reasons": [
                        {"reason": "lower cost", "reason_category": "pricing", "mention_count": 28},
                        {"reason": "simpler UI", "reason_category": "ux", "mention_count": 15},
                    ],
                },
                {
                    "challenger": "Pipedrive",
                    "battle_summary": "Pipedrive preferred for sales-led motions.",
                    "switch_reasons": [
                        {"reason": "better pipeline UX", "reason_category": "ux", "mention_count": 10}
                    ],
                },
            ]
        },
    })

    context = mod._briefing_context_from_data(briefing_data)

    dyn = context["competitive_dynamics"]
    assert len(dyn) == 2
    assert dyn[0]["challenger"] == "Zoho"
    assert dyn[0]["battle_summary"] == "Zoho wins on price in SMB."
    # switch_reasons in campaign context are extracted as plain reason strings
    assert dyn[0]["switch_reasons"] == ["lower cost", "simpler UI"]


def test_briefing_context_extracts_pain_urgency():
    import json
    briefing_data = json.dumps({
        "headline": "Pain signals.",
        "pain_breakdown": [
            {"category": "pricing", "count": 80, "avg_urgency": 7.8},
            {"category": "support", "count": 40, "avg_urgency": 5.5},
            {"category": "features", "count": 20},
        ],
    })

    context = mod._briefing_context_from_data(briefing_data)

    urgency = context["pain_urgency"]
    # Only entries with avg_urgency included
    assert len(urgency) == 2
    assert urgency[0]["category"] == "pricing"
    assert urgency[0]["avg_urgency"] == 7.8
    # features has no avg_urgency, excluded
    categories = [u["category"] for u in urgency]
    assert "features" not in categories


def test_briefing_context_surfaces_sanitized_reasoning_anchor_examples():
    briefing_data = json.dumps({
        "reasoning_anchor_examples": {
            "outlier_or_named_account": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "Hack Club said Slack tried to charge $200k/year at Q2 renewal.",
                    "reviewer_company": "Hack Club",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Google Chat",
                    "pain_category": "pricing",
                },
            ],
        },
        "reasoning_witness_highlights": [
            {
                "witness_id": "witness:r1:0",
                "excerpt_text": "Hack Club said Slack tried to charge $200k/year at Q2 renewal.",
                "reviewer_company": "Hack Club",
                "time_anchor": "Q2 renewal",
                "numeric_literals": {"currency_mentions": ["$200k/year"]},
                "competitor": "Google Chat",
                "pain_category": "pricing",
            },
        ],
        "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
    })

    context = mod._briefing_context_from_data(briefing_data)

    anchor = context["reasoning_anchor_examples"]["outlier_or_named_account"][0]
    assert anchor["excerpt_text"].startswith("a customer said Slack")
    assert "reviewer_company" not in anchor
    assert context["reasoning_reference_ids"]["witness_ids"] == ["witness:r1:0"]


# ---------------------------------------------------------------------------
# Phase 5: reasoning_context enrichment
# ---------------------------------------------------------------------------

def test_reasoning_context_includes_phase3_fields():
    """When synthesis has Phase 3 contracts, reasoning_context should include
    why_they_stay, timing, switch_triggers, confidence_limits, account_summary."""
    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import SynthesisView

    raw = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "medium",
                    "summary": "Pricing pressure from recent hike",
                    "key_signals": ["40% increase in complaints"],
                    "what_would_weaken_thesis": [
                        {"condition": "Price rollback", "signal_source": "temporal", "monitorable": True},
                    ],
                    "data_gaps": [],
                },
                "timing_intelligence": {
                    "best_timing_window": "Q2 renewal cycle",
                    "immediate_triggers": [
                        {"type": "deadline", "trigger": "Q2 renewal"},
                        {"type": "spike", "trigger": "Complaint surge"},
                    ],
                    "confidence": "medium",
                },
                "why_they_stay": {
                    "summary": "Retention anchored by integrations",
                    "strengths": [
                        {"area": "integrations", "evidence": "Broad ecosystem"},
                        {"area": "brand", "evidence": "Brand loyalty"},
                    ],
                },
                "confidence_posture": {
                    "overall": "medium",
                    "limits": ["thin enterprise sample"],
                },
            },
            "displacement_reasoning": {
                "switch_triggers": [
                    {"type": "deadline", "description": "Q2 contract renewal"},
                    {"type": "spike", "description": "March complaint surge"},
                ],
            },
            "account_reasoning": {
                "market_summary": "Active evaluation concentrated in mid-market ops teams.",
            },
        },
    }
    # Simulate what the campaign generation code does
    from datetime import date
    view = SynthesisView("TestVendor", raw, schema_version="v2", as_of_date=date(2026, 3, 28))

    cn = view.section("causal_narrative")
    wedge = view.primary_wedge
    wedge_label = wedge.value if wedge else cn.get("primary_wedge", "")

    reasoning_ctx = {
        "wedge": wedge_label,
        "confidence": view.confidence("causal_narrative"),
        "summary": cn.get("summary", ""),
        "key_signals": cn.get("key_signals", []),
    }

    wts = view.why_they_stay
    if wts:
        reasoning_ctx["why_they_stay"] = {
            "summary": wts.get("summary", ""),
            "strengths": [
                {"area": s.get("area", ""), "evidence": s.get("evidence", "")}
                for s in wts.get("strengths", []) if isinstance(s, dict)
            ][:5],
        }
    timing = view.section("timing_intelligence")
    if timing:
        reasoning_ctx["timing"] = {
            "best_window": timing.get("best_timing_window", ""),
            "trigger_count": len(timing.get("immediate_triggers") or []),
        }
    triggers = view.switch_triggers
    if triggers:
        reasoning_ctx["switch_triggers"] = [
            {"type": t.get("type", ""), "description": t.get("description", "")}
            for t in triggers[:3]
        ]
    cp = view.confidence_posture
    if cp and cp.get("limits"):
        reasoning_ctx["confidence_limits"] = cp["limits"]
    acct = view.contract("account_reasoning")
    if acct and acct.get("market_summary"):
        reasoning_ctx["account_summary"] = acct["market_summary"]

    # Assertions
    assert reasoning_ctx["wedge"] == "price_squeeze"
    assert reasoning_ctx["why_they_stay"]["summary"] == "Retention anchored by integrations"
    assert len(reasoning_ctx["why_they_stay"]["strengths"]) == 2
    assert reasoning_ctx["timing"]["best_window"] == "Q2 renewal cycle"
    assert reasoning_ctx["timing"]["trigger_count"] == 2
    assert len(reasoning_ctx["switch_triggers"]) == 2
    assert reasoning_ctx["confidence_limits"] == ["thin enterprise sample"]
    assert reasoning_ctx["account_summary"] == "Active evaluation concentrated in mid-market ops teams."


def test_challenger_ctx_includes_incumbent_reasoning():
    """When incumbent views have Phase 3 contracts, challenger_ctx should
    include incumbent_reasoning with per-vendor summaries."""
    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import SynthesisView

    inc_raw = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "medium",
                    "summary": "Incumbents losing on pricing",
                },
                "why_they_stay": {
                    "summary": "Ecosystem lock-in keeps customers",
                },
            },
            "displacement_reasoning": {
                "switch_triggers": [
                    {"type": "deadline", "description": "Q2 renewal"},
                ],
            },
        },
    }
    from datetime import date
    view = SynthesisView("IncumbentCRM", inc_raw, "v2", date(2026, 3, 28))

    # Simulate challenger path logic
    by_archetype = {}
    incumbent_reasoning = {}
    wedge = view.primary_wedge
    label = wedge.value if wedge else view.section("causal_narrative").get("primary_wedge", "")
    if label:
        by_archetype.setdefault(label, []).append("IncumbentCRM")

    cn = view.section("causal_narrative")
    inc_summary = {
        "wedge": label,
        "summary": cn.get("summary", ""),
    }
    wts = view.why_they_stay
    if wts:
        inc_summary["why_they_stay"] = wts.get("summary", "")
    triggers = view.switch_triggers
    if triggers:
        inc_summary["switch_triggers"] = [t.get("type", "") for t in triggers[:3]]
    if inc_summary.get("summary"):
        incumbent_reasoning["IncumbentCRM"] = inc_summary

    assert by_archetype == {"price_squeeze": ["IncumbentCRM"]}
    assert "IncumbentCRM" in incumbent_reasoning
    assert incumbent_reasoning["IncumbentCRM"]["why_they_stay"] == "Ecosystem lock-in keeps customers"
    assert incumbent_reasoning["IncumbentCRM"]["switch_triggers"] == ["deadline"]
