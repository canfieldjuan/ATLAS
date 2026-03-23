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
    assert payload["incumbent_archetypes"]["pricing_shock"] == ["IncumbentCRM"]
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
