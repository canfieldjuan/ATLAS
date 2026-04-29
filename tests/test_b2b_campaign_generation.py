import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
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


@pytest.mark.asyncio
async def test_compute_vendor_trend_reads_vendor_mentions():
    fetchval = AsyncMock(side_effect=[5, 3])
    pool = SimpleNamespace(fetchval=fetchval)

    result = await mod._compute_vendor_trend(pool, "Zendesk", products=["Zendesk Sell"])

    assert result == "increasing"
    assert len(fetchval.await_args_list) == 2
    current_sql, *current_args = fetchval.await_args_list[0].args
    previous_sql, *previous_args = fetchval.await_args_list[1].args
    assert current_args == ["Zendesk", "Zendesk Sell", 3.0]
    assert previous_args == ["Zendesk", "Zendesk Sell", 3.0]
    assert "COUNT(DISTINCT r.id)" in current_sql
    assert "JOIN b2b_review_vendor_mentions vm" in current_sql
    assert "vm.vendor_name ILIKE '%' || $1 || '%'" in current_sql
    assert "vm.vendor_name ILIKE '%' || $2 || '%'" in current_sql
    assert "COUNT(DISTINCT r.id)" in previous_sql
    assert "JOIN b2b_review_vendor_mentions vm" in previous_sql


def _html_body(*paragraphs: str) -> str:
    return "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)


def _html_body_with_min_words(*paragraphs: str, min_words: int = 80) -> str:
    filler = (
        "We can walk through the evidence, timing, costs, workflow impact, and next step "
        "in a short review so the message stays specific instead of generic."
    )
    parts = [paragraph for paragraph in paragraphs if paragraph]
    while len(" ".join(parts).split()) < min_words:
        parts.append(filler)
    return _html_body(*parts)


def _campaign_reasoning_view(vendor_name: str = "Asana"):
    from datetime import date

    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import SynthesisView

    raw = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "medium",
                    "summary": "Pricing pressure is clustering around renewal moments.",
                    "key_signals": ["Q2 renewal pressure"],
                },
            },
            "displacement_reasoning": {
                "switch_triggers": [
                    {"type": "deadline", "description": "Q2 renewal"},
                ],
            },
            "schema_version": "v2",
        },
        "reference_ids": {"witness_ids": ["witness:r1:0"]},
        "packet_artifacts": {
            "witness_pack": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "A customer hit a $200k/year pricing issue at Q2 renewal.",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "ClickUp",
                    "pain_category": "pricing",
                    "salience_score": 9.8,
                },
            ],
            "section_packets": {
                "anchor_examples": {
                    "outlier_or_named_account": ["witness:r1:0"],
                },
            },
        },
    }
    return SynthesisView(vendor_name, raw, schema_version="v2", as_of_date=date(2026, 3, 30))


def _campaign_sparse_account_preview_view(vendor_name: str = "Asana"):
    from datetime import date

    from atlas_brain.autonomous.tasks._b2b_synthesis_reader import SynthesisView

    raw = {
        "reasoning_contracts": {
            "vendor_core_reasoning": {
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "medium",
                    "summary": "Pricing pressure is clustering around renewal moments.",
                },
            },
            "displacement_reasoning": {
                "switch_triggers": [
                    {"type": "deadline", "description": "Q2 renewal"},
                ],
            },
            "account_reasoning": {
                "confidence": "insufficient",
                "market_summary": "A small set of named accounts is showing early churn pressure.",
                "total_accounts": {"value": 1, "source_id": "accounts:summary:total_accounts"},
                "top_accounts": [
                    {"name": "Concentrix", "source_id": "accounts:company:concentrix"},
                ],
            },
            "schema_version": "v2",
        },
    }
    return SynthesisView(vendor_name, raw, schema_version="v2", as_of_date=date(2026, 3, 30))


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


def test_campaign_vendor_and_challenger_contexts_are_order_stable():
    vendor_signals = [
        {
            "review_id": "r2",
            "urgency": 6,
            "pain_json": [{"category": "Support", "severity": "secondary"}],
            "competitors": [{"name": "ClickUp"}],
            "feature_gaps": [{"feature": "reporting"}],
            "quotable_phrases": ["Second quote"],
            "contract_end": None,
        },
        {
            "review_id": "r1",
            "urgency": 9,
            "pain_json": [{"category": "Pricing", "severity": "primary"}],
            "competitors": [{"name": "Asana"}],
            "feature_gaps": [{"feature": "automation"}],
            "quotable_phrases": ["First quote"],
            "contract_end": "Q2",
        },
    ]

    challenger_signals = [
        {
            "review_id": "c2",
            "buying_stage": "evaluation",
            "role_type": "champion",
            "pain_json": [{"category": "Support", "severity": "secondary"}],
            "vendor_name": "Incumbent B",
            "seat_count": 150,
            "competitors": [{"name": "ChallengerX", "reason": "workflow"}],
            "quotable_phrases": ["Beta quote"],
        },
        {
            "review_id": "c1",
            "buying_stage": "active_purchase",
            "role_type": "decision_maker",
            "pain_json": [{"category": "Pricing", "severity": "primary"}],
            "vendor_name": "Incumbent A",
            "seat_count": 600,
            "competitors": [{"name": "ChallengerX", "reason": "analytics"}],
            "quotable_phrases": ["Alpha quote"],
        },
    ]

    vendor_forward = mod._build_vendor_context("Acme", vendor_signals)
    vendor_reverse = mod._build_vendor_context("Acme", list(reversed(vendor_signals)))
    challenger_forward = mod._build_challenger_context("ChallengerX", challenger_signals)
    challenger_reverse = mod._build_challenger_context("ChallengerX", list(reversed(challenger_signals)))

    assert vendor_forward == vendor_reverse
    assert challenger_forward == challenger_reverse


def test_campaign_company_context_is_order_stable_and_keeps_strongest_pain_severity():
    best = {
        "vendor_name": "LegacyCRM",
        "reviewer_company": "Acme Co",
        "product_category": "CRM",
        "urgency": 8,
        "seat_count": 220,
        "contract_end": "Q2 renewal",
        "decision_timeline": "30-60 days",
        "buying_stage": "evaluation",
        "role_type": "decision_maker",
        "industry": "SaaS",
        "reviewer_title": "VP Ops",
        "company_size_raw": "201-500",
        "primary_workflow": "sales",
        "sentiment_direction": "down",
    }
    opps = [
        {
            "review_id": "o2",
            "pain_json": [{"category": "pricing", "severity": "secondary"}],
            "competitors": [{"name": "Asana", "reason": "lower cost"}],
            "quotable_phrases": ["Second quote"],
            "feature_gaps": [{"feature": "automation"}],
            "integration_stack": ["Slack"],
        },
        {
            "review_id": "o1",
            "pain_json": [{"category": "pricing", "severity": "primary"}],
            "competitors": [{"name": "ClickUp", "reason": "reporting"}],
            "quotable_phrases": ["First quote"],
            "feature_gaps": [{"feature": "analytics"}],
            "integration_stack": ["Salesforce"],
        },
    ]

    context_forward = mod._build_company_context(best, opps)
    context_reverse = mod._build_company_context(best, list(reversed(opps)))

    assert context_forward == context_reverse
    assert context_forward["pain_categories"] == [{"category": "pricing", "severity": "primary"}]


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
            "quotable_phrases": [{"text": "support has slowed down", "phrase_verbatim": True}],
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
async def test_generate_vendor_campaigns_injects_synthesis_anchor_examples(monkeypatch):
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
            "pain_json": [{"category": "pricing"}],
            "competitors": [{"name": "ClickUp", "reason": "pricing"}],
            "feature_gaps": [{"feature": "reporting"}],
            "quotable_phrases": [{"text": "pricing has become a renewal issue", "phrase_verbatim": True}],
            "contract_end": None,
            "decision_timeline": "30 days",
            "buying_stage": "evaluation",
            "seat_count": 120,
            "industry": "Software",
            "score_components": {"base": 1},
        }]

    async def _fake_generate_content(llm, skill_content, payload, max_tokens, temperature):
        captured_payloads.append(payload)
        return {
            "subject": "Renewal pressure",
            "body": "<p>The Q2 renewal now carries a $200k/year pricing issue and ClickUp is showing up.</p>",
            "cta": "Book time",
        }

    async def _fake_trend(pool, vendor_name, products=None):
        return "increasing"

    async def _fake_blog_posts(*args, **kwargs):
        return []

    async def _fake_reasoning_view(*args, **kwargs):
        return _campaign_reasoning_view("Asana")

    monkeypatch.setattr(mod, "_fetch_vendor_targets", _fake_fetch_vendor_targets)
    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(mod, "_generate_content", _fake_generate_content)
    monkeypatch.setattr(mod, "_compute_vendor_trend", _fake_trend)
    monkeypatch.setattr(mod, "_fetch_blog_posts", _fake_blog_posts)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.load_best_reasoning_view",
        _fake_reasoning_view,
    )
    monkeypatch.setattr(
        "atlas_brain.services.llm_router.get_llm",
        lambda workload: SimpleNamespace(model="test-model"),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: _FakeSkillRegistry(),
    )
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", False, raising=False)

    result = await mod._generate_vendor_campaigns(
        _FakePool({}),
        min_score=50,
        limit=1,
        vendor_filter=None,
    )

    payload = captured_payloads[0]
    anchor = payload["reasoning_anchor_examples"]["outlier_or_named_account"][0]
    assert anchor["witness_id"] == "witness:r1:0"
    assert payload["reasoning_witness_highlights"][0]["competitor"] == "ClickUp"
    assert payload["reasoning_reference_ids"]["witness_ids"] == ["witness:r1:0"]
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
            "quotable_phrases": [{"text": "we are evaluating alternatives", "phrase_verbatim": True}],
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
    assert "incumbent_archetypes" not in payload
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
            "quotable_phrases": [{"text": "pricing keeps going up", "phrase_verbatim": True}],
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
            "quotable_phrases": [{"text": "pricing keeps going up", "phrase_verbatim": True}],
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
            "quotable_phrases": [{"text": "pricing keeps going up", "phrase_verbatim": True}],
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

    def _allow_claim(row, *, as_of_date, analysis_window_days):
        row["opportunity_claim"] = {
            "claim_id": "claim-accounts-in-motion",
            "report_allowed": True,
            "render_allowed": True,
        }
        return SimpleNamespace(report_allowed=True)

    monkeypatch.setattr(mod, "_fetch_opportunities", _fake_fetch_opportunities)
    monkeypatch.setattr(mod, "_attach_campaign_opportunity_claim", _allow_claim)
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
            "quotable_phrases": [{"text": "pricing keeps going up", "phrase_verbatim": True}],
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
            "quotable_phrases": [{"text": "pricing keeps going up", "phrase_verbatim": True}],
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
            "quotable_phrases": [{"text": "support has slowed down", "phrase_verbatim": True}],
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
async def test_fetch_opportunities_filters_non_report_safe_account_claims(monkeypatch):
    from atlas_brain.autonomous.tasks import _b2b_shared

    review_id = "11111111-1111-1111-1111-111111111111"
    rows = [{
        "review_id": review_id,
        "vendor_name": "Zendesk",
        "reviewer_company": "Acme Corp",
        "urgency": 9.2,
        "is_dm": True,
        "role_type": "decision_maker",
        "buying_stage": "active_purchase",
        "seat_count": 750,
        "competitors": [{"name": "Intercom", "context": "renewal evaluation"}],
        "quotable_phrases": [{"text": "We are actively evaluating Intercom."}],
        "decision_timeline": "this quarter",
    }]

    async def _fake_read_campaign_opportunities(*args, **kwargs):
        return rows

    monkeypatch.setattr(_b2b_shared, "read_campaign_opportunities", _fake_read_campaign_opportunities)
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    result = await mod._fetch_opportunities(pool, min_score=1, limit=10)

    assert result == []
    claim = mod.build_account_opportunity_claim(
        mod._campaign_opportunity_claim_input(rows[0]),
        as_of_date=date.today(),
        analysis_window_days=90,
    )
    assert claim.report_allowed is False


@pytest.mark.asyncio
async def test_fetch_opportunities_keeps_report_safe_account_claims(monkeypatch):
    from atlas_brain.autonomous.tasks import _b2b_shared

    rows = [{
        "review_id": "11111111-1111-1111-1111-111111111111",
        "vendor_name": "Zendesk",
        "reviewer_company": "Acme Corp",
        "urgency": 9.2,
        "is_dm": True,
        "role_type": "decision_maker",
        "buying_stage": "active_purchase",
        "seat_count": 750,
        "competitors": [{"name": "Intercom", "context": "renewal evaluation"}],
        "quotable_phrases": [{"text": "We are actively evaluating Intercom."}],
        "decision_timeline": "this quarter",
    }]

    async def _fake_read_campaign_opportunities(*args, **kwargs):
        return rows

    def _allow_claim(row, *, as_of_date, analysis_window_days):
        row["opportunity_claim"] = {
            "claim_id": "claim-1",
            "report_allowed": True,
            "render_allowed": True,
        }
        return SimpleNamespace(report_allowed=True)

    monkeypatch.setattr(_b2b_shared, "read_campaign_opportunities", _fake_read_campaign_opportunities)
    monkeypatch.setattr(mod, "_attach_campaign_opportunity_claim", _allow_claim)
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    result = await mod._fetch_opportunities(pool, min_score=1, limit=10)

    assert len(result) == 1
    assert result[0]["opportunity_claim"]["claim_id"] == "claim-1"
    assert result[0]["opportunity_score"] >= 1


@pytest.mark.asyncio
async def test_accounts_in_motion_opportunities_filter_non_report_safe_account_claims():
    review_id = "11111111-1111-1111-1111-111111111111"
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[{
        "vendor_filter": "Zendesk",
        "intelligence_data": {
            "category": "Helpdesk",
            "accounts": [{
                "company": "Acme Corp",
                "opportunity_score": 91,
                "urgency": 9,
                "top_quote": "We are actively evaluating Intercom.",
                "source_reviews": [review_id],
                "alternatives_considering": ["Intercom"],
                "buying_stage": "active_purchase",
                "decision_timeline": "this quarter",
                "role_level": "decision_maker",
            }],
        },
    }]))

    result = await mod._fetch_accounts_in_motion_opportunities(
        pool,
        min_score=1,
        limit=10,
    )

    assert result == []


@pytest.mark.asyncio
async def test_generate_content_appends_signoff_when_missing(monkeypatch):
    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        return json.dumps({
            "subject": "CrowdStrike vs SentinelOne",
            "body": _html_body_with_min_words(
                "Hi there,",
                "Here is the comparison between CrowdStrike and SentinelOne, with the switching pressure showing up in pricing, security workflow fit, and the timing of the current evaluation cycle.",
            ),
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
            "body": _html_body_with_min_words(
                "Teams are hitting a $200k/year renewal flashpoint in Q2, which makes the pricing risk concrete instead of theoretical. That timing window is where we keep seeing the signal compress into active evaluation.",
                "Freshdesk keeps showing up in those pricing conversations, so the competitive angle is now visible enough to act on before the renewal locks in.",
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
            "body": _html_body_with_min_words(
                "Teams are running into a 120-seat pricing problem inside a 30-day evaluation window, which turns the budget issue into an operational decision instead of a vague complaint.",
                "ClickUp is showing up in that workflow-specific comparison often enough to sharpen the next step and give the team a credible alternative during the active evaluation.",
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
            "body": _html_body_with_min_words(
                "The Q2 renewal now carries a $200k/year pricing issue, and the account is revisiting whether the current spend still matches day-to-day usage across the team.",
                "Freshdesk is showing up in those active evaluations, which gives the buying group a real alternative before the renewal window closes.",
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
    assert payload["campaign_proof_terms"][:2] == ["$200k/year", "q2 renewal"]
    assert payload["_campaign_specificity_context"]["reference_ids"]["witness_ids"] == ["witness:r1:0"]


@pytest.mark.asyncio
async def test_generate_content_specificity_retry_names_exact_anchor_terms(monkeypatch):
    user_contents = []
    responses = [
        json.dumps({
            "subject": "Renewal pressure",
            "body": (
                "<p>Teams are seeing broad pressure and the pattern is widening.</p>"
                "<p>We can share the signal map if it helps.</p>"
            ),
            "cta": "Book time",
        }),
        json.dumps({
            "subject": "Renewal pressure",
            "body": _html_body_with_min_words(
                "Teams are hitting a $200k/year renewal issue in Q2 and that is when the signal turns active instead of staying theoretical.",
                "Freshdesk is showing up in those pricing conversations often enough to act on it before the renewal closes the current path.",
            ),
            "cta": "Book time",
        }),
    ]

    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        user_contents.append(user_content)
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
    assert len(user_contents) == 2
    assert "campaign_proof_terms" in user_contents[0]
    assert "$200k/year" in user_contents[1]
    assert "Q2 renewal" in user_contents[1]


@pytest.mark.asyncio
async def test_call_llm_uses_configured_campaign_timeout(monkeypatch):
    import asyncio

    captured = {}

    async def _fake_to_thread(fn, *args, **kwargs):
        return {"response": "ok"}

    async def _fake_wait_for(coro, timeout):
        captured["timeout"] = timeout
        return await coro

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setattr(mod.settings.b2b_campaign, "llm_timeout_seconds", 42.0)

    class _FakeSyncLLM:
        def chat(self, *, messages, max_tokens, temperature):
            return {"response": "unused"}

    result = await mod._call_llm(
        _FakeSyncLLM(),
        "system",
        "user",
        256,
        0.1,
    )

    assert result == "ok"
    assert captured["timeout"] == 42.0


@pytest.mark.asyncio
async def test_call_llm_populates_usage_out(monkeypatch):
    import asyncio

    async def _fake_to_thread(fn, *args, **kwargs):
        return {
            "response": "ok",
            "usage": {"input_tokens": 12, "output_tokens": 4},
            "_trace_meta": {
                "provider_request_id": "req_campaign_123",
                "billable_input_tokens": 9,
            },
        }

    async def _fake_wait_for(coro, timeout):
        return await coro

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(asyncio, "wait_for", _fake_wait_for)

    class _FakeSyncLLM:
        model = "claude-sonnet-4-5"
        name = "anthropic"

        def chat(self, *, messages, max_tokens, temperature):
            return {"response": "unused"}

    usage_out = {}
    result = await mod._call_llm(
        _FakeSyncLLM(),
        "system",
        "user",
        256,
        0.1,
        usage_out=usage_out,
    )

    assert result == "ok"
    assert usage_out["input_tokens"] == 12
    assert usage_out["output_tokens"] == 4
    assert usage_out["billable_input_tokens"] == 9
    assert usage_out["provider"] == "anthropic"
    assert usage_out["model"] == "claude-sonnet-4-5"
    assert usage_out["provider_request_id"] == "req_campaign_123"


@pytest.mark.asyncio
async def test_run_campaign_batch_task_metadata_can_enable_batch(monkeypatch):
    class FakeAnthropicLLM:
        name = "anthropic"
        model = "claude-sonnet-4-5"

    batch_llm = FakeAnthropicLLM()
    batch_calls = []

    monkeypatch.setattr(mod.settings.b2b_churn, "anthropic_batch_enabled", False, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "anthropic_batch_enabled", False, raising=False)
    monkeypatch.setattr(mod.settings.b2b_campaign, "anthropic_batch_detached_enabled", False, raising=False)
    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.mark_batch_fallback_result",
        AsyncMock(),
    )

    async def _fake_run_batch(**kwargs):
        batch_calls.append(kwargs)
        return SimpleNamespace(
            local_batch_id="campaign-batch-1",
            provider_batch_id="provider-campaign-batch-1",
            results_by_custom_id={
                "campaign:1": SimpleNamespace(
                    response_text=json.dumps(
                        {
                            "subject": "Subject",
                            "body": _html_body_with_min_words("Body"),
                            "proof_points": ["Proof"],
                            "cta": "CTA",
                        }
                    ),
                    usage={"input_tokens": 10, "output_tokens": 5},
                    error_text=None,
                )
            },
            submitted_items=1,
            cache_prefiltered_items=0,
            fallback_single_call_items=0,
            completed_items=1,
            failed_items=0,
        )

    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        _fake_run_batch,
    )

    results, metrics = await mod._run_campaign_batch(
        batch_llm,
        "system",
        [
            {
                "custom_id": "campaign:1",
                "artifact_id": "artifact-1",
                "payload": {"channel": "email_cold", "target_mode": "vendor_retention"},
                "channel": "email_cold",
                "max_tokens": 512,
                "temperature": 0.3,
                "trace_metadata": {"vendor_name": "Asana"},
            }
        ],
        run_id="run-campaign-batch",
        task=SimpleNamespace(
            metadata={
                "anthropic_batch_enabled": True,
                "campaign_anthropic_batch_enabled": True,
                "campaign_anthropic_batch_min_items": 1,
            }
        ),
    )

    assert batch_calls
    assert batch_calls[0]["llm"] is batch_llm
    assert batch_calls[0]["min_batch_size"] == 1
    assert metrics["jobs"] == 1
    assert results["campaign:1"]["subject"] == "Subject"


@pytest.mark.asyncio
async def test_generate_content_usage_out_aggregates_retry_attempts(monkeypatch):
    responses = [
        json.dumps({
            "subject": "Renewal pressure",
            "body": "<p>Short body.</p>",
            "cta": "Book time",
        }),
        json.dumps({
            "subject": "Renewal pressure",
            "body": _html_body_with_min_words(
                "Teams are hitting a concrete renewal issue with a $200k/year pricing increase during Q2, and that timing is forcing a real evaluation instead of a soft complaint.",
                "Freshdesk keeps showing up in those conversations often enough to make the next action credible before the renewal closes.",
            ),
            "cta": "Book time",
        }),
    ]
    usage_samples = [
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "billable_input_tokens": 8,
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "provider_request_id": "req_campaign_first",
        },
        {
            "input_tokens": 12,
            "output_tokens": 5,
            "billable_input_tokens": 9,
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "provider_request_id": "req_campaign_second",
        },
    ]

    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature, **kwargs):
        usage_out = kwargs.get("usage_out")
        sample = usage_samples.pop(0)
        if usage_out is not None:
            usage_out.clear()
            usage_out.update(sample)
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
    usage_out = {}

    result = await mod._generate_content(
        llm=object(),
        system_prompt="skill",
        payload=payload,
        max_tokens=256,
        temperature=0.1,
        usage_out=usage_out,
    )

    assert result is not None
    assert usage_out["input_tokens"] == 22
    assert usage_out["output_tokens"] == 9
    assert usage_out["billable_input_tokens"] == 17
    assert usage_out["provider_request_id"] == "req_campaign_second"


@pytest.mark.asyncio
async def test_generate_content_retries_when_body_is_too_short_for_mode(monkeypatch):
    user_contents = []
    responses = [
        json.dumps({
            "subject": "Switch signal",
            "body": "<p>Teams are seeing pressure. We can share details.</p>",
            "cta": "Book time",
        }),
        json.dumps({
            "subject": "Switch signal",
            "body": _html_body_with_min_words(
                "Teams are seeing pricing pressure during the renewal window, and the operating pain is now concrete enough to plan around instead of leaving in abstract budget language.",
                "We can share the workflow-specific comparison and the timing details if that helps your team evaluate next steps with more concrete evidence.",
            ),
            "cta": "Book time",
        }),
    ]

    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        user_contents.append(user_content)
        return responses.pop(0)

    monkeypatch.setattr(mod, "_call_llm", _fake_call_llm)

    payload = {
        "channel": "email_followup",
        "target_mode": "churning_company",
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
    assert len(user_contents) == 2
    assert "MUST be at least 75 words" in user_contents[1]


@pytest.mark.asyncio
async def test_generate_content_blocks_vendor_cold_email_with_competitor_name(monkeypatch):
    responses = [
        json.dumps({
            "subject": "Renewal pressure",
            "body": _html_body_with_min_words(
                "Freshdesk keeps showing up in the active conversations around pricing pressure, and that comparison is becoming hard to ignore during the renewal cycle.",
            ),
            "cta": "Book time",
        }),
        json.dumps({
            "subject": "Renewal pressure",
            "body": _html_body_with_min_words(
                "Freshdesk keeps showing up in the active conversations around pricing pressure, and that comparison is becoming hard to ignore during the renewal cycle.",
            ),
            "cta": "Book time",
        }),
    ]

    async def _fake_call_llm(llm, system_prompt, user_content, max_tokens, temperature):
        return responses.pop(0)

    monkeypatch.setattr(mod, "_call_llm", _fake_call_llm)

    payload = {
        "channel": "email_cold",
        "target_mode": "vendor_retention",
        "competitors_considering": [{"name": "Freshdesk", "reason": "pricing"}],
        "tier": "report",
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

    assert result is None
    assert payload["_generation_audit"]["failure_reason"] == "specificity_gate"
    assert "competitor_name_in_email_cold:Freshdesk" in payload["_generation_audit"]["specificity"]["blocking_issues"]


def test_campaign_prompt_contracts_include_campaign_proof_terms():
    prompt_files = [
        Path("atlas_brain/skills/digest/b2b_campaign_generation.md"),
        Path("atlas_brain/skills/digest/b2b_challenger_outreach.md"),
        Path("atlas_brain/skills/digest/b2b_vendor_outreach.md"),
    ]

    for path in prompt_files:
        content = path.read_text(encoding="utf-8")
        assert "campaign_proof_terms" in content, path.as_posix()
        assert "authoritative when present" in content.lower(), path.as_posix()


def test_campaign_word_limits_are_mode_aware():
    assert mod._campaign_word_limits(
        channel="email_cold",
        target_mode="vendor_retention",
    ) == (50, 125)
    assert mod._campaign_word_limits(
        channel="email_followup",
        target_mode="challenger_intel",
    ) == (75, 150)
    assert mod._campaign_word_limits(
        channel="email_followup",
        target_mode="churning_company",
    ) == (75, 125)
    assert mod._campaign_word_limits(
        channel="linkedin",
        target_mode="churning_company",
    ) == (0, 100)


def test_validate_campaign_content_uses_target_mode_specific_word_limits():
    over_limit = " ".join(f"word{i}" for i in range(130))
    under_limit = " ".join(f"word{i}" for i in range(60))

    _, too_long = mod._validate_campaign_content(
        {
            "subject": "Renewal pressure",
            "body": f"<p>{over_limit}</p>",
            "cta": "Book time",
        },
        "email_followup",
        target_mode="churning_company",
    )
    _, too_short = mod._validate_campaign_content(
        {
            "subject": "Renewal pressure",
            "body": f"<p>{under_limit}</p>",
            "cta": "Book time",
        },
        "email_cold",
        target_mode="churning_company",
    )

    assert too_long["word_count"] == 130
    assert too_long["max_words"] == 125
    assert too_short["word_count"] == 60
    assert too_short["min_words"] == 75


def test_campaign_storage_metadata_includes_specificity_context_and_audit():
    payload = {
        "tier": "report",
        "campaign_proof_terms": ["$200k/year", "q2 renewal"],
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
    assert metadata["campaign_proof_terms"] == ["$200k/year", "q2 renewal"]
    assert metadata["tier"] == "report"
    assert metadata["generation_audit"]["status"] == "succeeded"
    assert metadata["latest_specificity_audit"]["status"] == "pass"
    assert metadata["latest_specificity_audit"]["boundary"] == "generation"


def test_campaign_storage_metadata_includes_product_claim_gate_payloads():
    claim = {
        "claim_id": "claim-1",
        "render_allowed": True,
        "report_allowed": True,
        "confidence": "medium",
        "evidence_posture": "usable",
        "suppression_reason": None,
    }
    sibling_claim = {
        "claim_id": "claim-2",
        "render_allowed": True,
        "report_allowed": True,
    }

    metadata = mod._campaign_storage_metadata({
        "opportunity_claim": claim,
        "opportunity_claims": [sibling_claim, "not-a-claim"],
    })

    assert metadata["opportunity_claim"] == claim
    assert metadata["opportunity_claim_gate"] == {
        "claim_id": "claim-1",
        "render_allowed": True,
        "report_allowed": True,
        "confidence": "medium",
        "evidence_posture": "usable",
        "suppression_reason": None,
    }
    assert metadata["opportunity_claims"] == [sibling_claim]


def test_campaign_payload_report_safe_gate_requires_product_claim_context():
    assert mod._campaign_payload_has_report_safe_product_claim({}) is False
    assert mod._campaign_payload_has_report_safe_product_claim({
        "opportunity_claim": {"claim_id": "claim-1", "report_allowed": False},
    }) is False
    assert mod._campaign_payload_has_report_safe_product_claim({
        "opportunity_claim": {"claim_id": "claim-1", "report_allowed": True},
    }) is True
    assert mod._campaign_payload_has_report_safe_product_claim({
        "opportunity_claims": [
            {"claim_id": "claim-1", "report_allowed": True},
            {"claim_id": "claim-2", "report_allowed": False},
        ],
    }) is False
    assert mod._campaign_payload_has_report_safe_product_claim({
        "opportunity_claims": [
            {"claim_id": "claim-1", "report_allowed": True},
            {"claim_id": "claim-2", "report_allowed": True},
        ],
    }) is True
    assert mod._campaign_payload_has_report_safe_product_claim({
        "opportunity_claim": {"claim_id": "claim-1", "report_allowed": True},
        "opportunity_claims": [
            {"claim_id": "claim-1", "report_allowed": True},
            {"claim_id": "claim-2", "report_allowed": False},
        ],
    }) is False
    assert mod._campaign_payload_has_report_safe_product_claim({
        "opportunity_claim": {"claim_id": "claim-1", "report_allowed": True},
        "opportunity_claim_gate": {"claim_id": "claim-1", "report_allowed": False},
    }) is False


@pytest.mark.asyncio
async def test_store_replayed_campaign_entry_blocks_missing_product_claim_before_db_touch():
    pool = SimpleNamespace(execute=AsyncMock(side_effect=AssertionError("db touched")))

    with pytest.raises(ValueError, match="missing report-safe ProductClaim gate"):
        await mod._store_replayed_campaign_entry(
            pool,
            entry={
                "payload": {"target_mode": "churning_company"},
                "company_name": "Acme Corp",
                "artifact_id": "artifact-1",
            },
            content={"subject": "Subject", "body": "<p>Body</p>", "cta": "Book time"},
            llm_model_name="test-model",
        )

    pool.execute.assert_not_awaited()


def test_build_company_context_carries_best_opportunity_claim():
    claim = {
        "claim_id": "claim-1",
        "render_allowed": True,
        "report_allowed": True,
    }
    row = {
        "review_id": "rev-1",
        "vendor_name": "Zendesk",
        "reviewer_company": "Acme Corp",
        "product_category": "Helpdesk",
        "urgency": 9,
        "pain_json": [{"category": "support", "severity": "primary"}],
        "competitors": [{"name": "Intercom", "reason": "support"}],
        "quotable_phrases": [{"text": "Support is too slow.", "phrase_verbatim": True}],
        "opportunity_claim": claim,
    }

    context = mod._build_company_context(row, [row])

    assert context["opportunity_claim"] == claim


def test_vendor_and_challenger_contexts_carry_opportunity_claim_lists():
    claim = {
        "claim_id": "claim-1",
        "render_allowed": True,
        "report_allowed": True,
    }
    row = {
        "review_id": "rev-1",
        "vendor_name": "Zendesk",
        "urgency": 9,
        "buying_stage": "evaluation",
        "role_type": "decision_maker",
        "pain_json": [{"category": "support", "severity": "primary"}],
        "competitors": [{"name": "Intercom", "reason": "support"}],
        "quotable_phrases": [{"text": "Support is too slow.", "phrase_verbatim": True}],
        "opportunity_claim": claim,
    }

    vendor_context = mod._build_vendor_context("Zendesk", [row])
    challenger_context = mod._build_challenger_context("Intercom", [row])

    assert vendor_context["opportunity_claims"] == [claim]
    assert challenger_context["opportunity_claims"] == [claim]


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
                    "grounding_status": "grounded",
                    "phrase_polarity": "negative",
                    "phrase_subject": "subject_vendor",
                    "phrase_role": "primary_driver",
                    "phrase_verbatim": True,
                    "pain_confidence": "strong",
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
                "grounding_status": "grounded",
                "phrase_polarity": "negative",
                "phrase_subject": "subject_vendor",
                "phrase_role": "primary_driver",
                "phrase_verbatim": True,
                "pain_confidence": "strong",
            },
        ],
        "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
    })

    context = mod._briefing_context_from_data(briefing_data)

    anchor = context["reasoning_anchor_examples"]["outlier_or_named_account"][0]
    assert anchor["excerpt_text"].startswith("a customer said Slack")
    assert "reviewer_company" not in anchor
    assert anchor["grounding_status"] == "grounded"
    assert anchor["phrase_polarity"] == "negative"
    assert anchor["phrase_subject"] == "subject_vendor"
    assert anchor["phrase_role"] == "primary_driver"
    assert anchor["phrase_verbatim"] is True
    assert anchor["pain_confidence"] == "strong"
    assert context["reasoning_reference_ids"]["witness_ids"] == ["witness:r1:0"]


def test_campaign_quote_texts_requires_explicit_verbatim_marker():
    quotes = mod._campaign_quote_texts([
        "legacy string quote",
        {"text": "unmarked dict quote"},
        {"text": "false dict quote", "phrase_verbatim": False},
        {"text": "safe quote", "phrase_verbatim": True},
    ])

    assert quotes == ["safe quote"]


def test_churning_company_anchor_context_drops_unmarked_quotes():
    context = mod._build_churning_company_anchor_context(
        {"vendor_name": "Monday.com"},
        [{
            "review_id": "rev-1",
            "reviewer_company": "Acme Co",
            "vendor_name": "Monday.com",
            "urgency": 8,
            "pain_json": [{"category": "pricing", "severity": "primary"}],
            "competitors": [{"name": "ClickUp", "reason": "lower cost"}],
            "quotable_phrases": [{"text": "legacy quote should not render"}],
        }],
    )

    assert context == {}


def test_inject_reasoning_campaign_context_surfaces_section_disclaimers():
    target = {}
    mod._inject_reasoning_campaign_context(
        target,
        {
            "reasoning_section_disclaimers": {
                "timing_intelligence": "Timing guidance is based on limited direct evidence.",
            },
        },
    )

    assert target["reasoning_section_disclaimers"]["timing_intelligence"]


def test_inject_reasoning_campaign_context_surfaces_atoms_and_delta():
    target = {}
    mod._inject_reasoning_campaign_context(
        target,
        {
            "scope_manifest": {"selection_strategy": "vendor_facet_packet_v1"},
            "theses": [{"thesis_id": "primary_wedge", "summary": "Pricing pressure is clustering"}],
            "timing_windows": [{"window_id": "trigger_1", "start_or_anchor": "Q2 renewal"}],
            "proof_points": [{"label": "switch_volume"}],
            "account_signals": [{"company": "Acme"}],
            "counterevidence": [{"counterevidence_id": "counterevidence_1"}],
            "coverage_limits": [{"coverage_limit_id": "limit_1"}],
            "reasoning_delta": {"changed": True},
        },
    )

    assert target["reasoning_scope_summary"]["selection_strategy"] == "vendor_facet_packet_v1"
    assert target["reasoning_atom_context"]["top_theses"]
    assert target["reasoning_atom_context"]["proof_points"][0]["label"] == "switch_volume"
    assert target["reasoning_delta_summary"]["changed"] is True


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


def test_campaign_account_summary_uses_sparse_preview_when_contract_suppressed():
    view = _campaign_sparse_account_preview_view("Salesforce")

    consumer_context = view.filtered_consumer_context("campaign")
    account_summary = mod._campaign_account_summary_from_consumer_context(
        consumer_context,
    )

    assert account_summary["account_summary"] == (
        "A small set of named accounts is showing early churn pressure."
    )
    assert account_summary["account_summary_source"] == "account_reasoning_preview"
    assert account_summary["priority_account_names"] == ["Concentrix"]
    assert "Early account signal only" in account_summary["account_summary_disclaimer"]


def test_campaign_account_summary_preserves_actionability_fields_from_contract():
    consumer_context = {
        "reasoning_contracts": {
            "account_reasoning": {
                "market_summary": "Two enterprise accounts are evaluating alternatives.",
                "account_actionability_note": (
                    "Mixed confidence: 2 of 6 named accounts are backed by trusted identity anchors."
                ),
                "account_actionability_tier": "mixed",
            },
        },
    }

    account_summary = mod._campaign_account_summary_from_consumer_context(
        consumer_context,
    )

    assert account_summary["account_summary"] == (
        "Two enterprise accounts are evaluating alternatives."
    )
    assert account_summary["account_summary_source"] == "account_reasoning"
    assert account_summary["account_summary_disclaimer"] == (
        "Mixed confidence: 2 of 6 named accounts are backed by trusted identity anchors."
    )
    assert account_summary["account_actionability_tier"] == "mixed"


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
