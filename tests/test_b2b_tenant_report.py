import json
from datetime import date

import pytest

from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
    _canonicalize_vendor,
    _compact_vendor_churn_scores_for_llm,
)
from atlas_brain.autonomous.tasks.b2b_tenant_report import (
    _apply_tenant_report_context,
    _apply_tenant_synthesis_context,
    _build_deterministic_tenant_report,
    _build_deterministic_tenant_report_from_raw,
    _apply_tenant_vendor_context,
    _filter_tenant_payload_for_vendors,
    _merge_tenant_chunk_outputs,
    _tenant_payload_vendor_chunks,
    _tenant_report_chunk_size,
    _tenant_report_data_density,
    _tenant_report_llm_model,
)


def test_tenant_report_llm_model_uses_actual_model():
    assert _tenant_report_llm_model({"model": "openai/gpt-oss-120b"}) == "openai/gpt-oss-120b"


def test_tenant_report_llm_model_defaults_when_missing():
    assert _tenant_report_llm_model({}) == "pipeline_deterministic"
    assert _tenant_report_llm_model(None) == "pipeline_deterministic"


def test_tenant_report_data_density_includes_llm_and_reasoning_counts():
    result = {
        "vendors_analyzed": 12,
        "high_intent_companies": 3,
        "competitive_flows": 9,
        "pain_categories": 4,
        "feature_gaps": 7,
    }
    density = json.loads(
        _tenant_report_data_density(
            result,
            llm_usage={"input_tokens": 123, "output_tokens": 45},
            narrative_evidence_count=5,
            stratified_reasoning_count=8,
            synthesis_contract_vendor_count=4,
        )
    )
    assert density["vendors_analyzed"] == 12
    assert density["narrative_evidence_vendors"] == 5
    assert density["stratified_reasoning_vendors"] == 8
    assert density["synthesis_contract_vendors"] == 4
    assert density["llm_input_tokens"] == 123
    assert density["llm_output_tokens"] == 45


def test_tenant_payload_vendor_chunks_groups_categories_before_splitting(monkeypatch):
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._tenant_report_chunk_size",
        lambda: 4,
    )
    payload = {
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "product_category": "Helpdesk"},
            {"vendor_name": "Intercom", "product_category": "Helpdesk"},
            {"vendor_name": "ClickUp", "product_category": "Project Management"},
            {"vendor_name": "Asana", "product_category": "Project Management"},
            {"vendor_name": "HubSpot", "product_category": "CRM"},
        ]
    }
    chunks = _tenant_payload_vendor_chunks(payload)
    assert chunks == [["Zendesk", "Intercom", "ClickUp", "Asana"], ["HubSpot"]]


def test_tenant_report_chunk_size_uses_smaller_chunks_for_gpt_oss(monkeypatch):
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report.settings.llm.openrouter_reasoning_model",
        "openai/gpt-oss-120b",
    )
    assert _tenant_report_chunk_size() == 3


def test_filter_tenant_payload_for_vendors_scopes_vendor_lists():
    payload = {
        "date": "2026-03-21",
        "data_context": {"x": 1},
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "category_council": {"winner": "Zoho Desk"}},
            {"vendor_name": "HubSpot"},
        ],
        "competitive_displacement": [
            {"from_vendor": "Zendesk", "to_vendor": "Freshdesk"},
            {"from_vendor": "HubSpot", "to_vendor": "Salesforce"},
        ],
        "high_intent_companies": [
            {"vendor": "Zendesk", "company": "Acme"},
            {"vendor": "HubSpot", "company": "Beta"},
        ],
        "prior_reports": [{"report_type": "weekly_churn_feed"}],
    }
    filtered = _filter_tenant_payload_for_vendors(payload, ["Zendesk"])
    assert filtered["date"] == "2026-03-21"
    assert filtered["data_context"] == {"x": 1}
    assert filtered["prior_reports"] == [
        {"type": "weekly_churn_feed", "date": "", "data": []},
    ]
    assert filtered["vendor_churn_scores"] == [
        {"vendor_name": "Zendesk", "category_council": {"winner": "Zoho Desk"}},
    ]
    assert filtered["high_intent_companies"] == [{"vendor": "Zendesk", "company": "Acme"}]
    assert filtered["competitive_displacement"] == [
        {"from_vendor": "Zendesk", "to_vendor": "Freshdesk"},
    ]


def test_filter_tenant_payload_for_vendors_compacts_prior_reports():
    payload = {
        "vendor_churn_scores": [{"vendor_name": "Zendesk"}],
        "prior_reports": [
            {
                "type": "vendor_scorecard",
                "date": "2026-03-14",
                "data": [
                    {"vendor": "Zendesk", "score": 42},
                    {"vendor": "HubSpot", "score": 30},
                ],
            }
        ],
    }
    filtered = _filter_tenant_payload_for_vendors(payload, ["Zendesk"])
    assert filtered["prior_reports"] == [
        {
            "type": "vendor_scorecard",
            "date": "2026-03-14",
            "data": [{"vendor": "Zendesk", "score": 42}],
        }
    ]


def test_merge_tenant_chunk_outputs_dedupes_and_builds_summary():
    partials = [
        {
            "executive_summary": "chunk 1",
            "weekly_churn_feed": [
                {"vendor": "Zendesk", "churn_pressure_score": 42.0, "avg_urgency": 6.0},
            ],
            "vendor_scorecards": [
                {"vendor": "Zendesk", "churn_pressure_score": 42.0},
            ],
            "displacement_map": [
                {"from_vendor": "Zendesk", "to_vendor": "Freshdesk", "mention_count": 9},
            ],
            "category_insights": [
                {"category": "Helpdesk", "highest_churn_risk": "Zendesk"},
            ],
        },
        {
            "executive_summary": "chunk 2",
            "weekly_churn_feed": [
                {"vendor": "HubSpot", "churn_pressure_score": 30.0, "avg_urgency": 5.0},
                {"vendor": "Zendesk", "churn_pressure_score": 42.0, "avg_urgency": 6.0},
            ],
            "vendor_scorecards": [
                {"vendor": "HubSpot", "churn_pressure_score": 30.0},
            ],
            "displacement_map": [
                {"from_vendor": "HubSpot", "to_vendor": "Salesforce", "mention_count": 4},
            ],
            "category_insights": [
                {"category": "CRM", "highest_churn_risk": "HubSpot"},
            ],
        },
    ]
    merged = _merge_tenant_chunk_outputs(
        partials,
        data_context={
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
    )
    assert [row["vendor"] for row in merged["weekly_churn_feed"]] == ["Zendesk", "HubSpot"]
    assert [row["vendor"] for row in merged["vendor_scorecards"]] == ["Zendesk", "HubSpot"]
    assert len(merged["displacement_map"]) == 2
    assert len(merged["category_insights"]) == 2
    assert merged["executive_summary"]


def test_compact_vendor_churn_scores_for_llm_prefers_specific_category_rows():
    rows = [
        {
            "vendor_name": "Zendesk",
            "product_category": "B2B Software",
            "total_reviews": 300,
            "churn_intent": 50,
            "avg_urgency": 3.2,
            "avg_rating_normalized": 2.1,
            "recommend_yes": 3,
            "recommend_no": 90,
        },
        {
            "vendor_name": "Zendesk",
            "product_category": "Helpdesk",
            "total_reviews": 280,
            "churn_intent": 49,
            "avg_urgency": 3.8,
            "avg_rating_normalized": 2.4,
            "recommend_yes": 3,
            "recommend_no": 90,
        },
    ]
    compact = _compact_vendor_churn_scores_for_llm(
        rows,
        council_lookup={
            (_canonicalize_vendor("Zendesk"), "helpdesk"): {"winner": "Zoho Desk"},
        },
    )
    assert len(compact) == 1
    assert compact[0]["vendor_name"] == "Zendesk"
    assert compact[0]["category"] == "Helpdesk"
    assert compact[0]["category_council"] == {"winner": "Zoho Desk"}


def test_apply_tenant_vendor_context_overrides_generated_category_and_council():
    payload = {
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "category_council": {"winner": "Zoho Desk"},
            },
            {
                "vendor_name": "HubSpot",
                "category": "Marketing Automation",
            },
        ]
    }
    parsed = {
        "weekly_churn_feed": [
            {"vendor": "Zendesk", "category": "B2B Software"},
            {"vendor": "HubSpot", "category": "B2B Software", "category_council": {"winner": "Someone"}},
        ],
        "vendor_scorecards": [
            {"vendor": "Zendesk", "category": "Software", "category_council": {"winner": "Other"}},
            {"vendor": "HubSpot", "category": "Software", "category_council": {"winner": "Other"}},
        ],
    }

    applied = _apply_tenant_vendor_context(parsed, payload)

    assert applied["weekly_churn_feed"][0]["category"] == "Helpdesk"
    assert applied["weekly_churn_feed"][0]["category_council"] == {"winner": "Zoho Desk"}
    assert applied["weekly_churn_feed"][1]["category"] == "Marketing Automation"
    assert "category_council" not in applied["weekly_churn_feed"][1]

    assert applied["vendor_scorecards"][0]["category"] == "Helpdesk"
    assert applied["vendor_scorecards"][0]["category_council"] == {"winner": "Zoho Desk"}
    assert applied["vendor_scorecards"][1]["category"] == "Marketing Automation"
    assert "category_council" not in applied["vendor_scorecards"][1]


def test_apply_tenant_vendor_context_backfills_missing_vendor_rows():
    payload = {
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "reviews": 280,
                "churn": 49,
                "urgency": 3.8,
                "rec_yes": 3,
                "rec_no": 90,
                "category_council": {"winner": "Zoho Desk"},
            },
            {
                "vendor_name": "HubSpot",
                "category": "Marketing Automation",
                "reviews": 341,
                "churn": 91,
                "urgency": 3.8,
                "rec_yes": 20,
                "rec_no": 87,
            },
        ]
    }
    parsed = {
        "weekly_churn_feed": [{"vendor": "HubSpot", "category": "B2B Software"}],
        "vendor_scorecards": [{"vendor": "HubSpot", "category": "B2B Software"}],
    }

    applied = _apply_tenant_vendor_context(parsed, payload)

    weekly_vendors = [row["vendor"] for row in applied["weekly_churn_feed"]]
    scorecard_vendors = [row["vendor"] for row in applied["vendor_scorecards"]]
    assert weekly_vendors == ["HubSpot", "Zendesk"]
    assert scorecard_vendors == ["HubSpot", "Zendesk"]

    zendesk_feed = next(row for row in applied["weekly_churn_feed"] if row["vendor"] == "Zendesk")
    zendesk_scorecard = next(row for row in applied["vendor_scorecards"] if row["vendor"] == "Zendesk")
    assert zendesk_feed["category"] == "Helpdesk"
    assert zendesk_feed["category_council"] == {"winner": "Zoho Desk"}
    assert zendesk_scorecard["category"] == "Helpdesk"
    assert zendesk_scorecard["category_council"] == {"winner": "Zoho Desk"}
    assert zendesk_scorecard["expert_take"]


def test_apply_tenant_report_context_backfills_categories_and_displacement():
    payload = {
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "reviews": 280,
                "churn": 49,
                "urgency": 3.8,
                "rec_yes": 3,
                "rec_no": 90,
                "category_council": {
                    "winner": "Zoho Desk",
                    "market_regime": "price_competition",
                    "conclusion": "Pricing pressure is driving churn across Helpdesk.",
                },
            },
            {
                "vendor_name": "HubSpot",
                "category": "Marketing Automation",
                "reviews": 341,
                "churn": 91,
                "urgency": 3.8,
                "rec_yes": 20,
                "rec_no": 87,
            },
        ],
        "competitive_displacement": [
            {
                "vendor": "Zendesk",
                "competitor": "Freshdesk",
                "mention_count": 20,
                "explicit_switches": 0,
                "active_evaluations": 0,
                "reason_categories": {"pricing": 12},
            }
        ],
        "competitor_reasons": [
            {
                "vendor": "Zendesk",
                "competitor": "Freshdesk",
                "reason_category": "pricing",
            }
        ],
    }
    parsed = {
        "weekly_churn_feed": [],
        "vendor_scorecards": [],
        "category_insights": [{"category": "Helpdesk"}],
        "displacement_map": [],
    }

    applied = _apply_tenant_report_context(parsed, payload)

    categories = [row["category"] for row in applied["category_insights"]]
    edges = [
        (row["from_vendor"], row["to_vendor"])
        for row in applied["displacement_map"]
    ]
    assert categories == ["Marketing Automation", "Helpdesk"]
    assert edges == [("Zendesk", "Freshdesk")]
    helpdesk = next(row for row in applied["category_insights"] if row["category"] == "Helpdesk")
    assert helpdesk["dominant_pain"] == "pricing"
    assert helpdesk["emerging_challenger"] == "Zoho Desk"
    edge = applied["displacement_map"][0]
    assert edge["primary_driver"] == "pricing"
    assert edge["signal_strength"] == "strong"


def test_apply_tenant_synthesis_context_attaches_shared_contracts(monkeypatch):
    calls = []

    def _fake_attach(entry, view, *, consumer_name, requested_as_of, include_displacement):
        calls.append((entry["vendor"], consumer_name, include_displacement, requested_as_of))
        entry["reasoning_source"] = consumer_name

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_churn_reports._attach_synthesis_contracts_to_report_entry",
        _fake_attach,
    )

    parsed = {
        "weekly_churn_feed": [
            {"vendor": "Zendesk"},
            {"vendor": "HubSpot"},
        ],
        "vendor_scorecards": [
            {"vendor": "Zendesk"},
            {"vendor": "Asana"},
        ],
    }
    attached = _apply_tenant_synthesis_context(
        parsed,
        {"Zendesk": object(), "Asana": object()},
        requested_as_of=date(2026, 3, 21),
    )

    assert attached == 2
    assert parsed["weekly_churn_feed"][0]["reasoning_source"] == "weekly_churn_feed"
    assert "reasoning_source" not in parsed["weekly_churn_feed"][1]
    assert parsed["vendor_scorecards"][0]["reasoning_source"] == "vendor_scorecard"
    assert parsed["vendor_scorecards"][1]["reasoning_source"] == "vendor_scorecard"
    assert calls == [
        ("Zendesk", "weekly_churn_feed", False, date(2026, 3, 21)),
        ("Zendesk", "vendor_scorecard", True, date(2026, 3, 21)),
        ("Asana", "vendor_scorecard", True, date(2026, 3, 21)),
    ]


def test_build_deterministic_tenant_report_prefers_context_then_contracts(monkeypatch):
    calls = []

    def _fake_attach(parsed, synthesis_views, *, requested_as_of):
        calls.append((sorted(synthesis_views.keys()), requested_as_of))
        parsed["weekly_churn_feed"][0]["account_pressure_summary"] = "Two accounts are active."
        return 1

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_tenant_report._apply_tenant_synthesis_context",
        _fake_attach,
    )

    payload = {
        "data_context": {
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
        "vendor_churn_scores": [
            {
                "vendor_name": "Zendesk",
                "category": "Helpdesk",
                "reviews": 280,
                "churn": 49,
                "urgency": 3.8,
                "rec_yes": 3,
                "rec_no": 90,
            }
        ],
        "competitive_displacement": [],
        "competitor_reasons": [],
    }

    parsed, attached = _build_deterministic_tenant_report(
        payload,
        {"Zendesk": object()},
        requested_as_of=date(2026, 3, 21),
    )

    assert attached == 1
    assert parsed["weekly_churn_feed"][0]["vendor"] == "Zendesk"
    assert parsed["weekly_churn_feed"][0]["account_pressure_summary"] == "Two accounts are active."
    assert parsed["executive_summary"]
    assert calls == [(["Zendesk"], date(2026, 3, 21))]


@pytest.mark.asyncio
async def test_build_deterministic_tenant_report_from_raw_uses_shared_builders(monkeypatch):
    async def _fake_fetch_latest_evidence_vault(pool, *, as_of, analysis_window_days):
        return {}

    async def _fake_reconstruct_reasoning_lookup(pool, as_of=None):
        return {}

    async def _fake_reconstruct_cross_vendor_lookup(pool, as_of=None):
        return {"battles": {}, "councils": {}, "asymmetries": {}}

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared._fetch_latest_evidence_vault",
        _fake_fetch_latest_evidence_vault,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_churn_intelligence.reconstruct_reasoning_lookup",
        _fake_reconstruct_reasoning_lookup,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_churn_intelligence.reconstruct_cross_vendor_lookup",
        _fake_reconstruct_cross_vendor_lookup,
    )

    payload = {
        "data_context": {
            "enrichment_period": {"earliest": "2026-03-01", "latest": "2026-03-21"},
            "source_distribution": {},
        },
        "vendor_churn_scores": [
            {"vendor_name": "Zendesk", "category": "Helpdesk"},
        ],
        "competitive_displacement": [],
        "competitor_reasons": [],
    }
    raw_artifacts = {
        "vendor_scores_from_signals": [
            {
                "vendor_name": "Zendesk",
                "product_category": "Helpdesk",
                "total_reviews": 120,
                "churn_intent": 18,
                "avg_urgency": 7.2,
            }
        ],
        "competitive_disp": [],
        "pain_dist": [],
        "feature_gaps": [],
        "negative_counts": [],
        "price_rates": [],
        "dm_rates": [],
        "churning_companies": [],
        "quotable_evidence": [],
        "budget_signals": [],
        "use_case_dist": [],
        "sentiment_traj": [],
        "buyer_auth": [],
        "timeline_signals": [],
        "competitor_reasons": [],
        "data_context": payload["data_context"],
        "prior_reports": [],
        "keyword_spikes": [],
        "product_profiles_raw": [],
        "review_text_aggs": ([], []),
        "department_dist": [],
        "contract_ctx_aggs": ([], []),
        "sentiment_tenure_raw": [],
        "turning_points_raw": [],
    }

    parsed, attached = await _build_deterministic_tenant_report_from_raw(
        pool=None,
        raw_artifacts=raw_artifacts,
        payload=payload,
        synthesis_views={},
        requested_as_of=date(2026, 3, 21),
        analysis_window_days=30,
    )

    assert attached == 0
    assert parsed["weekly_churn_feed"][0]["vendor"] == "Zendesk"
    assert parsed["vendor_scorecards"][0]["vendor"] == "Zendesk"
    assert parsed["vendor_scorecards"][0]["expert_take"]
    assert parsed["category_insights"][0]["category"] == "Helpdesk"
    assert parsed["executive_summary"]
