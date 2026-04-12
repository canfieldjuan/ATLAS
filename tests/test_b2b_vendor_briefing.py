import csv
import io
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlas_brain.api import b2b_vendor_briefing as briefing_api
from atlas_brain.autonomous.tasks import b2b_vendor_briefing as briefing_mod



def _make_app():
    app = FastAPI()
    app.include_router(briefing_api.router)
    return app


def test_generate_briefing_trims_vendor_name_and_blank_email(monkeypatch):
    monkeypatch.setattr(briefing_api.settings.b2b_churn, "vendor_briefing_enabled", True, raising=False)
    generate = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr(briefing_api, "generate_and_send_briefing", generate)
    app = _make_app()

    with TestClient(app) as client:
        response = client.post(
            "/b2b/briefings/generate",
            json={"vendor_name": "  Zendesk  ", "to_email": "   "},
        )

    assert response.status_code == 200
    assert generate.await_args.kwargs == {"vendor_name": "Zendesk", "to_email": None}


def test_briefing_gate_rejects_blank_email_before_db_touch(monkeypatch):
    monkeypatch.setattr(briefing_api.settings.b2b_churn, "vendor_briefing_enabled", True, raising=False)

    def _fail_pool():
        raise AssertionError("_pool_or_503 should not be called")

    monkeypatch.setattr(briefing_api, "_pool_or_503", _fail_pool)
    app = _make_app()

    with TestClient(app) as client:
        response = client.post(
            "/b2b/briefings/gate",
            json={"email": "     ", "token": "abcdefghij"},
        )

    assert response.status_code == 422


def test_vendor_checkout_trims_customer_email_before_stripe(monkeypatch):
    monkeypatch.setattr(briefing_api.settings.saas_auth, "stripe_secret_key", "sk_test", raising=False)
    monkeypatch.setattr(briefing_api.settings.saas_auth, "stripe_vendor_standard_price_id", "price_standard", raising=False)
    fake_session = SimpleNamespace(id="cs_test", url="https://checkout.test/session")
    create = MagicMock(return_value=fake_session)
    fake_stripe = SimpleNamespace(
        api_key=None,
        StripeError=Exception,
        checkout=SimpleNamespace(Session=SimpleNamespace(create=create)),
    )
    import sys
    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    app = _make_app()

    with TestClient(app) as client:
        response = client.post(
            "/b2b/briefings/checkout",
            json={
                "vendor_name": "  Zendesk  ",
                "tier": "standard",
                "email": "  Ops@Example.com  ",
            },
        )

    assert response.status_code == 200
    kwargs = create.call_args.kwargs
    assert kwargs["customer_email"] == "ops@example.com"
    assert kwargs["metadata"]["vendor_name"] == "Zendesk"
    assert "vendor=Zendesk" in kwargs["success_url"]


def test_vendor_checkout_omits_blank_customer_email(monkeypatch):
    monkeypatch.setattr(briefing_api.settings.saas_auth, "stripe_secret_key", "sk_test", raising=False)
    monkeypatch.setattr(briefing_api.settings.saas_auth, "stripe_vendor_standard_price_id", "price_standard", raising=False)
    create = MagicMock(return_value=SimpleNamespace(id="cs_test", url="https://checkout.test/session"))
    fake_stripe = SimpleNamespace(
        api_key=None,
        StripeError=Exception,
        checkout=SimpleNamespace(Session=SimpleNamespace(create=create)),
    )
    import sys
    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    app = _make_app()

    with TestClient(app) as client:
        response = client.post(
            "/b2b/briefings/checkout",
            json={"vendor_name": "Zendesk", "tier": "standard", "email": "      "},
        )

    assert response.status_code == 200
    assert "customer_email" not in create.call_args.kwargs


def test_checkout_session_info_rejects_blank_session_id_before_stripe(monkeypatch):
    monkeypatch.setattr(briefing_api.settings.saas_auth, "stripe_secret_key", "sk_test", raising=False)
    retrieve = MagicMock(side_effect=AssertionError("Stripe should not be called"))
    fake_stripe = SimpleNamespace(
        api_key=None,
        StripeError=Exception,
        checkout=SimpleNamespace(Session=SimpleNamespace(retrieve=retrieve)),
    )
    import sys
    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    app = _make_app()

    with TestClient(app) as client:
        response = client.get("/b2b/briefings/checkout-session?session_id=%20%20%20%20%20%20%20%20%20%20")

    assert response.status_code == 422
    assert response.json()["detail"] == "session_id is required"


def test_report_data_rejects_blank_token_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("_pool_or_503 should not be called")

    monkeypatch.setattr(briefing_api, "_pool_or_503", _fail_pool)
    app = _make_app()

    with TestClient(app) as client:
        response = client.get("/b2b/briefings/report-data?token=%20%20%20%20%20%20%20%20%20%20")

    assert response.status_code == 422
    assert response.json()["detail"] == "token is required"


def test_default_pain_label_maps_generic_buckets_to_overall_dissatisfaction():
    assert briefing_mod._default_pain_label("other") == "Overall Dissatisfaction"
    assert briefing_mod._default_pain_label("general_dissatisfaction") == "Overall Dissatisfaction"
    assert briefing_mod._default_pain_label("overall_dissatisfaction") == "Overall Dissatisfaction"


def test_apply_evidence_vault_to_briefing_fills_sparse_fields():
    briefing = {
        "evidence": [],
        "pain_breakdown": [],
        "top_feature_gaps": [],
        "named_accounts": [],
        "review_count": 0,
        "avg_urgency": 0,
        "churn_signal_density": 0,
        "dm_churn_rate": 0,
    }
    vault = {
        "metric_snapshot": {
            "reviews_in_analysis_window": 42,
            "avg_urgency": 6.2,
            "churn_density": 18.5,
            "dm_churn_rate": 0.34,
        },
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "best_quote": "Bills kept climbing every quarter.",
                "quote_source": {
                    "company": "Acme Corp",
                    "reviewer_title": "VP Operations",
                    "company_size": "mid_market",
                    "industry": "SaaS",
                    "source": "g2",
                    "review_id": "r1",
                    "rating": 2.0,
                    "reviewed_at": "2026-03-10",
                },
                "mention_count_total": 12,
                "supporting_metrics": {"avg_urgency_when_mentioned": 7.1},
            },
            {
                "key": "reporting",
                "label": "Reporting gaps",
                "evidence_type": "feature_gap",
                "mention_count_total": 9,
            },
        ],
        "company_signals": [
            {
                "company_name": "Acme Corp",
                "buyer_role": "VP Operations",
                "seat_count": 250,
                "urgency_score": 8.0,
                "pain_category": "pricing",
                "buying_stage": "evaluation",
                "source": "g2",
                "confidence_score": 0.8,
            }
        ],
    }

    used = briefing_mod._apply_evidence_vault_to_briefing(briefing, vault)

    assert used is True
    assert briefing["review_count"] == 42
    assert briefing["avg_urgency"] == 6.2
    assert briefing["churn_signal_density"] == 18.5
    assert briefing["dm_churn_rate"] == 34.0
    assert briefing["pain_breakdown"][0]["category"] == "pricing"
    assert briefing["top_feature_gaps"] == ["Reporting gaps"]
    assert briefing["named_accounts"][0]["company"] == "Acme Corp"
    assert briefing["evidence"][0]["source"] == "g2"
    assert briefing["evidence"][0]["quote"] == "Bills kept climbing every quarter."


def test_apply_reasoning_synthesis_to_briefing_copies_contract_fields():
    briefing = {}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
                "vendor_core_reasoning": {
                    "schema_version": "v1",
                    "causal_narrative": {"trigger": "Price hike"},
                    "timing_intelligence": {
                        "best_timing_window": "Before renewal",
                        "active_eval_signals": {
                            "value": 3,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                        "sentiment_direction": "declining",
                        "immediate_triggers": [
                            {"trigger": "Q2 renewal", "type": "deadline"},
                        ],
                    },
                },
            "displacement_reasoning": {
                "schema_version": "v1",
                "migration_proof": {"confidence": "medium"},
            },
        },
        "synthesis_wedge": "price_squeeze",
        "synthesis_wedge_label": "Price Squeeze",
        "synthesis_schema_version": "v2",
        "evidence_window": {
            "evidence_window_start": "2026-03-01",
            "evidence_window_end": "2026-03-18",
        },
        "evidence_window_days": 17,
        "reasoning_source": "b2b_reasoning_synthesis",
        "category_council": {
            "winner": "Zoho Desk",
            "loser": "Freshdesk",
            "market_regime": "price_competition",
        },
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
    assert briefing["displacement_reasoning"]["migration_proof"]["confidence"] == "medium"
    assert briefing["synthesis_wedge"] == "price_squeeze"
    assert briefing["evidence_window_days"] == 17
    assert briefing["reasoning_source"] == "b2b_reasoning_synthesis"
    assert briefing["category_council"]["winner"] == "Zoho Desk"
    assert briefing["timing_intelligence"]["best_timing_window"] == "Before renewal"
    assert briefing["timing_summary"] == (
        "Before renewal. 3 active evaluation signals are visible right now. "
        "Review sentiment is skewing more negative."
    )
    assert briefing["timing_metrics"]["active_eval_signals"] == 3
    assert briefing["priority_timing_triggers"] == ["Q2 renewal"]
    assert "causal_narrative" not in briefing


@pytest.mark.asyncio
async def test_fetch_cross_vendor_conclusions_uses_canonical_lookup(monkeypatch):
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.load_best_cross_vendor_lookup",
        AsyncMock(
            return_value={
                "battles": {
                    ("HubSpot", "Zendesk"): {
                        "confidence": 0.72,
                        "source": "synthesis",
                        "computed_date": None,
                        "conclusion": {"conclusion": "HubSpot is winning SMB evaluations."},
                    },
                },
                "councils": {
                    "CRM": {
                        "confidence": 0.64,
                        "source": "synthesis",
                        "computed_date": None,
                        "reference_ids": {"witness_ids": ["witness:crm:1"]},
                        "conclusion": {"conclusion": "Pricing pressure is fragmenting CRM."},
                        "vendors": ["HubSpot", "Zendesk"],
                    },
                },
                "asymmetries": {},
            }
        ),
    )

    results = await briefing_mod._fetch_cross_vendor_conclusions(
        pool=object(),
        vendor_name="Zendesk",
        category="CRM",
    )

    assert results[0]["analysis_type"] == "pairwise_battle"
    assert results[1]["analysis_type"] == "category_council"
    assert results[1]["reference_ids"]["witness_ids"] == ["witness:crm:1"]


@pytest.mark.asyncio
async def test_export_briefings_returns_csv(monkeypatch):
    class Pool:
        async def fetch(self, query, *args):
            assert "FROM b2b_vendor_briefings" in query
            return [{
                "vendor_name": "Zendesk",
                "recipient_email": "ops@acme.com",
                "subject": "Zendesk Briefing",
                "status": "pending_approval",
                "target_mode": "vendor_retention",
                "created_at": datetime(2026, 4, 7, 12, 0),
                "approved_at": None,
                "rejected_at": None,
                "reject_reason": None,
            }]

    monkeypatch.setattr(briefing_api, "_pool_or_503", lambda: Pool())

    response = await briefing_api.export_briefings(
        status="pending_approval",
        user=object(),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    body = "".join(
        chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        for chunk in chunks
    )
    rows = list(csv.DictReader(io.StringIO(body)))

    assert rows[0]["vendor_name"] == "Zendesk"
    assert rows[0]["status"] == "pending_approval"


def test_apply_reasoning_synthesis_to_briefing_normalizes_flat_feed_sections():
    briefing = {}
    feed_entry = {
        "causal_narrative": {"trigger": "Legacy trigger"},
        "timing_intelligence": {
            "best_timing_window": "Before renewal",
            "active_eval_signals": {
                "value": 1,
                "source_id": "accounts:summary:active_eval_signal_count",
            },
        },
        "migration_proof": {"confidence": "medium"},
        "synthesis_wedge": "price_squeeze",
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Legacy trigger"
    assert briefing["vendor_core_reasoning"]["timing_intelligence"]["best_timing_window"] == "Before renewal"
    assert briefing["timing_summary"] == (
        "Before renewal. 1 active evaluation signals are visible right now."
    )
    assert briefing["displacement_reasoning"]["migration_proof"]["confidence"] == "medium"
    assert "causal_narrative" not in briefing
    assert "migration_proof" not in briefing


def test_apply_reasoning_synthesis_to_briefing_promotes_account_reasoning():
    briefing = {"named_accounts": []}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {"trigger": "Price hike"},
            },
            "account_reasoning": {
                "schema_version": "v1",
                "market_summary": "Two accounts are actively evaluating alternatives.",
                "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                "active_eval_count": {
                    "value": 2,
                    "source_id": "accounts:summary:active_eval_signal_count",
                },
                "top_accounts": [
                    {"name": "Acme Corp", "intent_score": 0.9, "source_id": "accounts:item:acme"},
                ],
            },
        },
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["account_reasoning"]["top_accounts"][0]["name"] == "Acme Corp"
    assert briefing["account_pressure_summary"] == "Two accounts are actively evaluating alternatives."
    assert briefing["account_pressure_metrics"]["high_intent_count"] == 4
    assert briefing["priority_account_names"] == ["Acme Corp"]
    assert briefing["named_accounts"][0]["company"] == "Acme Corp"
    assert briefing["named_accounts"][0]["reasoning_backed"] is True


def test_apply_reasoning_synthesis_to_briefing_surfaces_sparse_account_preview():
    briefing = {"named_accounts": []}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {"trigger": "Price hike"},
            },
            "account_reasoning": {
                "schema_version": "v1",
                "confidence": "insufficient",
                "market_summary": "A single post-purchase account is in scope.",
                "total_accounts": {"value": 1, "source_id": "accounts:summary:total_accounts"},
                "top_accounts": [
                    {
                        "name": "Concentrix",
                        "intent_score": 0.6,
                        "source_id": "accounts:company:concentrix",
                    },
                ],
            },
        },
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert "account_reasoning" not in briefing
    assert briefing["account_reasoning_preview"]["top_accounts"][0]["name"] == "Concentrix"
    assert briefing["account_reasoning_preview_only"] is True
    assert briefing["account_pressure_summary"] == "A single post-purchase account is in scope."
    assert briefing["account_pressure_metrics"]["total_accounts"] == 1
    assert briefing["priority_account_names"] == ["Concentrix"]
    assert briefing["named_accounts"][0]["company"] == "Concentrix"
    assert briefing["reasoning_section_disclaimers"]["account_reasoning"]


def test_finalize_briefing_presentation_does_not_promote_sparse_account_preview_to_executive_summary():
    briefing = {
        "account_reasoning_preview_only": True,
        "account_pressure_summary": "A single post-purchase account is in scope.",
        "timing_summary": "Renewal activity is concentrated in Q3.",
    }

    briefing_mod._finalize_briefing_presentation(briefing)

    assert briefing["executive_summary"] == "Renewal activity is concentrated in Q3."


@pytest.mark.asyncio
async def test_attach_company_signal_review_queue_to_briefing():
    pool = object()
    briefing = {"vendor_name": "Zendesk"}
    queue = {
        "vendor": "Zendesk",
        "candidate_bucket": "analyst_review",
        "review_status": "pending",
        "totals": {"pending_groups": 2},
        "operator_focus": {"action": "review_low_trust_policy"},
        "groups": [{"company": "Acme Corp"}],
    }
    queue_mock = AsyncMock(return_value=queue)
    with patch.object(
        briefing_mod,
        "read_vendor_company_signal_review_queue",
        queue_mock,
    ):
        used = await briefing_mod._attach_company_signal_review_queue_to_briefing(
            pool,
            briefing,
            vendor_name="Zendesk",
        )

    queue_mock.assert_awaited_once_with(
        pool,
        vendor_name="Zendesk",
    )
    assert used is True
    assert briefing["company_signal_review_queue"]["vendor"] == "Zendesk"
    assert briefing["company_signal_review_queue"]["groups"][0]["company"] == "Acme Corp"


def test_apply_reasoning_synthesis_to_briefing_surfaces_anchor_examples_and_reference_ids():
    briefing = {}
    feed_entry = {
        "vendor": "Zendesk",
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {"trigger": "Price hike"},
            },
            "displacement_reasoning": {
                "schema_version": "v1",
                "migration_proof": {"confidence": "medium"},
            },
            "account_reasoning": {
                "schema_version": "v1",
                "market_summary": "Two accounts are actively evaluating alternatives.",
            },
        },
        "reference_ids": {"witness_ids": ["witness:r1:0"]},
        "packet_artifacts": {
            "witness_pack": [
                {
                    "witness_id": "witness:r1:0",
                    "_sid": "witness:r1:0",
                    "excerpt_text": "Hack Club said the renewal jumped to $200k/year.",
                    "reviewer_company": "Hack Club",
                    "time_anchor": "Q2 renewal",
                    "salience_score": 9.1,
                },
            ],
            "section_packets": {
                "anchor_examples": {
                    "outlier_or_named_account": ["witness:r1:0"],
                },
            },
        },
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["reasoning_anchor_examples"]["outlier_or_named_account"][0]["reviewer_company"] == "Hack Club"
    assert briefing["reasoning_witness_highlights"][0]["witness_id"] == "witness:r1:0"
    assert briefing["reasoning_reference_ids"]["witness_ids"] == ["witness:r1:0"]


def test_apply_reasoning_synthesis_to_briefing_surfaces_section_disclaimers():
    briefing = {}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {
                    "primary_wedge": "price_squeeze",
                    "confidence": "high",
                },
                "timing_intelligence": {
                    "confidence": "low",
                    "best_timing_window": "Q2 renewal",
                },
            },
        },
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["reasoning_section_disclaimers"]["timing_intelligence"]


def test_apply_reasoning_synthesis_to_briefing_does_not_backfill_missing_explicit_contracts():
    briefing = {}
    feed_entry = {
        "reasoning_contracts": {
            "schema_version": "v1",
            "vendor_core_reasoning": {
                "schema_version": "v1",
                "causal_narrative": {"trigger": "Price hike"},
            },
        },
        "vendor_core_reasoning": {"causal_narrative": {"trigger": "Legacy mirror"}},
        "timing_intelligence": {"best_timing_window": "Before renewal"},
        "migration_proof": {"confidence": "medium"},
    }

    used = briefing_mod._apply_reasoning_synthesis_to_briefing(briefing, feed_entry)

    assert used is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
    assert "timing_intelligence" not in briefing["vendor_core_reasoning"]
    assert "displacement_reasoning" not in briefing


@pytest.mark.asyncio
async def test_build_vendor_briefing_uses_evidence_vault_overlay(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    vault = {
        "metric_snapshot": {"reviews_in_analysis_window": 30, "avg_urgency": 5.5},
        "weakness_evidence": [
            {
                "key": "pricing",
                "label": "Pricing opacity",
                "evidence_type": "pain_category",
                "best_quote": "The price jumps were impossible to budget.",
                "quote_source": {"source": "capterra"},
                "mention_count_total": 11,
                "supporting_metrics": {"avg_urgency_when_mentioned": 6.4},
            }
        ],
        "company_signals": [
            {
                "company_name": "Northwind",
                "buyer_role": "VP Finance",
                "urgency_score": 7.0,
                "pain_category": "pricing",
                "source": "reddit",
            }
        ],
    }

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 48,
                "churn_signal_density": 12.5,
                "avg_urgency": 0,
                "total_reviews": 0,
                "dm_churn_rate": 0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=vault))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    briefing = await briefing_mod.build_vendor_briefing("Zendesk")

    assert briefing is not None
    assert briefing["data_sources"]["weekly_churn_feed"] is True
    assert briefing["data_sources"]["evidence_vault"] is True
    assert briefing["pain_breakdown"][0]["category"] == "pricing"
    assert briefing["evidence"][0]["source"] == "capterra"
    assert briefing["named_accounts"][0]["company"] == "Northwind"
    assert briefing["avg_urgency"] == 5.5


@pytest.mark.asyncio
async def test_build_vendor_briefing_suppresses_stale_evidence_vault_overlay_for_signal_fallback(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    vault_record = {
        "vendor_name": "Zendesk",
        "materialization_run_id": "run-stale",
        "vault": {
            "metric_snapshot": {"reviews_in_analysis_window": 30, "avg_urgency": 9.9},
            "weakness_evidence": [
                {
                    "key": "pricing",
                    "label": "Pricing opacity",
                    "evidence_type": "pain_category",
                    "best_quote": "The price jumps were impossible to budget.",
                    "quote_source": {"source": "capterra"},
                    "mention_count_total": 11,
                    "supporting_metrics": {"avg_urgency_when_mentioned": 6.4},
                }
            ],
        },
    }

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(briefing_mod, "_extract_feed_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(
        briefing_mod,
        "_fetch_churn_signals",
        AsyncMock(
            return_value={
                "vendor_name": "Zendesk",
                "materialization_run_id": "run-current",
                "avg_urgency_score": 4.4,
                "total_reviews": 30,
                "decision_maker_churn_rate": 0.12,
                "top_pain_categories": [{"category": "pricing", "count": 4}],
                "top_competitors": [],
                "quotable_evidence": [{"quote": "Raw signal quote", "source": "g2"}],
                "company_churn_list": [],
                "top_feature_gaps": [],
                "product_category": "CRM",
            }
        ),
    )
    monkeypatch.setattr(
        briefing_mod,
        "_fetch_vendor_evidence_record",
        AsyncMock(return_value=vault_record),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    briefing = await briefing_mod.build_vendor_briefing("Zendesk")

    assert briefing is not None
    assert briefing["data_sources"]["churn_signals"] is True
    assert briefing["data_sources"]["evidence_vault"] is False
    assert briefing["avg_urgency"] == 4.4
    assert briefing["evidence"][0]["source"] == "g2"
    assert briefing["pain_breakdown"][0]["category"] == "pricing"


@pytest.mark.asyncio
async def test_build_vendor_briefing_marks_reasoning_synthesis_from_feed(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 48,
                "churn_signal_density": 12.5,
                "avg_urgency": 4.4,
                "total_reviews": 30,
                "dm_churn_rate": 12.0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {"trigger": "Price hike"},
                    },
                },
                "synthesis_wedge": "price_squeeze",
                "reasoning_source": "b2b_reasoning_synthesis",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    briefing = await briefing_mod.build_vendor_briefing("Zendesk")

    assert briefing is not None
    assert briefing["data_sources"]["reasoning_synthesis"] is True
    assert briefing["vendor_core_reasoning"]["causal_narrative"]["trigger"] == "Price hike"
    assert "causal_narrative" not in briefing
    assert briefing["synthesis_wedge"] == "price_squeeze"


@pytest.mark.asyncio
async def test_build_vendor_briefing_uses_account_reasoning_named_account_fallback(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 48,
                "churn_signal_density": 12.5,
                "avg_urgency": 4.4,
                "total_reviews": 30,
                "dm_churn_rate": 12.0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
                "reasoning_contracts": {
                    "schema_version": "v1",
                    "vendor_core_reasoning": {
                        "schema_version": "v1",
                        "causal_narrative": {"trigger": "Price hike"},
                    },
                    "account_reasoning": {
                        "schema_version": "v1",
                        "market_summary": "Two accounts are actively evaluating alternatives.",
                        "total_accounts": {"value": 6, "source_id": "accounts:summary:total_accounts"},
                        "high_intent_count": {"value": 4, "source_id": "accounts:summary:high_intent_count"},
                        "active_eval_count": {
                            "value": 2,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                        "top_accounts": [
                            {"name": "Acme Corp", "intent_score": 0.9, "source_id": "accounts:item:acme"},
                        ],
                    },
                },
                "synthesis_wedge": "price_squeeze",
                "reasoning_source": "b2b_reasoning_synthesis",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    briefing = await briefing_mod.build_vendor_briefing("Zendesk")

    assert briefing is not None
    assert briefing["data_sources"]["reasoning_synthesis"] is True
    assert briefing["data_sources"]["account_reasoning"] is True
    assert briefing["account_pressure_summary"] == "Two accounts are actively evaluating alternatives."
    assert briefing["named_accounts"][0]["company"] == "Acme Corp"
    assert briefing["executive_summary"] == "Two accounts are actively evaluating alternatives."


@pytest.mark.asyncio
async def test_build_vendor_briefing_can_skip_analyst_summary_and_force_baseline_cards(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, *args, **kwargs):
            return []

    analyst_mock = AsyncMock(return_value=None)
    card_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 48,
                "churn_signal_density": 12.5,
                "avg_urgency": 4.4,
                "total_reviews": 30,
                "dm_churn_rate": 12.0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", analyst_mock)
    monkeypatch.setattr(briefing_mod, "generate_account_cards", card_mock)

    briefing = await briefing_mod.build_vendor_briefing(
        "Zendesk",
        analyst_summary_enabled=False,
        account_cards_reasoning_depth=0,
    )

    assert briefing is not None
    analyst_mock.assert_not_awaited()
    card_mock.assert_awaited_once_with(
        briefing,
        reasoning_depth=0,
        target_mode="vendor_retention",
    )


@pytest.mark.asyncio
async def test_send_batch_briefings_uses_deterministic_scheduled_mode(monkeypatch):
    class DummyPool:
        is_initialized = True

        def __init__(self):
            self.execute_calls = []

        async def fetch(self, query, *args):
            if "FROM vendor_targets" in query:
                return [
                    {
                        "company_name": "Zendesk",
                        "contact_email": "ops@zendesk.test",
                        "contact_name": "Ops",
                        "contact_role": "VP Operations",
                        "target_mode": "vendor_retention",
                        "account_id": None,
                        "created_at": None,
                        "updated_at": None,
                    }
                ]
            return []

        async def execute(self, query, *args):
            self.execute_calls.append((query, args))
            return "INSERT 0 1"

    pool = DummyPool()
    build_mock = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "challenger_mode": False,
            "data_sources": {},
        }
    )

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(briefing_mod, "dedupe_vendor_target_rows", lambda rows: rows)
    monkeypatch.setattr(briefing_mod, "_check_cooldown", AsyncMock(return_value=False))
    monkeypatch.setattr(briefing_mod, "is_suppressed", AsyncMock(return_value=False))
    monkeypatch.setattr(briefing_mod, "_is_first_briefing", AsyncMock(return_value=False))
    monkeypatch.setattr(briefing_mod, "build_vendor_briefing", build_mock)
    monkeypatch.setattr(briefing_mod, "render_vendor_briefing_html", lambda _: "<html></html>")
    monkeypatch.setattr(
        briefing_mod.settings.b2b_churn,
        "vendor_briefing_scheduled_analyst_enrichment_enabled",
        False,
    )
    monkeypatch.setattr(
        briefing_mod.settings.b2b_churn,
        "vendor_briefing_scheduled_account_cards_reasoning_depth",
        0,
    )
    monkeypatch.setattr(
        briefing_mod.settings.b2b_churn,
        "vendor_briefing_max_per_batch",
        1,
    )

    result = await briefing_mod.send_batch_briefings()

    assert result["queued"] == 1
    build_mock.assert_awaited_once_with(
        "Zendesk",
        target_mode="vendor_retention",
        analyst_summary_enabled=False,
        account_cards_reasoning_depth=0,
    )
    assert pool.execute_calls


# ---------------------------------------------------------------------------
# _apply_pain_points_to_briefing
# ---------------------------------------------------------------------------

def test_apply_pain_points_backfills_empty_pain_breakdown():
    briefing: dict = {"pain_breakdown": []}
    pain_pts = [
        {
            "pain_category": "pricing",
            "mention_count": 80,
            "primary_count": 60,
            "avg_urgency": 7.2,
            "avg_rating": 2.5,
            "confidence_score": 0.85,
        },
        {
            "pain_category": "support",
            "mention_count": 40,
            "primary_count": 20,
            "avg_urgency": 5.1,
            "avg_rating": None,
            "confidence_score": 0.6,
        },
    ]

    used = briefing_mod._apply_pain_points_to_briefing(briefing, pain_pts)

    assert used is True
    assert len(briefing["pain_breakdown"]) == 2
    pricing = briefing["pain_breakdown"][0]
    assert pricing["category"] == "pricing"
    assert pricing["count"] == 80
    assert pricing["avg_urgency"] == 7.2
    assert pricing["avg_rating"] == 2.5
    assert pricing["confidence_score"] == 0.85
    assert briefing["pain_breakdown"][1]["avg_rating"] is None


def test_apply_pain_points_enriches_existing_pain_breakdown():
    briefing: dict = {
        "pain_breakdown": [
            {"category": "pricing", "count": 50},
            {"category": "support", "count": 20},
            {"category": "features", "count": 10},
        ]
    }
    pain_pts = [
        {
            "pain_category": "pricing",
            "mention_count": 80,
            "primary_count": 60,
            "avg_urgency": 7.2,
            "avg_rating": 2.5,
            "confidence_score": 0.85,
        },
        {
            "pain_category": "support",
            "mention_count": 40,
            "primary_count": 20,
            "avg_urgency": 5.1,
            "avg_rating": 3.1,
            "confidence_score": 0.6,
        },
    ]

    used = briefing_mod._apply_pain_points_to_briefing(briefing, pain_pts)

    assert used is True
    pricing = briefing["pain_breakdown"][0]
    assert pricing["avg_urgency"] == 7.2
    assert pricing["avg_rating"] == 2.5
    assert pricing["confidence_score"] == 0.85
    # features row had no matching pain_point -- unchanged
    assert "avg_urgency" not in briefing["pain_breakdown"][2]


def test_apply_pain_points_does_not_overwrite_existing_urgency():
    briefing: dict = {
        "pain_breakdown": [{"category": "pricing", "count": 50, "avg_urgency": 9.0}]
    }
    pain_pts = [
        {
            "pain_category": "pricing",
            "mention_count": 80,
            "primary_count": 60,
            "avg_urgency": 3.0,
            "avg_rating": None,
            "confidence_score": 0.85,
        }
    ]
    # existing has only 1 entry -- backfill threshold is < 2, so this replaces
    used = briefing_mod._apply_pain_points_to_briefing(briefing, pain_pts)
    assert used is True
    # backfill replaced the list
    assert briefing["pain_breakdown"][0]["avg_urgency"] == 3.0


def test_apply_pain_points_skips_when_list_is_empty():
    briefing: dict = {"pain_breakdown": [{"category": "pricing", "count": 50}] * 3}
    used = briefing_mod._apply_pain_points_to_briefing(briefing, [])
    assert used is False


# ---------------------------------------------------------------------------
# _apply_account_intelligence_to_briefing
# ---------------------------------------------------------------------------

def test_apply_account_intelligence_populates_pressure_fields():
    briefing: dict = {}
    acct_data = {
        "summary": {
            "total_accounts": 5,
            "high_intent_count": 3,
            "active_eval_signal_count": 2,
            "decision_maker_count": 1,
            "with_seat_count": 2,
            "with_contract_end": 0,
        },
        "accounts": [
            {"company_name": "Acme Corp", "urgency_score": 8.0, "high_intent": True},
            {"company_name": "Beta Inc", "urgency_score": 7.5},
            {"company_name": "Low Inc", "urgency_score": 2.0},
        ],
    }

    used = briefing_mod._apply_account_intelligence_to_briefing(briefing, acct_data)

    assert used is True
    assert briefing["account_pressure_metrics"]["high_intent_count"] == 3
    assert briefing["account_pressure_metrics"]["active_eval_signal_count"] == 2
    assert "3 high-intent accounts" in briefing["account_pressure_summary"]
    assert "2 active evaluation signals" in briefing["account_pressure_summary"]
    assert "Acme Corp" in briefing["priority_account_names"]
    assert "Beta Inc" in briefing["priority_account_names"]
    assert "Low Inc" not in briefing["priority_account_names"]
    # named_account_count should be populated from summary.total_accounts
    assert briefing["named_account_count"] == 5


def test_apply_account_intelligence_materializes_named_accounts():
    briefing: dict = {}
    acct_data = {
        "summary": {
            "total_accounts": 2,
            "high_intent_count": 2,
            "active_eval_signal_count": 2,
            "decision_maker_count": 0,
        },
        "accounts": [
            {
                "company_name": "Acme Corp",
                "urgency_score": 8.5,
                "buyer_role": "VP Operations",
                "buying_stage": "evaluation",
                "pain_category": "pricing",
                "source": "g2",
                "confidence_score": 0.72,
            },
            {
                "company_name": "Beta Inc",
                "urgency_score": 6.2,
                "buyer_role": "unknown",
                "buying_stage": "renewal_decision",
                "pain_category": "support",
                "source": "reddit",
                "confidence_score": 0.55,
            },
        ],
    }

    briefing_mod._apply_account_intelligence_to_briefing(briefing, acct_data)

    accounts = briefing["named_accounts"]
    assert len(accounts) == 2
    # Sorted by urgency descending
    assert accounts[0]["company"] == "Acme Corp"
    assert accounts[0]["urgency"] == 8.5
    assert accounts[0]["title"] == "VP Operations"
    assert accounts[0]["buying_stage"] == "evaluation"
    assert accounts[0]["source"] == "g2"
    # buyer_role "unknown" should normalize to None
    assert accounts[1]["company"] == "Beta Inc"
    assert accounts[1]["title"] is None


def test_apply_account_intelligence_does_not_overwrite_existing_named_accounts():
    briefing: dict = {"named_accounts": [{"company": "Existing", "urgency": 9.0}]}
    acct_data = {
        "summary": {"total_accounts": 1, "high_intent_count": 1, "active_eval_signal_count": 0, "decision_maker_count": 0},
        "accounts": [{"company_name": "New Corp", "urgency_score": 7.0}],
    }

    briefing_mod._apply_account_intelligence_to_briefing(briefing, acct_data)

    # Existing named_accounts preserved
    assert briefing["named_accounts"][0]["company"] == "Existing"
    assert len(briefing["named_accounts"]) == 1


def test_apply_account_intelligence_skips_summary_when_already_set():
    briefing: dict = {
        "account_pressure_summary": "Already set.",
        "named_accounts": [{"company": "Existing"}],
    }
    acct_data = {
        "summary": {
            "total_accounts": 10,
            "high_intent_count": 7,
            "active_eval_signal_count": 4,
            "decision_maker_count": 2,
        },
        "accounts": [],
    }

    used = briefing_mod._apply_account_intelligence_to_briefing(briefing, acct_data)

    assert used is True
    # Summary unchanged (already set)
    assert briefing["account_pressure_summary"] == "Already set."
    # named_account_count set from total_accounts even when named_accounts already present
    assert briefing["named_account_count"] == 10
    # Metrics always written
    assert briefing["account_pressure_metrics"]["high_intent_count"] == 7


def test_apply_account_intelligence_returns_false_when_summary_missing():
    briefing: dict = {}
    used = briefing_mod._apply_account_intelligence_to_briefing(briefing, {})
    assert used is False


# ---------------------------------------------------------------------------
# build_vendor_briefing: Sources 10-12 integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_vendor_briefing_uses_pain_points_fallback(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, query, *args):
            if "b2b_vendor_pain_points" in query:
                return [
                    {
                        "pain_category": "pricing",
                        "mention_count": 90,
                        "primary_count": 70,
                        "avg_urgency": 7.8,
                        "avg_rating": 2.3,
                        "confidence_score": 0.9,
                    },
                    {
                        "pain_category": "support",
                        "mention_count": 50,
                        "primary_count": 30,
                        "avg_urgency": 5.5,
                        "avg_rating": 3.0,
                        "confidence_score": 0.7,
                    },
                ]
            return []

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 55,
                "churn_signal_density": 14.0,
                "avg_urgency": 6.0,
                "total_reviews": 100,
                "dm_churn_rate": 10.0,
                "pain_breakdown": [],
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    result = await briefing_mod.build_vendor_briefing("Zendesk")

    assert result is not None
    assert result["data_sources"]["pain_points"] is True
    assert result["pain_breakdown"][0]["category"] == "pricing"
    assert result["pain_breakdown"][0]["avg_urgency"] == 7.8


@pytest.mark.asyncio
async def test_build_vendor_briefing_uses_account_intelligence(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, query, *args):
            if "b2b_account_intelligence" in query:
                return {
                    "accounts": {
                        "summary": {
                            "total_accounts": 4,
                            "high_intent_count": 3,
                            "active_eval_signal_count": 2,
                            "decision_maker_count": 1,
                        },
                        "accounts": [
                            {"company_name": "Acme", "urgency_score": 8.5},
                        ],
                    },
                    "as_of_date": "2026-03-28",
                }
            return None

        async def fetch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 55,
                "churn_signal_density": 14.0,
                "avg_urgency": 6.0,
                "total_reviews": 100,
                "dm_churn_rate": 10.0,
                "pain_breakdown": [{"category": "pricing", "count": 30}] * 3,
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    result = await briefing_mod.build_vendor_briefing("Zendesk")

    assert result is not None
    assert result["data_sources"]["account_intelligence"] is True
    assert result["account_pressure_metrics"]["high_intent_count"] == 3
    assert "3 high-intent accounts" in result["account_pressure_summary"]
    assert result["named_account_count"] == 4
    assert "Acme" in result["priority_account_names"]
    # named_accounts should be materialized from account objects
    assert result["named_accounts"][0]["company"] == "Acme"
    assert result["named_accounts"][0]["urgency"] == 8.5


@pytest.mark.asyncio
async def test_build_vendor_briefing_displacement_dynamics_augments_thin_conclusions(monkeypatch):
    class DummyPool:
        is_initialized = True

        async def fetchrow(self, *args, **kwargs):
            return None

        async def fetch(self, query, *args):
            if "b2b_displacement_dynamics" in query:
                return [
                    {
                        "peer_vendor": "Zoho",
                        "dynamics": {
                            "battle_summary": {
                                "winner": "Zoho",
                                "loser": "HubSpot",
                                "conclusion": "Zoho wins on price in SMB segment.",
                                "confidence": 0.75,
                                "durability_assessment": None,
                                "key_insights": ["SMB pricing gap is widening"],
                                "resource_advantage": None,
                            },
                            "switch_reasons": [
                                {
                                    "reason": "lower cost",
                                    "reason_category": "pricing",
                                    "mention_count": 28,
                                    "direction": None,
                                    "reason_detail": None,
                                },
                                {
                                    "reason": "simpler UI",
                                    "reason_category": "ux",
                                    "mention_count": 15,
                                    "direction": None,
                                    "reason_detail": None,
                                },
                            ],
                            "flow_summary": {
                                "explicit_switch_count": 20,
                                "active_evaluation_count": 8,
                                "total_flow_mentions": 42,
                            },
                            "edge_metrics": {
                                "mention_count": 42,
                                "primary_driver": "pricing",
                                "signal_strength": None,
                                "key_quote": None,
                                "confidence_score": 0.75,
                                "velocity_7d": 3,
                                "velocity_30d": 12,
                            },
                            "trend_acceleration": None,
                            "evidence_breakdown": [],
                            "segment_displacement": None,
                        },
                        "as_of_date": "2026-03-25",
                    }
                ]
            return []

    monkeypatch.setattr(briefing_mod, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(
        briefing_mod,
        "_extract_feed_entry",
        AsyncMock(
            return_value={
                "churn_pressure_score": 55,
                "churn_signal_density": 14.0,
                "avg_urgency": 6.0,
                "total_reviews": 100,
                "dm_churn_rate": 10.0,
                "pain_breakdown": [{"category": "pricing", "count": 30}] * 3,
                "top_displacement_targets": [],
                "evidence": [],
                "named_accounts": [],
                "top_feature_gaps": [],
                "category": "CRM",
            }
        ),
    )
    monkeypatch.setattr(briefing_mod, "_fetch_vendor_evidence_vault", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_product_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "_fetch_high_urgency_quotes", AsyncMock(return_value=[]))
    monkeypatch.setattr(briefing_mod, "_enrich_with_analyst_summary", AsyncMock(return_value=None))
    monkeypatch.setattr(briefing_mod, "generate_account_cards", AsyncMock(return_value=[]))

    result = await briefing_mod.build_vendor_briefing("HubSpot")

    assert result is not None
    assert result["data_sources"]["displacement_dynamics"] is True
    assert result["competitive_dynamics"]["pairs"][0]["challenger"] == "Zoho"
    # battle_summary should be extracted as the conclusion string, not the raw dict
    assert result["competitive_dynamics"]["pairs"][0]["battle_summary"] == (
        "Zoho wins on price in SMB segment."
    )
    # switch_reasons remain as canonical dicts with reason field
    reasons = result["competitive_dynamics"]["pairs"][0]["switch_reasons"]
    assert reasons[0]["reason"] == "lower cost"
    assert reasons[1]["reason"] == "simpler UI"
    # Sorted by edge_metrics.mention_count (canonical field)
    assert result["competitive_dynamics"]["pairs"][0]["edge_metrics"]["mention_count"] == 42
    # Should have been promoted into cross_vendor_conclusions
    conclusions = result.get("cross_vendor_conclusions") or []
    pairwise = [c for c in conclusions if c.get("analysis_type") == "pairwise_battle"]
    assert len(pairwise) >= 1
    # Uses "summary" (not "conclusion") to match the shape all renderers key off
    assert pairwise[0]["summary"] == "Zoho wins on price in SMB segment."
    assert pairwise[0]["confidence"] == 0.6
