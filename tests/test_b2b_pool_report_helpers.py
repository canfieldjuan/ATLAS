"""Regression tests for pool-backed comparison report helpers."""

import json
import sys
from datetime import date, datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest

# Pre-mock heavy deps before importing task modules.
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

import atlas_brain.autonomous.tasks.b2b_churn_intelligence as mod
import atlas_brain.autonomous.tasks._b2b_shared as shared_mod


class FakePool:
    def __init__(self, *, fetch_map=None, fetchrow_map=None, fetchval_map=None):
        self.fetch_map = fetch_map or {}
        self.fetchrow_map = fetchrow_map or {}
        self.fetchval_map = fetchval_map or {}
        self.calls = []

    async def fetch(self, query, *params):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        for needle, value in self.fetch_map.items():
            if needle in normalized:
                return value(*params) if callable(value) else value
        raise AssertionError(f"Unexpected fetch query: {normalized}")

    async def fetchrow(self, query, *params):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        for needle, value in self.fetchrow_map.items():
            if needle in normalized:
                return value(*params) if callable(value) else value
        raise AssertionError(f"Unexpected fetchrow query: {normalized}")

    async def fetchval(self, query, *params):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        for needle, value in self.fetchval_map.items():
            if needle in normalized:
                return value(*params) if callable(value) else value
        raise AssertionError(f"Unexpected fetchval query: {normalized}")


def test_count_analyzed_vendors_dedupes_canonical_vendor_rows():
    rows = [
        {"vendor_name": "HubSpot", "product_category": "CRM"},
        {"vendor_name": "hubspot", "product_category": "Marketing"},
        {"vendor": "Intercom"},
        {"vendor_name": ""},
        {"vendor_name": None},
    ]

    assert mod._count_analyzed_vendors(rows) == 2


@pytest.mark.asyncio
async def test_vendor_snapshot_parses_vendor_mentions_shape(monkeypatch):
    monkeypatch.setattr(mod, "_canonicalize_competitor", lambda raw: str(raw or "").strip())
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_evidence_vault": {
                "vendor_name": "Zendesk",
                "vault": json.dumps({
                    "metric_snapshot": {
                        "total_reviews": 40,
                        "churn_density": 20.0,
                        "avg_urgency": 6.5,
                        "positive_review_pct": 45.0,
                        "recommend_ratio": 12.5,
                    },
                    "weakness_evidence": [],
                    "company_signals": [{"company_name": "Acme", "urgency_score": 8.0}],
                }),
                "analysis_window_days": 30,
            },
            "FROM b2b_churn_signals": {
                "total_reviews": 40,
                "churn_intent_count": 8,
                "avg_urgency_score": 6.5,
                "avg_rating_normalized": 0.81,
            },
            "FROM b2b_product_profiles": {
                "commonly_compared_to": json.dumps([
                    {"vendor": "Freshdesk", "mentions": 23},
                    {"vendor": "Intercom", "mentions": 13},
                ]),
                "product_category": "Support",
            },
        },
    )

    snapshot = await mod._vendor_snapshot_from_pools(pool, "Zendesk", 90)

    assert snapshot is not None
    assert snapshot["top_competitors"] == [
        {"competitor": "Freshdesk", "count": 23},
        {"competitor": "Intercom", "count": 13},
    ]
    assert snapshot["high_urgency_count"] == 1


@pytest.mark.asyncio
async def test_read_vendor_scorecards_prefers_specific_category_rows():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {
                    "vendor_name": "Power BI",
                    "product_category": "B2B Software",
                    "total_reviews": 316,
                    "churn_intent": 18,
                    "avg_urgency": 4.0,
                    "materialization_run_id": "run-power-bi-generic",
                    "last_computed_at": "2026-03-21T23:35:44Z",
                    "review_window_end": "2026-03-21",
                },
                {
                    "vendor_name": "Power BI",
                    "product_category": "Data & Analytics",
                    "total_reviews": 278,
                    "churn_intent": 16,
                    "avg_urgency": 4.0,
                    "materialization_run_id": "run-power-bi-specific",
                    "last_computed_at": "2026-03-21T23:35:44Z",
                    "review_window_end": "2026-03-21",
                },
                {
                    "vendor_name": "Zendesk",
                    "product_category": "B2B Software",
                    "total_reviews": 45,
                    "churn_intent": 12,
                    "avg_urgency": 6.2,
                    "materialization_run_id": "run-zendesk",
                    "last_computed_at": "2026-03-21T23:35:44Z",
                    "review_window_end": "2026-03-21",
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_scorecards(pool, window_days=30, min_reviews=3)

    assert rows == [
        {
            "vendor_name": "Power BI",
            "product_category": "Data & Analytics",
            "total_reviews": 278,
            "churn_intent": 16,
            "avg_urgency": 4.0,
            "materialization_run_id": "run-power-bi-specific",
        },
        {
            "vendor_name": "Zendesk",
            "product_category": "B2B Software",
            "total_reviews": 45,
            "churn_intent": 12,
            "avg_urgency": 6.2,
            "materialization_run_id": "run-zendesk",
        },
    ]


@pytest.mark.asyncio
async def test_read_vendor_scorecards_applies_vendor_filter():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [],
        },
    )

    await shared_mod.read_vendor_scorecards(
        pool,
        window_days=30,
        min_reviews=3,
        vendor_names=["Zendesk"],
    )

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals" in call[0])
    assert score_call[1][2] == ["zendesk"]
    assert "LOWER(vendor_name) = ANY($3::text[])" in score_call[0]


def test_align_vendor_intelligence_records_to_scorecards_filters_mismatched_runs():
    vendor_scores = [
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-zendesk",
        },
        {
            "vendor_name": "Freshdesk",
            "materialization_run_id": "run-freshdesk",
        },
        {
            "vendor_name": "Intercom",
        },
    ]
    records = [
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-zendesk",
            "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
        },
        {
            "vendor_name": "Freshdesk",
            "materialization_run_id": "run-stale",
            "vault": {"metric_snapshot": {"avg_urgency": 6.1}},
        },
        {
            "vendor_name": "Intercom",
            "vault": {"metric_snapshot": {"avg_urgency": 5.4}},
        },
    ]

    lookup, alignment = shared_mod._align_vendor_intelligence_records_to_scorecards(
        vendor_scores,
        records,
    )

    assert lookup == {
        "Zendesk": {"metric_snapshot": {"avg_urgency": 7.2}},
        "Intercom": {"metric_snapshot": {"avg_urgency": 5.4}},
    }
    assert alignment["matched_vendor_count"] == 2
    assert alignment["mismatched_vendor_count"] == 1
    assert alignment["mismatched_vendors"] == ["Freshdesk"]
    assert alignment["legacy_scorecard_vendors"] == ["Intercom"]
    assert alignment["legacy_vault_vendors"] == ["Intercom"]


def test_align_vendor_intelligence_record_to_scorecard_filters_mismatched_run():
    vault, alignment = shared_mod._align_vendor_intelligence_record_to_scorecard(
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-current",
        },
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-stale",
            "vault": {"metric_snapshot": {"avg_urgency": 6.9}},
        },
    )

    assert vault is None
    assert alignment["matched_vendor_count"] == 0
    assert alignment["mismatched_vendor_count"] == 1
    assert alignment["mismatched_vendors"] == ["Zendesk"]


@pytest.mark.asyncio
async def test_has_complete_core_run_marker_requires_published_complete_row():
    pool = FakePool(
        fetchval_map={
            "FROM b2b_intelligence": 1,
        },
    )

    assert await shared_mod.has_complete_core_run_marker(pool, date(2026, 4, 10)) is True

    marker_call = next(call for call in pool.calls if "FROM b2b_intelligence" in call[0])
    assert marker_call[1] == (date(2026, 4, 10),)
    assert "status = 'published'" in marker_call[0]
    assert "materialization_complete" in marker_call[0]


@pytest.mark.asyncio
async def test_latest_complete_core_report_date_filters_incomplete_markers():
    pool = FakePool(
        fetchval_map={
            "FROM b2b_intelligence": date(2026, 4, 9),
        },
    )

    assert await shared_mod.latest_complete_core_report_date(pool) == date(2026, 4, 9)

    marker_call = next(call for call in pool.calls if "FROM b2b_intelligence" in call[0])
    assert "status = 'published'" in marker_call[0]
    assert "materialization_complete" in marker_call[0]


@pytest.mark.asyncio
async def test_describe_core_run_gap_reports_incomplete_phase_details():
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_intelligence": {
                "status": "incomplete",
                "intelligence_data": {
                    "materialization_complete": False,
                    "materialization_status": {
                        "failed_phases": ["b2b_evidence_vault", "b2b_churn_signals"],
                        "partial_phases": ["b2b_temporal_intelligence"],
                    },
                },
            },
        },
    )

    reason = await shared_mod.describe_core_run_gap(pool, date(2026, 4, 10))

    assert reason == (
        "Core churn materialization is incomplete for today "
        "(failed: b2b_evidence_vault, b2b_churn_signals; partial: b2b_temporal_intelligence)"
    )


@pytest.mark.asyncio
async def test_describe_core_run_gap_reports_missing_marker_when_absent():
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_intelligence": None,
        },
    )

    reason = await shared_mod.describe_core_run_gap(pool, date(2026, 4, 10))

    assert reason == "Core signals are not available for today"


def test_align_vendor_intelligence_records_to_scorecards_filters_mismatched_runs():
    vendor_scores = [
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-zendesk",
        },
        {
            "vendor_name": "Freshdesk",
            "materialization_run_id": "run-freshdesk",
        },
        {
            "vendor_name": "Intercom",
        },
    ]
    records = [
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-zendesk",
            "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
        },
        {
            "vendor_name": "Freshdesk",
            "materialization_run_id": "run-stale",
            "vault": {"metric_snapshot": {"avg_urgency": 6.1}},
        },
        {
            "vendor_name": "Intercom",
            "vault": {"metric_snapshot": {"avg_urgency": 5.4}},
        },
    ]

    lookup, alignment = shared_mod._align_vendor_intelligence_records_to_scorecards(
        vendor_scores,
        records,
    )

    assert lookup == {
        "Zendesk": {"metric_snapshot": {"avg_urgency": 7.2}},
        "Intercom": {"metric_snapshot": {"avg_urgency": 5.4}},
    }
    assert alignment["matched_vendor_count"] == 2
    assert alignment["mismatched_vendor_count"] == 1
    assert alignment["mismatched_vendors"] == ["Freshdesk"]
    assert alignment["legacy_scorecard_vendors"] == ["Intercom"]
    assert alignment["legacy_vault_vendors"] == ["Intercom"]


def test_align_vendor_intelligence_record_to_scorecard_filters_mismatched_run():
    vault, alignment = shared_mod._align_vendor_intelligence_record_to_scorecard(
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-current",
        },
        {
            "vendor_name": "Zendesk",
            "materialization_run_id": "run-stale",
            "vault": {"metric_snapshot": {"avg_urgency": 6.9}},
        },
    )

    assert vault is None
    assert alignment["matched_vendor_count"] == 0
    assert alignment["mismatched_vendor_count"] == 1
    assert alignment["mismatched_vendors"] == ["Zendesk"]


@pytest.mark.asyncio
async def test_read_vendor_scorecard_details_prefers_specific_category_rows():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {
                    "vendor_name": "Power BI",
                    "product_category": "B2B Software",
                    "total_reviews": 316,
                    "top_competitors": [{"name": "Tableau"}],
                },
                {
                    "vendor_name": "Power BI",
                    "product_category": "Data & Analytics",
                    "total_reviews": 278,
                    "top_competitors": [{"name": "Looker"}],
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_scorecard_details(pool, vendor_names=["Power BI"])

    assert len(rows) == 1
    assert rows[0]["product_category"] == "Data & Analytics"


@pytest.mark.asyncio
async def test_read_vendor_top_competitor_map_supports_jsonb_and_flat_rows():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {"vendor_name": "Acme", "product_category": "CRM", "total_reviews": 12, "top_competitors": [{"name": "Globex"}]},
                {"vendor_name": "Beta", "product_category": "CRM", "total_reviews": 8, "top_competitor": "Initech"},
            ],
        },
    )

    competitors = await shared_mod.read_vendor_top_competitor_map(pool)

    assert competitors == {"Acme": "Globex", "Beta": "Initech"}


@pytest.mark.asyncio
async def test_read_company_churn_context_parses_lists_and_rates():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {
                    "vendor_name": "HubSpot",
                    "product_category": "CRM",
                    "avg_urgency_score": "7.5",
                    "top_pain_categories": json.dumps(["pricing", "support"]),
                    "top_competitors": json.dumps([{"name": "Salesforce"}]),
                    "decision_maker_churn_rate": "0.4",
                    "price_complaint_rate": "0.2",
                },
            ],
        },
    )

    rows = await shared_mod.read_company_churn_context(pool, company_hint="acme", limit=5)

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals" in call[0])
    assert score_call[1] == ("acme", 5)
    assert rows == [
        {
            "vendor_name": "HubSpot",
            "product_category": "CRM",
            "avg_urgency_score": 7.5,
            "top_pain_categories": ["pricing", "support"],
            "top_competitors": [{"name": "Salesforce"}],
            "decision_maker_churn_rate": 0.4,
            "price_complaint_rate": 0.2,
        },
    ]


@pytest.mark.asyncio
async def test_read_signal_product_categories_returns_distinct_trimmed_values():
    pool = FakePool(
        fetch_map={
            "SELECT DISTINCT product_category": [
                {"product_category": "CRM"},
                {"product_category": "crm"},
                {"product_category": " Helpdesk "},
                {"product_category": ""},
            ],
        },
    )

    rows = await shared_mod.read_signal_product_categories(pool)

    assert rows == ["CRM", "Helpdesk"]


@pytest.mark.asyncio
async def test_read_category_vendor_signal_rows_dedupes_per_vendor_with_snapshot_metrics():
    pool = FakePool(
        fetch_map={
            "FROM ranked_signals rs": [
                {
                    "vendor_name": "Zendesk",
                    "total_reviews": 120,
                    "avg_urgency": 6.4,
                    "confidence_score": 0.8,
                    "churn_density": 28.0,
                    "positive_review_pct": 42.0,
                    "displacement_edge_count": 7,
                    "displacement_out": 12,
                    "displacement_in": 4,
                },
            ],
        },
    )

    rows = await shared_mod.read_category_vendor_signal_rows(
        pool,
        product_category="CRM",
    )

    score_call = next(call for call in pool.calls if "FROM ranked_signals rs" in call[0])
    assert score_call[1] == ("CRM",)
    assert "ROW_NUMBER() OVER" in score_call[0]
    assert "WHERE LOWER(product_category) = LOWER($1)" in score_call[0]
    assert "WHERE rs.vendor_row_rank = 1" in score_call[0]
    assert rows == [
        {
            "vendor_name": "Zendesk",
            "total_reviews": 120,
            "avg_urgency": 6.4,
            "confidence_score": 0.8,
            "churn_density": 28.0,
            "positive_review_pct": 42.0,
            "displacement_edge_count": 7,
            "displacement_out": 12,
            "displacement_in": 4,
        },
    ]


@pytest.mark.asyncio
async def test_read_vendor_graph_sync_rows_uses_ranked_scorecard_join():
    pool = FakePool(
        fetch_map={
            "FROM b2b_vendors v": [
                {
                    "canonical_name": "Zendesk",
                    "aliases": ["Zen"],
                    "product_category": "CRM",
                    "total_reviews": 120,
                    "avg_urgency": 6.4,
                    "confidence_score": 0.8,
                    "churn_density": 28.0,
                    "positive_review_pct": 42.0,
                    "recommend_ratio": 0.63,
                    "pain_count": 5,
                    "competitor_count": 3,
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_graph_sync_rows(pool)

    score_call = next(call for call in pool.calls if "FROM b2b_vendors v" in call[0])
    assert score_call[1] == ()
    assert "WITH ranked_signals AS" in score_call[0]
    assert "rs.vendor_row_rank = 1" in score_call[0]
    assert rows == [
        {
            "canonical_name": "Zendesk",
            "aliases": ["Zen"],
            "product_category": "CRM",
            "total_reviews": 120,
            "avg_urgency": 6.4,
            "confidence_score": 0.8,
            "churn_density": 28.0,
            "positive_review_pct": 42.0,
            "recommend_ratio": 0.63,
            "pain_count": 5,
            "competitor_count": 3,
        },
    ]


@pytest.mark.asyncio
async def test_fetch_vendor_vault_row_uses_shared_nearest_window_reader(monkeypatch):
    pool = object()
    reader = AsyncMock(
        return_value={
            "vendor_name": "Zendesk",
            "analysis_window_days": 60,
            "vault": {"metric_snapshot": {"total_reviews": 42}},
        },
    )
    monkeypatch.setattr(mod, "_read_vendor_intelligence_record_nearest_window", reader)

    vault, resolved_window = await mod._fetch_vendor_vault_row(pool, "Zendesk", 90)

    reader.assert_awaited_once_with(pool, "Zendesk", 90)
    assert vault == {"metric_snapshot": {"total_reviews": 42}}
    assert resolved_window == 60


@pytest.mark.asyncio
async def test_vendor_snapshot_from_pools_uses_shared_scorecard_metrics(monkeypatch):
    monkeypatch.setattr(
        mod,
        "_fetch_vendor_vault_row",
        AsyncMock(
            return_value=(
                {
                    "metric_snapshot": {
                        "total_reviews": 40,
                        "churn_density": 20.0,
                        "avg_urgency": 6.5,
                        "positive_review_pct": 45.0,
                        "recommend_ratio": 12.5,
                    },
                    "weakness_evidence": [],
                    "company_signals": [{"company_name": "Acme", "urgency_score": 8.0}],
                },
                30,
            ),
        ),
    )
    scorecard = AsyncMock(
        return_value={
            "total_reviews": 40,
            "churn_intent_count": 8,
            "avg_urgency_score": 6.5,
            "avg_rating_normalized": 0.81,
        },
    )
    monkeypatch.setattr(mod, "_read_vendor_scorecard_metrics", scorecard)

    class Pool:
        async def fetchrow(self, query, *args):
            assert "FROM b2b_product_profiles" in query
            assert args == ("Zendesk",)
            return {
                "commonly_compared_to": json.dumps([
                    {"vendor": "Freshdesk", "mentions": 23},
                    {"vendor": "Intercom", "mentions": 13},
                ]),
                "product_category": "Support",
            }

    snapshot = await mod._vendor_snapshot_from_pools(Pool(), "Zendesk", 90)

    scorecard.assert_awaited_once()
    assert snapshot is not None
    assert snapshot["avg_rating_normalized"] == 81.0
    assert snapshot["top_competitors"][0]["competitor"] == "Freshdesk"


@pytest.mark.asyncio
async def test_generate_company_deep_dive_report_uses_shared_archetype_lookup(monkeypatch):
    pool = object()
    snapshot = {
        "company_name": "Acme Corp",
        "signal_count": 2,
        "avg_urgency_score": 6.1,
        "max_urgency_score": 8.2,
        "decision_maker_signals": 1,
        "churn_intent_count": 1,
        "current_vendors": [{"vendor": "Zendesk"}],
        "product_categories": [],
        "top_pain_categories": [{"category": "pricing"}],
        "alternatives_considered": [{"name": "Freshdesk"}],
        "timeline_signals": [],
    }
    monkeypatch.setattr(mod, "_company_snapshot_from_signals", AsyncMock(return_value=snapshot))
    lookup = AsyncMock(return_value={"Zendesk": {"archetype": "pricing_pressure", "confidence": 0.82}})
    monkeypatch.setattr(mod, "_build_vendor_archetype_lookup", lookup)

    report = await mod.generate_company_deep_dive_report(
        pool,
        "Acme Corp",
        window_days=90,
        persist=False,
    )

    lookup.assert_awaited_once_with(pool, ["Zendesk"])
    assert report is not None
    assert report["vendor_archetypes"]["Zendesk"]["archetype"] == "pricing_pressure"


@pytest.mark.asyncio
async def test_generate_company_comparison_report_uses_shared_archetype_lookup(monkeypatch):
    pool = object()
    primary_snapshot = {
        "company_name": "Acme Corp",
        "signal_count": 2,
        "avg_urgency_score": 6.1,
        "max_urgency_score": 8.2,
        "decision_maker_signals": 1,
        "churn_intent_count": 1,
        "current_vendors": [{"vendor": "Zendesk"}],
        "product_categories": [],
        "top_pain_categories": [{"category": "pricing"}],
        "alternatives_considered": [{"name": "Freshdesk"}],
        "timeline_signals": [],
    }
    comparison_snapshot = {
        "company_name": "Globex",
        "signal_count": 3,
        "avg_urgency_score": 5.4,
        "max_urgency_score": 7.3,
        "decision_maker_signals": 2,
        "churn_intent_count": 1,
        "current_vendors": [{"vendor": "HubSpot"}],
        "product_categories": [],
        "top_pain_categories": [{"category": "automation"}],
        "alternatives_considered": [{"name": "Freshdesk"}],
        "timeline_signals": [],
    }
    monkeypatch.setattr(
        mod,
        "_company_snapshot_from_signals",
        AsyncMock(side_effect=[primary_snapshot, comparison_snapshot]),
    )
    lookup = AsyncMock(
        return_value={
            "HubSpot": {"archetype": "feature_gap", "confidence": 0.71},
            "Zendesk": {"archetype": "pricing_pressure", "confidence": 0.82},
        },
    )
    monkeypatch.setattr(mod, "_build_vendor_archetype_lookup", lookup)

    report = await mod.generate_company_comparison_report(
        pool,
        "Acme Corp",
        "Globex",
        window_days=90,
        persist=False,
    )

    lookup.assert_awaited_once()
    assert report is not None
    assert report["vendor_archetypes"]["Zendesk"]["archetype"] == "pricing_pressure"
    assert report["vendor_archetypes"]["HubSpot"]["archetype"] == "feature_gap"


@pytest.mark.asyncio
async def test_read_vendor_scorecard_inventory_rows_matches_provisioning_shape():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {
                    "vendor_name": "OpenProject",
                    "product_category": "Project Management",
                    "total_reviews_analyzed": 80,
                    "confidence_score": 0.0,
                    "last_computed_at": None,
                    "inventory_source": "b2b_churn_signals",
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_scorecard_inventory_rows(pool)

    assert rows == [
        {
            "vendor_name": "OpenProject",
            "product_category": "Project Management",
            "total_reviews_analyzed": 80,
            "confidence_score": 0.0,
            "last_computed_at": None,
            "inventory_source": "b2b_churn_signals",
        },
    ]


@pytest.mark.asyncio
async def test_read_vendor_scorecard_metrics_returns_latest_metric_row():
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_churn_signals": {
                "price_complaint_rate": 0.2,
                "decision_maker_churn_rate": 0.4,
                "total_reviews": 40,
                "signal_reviews": 20,
                "churn_intent_count": 5,
                "avg_urgency_score": 6.5,
                "avg_rating_normalized": 0.81,
                "top_competitors": [{"name": "Freshdesk"}],
                "sentiment_distribution": {"declining": 4, "improving": 1},
                "materialization_run_id": "run-123",
            },
        },
    )

    row = await shared_mod.read_vendor_scorecard_metrics(pool, vendor_name="Zendesk")

    assert row == {
        "price_complaint_rate": 0.2,
        "decision_maker_churn_rate": 0.4,
        "total_reviews": 40,
        "signal_reviews": 20,
        "churn_intent_count": 5,
        "avg_urgency_score": 6.5,
        "avg_rating_normalized": 0.81,
        "top_competitors": [{"name": "Freshdesk"}],
        "sentiment_distribution": {"declining": 4, "improving": 1},
        "materialization_run_id": "run-123",
    }


@pytest.mark.asyncio
async def test_read_vendor_intelligence_record_prefers_latest_exact_vendor_row():
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [
                {
                    "vendor_name": "Zendesk",
                    "as_of_date": date(2026, 3, 31),
                    "analysis_window_days": 30,
                    "schema_version": 2,
                    "materialization_run_id": "run-123",
                    "vault": json.dumps({"metric_snapshot": {"avg_urgency": 7.2}}),
                    "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
                },
            ],
        },
    )

    row = await shared_mod.read_vendor_intelligence_record(
        pool,
        "Zendesk",
        as_of=date(2026, 4, 1),
        analysis_window_days=30,
    )

    evidence_call = next(call for call in pool.calls if "FROM b2b_evidence_vault" in call[0])
    assert "SELECT DISTINCT ON (vendor_name)" in evidence_call[0]
    assert evidence_call[1] == (date(2026, 4, 1), 30, ["zendesk"])
    assert row == {
        "vendor_name": "Zendesk",
        "as_of_date": date(2026, 3, 31),
        "analysis_window_days": 30,
        "schema_version": 2,
        "materialization_run_id": "run-123",
        "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
        "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
    }


@pytest.mark.asyncio
async def test_search_vendor_intelligence_record_uses_partial_vendor_query():
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_evidence_vault": {
                "vendor_name": "Zendesk",
                "as_of_date": date(2026, 3, 31),
                "analysis_window_days": 30,
                "schema_version": 2,
                "materialization_run_id": "run-123",
                "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
                "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
            },
        },
    )

    row = await shared_mod.search_vendor_intelligence_record(
        pool,
        vendor_query="zen",
        as_of=date(2026, 4, 1),
        analysis_window_days=30,
    )

    evidence_call = next(call for call in pool.calls if "FROM b2b_evidence_vault" in call[0])
    assert "vendor_name ILIKE '%' || $1 || '%'" in evidence_call[0]
    assert "LIMIT 1" in evidence_call[0]
    assert evidence_call[1] == ("zen", date(2026, 4, 1), 30)
    assert row == {
        "vendor_name": "Zendesk",
        "as_of_date": date(2026, 3, 31),
        "analysis_window_days": 30,
        "schema_version": 2,
        "materialization_run_id": "run-123",
        "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
        "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
    }


@pytest.mark.asyncio
async def test_read_vendor_intelligence_record_nearest_window_prefers_closest_match():
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_evidence_vault": {
                "vendor_name": "Zendesk",
                "as_of_date": date(2026, 3, 31),
                "analysis_window_days": 60,
                "schema_version": 2,
                "materialization_run_id": "run-123",
                "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
                "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
            },
        },
    )

    row = await shared_mod.read_vendor_intelligence_record_nearest_window(
        pool,
        vendor_name="Zendesk",
        analysis_window_days=90,
    )

    evidence_call = next(call for call in pool.calls if "FROM b2b_evidence_vault" in call[0])
    assert "ABS(analysis_window_days - $2)" in evidence_call[0]
    assert evidence_call[1] == ("Zendesk", 90)
    assert row["analysis_window_days"] == 60
    assert row["vault"]["metric_snapshot"]["avg_urgency"] == 7.2


@pytest.mark.asyncio
async def test_read_vendor_intelligence_records_latest_returns_latest_rows():
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [
                {
                    "vendor_name": "Zendesk",
                    "as_of_date": date(2026, 3, 31),
                    "analysis_window_days": 60,
                    "schema_version": 2,
                    "materialization_run_id": "run-123",
                    "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
                    "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_intelligence_records_latest(
        pool,
        vendor_names=["Zendesk"],
    )

    evidence_call = next(call for call in pool.calls if "FROM b2b_evidence_vault" in call[0])
    assert "SELECT DISTINCT ON (vendor_name)" in evidence_call[0]
    assert "WHERE LOWER(vendor_name) = ANY($1::text[])" in evidence_call[0]
    assert evidence_call[1] == (["zendesk"],)
    assert rows == [
        {
            "vendor_name": "Zendesk",
            "as_of_date": date(2026, 3, 31),
            "analysis_window_days": 60,
            "schema_version": 2,
            "materialization_run_id": "run-123",
            "vault": {"metric_snapshot": {"avg_urgency": 7.2}},
            "created_at": datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        },
    ]


@pytest.mark.asyncio
async def test_read_vendor_scorecard_archetypes_returns_latest_non_null_rows():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {
                    "vendor_name": "Zendesk",
                    "archetype": "pricing_pressure",
                    "archetype_confidence": 0.82,
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_scorecard_archetypes(
        pool,
        vendor_names=["Zendesk"],
    )

    signal_call = next(call for call in pool.calls if "FROM b2b_churn_signals" in call[0])
    assert "SELECT DISTINCT ON (vendor_name)" in signal_call[0]
    assert "archetype IS NOT NULL" in signal_call[0]
    assert signal_call[1] == (["zendesk"],)
    assert rows == [
        {
            "vendor_name": "Zendesk",
            "archetype": "pricing_pressure",
            "archetype_confidence": 0.82,
        },
    ]


@pytest.mark.asyncio
async def test_read_vendor_signal_detail_applies_snapshot_scope_and_suppression():
    tracked_account_id = "11111111-1111-1111-1111-111111111111"
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_churn_signals sig": {
                "vendor_name": "Zendesk",
                "product_category": "CRM",
                "support_sentiment": -0.1,
            },
        },
    )

    row = await shared_mod.read_vendor_signal_detail(
        pool,
        vendor_name_query="Zendesk",
        product_category="CRM",
        tracked_account_id=tracked_account_id,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
    )

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals sig" in call[0])
    assert score_call[1] == ("Zendesk", "CRM", tracked_account_id)
    assert "LEFT JOIN LATERAL" in score_call[0]
    assert "sig.product_category = $2" in score_call[0]
    assert "tracked_vendors WHERE account_id = $3::uuid" in score_call[0]
    assert "dc.entity_type = 'churn_signal'" in score_call[0]
    assert row == {
        "vendor_name": "Zendesk",
        "product_category": "CRM",
        "support_sentiment": -0.1,
    }


@pytest.mark.asyncio
async def test_read_vendor_signal_detail_exact_applies_exact_scope_and_suppression():
    tracked_account_id = "11111111-1111-1111-1111-111111111111"
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_churn_signals sig": {
                "vendor_name": "Zendesk",
                "product_category": "CRM",
                "support_sentiment": -0.1,
            },
        },
    )

    row = await shared_mod.read_vendor_signal_detail_exact(
        pool,
        vendor_name="Zendesk",
        product_category="CRM",
        tracked_account_id=tracked_account_id,
        include_snapshot_metrics=True,
        exclude_suppressed=True,
    )

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals sig" in call[0])
    assert score_call[1] == ("Zendesk", "CRM", tracked_account_id)
    assert "LEFT JOIN LATERAL" in score_call[0]
    assert "LOWER(sig.vendor_name) = LOWER($1)" in score_call[0]
    assert "sig.product_category = $2" in score_call[0]
    assert "tracked_vendors WHERE account_id = $3::uuid" in score_call[0]
    assert "dc.entity_type = 'churn_signal'" in score_call[0]
    assert row == {
        "vendor_name": "Zendesk",
        "product_category": "CRM",
        "support_sentiment": -0.1,
    }


@pytest.mark.asyncio
async def test_read_vendor_signal_rows_applies_snapshot_filters_and_limit():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals sig": [
                {
                    "vendor_name": "Zendesk",
                    "product_category": "CRM",
                    "total_reviews": 120,
                    "churn_intent_count": 22,
                    "avg_urgency_score": 6.4,
                    "avg_rating_normalized": 0.41,
                    "nps_proxy": -0.2,
                    "price_complaint_rate": 0.18,
                    "decision_maker_churn_rate": 0.12,
                    "keyword_spike_count": 2,
                    "insider_signal_count": 1,
                    "last_computed_at": None,
                    "support_sentiment": -0.1,
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_signal_rows(
        pool,
        vendor_name_query="Zendesk",
        min_urgency=5.0,
        product_category="CRM",
        tracked_account_id="11111111-1111-1111-1111-111111111111",
        include_snapshot_metrics=True,
        exclude_suppressed=True,
        limit=15,
    )

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals sig" in call[0])
    assert score_call[1] == ("11111111-1111-1111-1111-111111111111", "Zendesk", 5.0, "CRM", 15)
    assert "LEFT JOIN LATERAL" in score_call[0]
    assert "sig.vendor_name ILIKE '%' || $2 || '%'" in score_call[0]
    assert "sig.avg_urgency_score >= $3" in score_call[0]
    assert "sig.product_category = $4" in score_call[0]
    assert "dc.entity_type = 'churn_signal'" in score_call[0]
    assert rows[0]["keyword_spike_count"] == 2
    assert rows[0]["support_sentiment"] == -0.1


@pytest.mark.asyncio
async def test_read_vendor_signal_rows_caps_export_limit_at_ten_thousand():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals sig": [],
        },
    )

    await shared_mod.read_vendor_signal_rows(
        pool,
        limit=25_000,
    )

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals sig" in call[0])
    assert score_call[1] == (10_000,)


@pytest.mark.asyncio
async def test_read_best_vendor_signal_rows_dedupes_per_vendor_and_applies_filters():
    pool = FakePool(
        fetch_map={
            "WITH ranked_signals AS": [
                {
                    "vendor_name": "Zendesk",
                    "product_category": "CRM",
                    "total_reviews": 120,
                    "churn_intent_count": 22,
                    "avg_urgency_score": 6.4,
                    "nps_proxy": -0.2,
                    "last_computed_at": None,
                },
            ],
        },
    )

    rows = await shared_mod.read_best_vendor_signal_rows(
        pool,
        vendor_name_query="Zendesk",
        tracked_account_id="11111111-1111-1111-1111-111111111111",
        exclude_suppressed=True,
        limit=15,
    )

    score_call = next(call for call in pool.calls if "WITH ranked_signals AS" in call[0])
    assert score_call[1] == ("11111111-1111-1111-1111-111111111111", "Zendesk", 15)
    assert "sig.vendor_name ILIKE '%' || $2 || '%'" in score_call[0]
    assert "tracked_vendors WHERE account_id = $1::uuid" in score_call[0]
    assert "dc.entity_type = 'churn_signal'" in score_call[0]
    assert "PARTITION BY sig.vendor_name" in score_call[0]
    assert rows == [
        {
            "vendor_name": "Zendesk",
            "product_category": "CRM",
            "total_reviews": 120,
            "churn_intent_count": 22,
            "avg_urgency_score": 6.4,
            "nps_proxy": -0.2,
            "last_computed_at": None,
        },
    ]


@pytest.mark.asyncio
async def test_read_vendor_signal_summary_applies_exact_vendor_filters():
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_churn_signals sig": {
                "total_vendors": 2,
                "high_urgency_count": 1,
                "total_signal_reviews": 180,
            },
        },
    )

    row = await shared_mod.read_vendor_signal_summary(
        pool,
        vendor_names=["Zendesk", "Freshdesk"],
        tracked_account_id="11111111-1111-1111-1111-111111111111",
    )

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals sig" in call[0])
    assert score_call[1] == ("11111111-1111-1111-1111-111111111111", ["freshdesk", "zendesk"])
    assert "LOWER(sig.vendor_name) = ANY($2::text[])" in score_call[0]
    assert row == {
        "total_vendors": 2,
        "high_urgency_count": 1,
        "total_signal_reviews": 180,
    }


@pytest.mark.asyncio
async def test_read_vendor_signal_overview_applies_account_scope():
    pool = FakePool(
        fetchrow_map={
            "FROM b2b_churn_signals sig": {
                "avg_urgency": 4.2,
                "total_churn_signals": 22,
                "total_reviews": 330,
            },
        },
    )

    row = await shared_mod.read_vendor_signal_overview(
        pool,
        tracked_account_id="11111111-1111-1111-1111-111111111111",
    )

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals sig" in call[0])
    assert score_call[1] == ("11111111-1111-1111-1111-111111111111",)
    assert "tracked_vendors WHERE account_id = $1::uuid" in score_call[0]
    assert row == {
        "avg_urgency": 4.2,
        "total_churn_signals": 22,
        "total_reviews": 330,
    }


@pytest.mark.asyncio
async def test_read_ranked_vendor_signal_rows_applies_snapshot_activity_and_suppression():
    pool = FakePool(
        fetch_map={
            "WITH ranked_signals AS": [
                {
                    "vendor_name": "Zendesk",
                    "product_category": "Helpdesk",
                    "total_reviews": 300,
                    "churn_intent_count": 22,
                    "avg_urgency_score": 8.2,
                    "avg_rating_normalized": 4.1,
                    "nps_proxy": 18.2,
                    "price_complaint_rate": 0.3,
                    "decision_maker_churn_rate": 0.2,
                    "keyword_spike_count": 2,
                    "insider_signal_count": 1,
                    "last_computed_at": None,
                    "support_sentiment": 2.4,
                    "legacy_support_score": 1.8,
                    "new_feature_velocity": 0.4,
                    "employee_growth_rate": 3.2,
                },
            ],
        },
    )

    rows = await shared_mod.read_ranked_vendor_signal_rows(
        pool,
        vendor_names=["Zendesk"],
        tracked_account_id="11111111-1111-1111-1111-111111111111",
        exclude_suppressed=True,
        require_snapshot_activity=True,
        limit=10,
    )

    score_call = next(call for call in pool.calls if "WITH ranked_signals AS" in call[0])
    assert score_call[1] == ("11111111-1111-1111-1111-111111111111", ["zendesk"], 10)
    assert "snap.support_sentiment IS NOT NULL" in score_call[0]
    assert "LOWER(sig.vendor_name) = ANY($2::text[])" in score_call[0]
    assert "dc.entity_type = 'churn_signal'" in score_call[0]
    assert rows[0]["employee_growth_rate"] == 3.2
    assert rows[0]["keyword_spike_count"] == 2


@pytest.mark.asyncio
async def test_read_market_landscape_candidates_uses_category_vendor_floor():
    pool = FakePool(
        fetch_map={
            "FROM b2b_product_profiles pp": [
                {
                    "category": "CRM",
                    "vendor_count": 4,
                    "total_reviews": 220,
                    "avg_urgency": 6.4,
                },
            ],
        },
    )

    rows = await shared_mod.read_market_landscape_candidates(
        pool,
        min_vendor_profiles=4,
    )

    score_call = next(call for call in pool.calls if "FROM b2b_product_profiles pp" in call[0])
    assert score_call[1] == (4, 10)
    assert "JOIN b2b_churn_signals cs" in score_call[0]
    assert rows == [
        {
            "category": "CRM",
            "vendor_count": 4,
            "total_reviews": 220,
            "avg_urgency": 6.4,
        },
    ]


@pytest.mark.asyncio
async def test_read_vendor_alternative_candidates_uses_scorecard_thresholds():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals cs": [
                {
                    "vendor": "HubSpot",
                    "category": "CRM",
                    "urgency": 8.4,
                    "review_count": 42,
                    "affiliate_id": uuid4(),
                    "affiliate_name": "Partner",
                    "affiliate_product": "CRM Boost",
                    "affiliate_url": "https://example.com",
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_alternative_candidates(pool)

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals cs" in call[0])
    assert score_call[1] == (6.0, 5, 15)
    assert "LEFT JOIN affiliate_partners ap" in score_call[0]
    assert rows[0]["vendor"] == "HubSpot"
    assert rows[0]["has_affiliate"] is True


@pytest.mark.asyncio
async def test_read_vendor_showdown_candidates_uses_review_floor():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals a": [
                {
                    "vendor_a": "Zendesk",
                    "vendor_b": "Freshdesk",
                    "category": "Support",
                    "reviews_a": 50,
                    "reviews_b": 47,
                    "total_reviews": 97,
                    "urgency_a": 7.1,
                    "urgency_b": 6.4,
                    "pain_diff": 0.7,
                },
            ],
        },
    )

    rows = await shared_mod.read_vendor_showdown_candidates(pool)

    score_call = next(call for call in pool.calls if "FROM b2b_churn_signals a" in call[0])
    assert score_call[1] == (10, 80)
    assert "JOIN b2b_churn_signals b" in score_call[0]
    assert rows == [
        {
            "vendor_a": "Zendesk",
            "vendor_b": "Freshdesk",
            "category": "Support",
            "reviews_a": 50,
            "reviews_b": 47,
            "total_reviews": 97,
            "urgency_a": 7.1,
            "urgency_b": 6.4,
            "pain_diff": 0.7,
        },
    ]


@pytest.mark.asyncio
async def test_read_churn_report_candidates_uses_negative_review_thresholds():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {
                    "vendor": "HubSpot",
                    "category": "CRM",
                    "negative_reviews": 11,
                    "avg_urgency": 7.2,
                    "total_reviews": 52,
                },
            ],
        },
    )

    rows = await shared_mod.read_churn_report_candidates(pool)

    score_call = next(
        call for call in pool.calls
        if "FROM b2b_churn_signals" in call[0] and "negative_reviews >=" in call[0]
    )
    assert score_call[1] == (8, 6.0, 10)
    assert rows == [
        {
            "vendor": "HubSpot",
            "category": "CRM",
            "negative_reviews": 11,
            "avg_urgency": 7.2,
            "total_reviews": 52,
        },
    ]


@pytest.mark.asyncio
async def test_read_category_vendor_rows_uses_profile_lookup_by_category():
    pool = FakePool(
        fetch_map={
            "FROM b2b_product_profiles": [
                {"vendor_name": "Zendesk", "product_category": "Helpdesk"},
            ],
        },
    )

    rows = await shared_mod.read_category_vendor_rows(
        pool,
        category_names=["Helpdesk"],
        limit=10,
    )

    score_call = next(call for call in pool.calls if "FROM b2b_product_profiles" in call[0])
    assert score_call[1] == (["helpdesk"], 10)
    assert "JOIN b2b_churn_signals cs" not in score_call[0]
    assert rows == [{"vendor_name": "Zendesk", "product_category": "Helpdesk"}]


@pytest.mark.asyncio
async def test_read_category_vendor_rows_can_require_scorecard_match():
    pool = FakePool(
        fetch_map={
            "JOIN b2b_churn_signals cs": [
                {"vendor_name": "HubSpot", "product_category": "CRM"},
            ],
        },
    )

    rows = await shared_mod.read_category_vendor_rows(
        pool,
        category_names=["CRM"],
        require_scorecard_match=True,
    )

    score_call = next(call for call in pool.calls if "JOIN b2b_churn_signals cs" in call[0])
    assert score_call[1] == (["crm"],)
    assert rows == [{"vendor_name": "HubSpot", "product_category": "CRM"}]


@pytest.mark.asyncio
async def test_read_known_vendor_names_orders_non_empty_vendors():
    pool = FakePool(
        fetch_map={
            "FROM b2b_churn_signals": [
                {"vendor_name": "HubSpot"},
                {"vendor_name": ""},
                {"vendor_name": None},
                {"vendor_name": "Zendesk"},
            ],
        },
    )

    rows = await shared_mod.read_known_vendor_names(pool)

    score_call = next(
        call for call in pool.calls
        if "SELECT DISTINCT vendor_name" in call[0] and "FROM b2b_churn_signals" in call[0]
    )
    assert "ORDER BY vendor_name" in score_call[0]
    assert rows == ["HubSpot", "Zendesk"]


@pytest.mark.asyncio
async def test_fetch_all_pool_layers_prefers_specific_profile_category_over_generic_vault():
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [
                {"vendor_name": "Zendesk", "vault": {"product_category": "B2B Software"}},
            ],
            "FROM b2b_segment_intelligence": [],
            "FROM b2b_temporal_intelligence": [],
            "FROM b2b_account_intelligence": [],
            "FROM b2b_displacement_dynamics": [],
            "FROM b2b_product_profiles": [
                {"vendor_name": "Zendesk", "product_category": "Helpdesk"},
            ],
            "FROM b2b_category_dynamics": [
                {
                    "category": "B2B Software",
                    "dynamics": {"category": "B2B Software", "vendor_count": 1},
                },
                {
                    "category": "Helpdesk",
                    "dynamics": {"category": "Helpdesk", "vendor_count": 2},
                },
            ],
        },
    )

    layers = await shared_mod.fetch_all_pool_layers(
        pool,
        as_of=mod.date.today(),
        analysis_window_days=30,
    )

    assert layers["Zendesk"]["evidence_vault"]["product_category"] == "B2B Software"
    assert layers["Zendesk"]["category"] == {"category": "Helpdesk", "vendor_count": 2}


@pytest.mark.asyncio
async def test_fetch_all_pool_layers_skips_generic_category_without_specific_profile():
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [
                {"vendor_name": "Unknown Vendor", "vault": {"product_category": "B2B Software"}},
            ],
            "FROM b2b_segment_intelligence": [],
            "FROM b2b_temporal_intelligence": [],
            "FROM b2b_account_intelligence": [],
            "FROM b2b_displacement_dynamics": [],
            "FROM b2b_product_profiles": [],
            "FROM b2b_category_dynamics": [
                {
                    "category": "B2B Software",
                    "dynamics": {"category": "B2B Software", "vendor_count": 9},
                },
            ],
        },
    )

    layers = await shared_mod.fetch_all_pool_layers(
        pool,
        as_of=mod.date.today(),
        analysis_window_days=30,
    )

    assert "category" not in layers["Unknown Vendor"]


@pytest.mark.asyncio
async def test_fetch_all_pool_layers_canonicalizes_displacement_from_vendor_key(monkeypatch):
    monkeypatch.setattr(
        shared_mod,
        "_canonicalize_vendor",
        lambda raw: "Amazon Web Services" if str(raw or "").strip().lower() == "aws" else str(raw or "").strip(),
    )
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [],
            "FROM b2b_segment_intelligence": [],
            "FROM b2b_temporal_intelligence": [],
            "FROM b2b_account_intelligence": [],
            "FROM b2b_displacement_dynamics": [
                {"from_vendor": "AWS", "to_vendor": "GCP", "dynamics": {"to_vendor": "Google Cloud Platform"}},
            ],
            "FROM b2b_product_profiles": [],
            "FROM b2b_category_dynamics": [],
        },
    )

    layers = await shared_mod.fetch_all_pool_layers(
        pool,
        as_of=mod.date.today(),
        analysis_window_days=30,
    )

    assert "Amazon Web Services" in layers
    assert layers["Amazon Web Services"]["displacement"][0]["to_vendor"] == "Google Cloud Platform"


@pytest.mark.asyncio
async def test_fetch_all_pool_layers_applies_vendor_filter_to_queries():
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [
                {"vendor_name": "Zendesk", "vault": {"product_category": "Helpdesk"}},
            ],
            "FROM b2b_segment_intelligence": [],
            "FROM b2b_temporal_intelligence": [],
            "FROM b2b_account_intelligence": [],
            "FROM b2b_displacement_dynamics": [],
            "FROM b2b_product_profiles": [
                {"vendor_name": "Zendesk", "product_category": "Helpdesk"},
            ],
            "FROM b2b_category_dynamics": [],
            "FROM b2b_reviews r LEFT JOIN b2b_account_resolution ar": [],
        },
    )

    layers = await shared_mod.fetch_all_pool_layers(
        pool,
        as_of=mod.date.today(),
        analysis_window_days=30,
        vendor_names=["Zendesk"],
    )

    assert "Zendesk" in layers
    evidence_call = next(call for call in pool.calls if "FROM b2b_evidence_vault" in call[0])
    assert evidence_call[1][2] == ["zendesk"]
    profile_call = next(call for call in pool.calls if "FROM b2b_product_profiles" in call[0])
    assert profile_call[1][0] == ["zendesk"]
    review_call = next(call for call in pool.calls if "FROM b2b_reviews r LEFT JOIN b2b_account_resolution ar" in call[0])
    assert review_call[1][2] == ["zendesk"]
    assert "LOWER(r.vendor_name) = ANY($3::text[])" in review_call[0]


@pytest.mark.asyncio
async def test_fetch_all_pool_layers_uses_resolved_company_for_review_candidates():
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [],
            "FROM b2b_segment_intelligence": [],
            "FROM b2b_temporal_intelligence": [],
            "FROM b2b_account_intelligence": [],
            "FROM b2b_displacement_dynamics": [],
            "FROM b2b_product_profiles": [],
            "FROM b2b_category_dynamics": [],
            "FROM b2b_reviews r LEFT JOIN b2b_account_resolution ar": [
                {
                    "vendor_name": "Zendesk",
                    "id": uuid4(),
                    "source": "g2",
                    "rating": 2.0,
                    "rating_max": 5.0,
                    "summary": "We are evaluating options.",
                    "review_text": "The renewal quote forced us to revisit the stack.",
                    "pros": "",
                    "cons": "",
                    "reviewer_title": "IT Manager",
                    "reviewer_company": "Acme Corp",
                    "raw_reviewer_company": "",
                    "resolution_confidence": "medium",
                    "reviewed_at": None,
                    "imported_at": None,
                    "raw_metadata": json.dumps({"source_weight": 0.8}),
                    "enrichment": json.dumps({"pain_category": "pricing"}),
                }
            ],
        },
    )

    layers = await shared_mod.fetch_all_pool_layers(
        pool,
        as_of=mod.date.today(),
        analysis_window_days=30,
    )

    review = layers["Zendesk"]["reviews"][0]
    assert review["reviewer_company"] == "Acme Corp"
    assert review["raw_reviewer_company"] == ""
    assert review["resolution_confidence"] == "medium"


@pytest.mark.asyncio
async def test_fetch_all_pool_layers_ignores_low_confidence_resolved_company_for_review_candidates():
    pool = FakePool(
        fetch_map={
            "FROM b2b_evidence_vault": [],
            "FROM b2b_segment_intelligence": [],
            "FROM b2b_temporal_intelligence": [],
            "FROM b2b_account_intelligence": [],
            "FROM b2b_displacement_dynamics": [],
            "FROM b2b_product_profiles": [],
            "FROM b2b_category_dynamics": [],
            "FROM b2b_reviews r LEFT JOIN b2b_account_resolution ar": [
                {
                    "vendor_name": "Zendesk",
                    "id": uuid4(),
                    "source": "g2",
                    "rating": 2.0,
                    "rating_max": 5.0,
                    "summary": "We are evaluating options.",
                    "review_text": "The renewal quote forced us to revisit the stack.",
                    "pros": "",
                    "cons": "",
                    "reviewer_title": "IT Manager",
                    "reviewer_company": "Raw Acme",
                    "raw_reviewer_company": "Raw Acme",
                    "resolution_confidence": "low",
                    "reviewed_at": None,
                    "imported_at": None,
                    "raw_metadata": json.dumps({"source_weight": 0.8}),
                    "enrichment": json.dumps({"pain_category": "pricing"}),
                }
            ],
        },
    )

    layers = await shared_mod.fetch_all_pool_layers(
        pool,
        as_of=mod.date.today(),
        analysis_window_days=30,
    )

    review = layers["Zendesk"]["reviews"][0]
    assert review["reviewer_company"] == "Raw Acme"
    assert review["raw_reviewer_company"] == "Raw Acme"
    assert review["resolution_confidence"] == "low"


def test_resolve_vendor_category_prefers_specific_profile_over_generic():
    preferred = {"Zendesk": "Helpdesk", "HubSpot": "Marketing Automation"}
    assert mod._resolve_vendor_category("Zendesk", "B2B Software", preferred) == "Helpdesk"
    assert mod._resolve_vendor_category("HubSpot", "Marketing Automation", preferred) == "Marketing Automation"
    assert mod._resolve_vendor_category("Unknown Vendor", "B2B Software", preferred) == "B2B Software"


def test_should_persist_category_dynamics_false_for_scoped_run():
    assert mod._should_persist_category_dynamics(["Zendesk"]) is False
    assert mod._should_persist_category_dynamics(["Zendesk", "Freshdesk"]) is False


def test_should_persist_category_dynamics_true_for_full_run():
    assert mod._should_persist_category_dynamics([]) is True
    assert mod._should_persist_category_dynamics(None) is True


def test_should_persist_cross_vendor_conclusions_false_for_scoped_run():
    assert mod._should_persist_cross_vendor_conclusions(["Zendesk"]) is False
    assert mod._should_persist_cross_vendor_conclusions(["Zendesk", "Freshdesk"]) is False


def test_should_persist_cross_vendor_conclusions_true_for_full_run():
    assert mod._should_persist_cross_vendor_conclusions([]) is True
    assert mod._should_persist_cross_vendor_conclusions(None) is True


def test_should_use_cross_vendor_category_rejects_generic():
    assert mod._should_use_cross_vendor_category("B2B Software") is False
    assert mod._should_use_cross_vendor_category("unknown") is False
    assert mod._should_use_cross_vendor_category("Helpdesk") is True


def test_competitive_disp_from_dynamics_recovers_counts_from_reason_directions():
    rows = shared_mod._competitive_disp_from_dynamics({
        "HubSpot": [
            {
                "to_vendor": "ActiveCampaign",
                "flow_summary": {
                    "total_flow_mentions": 0,
                    "explicit_switch_count": 0,
                    "active_evaluation_count": 0,
                },
                "edge_metrics": {"mention_count": 0},
                "switch_reasons": [
                    {"direction": "switched_to", "mention_count": 1, "reason_category": "pricing"},
                    {"direction": "considering", "mention_count": 2, "reason_category": "features"},
                    {"direction": "compared", "mention_count": 1, "reason_category": "integration"},
                ],
                "evidence_breakdown": [
                    {
                        "evidence_type": "unknown",
                        "mention_count": 0,
                        "reason_categories": {"pricing": 1, "features": 2, "integration": 1},
                    }
                ],
            }
        ]
    })

    assert len(rows) == 1
    row = rows[0]
    assert row["vendor"] == "HubSpot"
    assert row["competitor"] == "ActiveCampaign"
    assert row["mention_count"] == 4
    assert row["explicit_switches"] == 1
    assert row["active_evaluations"] == 2
    assert row["implied_preferences"] == 1
    assert row["reason_categories"] == {
        "pricing": 1,
        "features": 2,
        "integration": 1,
    }
    assert row["evidence_breakdown"] == [
        {
            "evidence_type": "explicit_switch",
            "mention_count": 1,
            "reason_categories": {},
            "industries": [],
            "company_sizes": [],
        },
        {
            "evidence_type": "active_evaluation",
            "mention_count": 2,
            "reason_categories": {},
            "industries": [],
            "company_sizes": [],
        },
        {
            "evidence_type": "implied_preference",
            "mention_count": 1,
            "reason_categories": {},
            "industries": [],
            "company_sizes": [],
        },
    ]


def test_aggregate_competitive_disp_preserves_breakdown_for_dynamics_builder():
    aggregated = mod._aggregate_competitive_disp([
        {
            "vendor": "HubSpot",
            "competitor": "ActiveCampaign",
            "mention_count": 4,
            "explicit_switches": 1,
            "active_evaluations": 2,
            "implied_preferences": 1,
            "reason_categories": {"pricing": 1, "features": 2, "integration": 1},
            "industries": ["SaaS"],
            "company_sizes": ["SMB"],
            "evidence_breakdown": [
                {
                    "evidence_type": "explicit_switch",
                    "mention_count": 1,
                    "reason_categories": {"pricing": 1},
                    "industries": ["SaaS"],
                    "company_sizes": ["SMB"],
                },
                {
                    "evidence_type": "active_evaluation",
                    "mention_count": 2,
                    "reason_categories": {"features": 2},
                    "industries": ["SaaS"],
                    "company_sizes": ["SMB"],
                },
                {
                    "evidence_type": "implied_preference",
                    "mention_count": 1,
                    "reason_categories": {"integration": 1},
                    "industries": ["SaaS"],
                    "company_sizes": ["SMB"],
                },
            ],
        }
    ])

    assert len(aggregated) == 1
    dyn = mod.build_displacement_dynamics(
        from_vendor="HubSpot",
        to_vendor="ActiveCampaign",
        edge={"mention_count": 0},
        displacement_flows=aggregated,
        reasons=[],
        battle_conclusion=None,
        analysis_window_days=30,
    )

    assert dyn["flow_summary"] == {
        "explicit_switch_count": 1,
        "active_evaluation_count": 2,
        "total_flow_mentions": 4,
    }
    assert dyn["evidence_breakdown"] == [
        {
            "evidence_type": "active_evaluation",
            "mention_count": 2,
            "reason_categories": {"features": 2},
            "industries": ["SaaS"],
            "company_sizes": ["SMB"],
        },
        {
            "evidence_type": "explicit_switch",
            "mention_count": 1,
            "reason_categories": {"pricing": 1},
            "industries": ["SaaS"],
            "company_sizes": ["SMB"],
        },
        {
            "evidence_type": "implied_preference",
            "mention_count": 1,
            "reason_categories": {"integration": 1},
            "industries": ["SaaS"],
            "company_sizes": ["SMB"],
        },
    ]


def test_deterministic_displacement_map_filters_tool_targets():
    results = mod._build_deterministic_displacement_map(
        competitive_disp=[
            {
                "vendor": "Intercom",
                "competitor": "Custom ChatGPT Integration",
                "mention_count": 5,
                "explicit_switches": 0,
                "active_evaluations": 5,
                "implied_preferences": 0,
                "reason_categories": {"features": 5},
            },
            {
                "vendor": "ClickUp",
                "competitor": "Asana",
                "mention_count": 22,
                "explicit_switches": 0,
                "active_evaluations": 3,
                "implied_preferences": 19,
                "reason_categories": {"pricing": 1},
            },
        ],
        competitor_reasons=[],
        quote_lookup={},
    )

    assert results == [
        {
            "from_vendor": "ClickUp",
            "to_vendor": "Asana",
            "mention_count": 22,
            "primary_driver": "pricing",
            "signal_strength": "strong",
            "evidence_breakdown": {
                "explicit_switches": 0,
                "active_evaluations": 3,
                "implied_preferences": 19,
            },
            "key_quote": None,
            "industries": [],
            "company_sizes": [],
        }
    ]


def test_deterministic_displacement_map_normalizes_primary_driver_labels():
    results = mod._build_deterministic_displacement_map(
        competitive_disp=[
            {
                "vendor": "Intercom",
                "competitor": "SendSafely",
                "mention_count": 6,
                "explicit_switches": 0,
                "active_evaluations": 4,
                "implied_preferences": 2,
                "reason_categories": {
                    "files stored on their servers": 2,
                    "Close solution but pricey": 2,
                    "pricing": 2,
                },
            }
        ],
        competitor_reasons=[],
        quote_lookup={},
    )

    assert results == [
        {
            "from_vendor": "Intercom",
            "to_vendor": "SendSafely",
            "mention_count": 6,
            "primary_driver": "pricing",
            "signal_strength": "moderate",
            "evidence_breakdown": {
                "explicit_switches": 0,
                "active_evaluations": 4,
                "implied_preferences": 2,
            },
            "key_quote": None,
            "industries": [],
            "company_sizes": [],
        }
    ]


def test_deterministic_displacement_map_uses_synthesis_views_for_source_wedge():
    class _FakeWedge:
        def __init__(self, value):
            self.value = value

    class _FakeView:
        def __init__(self, wedge_value):
            self.primary_wedge = _FakeWedge(wedge_value)
            self.schema_version = "v2.3"

        def section(self, name):
            assert name == "causal_narrative"
            return {"primary_wedge": self.primary_wedge.value}

        def confidence(self, section):
            assert section == "causal_narrative"
            return "medium"

        def falsification_conditions(self):
            return []

        @property
        def confidence_limits(self):
            return []

    results = mod._build_deterministic_displacement_map(
        competitive_disp=[
            {
                "vendor": "Intercom",
                "competitor": "Asana",
                "mention_count": 3,
                "explicit_switches": 0,
                "active_evaluations": 1,
                "implied_preferences": 0,
                "reason_categories": {"pricing": 1},
            }
        ],
        competitor_reasons=[],
        quote_lookup={},
        synthesis_views={"Intercom": _FakeView("category_disruption")},
    )

    assert results[0]["signal_strength"] == "moderate"


@pytest.mark.asyncio
async def test_head_to_head_uses_dynamics_counts_and_companies():
    review_a = uuid4()
    review_b = uuid4()
    pool = FakePool(
        fetch_map={
            "FROM b2b_displacement_edges": [
                {
                    "from_vendor": "Zendesk",
                    "to_vendor": "Freshdesk",
                    "mention_count": 5,
                    "sample_review_ids": [review_a],
                },
                {
                    "from_vendor": "Freshdesk",
                    "to_vendor": "Zendesk",
                    "mention_count": 4,
                    "sample_review_ids": [review_b],
                },
            ],
            "FROM b2b_displacement_dynamics": [
                {
                    "from_vendor": "Zendesk",
                    "to_vendor": "Freshdesk",
                    "dynamics": {"edge_metrics": {"mention_count": 20}},
                },
                {
                    "from_vendor": "Freshdesk",
                    "to_vendor": "Zendesk",
                    "dynamics": {"edge_metrics": {"mention_count": 12}},
                },
            ],
            "FROM b2b_company_signals": [
                {"vendor_name": "Zendesk", "review_id": review_a, "company_name": "Acme Corp"},
                {"vendor_name": "Freshdesk", "review_id": review_b, "company_name": "Beta LLC"},
            ],
            "FROM b2b_reviews": [],
        },
    )

    flows = await mod._head_to_head_from_edges(pool, "Zendesk", "Freshdesk", 30)

    assert flows == [
        {"name": "Zendesk -> Freshdesk", "count": 20, "companies": ["Acme Corp"]},
        {"name": "Freshdesk -> Zendesk", "count": 12, "companies": ["Beta LLC"]},
    ]


@pytest.mark.asyncio
async def test_head_to_head_falls_back_to_reviewer_company_when_signal_missing(monkeypatch):
    review_a = uuid4()
    monkeypatch.setattr(mod, "_company_signal_name_is_eligible", lambda *args, **kwargs: True)
    pool = FakePool(
        fetch_map={
            "FROM b2b_displacement_edges": [
                {
                    "from_vendor": "Zendesk",
                    "to_vendor": "Freshdesk",
                    "mention_count": 5,
                    "sample_review_ids": [review_a],
                },
                {
                    "from_vendor": "Freshdesk",
                    "to_vendor": "Zendesk",
                    "mention_count": 4,
                    "sample_review_ids": [],
                },
            ],
            "FROM b2b_displacement_dynamics": [
                {
                    "from_vendor": "Zendesk",
                    "to_vendor": "Freshdesk",
                    "dynamics": {"edge_metrics": {"mention_count": 20}},
                },
                {
                    "from_vendor": "Freshdesk",
                    "to_vendor": "Zendesk",
                    "dynamics": {"edge_metrics": {"mention_count": 12}},
                },
            ],
            "FROM b2b_company_signals": [],
            "FROM b2b_reviews": [
                {"vendor_name": "Zendesk", "review_id": review_a, "reviewer_company": "Gamma Inc"},
            ],
        },
    )

    flows = await mod._head_to_head_from_edges(pool, "Zendesk", "Freshdesk", 30)

    assert flows == [
        {"name": "Zendesk -> Freshdesk", "count": 20, "companies": ["Gamma Inc"]},
        {"name": "Freshdesk -> Zendesk", "count": 12, "companies": []},
    ]


@pytest.mark.asyncio
async def test_switching_triggers_merge_duplicate_competitor_labels(monkeypatch):
    def _canon(raw):
        raw = str(raw or "").strip().lower()
        if raw == "custom chatgpt integration":
            return "Custom ChatGPT Integration"
        if raw == "freshdesk":
            return "Freshdesk"
        return raw.title()

    monkeypatch.setattr(mod, "_canonicalize_competitor", _canon)
    monkeypatch.setattr(
        mod,
        "_battle_card_competitor_is_eligible",
        lambda label: str(label or "").strip().lower() != "custom chatgpt integration",
    )
    pool = FakePool(
        fetch_map={
            "FROM b2b_displacement_dynamics": [
                {
                    "to_vendor": "Custom ChatGPT integration",
                    "dynamics": {
                        "edge_metrics": {"mention_count": 8, "primary_driver": "features"},
                        "switch_reasons": [{"reason_category": "features", "mention_count": 8}],
                    },
                },
                {
                    "to_vendor": "Custom ChatGPT Integration",
                    "dynamics": {
                        "edge_metrics": {"mention_count": 5, "primary_driver": "features"},
                        "switch_reasons": [{"reason_category": "features", "mention_count": 5}],
                    },
                },
                {
                    "to_vendor": "Freshdesk",
                    "dynamics": {
                        "edge_metrics": {"mention_count": 7, "primary_driver": "pricing"},
                        "switch_reasons": [{"reason_category": "pricing", "mention_count": 4}],
                    },
                },
                {
                    "to_vendor": "Homegrown Support Utility",
                    "dynamics": {
                        "edge_metrics": {"mention_count": 9, "primary_driver": "reliability"},
                        "switch_reasons": [{"reason_category": "reliability", "mention_count": 3}],
                    },
                },
            ],
        },
    )

    triggers = await mod._switching_triggers_from_dynamics(pool, "Zendesk", 30)

    assert triggers == [
        {
            "competitor": "Freshdesk",
            "primary_reason": "pricing",
            "mention_count": 4,
            "total_mentions": 7,
        }
    ]


@pytest.mark.asyncio
async def test_company_snapshot_uses_review_context_for_company_specific_fields(monkeypatch):
    monkeypatch.setattr(mod, "_canonicalize_competitor", lambda raw: str(raw or "").strip())
    review_a = uuid4()
    review_b = uuid4()
    pool = FakePool(
        fetch_map={
            "FROM b2b_company_signals": [
                {
                    "vendor_name": "Zendesk",
                    "urgency_score": 8.0,
                    "pain_category": "pricing",
                    "buyer_role": "VP Ops",
                    "decision_maker": True,
                    "seat_count": 220,
                    "contract_end": "2026-05-01",
                    "buying_stage": "evaluation",
                    "source": "g2",
                    "review_id": review_a,
                },
                {
                    "vendor_name": "Zendesk",
                    "urgency_score": 7.0,
                    "pain_category": "ux",
                    "buyer_role": "Support Lead",
                    "decision_maker": False,
                    "seat_count": None,
                    "contract_end": None,
                    "buying_stage": "consideration",
                    "source": "g2",
                    "review_id": review_b,
                },
            ],
            "FROM b2b_reviews": [
                {
                    "id": review_a,
                    "vendor_name": "Zendesk",
                    "product_category": "Support",
                    "competitors_json": [{"name": "Freshdesk"}],
                    "quotable_phrases": [{"text": "Pricing became hard to justify"}],
                    "feature_gaps": [{"feature": "Missing SLA reporting"}],
                    "evaluation_deadline": "2026-04-15",
                    "decision_timeline": "within_quarter",
                    "contract_value_signal": "enterprise_high",
                    "reviewer_title": "VP Operations",
                    "company_size_raw": "Mid-Market",
                    "industry": "SaaS",
                },
                {
                    "id": review_b,
                    "vendor_name": "Zendesk",
                    "product_category": "Support",
                    "competitors_json": [{"name": "Intercom"}],
                    "quotable_phrases": [{"text": "The interface slowed the team down"}],
                    "feature_gaps": [{"feature": "Weak automation"}],
                    "evaluation_deadline": None,
                    "decision_timeline": "immediate",
                    "contract_value_signal": "mid_market",
                    "reviewer_title": "Support Manager",
                    "company_size_raw": "Mid-Market",
                    "industry": "Fintech",
                },
            ],
            "FROM b2b_product_profiles": [],
            "FROM b2b_evidence_vault": [],
        },
    )

    snapshot = await mod._company_snapshot_from_signals(pool, "Acme Corp", 30)

    assert snapshot is not None
    assert snapshot["industries"] == [
        {"industry": "SaaS", "count": 1},
        {"industry": "Fintech", "count": 1},
    ]
    assert snapshot["contract_value_signals"] == ["enterprise_high", "mid_market"]
    assert snapshot["alternatives_considered"] == [
        {"name": "Freshdesk", "count": 1},
        {"name": "Intercom", "count": 1},
    ]
    assert snapshot["quote_highlights"] == [
        "Pricing became hard to justify",
        "The interface slowed the team down",
    ]
    assert snapshot["timeline_signals"][0]["evaluation_deadline"] == "2026-04-15"
    assert snapshot["timeline_signals"][0]["title"] == "VP Operations"
