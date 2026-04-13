import inspect
from types import SimpleNamespace
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

from atlas_brain.autonomous.tasks import b2b_churn_intelligence as mod


class CapturePool:
    def __init__(self):
        self.execute_calls = []
        self.executemany_calls = []

    async def execute(self, query, *params):
        self.execute_calls.append((str(query), params))

    async def executemany(self, query, rows):
        self.executemany_calls.append((str(query), list(rows)))


@pytest.mark.asyncio
async def test_generate_vendor_report_reads_vendor_mentions():
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    result = await mod.generate_vendor_report(pool, "Zendesk", window_days=90)

    assert result is None
    sql, window_days, vendor_name, sources = pool.fetch.await_args.args
    assert window_days == 90
    assert vendor_name == "Zendesk"
    assert isinstance(sources, list)
    assert "JOIN LATERAL" in sql
    assert "FROM b2b_review_vendor_mentions vm" in sql
    assert "vm.vendor_name ILIKE '%' || $2 || '%'" in sql


@pytest.mark.asyncio
async def test_generate_challenger_report_reads_vendor_mentions():
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]))

    result = await mod.generate_challenger_report(pool, "HubSpot", window_days=90)

    assert result is None
    sql, window_days, challenger_name, sources = pool.fetch.await_args.args
    assert window_days == 90
    assert challenger_name == "HubSpot"
    assert isinstance(sources, list)
    assert "JOIN LATERAL" in sql
    assert "FROM b2b_review_vendor_mentions vm" in sql
    assert "matched_vm.vendor_name AS vendor_name" in sql


@pytest.mark.asyncio
async def test_run_skips_when_b2b_churn_intelligence_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_churn, "enabled", False)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_enabled", False)

    task = SimpleNamespace(id=uuid4(), metadata={})

    result = await mod.run(task)

    assert result == {"_skip_synthesis": "B2B churn intelligence disabled"}


@pytest.mark.asyncio
async def test_run_allows_maintenance_override_when_disabled(monkeypatch):
    monkeypatch.setattr(mod.settings.b2b_churn, "enabled", False)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_enabled", False)

    class DummyPool:
        is_initialized = False

    monkeypatch.setattr(mod, "get_db_pool", lambda: DummyPool())

    task = SimpleNamespace(id=uuid4(), metadata={"maintenance_run": True})

    result = await mod.run(task)

    assert result == {"_skip_synthesis": "DB not ready"}


@pytest.mark.asyncio
async def test_load_category_council_lookup_uses_synthesis_first_loader(monkeypatch):
    fake_lookup = {
        "councils": {
            "CRM": {
                "confidence": 0.82,
                "conclusion": {
                    "winner": "HubSpot",
                    "loser": "Salesforce",
                    "conclusion": "HubSpot is gaining share in CRM.",
                    "market_regime": "fragmenting",
                    "durability_assessment": "structural",
                    "key_insights": ["signal-1", "signal-2", "signal-3", "signal-4"],
                },
            },
        },
    }
    loader = AsyncMock(return_value=fake_lookup)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.load_best_cross_vendor_lookup",
        loader,
    )

    result = await mod._load_category_council_lookup(
        object(),
        [{"vendor_name": "Salesforce", "product_category": "CRM"}],
        as_of=mod.date(2026, 4, 7),
    )

    loader.assert_awaited_once()
    assert result[("Salesforce", "crm")]["winner"] == "HubSpot"
    assert result[("Salesforce", "crm")]["confidence"] == 0.82
    assert result[("Salesforce", "crm")]["key_insights"] == ["signal-1", "signal-2", "signal-3"]


@pytest.mark.asyncio
async def test_load_category_council_lookup_skips_generic_categories(monkeypatch):
    loader = AsyncMock(return_value={"councils": {"B2B Software": {"conclusion": {"winner": "A"}}}})
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_cross_vendor_synthesis.load_best_cross_vendor_lookup",
        loader,
    )

    result = await mod._load_category_council_lookup(
        object(),
        [{"vendor_name": "Salesforce", "product_category": "B2B Software"}],
        as_of=mod.date(2026, 4, 7),
    )

    assert result == {}


def test_task_run_id_prefers_scheduler_execution_id():
    execution_id = str(uuid4())
    task = SimpleNamespace(id=uuid4(), metadata={"_execution_id": execution_id, "run_id": "legacy-run"})

    assert mod._task_run_id(task) == execution_id


@pytest.mark.asyncio
async def test_upsert_churn_signals_persists_materialization_run_id():
    pool = CapturePool()

    failures = await mod._upsert_churn_signals(
        pool,
        vendor_scores=[
            {
                "vendor_name": "Zendesk",
                "total_reviews": 12,
                "churn_intent": 3,
                "avg_urgency": 7.5,
                "signal_reviews": 5,
            },
        ],
        neg_lookup={},
        pain_lookup={},
        competitor_lookup={},
        feature_gap_lookup={},
        price_lookup={},
        dm_lookup={},
        company_lookup={},
        quote_lookup={},
        materialization_run_id="run-123",
    )

    assert failures == 0
    assert len(pool.execute_calls) == 1
    query, params = pool.execute_calls[0]
    assert "materialization_run_id" in query
    assert params[-2] == "run-123"


@pytest.mark.asyncio
async def test_upsert_company_signal_candidates_persists_materialization_run_id():
    pool = CapturePool()

    persisted = await mod._upsert_company_signal_candidates(
        pool,
        candidates=[
            {
                "review_id": str(uuid4()),
                "company_name": "Acme Corp",
                "company_name_raw": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "source": "reddit",
                "reviewed_at": "2026-04-10T12:00:00+00:00",
                "urgency_score": 6.8,
                "relevance_score": 0.71,
                "pain_category": "pricing",
                "role_level": "vp",
                "decision_maker": True,
                "seat_count": 120,
                "contract_end": "2026-07-01",
                "buying_stage": "evaluation",
                "resolution_confidence": "medium",
                "confidence_score": 0.26,
                "confidence_tier": "low",
                "signal_evidence_present": False,
                "canonical_gap_reason": "low_confidence_low_trust_source",
                "candidate_bucket": "analyst_review",
            },
        ],
        materialization_run_id="run-456",
    )

    assert persisted == 1
    assert len(pool.executemany_calls) == 1
    query, rows = pool.executemany_calls[0]
    assert "materialization_run_id" in query
    assert "review_status =" not in query
    assert "review_status_updated_at =" not in query
    assert rows[0][-1] == "run-456"
    assert rows[0][-2] == "analyst_review"


def test_build_company_signal_candidate_groups_corroborates_low_trust_cluster():
    review_ids = [str(uuid4()) for _ in range(4)]

    groups = mod._build_company_signal_candidate_groups(
        [
            {
                "review_id": review_ids[0],
                "company_name": "Acme Corp",
                "company_name_raw": "Acme Corp",
                "vendor_name": "Zendesk",
                "source": "reddit",
                "urgency_score": 8.1,
                "confidence_score": 0.26,
                "signal_evidence_present": True,
                "decision_maker": True,
                "candidate_bucket": "analyst_review",
                "canonical_gap_reason": "low_confidence_low_trust_source",
                "role_level": "vp",
                "pain_category": "pricing",
            },
            {
                "review_id": review_ids[1],
                "company_name": "Acme Corp",
                "company_name_raw": "Acme Corp",
                "vendor_name": "Zendesk",
                "source": "reddit",
                "urgency_score": 7.9,
                "confidence_score": 0.26,
                "signal_evidence_present": True,
                "decision_maker": False,
                "candidate_bucket": "analyst_review",
                "canonical_gap_reason": "low_confidence_low_trust_source",
                "role_level": "director",
                "pain_category": "pricing",
            },
            {
                "review_id": review_ids[2],
                "company_name": "Acme Corp",
                "company_name_raw": "Acme Corp",
                "vendor_name": "Zendesk",
                "source": "reddit",
                "urgency_score": 7.2,
                "confidence_score": 0.26,
                "signal_evidence_present": False,
                "decision_maker": False,
                "candidate_bucket": "analyst_review",
                "canonical_gap_reason": "low_confidence_low_trust_source",
                "role_level": "manager",
                "pain_category": "pricing",
            },
            {
                "review_id": review_ids[3],
                "company_name": "Acme Corp",
                "company_name_raw": "Acme Corp",
                "vendor_name": "Zendesk",
                "source": "reddit",
                "urgency_score": 7.0,
                "confidence_score": 0.26,
                "signal_evidence_present": False,
                "decision_maker": False,
                "candidate_bucket": "analyst_review",
                "canonical_gap_reason": "low_confidence_low_trust_source",
                "role_level": "manager",
                "pain_category": "pricing",
            },
        ],
    )

    assert len(groups) == 1
    group = groups[0]
    assert group["company_name"] == "acme"
    assert group["review_count"] == 4
    assert group["candidate_bucket"] == "canonical_ready"
    assert group["canonical_gap_reason"] is None
    assert group["corroborated_confidence_score"] >= 0.6


@pytest.mark.asyncio
async def test_upsert_company_signal_candidate_groups_persists_materialization_run_id():
    pool = CapturePool()

    persisted = await mod._upsert_company_signal_candidate_groups(
        pool,
        groups=[
            {
                "company_name": "acme",
                "display_company_name": "Acme Corp",
                "vendor_name": "Zendesk",
                "product_category": "Customer Support",
                "review_count": 4,
                "distinct_source_count": 1,
                "decision_maker_count": 1,
                "signal_evidence_count": 2,
                "canonical_ready_review_count": 0,
                "avg_urgency_score": 7.55,
                "max_urgency_score": 8.1,
                "avg_confidence_score": 0.26,
                "max_confidence_score": 0.26,
                "corroborated_confidence_score": 0.61,
                "confidence_tier": "high",
                "source_distribution": {"reddit": 4},
                "gap_reason_distribution": {"low_confidence_low_trust_source": 4},
                "sample_review_ids": [str(uuid4())],
                "representative_review_id": str(uuid4()),
                "representative_source": "reddit",
                "representative_pain_category": "pricing",
                "representative_buyer_role": "vp",
                "representative_decision_maker": True,
                "representative_seat_count": 120,
                "representative_contract_end": "2026-07-01",
                "representative_buying_stage": "evaluation",
                "representative_confidence_score": 0.26,
                "representative_urgency_score": 8.1,
                "canonical_gap_reason": None,
                "candidate_bucket": "canonical_ready",
            },
        ],
        materialization_run_id="run-789",
    )

    assert persisted == 1
    assert len(pool.executemany_calls) == 1
    query, rows = pool.executemany_calls[0]
    assert "b2b_company_signal_candidate_groups" in query
    assert "materialization_run_id" in query
    assert rows[0][-1] == "run-789"
    assert rows[0][-2] == "canonical_ready"


@pytest.mark.asyncio
async def test_rebuild_company_signal_candidate_materializations_returns_counts(monkeypatch):
    pool = CapturePool()
    fetch_mock = AsyncMock(
        return_value=[
            {
                "review_id": str(uuid4()),
                "company_name": "Acme Corp",
                "vendor_name": "Zendesk",
                "candidate_bucket": "canonical_ready",
            },
            {
                "review_id": str(uuid4()),
                "company_name": "Acme Corp",
                "vendor_name": "Zendesk",
                "candidate_bucket": "analyst_review",
            },
        ]
    )
    upsert_candidates_mock = AsyncMock(return_value=2)
    upsert_groups_mock = AsyncMock(return_value=1)
    sync_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(mod, "_fetch_company_signal_review_candidates", fetch_mock)
    monkeypatch.setattr(
        mod,
        "_build_company_signal_candidate_groups",
        lambda candidates: [
            {
                "company_name": "acme",
                "vendor_name": "Zendesk",
                "candidate_bucket": "canonical_ready",
            },
        ],
    )
    monkeypatch.setattr(mod, "_upsert_company_signal_candidates", upsert_candidates_mock)
    monkeypatch.setattr(mod, "_upsert_company_signal_candidate_groups", upsert_groups_mock)
    monkeypatch.setattr(mod, "_sync_company_signal_candidate_member_status_from_groups", sync_mock)

    result = await mod.rebuild_company_signal_candidate_materializations(
        pool,
        window_days=90,
        vendors=["Zendesk"],
        materialization_run_id="run-backfill",
    )

    assert result == {
        "company_signal_candidates": 2,
        "canonical_ready_company_signal_candidates": 1,
        "company_signal_candidates_persisted": 2,
        "company_signal_candidate_groups": 1,
        "canonical_ready_company_signal_candidate_groups": 1,
        "company_signal_candidate_groups_persisted": 1,
    }
    fetch_mock.assert_awaited_once()
    upsert_candidates_mock.assert_awaited_once()
    upsert_groups_mock.assert_awaited_once()
    sync_mock.assert_awaited_once_with(pool, materialization_run_id="run-backfill")


def test_run_uses_scoped_vendors_for_company_signal_candidate_materializations():
    source = inspect.getsource(mod.run)

    assert "vendors=scoped_vendors or None" in source
