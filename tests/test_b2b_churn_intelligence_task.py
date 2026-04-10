import json
import datetime as datetime_module
from types import SimpleNamespace
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

from atlas_brain.autonomous.tasks import b2b_churn_intelligence as mod


class CapturePool:
    def __init__(self):
        self.execute_calls = []

    async def execute(self, query, *params):
        self.execute_calls.append((str(query), params))


class RunCapturePool(CapturePool):
    def __init__(self):
        super().__init__()
        self.executemany_calls = []
        self.is_initialized = True

    async def executemany(self, query, rows):
        self.executemany_calls.append((str(query), [tuple(row) for row in rows]))

    async def fetch(self, *args, **kwargs):
        return []

    async def fetchrow(self, *args, **kwargs):
        return None

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


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

    result = await mod._upsert_churn_signals(
        pool,
        vendor_scores=[
            {
                "vendor_name": "Zendesk",
                "product_category": "Helpdesk",
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

    assert result.phase_name == "churn_signals"
    assert result.failed == 0
    assert result.succeeded == 1
    assert len(pool.execute_calls) == 1
    query, params = pool.execute_calls[0]
    assert "materialization_run_id" in query
    assert "product_category = EXCLUDED.product_category" in query
    assert params[1] == "Helpdesk"
    assert params[-2] == "run-123"


def test_build_materialization_status_tracks_failed_and_partial_phases():
    status = mod._build_materialization_status(
        materialization_run_id="run-123",
        run_date=mod.date(2026, 4, 10),
        phase_results=[
            mod._complete_phase_result("evidence_vault", attempted=2, succeeded=2),
            mod._MaterializationPhaseResult(
                phase_name="churn_signals",
                attempted=3,
                succeeded=2,
                failed=1,
                state="incomplete",
                failed_targets=["Freshdesk"],
                reason="vendor_upsert_failure",
            ),
            mod._failed_phase_result(
                "account_intelligence",
                attempted=2,
                failed_targets=["Zendesk", "Freshdesk"],
                reason="persistence_exception",
            ),
            mod._skipped_phase_result("category_dynamics", reason="scoped_run"),
        ],
    )

    assert status["is_complete"] is False
    assert status["partial_phases"] == ["churn_signals"]
    assert status["failed_phases"] == ["account_intelligence"]
    assert status["skipped_phases"] == ["category_dynamics"]
    assert status["phases"]["churn_signals"]["failed_targets"] == ["Freshdesk"]


@pytest.mark.asyncio
async def test_write_core_run_marker_marks_incomplete_runs():
    pool = CapturePool()
    materialization_status = {
        "is_complete": False,
        "failed_phases": ["account_intelligence"],
        "partial_phases": ["churn_signals"],
        "phases": {},
    }

    await mod._write_core_run_marker(
        pool,
        report_date=mod.date(2026, 4, 10),
        analyzed_vendor_count=7,
        upsert_failures=1,
        elapsed_seconds=12.4,
        materialization_status=materialization_status,
    )

    assert len(pool.execute_calls) == 1
    query, params = pool.execute_calls[0]
    assert "status = EXCLUDED.status" in query
    assert params[0] == mod.date(2026, 4, 10)
    payload = json.loads(params[1])
    assert payload["materialization_complete"] is False
    assert payload["materialization_status"]["failed_phases"] == ["account_intelligence"]
    assert params[2] == "incomplete"


@pytest.mark.asyncio
async def test_run_uses_captured_run_date_for_all_persistence(monkeypatch):
    real_date = datetime_module.date

    class FakeDate(real_date):
        calls = 0

        @classmethod
        def today(cls):
            cls.calls += 1
            if cls.calls == 1:
                return real_date(2026, 4, 10)
            return real_date(2026, 4, 11)

    async def fake_gather(*coroutines, **_kwargs):
        for coro in coroutines:
            close = getattr(coro, "close", None)
            if close:
                close()
        return (
            [{"vendor_name": "Zendesk", "product_category": "Helpdesk", "total_reviews": 12, "churn_intent": 3, "avg_urgency": 7.5, "signal_reviews": 5}],
            [],
            {},
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    pool = RunCapturePool()
    monkeypatch.setattr(mod.settings.b2b_churn, "enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_churn, "snapshot_enabled", False, raising=False)
    monkeypatch.setattr(mod.settings.b2b_churn, "change_detection_enabled", False, raising=False)
    monkeypatch.setattr(mod.settings.b2b_churn, "temporal_analysis_vendor_limit", 0, raising=False)
    monkeypatch.setattr(mod, "date", FakeDate)
    monkeypatch.setattr(datetime_module, "date", FakeDate)
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_warm_vendor_cache", AsyncMock())
    monkeypatch.setattr(mod, "_sync_vendor_firmographics", AsyncMock(return_value=0))
    monkeypatch.setattr(mod, "_update_execution_progress", AsyncMock())
    monkeypatch.setattr(mod, "_send_notification", AsyncMock())
    monkeypatch.setattr(mod, "_emit_reasoning_events", AsyncMock())
    monkeypatch.setattr(mod, "_fetch_prior_reports", AsyncMock(return_value=[]))
    monkeypatch.setattr(mod.asyncio, "gather", fake_gather)
    monkeypatch.setattr(mod, "build_evidence_vault", lambda **_kwargs: {"schema_version": "v1"})
    monkeypatch.setattr(mod, "build_segment_intelligence", lambda **_kwargs: {"schema_version": "v1"})
    monkeypatch.setattr(mod, "build_temporal_intelligence", lambda **_kwargs: {"schema_version": "v1"})
    monkeypatch.setattr(mod, "build_account_intelligence", lambda **_kwargs: {"schema_version": "v1"})
    monkeypatch.setattr(mod, "build_category_dynamics", lambda **_kwargs: {"schema_version": "v1"})
    monkeypatch.setattr(mod, "_build_exploratory_payload", lambda *_args, **_kwargs: ({}, 0))
    monkeypatch.setattr(mod, "_build_deterministic_vendor_feed", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, "_build_validated_executive_summary", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("atlas_brain.pipelines.llm.get_pipeline_llm", lambda workload=None: None)

    result = await mod.run(SimpleNamespace(id=uuid4(), metadata={}))

    assert result["date"] == "2026-04-10"
    persisted_dates = []
    for query, rows in pool.executemany_calls:
        if "INSERT INTO b2b_evidence_vault" in query:
            persisted_dates.extend(row[1] for row in rows)
        elif "INSERT INTO b2b_segment_intelligence" in query:
            persisted_dates.extend(row[1] for row in rows)
        elif "INSERT INTO b2b_temporal_intelligence" in query:
            persisted_dates.extend(row[1] for row in rows)
        elif "INSERT INTO b2b_category_dynamics" in query:
            persisted_dates.extend(row[1] for row in rows)
        elif "INSERT INTO b2b_account_intelligence" in query:
            persisted_dates.extend(row[1] for row in rows)
    completion_dates = [
        params[0]
        for query, params in pool.execute_calls
        if "INSERT INTO b2b_intelligence" in query
    ]

    assert persisted_dates
    assert all(value == real_date(2026, 4, 10) for value in persisted_dates)
    assert completion_dates == [real_date(2026, 4, 10)]
