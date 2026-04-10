import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.auth.dependencies import AuthUser
from atlas_brain.services.b2b_competitive_sets import (
    build_competitive_set_plan,
    estimate_competitive_set_plan,
    plan_to_synthesis_metadata,
)
from atlas_brain.storage.models import CompetitiveSet


def _competitive_set(**overrides) -> CompetitiveSet:
    payload = {
        "id": uuid4(),
        "account_id": uuid4(),
        "name": "Salesforce Core Competitors",
        "focal_vendor_name": "Salesforce",
        "competitor_vendor_names": ["HubSpot", "Microsoft Dynamics", "HubSpot"],
        "active": True,
        "refresh_mode": "scheduled",
        "refresh_interval_hours": 24,
        "vendor_synthesis_enabled": True,
        "pairwise_enabled": True,
        "category_council_enabled": True,
        "asymmetry_enabled": True,
    }
    payload.update(overrides)
    return CompetitiveSet(**payload)


def test_build_competitive_set_plan_scopes_to_focal_and_named_competitors():
    plan = build_competitive_set_plan(
        _competitive_set(),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    assert plan.vendor_names == ["Salesforce", "HubSpot", "Microsoft Dynamics"]
    assert plan.pairwise_pairs == [
        ["Salesforce", "HubSpot"],
        ["Salesforce", "Microsoft Dynamics"],
    ]
    assert plan.category_names == ["CRM"]
    assert plan.asymmetry_pairs == [
        ["Salesforce", "HubSpot"],
        ["Salesforce", "Microsoft Dynamics"],
    ]


def test_build_competitive_set_plan_disables_optional_jobs_when_toggles_off():
    plan = build_competitive_set_plan(
        _competitive_set(
            pairwise_enabled=False,
            category_council_enabled=False,
            asymmetry_enabled=False,
        ),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    assert plan.pairwise_pairs == []
    assert plan.category_names == []
    assert plan.asymmetry_pairs == []


def test_plan_to_synthesis_metadata_emits_explicit_scope_contract():
    plan = build_competitive_set_plan(
        _competitive_set(),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    metadata = plan_to_synthesis_metadata(plan)

    assert metadata["scope_type"] == "competitive_set"
    assert metadata["scope_id"] == str(plan.competitive_set_id)
    assert metadata["scope_vendor_names"] == ["Salesforce", "HubSpot", "Microsoft Dynamics"]
    assert metadata["scope_pairwise_pairs"] == [
        ["Salesforce", "HubSpot"],
        ["Salesforce", "Microsoft Dynamics"],
    ]
    assert metadata["scope_category_names"] == ["CRM"]


@pytest.mark.asyncio
async def test_load_vendor_category_map_prefers_profiles_before_scorecard_fallback(monkeypatch):
    from atlas_brain.services import b2b_competitive_sets as mod

    calls: list[tuple[str, tuple[Any, ...]]] = []

    class FakePool:
        async def fetch(self, query, *args):
            normalized = " ".join(str(query).split())
            calls.append((normalized, args))
            if "FROM requested r LEFT JOIN b2b_product_profiles p" not in normalized:
                raise AssertionError(f"Unexpected query: {normalized}")
            return [
                {"vendor_name": "Salesforce", "product_category": "CRM"},
                {"vendor_name": "HubSpot", "product_category": None},
            ]

    adapter = AsyncMock(return_value=[
        {"vendor_name": "Salesforce", "product_category": "B2B Software"},
        {"vendor_name": "HubSpot", "product_category": "CRM"},
    ])
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.read_vendor_scorecard_details",
        adapter,
    )

    result = await mod.load_vendor_category_map(FakePool(), ["Salesforce", "HubSpot"])

    assert adapter.await_count == 1
    assert adapter.await_args.kwargs == {"vendor_names": ["Salesforce", "HubSpot"]}
    assert len(calls) == 1
    assert result == {"salesforce": "CRM", "hubspot": "CRM"}


@pytest.mark.asyncio
async def test_create_competitive_set_trims_name_before_duplicate_lookup(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    repo = SimpleNamespace(
        get_by_name_for_account=AsyncMock(return_value=_competitive_set(name="Sales Ops")),
    )
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())

    req = mod.CompetitiveSetRequest(
        name="  Sales Ops  ",
        focal_vendor_name="Salesforce",
        competitor_vendor_names=["HubSpot"],
    )

    with pytest.raises(mod.HTTPException) as exc:
        await mod.create_competitive_set(req, user=user)

    assert exc.value.status_code == 409
    assert repo.get_by_name_for_account.await_args.args[1] == "Sales Ops"


@pytest.mark.asyncio
async def test_list_competitive_sets_returns_backend_defaults(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    repo = SimpleNamespace(
        list_for_account=AsyncMock(return_value=[_competitive_set()]),
    )
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(uuid4()),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "competitive_set_refresh_interval_seconds",
        7200,
        raising=False,
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "competitive_set_max_competitors",
        12,
        raising=False,
    )

    result = await mod.list_competitive_sets(include_inactive=False, user=user)

    assert result["count"] == 1
    assert result["defaults"] == {
        "default_refresh_interval_hours": 2,
        "max_competitors": 12,
        "default_changed_vendors_only": True,
    }


def test_competitive_scope_run_id_prefers_explicit_scope_run_id():
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    task = SimpleNamespace(
        metadata={
            "_execution_id": str(uuid4()),
            "run_id": "scope-run-123",
            "scope_type": "competitive_set",
            "scope_id": str(uuid4()),
        }
    )

    scope_meta = mod._competitive_scope_metadata(task)

    assert mod._competitive_scope_run_id(task, scope_meta) == "scope-run-123"


def test_competitive_scope_run_id_prefers_scheduler_execution_id_when_present():
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    execution_id = str(uuid4())
    task = SimpleNamespace(
        id=uuid4(),
        metadata={
            "_execution_id": execution_id,
            "scope_type": "competitive_set",
            "scope_id": str(uuid4()),
        },
    )

    scope_meta = mod._competitive_scope_metadata(task)

    assert mod._competitive_scope_run_id(task, scope_meta) == execution_id


def test_competitive_scope_run_id_generates_unique_id_for_direct_scoped_runs():
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    task = SimpleNamespace(
        id=uuid4(),
        metadata={
            "scope_type": "competitive_set",
            "scope_id": str(uuid4()),
        },
    )

    scope_meta = mod._competitive_scope_metadata(task)
    run_id_a = mod._competitive_scope_run_id(task, scope_meta)
    run_id_b = mod._competitive_scope_run_id(task, scope_meta)

    assert run_id_a is not None
    assert run_id_b is not None
    assert run_id_a != str(task.id)
    assert run_id_b != str(task.id)
    assert run_id_a != run_id_b


def test_competitive_set_preview_uses_canonical_v2_schema_predicate():
    from atlas_brain.services import b2b_competitive_sets as mod

    predicate = mod._reasoning_v2_schema_predicate("schema_version")

    assert "IN ('v2', '2')" in predicate
    assert "LIKE 'v2.%'" in predicate
    assert "LIKE '2.%'" in predicate


def test_scheduled_scope_strategy_prefers_runtime_config(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    task = SimpleNamespace(metadata={"scheduled_scope_strategy": "competitive_sets"})

    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "reasoning_synthesis_scheduled_scope_strategy",
        "full_universe",
        raising=False,
    )

    assert mod._scheduled_scope_strategy(task) == "full_universe"


def test_scheduled_scope_strategy_falls_back_to_task_metadata(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    task = SimpleNamespace(metadata={"scheduled_scope_strategy": "competitive_sets"})

    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "reasoning_synthesis_scheduled_scope_strategy",
        "",
        raising=False,
    )

    assert mod._scheduled_scope_strategy(task) == "competitive_sets"


@pytest.mark.asyncio
async def test_estimate_competitive_set_plan_uses_history_and_fallback(monkeypatch):
    plan = build_competitive_set_plan(
        _competitive_set(),
        category_by_vendor={
            "salesforce": "CRM",
            "hubspot": "CRM",
            "microsoft dynamics": "CRM",
        },
    )

    class FakePool:
        async def fetch(self, query, *args):
            if "FROM b2b_reasoning_synthesis" in query:
                return [
                    {"vendor_name": "Salesforce", "tokens_used": 1000},
                    {"vendor_name": "HubSpot", "tokens_used": 1200},
                ]
            if "FROM llm_usage" in query:
                return [
                    {
                        "span_name": "task.b2b_reasoning_synthesis",
                        "avg_total_tokens": 900.0,
                        "avg_cost_usd": 0.09,
                        "sample_count": 20,
                    },
                    {
                        "span_name": "task.b2b_reasoning_synthesis.cross_vendor",
                        "avg_total_tokens": 200.0,
                        "avg_cost_usd": 0.02,
                        "sample_count": 10,
                    },
                ]
            if "FROM b2b_cross_vendor_reasoning_synthesis" in query:
                return [
                    {"analysis_type": "pairwise_battle", "avg_tokens_used": 250.0, "sample_count": 8},
                    {"analysis_type": "category_council", "avg_tokens_used": 400.0, "sample_count": 3},
                ]
            raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(
        "atlas_brain.services.b2b_competitive_sets.settings.b2b_churn.competitive_set_preview_lookback_days",
        14,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b_competitive_sets._estimate_vendor_reuse_for_plan",
        AsyncMock(return_value={
            "vendor_jobs_with_matching_pools": 3,
            "vendor_jobs_missing_pools": 0,
            "vendor_jobs_likely_to_reason": 1,
            "vendor_jobs_likely_hash_reuse": 1,
            "vendor_jobs_likely_stale_reuse": 1,
            "vendor_jobs_likely_missing_prior": 0,
            "vendor_jobs_likely_hash_changed": 1,
            "vendor_jobs_likely_prior_quality_weak": 0,
            "vendor_jobs_likely_missing_packet_artifacts": 0,
            "vendor_jobs_likely_missing_reference_ids": 0,
            "likely_rerun_vendors": ["Microsoft Dynamics:hash_changed"],
            "likely_reuse_vendors": ["Salesforce:hash_reuse", "HubSpot:stale_reused"],
        }),
        raising=False,
    )

    estimate = await estimate_competitive_set_plan(FakePool(), plan)

    assert estimate["estimated_vendor_tokens"] == 3100
    assert estimate["estimated_cross_vendor_tokens"] == 1300
    assert estimate["estimated_total_tokens"] == 4400
    assert estimate["estimated_total_cost_usd"] == 0.44
    assert estimate["vendor_jobs_with_history"] == 2
    assert estimate["vendor_jobs_using_fallback"] == 1
    assert estimate["cross_vendor_jobs_with_history"] == 3
    assert estimate["cross_vendor_jobs_using_fallback"] == 2
    assert estimate["estimated_vendor_tokens_likely_to_reason"] == 900
    assert estimate["estimated_vendor_cost_usd_likely_to_reason"] == 0.09
    assert estimate["vendor_jobs_likely_to_reason"] == 1
    assert estimate["vendor_jobs_likely_hash_reuse"] == 1
    assert estimate["vendor_jobs_likely_stale_reuse"] == 1
    assert estimate["likely_rerun_vendors"] == ["Microsoft Dynamics:hash_changed"]


@pytest.mark.asyncio
async def test_preview_competitive_set_plan_returns_recent_runs(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    competitive_set = _competitive_set()
    repo = SimpleNamespace(
        get_by_id_for_account=AsyncMock(return_value=competitive_set),
        list_runs_for_account_set=AsyncMock(return_value=[
            SimpleNamespace(to_dict=lambda: {
                "id": str(uuid4()),
                "competitive_set_id": str(competitive_set.id),
                "account_id": str(competitive_set.account_id),
                "run_id": "scope-run-1",
                "trigger": "manual",
                "status": "succeeded",
                "execution_id": None,
                "summary": {"vendors_reasoned": 2, "vendors_skipped_hash_reuse": 1, "total_tokens": 12345},
                "started_at": "2026-04-05T12:00:00+00:00",
                "completed_at": "2026-04-05T12:01:00+00:00",
                "created_at": "2026-04-05T12:00:00+00:00",
            }),
        ]),
    )
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(competitive_set.account_id),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )
    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(
        mod,
        "_competitive_set_plan_payload",
        AsyncMock(return_value={"estimated_total_jobs": 2, "estimate": {"estimated_total_cost_usd": 0.12}}),
    )

    result = await mod.preview_competitive_set_plan(competitive_set.id, user=user)

    assert result["competitive_set"]["id"] == str(competitive_set.id)
    assert result["plan"]["estimated_total_jobs"] == 2
    assert len(result["recent_runs"]) == 1
    assert result["recent_runs"][0]["summary"]["total_tokens"] == 12345


@pytest.mark.asyncio
async def test_run_competitive_set_now_forwards_changed_only_flag(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    competitive_set = _competitive_set()
    repo = SimpleNamespace(
        get_by_id_for_account=AsyncMock(return_value=competitive_set),
    )
    scheduler = SimpleNamespace(
        run_now=AsyncMock(return_value={"status": "started", "message": "ok", "execution_id": "exec-1"}),
    )
    task = SimpleNamespace(metadata={"existing": True})
    task_repo = SimpleNamespace(get_by_name=AsyncMock(return_value=task))
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(competitive_set.account_id),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )

    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(mod, "get_task_scheduler", lambda: scheduler)
    monkeypatch.setattr(mod, "get_scheduled_task_repo", lambda: task_repo)
    monkeypatch.setattr(
        mod,
        "load_vendor_category_map",
        AsyncMock(return_value={"salesforce": "CRM", "hubspot": "CRM"}),
    )

    req = mod.CompetitiveSetRunRequest(changed_vendors_only=True)
    result = await mod.run_competitive_set_now(competitive_set.id, req, user=user)

    assert result["competitive_set_id"] == str(competitive_set.id)
    assert scheduler.run_now.await_count == 1
    assert task.metadata["changed_vendors_only"] is True
    assert task.metadata["scope_trigger"] == "manual"


@pytest.mark.asyncio
async def test_run_competitive_set_now_forwards_force_flags(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    competitive_set = _competitive_set()
    repo = SimpleNamespace(
        get_by_id_for_account=AsyncMock(return_value=competitive_set),
    )
    scheduler = SimpleNamespace(
        run_now=AsyncMock(return_value={"status": "started", "message": "ok", "execution_id": "exec-1"}),
    )
    task = SimpleNamespace(metadata={"existing": True})
    task_repo = SimpleNamespace(get_by_name=AsyncMock(return_value=task))
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(competitive_set.account_id),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )

    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(mod, "get_task_scheduler", lambda: scheduler)
    monkeypatch.setattr(mod, "get_scheduled_task_repo", lambda: task_repo)
    monkeypatch.setattr(
        mod,
        "load_vendor_category_map",
        AsyncMock(return_value={"salesforce": "CRM", "hubspot": "CRM"}),
    )

    req = mod.CompetitiveSetRunRequest(
        force=True,
        force_cross_vendor=True,
        changed_vendors_only=False,
    )
    result = await mod.run_competitive_set_now(competitive_set.id, req, user=user)

    assert result["competitive_set_id"] == str(competitive_set.id)
    assert scheduler.run_now.await_count == 1
    assert task.metadata["force"] is True
    assert task.metadata["force_cross_vendor"] is True
    assert task.metadata["changed_vendors_only"] is False
    assert task.metadata["scope_trigger"] == "manual"


@pytest.mark.asyncio
async def test_run_competitive_set_now_uses_config_default_when_flag_omitted(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    competitive_set = _competitive_set()
    repo = SimpleNamespace(
        get_by_id_for_account=AsyncMock(return_value=competitive_set),
    )
    scheduler = SimpleNamespace(
        run_now=AsyncMock(return_value={"status": "started", "message": "ok", "execution_id": "exec-1"}),
    )
    task = SimpleNamespace(metadata={"existing": True})
    task_repo = SimpleNamespace(get_by_name=AsyncMock(return_value=task))
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(competitive_set.account_id),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )

    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(mod, "get_task_scheduler", lambda: scheduler)
    monkeypatch.setattr(mod, "get_scheduled_task_repo", lambda: task_repo)
    monkeypatch.setattr(
        mod,
        "load_vendor_category_map",
        AsyncMock(return_value={"salesforce": "CRM", "hubspot": "CRM"}),
    )
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "competitive_set_changed_vendors_only_default",
        False,
        raising=False,
    )

    req = mod.CompetitiveSetRunRequest()
    await mod.run_competitive_set_now(competitive_set.id, req, user=user)

    assert task.metadata["changed_vendors_only"] is False


@pytest.mark.asyncio
async def test_run_competitive_set_now_passes_through_already_running(monkeypatch):
    from atlas_brain.api import b2b_tenant_dashboard as mod

    competitive_set = _competitive_set()
    repo = SimpleNamespace(
        get_by_id_for_account=AsyncMock(return_value=competitive_set),
    )
    scheduler = SimpleNamespace(
        run_now=AsyncMock(return_value={
            "status": "running",
            "message": "Task 'b2b_reasoning_synthesis' is already running.",
            "execution_id": "exec-existing",
            "already_running": True,
        }),
    )
    task = SimpleNamespace(metadata={"existing": True})
    task_repo = SimpleNamespace(get_by_name=AsyncMock(return_value=task))
    user = AuthUser(
        user_id=str(uuid4()),
        account_id=str(competitive_set.account_id),
        plan="b2b_pro",
        plan_status="active",
        role="owner",
        product="b2b_retention",
        is_admin=True,
    )

    monkeypatch.setattr(mod, "_pool_or_503", lambda: object())
    monkeypatch.setattr(mod, "get_competitive_set_repo", lambda: repo)
    monkeypatch.setattr(mod, "get_task_scheduler", lambda: scheduler)
    monkeypatch.setattr(mod, "get_scheduled_task_repo", lambda: task_repo)
    monkeypatch.setattr(
        mod,
        "load_vendor_category_map",
        AsyncMock(return_value={"salesforce": "CRM", "hubspot": "CRM"}),
    )

    result = await mod.run_competitive_set_now(
        competitive_set.id,
        mod.CompetitiveSetRunRequest(),
        user=user,
    )

    assert result["competitive_set_id"] == str(competitive_set.id)
    assert result["execution_id"] == "exec-existing"
    assert result["already_running"] is True


@pytest.mark.asyncio
async def test_competitive_scope_finalize_marks_cross_vendor_only_failures_as_failed(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    marked = {}
    scope_id = uuid4()

    class FakeRepo:
        async def mark_run_started(self, *args, **kwargs):
            return None

        async def mark_run_completed(self, competitive_set_id, *, status, summary):
            marked["competitive_set_id"] = str(competitive_set_id)
            marked["status"] = status
            marked["summary"] = summary

    class FakePool:
        is_initialized = True

        async def fetch(self, query, *args):
            return []

        async def execute(self, query, *args):
            return None

    class FakeLLM:
        model = "fake-reasoner"

    async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
        return {
            "Salesforce": {"evidence_vault": {"product_category": "CRM"}},
            "HubSpot": {"evidence_vault": {"product_category": "CRM"}},
        }

    async def _fake_run_cross_vendor_synthesis(**kwargs):
        return (0, 1, 0, 0, 0)

    monkeypatch.setattr(mod, "get_db_pool", lambda: FakePool())
    monkeypatch.setattr(mod.settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False)
    monkeypatch.setattr(mod.settings.b2b_churn, "cross_vendor_synthesis_enabled", True, raising=False)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
        _fake_fetch_all_pool_layers,
    )
    monkeypatch.setattr(
        "atlas_brain.pipelines.llm.get_pipeline_llm",
        lambda **kw: FakeLLM(),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_reasoning_synthesis._run_cross_vendor_synthesis",
        _fake_run_cross_vendor_synthesis,
    )
    monkeypatch.setattr(
        "atlas_brain.storage.repositories.competitive_set.get_competitive_set_repo",
        lambda: FakeRepo(),
    )

    result = await mod.run(SimpleNamespace(metadata={
        "scope_type": "competitive_set",
        "scope_id": str(scope_id),
        "scope_vendor_names": ["Salesforce", "HubSpot"],
        "scope_pairwise_pairs": [["Salesforce", "HubSpot"]],
        "vendor_synthesis_enabled": False,
        "changed_vendors_only": False,
        "scope_trigger": "manual",
    }))

    assert result["cross_vendor_failed"] == 1
    assert marked["competitive_set_id"] == str(scope_id)
    assert marked["status"] == "failed"


@pytest.mark.asyncio
async def test_competitive_scope_finalize_preserves_force_flags_on_skip(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    marked = {}
    scope_id = uuid4()

    class FakeRepo:
        async def mark_run_completed(self, competitive_set_id, *, status, summary):
            marked["competitive_set_id"] = str(competitive_set_id)
            marked["status"] = status
            marked["summary"] = summary

    class FakePool:
        is_initialized = True

    async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
        return {}

    monkeypatch.setattr(mod, "get_db_pool", lambda: FakePool())
    monkeypatch.setattr(mod.settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
        _fake_fetch_all_pool_layers,
    )
    monkeypatch.setattr(
        "atlas_brain.storage.repositories.competitive_set.get_competitive_set_repo",
        lambda: FakeRepo(),
    )

    result = await mod.run(SimpleNamespace(metadata={
        "scope_type": "competitive_set",
        "scope_id": str(scope_id),
        "scope_vendor_names": ["Salesforce", "HubSpot"],
        "force": True,
        "force_cross_vendor": True,
        "changed_vendors_only": True,
        "scope_trigger": "manual",
    }))

    assert result["_skip_synthesis"] == "No pool data available"
    assert marked["competitive_set_id"] == str(scope_id)
    assert marked["status"] == "failed"
    assert marked["summary"]["force"] is True
    assert marked["summary"]["force_cross_vendor"] is True
    assert marked["summary"]["changed_vendors_only"] is True
    assert marked["summary"]["_skip_synthesis"] == "No pool data available"


@pytest.mark.asyncio
async def test_competitive_scope_start_forwards_force_flags(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    started = {}
    scope_id = uuid4()

    class FakeRepo:
        async def mark_run_started(self, competitive_set_id, *, run_id, trigger, execution_id=None, summary=None):
            started["competitive_set_id"] = str(competitive_set_id)
            started["run_id"] = run_id
            started["trigger"] = trigger
            started["execution_id"] = execution_id
            started["summary"] = summary or {}

        async def mark_run_completed(self, competitive_set_id, *, status, summary):
            return None

    class FakePool:
        is_initialized = True

    async def _fake_fetch_all_pool_layers(pool, *, as_of, analysis_window_days, vendor_names=None):
        return {}

    monkeypatch.setattr(mod, "get_db_pool", lambda: FakePool())
    monkeypatch.setattr(mod.settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_shared.fetch_all_pool_layers",
        _fake_fetch_all_pool_layers,
    )
    monkeypatch.setattr(
        "atlas_brain.storage.repositories.competitive_set.get_competitive_set_repo",
        lambda: FakeRepo(),
    )

    await mod.run(SimpleNamespace(metadata={
        "scope_type": "competitive_set",
        "scope_id": str(scope_id),
        "scope_vendor_names": ["Salesforce", "HubSpot"],
        "force": True,
        "force_cross_vendor": True,
        "changed_vendors_only": True,
        "scope_trigger": "manual",
        "_execution_id": "exec-123",
    }))

    assert started["competitive_set_id"] == str(scope_id)
    assert started["trigger"] == "manual"
    assert started["execution_id"] == "exec-123"
    assert started["summary"]["force"] is True
    assert started["summary"]["force_cross_vendor"] is True
    assert started["summary"]["changed_vendors_only"] is True


@pytest.mark.asyncio
async def test_competitive_set_repo_mark_run_started_persists_scope_flags(monkeypatch):
    from atlas_brain.storage.repositories import competitive_set as mod

    executed = []
    competitive_set = _competitive_set()

    class FakeConn:
        async def execute(self, query, *args):
            executed.append((query, args))

    class FakeTransaction:
        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        is_initialized = True

        def transaction(self):
            return FakeTransaction()

    repo = mod.CompetitiveSetRepository()

    monkeypatch.setattr(mod, "get_db_pool", lambda: FakePool())
    monkeypatch.setattr(repo, "get_by_id", AsyncMock(return_value=competitive_set))

    await repo.mark_run_started(
        competitive_set.id,
        run_id="run-123",
        trigger="manual",
        execution_id="exec-123",
        summary={
            "force": True,
            "force_cross_vendor": True,
            "changed_vendors_only": True,
        },
    )

    insert_summary = json.loads(executed[0][1][5])
    update_summary = json.loads(executed[1][1][1])

    assert insert_summary["run_id"] == "run-123"
    assert insert_summary["trigger"] == "manual"
    assert insert_summary["execution_id"] == "exec-123"
    assert insert_summary["force"] is True
    assert insert_summary["force_cross_vendor"] is True
    assert insert_summary["changed_vendors_only"] is True
    assert update_summary == insert_summary


@pytest.mark.asyncio
async def test_scheduled_competitive_sets_use_configured_changed_only_default(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_reasoning_synthesis as mod

    due_set = _competitive_set()
    repo = SimpleNamespace(
        list_due_scheduled=AsyncMock(return_value=[due_set]),
    )
    child_meta = {}
    original_run = mod.run

    class FakePool:
        is_initialized = True

    async def _fake_run(task):
        metadata = getattr(task, "metadata", {}) or {}
        if metadata.get("scope_type") == "competitive_set":
            child_meta.update(metadata)
            return {
                "vendors_reasoned": 0,
                "vendors_failed": 0,
                "vendors_skipped": 0,
                "cross_vendor_succeeded": 0,
                "cross_vendor_failed": 0,
                "total_tokens": 0,
            }
        return await original_run(task)

    monkeypatch.setattr(mod, "get_db_pool", lambda: FakePool())
    monkeypatch.setattr(mod.settings.b2b_churn, "reasoning_synthesis_enabled", True, raising=False)
    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "competitive_set_changed_vendors_only_default",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "atlas_brain.storage.repositories.competitive_set.get_competitive_set_repo",
        lambda: repo,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b_competitive_sets.load_vendor_category_map",
        AsyncMock(return_value={"salesforce": "CRM", "hubspot": "CRM", "microsoft dynamics": "CRM"}),
        raising=False,
    )
    monkeypatch.setattr(mod, "run", _fake_run)

    result = await _fake_run(SimpleNamespace(
        metadata={"scheduled_scope_strategy": "competitive_sets"},
        id=uuid4(),
        name="b2b_reasoning_synthesis",
        task_type="builtin",
        schedule_type="interval",
        description=None,
        prompt=None,
        agent_type=None,
        cron_expression=None,
        interval_seconds=3600,
        run_at=None,
        timezone="UTC",
        enabled=True,
        max_retries=0,
        retry_delay_seconds=0,
        timeout_seconds=0,
        created_at=None,
        updated_at=None,
        last_run_at=None,
        next_run_at=None,
    ))

    assert result["competitive_sets_processed"] == 1
    assert child_meta["changed_vendors_only"] is False
