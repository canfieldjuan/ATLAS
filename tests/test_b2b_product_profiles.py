import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks import b2b_product_profiles as profiles


@pytest.mark.asyncio
async def test_run_scopes_to_test_vendors(monkeypatch):
    monkeypatch.setattr(profiles.settings.b2b_churn, "enabled", True)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_enabled", True)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_min_reviews", 3)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_max_tokens", 256)

    class DummyPool:
        is_initialized = True

    class DummyClient:
        async def aclose(self):
            return None

    async def _empty(*_args, **_kwargs):
        return {}

    async def _aggregate(*_args, **_kwargs):
        return {
            "Zendesk": {"product_category": "Help Desk", "total_reviews": 25},
            "HubSpot": {"product_category": "CRM", "total_reviews": 40},
        }

    async def _resolve(vendor):
        return vendor

    async def _synthesize_profile(**_kwargs):
        return "summary", []

    captured = []

    async def _upsert(_pool, profile):
        captured.append(profile["vendor_name"])

    async def _snapshots(_pool, generated):
        return generated

    monkeypatch.setattr(profiles, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(profiles.httpx, "AsyncClient", lambda timeout=120: DummyClient())
    monkeypatch.setattr(profiles, "_fetch_satisfaction_by_area", _empty)
    monkeypatch.setattr(profiles, "_fetch_pain_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_use_case_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_company_size_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_competitive_flows", _empty)
    monkeypatch.setattr(profiles, "_fetch_integration_stacks", _empty)
    monkeypatch.setattr(profiles, "_fetch_aggregate_metrics", _aggregate)
    monkeypatch.setattr(profiles, "_fetch_source_distribution", _empty)
    monkeypatch.setattr(profiles, "resolve_vendor_name", _resolve)
    monkeypatch.setattr(profiles, "_synthesize_profile", _synthesize_profile)
    monkeypatch.setattr(profiles, "_upsert_profile", _upsert)
    monkeypatch.setattr(profiles, "_persist_profile_snapshots", _snapshots)

    task = type("Task", (), {"metadata": {"test_vendors": ["Zendesk"]}})()
    result = await profiles.run(task)

    assert captured == ["Zendesk"]
    assert result["vendors_processed"] == 1
    assert result["total_eligible"] == 1
    assert result["snapshots_persisted"] == 1


def test_profile_payload_builders_are_deterministic_for_ties():
    strengths, weaknesses = profiles._build_strengths_weaknesses(
        [
            {"area": "ux", "score": 4.2, "evidence_count": 4},
            {"area": "pricing", "score": 4.2, "evidence_count": 4},
            {"area": "support", "score": 2.8, "evidence_count": 3},
            {"area": "api", "score": 2.8, "evidence_count": 3},
        ]
    )
    assert [item["area"] for item in strengths] == ["pricing", "ux"]
    assert [item["area"] for item in weaknesses] == ["api", "support"]

    use_cases = profiles._build_use_cases(
        [
            {"use_case": "Ticketing", "count": 8},
            {"use_case": "Automation", "count": 8},
            {"use_case": "Reporting", "count": 5},
        ],
        total_reviews=20,
    )
    assert [item["use_case"] for item in use_cases] == ["Automation", "Ticketing", "Reporting"]

    compared_to, switched_from = profiles._build_competitive_positioning(
        {
            "compared_to": {
                "Zendesk": {"mentions": 6, "reasons": ["market leader"]},
                "Freshdesk": {"mentions": 6, "reasons": ["lower cost"]},
            },
            "switched_from": {
                "Intercom": {"count": 3, "reasons": ["pricing"]},
                "Help Scout": {"count": 3, "reasons": ["support"]},
            },
        }
    )
    assert [item["vendor"] for item in compared_to] == ["Freshdesk", "Zendesk"]
    assert [item["vendor"] for item in switched_from] == ["Help Scout", "Intercom"]

    integrations = profiles._build_top_integrations(
        {
            "Slack": 5,
            "HubSpot": 5,
            "Salesforce": 3,
        }
    )
    assert integrations == ["HubSpot", "Slack", "Salesforce"]


def test_profile_batch_custom_id_is_anthropic_safe():
    assert profiles._profile_batch_custom_id("Microsoft Teams") == "profile_Microsoft_Teams"
    assert profiles._profile_batch_custom_id("ACME: CRM / Core") == "profile_ACME_CRM_Core"


@pytest.mark.asyncio
async def test_profile_fetchers_read_vendor_mentions():
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]), fetchrow=AsyncMock(return_value=None))

    await profiles._fetch_satisfaction_by_area(pool, 30)
    assert "JOIN b2b_review_vendor_mentions vm" in pool.fetch.await_args_list[0].args[0]

    await profiles._fetch_pain_distribution(pool, 30)
    assert "JOIN b2b_review_vendor_mentions vm" in pool.fetch.await_args_list[1].args[0]

    await profiles._fetch_use_case_distribution(pool, 30)
    assert "JOIN b2b_review_vendor_mentions vm" in pool.fetch.await_args_list[2].args[0]

    await profiles._fetch_company_size_distribution(pool, 30)
    assert "JOIN b2b_review_vendor_mentions vm" in pool.fetch.await_args_list[3].args[0]

    await profiles._fetch_competitive_flows(pool, 30)
    assert "JOIN b2b_review_vendor_mentions vm" in pool.fetch.await_args_list[4].args[0]

    await profiles._fetch_integration_stacks(pool, 30)
    assert "JOIN b2b_review_vendor_mentions vm" in pool.fetch.await_args_list[5].args[0]


@pytest.mark.asyncio
async def test_profile_aggregate_and_source_fetchers_read_vendor_mentions():
    pool = SimpleNamespace(fetch=AsyncMock(return_value=[]), fetchrow=AsyncMock(return_value=None))

    await profiles._fetch_aggregate_metrics(pool, 30, 3)
    aggregate_sql = pool.fetch.await_args_list[0].args[0]
    assert "JOIN b2b_review_vendor_mentions vm" in aggregate_sql
    assert "GROUP BY vm.vendor_name" in aggregate_sql

    await profiles._fetch_source_distribution(pool, 30)
    source_sql = pool.fetch.await_args_list[1].args[0]
    assert "JOIN b2b_review_vendor_mentions vm" in source_sql
    assert "GROUP BY vm.vendor_name, r.source" in source_sql


@pytest.mark.asyncio
async def test_run_uses_anthropic_batch_for_product_profiles(monkeypatch):
    class DummyPool:
        is_initialized = True

    class DummyClient:
        async def aclose(self):
            return None

    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    async def _empty(*_args, **_kwargs):
        return {}

    async def _aggregate(*_args, **_kwargs):
        return {
            "Zendesk": {"product_category": "Help Desk", "total_reviews": 25},
        }

    async def _resolve(vendor):
        return vendor

    captured = []

    async def _upsert(_pool, profile):
        captured.append(profile["vendor_name"])

    async def _snapshots(_pool, generated):
        return generated

    monkeypatch.setattr(profiles.settings.b2b_churn, "enabled", True)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_enabled", True)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_min_reviews", 3)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_max_tokens", 256)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_llm_backend", "openrouter")
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_openrouter_model", "anthropic/claude-haiku-4-5")
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_anthropic_batch_enabled", True)

    monkeypatch.setattr(profiles, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(profiles.httpx, "AsyncClient", lambda timeout=120: DummyClient())
    monkeypatch.setattr(profiles, "_fetch_satisfaction_by_area", _empty)
    monkeypatch.setattr(profiles, "_fetch_pain_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_use_case_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_company_size_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_competitive_flows", _empty)
    monkeypatch.setattr(profiles, "_fetch_integration_stacks", _empty)
    monkeypatch.setattr(profiles, "_fetch_aggregate_metrics", _aggregate)
    monkeypatch.setattr(profiles, "_fetch_source_distribution", _empty)
    monkeypatch.setattr(profiles, "resolve_vendor_name", _resolve)
    monkeypatch.setattr(profiles, "_upsert_profile", _upsert)
    monkeypatch.setattr(profiles, "_persist_profile_snapshots", _snapshots)
    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: FakeAnthropicLLM(),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(get=lambda name: SimpleNamespace(content="profile prompt") if name == "b2b/product_profile_synthesis" else None),
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.prepare_b2b_exact_stage_request",
        lambda *args, **_kwargs: SimpleNamespace(namespace="ns", request_envelope={}, provider="openrouter", model="anthropic/claude-haiku-4-5"),
    )
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.lookup_b2b_exact_stage_text", AsyncMock(return_value=None))
    monkeypatch.setattr("atlas_brain.services.b2b.cache_runner.store_b2b_exact_stage_text", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        AsyncMock(
            return_value=SimpleNamespace(
                local_batch_id="batch-1",
                provider_batch_id="provider-batch-1",
                submitted_items=1,
                cache_prefiltered_items=0,
                fallback_single_call_items=0,
                completed_items=1,
                failed_items=0,
                results_by_custom_id={
                    "profile_Zendesk": SimpleNamespace(
                        response_text=json.dumps(
                            {"summary": "Strong on workflow coverage, weaker on support.", "pain_addressed": {"support": 0.4}}
                        ),
                        cached=False,
                        usage={},
                        error_text=None,
                    )
                },
            )
        ),
    )

    task = type(
        "Task",
        (),
        {
            "metadata": {
                "test_vendors": ["Zendesk"],
                "anthropic_batch_enabled": True,
                "product_profile_anthropic_batch_enabled": True,
            }
        },
    )()
    result = await profiles.run(task)

    assert captured == ["Zendesk"]
    assert result["vendors_processed"] == 1
    assert result["anthropic_batch_jobs"] == 1
    assert result["anthropic_batch_items_submitted"] == 1


@pytest.mark.asyncio
async def test_run_reuses_existing_completed_batch_result(monkeypatch):
    class DummyPool:
        is_initialized = True

    class DummyClient:
        async def aclose(self):
            return None

    class FakeAnthropicLLM:
        def __init__(self, model: str = "claude-haiku-4-5"):
            self.model = model
            self.name = "anthropic"

    async def _empty(*_args, **_kwargs):
        return {}

    async def _aggregate(*_args, **_kwargs):
        return {
            "Zendesk": {"product_category": "Help Desk", "total_reviews": 25},
        }

    async def _resolve(vendor):
        return vendor

    captured = []

    async def _upsert(_pool, profile):
        captured.append(profile["vendor_name"])

    async def _snapshots(_pool, generated):
        return generated

    monkeypatch.setattr(profiles.settings.b2b_churn, "enabled", True)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_enabled", True)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_min_reviews", 3)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_max_tokens", 256)
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_llm_backend", "openrouter")
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_openrouter_model", "anthropic/claude-haiku-4-5")
    monkeypatch.setattr(profiles.settings.b2b_churn, "product_profile_anthropic_batch_enabled", True)

    monkeypatch.setattr(profiles, "get_db_pool", lambda: DummyPool())
    monkeypatch.setattr(profiles.httpx, "AsyncClient", lambda timeout=120: DummyClient())
    monkeypatch.setattr(profiles, "_fetch_satisfaction_by_area", _empty)
    monkeypatch.setattr(profiles, "_fetch_pain_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_use_case_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_company_size_distribution", _empty)
    monkeypatch.setattr(profiles, "_fetch_competitive_flows", _empty)
    monkeypatch.setattr(profiles, "_fetch_integration_stacks", _empty)
    monkeypatch.setattr(profiles, "_fetch_aggregate_metrics", _aggregate)
    monkeypatch.setattr(profiles, "_fetch_source_distribution", _empty)
    monkeypatch.setattr(profiles, "resolve_vendor_name", _resolve)
    monkeypatch.setattr(profiles, "_upsert_profile", _upsert)
    monkeypatch.setattr(profiles, "_persist_profile_snapshots", _snapshots)
    monkeypatch.setattr("atlas_brain.services.llm.anthropic.AnthropicLLM", FakeAnthropicLLM)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.resolve_anthropic_batch_llm",
        lambda **_kwargs: FakeAnthropicLLM(),
    )
    monkeypatch.setattr(
        "atlas_brain.skills.get_skill_registry",
        lambda: SimpleNamespace(get=lambda name: SimpleNamespace(content="profile prompt") if name == "b2b/product_profile_synthesis" else None),
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_batch_utils.reconcile_existing_batch_artifacts",
        AsyncMock(
            return_value={
                "Zendesk": {
                    "state": "succeeded",
                    "response_text": json.dumps(
                        {
                            "summary": "Recovered from prior batch output.",
                            "pain_addressed": {"support": 0.6},
                        }
                    ),
                    "custom_id": "profile_Zendesk",
                }
            }
        ),
    )
    run_batch = AsyncMock()
    monkeypatch.setattr(
        "atlas_brain.services.b2b.anthropic_batch.run_anthropic_message_batch",
        run_batch,
    )

    task = type(
        "Task",
        (),
        {
            "metadata": {
                "test_vendors": ["Zendesk"],
                "anthropic_batch_enabled": True,
                "product_profile_anthropic_batch_enabled": True,
            }
        },
    )()
    result = await profiles.run(task)

    assert captured == ["Zendesk"]
    assert result["vendors_processed"] == 1
    assert result["anthropic_batch_reused_completed_items"] == 1
    run_batch.assert_not_awaited()
