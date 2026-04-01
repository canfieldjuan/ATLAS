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
