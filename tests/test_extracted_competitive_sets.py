from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from extracted_competitive_intelligence.services import b2b_competitive_sets as mod
from extracted_competitive_intelligence.services.b2b.competitive_set_ports import (
    CompetitiveSetReasoningPortNotConfigured,
    configure_competitive_set_reasoning_port,
    get_competitive_set_reasoning_port,
)


def _competitive_set(**overrides: Any) -> SimpleNamespace:
    payload = {
        "id": "set-1",
        "focal_vendor_name": "Salesforce",
        "competitor_vendor_names": ["HubSpot", "Microsoft Dynamics", "HubSpot"],
        "vendor_synthesis_enabled": True,
        "pairwise_enabled": True,
        "category_council_enabled": True,
        "asymmetry_enabled": True,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class FakeReasoningPort:
    schema_version = "v2"

    def __init__(self) -> None:
        self.pool_fetch_calls: list[dict[str, Any]] = []
        self.scorecard_calls: list[list[str]] = []

    async def fetch_all_pool_layers(
        self,
        pool: Any,
        *,
        as_of: date,
        analysis_window_days: int,
        vendor_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        self.pool_fetch_calls.append(
            {
                "as_of": as_of,
                "analysis_window_days": analysis_window_days,
                "vendor_names": vendor_names,
            }
        )
        return {
            vendor_name: {"vendor_name": vendor_name}
            for vendor_name in vendor_names
        }

    async def read_vendor_scorecard_details(
        self,
        pool: Any,
        *,
        vendor_names: list[str],
    ) -> list[dict[str, Any]]:
        self.scorecard_calls.append(vendor_names)
        return [
            {"vendor_name": "HubSpot", "product_category": "CRM"},
        ]

    def compute_pool_hash(self, layers: dict[str, Any]) -> str:
        return f"hash:{layers['vendor_name']}"

    def compute_pool_hash_legacy(self, layers: dict[str, Any]) -> str:
        return f"legacy:{layers['vendor_name']}"

    def coerce_as_of_date(self, value: Any) -> date | None:
        return value if isinstance(value, date) else None

    def classify_vendor_reasoning_decision(
        self,
        *,
        vendor_name: str,
        today: date,
        evidence_hash: str,
        latest_row: dict[str, Any] | None,
        force: bool,
        max_stale_days: int,
        rerun_if_missing_packet_artifacts: bool,
        rerun_if_missing_reference_ids: bool,
        hash_matches_prior: bool,
    ) -> dict[str, Any]:
        if hash_matches_prior:
            return {"should_reason": False, "reason": "hash_reuse"}
        return {"should_reason": True, "reason": "hash_changed"}


@pytest.fixture(autouse=True)
def _reset_competitive_set_port() -> None:
    configure_competitive_set_reasoning_port(None)
    yield
    configure_competitive_set_reasoning_port(None)


def test_build_competitive_set_plan_scopes_jobs() -> None:
    plan = mod.build_competitive_set_plan(
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


def test_plan_to_synthesis_metadata_emits_scope_contract() -> None:
    plan = mod.build_competitive_set_plan(_competitive_set())

    metadata = mod.plan_to_synthesis_metadata(plan)

    assert metadata["scope_type"] == "competitive_set"
    assert metadata["scope_id"] == "set-1"
    assert metadata["scope_vendor_names"] == [
        "Salesforce",
        "HubSpot",
        "Microsoft Dynamics",
    ]


def test_competitive_set_reasoning_port_fails_closed_in_standalone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXTRACTED_COMP_INTEL_STANDALONE", "1")

    with pytest.raises(CompetitiveSetReasoningPortNotConfigured):
        get_competitive_set_reasoning_port()


@pytest.mark.asyncio
async def test_load_vendor_category_map_uses_configured_scorecard_port() -> None:
    reasoning_port = FakeReasoningPort()
    configure_competitive_set_reasoning_port(reasoning_port)

    class FakePool:
        async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
            assert "FROM requested r" in query
            return [
                {"vendor_name": "Salesforce", "product_category": "CRM"},
                {"vendor_name": "HubSpot", "product_category": None},
            ]

    result = await mod.load_vendor_category_map(FakePool(), ["Salesforce", "HubSpot"])

    assert reasoning_port.scorecard_calls == [["Salesforce", "HubSpot"]]
    assert result == {"salesforce": "CRM", "hubspot": "CRM"}


@pytest.mark.asyncio
async def test_vendor_reuse_estimate_uses_configured_reasoning_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning_port = FakeReasoningPort()
    configure_competitive_set_reasoning_port(reasoning_port)
    monkeypatch.setattr(mod.settings.b2b_churn, "intelligence_window_days", 30)

    class FakePool:
        async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
            assert args[0] == 30
            assert args[1] == "v2"
            return [
                {
                    "vendor_name": "Salesforce",
                    "as_of_date": date.today(),
                    "evidence_hash": "hash:Salesforce",
                    "synthesis": {},
                    "has_witness_pack": True,
                    "has_metric_refs": True,
                    "has_witness_refs": True,
                },
                {
                    "vendor_name": "HubSpot",
                    "as_of_date": date.today(),
                    "evidence_hash": "old-hubspot",
                    "synthesis": {},
                    "has_witness_pack": True,
                    "has_metric_refs": True,
                    "has_witness_refs": True,
                },
            ]

    plan = mod.build_competitive_set_plan(
        _competitive_set(competitor_vendor_names=["HubSpot"])
    )

    estimate = await mod._estimate_vendor_reuse_for_plan(FakePool(), plan)

    assert estimate["vendor_jobs_with_matching_pools"] == 2
    assert estimate["vendor_jobs_likely_to_reason"] == 1
    assert estimate["vendor_jobs_likely_hash_reuse"] == 1
    assert estimate["vendor_jobs_likely_hash_changed"] == 1
    assert estimate["likely_reuse_vendors"] == ["Salesforce:hash_reuse"]
    assert estimate["likely_rerun_vendors"] == ["HubSpot:hash_changed"]
