from __future__ import annotations

import importlib
import sys
from datetime import date

import pytest


_COMP_INTEL_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"
_PORT_MODULE = "extracted_competitive_intelligence.services.b2b.battle_card_ports"
_BATTLE_CARD_MODULE = (
    "extracted_competitive_intelligence.autonomous.tasks.b2b_battle_cards"
)
_EXTRACTED_SHARED_MODULE = "extracted_competitive_intelligence.autonomous.tasks._b2b_shared"
_ATLAS_SHARED_MODULE = "atlas_brain.autonomous.tasks._b2b_shared"
_EXTRACTED_CHURN_MODULE = (
    "extracted_competitive_intelligence.autonomous.tasks.b2b_churn_intelligence"
)
_ATLAS_CHURN_MODULE = "atlas_brain.autonomous.tasks.b2b_churn_intelligence"
_EXTRACTED_PROGRESS_MODULE = (
    "extracted_competitive_intelligence.autonomous.tasks._execution_progress"
)
_ATLAS_PROGRESS_MODULE = "atlas_brain.autonomous.tasks._execution_progress"
_EXTRACTED_SYNTHESIS_MODULE = (
    "extracted_competitive_intelligence.autonomous.tasks._b2b_synthesis_reader"
)
_ATLAS_SYNTHESIS_MODULE = "atlas_brain.autonomous.tasks._b2b_synthesis_reader"


def _drop_package_attr(package_name: str, attr_name: str) -> None:
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attr_name):
        delattr(package, attr_name)


def _reset_modules() -> None:
    for module_name in (
        _PORT_MODULE,
        _BATTLE_CARD_MODULE,
        _EXTRACTED_SHARED_MODULE,
        _ATLAS_SHARED_MODULE,
        _EXTRACTED_CHURN_MODULE,
        _ATLAS_CHURN_MODULE,
        _EXTRACTED_PROGRESS_MODULE,
        _ATLAS_PROGRESS_MODULE,
        _EXTRACTED_SYNTHESIS_MODULE,
        _ATLAS_SYNTHESIS_MODULE,
    ):
        sys.modules.pop(module_name, None)
    _drop_package_attr("extracted_competitive_intelligence.services.b2b", "battle_card_ports")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "b2b_battle_cards")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "_b2b_shared")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "b2b_churn_intelligence")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "_execution_progress")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "_b2b_synthesis_reader")
    _drop_package_attr("atlas_brain.autonomous.tasks", "_b2b_shared")
    _drop_package_attr("atlas_brain.autonomous.tasks", "b2b_churn_intelligence")
    _drop_package_attr("atlas_brain.autonomous.tasks", "_execution_progress")
    _drop_package_attr("atlas_brain.autonomous.tasks", "_b2b_synthesis_reader")


class FakeSynthesisView:
    def __init__(self, vendor_name: str, schema_version: str) -> None:
        self.vendor_name = vendor_name
        self.schema_version = schema_version


class RecordingPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def has_complete_core_run_marker(self, pool, report_date):
        self.calls.append(("has_complete_core_run_marker", report_date.isoformat()))
        return True

    async def latest_complete_core_report_date(self, pool):
        self.calls.append(("latest_complete_core_report_date", ""))
        return date(2026, 5, 1)

    async def describe_core_run_gap(self, pool, report_date):
        self.calls.append(("describe_core_run_gap", report_date.isoformat()))
        return "core gap"

    async def update_execution_progress(
        self,
        task,
        *,
        stage,
        progress_current=None,
        progress_total=None,
        progress_message=None,
        **counters,
    ):
        self.calls.append(("update_execution_progress", stage))
        task.progress_payload = {
            "stage": stage,
            "progress_current": progress_current,
            "progress_total": progress_total,
            "progress_message": progress_message,
            **counters,
        }

    def normalize_test_vendors(self, raw):
        self.calls.append(("normalize_test_vendors", str(raw)))
        return ["Acme"]

    def apply_vendor_scope_to_churn_inputs(self, data, vendor_names):
        self.calls.append(("apply_vendor_scope", ",".join(vendor_names or [])))
        return ({"vendor_scores": [{"vendor_name": "Acme"}]}, list(vendor_names or []))

    async def load_best_reasoning_views(
        self,
        pool,
        vendor_names,
        *,
        as_of=None,
        analysis_window_days=30,
    ):
        self.calls.append((
            "load_best_reasoning_views",
            f"{','.join(vendor_names)}:{analysis_window_days}",
        ))
        return {
            vendor: FakeSynthesisView(vendor, "synthesis_v2")
            for vendor in vendor_names
        }

    def build_reasoning_lookup_from_views(self, synthesis_views):
        self.calls.append(("build_reasoning_lookup_from_views", str(len(synthesis_views))))
        return {
            vendor: {"vendor": vendor, "schema_version": view.schema_version}
            for vendor, view in synthesis_views.items()
        }

    def _build_battle_card_locked_facts(self, card):
        self.calls.append(("locked_facts", str(card.get("vendor") or "")))
        return {"vendor": card.get("vendor"), "locked": True}

    def _build_metric_ledger(self, card):
        self.calls.append(("metric_ledger", str(card.get("vendor") or "")))
        return [{"label": "Reviews analyzed", "value": card.get("total_reviews")}]

    async def _fetch_pain_distribution(self, pool, window_days):
        self.calls.append(("pain_distribution", str(window_days)))
        return [{"vendor": "Acme", "pain": "support"}]


@pytest.fixture(autouse=True)
def reset_configured_port():
    yield
    module = sys.modules.get(_PORT_MODULE)
    if module is not None:
        module.configure_battle_card_support_port(None)


def test_battle_card_support_port_fails_closed_in_standalone(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    _reset_modules()

    module = importlib.import_module(_PORT_MODULE)

    with pytest.raises(module.BattleCardSupportPortNotConfigured):
        module.get_battle_card_support_port()
    with pytest.raises(module.BattleCardSupportPortNotConfigured):
        module._build_metric_ledger({})

    assert _EXTRACTED_SHARED_MODULE not in sys.modules
    assert _ATLAS_SHARED_MODULE not in sys.modules
    assert _EXTRACTED_CHURN_MODULE not in sys.modules
    assert _ATLAS_CHURN_MODULE not in sys.modules
    assert _EXTRACTED_PROGRESS_MODULE not in sys.modules
    assert _ATLAS_PROGRESS_MODULE not in sys.modules
    assert _EXTRACTED_SYNTHESIS_MODULE not in sys.modules
    assert _ATLAS_SYNTHESIS_MODULE not in sys.modules


@pytest.mark.asyncio
async def test_battle_card_task_uses_configured_support_port(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    _reset_modules()

    port_module = importlib.import_module(_PORT_MODULE)
    port = RecordingPort()
    port_module.configure_battle_card_support_port(port)
    battle_cards = importlib.import_module(_BATTLE_CARD_MODULE)

    pool = object()

    assert await battle_cards._check_freshness(pool) == date.today()
    assert await battle_cards._latest_core_report_date(pool) == date(2026, 5, 1)
    assert await port_module.describe_core_run_gap(pool, date(2026, 5, 2)) == "core gap"
    assert await port_module._fetch_pain_distribution(pool, 30) == [
        {"vendor": "Acme", "pain": "support"}
    ]
    assert port_module.normalize_test_vendors("Acme, Acme") == ["Acme"]
    assert port_module.apply_vendor_scope_to_churn_inputs(
        {"vendor_scores": [{"vendor_name": "Acme"}, {"vendor_name": "Other"}]},
        ["Acme"],
    ) == ({"vendor_scores": [{"vendor_name": "Acme"}]}, ["Acme"])
    synthesis_views = await port_module.load_best_reasoning_views(
        pool,
        ["Acme"],
        as_of=date(2026, 5, 2),
        analysis_window_days=45,
    )
    assert list(synthesis_views) == ["Acme"]
    assert port_module.build_reasoning_lookup_from_views(synthesis_views) == {
        "Acme": {"vendor": "Acme", "schema_version": "synthesis_v2"}
    }

    task = type("Task", (), {})()
    await battle_cards._update_execution_progress(
        task,
        stage="loading_inputs",
        progress_current=1,
        progress_total=3,
        progress_message="Loading",
        cards_built=2,
    )
    assert task.progress_payload == {
        "stage": "loading_inputs",
        "progress_current": 1,
        "progress_total": 3,
        "progress_message": "Loading",
        "cards_built": 2,
    }

    payload = battle_cards._build_battle_card_render_payload(
        {"vendor": "Acme", "total_reviews": 12}
    )
    assert payload["locked_facts"] == {"vendor": "Acme", "locked": True}
    assert payload["metric_ledger"] == [{"label": "Reviews analyzed", "value": 12}]

    assert ("has_complete_core_run_marker", date.today().isoformat()) in port.calls
    assert ("normalize_test_vendors", "Acme, Acme") in port.calls
    assert ("apply_vendor_scope", "Acme") in port.calls
    assert ("load_best_reasoning_views", "Acme:45") in port.calls
    assert ("build_reasoning_lookup_from_views", "1") in port.calls
    assert ("update_execution_progress", "loading_inputs") in port.calls
    assert ("locked_facts", "Acme") in port.calls
    assert ("metric_ledger", "Acme") in port.calls
    assert _EXTRACTED_SHARED_MODULE not in sys.modules
    assert _ATLAS_SHARED_MODULE not in sys.modules
    assert _EXTRACTED_CHURN_MODULE not in sys.modules
    assert _ATLAS_CHURN_MODULE not in sys.modules
    assert _EXTRACTED_PROGRESS_MODULE not in sys.modules
    assert _ATLAS_PROGRESS_MODULE not in sys.modules
    assert _EXTRACTED_SYNTHESIS_MODULE not in sys.modules
    assert _ATLAS_SYNTHESIS_MODULE not in sys.modules
