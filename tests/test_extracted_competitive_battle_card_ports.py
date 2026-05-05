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
    ):
        sys.modules.pop(module_name, None)
    _drop_package_attr("extracted_competitive_intelligence.services.b2b", "battle_card_ports")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "b2b_battle_cards")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "_b2b_shared")
    _drop_package_attr("atlas_brain.autonomous.tasks", "_b2b_shared")


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

    payload = battle_cards._build_battle_card_render_payload(
        {"vendor": "Acme", "total_reviews": 12}
    )
    assert payload["locked_facts"] == {"vendor": "Acme", "locked": True}
    assert payload["metric_ledger"] == [{"label": "Reviews analyzed", "value": 12}]

    assert ("has_complete_core_run_marker", date.today().isoformat()) in port.calls
    assert ("locked_facts", "Acme") in port.calls
    assert ("metric_ledger", "Acme") in port.calls
    assert _EXTRACTED_SHARED_MODULE not in sys.modules
    assert _ATLAS_SHARED_MODULE not in sys.modules
