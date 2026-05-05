from __future__ import annotations

import importlib
import sys
from datetime import date
from types import SimpleNamespace

import pytest


_COMP_INTEL_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"
_PORT_MODULE = "extracted_competitive_intelligence.services.b2b.vendor_briefing_ports"
_VENDOR_BRIEFING_MODULE = (
    "extracted_competitive_intelligence.autonomous.tasks.b2b_vendor_briefing"
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
        _VENDOR_BRIEFING_MODULE,
        _EXTRACTED_SHARED_MODULE,
        _ATLAS_SHARED_MODULE,
    ):
        sys.modules.pop(module_name, None)
    _drop_package_attr("extracted_competitive_intelligence.services.b2b", "vendor_briefing_ports")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "b2b_vendor_briefing")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "_b2b_shared")
    _drop_package_attr("atlas_brain.autonomous.tasks", "_b2b_shared")


class RecordingPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def reasoning_int(self, value):
        self.calls.append(("reasoning_int", str(value)))
        return 42

    def timing_summary_payload(self, timing_intelligence):
        self.calls.append(("timing_summary_payload", ""))
        return ("timing", {"active_eval_signals": 2}, ["renewal"])

    def align_vendor_intelligence_record_to_scorecard(self, scorecard, record):
        self.calls.append(("align", str((scorecard or {}).get("vendor_name") or "")))
        return ({"vault": True}, {"matched_vendor_count": 1})

    async def read_vendor_company_signal_review_queue(
        self,
        pool,
        *,
        vendor_name,
        window_days=None,
        preview_limit=None,
    ):
        self.calls.append(("company_signal_queue", vendor_name))
        return {"vendor": vendor_name, "groups": []}

    async def read_vendor_intelligence_record(
        self,
        pool,
        vendor_name,
        *,
        as_of,
        analysis_window_days,
    ):
        self.calls.append(("intelligence_record", vendor_name))
        return {"vendor_name": vendor_name, "as_of": as_of.isoformat()}

    async def read_vendor_intelligence(
        self,
        pool,
        vendor_name,
        *,
        as_of,
        analysis_window_days,
    ):
        self.calls.append(("intelligence", vendor_name))
        return {"vendor_name": vendor_name, "vault": {}}

    async def read_vendor_scorecard_detail(self, pool, vendor_name):
        self.calls.append(("scorecard", vendor_name))
        return {"vendor_name": vendor_name, "score": 9}

    async def read_vendor_quote_evidence(
        self,
        pool,
        *,
        vendor_name,
        window_days=90,
        min_urgency=5.0,
        limit=10,
        sources=None,
        pain_filter=None,
        require_quotes=False,
        recency_column="enriched_at",
    ):
        self.calls.append(("quote_evidence", vendor_name))
        return [{"vendor_name": vendor_name, "urgency": min_urgency}]


@pytest.fixture(autouse=True)
def reset_configured_port():
    yield
    module = sys.modules.get(_PORT_MODULE)
    if module is not None:
        module.configure_vendor_briefing_intelligence_port(None)


def test_vendor_briefing_port_fails_closed_in_standalone(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    _reset_modules()

    module = importlib.import_module(_PORT_MODULE)

    with pytest.raises(module.VendorBriefingIntelligencePortNotConfigured):
        module.get_vendor_briefing_intelligence_port()

    assert _EXTRACTED_SHARED_MODULE not in sys.modules
    assert _ATLAS_SHARED_MODULE not in sys.modules


@pytest.mark.asyncio
async def test_vendor_briefing_uses_configured_intelligence_port(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    _reset_modules()

    port_module = importlib.import_module(_PORT_MODULE)
    port = RecordingPort()
    port_module.configure_vendor_briefing_intelligence_port(port)

    vendor_briefing = importlib.import_module(_VENDOR_BRIEFING_MODULE)
    monkeypatch.setattr(
        vendor_briefing.settings,
        "b2b_churn",
        SimpleNamespace(intelligence_window_days=30),
    )

    pool = object()

    assert vendor_briefing._reasoning_int({"value": "7"}) == 42
    assert vendor_briefing._timing_summary_payload({})[0] == "timing"
    assert vendor_briefing._align_vendor_intelligence_record_to_scorecard(
        {"vendor_name": "Acme"},
        {"vendor_name": "Acme"},
    )[1]["matched_vendor_count"] == 1
    assert await vendor_briefing._fetch_vendor_evidence_vault(pool, "Acme") == {
        "vendor_name": "Acme",
        "vault": {},
    }
    assert await vendor_briefing._fetch_vendor_evidence_record(pool, "Acme") == {
        "vendor_name": "Acme",
        "as_of": date.today().isoformat(),
    }
    assert await vendor_briefing._fetch_churn_signals(pool, "Acme") == {
        "vendor_name": "Acme",
        "score": 9,
    }
    assert await port_module.read_vendor_quote_evidence(
        pool,
        vendor_name="Acme",
        min_urgency=7.0,
    ) == [{"vendor_name": "Acme", "urgency": 7.0}]

    assert _EXTRACTED_SHARED_MODULE not in sys.modules
    assert _ATLAS_SHARED_MODULE not in sys.modules
