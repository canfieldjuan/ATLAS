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
        _VENDOR_BRIEFING_MODULE,
        _EXTRACTED_SHARED_MODULE,
        _ATLAS_SHARED_MODULE,
        _EXTRACTED_SYNTHESIS_MODULE,
        _ATLAS_SYNTHESIS_MODULE,
    ):
        sys.modules.pop(module_name, None)
    _drop_package_attr("extracted_competitive_intelligence.services.b2b", "vendor_briefing_ports")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "b2b_vendor_briefing")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "_b2b_shared")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "_b2b_synthesis_reader")
    _drop_package_attr("atlas_brain.autonomous.tasks", "_b2b_shared")
    _drop_package_attr("atlas_brain.autonomous.tasks", "_b2b_synthesis_reader")


class FakeSynthesisView:
    def __init__(
        self,
        vendor_name: str,
        *,
        schema_version: str = "synthesis_v2",
        as_of_date: date | None = None,
    ) -> None:
        self.vendor_name = vendor_name
        self.schema_version = schema_version
        self.as_of_date = as_of_date
        self.primary_wedge = None
        self.wedge_label = ""
        self.meta = {"evidence_window_start": "2026-04-01", "evidence_window_end": "2026-05-01"}
        self.why_they_stay = ["workflow inertia"]
        self.confidence_posture = {"limits": ["limited sample"]}
        self.switch_triggers = ["renewal"]
        self.coverage_gaps = ["pricing"]

    def filtered_consumer_context(self, consumer: str) -> dict:
        return {
            "reasoning_contracts": {"schema_version": self.schema_version},
            "vendor_core_reasoning": {
                "timing_intelligence": {"renewal_windows": ["Q2"]}
            },
            "displacement_reasoning": {"competitive_reframes": ["service"]},
            "account_reasoning": {},
            "anchor_examples": {"timing": ["renewal mention"]},
            "reference_ids": {"timing": ["ref-1"]},
        }

    def materialized_contracts(self) -> dict:
        return {
            "vendor_core_reasoning": {
                "timing_intelligence": {"renewal_windows": ["Q2"]}
            },
            "schema_version": self.schema_version,
        }


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

    def inject_synthesis_freshness(self, entry, view, *, requested_as_of=None):
        self.calls.append(("inject_synthesis_freshness", view.vendor_name))
        entry["data_as_of_date"] = (
            view.as_of_date.isoformat() if view.as_of_date else "2026-05-01"
        )
        entry["data_stale"] = False

    def load_synthesis_view(
        self,
        raw,
        vendor_name,
        schema_version="",
        as_of_date=None,
    ):
        self.calls.append(("load_synthesis_view", vendor_name))
        return FakeSynthesisView(
            vendor_name,
            schema_version=schema_version or "synthesis_v2",
            as_of_date=as_of_date if isinstance(as_of_date, date) else date(2026, 5, 1),
        )

    async def load_best_reasoning_view(
        self,
        pool,
        vendor_name,
        *,
        as_of=None,
        analysis_window_days=30,
    ):
        self.calls.append(("load_best_reasoning_view", vendor_name))
        return FakeSynthesisView(vendor_name, as_of_date=as_of or date(2026, 5, 1))

    async def load_prior_reasoning_snapshots(
        self,
        pool,
        vendor_names,
        *,
        before_date=None,
        analysis_window_days=30,
    ):
        self.calls.append(("load_prior_reasoning_snapshots", ",".join(vendor_names)))
        return {vendor_names[0]: {"archetype": "legacy"}}

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
    assert _EXTRACTED_SYNTHESIS_MODULE not in sys.modules
    assert _ATLAS_SYNTHESIS_MODULE not in sys.modules


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
    briefing = {"vendor": "Acme", "data_sources": {}}
    assert vendor_briefing._apply_reasoning_synthesis_to_briefing(
        briefing,
        {"vendor": "Acme", "schema_version": "synthesis_v2"},
    )
    assert briefing["data_as_of_date"] == "2026-05-01"
    assert briefing["reasoning_source"] == "b2b_reasoning_synthesis"

    reasoning = await vendor_briefing._fetch_reasoning_synthesis(pool, "Acme")
    assert reasoning is not None
    assert reasoning["synthesis_schema_version"] == "synthesis_v2"
    assert reasoning["data_as_of_date"] == "2026-05-01"
    assert await port_module.load_prior_reasoning_snapshots(pool, ["Acme"]) == {
        "Acme": {"archetype": "legacy"}
    }

    assert _EXTRACTED_SHARED_MODULE not in sys.modules
    assert _ATLAS_SHARED_MODULE not in sys.modules
    assert _EXTRACTED_SYNTHESIS_MODULE not in sys.modules
    assert _ATLAS_SYNTHESIS_MODULE not in sys.modules
