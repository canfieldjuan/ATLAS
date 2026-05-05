from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


_COMP_INTEL_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"
_LLM_INFRA_ENV_VAR = "EXTRACTED_LLM_INFRA_STANDALONE"
_ROUTER_MODULE = "extracted_competitive_intelligence.services.llm_router"
_INFRA_ROUTER_MODULE = "extracted_llm_infrastructure.services.llm_router"
_VENDOR_BRIEFING_MODULE = (
    "extracted_competitive_intelligence.autonomous.tasks.b2b_vendor_briefing"
)
_ATLAS_ROUTER_MODULE = "atlas_brain.services.llm_router"


def _drop_package_attr(package_name: str, attr_name: str) -> None:
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attr_name):
        delattr(package, attr_name)


def _reset_modules() -> None:
    for module_name in (
        _ROUTER_MODULE,
        _INFRA_ROUTER_MODULE,
        _VENDOR_BRIEFING_MODULE,
        _ATLAS_ROUTER_MODULE,
    ):
        sys.modules.pop(module_name, None)
    _drop_package_attr("extracted_competitive_intelligence.services", "llm_router")
    _drop_package_attr("extracted_llm_infrastructure.services", "llm_router")
    _drop_package_attr("extracted_competitive_intelligence.autonomous.tasks", "b2b_vendor_briefing")


def test_competitive_llm_router_uses_extracted_infrastructure_in_standalone(
    monkeypatch,
) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    _reset_modules()

    module = importlib.import_module(_ROUTER_MODULE)

    assert module.get_llm.__module__ == _INFRA_ROUTER_MODULE
    assert _ATLAS_ROUTER_MODULE not in sys.modules


def test_vendor_briefing_get_llm_uses_competitive_router(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    _reset_modules()

    router = importlib.import_module(_ROUTER_MODULE)
    expected = SimpleNamespace(name="anthropic", model="claude-test")
    monkeypatch.setattr(router, "get_llm", lambda workflow_type=None: expected)

    vendor_briefing = importlib.import_module(_VENDOR_BRIEFING_MODULE)

    assert vendor_briefing._get_llm() is expected
    assert _ATLAS_ROUTER_MODULE not in sys.modules
