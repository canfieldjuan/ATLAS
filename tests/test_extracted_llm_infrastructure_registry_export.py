from __future__ import annotations

import importlib
import sys


_LLM_INFRA_ENV_VAR = "EXTRACTED_LLM_INFRA_STANDALONE"


def _drop_package_attr(package_name: str, attr_name: str) -> None:
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attr_name):
        delattr(package, attr_name)


def _reset_modules() -> None:
    for module_name in (
        "extracted_llm_infrastructure._standalone.registry",
        "extracted_llm_infrastructure.services",
        "extracted_llm_infrastructure.services.registry",
        "extracted_llm_infrastructure.pipelines.llm",
        "atlas_brain.services.registry",
    ):
        sys.modules.pop(module_name, None)
    _drop_package_attr("extracted_llm_infrastructure", "services")
    _drop_package_attr("extracted_llm_infrastructure.pipelines", "llm")


def test_services_namespace_exports_standalone_llm_registry(monkeypatch) -> None:
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    _reset_modules()

    services = importlib.import_module("extracted_llm_infrastructure.services")
    registry = importlib.import_module(
        "extracted_llm_infrastructure.services.registry"
    )

    assert services.llm_registry is registry.llm_registry
    assert services.register_llm is registry.register_llm


def test_pipeline_local_fast_resolver_uses_services_registry_export(
    monkeypatch,
) -> None:
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    _reset_modules()

    llm = importlib.import_module("extracted_llm_infrastructure.pipelines.llm")

    assert (
        llm.get_pipeline_llm(
            workload="local_fast",
            auto_activate_ollama=False,
        )
        is None
    )
    assert "atlas_brain.services.registry" not in sys.modules
