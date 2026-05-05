from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


_COMP_INTEL_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"
_LLM_INFRA_ENV_VAR = "EXTRACTED_LLM_INFRA_STANDALONE"
_MODULE = "extracted_competitive_intelligence.autonomous.tasks._b2b_batch_utils"
_ATLAS_MODULE = "atlas_brain.autonomous.tasks._b2b_batch_utils"


def _drop_package_attr(package_name: str, attr_name: str) -> None:
    package = sys.modules.get(package_name)
    if package is not None and hasattr(package, attr_name):
        delattr(package, attr_name)


def _reset_modules() -> None:
    for module_name in (
        _MODULE,
        _ATLAS_MODULE,
        "extracted_competitive_intelligence.pipelines.llm",
        "extracted_llm_infrastructure.services.registry",
    ):
        sys.modules.pop(module_name, None)
    _drop_package_attr("extracted_competitive_intelligence.pipelines", "llm")
    _drop_package_attr("extracted_llm_infrastructure.services", "registry")


def test_competitive_batch_utils_import_without_atlas(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    _reset_modules()

    module = importlib.import_module(_MODULE)

    assert module.__name__ == _MODULE
    assert _ATLAS_MODULE not in sys.modules
    assert module.anthropic_batch_requested(
        SimpleNamespace(metadata={"anthropic_batch_enabled": "yes", "task_enabled": 1}),
        global_default=False,
        task_default=False,
        task_keys=("task_enabled",),
    ) is True
    assert module.anthropic_batch_min_items(
        SimpleNamespace(metadata={"min_items": "0"}),
        default=5,
        keys=("min_items",),
        min_value=2,
    ) == 2
    assert module.is_anthropic_llm(
        SimpleNamespace(name="anthropic", model="claude-sonnet-4-5")
    ) is True


def test_competitive_batch_utils_resolves_standalone_registry_slot(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    monkeypatch.setenv("EXTRACTED_LLM_ANTHROPIC_API_KEY", "product-key")
    _reset_modules()

    module = importlib.import_module(_MODULE)

    llm_mod = importlib.import_module("extracted_competitive_intelligence.pipelines.llm")
    registry_mod = importlib.import_module("extracted_llm_infrastructure.services.registry")

    activation: dict[str, object] = {}

    class Registry:
        def get_slot(self, slot_name):
            activation["checked"] = slot_name
            return None

        def activate_slot(self, slot_name, provider, **kwargs):
            activation.update(
                {
                    "slot_name": slot_name,
                    "provider": provider,
                    "model": kwargs.get("model"),
                    "api_key": kwargs.get("api_key"),
                }
            )
            return SimpleNamespace(name=provider, model=kwargs.get("model"))

    monkeypatch.setattr(llm_mod, "get_pipeline_llm", lambda **_kwargs: None)
    monkeypatch.setattr(registry_mod, "llm_registry", Registry())

    resolved = module.resolve_anthropic_batch_llm(
        target_model_candidates=("anthropic/claude-sonnet-4-5",),
    )

    assert resolved.name == "anthropic"
    assert resolved.model == "claude-sonnet-4-5"
    assert activation == {
        "checked": "b2b_batch_anthropic::claude-sonnet-4-5",
        "slot_name": "b2b_batch_anthropic::claude-sonnet-4-5",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_key": "product-key",
    }
    assert _ATLAS_MODULE not in sys.modules
