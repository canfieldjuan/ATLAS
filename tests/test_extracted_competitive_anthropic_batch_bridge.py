from __future__ import annotations

import importlib
import sys

import pytest


_COMP_INTEL_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"
_LLM_INFRA_ENV_VAR = "EXTRACTED_LLM_INFRA_STANDALONE"
_BRIDGE_MODULE = "extracted_competitive_intelligence.services.b2b.anthropic_batch"
_ATLAS_MODULE = "atlas_brain.services.b2b.anthropic_batch"


def _reset_modules() -> None:
    for module_name in (
        _BRIDGE_MODULE,
        _ATLAS_MODULE,
        "extracted_llm_infrastructure.services.b2b.anthropic_batch",
    ):
        sys.modules.pop(module_name, None)


@pytest.fixture
def standalone_batch(monkeypatch) -> None:
    monkeypatch.setenv(_COMP_INTEL_ENV_VAR, "1")
    monkeypatch.setenv(_LLM_INFRA_ENV_VAR, "1")
    _reset_modules()
    try:
        yield
    finally:
        _reset_modules()


def test_competitive_anthropic_batch_uses_extracted_llm_infrastructure(
    standalone_batch: None,
) -> None:
    module = importlib.import_module(_BRIDGE_MODULE)

    item = module.AnthropicBatchItem(
        custom_id="battle-card:acme",
        artifact_type="battle_card",
        artifact_id="Acme",
        messages=[{"role": "user", "content": "Build seller copy."}],
        max_tokens=256,
        temperature=0.2,
        vendor_name="Acme",
    )

    assert module.AnthropicBatchItem.__module__ == (
        "extracted_llm_infrastructure.services.b2b.anthropic_batch"
    )
    assert module.run_anthropic_message_batch.__module__ == (
        "extracted_llm_infrastructure.services.b2b.anthropic_batch"
    )
    assert module.mark_batch_fallback_result.__module__ == (
        "extracted_llm_infrastructure.services.b2b.anthropic_batch"
    )
    assert _ATLAS_MODULE not in sys.modules
    assert item.custom_id == "battle-card:acme"
    assert item.artifact_type == "battle_card"
    assert item.vendor_name == "Acme"
