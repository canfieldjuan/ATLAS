"""Shared LLM helpers for stratified reasoning."""

from __future__ import annotations

import logging
import os
from typing import Any

from ..pipelines.llm import get_pipeline_llm

logger = logging.getLogger("atlas.reasoning.llm_utils")

REASONING_CONCLUSION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "archetype",
        "secondary_archetype",
        "confidence",
        "risk_level",
        "executive_summary",
        "key_signals",
        "falsification_conditions",
        "uncertainty_sources",
    ],
    "properties": {
        "archetype": {
            "type": "string",
            "enum": [
                "pricing_shock",
                "feature_gap",
                "acquisition_decay",
                "leadership_redesign",
                "integration_break",
                "support_collapse",
                "category_disruption",
                "compliance_gap",
                "mixed",
                "stable",
            ],
        },
        "secondary_archetype": {
            "type": ["string", "null"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "executive_summary": {
            "type": "string",
        },
        "key_signals": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "falsification_conditions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "uncertainty_sources": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


def resolve_stratified_llm(cfg: Any) -> Any:
    """Resolve the configured LLM for stratified reasoning.

    Workload options:
    - ``openrouter``: OpenRouter with configured model (default: openai/o4-mini)
    - ``vllm``: local vLLM server
    - ``anthropic``: configured Anthropic model
    - ``auto``: try Anthropic, then reasoning router, then pipeline fallback
    """
    from ..services import llm_registry

    workload = (cfg.stratified_llm_workload or "openrouter").strip().lower()

    if workload == "openrouter":
        return _activate_openrouter(cfg, llm_registry)

    if workload == "vllm":
        return get_pipeline_llm(workload="vllm", auto_activate_ollama=False)

    if workload == "anthropic":
        llm = _activate_configured_anthropic(cfg, llm_registry)
        if llm is not None:
            return llm
        return get_pipeline_llm(workload="anthropic", auto_activate_ollama=False)

    if workload == "auto":
        llm = _activate_configured_anthropic(cfg, llm_registry)
        if llm is not None:
            return llm
        from ..services.llm_router import get_reasoning_llm
        llm = get_reasoning_llm()
        if llm is not None and _matches_model(llm, cfg.model):
            return llm
        return get_pipeline_llm(workload="reasoning", auto_activate_ollama=False)

    return get_pipeline_llm(workload="reasoning", auto_activate_ollama=False)


def _activate_openrouter(cfg: Any, llm_registry: Any) -> Any:
    """Activate OpenRouter with the configured stratified reasoning model."""
    api_key = _openrouter_api_key()
    if not api_key:
        logger.warning("No OpenRouter API key found, falling back to pipeline LLM")
        return get_pipeline_llm(workload="reasoning", auto_activate_ollama=False)

    model = cfg.stratified_openrouter_model or "openai/o4-mini"

    # Check if already active with the right model
    active = llm_registry.get_active()
    if (
        active is not None
        and getattr(active, "name", "") == "openrouter"
        and _matches_model(active, model)
    ):
        return active

    try:
        llm_registry.activate("openrouter", model=model, api_key=api_key)
        llm = llm_registry.get_active()
        logger.info("Activated OpenRouter for stratified reasoning: %s", model)
        return llm
    except Exception:
        logger.warning("Failed to activate OpenRouter (%s), falling back", model, exc_info=True)
        return get_pipeline_llm(workload="reasoning", auto_activate_ollama=False)


def _activate_configured_anthropic(cfg: Any, llm_registry: Any) -> Any:
    api_key = _anthropic_api_key()
    if not api_key:
        return None

    active = llm_registry.get_active()
    if active is not None and getattr(active, "name", "") == "anthropic" and _matches_model(active, cfg.model):
        return active

    try:
        llm_registry.activate("anthropic", model=cfg.model, api_key=api_key)
        return llm_registry.get_active()
    except Exception:
        return None


def _openrouter_api_key() -> str:
    from ..config import settings
    return (
        settings.b2b_churn.openrouter_api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ATLAS_B2B_CHURN_OPENROUTER_API_KEY", "")
    )


def _anthropic_api_key() -> str:
    from ..config import settings
    return (
        settings.llm.anthropic_api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ATLAS_LLM_ANTHROPIC_API_KEY", "")
    )


def _matches_model(llm: Any, expected_model: str) -> bool:
    if not expected_model:
        return True
    return expected_model in {
        getattr(llm, "model", ""),
        getattr(llm, "model_id", ""),
    }
