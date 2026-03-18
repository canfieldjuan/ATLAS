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


_SLOT_HEAVY = "reasoning_heavy"
_SLOT_LIGHT = "reasoning_light"


def resolve_stratified_llm(cfg: Any) -> Any:
    """Resolve the Tier 1 (heavy) LLM for stratified reasoning.

    Used for: archetype classification (Pass 1), pairwise battles.

    Uses a named registry slot so it does not evict the primary LLM
    used by voice, chat, and other subsystems.

    Workload options:
    - ``openrouter``: OpenRouter with ``stratified_openrouter_model``
    - ``vllm``: local vLLM server
    - ``anthropic``: configured Anthropic model
    - ``auto``: try Anthropic, then reasoning router, then pipeline fallback
    """
    from ..services import llm_registry

    workload = (cfg.stratified_llm_workload or "openrouter").strip().lower()

    if workload == "openrouter":
        return _activate_openrouter_slot(cfg, llm_registry, _SLOT_HEAVY)

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


def resolve_stratified_llm_light(cfg: Any) -> Any:
    """Resolve the Tier 2 (light) LLM for secondary reasoning tasks.

    Used for: challenge/ground passes, reconstitute, category councils,
    resource asymmetry.  Falls back to the heavy model if not configured.

    Uses a separate named registry slot so heavy and light models
    coexist without evicting each other or the primary system LLM.
    """
    from ..services import llm_registry

    workload = (cfg.stratified_llm_workload or "openrouter").strip().lower()
    if workload != "openrouter":
        return resolve_stratified_llm(cfg)

    light_model = getattr(cfg, "stratified_openrouter_model_light", "") or ""
    if not light_model:
        return resolve_stratified_llm(cfg)

    # If heavy and light are the same model, share one slot
    heavy_model = cfg.stratified_openrouter_model or ""
    if light_model == heavy_model:
        return resolve_stratified_llm(cfg)

    api_key = _openrouter_api_key()
    if not api_key:
        return resolve_stratified_llm(cfg)

    return _activate_openrouter_slot(cfg, llm_registry, _SLOT_LIGHT, model_override=light_model)


def _activate_openrouter_slot(
    cfg: Any, llm_registry: Any, slot_name: str, *, model_override: str = "",
) -> Any:
    """Ensure an OpenRouter model is loaded in a named registry slot.

    Reuses the existing slot instance when the model already matches.
    """
    api_key = _openrouter_api_key()
    if not api_key:
        logger.warning("No OpenRouter API key found, falling back to pipeline LLM")
        return get_pipeline_llm(workload="reasoning", auto_activate_ollama=False)

    model = model_override or cfg.stratified_openrouter_model or "openai/o4-mini"

    existing = llm_registry.get_slot(slot_name)
    if existing is not None and _matches_model(existing, model):
        return existing

    try:
        instance = llm_registry.activate_slot(
            slot_name, "openrouter", model=model, api_key=api_key,
        )
        logger.info("Reasoning slot '%s' -> %s", slot_name, model)
        return instance
    except Exception:
        logger.warning(
            "Failed to activate OpenRouter slot '%s' (%s), falling back",
            slot_name, model, exc_info=True,
        )
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
