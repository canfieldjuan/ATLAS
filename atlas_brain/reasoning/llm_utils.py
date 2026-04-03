"""Shared LLM helpers for legacy reasoning utilities and synthesis tasks."""

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
        "trend_direction",
        "displacement_net_direction",
        "displacement_winner",
        "executive_summary",
        "key_signals",
        "compound_signals",
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
        },
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "trend_direction": {
            "type": "string",
            "enum": ["accelerating", "stable", "decelerating", "unknown"],
        },
        "displacement_net_direction": {
            "type": "string",
            "enum": ["positive", "negative", "balanced", "insufficient_data"],
        },
        "displacement_winner": {
            "type": ["string", "null"],
        },
        "executive_summary": {
            "type": "string",
        },
        "key_signals": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "compound_signals": {
            "type": "array",
            "items": {"type": "string"},
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
