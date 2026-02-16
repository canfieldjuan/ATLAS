"""Per-workflow LLM routing.

Holds a cloud LLM singleton alongside the local LLM in llm_registry.
Routes workflow types to the appropriate backend.

Routing map:
    LOCAL  (Ollama qwen3:14b): conversation, reminder, calendar, intent
    CLOUD  (Ollama cloud model): booking, email, security escalation
    NO LLM (unchanged): security workflow, presence workflow
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .protocols import LLMService

logger = logging.getLogger("atlas.services.llm_router")

# Cloud LLM singleton -- initialized at startup
_cloud_llm: Optional[LLMService] = None

# Workflows that require cloud reasoning
CLOUD_WORKFLOWS = frozenset({"booking", "email"})


def init_cloud_llm(
    model: str = "glm-5:cloud",
    base_url: str = "http://localhost:11434",
) -> Optional[LLMService]:
    """Initialize the cloud LLM singleton via Ollama. Called from main.py lifespan."""
    global _cloud_llm
    from .llm.ollama import OllamaLLM

    try:
        _cloud_llm = OllamaLLM(model=model, base_url=base_url)
        _cloud_llm.load()
        logger.info("Cloud LLM initialized: %s via %s", model, base_url)
        return _cloud_llm
    except Exception as e:
        logger.error("Failed to initialize cloud LLM: %s", e)
        return None


def shutdown_cloud_llm() -> None:
    """Unload the cloud LLM. Called from main.py shutdown."""
    global _cloud_llm
    if _cloud_llm:
        _cloud_llm.unload()
        _cloud_llm = None
        logger.info("Cloud LLM shut down")


def get_cloud_llm() -> Optional[LLMService]:
    """Get the cloud LLM instance (or None if not loaded)."""
    return _cloud_llm


def get_llm(workflow_type: Optional[str] = None) -> Optional[LLMService]:
    """Get the right LLM for a workflow type.

    Returns cloud LLM for business workflows, local for everything else.
    Falls back to local if cloud is unavailable.
    """
    from . import llm_registry

    if workflow_type and workflow_type in CLOUD_WORKFLOWS and _cloud_llm:
        return _cloud_llm

    # Default: local from registry
    return llm_registry.get_active()
