"""Per-workflow LLM routing.

Holds a cloud LLM singleton alongside the local LLM in llm_registry.
Routes workflow types to the appropriate backend.

Routing map:
    LOCAL     (Ollama qwen3:14b): conversation, reminder, calendar, intent
    CLOUD     (Ollama cloud minimax-m2): booking, email, security escalation
    DRAFT     (Anthropic Sonnet): email_draft
    TRIAGE    (Anthropic Haiku): email_triage, email_query
    REASONING (Anthropic Sonnet): reasoning agent deep analysis
    NO LLM    (unchanged): security workflow, presence workflow
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .protocols import LLMService

logger = logging.getLogger("atlas.services.llm_router")

# Cloud LLM singleton -- initialized at startup
_cloud_llm: Optional[LLMService] = None

# Draft LLM singleton -- Anthropic for email draft generation
_draft_llm: Optional[LLMService] = None

# Triage LLM singleton -- Anthropic Haiku for cheap email replyable classification
_triage_llm: Optional[LLMService] = None

# Reasoning LLM singleton -- Anthropic Sonnet for cross-domain reasoning agent
_reasoning_llm: Optional[LLMService] = None

# Workflows that require cloud reasoning
CLOUD_WORKFLOWS = frozenset({"booking", "email"})

# Workflows that use the draft LLM (Anthropic)
DRAFT_WORKFLOWS = frozenset({"email_draft"})

# Workflows that use the triage LLM (Anthropic Haiku)
TRIAGE_WORKFLOWS = frozenset({"email_triage", "email_query"})

# Workflows that use the reasoning LLM (Anthropic Sonnet)
REASONING_WORKFLOWS = frozenset({"reasoning"})


def init_cloud_llm(
    model: str = "minimax-m2:cloud",
    base_url: str = "http://localhost:11434",
    think: bool = True,
) -> Optional[LLMService]:
    """Initialize the cloud LLM singleton via Ollama. Called from main.py lifespan."""
    global _cloud_llm
    from .llm.ollama import OllamaLLM

    try:
        _cloud_llm = OllamaLLM(model=model, base_url=base_url, think=think)
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


def init_draft_llm(
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
) -> Optional[LLMService]:
    """Initialize the draft LLM singleton (Anthropic). Called from main.py lifespan."""
    global _draft_llm
    from .llm.anthropic import AnthropicLLM

    try:
        _draft_llm = AnthropicLLM(model=model, api_key=api_key)
        _draft_llm.load()
        logger.info("Draft LLM initialized: %s (Anthropic)", model)
        return _draft_llm
    except Exception as e:
        logger.error("Failed to initialize draft LLM: %s", e)
        return None


def shutdown_draft_llm() -> None:
    """Unload the draft LLM. Called from main.py shutdown."""
    global _draft_llm
    if _draft_llm:
        _draft_llm.unload()
        _draft_llm = None
        logger.info("Draft LLM shut down")


def get_draft_llm() -> Optional[LLMService]:
    """Get the draft LLM instance (or None if not loaded)."""
    return _draft_llm


def init_triage_llm(
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
) -> Optional[LLMService]:
    """Initialize the triage LLM singleton (Anthropic Haiku). Called from main.py lifespan."""
    global _triage_llm
    from .llm.anthropic import AnthropicLLM

    try:
        _triage_llm = AnthropicLLM(model=model, api_key=api_key)
        _triage_llm.load()
        logger.info("Triage LLM initialized: %s (Anthropic)", model)
        return _triage_llm
    except Exception as e:
        logger.error("Failed to initialize triage LLM: %s", e)
        return None


def shutdown_triage_llm() -> None:
    """Unload the triage LLM. Called from main.py shutdown."""
    global _triage_llm
    if _triage_llm:
        _triage_llm.unload()
        _triage_llm = None
        logger.info("Triage LLM shut down")


def get_triage_llm() -> Optional[LLMService]:
    """Get the triage LLM instance (or None if not loaded)."""
    return _triage_llm


def init_reasoning_llm(
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
) -> Optional[LLMService]:
    """Initialize the reasoning LLM singleton (Anthropic). Called from main.py lifespan."""
    global _reasoning_llm
    from .llm.anthropic import AnthropicLLM

    try:
        _reasoning_llm = AnthropicLLM(model=model, api_key=api_key)
        _reasoning_llm.load()
        logger.info("Reasoning LLM initialized: %s (Anthropic)", model)
        return _reasoning_llm
    except Exception as e:
        logger.error("Failed to initialize reasoning LLM: %s", e)
        return None


def shutdown_reasoning_llm() -> None:
    """Unload the reasoning LLM. Called from main.py shutdown."""
    global _reasoning_llm
    if _reasoning_llm:
        _reasoning_llm.unload()
        _reasoning_llm = None
        logger.info("Reasoning LLM shut down")


def get_reasoning_llm() -> Optional[LLMService]:
    """Get the reasoning LLM instance (or None if not loaded)."""
    return _reasoning_llm


def get_llm(workflow_type: Optional[str] = None) -> Optional[LLMService]:
    """Get the right LLM for a workflow type.

    Returns cloud LLM for business workflows, local for everything else.
    Falls back to local if cloud is unavailable.
    """
    from . import llm_registry

    if workflow_type and workflow_type in REASONING_WORKFLOWS:
        if _reasoning_llm:
            return _reasoning_llm
        logger.warning(
            "Reasoning LLM not initialized; falling back to local for workflow '%s'",
            workflow_type,
        )

    if workflow_type and workflow_type in TRIAGE_WORKFLOWS:
        if _triage_llm:
            return _triage_llm
        logger.warning(
            "Triage LLM not initialized; falling back to local for workflow '%s'",
            workflow_type,
        )

    if workflow_type and workflow_type in DRAFT_WORKFLOWS:
        if _draft_llm:
            return _draft_llm
        logger.warning(
            "Draft LLM not initialized; falling back to local for workflow '%s'",
            workflow_type,
        )

    if workflow_type and workflow_type in CLOUD_WORKFLOWS:
        if _cloud_llm:
            return _cloud_llm
        logger.warning(
            "Cloud LLM not initialized; falling back to local for workflow '%s'",
            workflow_type,
        )

    # Default: local from registry
    return llm_registry.get_active()
