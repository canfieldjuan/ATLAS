"""
AI model services for Atlas Brain.

This module provides:
- Protocol definitions for LLM services
- Service registries for runtime model management
- Concrete implementations (ollama, etc.)

Note: VLM (moondream) and VOS were removed — they are not used.
"""

from importlib import import_module
from typing import Any

__all__ = [
    # Protocols
    "LLMService",
    "ModelInfo",
    "InferenceMetrics",
    "Message",
    # Registries
    "llm_registry",
    # Decorators
    "register_llm",
    # Embedding
    "SentenceTransformerEmbedding",
    # Reminders
    "ReminderService",
    "get_reminder_service",
]


def __getattr__(name: str) -> Any:
    if name in {"InferenceMetrics", "LLMService", "Message", "ModelInfo"}:
        module = import_module(".protocols", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {"llm_registry", "register_llm"}:
        # Preserve the old public API side effect for callers that request the
        # registry directly, without forcing unrelated service submodules (for
        # example receivables) to import every LLM implementation.
        import_module(".llm", __name__)
        module = import_module(".registry", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "SentenceTransformerEmbedding":
        module = import_module(".embedding", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {"ReminderService", "get_reminder_service"}:
        module = import_module(".reminders", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
