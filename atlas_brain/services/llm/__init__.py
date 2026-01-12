"""LLM service implementations."""

from .llama_cpp import LlamaCppLLM

# Import transformers flash backend (optional - requires transformers)
try:
    from .transformers_flash import TransformersFlashLLM
    __all__ = ["LlamaCppLLM", "TransformersFlashLLM"]
except ImportError:
    __all__ = ["LlamaCppLLM"]
