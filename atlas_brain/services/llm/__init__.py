"""LLM service implementations."""

from .llama_cpp import LlamaCppLLM
from .ollama import OllamaLLM
from .together import TogetherLLM

# Import transformers flash backend (optional - requires transformers)
try:
    from .transformers_flash import TransformersFlashLLM
    __all__ = ["LlamaCppLLM", "OllamaLLM", "TogetherLLM", "TransformersFlashLLM"]
except ImportError:
    __all__ = ["LlamaCppLLM", "OllamaLLM", "TogetherLLM"]
