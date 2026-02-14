"""LLM service implementations."""

from .llama_cpp import LlamaCppLLM
from .ollama import OllamaLLM
from .together import TogetherLLM
from .groq import GroqLLM
from .cloud import CloudLLM
from .hybrid import HybridLLM

# Import transformers flash backend (optional - requires transformers)
try:
    from .transformers_flash import TransformersFlashLLM
    __all__ = [
        "LlamaCppLLM", "OllamaLLM", "TogetherLLM",
        "GroqLLM", "CloudLLM", "HybridLLM", "TransformersFlashLLM",
    ]
except ImportError:
    __all__ = [
        "LlamaCppLLM", "OllamaLLM", "TogetherLLM",
        "GroqLLM", "CloudLLM", "HybridLLM",
    ]
