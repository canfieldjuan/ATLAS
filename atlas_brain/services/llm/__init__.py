"""LLM service implementations."""

from .llama_cpp import LlamaCppLLM
from .ollama import OllamaLLM
from .together import TogetherLLM
from .groq import GroqLLM
from .vllm import VLLMLLM
from .openrouter import OpenRouterLLM
from .cloud import CloudLLM
from .hybrid import HybridLLM

# Import Anthropic backend (optional - requires anthropic SDK)
try:
    from .anthropic import AnthropicLLM
    _has_anthropic = True
except ImportError:
    _has_anthropic = False

# Import transformers flash backend (optional - requires transformers)
try:
    from .transformers_flash import TransformersFlashLLM
    _has_transformers = True
except ImportError:
    _has_transformers = False

__all__ = [
    "LlamaCppLLM", "OllamaLLM", "TogetherLLM",
    "GroqLLM", "VLLMLLM", "OpenRouterLLM", "CloudLLM", "HybridLLM",
]
if _has_anthropic:
    __all__.append("AnthropicLLM")
if _has_transformers:
    __all__.append("TransformersFlashLLM")
