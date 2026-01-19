"""
Pipecat integration for Atlas voice pipeline.

This module provides custom Pipecat services for local models:
- ParakeetSTTService: NVIDIA Parakeet-TDT for fast STT
- KokoroTTSService: Kokoro for fast TTS
- FunctionGemmaRouter: Fast tool routing (bypass LLM for simple queries)
- GptOssToolService: Multi-turn tool calling for complex queries
- Uses Pipecat's built-in OllamaLLMService for LLM

Two-tier architecture:
- Simple queries: FunctionGemma (~120ms) + direct execution (~80ms) = ~200ms
- Complex queries: gpt-oss:20b multi-turn tool calling (~1.5s)
"""

from .stt import ParakeetSTTService
from .tts import KokoroTTSService
from .pipeline import create_voice_pipeline
from .router import (
    FunctionGemmaRouter,
    ToolRouterProcessor,
    ProcessedQueryResult,
    is_complex_query,
    route_query,
)
from .llm import GptOssToolService, ToolCallResult, get_gptoss_service, process_complex_query

__all__ = [
    # STT/TTS
    "ParakeetSTTService",
    "KokoroTTSService",
    # Pipeline
    "create_voice_pipeline",
    # Router (fast path)
    "FunctionGemmaRouter",
    "ToolRouterProcessor",
    "ProcessedQueryResult",
    "is_complex_query",
    "route_query",
    # LLM tool calling (complex path)
    "GptOssToolService",
    "ToolCallResult",
    "get_gptoss_service",
    "process_complex_query",
]
