"""
Pipecat integration for Atlas voice pipeline.

This module provides custom Pipecat services for local models:
- ParakeetSTTService: NVIDIA Parakeet-TDT for fast STT
- KokoroTTSService: Kokoro for fast TTS
- FunctionGemmaRouter: Fast tool routing (bypass LLM for tool queries)
- Uses Pipecat's built-in OllamaLLMService for LLM

Target: Sub-200ms voice-to-voice latency for tool queries.
"""

from .stt import ParakeetSTTService
from .tts import KokoroTTSService
from .pipeline import create_voice_pipeline
from .router import FunctionGemmaRouter, ToolRouterProcessor, route_query

__all__ = [
    "ParakeetSTTService",
    "KokoroTTSService",
    "create_voice_pipeline",
    "FunctionGemmaRouter",
    "ToolRouterProcessor",
    "route_query",
]
