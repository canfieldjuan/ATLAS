"""
Orchestration layer for Atlas Brain.

Manages the complete pipeline from audio input to device action:
- Wake word detection
- Voice activity detection
- Speech-to-text
- Intent parsing / LLM reasoning
- Action execution
- Text-to-speech response
"""

from .audio_buffer import AudioBuffer, VADConfig
from .context import ContextAggregator, get_context
from .orchestrator import Orchestrator, OrchestratorConfig, OrchestratorResult
from .states import PipelineContext, PipelineEvent, PipelineState, PipelineStateMachine
from .wake_word import WakeWordConfig, WakeWordManager

__all__ = [
    # Core orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    # State machine
    "PipelineState",
    "PipelineEvent",
    "PipelineContext",
    "PipelineStateMachine",
    # Audio
    "AudioBuffer",
    "VADConfig",
    # Wake word
    "WakeWordManager",
    "WakeWordConfig",
    # Context
    "ContextAggregator",
    "get_context",
]
