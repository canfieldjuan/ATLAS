"""
Context aggregation for Atlas Brain.

Provides runtime context tracking for:
- Who's in the room (face IDs, speaker IDs)
- What's visible (objects, scenes)
- Recent audio events
- Device states
- Conversation history

Note: The old orchestration pipeline has been replaced by Pipecat.
See atlas_brain/pipecat/ for the voice pipeline.
"""

from .context import ContextAggregator, get_context

__all__ = [
    "ContextAggregator",
    "get_context",
]
