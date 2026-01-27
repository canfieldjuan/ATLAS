"""
Context aggregation for Atlas Brain.

Provides:
- Runtime context tracking (faces, speakers, objects, devices)
- Global CUDA lock to prevent conflicts between services
"""

import asyncio
import logging

logger = logging.getLogger("atlas.orchestration")

from .context import ContextAggregator, get_context

# Global CUDA lock to prevent conflicts between vision services (YOLO)
cuda_lock = asyncio.Lock()

__all__ = [
    "ContextAggregator",
    "get_context",
    "cuda_lock",
]
