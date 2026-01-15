"""
Frame buffer for efficient frame storage and retrieval.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("atlas.vision.processing.frame_buffer")


@dataclass
class Frame:
    """Single video frame with metadata."""

    data: np.ndarray
    timestamp: float
    source_id: str
    frame_id: int
    width: int = 0
    height: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.data is not None and len(self.data.shape) >= 2:
            self.height, self.width = self.data.shape[:2]


class FrameBuffer:
    """Thread-safe ring buffer for video frames."""

    def __init__(self, max_size: int = 30):
        self._max_size = max_size
        self._buffers: dict[str, deque[Frame]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._frame_counters: dict[str, int] = {}

    def _get_lock(self, source_id: str) -> asyncio.Lock:
        """Get or create lock for a source."""
        if source_id not in self._locks:
            self._locks[source_id] = asyncio.Lock()
        return self._locks[source_id]

    def _get_buffer(self, source_id: str) -> deque[Frame]:
        """Get or create buffer for a source."""
        if source_id not in self._buffers:
            self._buffers[source_id] = deque(maxlen=self._max_size)
            self._frame_counters[source_id] = 0
        return self._buffers[source_id]

    async def push(self, source_id: str, data: np.ndarray, metadata: Optional[dict] = None) -> Frame:
        """Push a new frame to the buffer."""
        lock = self._get_lock(source_id)

        async with lock:
            buffer = self._get_buffer(source_id)
            self._frame_counters[source_id] += 1

            frame = Frame(
                data=data,
                timestamp=time.time(),
                source_id=source_id,
                frame_id=self._frame_counters[source_id],
                metadata=metadata or {},
            )

            buffer.append(frame)
            return frame

    async def get_latest(self, source_id: str) -> Optional[Frame]:
        """Get the most recent frame from a source."""
        lock = self._get_lock(source_id)

        async with lock:
            buffer = self._get_buffer(source_id)
            if buffer:
                return buffer[-1]
            return None

    async def get_recent(self, source_id: str, count: int = 5) -> list[Frame]:
        """Get recent frames from a source."""
        lock = self._get_lock(source_id)

        async with lock:
            buffer = self._get_buffer(source_id)
            return list(buffer)[-count:]

    async def get_all_latest(self) -> dict[str, Optional[Frame]]:
        """Get latest frame from all sources."""
        result = {}
        for source_id in self._buffers:
            result[source_id] = await self.get_latest(source_id)
        return result

    def get_sources(self) -> list[str]:
        """Get list of all source IDs."""
        return list(self._buffers.keys())

    async def clear(self, source_id: Optional[str] = None) -> None:
        """Clear buffer for a source or all sources."""
        if source_id:
            lock = self._get_lock(source_id)
            async with lock:
                if source_id in self._buffers:
                    self._buffers[source_id].clear()
        else:
            for sid in list(self._buffers.keys()):
                await self.clear(sid)


# Global frame buffer instance
_frame_buffer: Optional[FrameBuffer] = None


def get_frame_buffer() -> FrameBuffer:
    """Get or create global frame buffer."""
    global _frame_buffer
    if _frame_buffer is None:
        _frame_buffer = FrameBuffer()
    return _frame_buffer
