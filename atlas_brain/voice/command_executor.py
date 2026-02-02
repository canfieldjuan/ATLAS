"""
Command executor for voice pipeline.

Runs command handling on a thread pool to keep audio capture responsive.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

logger = logging.getLogger("atlas.voice.command_executor")


class CommandExecutor:
    """Runs command handling on a thread pool."""

    def __init__(
        self,
        handler: Callable[[bytes], None],
        max_workers: int,
        streaming_handler: Optional[Callable[[str, bytes], None]] = None,
    ):
        self.handler = handler
        self.streaming_handler = streaming_handler
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _on_done(self, future: Future) -> None:
        """Callback to log exceptions from completed futures."""
        exc = future.exception()
        if exc is not None:
            logger.error("Command handler failed: %s", exc, exc_info=exc)

    def submit(self, payload: bytes):
        """Submit audio payload for processing."""
        future = self.executor.submit(self.handler, payload)
        future.add_done_callback(self._on_done)

    def submit_streaming(self, transcript: str, audio_bytes: bytes):
        """Submit transcript and audio from streaming ASR for processing.

        Args:
            transcript: Final transcript text from streaming ASR
            audio_bytes: Raw PCM audio bytes for speaker verification
        """
        if self.streaming_handler is not None:
            future = self.executor.submit(self.streaming_handler, transcript, audio_bytes)
            future.add_done_callback(self._on_done)

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False, cancel_futures=True)
