"""
Command executor for voice pipeline.

Runs command handling on a thread pool to keep audio capture responsive.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable


class CommandExecutor:
    """Runs command handling on a thread pool."""

    def __init__(self, handler: Callable[[bytes], None], max_workers: int):
        self.handler = handler
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, payload: bytes):
        """Submit audio payload for processing."""
        self.executor.submit(self.handler, payload)

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False, cancel_futures=True)
