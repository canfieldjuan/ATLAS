"""
Playback controller for voice pipeline.

Manages TTS playback in a background thread with stop support.
"""

import threading
from typing import Callable, Optional, Protocol


class SpeechEngine(Protocol):
    """Protocol for TTS engines."""

    def speak(self, text: str) -> None:
        """Speak the given text."""
        ...

    def stop(self) -> None:
        """Stop current playback."""
        ...


class PlaybackController:
    """Runs TTS playback in a background thread with stop support."""

    def __init__(self, engine: SpeechEngine):
        self.engine = engine
        self._thread: Optional[threading.Thread] = None
        self.speaking = threading.Event()
        self._lock = threading.Lock()

    def speak(
        self,
        text: str,
        on_start: Optional[Callable[[], None]] = None,
        on_done: Optional[Callable[[], None]] = None,
    ):
        """
        Speak text in background thread.

        Args:
            text: Text to speak
            on_start: Callback when speech starts
            on_done: Callback when speech ends
        """
        def runner():
            self.speaking.set()
            try:
                if on_start:
                    on_start()
                self.engine.speak(text)
            finally:
                if on_done:
                    on_done()
                self.speaking.clear()

        with self._lock:
            self._stop_locked()
            thread = threading.Thread(target=runner, daemon=True)
            self._thread = thread
            thread.start()

    def stop(self):
        """Stop current playback."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        """Internal stop without lock."""
        self.engine.stop()
        self._thread = None
        self.speaking.clear()
