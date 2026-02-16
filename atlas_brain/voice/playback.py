"""
Playback controller for voice pipeline.

Manages TTS playback in a background thread with stop support.
"""

import queue
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
        self._stream_stop = threading.Event()

    def speak(
        self,
        text: str,
        target_node: Optional[str] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_done: Optional[Callable[[], None]] = None,
    ):
        """
        Speak text in background thread.

        Args:
            text: Text to speak
            target_node: Node ID for remote routing (None=local, future use)
            on_start: Callback when speech starts
            on_done: Callback when speech ends
        """
        # target_node reserved for future remote node routing
        _ = target_node

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

    def speak_streamed(
        self,
        sentence_queue: queue.Queue,
        on_start: Optional[Callable[[], None]] = None,
        on_done: Optional[Callable[[], None]] = None,
    ):
        """Play sentences from a queue as they arrive.

        Pulls sentences from the queue and speaks them sequentially.
        A None sentinel in the queue signals normal completion.
        on_start fires before the first sentence, on_done fires
        after the sentinel (but NOT if stopped externally via stop()).

        Args:
            sentence_queue: Queue of sentence strings (None = end)
            on_start: Callback before first sentence plays
            on_done: Callback after all sentences finish
        """
        def runner():
            self.speaking.set()
            first = True
            try:
                while not self._stream_stop.is_set():
                    try:
                        sentence = sentence_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if sentence is None:
                        break
                    if first:
                        if on_start:
                            on_start()
                        first = False
                    self.engine.speak(sentence)
            finally:
                # Only fire on_done for normal completion, not forced stop
                if on_done and not self._stream_stop.is_set():
                    on_done()
                self.speaking.clear()

        with self._lock:
            self._stop_locked()
            self._stream_stop.clear()
            thread = threading.Thread(target=runner, daemon=True,
                                      name="playback-stream")
            self._thread = thread
            thread.start()

    def stop(self):
        """Stop current playback."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        """Internal stop without lock."""
        self._stream_stop.set()
        self.engine.stop()
        thread = self._thread
        self._thread = None
        self.speaking.clear()
        # Join thread with timeout to ensure clean shutdown
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
