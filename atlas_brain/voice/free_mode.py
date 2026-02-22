"""
Free Conversation Mode evaluator.

Periodically checks ambient conditions and speaker identity to decide
whether to activate or deactivate free conversation mode on the pipeline.

Free mode entry conditions (all must be met):
  - require_known_speaker=True: a known speaker was identified recently
  - ambient_rms <= ambient_rms_max  (only meaningful with rms_adaptive=True)

Free mode exit conditions (any one triggers exit):
  - No known speaker confirmation within speaker_id_expiry_s seconds
  - Ambient RMS exceeds ambient_rms_max (noisy environment)
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .pipeline import VoicePipeline

from ..config import FreeModeConfig

logger = logging.getLogger("atlas.voice.free_mode")


class FreeModeManager:
    """Background evaluator that activates/deactivates free conversation mode.

    Runs a low-frequency poll loop (default 10s) that checks whether
    the conditions for hands-free always-on listening are satisfied.

    Thread-safe: all pipeline mutations go through pipeline.enter_free_mode()
    and pipeline.exit_free_mode(), which are themselves thread-safe because
    they only flip flags and call frame_processor methods (which use locks).
    """

    def __init__(
        self,
        pipeline: "VoicePipeline",
        config: FreeModeConfig,
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Monotonic time of last known-speaker confirmation
        self._last_known_speaker_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background evaluator thread."""
        if not self.config.enabled:
            return
        if not self.pipeline.conversation_mode_enabled:
            logger.warning(
                "FreeModeManager: conversation_mode_enabled=False — "
                "free mode requires conversation mode. Evaluator will not start."
            )
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="free-mode-eval",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Free mode evaluator started (poll=%.0fs, speaker_expiry=%.0fs, "
            "ambient_max=%.4f, extended_timeout=%dms)",
            self.config.poll_interval_s,
            self.config.speaker_id_expiry_s,
            self.config.ambient_rms_max,
            self.config.extended_timeout_ms,
        )

    def stop(self) -> None:
        """Stop the evaluator thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main evaluation loop — runs every poll_interval_s seconds."""
        # Initial delay so pipeline has time to fully start
        self._stop.wait(timeout=5.0)
        while not self._stop.is_set():
            try:
                self._evaluate()
            except Exception as e:
                logger.warning("Free mode evaluation error: %s", e)
            self._stop.wait(timeout=self.config.poll_interval_s)

    def _evaluate(self) -> None:
        """Check conditions and enter or exit free mode as needed."""
        should_be_free = self._check_conditions()
        active = self.pipeline._free_mode_active

        if should_be_free and not active:
            logger.info("Free mode: conditions met -- activating")
            self.pipeline.enter_free_mode(timeout_ms=self.config.extended_timeout_ms)
        elif not should_be_free and active:
            reason = self._exit_reason()
            logger.info("Free mode: conditions no longer met (%s) -- deactivating", reason)
            self.pipeline.exit_free_mode()
        else:
            logger.debug(
                "Free mode: poll (active=%s, conditions_met=%s)",
                active, should_be_free,
            )

    def _check_conditions(self) -> bool:
        """Return True if all free mode entry conditions are satisfied."""
        # Speaker ID check
        if self.config.require_known_speaker:
            if not self._speaker_ok():
                return False

        # Ambient RMS check — only meaningful when rms_adaptive=True
        if not self._ambient_ok():
            return False

        return True

    def _speaker_ok(self) -> bool:
        """Check whether a known speaker has been seen recently enough.

        The heartbeat (_last_known_speaker_time) is updated ONLY by
        notify_speaker_confirmed(), which is called each time the pipeline
        runs speaker ID on a completed command. Reading _last_speaker_match
        here is only used to bootstrap on the very first evaluation cycle
        when notify_speaker_confirmed() hasn't fired yet.
        """
        now = time.monotonic()

        if self._last_known_speaker_time > 0.0:
            # Primary path: check heartbeat age
            elapsed = now - self._last_known_speaker_time
            if elapsed <= self.config.speaker_id_expiry_s:
                logger.debug("Free mode: speaker heartbeat OK (%.0fs ago)", elapsed)
                return True
            logger.debug(
                "Free mode: speaker heartbeat expired (%.0fs > %.0fs expiry)",
                elapsed, self.config.speaker_id_expiry_s,
            )
            return False

        # Bootstrap path: no heartbeat yet — check if there's a current positive match
        match = self.pipeline._last_speaker_match
        if (match is not None
                and match.matched
                and match.confidence >= self.config.min_speaker_confidence):
            # Bootstrap the heartbeat from the existing match
            self._last_known_speaker_time = now
            logger.debug("Free mode: bootstrapped speaker heartbeat from existing match")
            return True

        logger.debug("Free mode: no known speaker yet (no heartbeat, no current match)")
        return False

    def _ambient_ok(self) -> bool:
        """Check whether ambient noise is below threshold."""
        ambient = self.pipeline.frame_processor.get_ambient_rms()
        if ambient > self.config.ambient_rms_max:
            logger.debug(
                "Free mode: ambient RMS too high (%.4f > %.4f)",
                ambient, self.config.ambient_rms_max,
            )
            return False
        logger.debug(
            "Free mode: ambient RMS OK (%.4f <= %.4f)", ambient, self.config.ambient_rms_max
        )
        return True

    def _exit_reason(self) -> str:
        """Describe why free mode should exit (for logging)."""
        if self.config.require_known_speaker and not self._speaker_ok():
            return "speaker absent/expired"
        if not self._ambient_ok():
            return "ambient RMS too high"
        return "unknown"

    # ------------------------------------------------------------------
    # External notification (called by pipeline when speaker ID runs)
    # ------------------------------------------------------------------

    def notify_speaker_confirmed(self, confidence: float) -> None:
        """Update heartbeat when a speaker is positively identified.

        Call this from VoicePipeline after a successful speaker ID match
        so free mode doesn't expire during active conversation.
        """
        if confidence >= self.config.min_speaker_confidence:
            self._last_known_speaker_time = time.monotonic()
            logger.debug(
                "Free mode: speaker heartbeat updated (confidence=%.3f)", confidence
            )
        else:
            logger.debug(
                "Free mode: speaker confidence %.3f below threshold %.3f, heartbeat not updated",
                confidence, self.config.min_speaker_confidence,
            )
