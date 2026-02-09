"""VAD (Voice Activity Detection) backends for Atlas voice pipeline."""

from .silero import SileroVAD

__all__ = ["SileroVAD"]
