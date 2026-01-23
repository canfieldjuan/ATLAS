"""
Audio conversion utilities for PersonaPlex integration.

Handles conversion between:
- SignalWire: mulaw 8kHz mono
- PersonaPlex: Opus 24kHz mono
"""

import audioop
import logging
from typing import Optional

logger = logging.getLogger("atlas.personaplex.audio")

SIGNALWIRE_SAMPLE_RATE = 8000
PERSONAPLEX_SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2


OPUS_FRAME_SIZE = 1920  # 80ms at 24kHz - matches Mimi codec frame rate (12.5 Hz)


class AudioConverter:
    """Bidirectional audio converter for SignalWire <-> PersonaPlex."""

    def __init__(self):
        self._opus_encoder: Optional[object] = None
        self._opus_decoder: Optional[object] = None
        self._sphn_available = False
        self._pcm_buffer: bytes = b""  # Buffer for accumulating PCM samples
        self._check_sphn()

    def _check_sphn(self) -> None:
        """Check if sphn library is available for Opus encoding."""
        try:
            import sphn
            self._sphn_available = True
            logger.debug("sphn library available for Opus codec")
        except ImportError:
            self._sphn_available = False
            logger.warning("sphn library not available, Opus codec disabled")

    def _get_opus_encoder(self) -> object:
        """Get or create Opus encoder."""
        if self._opus_encoder is None:
            import sphn
            self._opus_encoder = sphn.OpusStreamWriter(PERSONAPLEX_SAMPLE_RATE)
        return self._opus_encoder

    def _get_opus_decoder(self) -> object:
        """Get or create Opus decoder."""
        if self._opus_decoder is None:
            import sphn
            self._opus_decoder = sphn.OpusStreamReader(PERSONAPLEX_SAMPLE_RATE)
        return self._opus_decoder

    def mulaw_to_pcm(self, mulaw_data: bytes) -> bytes:
        """Convert mulaw audio to 16-bit PCM."""
        return audioop.ulaw2lin(mulaw_data, SAMPLE_WIDTH)

    def pcm_to_mulaw(self, pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM audio to mulaw."""
        return audioop.lin2ulaw(pcm_data, SAMPLE_WIDTH)

    def resample_8k_to_24k(self, pcm_8k: bytes) -> bytes:
        """Resample PCM audio from 8kHz to 24kHz."""
        pcm_24k, _ = audioop.ratecv(
            pcm_8k,
            SAMPLE_WIDTH,
            1,
            SIGNALWIRE_SAMPLE_RATE,
            PERSONAPLEX_SAMPLE_RATE,
            None,
        )
        return pcm_24k

    def resample_24k_to_8k(self, pcm_24k: bytes) -> bytes:
        """Resample PCM audio from 24kHz to 8kHz."""
        pcm_8k, _ = audioop.ratecv(
            pcm_24k,
            SAMPLE_WIDTH,
            1,
            PERSONAPLEX_SAMPLE_RATE,
            SIGNALWIRE_SAMPLE_RATE,
            None,
        )
        return pcm_8k

    def pcm_to_opus(self, pcm_24k: bytes) -> bytes:
        """Encode PCM audio (24kHz) to Opus with buffering for valid frame sizes."""
        if not self._sphn_available:
            raise RuntimeError("sphn library required for Opus encoding")
        import numpy as np

        # Add to buffer
        self._pcm_buffer += pcm_24k

        # Calculate bytes needed for one frame (2 bytes per sample)
        frame_bytes = OPUS_FRAME_SIZE * SAMPLE_WIDTH
        if len(self._pcm_buffer) < frame_bytes:
            return b""  # Not enough data yet

        # Process complete frames
        encoder = self._get_opus_encoder()
        result = b""

        while len(self._pcm_buffer) >= frame_bytes:
            frame_data = self._pcm_buffer[:frame_bytes]
            self._pcm_buffer = self._pcm_buffer[frame_bytes:]

            samples = np.frombuffer(frame_data, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
            encoder.append_pcm(samples)
            result += encoder.read_bytes()

        return result

    def opus_to_pcm(self, opus_data: bytes) -> bytes:
        """Decode Opus audio to PCM (24kHz)."""
        if not self._sphn_available:
            raise RuntimeError("sphn library required for Opus decoding")
        import numpy as np
        decoder = self._get_opus_decoder()
        decoder.append_bytes(opus_data)
        samples = decoder.read_pcm()
        if samples is None or len(samples) == 0:
            return b""
        int_samples = (samples * 32768.0).astype(np.int16)
        return int_samples.tobytes()

    def signalwire_to_personaplex(self, mulaw_8k: bytes) -> bytes:
        """Convert SignalWire audio (mulaw 8kHz) to PersonaPlex (Opus 24kHz)."""
        pcm_8k = self.mulaw_to_pcm(mulaw_8k)
        pcm_24k = self.resample_8k_to_24k(pcm_8k)
        return self.pcm_to_opus(pcm_24k)

    def personaplex_to_signalwire(self, opus_24k: bytes) -> bytes:
        """Convert PersonaPlex audio (Opus 24kHz) to SignalWire (mulaw 8kHz)."""
        pcm_24k = self.opus_to_pcm(opus_24k)
        if not pcm_24k:
            return b""
        pcm_8k = self.resample_24k_to_8k(pcm_24k)
        return self.pcm_to_mulaw(pcm_8k)

    def reset(self) -> None:
        """Reset encoder and decoder state for new conversation."""
        self._opus_encoder = None
        self._opus_decoder = None
        self._pcm_buffer = b""
