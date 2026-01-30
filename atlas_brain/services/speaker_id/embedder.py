"""
Voice embedder using Resemblyzer.

Extracts speaker embeddings from audio for identification.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("atlas.services.speaker_id.embedder")

# Lazy load Resemblyzer to avoid slow startup
_encoder = None
_encoder_lock = None


def _get_encoder():
    """Lazy load the Resemblyzer encoder."""
    global _encoder, _encoder_lock

    if _encoder is None:
        import threading
        if _encoder_lock is None:
            _encoder_lock = threading.Lock()

        with _encoder_lock:
            if _encoder is None:
                logger.info("Loading Resemblyzer voice encoder...")
                from resemblyzer import VoiceEncoder
                _encoder = VoiceEncoder()
                logger.info("Resemblyzer encoder loaded")

    return _encoder


class VoiceEmbedder:
    """
    Extracts voice embeddings using Resemblyzer.

    Resemblyzer produces 256-dimensional embeddings that capture
    speaker identity independent of what is being said.
    """

    EMBEDDING_DIM = 256
    MIN_AUDIO_SAMPLES = 8000  # 0.5 seconds at 16kHz
    TARGET_SAMPLE_RATE = 16000

    def __init__(self):
        """Initialize the voice embedder."""
        self._encoder = None

    @property
    def encoder(self):
        """Get the encoder (lazy loaded)."""
        if self._encoder is None:
            self._encoder = _get_encoder()
        return self._encoder

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio samples as float32 numpy array (normalized -1 to 1)
            sample_rate: Sample rate of audio (will resample if not 16kHz)

        Returns:
            256-dimensional embedding vector, or None if audio too short
        """
        if audio is None or len(audio) == 0:
            logger.warning("Empty audio provided")
            return None

        # Convert to float32 if needed
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        # Resample if needed
        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, self.TARGET_SAMPLE_RATE)

        # Check minimum length
        if len(audio) < self.MIN_AUDIO_SAMPLES:
            logger.warning(
                "Audio too short: %d samples (need %d)",
                len(audio), self.MIN_AUDIO_SAMPLES
            )
            return None

        try:
            # Extract embedding using Resemblyzer
            embedding = self.encoder.embed_utterance(audio)
            return embedding
        except Exception as e:
            logger.error("Failed to extract embedding: %s", e)
            return None

    def extract_embedding_from_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """
        Extract embedding from raw PCM bytes (int16).

        Args:
            pcm_bytes: Raw PCM audio as bytes (int16, mono)
            sample_rate: Sample rate

        Returns:
            256-dimensional embedding vector, or None if failed
        """
        if not pcm_bytes:
            return None

        # Convert bytes to numpy array
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        return self.extract_embedding(audio_float, sample_rate)

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp to [0, 1] (cosine can be negative for very different speakers)
        return float(max(0.0, min(1.0, similarity)))

    def average_embeddings(
        self,
        embeddings: list[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Average multiple embeddings into a single representative embedding.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Averaged and normalized embedding, or None if empty
        """
        if not embeddings:
            return None

        # Filter out None values
        valid_embeddings = [e for e in embeddings if e is not None]
        if not valid_embeddings:
            return None

        # Average
        avg_embedding = np.mean(valid_embeddings, axis=0)

        # Normalize to unit length
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: simple linear interpolation
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)


# Global embedder instance
_embedder: Optional[VoiceEmbedder] = None


def get_voice_embedder() -> VoiceEmbedder:
    """Get the global voice embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = VoiceEmbedder()
    return _embedder
