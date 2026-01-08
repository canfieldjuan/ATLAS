"""
Resemblyzer-based speaker identification service.

Uses deep learning voice embeddings for:
- Speaker enrollment (register known voices)
- Speaker identification (who is speaking?)
- Voice similarity comparison
"""

import io
import json
import logging
import struct
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo, SpeakerInfo, SpeakerMatch
from ..registry import register_speaker_id

logger = logging.getLogger("atlas.services.speaker_id")


@register_speaker_id("resemblyzer")
class ResemblyzerSpeakerID(BaseModelService):
    """
    Speaker identification using Resemblyzer.

    Resemblyzer uses a deep speaker encoder trained on thousands of speakers
    to generate 256-dimensional embeddings that capture voice characteristics.
    These embeddings can be compared using cosine similarity.
    """

    CAPABILITIES = ["speaker_embedding", "speaker_identification", "speaker_verification"]
    EMBEDDING_DIM = 256
    SAMPLE_RATE = 16000

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        enrollment_path: Optional[Path] = None,
        use_gpu: bool = False,  # Resemblyzer is fast on CPU
    ):
        super().__init__(
            name="resemblyzer",
            model_id="resemblyzer/voice-encoder",
            cache_path=cache_path or Path("models/resemblyzer"),
            log_file=Path("logs/atlas_speaker_id.log"),
        )
        self._use_gpu = use_gpu
        self._enrolled_speakers: dict[str, SpeakerInfo] = {}
        self._enrollment_path = enrollment_path or Path("data/speakers.json")

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device="cuda" if self._use_gpu else "cpu",
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        """Load the Resemblyzer voice encoder."""
        if self._model is not None:
            self.logger.info("Encoder already loaded")
            return

        self.logger.info("Loading Resemblyzer voice encoder...")

        try:
            from resemblyzer import VoiceEncoder

            # Load the pretrained encoder
            self._model = VoiceEncoder(device="cuda" if self._use_gpu else "cpu")
            self.logger.info("Resemblyzer encoder loaded")

            # Load enrolled speakers from disk
            self._load_enrollments()

        except ImportError as e:
            raise ImportError(
                f"resemblyzer not installed. Install with: pip install resemblyzer\n{e}"
            )

    def unload(self) -> None:
        """Unload the encoder and save enrollments."""
        if self._model is not None:
            self.logger.info("Unloading Resemblyzer")
            self._save_enrollments()
            del self._model
            self._model = None
            self._clear_gpu_memory()

    def get_embedding(self, audio_bytes: bytes) -> np.ndarray:
        """
        Extract speaker embedding from audio.

        Args:
            audio_bytes: Raw audio (WAV or 16-bit PCM at 16kHz)

        Returns:
            256-dimensional speaker embedding
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from resemblyzer import preprocess_wav

        # Convert audio bytes to waveform
        wav = self._load_audio(audio_bytes)

        # Preprocess for resemblyzer
        wav = preprocess_wav(wav)

        # Generate embedding
        embedding = self._model.embed_utterance(wav)

        return embedding

    def enroll(
        self,
        name: str,
        audio_bytes: bytes,
        merge_existing: bool = True,
    ) -> SpeakerInfo:
        """
        Enroll a speaker with a voice sample.

        For best results, use 5-10 seconds of clear speech.
        Multiple enrollments for the same speaker are averaged.

        Args:
            name: Speaker's name/identifier
            audio_bytes: Voice sample audio
            merge_existing: If True, average with existing embedding

        Returns:
            SpeakerInfo for the enrolled speaker
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.logger.info("Enrolling speaker: %s", name)

        # Get embedding for this sample
        embedding = self.get_embedding(audio_bytes)

        if merge_existing and name in self._enrolled_speakers:
            # Average with existing embedding
            existing = self._enrolled_speakers[name]
            old_embedding = existing.embedding
            sample_count = existing.sample_count

            # Weighted average (more weight to existing if many samples)
            new_embedding = (old_embedding * sample_count + embedding) / (sample_count + 1)
            new_embedding = new_embedding / np.linalg.norm(new_embedding)  # Normalize

            speaker_info = SpeakerInfo(
                name=name,
                embedding=new_embedding,
                enrolled_at=existing.enrolled_at,
                sample_count=sample_count + 1,
            )
            self.logger.info(
                "Updated enrollment for %s (now %d samples)",
                name,
                sample_count + 1,
            )
        else:
            # New enrollment
            speaker_info = SpeakerInfo(
                name=name,
                embedding=embedding,
                enrolled_at=time.time(),
                sample_count=1,
            )
            self.logger.info("New enrollment for %s", name)

        self._enrolled_speakers[name] = speaker_info
        self._save_enrollments()

        return speaker_info

    def identify(
        self,
        audio_bytes: bytes,
        threshold: float = 0.75,
    ) -> SpeakerMatch:
        """
        Identify the speaker in an audio sample.

        Compares the audio against all enrolled speakers and returns
        the best match if above the threshold.

        Args:
            audio_bytes: Voice sample to identify
            threshold: Minimum cosine similarity (0.0-1.0) to consider a match
                      Typical values: 0.75 (relaxed), 0.85 (strict)

        Returns:
            SpeakerMatch with the best matching speaker or "unknown"
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not self._enrolled_speakers:
            return SpeakerMatch(name="unknown", confidence=0.0, is_known=False)

        with InferenceTimer() as timer:
            # Get embedding for the input audio
            query_embedding = self.get_embedding(audio_bytes)

            # Compare against all enrolled speakers
            best_match = None
            best_similarity = -1.0

            for name, speaker_info in self._enrolled_speakers.items():
                # Cosine similarity
                similarity = float(
                    np.dot(query_embedding, speaker_info.embedding)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(speaker_info.embedding))
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name

        self.logger.debug(
            "Speaker identification took %.2fms, best match: %s (%.2f)",
            timer.duration * 1000,
            best_match,
            best_similarity,
        )

        # Check if above threshold
        if best_similarity >= threshold:
            return SpeakerMatch(
                name=best_match,
                confidence=best_similarity,
                is_known=True,
            )
        else:
            return SpeakerMatch(
                name="unknown",
                confidence=best_similarity,
                is_known=False,
            )

    def verify(
        self,
        audio_bytes: bytes,
        claimed_name: str,
        threshold: float = 0.80,
    ) -> tuple[bool, float]:
        """
        Verify if audio matches a claimed speaker.

        Args:
            audio_bytes: Voice sample
            claimed_name: Name of the claimed speaker
            threshold: Minimum similarity for verification

        Returns:
            (is_verified, similarity_score)
        """
        if claimed_name not in self._enrolled_speakers:
            return False, 0.0

        query_embedding = self.get_embedding(audio_bytes)
        enrolled_embedding = self._enrolled_speakers[claimed_name].embedding

        similarity = float(
            np.dot(query_embedding, enrolled_embedding)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(enrolled_embedding))
        )

        return similarity >= threshold, similarity

    def list_enrolled(self) -> list[SpeakerInfo]:
        """List all enrolled speakers."""
        return list(self._enrolled_speakers.values())

    def remove_speaker(self, name: str) -> bool:
        """Remove an enrolled speaker."""
        if name in self._enrolled_speakers:
            del self._enrolled_speakers[name]
            self._save_enrollments()
            self.logger.info("Removed speaker: %s", name)
            return True
        return False

    def _load_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Load audio from bytes (WAV or raw PCM)."""
        import wave

        # Try WAV format first
        try:
            buffer = io.BytesIO(audio_bytes)
            with wave.open(buffer, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())

                # Convert to numpy
                if sample_width == 2:
                    samples = np.frombuffer(frames, dtype=np.int16)
                elif sample_width == 4:
                    samples = np.frombuffer(frames, dtype=np.int32)
                else:
                    samples = np.frombuffer(frames, dtype=np.uint8)

                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max

                # Convert to mono if stereo
                if n_channels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)

                # Resample if needed
                if sample_rate != self.SAMPLE_RATE:
                    samples = self._resample(samples, sample_rate, self.SAMPLE_RATE)

                return samples
        except Exception:
            pass

        # Assume raw 16-bit PCM at 16kHz
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio

        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)

    def _save_enrollments(self) -> None:
        """Save enrolled speakers to disk."""
        if not self._enrolled_speakers:
            return

        self._enrollment_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {}
        for name, info in self._enrolled_speakers.items():
            data[name] = {
                "embedding": info.embedding.tolist(),
                "enrolled_at": info.enrolled_at,
                "sample_count": info.sample_count,
            }

        with open(self._enrollment_path, 'w') as f:
            json.dump(data, f)

        self.logger.debug("Saved %d enrollments to %s", len(data), self._enrollment_path)

    def _load_enrollments(self) -> None:
        """Load enrolled speakers from disk."""
        if not self._enrollment_path.exists():
            self.logger.info("No existing enrollments found")
            return

        try:
            with open(self._enrollment_path, 'r') as f:
                data = json.load(f)

            for name, info in data.items():
                self._enrolled_speakers[name] = SpeakerInfo(
                    name=name,
                    embedding=np.array(info["embedding"]),
                    enrolled_at=info.get("enrolled_at", 0.0),
                    sample_count=info.get("sample_count", 1),
                )

            self.logger.info("Loaded %d enrolled speakers", len(self._enrolled_speakers))

        except Exception as e:
            self.logger.warning("Failed to load enrollments: %s", e)
