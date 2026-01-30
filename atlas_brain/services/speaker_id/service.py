"""
Speaker identification service.

Provides speaker enrollment and verification using voice embeddings.
"""

import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import numpy as np

from ...config import settings
from .embedder import VoiceEmbedder, get_voice_embedder

logger = logging.getLogger("atlas.services.speaker_id.service")


@dataclass
class SpeakerMatch:
    """Result of speaker identification."""

    matched: bool
    user_id: Optional[UUID]
    user_name: Optional[str]
    confidence: float
    is_known: bool


@dataclass
class EnrollmentSession:
    """Tracks an in-progress enrollment."""

    user_id: UUID
    user_name: str
    embeddings: list[np.ndarray]
    samples_needed: int


class SpeakerIDService:
    """
    Speaker identification service.

    Handles voice enrollment and verification using Resemblyzer embeddings.
    Designed for parallel execution with ASR to add zero latency.
    """

    def __init__(
        self,
        embedder: Optional[VoiceEmbedder] = None,
        confidence_threshold: Optional[float] = None,
        min_samples: Optional[int] = None,
    ):
        """
        Initialize speaker ID service.

        Args:
            embedder: Voice embedder instance (uses global if None)
            confidence_threshold: Match threshold (uses config if None)
            min_samples: Minimum samples for enrollment (uses config if None)
        """
        self._embedder = embedder
        self._threshold = confidence_threshold
        self._min_samples = min_samples
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._enrollment_sessions: dict[str, EnrollmentSession] = {}
        self._speaker_cache: dict[UUID, np.ndarray] = {}
        self._initialized = False

    @property
    def embedder(self) -> VoiceEmbedder:
        """Get the voice embedder."""
        if self._embedder is None:
            self._embedder = get_voice_embedder()
        return self._embedder

    @property
    def threshold(self) -> float:
        """Get confidence threshold."""
        if self._threshold is None:
            self._threshold = settings.speaker_id.confidence_threshold
        return self._threshold

    @property
    def min_samples(self) -> int:
        """Get minimum enrollment samples."""
        if self._min_samples is None:
            self._min_samples = settings.speaker_id.min_enrollment_samples
        return self._min_samples

    async def initialize(self) -> None:
        """Initialize service and preload model."""
        if self._initialized:
            return

        logger.info("Initializing speaker ID service...")

        # Preload encoder in background thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._preload_encoder)

        self._initialized = True
        logger.info("Speaker ID service initialized")

    def _preload_encoder(self) -> None:
        """Preload the Resemblyzer encoder."""
        _ = self.embedder.encoder

    async def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> SpeakerMatch:
        """
        Identify speaker from audio.

        Args:
            audio: Audio samples (float32 or int16)
            sample_rate: Sample rate

        Returns:
            SpeakerMatch with identification result
        """
        # Extract embedding in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self._executor,
            self.embedder.extract_embedding,
            audio,
            sample_rate,
        )

        if embedding is None:
            return SpeakerMatch(
                matched=False,
                user_id=None,
                user_name=None,
                confidence=0.0,
                is_known=False,
            )

        # Compare against enrolled speakers
        return await self._match_embedding(embedding)

    async def identify_speaker_from_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
    ) -> SpeakerMatch:
        """
        Identify speaker from raw PCM bytes.

        Args:
            pcm_bytes: Raw PCM audio (int16, mono)
            sample_rate: Sample rate

        Returns:
            SpeakerMatch with identification result
        """
        # Extract embedding in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self._executor,
            self.embedder.extract_embedding_from_pcm,
            pcm_bytes,
            sample_rate,
        )

        if embedding is None:
            return SpeakerMatch(
                matched=False,
                user_id=None,
                user_name=None,
                confidence=0.0,
                is_known=False,
            )

        return await self._match_embedding(embedding)

    async def _match_embedding(
        self,
        embedding: np.ndarray,
    ) -> SpeakerMatch:
        """Match embedding against enrolled speakers."""
        from ...storage.repositories.speaker import get_speaker_repo

        repo = get_speaker_repo()
        enrolled = await repo.get_all_speaker_embeddings()

        if not enrolled:
            return SpeakerMatch(
                matched=False,
                user_id=None,
                user_name=None,
                confidence=0.0,
                is_known=False,
            )

        best_match = None
        best_score = 0.0

        for user_id, user_name, stored_embedding in enrolled:
            score = self.embedder.compute_similarity(embedding, stored_embedding)
            if score > best_score:
                best_score = score
                best_match = (user_id, user_name)

        if best_match and best_score >= self.threshold:
            return SpeakerMatch(
                matched=True,
                user_id=best_match[0],
                user_name=best_match[1],
                confidence=best_score,
                is_known=True,
            )

        return SpeakerMatch(
            matched=False,
            user_id=best_match[0] if best_match else None,
            user_name=best_match[1] if best_match else None,
            confidence=best_score,
            is_known=False,
        )

    def start_enrollment(
        self,
        session_id: str,
        user_id: UUID,
        user_name: str,
    ) -> dict:
        """
        Start enrollment session for a user.

        Args:
            session_id: Unique session identifier
            user_id: User ID to enroll
            user_name: User display name

        Returns:
            Session info dict
        """
        self._enrollment_sessions[session_id] = EnrollmentSession(
            user_id=user_id,
            user_name=user_name,
            embeddings=[],
            samples_needed=self.min_samples,
        )

        logger.info("Started enrollment for %s (session=%s)", user_name, session_id)

        return {
            "session_id": session_id,
            "user_id": str(user_id),
            "user_name": user_name,
            "samples_collected": 0,
            "samples_needed": self.min_samples,
        }

    async def add_enrollment_sample(
        self,
        session_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Add voice sample to enrollment session.

        Args:
            session_id: Enrollment session ID
            audio: Audio samples
            sample_rate: Sample rate

        Returns:
            Updated session status
        """
        session = self._enrollment_sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "session_id": session_id}

        # Extract embedding
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self._executor,
            self.embedder.extract_embedding,
            audio,
            sample_rate,
        )

        if embedding is None:
            return {
                "error": "Failed to extract embedding - audio too short or invalid",
                "session_id": session_id,
                "samples_collected": len(session.embeddings),
                "samples_needed": session.samples_needed,
            }

        session.embeddings.append(embedding)
        collected = len(session.embeddings)
        is_ready = collected >= session.samples_needed

        logger.info(
            "Enrollment sample %d/%d for %s",
            collected, session.samples_needed, session.user_name
        )

        return {
            "session_id": session_id,
            "samples_collected": collected,
            "samples_needed": session.samples_needed,
            "is_ready": is_ready,
        }

    async def complete_enrollment(self, session_id: str) -> dict:
        """
        Complete enrollment and store embedding.

        Args:
            session_id: Enrollment session ID

        Returns:
            Enrollment result
        """
        session = self._enrollment_sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "success": False}

        if len(session.embeddings) < session.samples_needed:
            return {
                "error": "Not enough samples",
                "success": False,
                "samples_collected": len(session.embeddings),
                "samples_needed": session.samples_needed,
            }

        # Average embeddings
        avg_embedding = self.embedder.average_embeddings(session.embeddings)
        if avg_embedding is None:
            return {"error": "Failed to compute average embedding", "success": False}

        # Store in database
        from ...storage.repositories.speaker import get_speaker_repo

        repo = get_speaker_repo()
        await repo.save_speaker_embedding(session.user_id, avg_embedding)

        # Update cache
        self._speaker_cache[session.user_id] = avg_embedding

        # Cleanup session
        del self._enrollment_sessions[session_id]

        logger.info("Completed enrollment for %s", session.user_name)

        return {
            "success": True,
            "user_id": str(session.user_id),
            "user_name": session.user_name,
            "samples_used": len(session.embeddings),
        }

    def cancel_enrollment(self, session_id: str) -> dict:
        """Cancel an enrollment session."""
        if session_id in self._enrollment_sessions:
            session = self._enrollment_sessions.pop(session_id)
            logger.info("Cancelled enrollment for %s", session.user_name)
            return {"success": True, "session_id": session_id}
        return {"success": False, "error": "Session not found"}

    async def delete_enrollment(self, user_id: UUID) -> bool:
        """Delete a user's voice enrollment."""
        from ...storage.repositories.speaker import get_speaker_repo

        repo = get_speaker_repo()
        success = await repo.delete_speaker_embedding(user_id)

        if success and user_id in self._speaker_cache:
            del self._speaker_cache[user_id]

        return success

    def shutdown(self) -> None:
        """Shutdown the service."""
        self._executor.shutdown(wait=False)
        self._enrollment_sessions.clear()
        self._speaker_cache.clear()


# Global service instance
_service: Optional[SpeakerIDService] = None


def get_speaker_id_service() -> SpeakerIDService:
    """Get the global speaker ID service."""
    global _service
    if _service is None:
        _service = SpeakerIDService()
    return _service


async def initialize_speaker_id() -> None:
    """Initialize the speaker ID service."""
    if settings.speaker_id.enabled:
        service = get_speaker_id_service()
        await service.initialize()
