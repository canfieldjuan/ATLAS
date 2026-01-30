"""
Speaker embedding repository.

Manages voice embeddings for speaker identification.
"""

import logging
import pickle
from typing import Optional
from uuid import UUID

import numpy as np

from ..database import get_db_pool

logger = logging.getLogger("atlas.storage.speaker")


class SpeakerRepository:
    """Repository for speaker voice embeddings."""

    async def get_speaker_embedding(
        self,
        user_id: UUID,
    ) -> Optional[np.ndarray]:
        """
        Get a user's voice embedding.

        Args:
            user_id: User ID

        Returns:
            Embedding array or None if not enrolled
        """
        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT speaker_embedding FROM users WHERE id = $1",
            user_id,
        )

        if not row or not row["speaker_embedding"]:
            return None

        try:
            return pickle.loads(row["speaker_embedding"])
        except Exception as e:
            logger.error("Failed to load embedding for user %s: %s", user_id, e)
            return None

    async def save_speaker_embedding(
        self,
        user_id: UUID,
        embedding: np.ndarray,
    ) -> bool:
        """
        Save a user's voice embedding.

        Args:
            user_id: User ID
            embedding: Voice embedding array

        Returns:
            True if successful
        """
        pool = get_db_pool()

        try:
            embedding_bytes = pickle.dumps(embedding)
            await pool.execute(
                """
                UPDATE users
                SET speaker_embedding = $1, updated_at = NOW()
                WHERE id = $2
                """,
                embedding_bytes,
                user_id,
            )
            logger.info("Saved speaker embedding for user %s", user_id)
            return True
        except Exception as e:
            logger.error("Failed to save embedding for user %s: %s", user_id, e)
            return False

    async def delete_speaker_embedding(self, user_id: UUID) -> bool:
        """
        Delete a user's voice embedding.

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        pool = get_db_pool()

        try:
            await pool.execute(
                """
                UPDATE users
                SET speaker_embedding = NULL, updated_at = NOW()
                WHERE id = $1
                """,
                user_id,
            )
            logger.info("Deleted speaker embedding for user %s", user_id)
            return True
        except Exception as e:
            logger.error("Failed to delete embedding for user %s: %s", user_id, e)
            return False

    async def get_all_speaker_embeddings(
        self,
    ) -> list[tuple[UUID, str, np.ndarray]]:
        """
        Get all enrolled speaker embeddings.

        Returns:
            List of (user_id, user_name, embedding) tuples
        """
        pool = get_db_pool()

        rows = await pool.fetch(
            """
            SELECT id, name, speaker_embedding
            FROM users
            WHERE speaker_embedding IS NOT NULL
            """
        )

        results = []
        for row in rows:
            try:
                embedding = pickle.loads(row["speaker_embedding"])
                results.append((row["id"], row["name"], embedding))
            except Exception as e:
                logger.warning(
                    "Failed to load embedding for user %s: %s",
                    row["id"], e
                )

        return results

    async def get_enrolled_users(self) -> list[dict]:
        """
        Get list of users with voice enrollment.

        Returns:
            List of user info dicts
        """
        pool = get_db_pool()

        rows = await pool.fetch(
            """
            SELECT id, name, created_at, updated_at
            FROM users
            WHERE speaker_embedding IS NOT NULL
            ORDER BY name
            """
        )

        return [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def is_user_enrolled(self, user_id: UUID) -> bool:
        """Check if a user has voice enrollment."""
        pool = get_db_pool()

        result = await pool.fetchval(
            """
            SELECT speaker_embedding IS NOT NULL
            FROM users
            WHERE id = $1
            """,
            user_id,
        )

        return bool(result)

    async def get_or_create_user(self, name: str) -> UUID:
        """
        Get existing user by name or create new one.

        Args:
            name: User name

        Returns:
            User ID
        """
        pool = get_db_pool()

        # Try to find existing user
        existing = await pool.fetchval(
            "SELECT id FROM users WHERE name = $1",
            name,
        )

        if existing:
            return existing

        # Create new user
        from uuid import uuid4
        user_id = uuid4()

        await pool.execute(
            """
            INSERT INTO users (id, name, created_at, updated_at)
            VALUES ($1, $2, NOW(), NOW())
            """,
            user_id,
            name,
        )

        logger.info("Created new user: %s (%s)", name, user_id)
        return user_id


# Global repository instance
_repo: Optional[SpeakerRepository] = None


def get_speaker_repo() -> SpeakerRepository:
    """Get the global speaker repository."""
    global _repo
    if _repo is None:
        _repo = SpeakerRepository()
    return _repo
