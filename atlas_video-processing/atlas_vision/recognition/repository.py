"""
Database repository for person recognition.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

import numpy as np

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.vision.recognition.repository")


class PersonRepository:
    """Repository for person recognition database operations."""

    def _to_pgvector(self, embedding: np.ndarray) -> str:
        """Convert numpy embedding to pgvector format string."""
        return "[" + ",".join(str(float(x)) for x in embedding.tolist()) + "]"

    async def create_person(
        self,
        name: str,
        is_known: bool = True,
        auto_created: bool = False,
        metadata: Optional[dict] = None,
    ) -> UUID:
        """Create a new person record."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO persons (name, is_known, auto_created, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id
            """,
            name,
            is_known,
            auto_created,
            json.dumps(metadata or {}),
        )
        return row["id"]

    async def get_person(self, person_id: UUID) -> Optional[dict]:
        """Get person by ID."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT * FROM persons WHERE id = $1",
            person_id,
        )
        return dict(row) if row else None

    async def get_person_by_name(self, name: str) -> Optional[dict]:
        """Get person by name."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT * FROM persons WHERE name = $1",
            name,
        )
        return dict(row) if row else None

    async def list_persons(self, include_unknown: bool = True) -> list[dict]:
        """List all persons."""
        pool = get_db_pool()
        if include_unknown:
            rows = await pool.fetch(
                "SELECT * FROM persons ORDER BY name"
            )
        else:
            rows = await pool.fetch(
                "SELECT * FROM persons WHERE is_known = TRUE ORDER BY name"
            )
        return [dict(row) for row in rows]

    async def update_last_seen(self, person_id: UUID) -> None:
        """Update last seen timestamp."""
        pool = get_db_pool()
        await pool.execute(
            "UPDATE persons SET last_seen_at = NOW() WHERE id = $1",
            person_id,
        )

    async def add_face_embedding(
        self,
        person_id: UUID,
        embedding: np.ndarray,
        quality_score: float = 0.0,
        source: str = "enrollment",
        reference_image: Optional[bytes] = None,
        image_format: str = "jpeg",
    ) -> UUID:
        """Add a face embedding for a person and update centroid."""
        pool = get_db_pool()
        embedding_str = self._to_pgvector(embedding)

        # Insert the embedding
        row = await pool.fetchrow(
            """
            INSERT INTO face_embeddings
            (person_id, embedding, quality_score, source, reference_image, image_format)
            VALUES ($1, $2::vector, $3, $4, $5, $6)
            RETURNING id
            """,
            person_id,
            embedding_str,
            quality_score,
            source,
            reference_image,
            image_format,
        )
        embedding_id = row["id"]

        # Update centroid using weighted average (same pattern as speaker ID)
        await self._update_face_centroid(person_id, embedding)

        return embedding_id

    async def _update_face_centroid(
        self,
        person_id: UUID,
        new_embedding: np.ndarray,
    ) -> None:
        """Update person's face centroid with weighted average."""
        pool = get_db_pool()

        # Get current centroid and sample count
        person = await pool.fetchrow(
            "SELECT face_centroid, face_sample_count FROM persons WHERE id = $1",
            person_id,
        )
        if not person:
            return

        sample_count = person["face_sample_count"] or 0
        old_centroid_str = person["face_centroid"]

        if old_centroid_str and sample_count > 0:
            # Parse existing centroid
            centroid_list = [float(x) for x in str(old_centroid_str).strip("[]").split(",")]
            old_centroid = np.array(centroid_list, dtype=np.float32)

            # Weighted average (more weight to existing if many samples)
            new_centroid = (old_centroid * sample_count + new_embedding) / (sample_count + 1)
        else:
            new_centroid = new_embedding.copy()

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm

        centroid_str = self._to_pgvector(new_centroid)

        await pool.execute(
            """
            UPDATE persons
            SET face_centroid = $1::vector, face_sample_count = $2
            WHERE id = $3
            """,
            centroid_str,
            sample_count + 1,
            person_id,
        )

    async def find_matching_face(
        self,
        embedding: np.ndarray,
        threshold: float = 0.6,
    ) -> Optional[dict]:
        """Find matching face using cosine similarity."""
        pool = get_db_pool()
        embedding_str = self._to_pgvector(embedding)
        row = await pool.fetchrow(
            """
            SELECT
                fe.id as embedding_id,
                fe.person_id,
                p.name,
                p.is_known,
                1 - (fe.embedding <=> $1::vector) as similarity
            FROM face_embeddings fe
            JOIN persons p ON fe.person_id = p.id
            WHERE 1 - (fe.embedding <=> $1::vector) > $2
            ORDER BY fe.embedding <=> $1::vector
            LIMIT 1
            """,
            embedding_str,
            threshold,
        )
        return dict(row) if row else None

    async def add_gait_embedding(
        self,
        person_id: UUID,
        embedding: np.ndarray,
        capture_duration_ms: int = 0,
        frame_count: int = 0,
        walking_direction: Optional[str] = None,
        source: str = "enrollment",
    ) -> UUID:
        """Add a gait embedding for a person and update centroid."""
        pool = get_db_pool()
        embedding_str = self._to_pgvector(embedding)

        # Insert the embedding
        row = await pool.fetchrow(
            """
            INSERT INTO gait_embeddings
            (person_id, embedding, capture_duration_ms, frame_count, walking_direction, source)
            VALUES ($1, $2::vector, $3, $4, $5, $6)
            RETURNING id
            """,
            person_id,
            embedding_str,
            capture_duration_ms,
            frame_count,
            walking_direction,
            source,
        )
        embedding_id = row["id"]

        # Update centroid using weighted average
        await self._update_gait_centroid(person_id, embedding)

        return embedding_id

    async def _update_gait_centroid(
        self,
        person_id: UUID,
        new_embedding: np.ndarray,
    ) -> None:
        """Update person's gait centroid with weighted average."""
        pool = get_db_pool()

        # Get current centroid and sample count
        person = await pool.fetchrow(
            "SELECT gait_centroid, gait_sample_count FROM persons WHERE id = $1",
            person_id,
        )
        if not person:
            return

        sample_count = person["gait_sample_count"] or 0
        old_centroid_str = person["gait_centroid"]

        if old_centroid_str and sample_count > 0:
            # Parse existing centroid
            centroid_list = [float(x) for x in str(old_centroid_str).strip("[]").split(",")]
            old_centroid = np.array(centroid_list, dtype=np.float32)

            # Weighted average
            new_centroid = (old_centroid * sample_count + new_embedding) / (sample_count + 1)
        else:
            new_centroid = new_embedding.copy()

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm

        centroid_str = self._to_pgvector(new_centroid)

        await pool.execute(
            """
            UPDATE persons
            SET gait_centroid = $1::vector, gait_sample_count = $2
            WHERE id = $3
            """,
            centroid_str,
            sample_count + 1,
            person_id,
        )

    async def find_matching_gait(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> Optional[dict]:
        """Find matching gait using cosine similarity."""
        pool = get_db_pool()
        embedding_str = self._to_pgvector(embedding)
        row = await pool.fetchrow(
            """
            SELECT
                ge.id as embedding_id,
                ge.person_id,
                p.name,
                p.is_known,
                1 - (ge.embedding <=> $1::vector) as similarity
            FROM gait_embeddings ge
            JOIN persons p ON ge.person_id = p.id
            WHERE 1 - (ge.embedding <=> $1::vector) > $2
            ORDER BY ge.embedding <=> $1::vector
            LIMIT 1
            """,
            embedding_str,
            threshold,
        )
        return dict(row) if row else None

    async def find_matching_gait_averaged(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> Optional[dict]:
        """
        Find matching gait using pre-computed centroid embeddings.

        Uses pgvector native search against gait_centroid column for O(log n)
        performance with ivfflat index. Scales to thousands of persons.
        """
        pool = get_db_pool()
        embedding_str = self._to_pgvector(embedding)

        row = await pool.fetchrow(
            """
            SELECT
                id as person_id,
                name,
                is_known,
                1 - (gait_centroid <=> $1::vector) as similarity
            FROM persons
            WHERE gait_centroid IS NOT NULL
            AND 1 - (gait_centroid <=> $1::vector) > $2
            ORDER BY gait_centroid <=> $1::vector
            LIMIT 1
            """,
            embedding_str,
            threshold,
        )
        return dict(row) if row else None

    async def log_recognition_event(
        self,
        person_id: Optional[UUID],
        recognition_type: str,
        confidence: float,
        camera_source: Optional[str] = None,
        matched: bool = False,
        metadata: Optional[dict] = None,
    ) -> UUID:
        """Log a recognition event."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO recognition_events
            (person_id, recognition_type, confidence, camera_source, matched, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            RETURNING id
            """,
            person_id,
            recognition_type,
            confidence,
            camera_source,
            matched,
            json.dumps(metadata or {}),
        )
        return row["id"]

    async def get_unknown_person_count(self) -> int:
        """Get count of auto-created unknown persons."""
        pool = get_db_pool()
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM persons WHERE auto_created = TRUE"
        )
        return count

    async def update_person(
        self,
        person_id: UUID,
        name: Optional[str] = None,
        is_known: Optional[bool] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Update person record."""
        pool = get_db_pool()
        updates = []
        params = []
        param_idx = 1

        if name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1
        if is_known is not None:
            updates.append(f"is_known = ${param_idx}")
            params.append(is_known)
            param_idx += 1
        if metadata is not None:
            updates.append(f"metadata = ${param_idx}::jsonb")
            params.append(json.dumps(metadata))
            param_idx += 1

        if not updates:
            return False

        updates.append("updated_at = NOW()")
        params.append(person_id)

        query = f"UPDATE persons SET {', '.join(updates)} WHERE id = ${param_idx}"
        result = await pool.execute(query, *params)
        return result == "UPDATE 1"

    async def delete_person(self, person_id: UUID) -> bool:
        """Delete person and all associated embeddings (cascade)."""
        pool = get_db_pool()
        result = await pool.execute(
            "DELETE FROM persons WHERE id = $1",
            person_id,
        )
        return result == "DELETE 1"

    async def get_person_embedding_counts(self, person_id: UUID) -> dict:
        """Get count of face and gait embeddings for a person."""
        pool = get_db_pool()
        face_count = await pool.fetchval(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id = $1",
            person_id,
        )
        gait_count = await pool.fetchval(
            "SELECT COUNT(*) FROM gait_embeddings WHERE person_id = $1",
            person_id,
        )
        return {"face_embeddings": face_count, "gait_embeddings": gait_count}

    async def get_recent_recognition_events(
        self,
        person_id: Optional[UUID] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent recognition events."""
        pool = get_db_pool()
        if person_id:
            rows = await pool.fetch(
                """
                SELECT re.*, p.name as person_name
                FROM recognition_events re
                LEFT JOIN persons p ON re.person_id = p.id
                WHERE re.person_id = $1
                ORDER BY re.created_at DESC
                LIMIT $2
                """,
                person_id,
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT re.*, p.name as person_name
                FROM recognition_events re
                LEFT JOIN persons p ON re.person_id = p.id
                ORDER BY re.created_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [dict(row) for row in rows]

    async def get_person_face_embeddings(self, person_id: UUID) -> list[np.ndarray]:
        """Get all face embeddings for a person."""
        pool = get_db_pool()
        rows = await pool.fetch(
            "SELECT embedding FROM face_embeddings WHERE person_id = $1",
            person_id,
        )
        embeddings = []
        for row in rows:
            # pgvector returns as string, parse it
            emb_str = str(row["embedding"])
            emb_list = [float(x) for x in emb_str.strip("[]").split(",")]
            embeddings.append(np.array(emb_list, dtype=np.float32))
        return embeddings

    async def get_averaged_face_embedding(self, person_id: UUID) -> Optional[np.ndarray]:
        """
        Compute averaged (centroid) face embedding for a person.

        Uses weighted averaging like speaker ID for more stable recognition.
        Multiple samples create a more robust representation.
        """
        embeddings = await self.get_person_face_embeddings(person_id)
        if not embeddings:
            return None

        # Stack and average
        stacked = np.stack(embeddings)
        averaged = np.mean(stacked, axis=0)

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return averaged

    async def find_matching_face_averaged(
        self,
        embedding: np.ndarray,
        threshold: float = 0.6,
    ) -> Optional[dict]:
        """
        Find matching face using pre-computed centroid embeddings.

        Uses pgvector native search against face_centroid column for O(log n)
        performance with ivfflat index. Scales to thousands of persons.
        """
        pool = get_db_pool()
        embedding_str = self._to_pgvector(embedding)

        row = await pool.fetchrow(
            """
            SELECT
                id as person_id,
                name,
                is_known,
                1 - (face_centroid <=> $1::vector) as similarity
            FROM persons
            WHERE face_centroid IS NOT NULL
            AND 1 - (face_centroid <=> $1::vector) > $2
            ORDER BY face_centroid <=> $1::vector
            LIMIT 1
            """,
            embedding_str,
            threshold,
        )
        return dict(row) if row else None

    async def backfill_face_centroids(self) -> int:
        """
        Backfill face centroids for all persons with face embeddings.

        Call this after migration to populate centroids for existing data.
        Returns the number of persons updated.
        """
        pool = get_db_pool()

        # Get all persons with face embeddings but no centroid
        persons = await pool.fetch(
            """
            SELECT DISTINCT p.id
            FROM persons p
            JOIN face_embeddings fe ON p.id = fe.person_id
            WHERE p.face_centroid IS NULL
            """
        )

        updated = 0
        for person in persons:
            person_id = person["id"]
            embeddings = await self.get_person_face_embeddings(person_id)

            if not embeddings:
                continue

            # Compute centroid
            stacked = np.stack(embeddings)
            centroid = np.mean(stacked, axis=0)

            # L2 normalize
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            centroid_str = self._to_pgvector(centroid)

            await pool.execute(
                """
                UPDATE persons
                SET face_centroid = $1::vector, face_sample_count = $2
                WHERE id = $3
                """,
                centroid_str,
                len(embeddings),
                person_id,
            )
            updated += 1
            logger.info("Backfilled face centroid for person %s (%d samples)", person_id, len(embeddings))

        return updated

    async def get_person_gait_embeddings(self, person_id: UUID) -> list[np.ndarray]:
        """Get all gait embeddings for a person."""
        pool = get_db_pool()
        rows = await pool.fetch(
            "SELECT embedding FROM gait_embeddings WHERE person_id = $1",
            person_id,
        )
        embeddings = []
        for row in rows:
            emb_str = str(row["embedding"])
            emb_list = [float(x) for x in emb_str.strip("[]").split(",")]
            embeddings.append(np.array(emb_list, dtype=np.float32))
        return embeddings

    async def get_averaged_gait_embedding(self, person_id: UUID) -> Optional[np.ndarray]:
        """Compute averaged (centroid) gait embedding for a person."""
        embeddings = await self.get_person_gait_embeddings(person_id)
        if not embeddings:
            return None

        stacked = np.stack(embeddings)
        averaged = np.mean(stacked, axis=0)

        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return averaged

    async def backfill_gait_centroids(self) -> int:
        """
        Backfill gait centroids for all persons with gait embeddings.

        Call this after migration to populate centroids for existing data.
        Returns the number of persons updated.
        """
        pool = get_db_pool()

        persons = await pool.fetch(
            """
            SELECT DISTINCT p.id
            FROM persons p
            JOIN gait_embeddings ge ON p.id = ge.person_id
            WHERE p.gait_centroid IS NULL
            """
        )

        updated = 0
        for person in persons:
            person_id = person["id"]
            embeddings = await self.get_person_gait_embeddings(person_id)

            if not embeddings:
                continue

            stacked = np.stack(embeddings)
            centroid = np.mean(stacked, axis=0)

            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            centroid_str = self._to_pgvector(centroid)

            await pool.execute(
                """
                UPDATE persons
                SET gait_centroid = $1::vector, gait_sample_count = $2
                WHERE id = $3
                """,
                centroid_str,
                len(embeddings),
                person_id,
            )
            updated += 1
            logger.info("Backfilled gait centroid for person %s (%d samples)", person_id, len(embeddings))

        return updated


# Singleton instance
_repository: Optional[PersonRepository] = None


def get_person_repository() -> PersonRepository:
    """Get the person repository singleton."""
    global _repository
    if _repository is None:
        _repository = PersonRepository()
    return _repository
