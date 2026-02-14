"""
Identity embedding repository for edge node sync.

Stores face/gait/speaker embeddings by (name, modality) as the master
registry.  Edge nodes keep local .npy copies and sync via WebSocket.
"""

import logging
import pickle
from typing import Optional

import numpy as np

from ..database import get_db_pool

logger = logging.getLogger("atlas.storage.identity")

VALID_MODALITIES = ("face", "gait", "speaker")

# Expected embedding dimensions per modality
_DIMS = {"face": 512, "gait": 256, "speaker": 192}


class IdentityRepository:
    """Repository for the identity_embeddings table."""

    async def upsert(
        self,
        name: str,
        modality: str,
        embedding: np.ndarray,
        source_node: Optional[str] = None,
    ) -> None:
        """Insert or update an identity embedding."""
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality: {modality}")

        expected_dim = _DIMS.get(modality)
        if expected_dim and len(embedding) != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch for {modality}: "
                f"expected {expected_dim}, got {len(embedding)}"
            )

        pool = get_db_pool()
        embedding_bytes = pickle.dumps(embedding)

        await pool.execute(
            """
            INSERT INTO identity_embeddings (name, modality, embedding, embedding_dim, source_node)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (name, modality)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_dim = EXCLUDED.embedding_dim,
                source_node = EXCLUDED.source_node,
                updated_at = NOW()
            """,
            name,
            modality,
            embedding_bytes,
            len(embedding),
            source_node,
        )
        logger.info("Upserted identity: %s/%s (dim=%d, source=%s)", modality, name, len(embedding), source_node)

    async def get(self, name: str, modality: str) -> Optional[np.ndarray]:
        """Get a single embedding by name and modality."""
        pool = get_db_pool()
        row = await pool.fetchrow(
            "SELECT embedding FROM identity_embeddings WHERE name = $1 AND modality = $2",
            name,
            modality,
        )
        if not row:
            return None
        return pickle.loads(row["embedding"])

    async def get_all(self, modality: str) -> dict[str, np.ndarray]:
        """Get all embeddings for a modality. Returns {name: embedding}."""
        pool = get_db_pool()
        rows = await pool.fetch(
            "SELECT name, embedding FROM identity_embeddings WHERE modality = $1",
            modality,
        )
        result = {}
        for row in rows:
            try:
                result[row["name"]] = pickle.loads(row["embedding"])
            except Exception as e:
                logger.warning("Failed to load %s/%s: %s", modality, row["name"], e)
        return result

    async def get_names(self, modality: str) -> list[str]:
        """Get all identity names for a modality."""
        pool = get_db_pool()
        rows = await pool.fetch(
            "SELECT name FROM identity_embeddings WHERE modality = $1 ORDER BY name",
            modality,
        )
        return [row["name"] for row in rows]

    async def get_all_names(self) -> dict[str, list[str]]:
        """Get {modality: [names]} for all modalities."""
        result = {}
        for mod in VALID_MODALITIES:
            result[mod] = await self.get_names(mod)
        return result

    async def delete(self, name: str, modality: str) -> bool:
        """Delete an identity embedding. Returns True if it existed."""
        pool = get_db_pool()
        status = await pool.execute(
            "DELETE FROM identity_embeddings WHERE name = $1 AND modality = $2",
            name,
            modality,
        )
        count_str = status.split()[-1] if status else "0"
        deleted = count_str != "0"
        if deleted:
            logger.info("Deleted identity: %s/%s", modality, name)
        return deleted

    async def delete_person(self, name: str) -> int:
        """Delete all embeddings for a person across all modalities."""
        pool = get_db_pool()
        status = await pool.execute(
            "DELETE FROM identity_embeddings WHERE name = $1",
            name,
        )
        # status like "DELETE 3"
        try:
            count = int(status.split()[-1]) if status else 0
        except (ValueError, IndexError):
            count = 0
        if count:
            logger.info("Deleted all identities for '%s' (%d removed)", name, count)
        return count

    async def list_all(self) -> list[dict]:
        """List all identities with metadata (no embedding data)."""
        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT name, modality, embedding_dim, source_node, created_at, updated_at
            FROM identity_embeddings
            ORDER BY name, modality
            """
        )
        return [
            {
                "name": row["name"],
                "modality": row["modality"],
                "embedding_dim": row["embedding_dim"],
                "source_node": row["source_node"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }
            for row in rows
        ]

    async def diff_manifest(
        self, edge_manifest: dict[str, list[str]]
    ) -> tuple[dict[str, dict[str, list]], dict[str, list[str]], dict[str, list[str]]]:
        """Compare an edge manifest against the master DB.

        Args:
            edge_manifest: {modality: [names]} from the edge node

        Returns:
            (to_send, to_delete, need_from_edge) where:
            - to_send = {modality: {name: embedding_list}} -- embeddings the edge is missing
            - to_delete = {modality: [names]} -- always empty (deletions are push-only
              via explicit identity_delete messages to avoid wiping edge data when
              Brain DB is empty or freshly migrated)
            - need_from_edge = {modality: [names]} -- identities the edge has
              that Brain is missing (edge should push these)
        """
        to_send: dict[str, dict[str, list]] = {}
        to_delete: dict[str, list[str]] = {}
        need_from_edge: dict[str, list[str]] = {}

        for mod in VALID_MODALITIES:
            edge_names = set(edge_manifest.get(mod, []))
            brain_embeddings = await self.get_all(mod)
            brain_names = set(brain_embeddings.keys())

            # Edge is missing these -- send them
            missing = brain_names - edge_names
            if missing:
                to_send[mod] = {
                    name: brain_embeddings[name].tolist()
                    for name in missing
                }

            # NOTE: We intentionally do NOT tell edges to delete identities
            # they have but Brain doesn't.  Deletions are only triggered by
            # explicit identity_delete messages (from REST API or admin).
            # This prevents wiping edge data when Brain DB is empty/new.

            # Edge has these but Brain doesn't -- ask edge to push them
            edge_only = edge_names - brain_names
            if edge_only:
                need_from_edge[mod] = sorted(edge_only)

        return to_send, to_delete, need_from_edge


# Global instance
_repo: Optional[IdentityRepository] = None


def get_identity_repo() -> IdentityRepository:
    """Get the global identity repository."""
    global _repo
    if _repo is None:
        _repo = IdentityRepository()
    return _repo
