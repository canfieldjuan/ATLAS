"""
Sentence Transformer embedding service.

Uses sentence-transformers models (default: all-MiniLM-L6-v2, 384 dimensions) for semantic embeddings.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("atlas.services.embedding")


class SentenceTransformerEmbedding:
    """
    Embedding service using sentence-transformers.

    Produces 384-dimensional vectors optimized for semantic similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = 384

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            logger.debug("Embedding model already loaded")
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Embedding model loaded (dim=%d, device=%s)",
                self._dimension,
                self._model.device,
            )
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Embedding model unloaded")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (dimension,)
        """
        if self._model is None:
            self.load()

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if self._model is None:
            self.load()

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings

    def to_pgvector(self, embedding: np.ndarray) -> str:
        """
        Convert numpy embedding to pgvector format string.

        Args:
            embedding: numpy array of shape (dimension,)

        Returns:
            String in format "[0.1, 0.2, ...]" for pgvector
        """
        return "[" + ",".join(str(float(x)) for x in embedding) + "]"


# Global instance
_embedding_service: Optional[SentenceTransformerEmbedding] = None


def get_embedding_service() -> SentenceTransformerEmbedding:
    """Get or create the global embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = SentenceTransformerEmbedding()
    return _embedding_service
