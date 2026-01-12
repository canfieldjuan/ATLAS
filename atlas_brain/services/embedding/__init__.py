"""
Embedding service for Atlas Brain.

Provides text-to-vector embeddings for semantic search and RAG.
"""

from .sentence_transformer import SentenceTransformerEmbedding

__all__ = ["SentenceTransformerEmbedding"]
