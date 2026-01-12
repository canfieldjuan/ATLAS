"""
Memory service for Atlas Brain.

Connects to atlas-memory (graphiti-wrapper) for long-term knowledge storage.
"""

from .client import (
    MemoryClient,
    get_memory_client,
    SearchResult,
    EnhancedSearchResult,
    EpisodeResult,
)

__all__ = [
    "MemoryClient",
    "get_memory_client",
    "SearchResult",
    "EnhancedSearchResult",
    "EpisodeResult",
]
