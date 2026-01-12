"""
Query classifier for determining if RAG should be used.

Classifies queries into:
- Device commands (no RAG needed)
- Simple queries (no RAG needed)
- Knowledge queries (use RAG)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("atlas.memory.query_classifier")


@dataclass
class ClassificationResult:
    """Result of query classification."""

    use_rag: bool
    category: str
    reason: str
    confidence: float = 1.0


class QueryClassifier:
    """
    Classify queries to determine if RAG should be used.

    Device commands and simple queries skip RAG for low latency.
    Knowledge queries use RAG for enhanced responses.
    """

    # Device command patterns - skip RAG
    DEVICE_PATTERNS = [
        r"\b(turn|switch)\s+(on|off)\b",
        r"\b(dim|brighten)\s+",
        r"\b(open|close)\s+(the\s+)?(door|garage|blinds|curtains)\b",
        r"\b(set|change)\s+(the\s+)?(temperature|volume|brightness)\b",
        r"\b(play|pause|stop|skip|next|previous)\b",
        r"\b(lock|unlock)\s+(the\s+)?(door|car)\b",
        r"\barm\s+(the\s+)?alarm\b",
        r"\b(mute|unmute)\b",
    ]

    # Simple/greeting patterns - skip RAG
    SIMPLE_PATTERNS = [
        r"^(hi|hello|hey|thanks|thank you|bye|goodbye|good\s*(morning|night|afternoon|evening))\b",
        r"^(what\s+time\s+is\s+it|what\'?s\s+the\s+time)\b",
        r"^(what\s+day\s+is\s+(it|today))\b",
        r"^(how\s+are\s+you)\b",
        r"^(yes|no|ok|okay|sure|maybe|never\s*mind|cancel)\b",
    ]

    # Patterns that suggest knowledge query - use RAG
    KNOWLEDGE_PATTERNS = [
        r"\b(what|who|where|when|why|how)\s+(is|are|was|were|did|does|do)\b",
        r"\b(explain|describe|tell\s+me\s+about)\b",
        r"\b(remember|recall|did\s+I|did\s+we|last\s+time)\b",
        r"\b(help\s+me\s+(with|understand))\b",
        r"\b(compare|difference\s+between)\b",
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._device_re = [re.compile(p, re.IGNORECASE) for p in self.DEVICE_PATTERNS]
        self._simple_re = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        self._knowledge_re = [re.compile(p, re.IGNORECASE) for p in self.KNOWLEDGE_PATTERNS]

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query to determine if RAG should be used.

        Args:
            query: The user's query text

        Returns:
            ClassificationResult with use_rag flag and category
        """
        if not query or not query.strip():
            return ClassificationResult(
                use_rag=False,
                category="empty",
                reason="Empty query",
            )

        query_lower = query.lower().strip()

        # Check device commands first (highest priority)
        for pattern in self._device_re:
            if pattern.search(query_lower):
                return ClassificationResult(
                    use_rag=False,
                    category="device_command",
                    reason="Device control command detected",
                )

        # Check simple/greeting patterns
        for pattern in self._simple_re:
            if pattern.search(query_lower):
                return ClassificationResult(
                    use_rag=False,
                    category="simple",
                    reason="Simple query or greeting",
                )

        # Check if it looks like a knowledge query
        for pattern in self._knowledge_re:
            if pattern.search(query_lower):
                return ClassificationResult(
                    use_rag=True,
                    category="knowledge",
                    reason="Knowledge or memory query detected",
                )

        # Default: use RAG for longer queries, skip for short ones
        word_count = len(query.split())
        if word_count <= 3:
            return ClassificationResult(
                use_rag=False,
                category="short",
                reason="Short query, skipping RAG",
                confidence=0.7,
            )

        # Default to using RAG for medium/longer queries
        return ClassificationResult(
            use_rag=True,
            category="general",
            reason="General query, using RAG",
            confidence=0.6,
        )


# Global classifier instance
_query_classifier: Optional[QueryClassifier] = None


def get_query_classifier() -> QueryClassifier:
    """Get the global query classifier instance."""
    global _query_classifier
    if _query_classifier is None:
        _query_classifier = QueryClassifier()
    return _query_classifier
