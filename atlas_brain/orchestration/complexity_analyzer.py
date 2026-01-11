"""
Query Complexity Analyzer for intelligent model routing.

Analyzes query complexity to determine the appropriate LLM tier:
- SIMPLE: Greetings, basic questions, short responses
- MEDIUM: Multi-step reasoning, context-dependent queries
- COMPLEX: Math, coding, deep analysis, creative tasks
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .states import PipelineContext


class QueryComplexity(Enum):
    """Query complexity levels for model routing."""
    SIMPLE = 1   # Greetings, basic questions
    MEDIUM = 2   # Multi-step reasoning, context
    COMPLEX = 3  # Math, coding, deep analysis


@dataclass
class ComplexityResult:
    """Result of complexity analysis."""
    level: QueryComplexity
    score: float  # 0.0 to 1.0
    factors: dict[str, float]  # Contributing factors


class ComplexityAnalyzer:
    """
    Analyzes query complexity for intelligent model routing.

    Uses a weighted scoring system based on:
    - Query length and structure
    - Keyword patterns (math, code, analysis)
    - Question complexity (what/why/how)
    - Context depth (multi-turn conversations)
    """

    # Simple query patterns (greetings, basic questions)
    SIMPLE_PATTERNS = [
        r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|thanks|thank\s*you)\b",
        r"^what\s*(time|day|date)\s*is\s*it",
        r"^(yes|no|ok|okay|sure|alright)\b",
        r"^(turn\s*(on|off)|switch|toggle)\s+",  # Device commands
        r"^set\s+(the\s+)?(temperature|brightness|volume)\s+to\s+\d+",
        r"^(open|close)\s+(the\s+)?",
    ]

    # Complex query indicators
    COMPLEX_KEYWORDS = [
        # Math and logic
        r"\b(calculate|compute|solve|equation|formula|derivative|integral)\b",
        r"\b(algorithm|complexity|optimize|efficient)\b",
        # Coding
        r"\b(code|program|function|class|debug|refactor|implement)\b",
        r"\b(python|javascript|rust|java|c\+\+|typescript)\b",
        r"\b(api|database|server|backend|frontend)\b",
        # Deep analysis
        r"\b(analyze|compare|contrast|evaluate|synthesize)\b",
        r"\b(explain\s+(how|why|the\s+difference))\b",
        r"\b(what\s+are\s+the\s+(pros|cons|advantages|disadvantages))\b",
        # Creative/generative
        r"\b(write|compose|create|generate|draft)\s+(a|an|the)?\s*(story|poem|essay|article)\b",
        r"\b(summarize|paraphrase)\s+(this|the)\b",
    ]

    # Medium complexity indicators
    MEDIUM_KEYWORDS = [
        r"\b(why|how\s+does|how\s+do|how\s+can|could\s+you)\b",
        r"\b(explain|describe|tell\s+me\s+about)\b",
        r"\b(difference\s+between|compared\s+to)\b",
        r"\b(should\s+i|what\s+should|recommend)\b",
        r"\b(help\s+me|assist|guide)\b",
    ]

    # Thresholds for tier selection
    SIMPLE_THRESHOLD = 0.3
    COMPLEX_THRESHOLD = 0.7

    def __init__(self):
        # Pre-compile patterns for efficiency
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        self._complex_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_KEYWORDS]
        self._medium_patterns = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_KEYWORDS]

    def analyze(
        self,
        query: str,
        context: Optional["PipelineContext"] = None,
    ) -> ComplexityResult:
        """
        Analyze query complexity.

        Args:
            query: The user's query text
            context: Optional pipeline context for conversation history

        Returns:
            ComplexityResult with level, score, and contributing factors
        """
        if not query or not query.strip():
            return ComplexityResult(
                level=QueryComplexity.SIMPLE,
                score=0.0,
                factors={"empty_query": 1.0}
            )

        query = query.strip()
        factors: dict[str, float] = {}

        # Factor 1: Length-based complexity (0.0 - 0.3)
        length_score = self._score_length(query)
        factors["length"] = length_score

        # Factor 2: Simple pattern match (negative score)
        simple_score = self._score_simple_patterns(query)
        factors["simple_patterns"] = -simple_score

        # Factor 3: Medium keyword patterns (0.0 - 0.3)
        medium_score = self._score_medium_patterns(query)
        factors["medium_keywords"] = medium_score

        # Factor 4: Complex keyword patterns (0.0 - 0.5)
        complex_score = self._score_complex_patterns(query)
        factors["complex_keywords"] = complex_score

        # Factor 5: Question structure complexity (0.0 - 0.2)
        structure_score = self._score_structure(query)
        factors["structure"] = structure_score

        # Factor 6: Context depth (0.0 - 0.2)
        context_score = self._score_context(context)
        factors["context_depth"] = context_score

        # Calculate weighted total
        total_score = (
            length_score * 0.15 +
            (-simple_score) * 0.25 +  # Simple patterns reduce complexity
            medium_score * 0.20 +
            complex_score * 0.25 +
            structure_score * 0.10 +
            context_score * 0.05
        )

        # Clamp to 0.0 - 1.0
        total_score = max(0.0, min(1.0, total_score))

        # Determine level based on thresholds
        if total_score < self.SIMPLE_THRESHOLD:
            level = QueryComplexity.SIMPLE
        elif total_score < self.COMPLEX_THRESHOLD:
            level = QueryComplexity.MEDIUM
        else:
            level = QueryComplexity.COMPLEX

        return ComplexityResult(
            level=level,
            score=total_score,
            factors=factors,
        )

    def _score_length(self, query: str) -> float:
        """Score based on query length."""
        word_count = len(query.split())

        if word_count <= 3:
            return 0.0
        elif word_count <= 10:
            return 0.2
        elif word_count <= 25:
            return 0.5
        elif word_count <= 50:
            return 0.7
        else:
            return 1.0

    def _score_simple_patterns(self, query: str) -> float:
        """Score simple patterns (higher = more simple)."""
        for pattern in self._simple_patterns:
            if pattern.search(query):
                return 1.0
        return 0.0

    def _score_medium_patterns(self, query: str) -> float:
        """Score medium complexity patterns."""
        matches = sum(1 for p in self._medium_patterns if p.search(query))
        return min(1.0, matches * 0.3)

    def _score_complex_patterns(self, query: str) -> float:
        """Score complex patterns."""
        matches = sum(1 for p in self._complex_patterns if p.search(query))
        return min(1.0, matches * 0.4)

    def _score_structure(self, query: str) -> float:
        """Score query structure complexity."""
        score = 0.0

        # Multiple sentences
        sentences = query.count(".") + query.count("?") + query.count("!")
        if sentences > 1:
            score += 0.3

        # Nested clauses (commas, parentheses)
        if query.count(",") >= 2:
            score += 0.2
        if "(" in query and ")" in query:
            score += 0.2

        # Lists or enumerations
        if re.search(r"\b(first|second|third|1\.|2\.|3\.)\b", query, re.IGNORECASE):
            score += 0.3

        return min(1.0, score)

    def _score_context(self, context: Optional["PipelineContext"]) -> float:
        """Score based on conversation context depth."""
        if context is None:
            return 0.0

        # Could expand this to look at conversation history
        # For now, return a base score if we have any context
        return 0.1


# Module-level singleton
_analyzer: Optional[ComplexityAnalyzer] = None


def get_complexity_analyzer() -> ComplexityAnalyzer:
    """Get or create the complexity analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ComplexityAnalyzer()
    return _analyzer
