"""
Intent Router - Routes queries to the appropriate model tier.

Uses fast intent classification to determine:
- Device commands → ACTION tier (no LLM)
- Simple queries → FAST tier (1B model)
- Reasoning → BALANCED tier (8B model)
- Complex/coding → CLOUD tier (OpenAI/Claude)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .model_pool import ModelTier

logger = logging.getLogger("atlas.services.intent_router")


@dataclass
class RoutingDecision:
    """Result of intent routing."""
    tier: ModelTier
    confidence: float
    reason: str
    is_device_command: bool = False
    intent: Optional[dict] = None  # For device commands


class IntentRouter:
    """
    Routes queries to appropriate model tiers.

    Classification hierarchy:
    1. Device command patterns → ACTION (no LLM needed)
    2. Simple patterns → FAST (1B)
    3. Reasoning patterns → BALANCED (8B)
    4. Complex patterns → CLOUD (API)
    """

    # Device command patterns (no LLM needed)
    DEVICE_PATTERNS = [
        r"^(turn\s*(on|off)|switch|toggle)\s+",
        r"^(dim|brighten)\s+",
        r"^set\s+(the\s+)?(brightness|temperature|volume)\s+(to|at)\s+\d+",
        r"^(open|close)\s+(the\s+)?",
        r"^(lock|unlock)\s+(the\s+)?",
        r"^(start|stop|pause|play|resume)\s+",
    ]

    # Simple patterns → FAST tier (1B)
    SIMPLE_PATTERNS = [
        r"^(hi|hello|hey|good\s*(morning|afternoon|evening))\b",
        r"^(thanks|thank\s*you|bye|goodbye)\b",
        r"^what\s*(time|day|date)\s*is\s*it",
        r"^(yes|no|ok|okay|sure|alright|yep|nope)\b",
        r"^(who|what|where|when)\s+is\s+\w+\??$",  # Simple factual
        r"^tell\s+me\s+(a\s+)?joke",
        r"^(how|what)('s| is)\s+(the\s+)?weather",
        r"^set\s+(a\s+)?(timer|alarm|reminder)",
    ]

    # Reasoning patterns → BALANCED tier (8B)
    REASONING_PATTERNS = [
        r"\b(why|how\s+does|how\s+do|how\s+can|could\s+you)\b",
        r"\b(explain|describe|elaborate)\b",
        r"\b(difference\s+between|compared?\s+to|versus|vs\.?)\b",
        r"\b(should\s+i|what\s+should|recommend|suggest)\b",
        r"\b(help\s+me\s+(understand|with)|guide\s+me)\b",
        r"\b(summarize|overview|brief)\b",
        r"\b(pros?\s+and\s+cons?|advantages?|disadvantages?)\b",
    ]

    # Complex patterns → CLOUD tier (API)
    COMPLEX_PATTERNS = [
        # Coding - more flexible patterns
        r"\b(write|create|generate|implement)\s+.{0,20}(code|script|function|program|class|method)\b",
        r"\b(debug|fix|refactor|optimize)\s+.{0,20}(code|script|function|bug|error|issue)?\b",
        r"\b(python|javascript|typescript|rust|java|c\+\+|golang|sql|html|css)\b",
        r"\b(api|endpoint|database|server|backend|frontend|deploy|docker|kubernetes)\b",
        # Math/Analysis
        r"\b(calculate|compute|solve|derive|prove)\b",
        r"\b(equation|formula|algorithm|complexity)\b",
        r"\b(analyze|evaluate|assess)\s+(this|the|my)?\s*(data|results?|performance)\b",
        # Creative/Long-form
        r"\b(write|compose|draft)\s+(a\s+)?(essay|article|story|report|email|letter)\b",
        r"\b(rewrite|paraphrase|translate)\s+(this|the)\b",
        # Multi-step
        r"\b(step\s*by\s*step|detailed|comprehensive|thorough)\b",
        r"\b(plan|design|architect|structure)\s+(a\s+)?(system|project|app)\b",
    ]

    def __init__(self):
        # Pre-compile patterns for efficiency
        self._device_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEVICE_PATTERNS]
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        self._reasoning_patterns = [re.compile(p, re.IGNORECASE) for p in self.REASONING_PATTERNS]
        self._complex_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS]

    def route(self, query: str, context: Optional[dict] = None) -> RoutingDecision:
        """
        Route a query to the appropriate model tier.

        Args:
            query: User's query text
            context: Optional context (conversation history, etc.)

        Returns:
            RoutingDecision with tier and reasoning
        """
        if not query or not query.strip():
            return RoutingDecision(
                tier=ModelTier.FAST,
                confidence=0.0,
                reason="empty_query",
            )

        query = query.strip()

        # Check device commands first (highest priority, no LLM)
        if self._matches_any(query, self._device_patterns):
            return RoutingDecision(
                tier=ModelTier.ACTION,
                confidence=0.95,
                reason="device_command",
                is_device_command=True,
            )

        # Check for complex patterns (cloud API)
        complex_count = self._count_matches(query, self._complex_patterns)
        if complex_count >= 2 or (complex_count >= 1 and len(query.split()) > 20):
            return RoutingDecision(
                tier=ModelTier.CLOUD,
                confidence=0.85,
                reason=f"complex_query (matches={complex_count})",
            )

        # Check for reasoning patterns (8B model)
        reasoning_count = self._count_matches(query, self._reasoning_patterns)
        if reasoning_count >= 1:
            return RoutingDecision(
                tier=ModelTier.BALANCED,
                confidence=0.80,
                reason=f"reasoning_query (matches={reasoning_count})",
            )

        # Check for simple patterns (1B model)
        if self._matches_any(query, self._simple_patterns):
            return RoutingDecision(
                tier=ModelTier.FAST,
                confidence=0.90,
                reason="simple_query",
            )

        # Default based on length
        word_count = len(query.split())
        if word_count <= 5:
            return RoutingDecision(
                tier=ModelTier.FAST,
                confidence=0.70,
                reason="short_query",
            )
        elif word_count <= 15:
            return RoutingDecision(
                tier=ModelTier.BALANCED,
                confidence=0.65,
                reason="medium_query",
            )
        else:
            return RoutingDecision(
                tier=ModelTier.BALANCED,
                confidence=0.60,
                reason="long_query",
            )

    def _matches_any(self, query: str, patterns: list) -> bool:
        """Check if query matches any pattern."""
        return any(p.search(query) for p in patterns)

    def _count_matches(self, query: str, patterns: list) -> int:
        """Count how many patterns match."""
        return sum(1 for p in patterns if p.search(query))

    def route_with_fallback(
        self,
        query: str,
        available_tiers: list[ModelTier],
        context: Optional[dict] = None,
    ) -> RoutingDecision:
        """
        Route with fallback to available tiers.

        If the ideal tier isn't available, falls back to the next best option.
        """
        decision = self.route(query, context)

        if decision.tier in available_tiers:
            return decision

        # Fallback logic - depends on the original tier
        # For complex queries (CLOUD/POWERFUL), prefer higher capability
        # For simple queries, prefer faster tiers
        if decision.tier in (ModelTier.CLOUD, ModelTier.POWERFUL, ModelTier.BALANCED):
            # Complex query - prefer capability over speed
            tier_order = [
                ModelTier.BALANCED,
                ModelTier.POWERFUL,
                ModelTier.CLOUD,
                ModelTier.FAST,
                ModelTier.ACTION,
            ]
        else:
            # Simple query - prefer speed
            tier_order = [
                ModelTier.FAST,
                ModelTier.BALANCED,
                ModelTier.ACTION,
                ModelTier.POWERFUL,
                ModelTier.CLOUD,
            ]

        # Find next available tier
        for fallback_tier in tier_order:
            if fallback_tier in available_tiers:
                return RoutingDecision(
                    tier=fallback_tier,
                    confidence=decision.confidence * 0.8,
                    reason=f"fallback from {decision.tier.name} to {fallback_tier.name}",
                    is_device_command=decision.is_device_command,
                )

        # Last resort
        return RoutingDecision(
            tier=ModelTier.FAST,
            confidence=0.5,
            reason="no_tiers_available",
        )


# Global instance
_router: Optional[IntentRouter] = None


def get_intent_router() -> IntentRouter:
    """Get or create the global intent router."""
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router
