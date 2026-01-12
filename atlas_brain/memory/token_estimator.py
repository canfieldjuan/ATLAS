"""
Token estimation and budget management for context injection.

Provides accurate token counting and context trimming to stay within limits.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("atlas.memory.token_estimator")

# Average characters per token for different content types
CHARS_PER_TOKEN = {
    "english_text": 4.0,
    "code": 3.5,
    "structured": 3.0,
}

# Default token budgets by component
DEFAULT_BUDGETS = {
    "system_prompt": 200,
    "user_profile": 50,
    "physical_context": 100,
    "conversation_history": 500,
    "rag_context": 800,
    "buffer": 100,
}


@dataclass
class TokenBudget:
    """Token budget configuration."""

    max_total: int = 2000
    max_history: int = 500
    max_rag_context: int = 800
    max_physical: int = 100
    reserve_for_response: int = 500

    @property
    def available_for_context(self) -> int:
        """Tokens available for context after reserving response space."""
        return self.max_total - self.reserve_for_response


@dataclass
class TokenUsage:
    """Tracks token usage by component."""

    system_prompt: int = 0
    user_profile: int = 0
    physical_context: int = 0
    conversation_history: int = 0
    rag_context: int = 0

    @property
    def total(self) -> int:
        """Total tokens used."""
        return (
            self.system_prompt
            + self.user_profile
            + self.physical_context
            + self.conversation_history
            + self.rag_context
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "system_prompt": self.system_prompt,
            "user_profile": self.user_profile,
            "physical_context": self.physical_context,
            "conversation_history": self.conversation_history,
            "rag_context": self.rag_context,
            "total": self.total,
        }


class TokenEstimator:
    """
    Estimates token counts and manages context budgets.

    Uses character-based estimation which is fast and reasonably accurate
    for most LLMs (within 10-15% of actual tokenization).
    """

    def __init__(self, budget: Optional[TokenBudget] = None):
        self.budget = budget or TokenBudget()

    def estimate_tokens(self, text: str, content_type: str = "english_text") -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate
            content_type: Type of content (english_text, code, structured)

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        chars_per_token = CHARS_PER_TOKEN.get(content_type, 4.0)
        return max(1, int(len(text) / chars_per_token))

    def estimate_messages(self, messages: list[dict]) -> int:
        """
        Estimate tokens for a list of chat messages.

        Accounts for message structure overhead.
        """
        total = 0
        for msg in messages:
            # Role overhead (~4 tokens per message)
            total += 4
            content = msg.get("content", "")
            total += self.estimate_tokens(content)
        return total

    def calculate_usage(self, context) -> TokenUsage:
        """
        Calculate token usage for a MemoryContext.

        Args:
            context: MemoryContext instance

        Returns:
            TokenUsage breakdown
        """
        usage = TokenUsage()

        # Base system prompt overhead
        usage.system_prompt = 50

        # User profile
        if context.user_name:
            usage.user_profile = 20
        if context.response_style != "balanced":
            usage.user_profile += 15
        if context.expertise_level != "intermediate":
            usage.user_profile += 15

        # Physical context
        if context.current_room:
            usage.physical_context += 10
        usage.physical_context += len(context.people_present) * 5
        usage.physical_context += min(len(context.devices), 5) * 15

        # Conversation history
        for turn in context.conversation_history:
            content = turn.get("content", "")
            usage.conversation_history += self.estimate_tokens(content) + 4

        # RAG context
        if context.rag_result and context.rag_result.context_used:
            usage.rag_context = self.estimate_tokens(context.rag_result.prompt)

        return usage

    def is_over_budget(self, usage: TokenUsage) -> bool:
        """Check if usage exceeds budget."""
        return usage.total > self.budget.available_for_context

    def trim_history(
        self,
        history: list[dict],
        max_tokens: Optional[int] = None,
    ) -> list[dict]:
        """
        Trim conversation history to fit within budget.

        Keeps most recent messages, removes oldest first.

        Args:
            history: List of conversation turns
            max_tokens: Maximum tokens for history

        Returns:
            Trimmed history list
        """
        max_tokens = max_tokens or self.budget.max_history

        if not history:
            return []

        # Calculate tokens for each turn
        turn_tokens = []
        for turn in history:
            content = turn.get("content", "")
            tokens = self.estimate_tokens(content) + 4
            turn_tokens.append((turn, tokens))

        # Keep most recent turns that fit
        trimmed = []
        total = 0

        for turn, tokens in reversed(turn_tokens):
            if total + tokens <= max_tokens:
                trimmed.insert(0, turn)
                total += tokens
            else:
                break

        if len(trimmed) < len(history):
            logger.debug(
                "Trimmed history from %d to %d turns (%d tokens)",
                len(history),
                len(trimmed),
                total,
            )

        return trimmed

    def trim_rag_context(
        self,
        sources: list,
        max_tokens: Optional[int] = None,
    ) -> list:
        """
        Trim RAG sources to fit within budget.

        Keeps highest confidence sources first.

        Args:
            sources: List of RAG sources
            max_tokens: Maximum tokens for RAG context

        Returns:
            Trimmed sources list
        """
        max_tokens = max_tokens or self.budget.max_rag_context

        if not sources:
            return []

        # Sort by confidence (highest first)
        sorted_sources = sorted(
            sources,
            key=lambda s: s.confidence if hasattr(s, "confidence") else s.get("confidence", 0),
            reverse=True,
        )

        # Keep sources that fit
        trimmed = []
        total = 0

        for source in sorted_sources:
            fact = source.fact if hasattr(source, "fact") else source.get("fact", "")
            tokens = self.estimate_tokens(fact) + 10  # Overhead for formatting
            if total + tokens <= max_tokens:
                trimmed.append(source)
                total += tokens
            else:
                break

        if len(trimmed) < len(sources):
            logger.debug(
                "Trimmed RAG sources from %d to %d (%d tokens)",
                len(sources),
                len(trimmed),
                total,
            )

        return trimmed

    def optimize_context(self, context) -> tuple:
        """
        Optimize context to fit within budget.

        Returns the context with trimmed components and usage stats.

        Priority order (last to cut first):
        1. RAG context (cut first if over budget)
        2. Conversation history (oldest first)
        3. Physical context (devices first)
        4. User profile (never cut)

        Args:
            context: MemoryContext to optimize

        Returns:
            Tuple of (optimized_context, token_usage, was_trimmed)
        """
        usage = self.calculate_usage(context)

        if not self.is_over_budget(usage):
            return context, usage, False

        logger.info(
            "Context over budget (%d > %d), optimizing...",
            usage.total,
            self.budget.available_for_context,
        )

        was_trimmed = False

        # Step 1: Trim RAG context
        if context.rag_result and context.rag_result.sources:
            available = self.budget.max_rag_context
            context.rag_result.sources = self.trim_rag_context(
                context.rag_result.sources,
                max_tokens=available,
            )
            was_trimmed = True

        # Recalculate
        usage = self.calculate_usage(context)
        if not self.is_over_budget(usage):
            return context, usage, was_trimmed

        # Step 2: Trim conversation history
        if context.conversation_history:
            available = self.budget.max_history
            context.conversation_history = self.trim_history(
                context.conversation_history,
                max_tokens=available,
            )
            was_trimmed = True

        # Recalculate
        usage = self.calculate_usage(context)
        if not self.is_over_budget(usage):
            return context, usage, was_trimmed

        # Step 3: Trim physical context (devices)
        if len(context.devices) > 3:
            context.devices = context.devices[:3]
            was_trimmed = True

        # Step 4: Aggressive history trimming
        if context.conversation_history and len(context.conversation_history) > 3:
            context.conversation_history = context.conversation_history[-3:]
            was_trimmed = True

        usage = self.calculate_usage(context)
        return context, usage, was_trimmed


# Global estimator instance
_token_estimator: Optional[TokenEstimator] = None


def get_token_estimator(budget: Optional[TokenBudget] = None) -> TokenEstimator:
    """Get the global token estimator instance."""
    global _token_estimator
    if _token_estimator is None:
        _token_estimator = TokenEstimator(budget)
    return _token_estimator
