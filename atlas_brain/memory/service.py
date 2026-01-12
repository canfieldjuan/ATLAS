"""
MemoryService - Unified memory layer for Atlas Brain.

Aggregates context from multiple sources:
- PostgreSQL: Session history, user profiles
- GraphRAG: Semantic memory from knowledge graph
- ContextAggregator: Real-time physical awareness
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from ..config import settings
from ..storage import db_settings
from .feedback import FeedbackContext, get_feedback_service
from .rag_client import EnhancedPromptResult, get_rag_client
from .token_estimator import TokenBudget, get_token_estimator

logger = logging.getLogger("atlas.memory.service")


@dataclass
class MemoryContext:
    """
    Unified context for LLM injection.

    Aggregates data from PostgreSQL, GraphRAG, and ContextAggregator.
    """

    # Session info
    session_id: Optional[str] = None

    # User profile (from PostgreSQL - future)
    user_name: Optional[str] = None
    user_timezone: str = "UTC"
    response_style: str = "balanced"
    expertise_level: str = "intermediate"

    # Conversation history (from PostgreSQL)
    conversation_history: list[dict] = field(default_factory=list)

    # Physical context (from ContextAggregator)
    current_time: str = ""
    current_room: Optional[str] = None
    people_present: list[str] = field(default_factory=list)
    devices: list[dict] = field(default_factory=list)

    # RAG context (from GraphRAG)
    rag_result: Optional[EnhancedPromptResult] = None
    rag_context_used: bool = False

    # Token tracking
    estimated_tokens: int = 0
    max_tokens: int = 2000
    was_trimmed: bool = False
    token_usage: Optional[dict] = None

    # Feedback tracking
    feedback_context: Optional[FeedbackContext] = None


class MemoryService:
    """
    Unified memory service that aggregates context from all sources.

    Sources:
    1. PostgreSQL - Session history, conversation turns
    2. GraphRAG - Semantic memory from knowledge graph
    3. ContextAggregator - Real-time physical awareness
    """

    def __init__(self, token_budget: Optional[TokenBudget] = None):
        self._rag_client = get_rag_client()
        self._token_estimator = get_token_estimator(token_budget)
        self._feedback_service = get_feedback_service()

    async def gather_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_rag: bool = True,
        include_history: bool = True,
        include_physical: bool = True,
        max_history: int = 10,
        max_tokens: int = 2000,
    ) -> MemoryContext:
        """
        Gather context from all memory sources.

        Args:
            query: The user query to gather context for
            session_id: Current session ID
            user_id: Current user ID
            include_rag: Include RAG context from knowledge graph
            include_history: Include conversation history
            include_physical: Include physical context
            max_history: Maximum conversation turns to include
            max_tokens: Maximum tokens for context (triggers trimming if exceeded)

        Returns:
            MemoryContext with aggregated data and token usage
        """
        context = MemoryContext(session_id=session_id)
        context.current_time = datetime.now().strftime("%I:%M %p")

        # Load user profile from PostgreSQL
        if user_id and db_settings.enabled:
            try:
                profile = await self._load_user_profile(user_id)
                if profile:
                    context.user_name = profile.display_name
                    context.user_timezone = profile.timezone
                    context.response_style = profile.response_style
                    context.expertise_level = profile.expertise_level
                    # Respect user RAG preference
                    if not profile.enable_rag:
                        include_rag = False
            except Exception as e:
                logger.warning("Failed to load user profile: %s", e)

        # Load conversation history from PostgreSQL
        if include_history and session_id and db_settings.enabled:
            try:
                history = await self._load_session_history(session_id, max_history)
                context.conversation_history = history
            except Exception as e:
                logger.warning("Failed to load session history: %s", e)

        # Load physical context from ContextAggregator
        if include_physical:
            try:
                physical = self._get_physical_context()
                context.current_room = physical.get("location")
                context.people_present = physical.get("people", [])
                context.devices = physical.get("devices", [])
            except Exception as e:
                logger.warning("Failed to get physical context: %s", e)

        # Get RAG context from GraphRAG
        if include_rag and settings.memory.enabled:
            try:
                rag_result = await self._rag_client.enhance_prompt(
                    query=query,
                    max_sources=settings.memory.context_results,
                )
                context.rag_result = rag_result
                context.rag_context_used = rag_result.context_used

                # Track sources for feedback
                if rag_result.sources:
                    context.feedback_context = await self._feedback_service.track_sources(
                        session_id=session_id,
                        query=query,
                        sources=rag_result.sources,
                    )
            except Exception as e:
                logger.warning("Failed to get RAG context: %s", e)

        # Optimize context to fit within token budget
        context.max_tokens = max_tokens
        context, usage, was_trimmed = self._token_estimator.optimize_context(context)
        context.estimated_tokens = usage.total
        context.was_trimmed = was_trimmed
        context.token_usage = usage.to_dict()

        return context

    async def _load_session_history(
        self,
        session_id: str,
        limit: int,
    ) -> list[dict]:
        """
        Load conversation history from PostgreSQL.

        Only loads 'conversation' turns, not 'command' turns.
        This keeps context focused on actual conversations.
        """
        from ..storage.database import get_db_pool
        from ..storage.repositories.conversation import get_conversation_repo

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        try:
            conv_repo = get_conversation_repo()
            session_uuid = UUID(session_id)
            # Filter to conversation turns only - exclude device commands
            turns = await conv_repo.get_history(
                session_uuid,
                limit=limit,
                turn_type="conversation",
            )

            return [
                {"role": t.role, "content": t.content}
                for t in turns
            ]
        except Exception as e:
            logger.warning("Failed to load history: %s", e)
            return []

    async def _load_user_profile(self, user_id: str):
        """Load user profile from PostgreSQL."""
        from ..storage.database import get_db_pool
        from ..storage.repositories.profile import get_profile_repo

        pool = get_db_pool()
        if not pool.is_initialized:
            return None

        try:
            profile_repo = get_profile_repo()
            user_uuid = UUID(user_id)
            return await profile_repo.get_profile(user_uuid)
        except Exception as e:
            logger.warning("Failed to load profile: %s", e)
            return None

    def _get_physical_context(self) -> dict[str, Any]:
        """Get physical context from ContextAggregator."""
        try:
            from ..orchestration.context import get_context

            ctx = get_context()
            ctx_dict = ctx.build_context_dict()

            people = []
            for p in ctx_dict.get("people", []):
                name = p.get("name") or f"Person_{p.get('id', 'unknown')[:6]}"
                people.append(name)

            devices = []
            for d in ctx_dict.get("devices", []):
                devices.append({
                    "name": d.get("name", ""),
                    "state": d.get("state", {}),
                })

            return {
                "location": ctx_dict.get("location"),
                "people": people,
                "devices": devices,
            }
        except Exception as e:
            logger.debug("ContextAggregator not available: %s", e)
            return {}

    def build_system_prompt(
        self,
        context: MemoryContext,
        base_prompt: Optional[str] = None,
    ) -> str:
        """
        Build a system prompt from gathered context.

        Args:
            context: The gathered MemoryContext
            base_prompt: Optional base system prompt to prepend

        Returns:
            Formatted system prompt string
        """
        parts = []

        if base_prompt:
            parts.append(base_prompt)

        # Current context section
        context_lines = [f"Current time: {context.current_time}"]

        if context.current_room:
            context_lines.append(f"Location: {context.current_room}")

        if context.people_present:
            context_lines.append(f"People present: {', '.join(context.people_present)}")

        if context.devices:
            device_strs = []
            for d in context.devices[:5]:
                state_str = ", ".join(f"{k}={v}" for k, v in d.get("state", {}).items())
                device_strs.append(f"{d.get('name', 'device')}: {state_str}")
            if device_strs:
                context_lines.append(f"Devices: {'; '.join(device_strs)}")

        parts.append("## Context\n" + "\n".join(context_lines))

        # User profile section (when available)
        if context.user_name or context.response_style != "balanced":
            profile_lines = []
            if context.user_name:
                profile_lines.append(f"User: {context.user_name}")
            if context.response_style == "brief":
                profile_lines.append("Preference: Keep responses short and concise")
            elif context.response_style == "detailed":
                profile_lines.append("Preference: Provide thorough explanations")
            if context.expertise_level == "beginner":
                profile_lines.append("Level: Explain concepts simply")
            elif context.expertise_level == "expert":
                profile_lines.append("Level: Use technical language freely")
            if profile_lines:
                parts.append("## User\n" + "\n".join(profile_lines))

        # Conversation history section
        if context.conversation_history:
            history_lines = []
            for turn in context.conversation_history[-5:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                history_lines.append(f"{role}: {content}")

            if history_lines:
                parts.append("## Recent Conversation\n" + "\n".join(history_lines))

        return "\n\n".join(parts)

    async def store_conversation(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        speaker_id: Optional[str] = None,
        intent: Optional[str] = None,
        turn_type: str = "conversation",
    ) -> None:
        """
        Store conversation turns in both PostgreSQL and GraphRAG.

        Args:
            session_id: Session ID
            user_content: User's message
            assistant_content: Assistant's response
            speaker_id: Optional speaker identifier
            intent: Optional detected intent
            turn_type: "conversation" or "command" (commands excluded from context)
        """
        # Store in PostgreSQL
        if db_settings.enabled:
            await self._store_to_postgresql(
                session_id, user_content, assistant_content, speaker_id, intent, turn_type
            )

        # Store in GraphRAG (only conversations, not commands)
        if turn_type == "conversation":
            if settings.memory.enabled and settings.memory.store_conversations:
                await self._store_to_graphrag(session_id, user_content, assistant_content)

    async def _store_to_postgresql(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        speaker_id: Optional[str],
        intent: Optional[str],
        turn_type: str = "conversation",
    ) -> None:
        """Store conversation turns in PostgreSQL."""
        from ..storage.database import get_db_pool
        from ..storage.repositories.conversation import get_conversation_repo

        pool = get_db_pool()
        if not pool.is_initialized:
            return

        try:
            conv_repo = get_conversation_repo()
            session_uuid = UUID(session_id)

            if user_content:
                await conv_repo.add_turn(
                    session_id=session_uuid,
                    role="user",
                    content=user_content,
                    speaker_id=speaker_id,
                    intent=intent,
                    turn_type=turn_type,
                )

            if assistant_content:
                await conv_repo.add_turn(
                    session_id=session_uuid,
                    role="assistant",
                    content=assistant_content,
                    turn_type=turn_type,
                )

            logger.debug("Stored %s in PostgreSQL session %s", turn_type, session_id)

        except Exception as e:
            logger.warning("Failed to store in PostgreSQL: %s", e)

    async def _store_to_graphrag(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Store conversation as episode in GraphRAG."""
        try:
            # Combine into single episode
            content = f"User: {user_content}\nAssistant: {assistant_content}"
            source_desc = f"Voice conversation session {session_id}"

            await self._rag_client.add_episode(
                content=content,
                source_description=source_desc,
            )

            logger.debug("Stored conversation in GraphRAG")

        except Exception as e:
            logger.warning("Failed to store in GraphRAG: %s", e)

    def get_citations(
        self,
        context: MemoryContext,
        max_citations: int = 3,
        min_confidence: float = 0.3,
    ) -> str:
        """
        Get formatted source citations for a response.

        Args:
            context: The MemoryContext with feedback tracking
            max_citations: Maximum citations to include
            min_confidence: Minimum confidence threshold

        Returns:
            Formatted citation string
        """
        if not context.feedback_context:
            return ""

        return self._feedback_service.format_citations(
            context.feedback_context,
            max_citations=max_citations,
            min_confidence=min_confidence,
        )

    async def mark_sources_helpful(
        self,
        context: MemoryContext,
        feedback_type: str = "implicit",
    ) -> None:
        """Mark all tracked sources as helpful."""
        if not context.feedback_context or not context.feedback_context.usage_ids:
            return

        await self._feedback_service.record_helpful(
            context.feedback_context.usage_ids,
            feedback_type=feedback_type,
        )

    async def mark_sources_not_helpful(
        self,
        context: MemoryContext,
        feedback_type: str = "implicit",
    ) -> None:
        """Mark all tracked sources as not helpful."""
        if not context.feedback_context or not context.feedback_context.usage_ids:
            return

        await self._feedback_service.record_not_helpful(
            context.feedback_context.usage_ids,
            feedback_type=feedback_type,
        )


# Global service instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get the global memory service instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
