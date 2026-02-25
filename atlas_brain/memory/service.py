"""
MemoryService - Unified memory layer for Atlas Brain.

Aggregates context from multiple sources:
- PostgreSQL: Session history, user profiles
- GraphRAG: Semantic memory from knowledge graph
- ContextAggregator: Real-time physical awareness
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from ..config import settings
from ..storage import db_settings
from .feedback import FeedbackContext, get_feedback_service
from .rag_client import EnhancedPromptResult, get_rag_client
from .token_estimator import TokenBudget, get_token_estimator

logger = logging.getLogger("atlas.memory.service")

# ---------------------------------------------------------------------------
# Response sanitization — strip LLM artifacts before persisting to history.
# Poisoned history causes cascading failures: wrong tool selection, language
# switching, and 30s timeouts on subsequent turns.
# ---------------------------------------------------------------------------
import re as _re

_TOOL_CALL_PATTERNS = _re.compile(
    r"<think>.*?</think>"         # reasoning model thinking tags
    r"|</?tool_call>"             # tool_call XML tags
    r"|<function=\w+>.*?</function>"  # text-based tool call syntax
    r"|\.encoder\s*\n?\s*\{.*?\}"    # hallucinated .encoder{"name":...} fragments
    r"|```json\s*\{[^}]*\"name\"\s*:.*?\}\s*```"  # fenced JSON tool calls
    , _re.DOTALL
)


def _sanitize_for_storage(text: str) -> str:
    """Remove LLM tool-call artifacts and non-response junk before storing.

    Applied to assistant content at both store points (agent path and
    streaming path) to prevent poisoned history from affecting later turns.
    """
    if not text:
        return text
    text = _TOOL_CALL_PATTERNS.sub("", text)
    # Collapse leftover blank lines from stripped content
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    # Recent entity context across all turn types (from PostgreSQL)
    recent_entities: list[dict] = field(default_factory=list)

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
        pre_fetched_sources: list | None = None,
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
            pre_fetched_sources: Pre-fetched SearchSource objects from retrieve_memory.
                When not None (even if empty list), skips RAG search and uses these
                sources directly. Source tracking still runs. None means no
                pre-fetch occurred and include_rag controls RAG behavior.
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

        # Load recent entity context from last N turns (any type)
        if include_history and session_id:
            try:
                context.recent_entities = await self._load_recent_entities(
                    session_id, limit=3
                )
            except Exception as e:
                logger.debug("Recent entities load failed: %s", e)

        # Cross-session fallback: entities from recent turns in other sessions
        if include_history and session_id and not context.recent_entities and db_settings.enabled:
            try:
                context.recent_entities = await self._load_cross_session_entities(limit=5)
            except Exception as e:
                logger.debug("Cross-session entities load failed: %s", e)

        # GraphRAG fallback: entity hints from knowledge graph (opt-in via config)
        if include_history and not context.recent_entities and settings.entity_context.graph_entity_fallback:
            try:
                context.recent_entities = await self._load_graph_entities()
            except Exception as e:
                logger.debug("Graph entity fallback failed: %s", e)

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
        if pre_fetched_sources is not None and include_rag:
            # Use pre-fetched sources (from retrieve_memory node) -- still track them
            try:
                rag_result = EnhancedPromptResult(
                    prompt=query,
                    context_used=bool(pre_fetched_sources),
                    sources=pre_fetched_sources,
                )
                context.rag_result = rag_result
                context.rag_context_used = rag_result.context_used

                if rag_result.sources:
                    context.feedback_context = await self._feedback_service.track_sources(
                        session_id=session_id,
                        query=query,
                        sources=rag_result.sources,
                    )
            except Exception as e:
                logger.warning("Failed to process pre-fetched RAG sources: %s", e)

        elif include_rag and settings.memory.enabled:
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

    async def _load_recent_entities(self, session_id: str, limit: int = 3) -> list[dict]:
        """
        Load the last `limit` turns of ANY type and return their stored entities.

        Queries both conversation and command turns so device context is included.
        Returns a flat list of entity dicts for format_entity_context().
        """
        from ..storage.database import get_db_pool
        from ..storage.repositories.conversation import get_conversation_repo

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        try:
            from datetime import timezone as _tz
            from ..config import settings as _settings
            conv_repo = get_conversation_repo()
            session_uuid = UUID(session_id)
            # turn_type=None -> all types (conversation + command)
            turns = await conv_repo.get_history(session_uuid, limit=limit, turn_type=None)
            max_age_s = _settings.entity_context.max_age_s
            now = datetime.now(_tz.utc)
            entities = []
            for t in reversed(turns):  # newest-first so dedup keeps latest action
                if max_age_s > 0 and t.created_at:
                    # created_at from asyncpg may be naive (UTC) -- normalise
                    ts = t.created_at
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=_tz.utc)
                    if (now - ts).total_seconds() > max_age_s:
                        continue
                for e in (t.metadata or {}).get("entities", []):
                    if e.get("name"):
                        entities.append(e)
            return entities
        except Exception as e:
            logger.warning("Failed to load recent entities: %s", e)
            return []

    async def _load_cross_session_entities(self, limit: int = 5) -> list[dict]:
        """
        Fallback: load entities from recent turns across ALL sessions.

        Used when the current session has no recent entities (e.g. start of a new
        session). Applies a longer staleness window (cross_session_max_age_s, default 24h).
        Works for single-user Atlas without needing a user/speaker identifier.
        """
        from ..storage.database import get_db_pool
        from datetime import timezone as _tz

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        try:
            max_age_s = settings.entity_context.cross_session_max_age_s
            if max_age_s > 0:
                cutoff = datetime.now(_tz.utc) - timedelta(seconds=max_age_s)
                rows = await pool.fetch(
                    """
                    SELECT metadata, created_at
                    FROM conversation_turns
                    WHERE created_at > $1
                      AND metadata ? 'entities'
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    cutoff,
                    limit * 6,
                )
            else:
                rows = await pool.fetch(
                    """
                    SELECT metadata, created_at
                    FROM conversation_turns
                    WHERE metadata ? 'entities'
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit * 6,
                )

            entities: list[dict] = []
            for row in rows:
                raw = row["metadata"]
                if not raw:
                    continue
                meta = json.loads(raw) if isinstance(raw, str) else raw
                for e in meta.get("entities", []):
                    if e.get("name"):
                        entities.append(e)
            return entities
        except Exception as e:
            logger.warning("Failed to load cross-session entities: %s", e)
            return []

    async def _load_graph_entities(self, max_facts: int = 5) -> list[dict]:
        """
        GraphRAG fallback: search for entity hints when no recent turns have entities.

        Runs a fixed query for home/device context and extracts locations from facts.
        Only runs when settings.memory.enabled is True.
        Uses a short timeout (3s) to avoid blocking the voice pipeline.
        """
        if not settings.memory.enabled:
            return []

        try:
            import asyncio
            from ..voice.entity_context import extract_location_from_text

            result = await asyncio.wait_for(
                self._rag_client.search(
                    query="home devices rooms smart lights thermostat frequently used",
                    max_facts=max_facts,
                ),
                timeout=3.0,
            )
            if not result.facts:
                return []

            entities: list[dict] = []
            seen: set[str] = set()
            for fact in result.facts:
                location = extract_location_from_text(fact.fact)
                if location and location.lower() not in seen:
                    seen.add(location.lower())
                    entities.append({
                        "type": "location",
                        "name": location,
                        "action": "",
                        "source": "graph",
                    })
            return entities
        except Exception as e:
            logger.debug("Graph entity fallback failed: %s", e)
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
        speaker_uuid: Optional[str] = None,
        intent: Optional[str] = None,
        turn_type: str = "conversation",
        user_metadata: Optional[dict] = None,
        assistant_metadata: Optional[dict] = None,
    ) -> None:
        """
        Store conversation turns in both PostgreSQL and GraphRAG.

        Args:
            session_id: Session ID
            user_content: User's message
            assistant_content: Assistant's response
            speaker_id: Optional speaker identifier (display name)
            speaker_uuid: Optional speaker users.id UUID string
            intent: Optional detected intent
            turn_type: "conversation" or "command" (commands excluded from context)
            user_metadata: Optional metadata dict for the user turn
            assistant_metadata: Optional metadata dict for the assistant turn
        """
        # Store in PostgreSQL
        if db_settings.enabled:
            await self._store_to_postgresql(
                session_id, user_content, assistant_content,
                speaker_id, intent, turn_type,
                speaker_uuid=speaker_uuid,
                user_metadata=user_metadata,
                assistant_metadata=assistant_metadata,
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
        speaker_uuid: Optional[str] = None,
        user_metadata: Optional[dict] = None,
        assistant_metadata: Optional[dict] = None,
    ) -> None:
        """Store conversation turns in PostgreSQL."""
        from ..storage.database import get_db_pool
        from ..storage.repositories.conversation import get_conversation_repo

        pool = get_db_pool()
        if not pool.is_initialized:
            return

        uuid_val = UUID(speaker_uuid) if speaker_uuid else None

        try:
            conv_repo = get_conversation_repo()
            session_uuid = UUID(session_id)

            if user_content:
                await conv_repo.add_turn(
                    session_id=session_uuid,
                    role="user",
                    content=user_content,
                    speaker_id=speaker_id,
                    speaker_uuid=uuid_val,
                    intent=intent,
                    turn_type=turn_type,
                    metadata=user_metadata,
                )

            if assistant_content:
                await conv_repo.add_turn(
                    session_id=session_uuid,
                    role="assistant",
                    content=_sanitize_for_storage(assistant_content),
                    speaker_uuid=uuid_val,
                    turn_type=turn_type,
                    metadata=assistant_metadata,
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
        """Store conversation messages in GraphRAG via /messages endpoint."""
        try:
            source_desc = "voice-session-%s" % session_id[:8]
            messages = []
            if user_content:
                messages.append({
                    "content": user_content,
                    "role_type": "user",
                    "role": None,
                    "source_description": source_desc,
                })
            if assistant_content:
                messages.append({
                    "content": assistant_content,
                    "role_type": "assistant",
                    "role": None,
                    "source_description": source_desc,
                })
            if not messages:
                return

            await self._rag_client.add_messages(messages=messages)
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
