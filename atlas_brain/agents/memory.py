"""
Agent memory system.

Wraps ContextAggregator (runtime context) and persistence repositories
to provide a unified memory interface for agents.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from .protocols import AgentMemory as AgentMemoryProtocol

logger = logging.getLogger("atlas.agents.memory")


class AtlasAgentMemory:
    """
    Memory system for Atlas Agent.

    Provides unified access to:
    - Runtime context (people, devices, objects, events)
    - Conversation history (persistent)
    - Session management

    This class wraps existing components rather than replacing them,
    allowing gradual migration to the agent architecture.
    """

    def __init__(
        self,
        context_aggregator: Optional[Any] = None,
        conversation_repo: Optional[Any] = None,
        session_repo: Optional[Any] = None,
    ):
        """
        Initialize agent memory.

        Args:
            context_aggregator: ContextAggregator instance (lazy-loaded if None)
            conversation_repo: ConversationRepository instance (lazy-loaded if None)
            session_repo: SessionRepository instance (lazy-loaded if None)
        """
        self._context = context_aggregator
        self._conversation_repo = conversation_repo
        self._session_repo = session_repo


    # Lazy loading of dependencies

    def _get_context(self) -> Any:
        """Get or create ContextAggregator."""
        if self._context is None:
            from ..orchestration.context import get_context
            self._context = get_context()
        return self._context

    def _get_conversation_repo(self) -> Any:
        """Get or create ConversationRepository."""
        if self._conversation_repo is None:
            from ..storage.repositories.conversation import get_conversation_repo
            self._conversation_repo = get_conversation_repo()
        return self._conversation_repo

    def _get_session_repo(self) -> Any:
        """Get or create SessionRepository."""
        if self._session_repo is None:
            from ..storage.repositories.session import get_session_repo
            self._session_repo = get_session_repo()
        return self._session_repo

    # Conversation history (persistent storage)

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
        turn_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get recent conversation turns from persistent storage.

        Args:
            session_id: Session UUID string
            limit: Maximum number of turns to return
            turn_type: Filter by type ("conversation" or "command"), None for all

        Returns:
            List of turn dictionaries with role, content, etc.
        """
        try:
            from ..utils.session_id import normalize_session_id
            repo = self._get_conversation_repo()
            session_uuid = UUID(normalize_session_id(session_id))

            turns = await repo.get_history(
                session_id=session_uuid,
                limit=limit,
                turn_type=turn_type,
            )

            return [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "speaker_id": turn.speaker_id,
                    "intent": turn.intent,
                    "turn_type": turn.turn_type,
                    "created_at": turn.created_at.isoformat() if turn.created_at else None,
                }
                for turn in turns
            ]

        except Exception as e:
            logger.warning("Failed to get conversation history: %s", e)
            return []

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        speaker_id: Optional[str] = None,
        speaker_uuid: Optional[str] = None,
        intent: Optional[str] = None,
        turn_type: str = "conversation",
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Add a conversation turn to persistent storage.

        Args:
            session_id: Session UUID string
            role: "user" or "assistant"
            content: The message content
            speaker_id: Identified speaker name (optional)
            speaker_uuid: Speaker's users.id UUID string (optional)
            intent: Parsed intent (optional)
            turn_type: "conversation" or "command"
            metadata: Additional metadata (optional)

        Returns:
            UUID string of created turn, or None on failure
        """
        try:
            from ..utils.session_id import normalize_session_id
            repo = self._get_conversation_repo()
            session_uuid = UUID(normalize_session_id(session_id))

            turn_id = await repo.add_turn(
                session_id=session_uuid,
                role=role,
                content=content,
                speaker_id=speaker_id,
                speaker_uuid=UUID(speaker_uuid) if speaker_uuid else None,
                intent=intent,
                turn_type=turn_type,
                metadata=metadata,
            )

            logger.debug("Added %s turn to session %s", role, session_id)
            return str(turn_id)

        except Exception as e:
            logger.warning("Failed to add conversation turn: %s", e)
            return None

    # Runtime context (in-memory, transient)

    def get_runtime_context(self) -> dict[str, Any]:
        """
        Get current runtime context as a dictionary.

        Includes: people present, visible objects, recent events, device states.
        """
        context = self._get_context()
        return context.build_context_dict()

    def build_context_string(self) -> str:
        """
        Build natural language context string for LLM input.

        Returns a formatted string describing current context.
        """
        context = self._get_context()
        return context.build_context_string()

    def get_in_memory_conversation(self, last_n: int = 10) -> list[dict[str, Any]]:
        """
        Get recent conversation from in-memory context.

        This is faster than database retrieval but not persistent.
        """
        context = self._get_context()
        turns = context.get_conversation_history(last_n=last_n)

        return [
            {
                "role": turn.role,
                "content": turn.content,
                "speaker_id": turn.speaker_id,
                "intent": turn.intent,
            }
            for turn in turns
        ]

    def add_in_memory_turn(
        self,
        role: str,
        content: str,
        speaker_id: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> None:
        """
        Add a turn to in-memory conversation context.

        This is for immediate context, not persistent storage.
        """
        context = self._get_context()
        context.add_conversation_turn(
            role=role,
            content=content,
            speaker_id=speaker_id,
            intent=intent,
        )

    def clear_in_memory_conversation(self) -> None:
        """Clear in-memory conversation history."""
        context = self._get_context()
        context.clear_conversation()

    # People tracking

    def update_person(
        self,
        person_id: str,
        name: Optional[str] = None,
        location: Optional[str] = None,
        confidence: float = 0.0,
    ) -> None:
        """Update person presence context."""
        context = self._get_context()
        context.update_person(
            person_id=person_id,
            name=name,
            location=location,
            confidence=confidence,
        )

    def get_present_people(self) -> list[dict[str, Any]]:
        """Get list of currently present people."""
        context = self._get_context()
        people = context.get_present_people()

        return [
            {
                "id": p.id,
                "name": p.name,
                "location": p.location,
                "confidence": p.confidence,
            }
            for p in people
        ]

    def get_person_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Find a person by name."""
        context = self._get_context()
        person = context.get_person_by_name(name)

        if person:
            return {
                "id": person.id,
                "name": person.name,
                "location": person.location,
                "confidence": person.confidence,
            }
        return None

    # Device state tracking

    def update_device(
        self,
        device_id: str,
        name: str,
        state: dict[str, Any],
    ) -> None:
        """Update device state in runtime context."""
        context = self._get_context()
        context.update_device(
            device_id=device_id,
            name=name,
            state=state,
        )

    def get_device_state(self, device_id: str) -> Optional[dict[str, Any]]:
        """Get state of a specific device."""
        context = self._get_context()
        device = context.get_device_state(device_id)

        if device:
            return {
                "device_id": device.device_id,
                "name": device.name,
                "state": device.state,
            }
        return None

    def get_all_devices(self) -> list[dict[str, Any]]:
        """Get all device states."""
        context = self._get_context()
        devices = context.get_all_devices()

        return [
            {
                "device_id": d.device_id,
                "name": d.name,
                "state": d.state,
            }
            for d in devices
        ]

    # Object detection

    def update_object(
        self,
        label: str,
        confidence: float,
        location: Optional[str] = None,
        bounding_box: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        """Update object detection in runtime context."""
        context = self._get_context()
        context.update_object(
            label=label,
            confidence=confidence,
            location=location,
            bounding_box=bounding_box,
        )

    def get_visible_objects(self) -> list[dict[str, Any]]:
        """Get currently visible objects."""
        context = self._get_context()
        objects = context.get_visible_objects()

        return [
            {
                "label": o.label,
                "confidence": o.confidence,
                "location": o.location,
            }
            for o in objects
        ]

    def is_object_visible(self, label: str) -> bool:
        """Check if an object type is currently visible."""
        context = self._get_context()
        return context.is_object_visible(label)

    # Audio events

    def add_audio_event(
        self,
        label: str,
        confidence: float,
        location: Optional[str] = None,
    ) -> None:
        """Record an audio event."""
        context = self._get_context()
        context.add_audio_event(
            label=label,
            confidence=confidence,
            location=location,
        )

    def get_recent_events(self, seconds: int = 60) -> list[dict[str, Any]]:
        """Get audio events from the last N seconds."""
        context = self._get_context()
        events = context.get_recent_events(seconds=seconds)

        return [
            {
                "label": e.label,
                "confidence": e.confidence,
                "location": e.location,
            }
            for e in events
        ]

    # Location

    def set_room(self, room: str) -> None:
        """Set the current room/location."""
        context = self._get_context()
        context.set_room(room)

    # Session management

    async def get_or_create_session(
        self,
        user_id: Optional[str] = None,
        terminal_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get or create a session for the user.

        Args:
            user_id: User UUID string (optional for anonymous)
            terminal_id: Terminal/device identifier

        Returns:
            Session UUID string, or None on failure
        """
        try:
            repo = self._get_session_repo()
            user_uuid = UUID(user_id) if user_id else None

            session = await repo.get_or_create_session(
                user_id=user_uuid,
                terminal_id=terminal_id,
            )

            return str(session.id) if session else None

        except Exception as e:
            logger.warning("Failed to get/create session: %s", e)
            return None

    async def touch_session(self, session_id: str) -> bool:
        """
        Update session activity timestamp.

        Args:
            session_id: Session UUID string

        Returns:
            True if successful, False otherwise
        """
        try:
            repo = self._get_session_repo()
            session_uuid = UUID(session_id)
            await repo.touch_session(session_uuid)
            return True

        except Exception as e:
            logger.warning("Failed to touch session: %s", e)
            return False


# Global memory instance
_agent_memory: Optional[AtlasAgentMemory] = None


def get_agent_memory() -> AtlasAgentMemory:
    """Get or create the global agent memory instance."""
    global _agent_memory
    if _agent_memory is None:
        _agent_memory = AtlasAgentMemory()
    return _agent_memory


def reset_agent_memory() -> None:
    """Reset the global agent memory instance."""
    global _agent_memory
    _agent_memory = None
