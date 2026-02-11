"""
Workflow state manager for multi-turn conversational workflows.

Persists partial workflow state to session.metadata for continuation
across voice turns without requiring all parameters upfront.
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger("atlas.agents.graphs.workflow_state")

# Default timeout for workflow state (minutes) — matches WorkflowConfig.timeout_minutes
DEFAULT_WORKFLOW_TIMEOUT_MINUTES = 10


@dataclass
class ActiveWorkflowState:
    """
    Represents an active workflow awaiting user input.

    Stored in session.metadata["active_workflow"].
    """

    workflow_type: str  # "booking", "email", "reminder", etc.
    current_step: str  # "awaiting_info", "awaiting_date", etc.
    started_at: str  # ISO format datetime
    partial_state: dict = field(default_factory=dict)
    conversation_context: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_type": self.workflow_type,
            "current_step": self.current_step,
            "started_at": self.started_at,
            "partial_state": self.partial_state,
            "conversation_context": self.conversation_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActiveWorkflowState":
        """Create instance from dictionary."""
        return cls(
            workflow_type=data.get("workflow_type", ""),
            current_step=data.get("current_step", ""),
            started_at=data.get("started_at", ""),
            partial_state=data.get("partial_state", {}),
            conversation_context=data.get("conversation_context", []),
        )

    def is_expired(self, timeout_minutes: int = DEFAULT_WORKFLOW_TIMEOUT_MINUTES) -> bool:
        """Check if workflow state has expired due to inactivity."""
        if not self.started_at:
            return True
        try:
            started = datetime.fromisoformat(self.started_at)
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
            return started < cutoff
        except (ValueError, TypeError):
            return True


class WorkflowStateManager:
    """
    Manages workflow state persistence via session.metadata.

    Enables multi-turn conversations by saving partial workflow state
    after clarification questions and restoring it on the next turn.
    """

    def __init__(self, timeout_minutes: int = DEFAULT_WORKFLOW_TIMEOUT_MINUTES):
        """
        Initialize the workflow state manager.

        Args:
            timeout_minutes: Minutes of inactivity before workflow auto-clears
        """
        self._timeout_minutes = timeout_minutes

    async def save_workflow_state(
        self,
        session_id: str,
        workflow_type: str,
        current_step: str,
        partial_state: dict,
        conversation_context: Optional[list] = None,
    ) -> bool:
        """
        Save workflow state to session metadata.

        Args:
            session_id: The session ID (string or UUID)
            workflow_type: Type of workflow ("booking", "email", etc.)
            current_step: Current step in workflow ("awaiting_info", etc.)
            partial_state: Dict of fields collected so far
            conversation_context: Optional list of context turns

        Returns:
            True if saved successfully, False otherwise
        """
        from ...storage.repositories.session import get_session_repo

        try:
            repo = get_session_repo()
            session_uuid = self._to_uuid(session_id)
            if session_uuid is None:
                logger.warning("Invalid session_id: %s", session_id)
                return False

            workflow_state = ActiveWorkflowState(
                workflow_type=workflow_type,
                current_step=current_step,
                started_at=datetime.now(timezone.utc).isoformat(),
                partial_state=partial_state,
                conversation_context=conversation_context or [],
            )

            updated = await repo.update_metadata(
                session_uuid,
                {"active_workflow": workflow_state.to_dict()},
            )

            if updated:
                logger.info(
                    "Saved workflow state: type=%s step=%s session=%s",
                    workflow_type,
                    current_step,
                    session_id,
                )
            return updated

        except Exception as e:
            logger.error("Failed to save workflow state: %s", e)
            return False

    async def restore_workflow_state(
        self,
        session_id: str,
    ) -> Optional[ActiveWorkflowState]:
        """
        Restore workflow state from session metadata.

        Automatically clears expired workflows.

        Args:
            session_id: The session ID

        Returns:
            ActiveWorkflowState if found and valid, None otherwise
        """
        from ...storage.repositories.session import get_session_repo

        try:
            repo = get_session_repo()
            session_uuid = self._to_uuid(session_id)
            if session_uuid is None:
                return None

            session = await repo.get_session(session_uuid)
            if session is None:
                return None

            workflow_data = session.metadata.get("active_workflow")
            if not workflow_data:
                return None

            workflow = ActiveWorkflowState.from_dict(workflow_data)

            # Check expiration
            if workflow.is_expired(self._timeout_minutes):
                logger.info(
                    "Workflow expired: type=%s session=%s",
                    workflow.workflow_type,
                    session_id,
                )
                await self.clear_workflow_state(session_id)
                return None

            logger.debug(
                "Restored workflow state: type=%s step=%s",
                workflow.workflow_type,
                workflow.current_step,
            )
            return workflow

        except Exception as e:
            logger.error("Failed to restore workflow state: %s", e)
            return None

    async def clear_workflow_state(self, session_id: str) -> bool:
        """
        Clear workflow state from session metadata.

        Args:
            session_id: The session ID

        Returns:
            True if cleared successfully, False otherwise
        """
        from ...storage.repositories.session import get_session_repo

        try:
            repo = get_session_repo()
            session_uuid = self._to_uuid(session_id)
            if session_uuid is None:
                return False

            cleared = await repo.clear_metadata_key(session_uuid, "active_workflow")
            if cleared:
                logger.info("Cleared workflow state: session=%s", session_id)
            return cleared

        except Exception as e:
            logger.error("Failed to clear workflow state: %s", e)
            return False

    async def add_context_turn(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> bool:
        """
        Add a conversation turn to the workflow context.

        Args:
            session_id: The session ID
            role: "user" or "assistant"
            content: The message content

        Returns:
            True if added successfully, False otherwise
        """
        workflow = await self.restore_workflow_state(session_id)
        if workflow is None:
            return False

        workflow.conversation_context.append({
            "role": role,
            "content": content,
        })

        return await self.save_workflow_state(
            session_id=session_id,
            workflow_type=workflow.workflow_type,
            current_step=workflow.current_step,
            partial_state=workflow.partial_state,
            conversation_context=workflow.conversation_context,
        )

    async def update_partial_state(
        self,
        session_id: str,
        updates: dict,
        new_step: Optional[str] = None,
    ) -> bool:
        """
        Update partial state fields in active workflow.

        Args:
            session_id: The session ID
            updates: Dict of field updates to merge
            new_step: Optional new step to set

        Returns:
            True if updated successfully, False otherwise
        """
        workflow = await self.restore_workflow_state(session_id)
        if workflow is None:
            return False

        workflow.partial_state.update(updates)
        if new_step:
            workflow.current_step = new_step

        return await self.save_workflow_state(
            session_id=session_id,
            workflow_type=workflow.workflow_type,
            current_step=workflow.current_step,
            partial_state=workflow.partial_state,
            conversation_context=workflow.conversation_context,
        )

    def _to_uuid(self, session_id: str) -> Optional[UUID]:
        """Convert session_id string to UUID."""
        if session_id is None:
            return None
        try:
            if isinstance(session_id, UUID):
                return session_id
            return UUID(str(session_id))
        except (ValueError, TypeError):
            return None


# Global instance
_workflow_state_manager: Optional[WorkflowStateManager] = None


def get_workflow_state_manager() -> WorkflowStateManager:
    """Get the global workflow state manager."""
    global _workflow_state_manager
    if _workflow_state_manager is None:
        from ...config import settings
        _workflow_state_manager = WorkflowStateManager(
            timeout_minutes=settings.workflows.timeout_minutes,
        )
    return _workflow_state_manager
