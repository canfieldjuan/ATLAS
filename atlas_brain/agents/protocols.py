"""
Protocol definitions for Agent system.

These protocols define the interface that all Agent implementations must follow,
enabling different agent types (Atlas, specialized agents, etc.) with consistent behavior.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, Protocol, runtime_checkable


class AgentState(Enum):
    """State of the agent during processing."""

    IDLE = auto()           # Waiting for input
    THINKING = auto()       # Analyzing input, deciding what to do
    EXECUTING = auto()      # Running tools/actions
    RESPONDING = auto()     # Generating response
    ERROR = auto()          # Error state


@dataclass
class AgentInfo:
    """Metadata about an agent."""

    name: str                                   # Agent identifier (e.g., "atlas")
    description: str                            # Human-readable description
    capabilities: list[str] = field(default_factory=list)  # What this agent can do
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "version": self.version,
        }


@dataclass
class AgentContext:
    """
    Context passed to agent during processing.

    Contains all information the agent needs to process a request,
    including the input, conversation history, and runtime state.
    """

    # Input
    input_text: str                             # User's input (transcript or text)
    input_type: str = "text"                    # "text", "voice", "vision"

    # Session info
    session_id: Optional[str] = None            # Session UUID for persistence
    user_id: Optional[str] = None               # User identifier
    speaker_id: Optional[str] = None            # Identified speaker name
    speaker_confidence: float = 0.0

    # Conversation history (loaded from persistence)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    # Runtime context (people present, device states, etc.)
    runtime_context: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "input_type": self.input_type,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "speaker_id": self.speaker_id,
            "speaker_confidence": self.speaker_confidence,
            "conversation_history_length": len(self.conversation_history),
            "has_runtime_context": bool(self.runtime_context),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ThinkResult:
    """
    Result of agent's thinking phase.

    Contains the agent's analysis of the input and decision on what to do.
    """

    # What the agent decided to do
    action_type: str                            # "conversation", "device_command", "tool_use", "none"

    # For device commands
    intent: Optional[Any] = None                # Parsed Intent object

    # For conversations
    needs_llm: bool = False                     # Whether LLM is needed for response
    system_prompt: Optional[str] = None         # Custom system prompt if needed

    # For tool use
    tools_to_call: list[str] = field(default_factory=list)  # Tool names to execute
    tool_params: dict[str, Any] = field(default_factory=dict)

    # Context gathered during thinking
    retrieved_context: Optional[str] = None     # Memory/RAG context

    # Confidence and reasoning
    confidence: float = 0.0
    reasoning: Optional[str] = None             # Why this decision was made

    # Timing
    duration_ms: float = 0.0


@dataclass
class ActResult:
    """
    Result of agent's action execution phase.

    Contains results from executing tools, device commands, etc.
    """

    success: bool
    action_type: str                            # What was executed

    # Results
    action_results: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)

    # For generating response
    response_data: dict[str, Any] = field(default_factory=dict)

    # Error info
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Timing
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """
    Final result from agent processing.

    This is what the Orchestrator receives after calling Agent.run().
    """

    success: bool

    # Response to user
    response_text: Optional[str] = None

    # What happened
    action_type: str = "none"                   # "conversation", "device_command", "tool_use", "none"
    intent: Optional[Any] = None                # Parsed intent if applicable
    action_results: list[dict[str, Any]] = field(default_factory=list)

    # Error info
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Detailed timing (all in milliseconds)
    total_ms: float = 0.0
    think_ms: float = 0.0                       # Intent parsing, context retrieval
    act_ms: float = 0.0                         # Action/tool execution
    llm_ms: float = 0.0                         # LLM response generation
    memory_ms: float = 0.0                      # Memory/RAG retrieval
    tools_ms: float = 0.0                       # Built-in tool execution
    storage_ms: float = 0.0                     # Database persistence

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def timing_breakdown(self) -> dict[str, float]:
        """Get timing breakdown as a dictionary."""
        return {
            "total": self.total_ms,
            "think": self.think_ms,
            "act": self.act_ms,
            "llm": self.llm_ms,
            "memory": self.memory_ms,
            "tools": self.tools_ms,
            "storage": self.storage_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "response_text": self.response_text,
            "action_type": self.action_type,
            "intent": self.intent.model_dump() if self.intent and hasattr(self.intent, "model_dump") else None,
            "action_results": self.action_results,
            "error": self.error,
            "error_code": self.error_code,
            "timing": self.timing_breakdown(),
        }


@runtime_checkable
class Agent(Protocol):
    """
    Protocol for Agent implementations.

    Agents are responsible for:
    - Understanding user intent (think)
    - Executing actions and tools (act)
    - Generating responses (respond)
    - Managing conversation context and memory

    The Orchestrator handles audio I/O (STT, TTS) and delegates
    reasoning/action to the Agent.
    """

    @property
    def info(self) -> AgentInfo:
        """Return metadata about this agent."""
        ...

    @property
    def state(self) -> AgentState:
        """Return current agent state."""
        ...

    async def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """
        Main entry point: Process input and return result.

        This is the primary method called by Orchestrator.
        Internally calls think() -> act() -> respond() as needed.

        Args:
            context: AgentContext with input, session info, and history

        Returns:
            AgentResult with response and action results
        """
        ...

    async def think(
        self,
        context: AgentContext,
    ) -> ThinkResult:
        """
        Analyze input and decide what to do.

        Responsibilities:
        - Parse intent from input
        - Retrieve relevant memory/context
        - Decide: conversation, device command, tool use, or none

        Args:
            context: AgentContext with input and history

        Returns:
            ThinkResult with decision and any retrieved context
        """
        ...

    async def act(
        self,
        context: AgentContext,
        think_result: ThinkResult,
    ) -> ActResult:
        """
        Execute actions based on think result.

        Responsibilities:
        - Execute device commands via ActionDispatcher
        - Run tools (weather, traffic, etc.)
        - Gather data for response generation

        Args:
            context: AgentContext with input and history
            think_result: Result from think() phase

        Returns:
            ActResult with execution results
        """
        ...

    async def respond(
        self,
        context: AgentContext,
        think_result: ThinkResult,
        act_result: Optional[ActResult],
    ) -> str:
        """
        Generate response text.

        Responsibilities:
        - For device commands: Generate confirmation message
        - For conversations: Call LLM with context
        - For errors: Generate error message

        Args:
            context: AgentContext with input and history
            think_result: Result from think() phase
            act_result: Result from act() phase (if any)

        Returns:
            Response text for the user
        """
        ...

    async def store_turn(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> None:
        """
        Persist the conversation turn.

        Stores both user input and agent response to the database
        for future context retrieval.

        Args:
            context: AgentContext with session info
            result: AgentResult with response
        """
        ...

    def reset(self) -> None:
        """
        Reset agent state.

        Called when starting a new interaction or recovering from error.
        """
        ...


@runtime_checkable
class AgentMemory(Protocol):
    """
    Protocol for Agent memory/context management.

    Wraps ContextAggregator and persistence repositories
    to provide unified memory interface for agents.
    """

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent conversation turns."""
        ...

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        speaker_id: Optional[str] = None,
        intent: Optional[str] = None,
        turn_type: str = "conversation",
    ) -> None:
        """Add a conversation turn."""
        ...

    def get_runtime_context(self) -> dict[str, Any]:
        """Get current runtime context (people, devices, events)."""
        ...

    def build_context_string(self) -> str:
        """Build natural language context string for LLM."""
        ...

    def update_person(
        self,
        person_id: str,
        name: Optional[str] = None,
        location: Optional[str] = None,
        confidence: float = 0.0,
    ) -> None:
        """Update person presence context."""
        ...

    def update_device(
        self,
        device_id: str,
        name: str,
        state: dict[str, Any],
    ) -> None:
        """Update device state context."""
        ...


@runtime_checkable
class AgentTools(Protocol):
    """
    Protocol for Agent tool management.

    Wraps IntentParser, ActionDispatcher, and built-in tools
    to provide unified tool interface for agents.
    """

    async def parse_intent(
        self,
        query: str,
    ) -> Optional[Any]:
        """Parse intent from natural language query."""
        ...

    async def execute_intent(
        self,
        intent: Any,
    ) -> dict[str, Any]:
        """Execute a parsed intent via ActionDispatcher."""
        ...

    async def execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a built-in tool (weather, traffic, etc.)."""
        ...

    def list_tools(self) -> list[str]:
        """List available tool names."""
        ...

    def get_tool_keywords(self, tool_name: str) -> list[str]:
        """Get trigger keywords for a tool."""
        ...
