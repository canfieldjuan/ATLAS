"""
Streaming support for LangGraph agents.

Provides async generators for token-by-token response streaming,
enabling real-time TTS synthesis and progressive UI updates.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger("atlas.agents.graphs.streaming")


@dataclass
class StreamingToken:
    """A single token from streaming response."""

    token: str
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingResult:
    """Complete result after streaming finishes."""

    success: bool
    response_text: str
    action_type: str = "conversation"
    tokens_streamed: int = 0
    total_ms: float = 0.0
    error: Optional[str] = None


class StreamingCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler for LLM token streaming.

    Collects tokens and pushes them to a queue for async consumption.
    """

    def __init__(self):
        """Initialize the streaming handler."""
        self._queue: asyncio.Queue[Optional[StreamingToken]] = asyncio.Queue()
        self._tokens: list[str] = []
        self._finished = False

    @property
    def queue(self) -> asyncio.Queue[Optional[StreamingToken]]:
        """Get the token queue."""
        return self._queue

    @property
    def tokens(self) -> list[str]:
        """Get all collected tokens."""
        return self._tokens

    @property
    def full_response(self) -> str:
        """Get the full response text."""
        return "".join(self._tokens)

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle new token from LLM."""
        self._tokens.append(token)
        await self._queue.put(StreamingToken(token=token))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle LLM completion."""
        self._finished = True
        await self._queue.put(StreamingToken(token="", is_final=True))

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Handle LLM error."""
        self._finished = True
        await self._queue.put(
            StreamingToken(
                token="",
                is_final=True,
                metadata={"error": type(error).__name__},
            )
        )


async def stream_tokens(
    handler: StreamingCallbackHandler,
    timeout: float = 30.0,
) -> AsyncGenerator[StreamingToken, None]:
    """
    Async generator that yields tokens from the callback handler.

    Args:
        handler: The streaming callback handler
        timeout: Maximum time to wait for next token

    Yields:
        StreamingToken objects
    """
    try:
        while True:
            try:
                token = await asyncio.wait_for(
                    handler.queue.get(),
                    timeout=timeout,
                )
                if token is None or token.is_final:
                    break
                yield token
            except asyncio.TimeoutError:
                logger.warning("Token streaming timed out")
                break
    except Exception as e:
        logger.error("Error in token streaming: %s", e)


class StreamingAgentMixin:
    """
    Mixin that adds streaming capabilities to LangGraph agents.

    Usage:
        class MyAgent(StreamingAgentMixin):
            async def stream(self, input_text: str) -> AsyncGenerator[str, None]:
                async for token in self._stream_response(input_text):
                    yield token
    """

    async def _stream_llm_response(
        self,
        messages: list,
        max_tokens: int = 100,
        temperature: float = 0.7,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response token by token.

        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            on_token: Optional callback for each token

        Yields:
            Token strings
        """
        from ...services import llm_registry

        llm = llm_registry.get_active()
        if llm is None:
            yield "I'm not able to respond right now."
            return

        # Check if LLM supports streaming
        if hasattr(llm, "stream"):
            try:
                async for chunk in llm.stream(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    token = chunk.get("token", "")
                    if token:
                        if on_token:
                            on_token(token)
                        yield token
            except Exception as e:
                logger.error("Streaming failed: %s", e)
                # Fall back to non-streaming
                result = llm.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response = result.get("response", "")
                yield response
        else:
            # Non-streaming fallback
            result = llm.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = result.get("response", "")
            yield response


class StreamingHomeAgent(StreamingAgentMixin):
    """HomeAgent with streaming support."""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize streaming HomeAgent."""
        from .home import HomeAgentGraph

        self._agent = HomeAgentGraph(session_id=session_id)
        self._session_id = session_id

    async def run(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        """Run agent (non-streaming)."""
        return await self._agent.run(input_text, **kwargs)

    async def stream(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent response token by token.

        For device commands (which use templates), yields the full response.
        For conversations, streams token by token.

        Args:
            input_text: User input text
            session_id: Session ID
            on_token: Optional callback for each token
            **kwargs: Additional context

        Yields:
            Response tokens
        """
        # First, classify the intent to determine if we need streaming
        from ...services.intent_router import route_query, ROUTE_TO_WORKFLOW
        from ...config import settings

        if settings.intent_router.enabled:
            route_result = await route_query(input_text)

            # Device commands use templates - no streaming needed
            if route_result.action_category == "device_command":
                result = await self._agent.run(
                    input_text,
                    session_id=session_id,
                    **kwargs,
                )
                yield result.get("response_text", "Done.")
                return

            # Fast path tools also don't need streaming
            if route_result.action_category == "tool_use" and route_result.fast_path_ok:
                result = await self._agent.run(
                    input_text,
                    session_id=session_id,
                    **kwargs,
                )
                yield result.get("response_text", "")
                return

            # Workflow and parameterized tool routes need the agent graph
            if route_result.raw_label in ROUTE_TO_WORKFLOW or (
                route_result.action_category == "tool_use" and not route_result.fast_path_ok
            ):
                result = await self._agent.run(
                    input_text,
                    session_id=session_id,
                    **kwargs,
                )
                yield result.get("response_text", "")
                return

        # For conversation or complex queries, run through the full agent
        # to get tracing, memory retrieval, and conversation history
        result = await self._agent.run(
            input_text,
            session_id=session_id,
            **kwargs,
        )
        response = result.get("response_text", "")
        if on_token:
            on_token(response)
        yield response


class StreamingAtlasAgent(StreamingAgentMixin):
    """AtlasAgent with streaming support."""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize streaming AtlasAgent."""
        from .atlas import AtlasAgentGraph

        self._agent = AtlasAgentGraph(session_id=session_id)
        self._session_id = session_id

    async def run(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        """Run agent (non-streaming)."""
        return await self._agent.run(input_text, **kwargs)

    async def stream(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent response token by token.

        Handles delegation to sub-agents appropriately.

        Args:
            input_text: User input text
            session_id: Session ID
            speaker_id: Speaker identifier
            on_token: Optional callback for each token
            **kwargs: Additional context

        Yields:
            Response tokens
        """
        from ...services.intent_router import route_query, ROUTE_TO_WORKFLOW
        from ...config import settings

        if settings.intent_router.enabled:
            route_result = await route_query(input_text)

            # Device commands are delegated to HomeAgent (template-based)
            if route_result.action_category == "device_command":
                result = await self._agent.run(
                    input_text,
                    session_id=session_id,
                    speaker_id=speaker_id,
                    **kwargs,
                )
                yield result.get("response_text", "Done.")
                return

            # Fast path tools
            if route_result.action_category == "tool_use" and route_result.fast_path_ok:
                result = await self._agent.run(
                    input_text,
                    session_id=session_id,
                    speaker_id=speaker_id,
                    **kwargs,
                )
                yield result.get("response_text", "")
                return

            # Workflow and parameterized tool routes need the agent graph
            if route_result.raw_label in ROUTE_TO_WORKFLOW or (
                route_result.action_category == "tool_use" and not route_result.fast_path_ok
            ):
                result = await self._agent.run(
                    input_text,
                    session_id=session_id,
                    speaker_id=speaker_id,
                    **kwargs,
                )
                yield result.get("response_text", "")
                return

        # For conversation, run through the full agent
        # to get tracing, memory retrieval, and conversation history
        result = await self._agent.run(
            input_text,
            session_id=session_id,
            speaker_id=speaker_id,
            **kwargs,
        )
        response = result.get("response_text", "")
        if on_token:
            on_token(response)
        yield response


# Factory functions


def get_streaming_home_agent(session_id: Optional[str] = None) -> StreamingHomeAgent:
    """Get HomeAgent with streaming support."""
    return StreamingHomeAgent(session_id=session_id)


def get_streaming_atlas_agent(session_id: Optional[str] = None) -> StreamingAtlasAgent:
    """Get AtlasAgent with streaming support."""
    return StreamingAtlasAgent(session_id=session_id)


async def stream_to_tts(
    token_stream: AsyncGenerator[str, None],
    tts_callback: Callable[[str], None],
    buffer_size: int = 3,
) -> str:
    """
    Stream tokens to TTS with buffering for natural speech.

    Buffers tokens until we have enough for a natural TTS chunk
    (e.g., end of sentence or clause).

    Args:
        token_stream: Async generator of tokens
        tts_callback: Callback to send text to TTS
        buffer_size: Minimum tokens before flushing

    Returns:
        Complete response text
    """
    buffer: list[str] = []
    full_response: list[str] = []
    sentence_enders = {".", "!", "?", ":", ";"}

    async for token in token_stream:
        buffer.append(token)
        full_response.append(token)

        # Check if we should flush to TTS
        text = "".join(buffer)
        should_flush = (
            # End of sentence
            any(text.rstrip().endswith(e) for e in sentence_enders)
            # Comma with enough content
            or (text.rstrip().endswith(",") and len(buffer) >= buffer_size)
            # Buffer getting large
            or len(buffer) >= buffer_size * 2
        )

        if should_flush and text.strip():
            tts_callback(text)
            buffer = []

    # Flush remaining buffer
    if buffer:
        text = "".join(buffer)
        if text.strip():
            tts_callback(text)

    return "".join(full_response)
