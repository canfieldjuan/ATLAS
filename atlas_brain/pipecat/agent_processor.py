"""
Pipecat processor that routes transcriptions through Atlas Agent.

This processor replaces the simple LLM in the Pipecat pipeline,
providing full Atlas capabilities: tools, device commands, memory.
"""

import logging
from typing import Optional

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TextFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    InputAudioRawFrame,
)

from ..agents import get_atlas_agent, AgentContext

logger = logging.getLogger("atlas.pipecat.agent_processor")


class AtlasAgentProcessor(FrameProcessor):
    """
    Pipecat processor that routes transcriptions through AtlasAgent.

    Replaces the OLLamaLLMService in the pipeline to provide:
    - Tool execution (weather, time, calendar, etc.)
    - Device commands (lights, TV, etc.)
    - Conversation memory
    - Intent parsing
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._agent = None
        self._started = False

    def _ensure_agent(self):
        """Lazy-load the agent on first use."""
        if self._agent is None:
            self._agent = get_atlas_agent(session_id=self._session_id)
            logger.info(
                "AtlasAgentProcessor initialized (session=%s)",
                self._session_id,
            )

    async def process_frame(self, frame: Frame, direction):
        """Process incoming frames."""
        # Handle StartFrame first - must call super() to set base class state
        if isinstance(frame, StartFrame):
            self._started = True
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Before StartFrame, drop all frames silently
        if not self._started:
            return

        # Handle EndFrame/CancelFrame
        if isinstance(frame, (EndFrame, CancelFrame)):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # Audio frames - pass through
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        # Transcriptions - route through Agent
        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame, direction)
            return

        # All other frames - pass through
        await self.push_frame(frame, direction)

    async def _handle_transcription(self, frame: TranscriptionFrame, direction):
        """Handle a transcription frame by routing through the Agent."""
        text = frame.text
        if not text or not text.strip():
            return

        logger.info("Processing transcription: %s", text[:50])
        self._ensure_agent()

        context = AgentContext(
            input_text=text,
            input_type="voice",
            session_id=self._session_id,
        )

        try:
            result = await self._agent.run(context)

            if result.response_text:
                logger.info(
                    "Agent response: %s (action=%s)",
                    result.response_text[:50],
                    result.action_type,
                )
                response_frame = TextFrame(text=result.response_text)
                await self.push_frame(response_frame, direction)
            else:
                logger.warning("Agent returned no response")

        except Exception as e:
            logger.error("Agent processing error: %s", e)
            error_frame = TextFrame(text="Sorry, I encountered an error.")
            await self.push_frame(error_frame, direction)
