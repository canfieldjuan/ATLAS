"""
Pipecat Voice Pipeline for Atlas.

Creates a complete voice pipeline using:
- Parakeet/Nemotron STT (local NVIDIA model)
- Ollama LLM (local)
- Kokoro TTS (local)

Target: Sub-1-second voice-to-voice latency.
"""

import asyncio
import logging
from typing import Optional

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator
from pipecat.processors.aggregators.user_response import UserResponseAggregator
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TextFrame,
    LLMMessagesFrame,
    EndFrame,
)
from pipecat.services.ollama import OLLamaLLMService

from .stt import ParakeetSTTService, NemotronSTTService
from .tts import KokoroTTSService, StreamingKokoroTTSService

logger = logging.getLogger("atlas.pipecat.pipeline")


class TranscriptLogger(FrameProcessor):
    """Logs transcriptions for debugging."""

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.info("[USER] %s", frame.text)
        elif isinstance(frame, TextFrame):
            logger.info("[ASSISTANT] %s", frame.text)

        await self.push_frame(frame, direction)


def create_voice_pipeline(
    *,
    stt_model: str = "parakeet",  # "parakeet" or "nemotron"
    ollama_model: str = "qwen3-coder-tools:latest",
    ollama_url: str = "http://localhost:11434",
    tts_voice: str = "am_michael",
    tts_speed: float = 1.15,
    tts_streaming: bool = True,
    device: str = "cuda",
    system_prompt: Optional[str] = None,
) -> tuple[Pipeline, PipelineTask]:
    """
    Create a complete Pipecat voice pipeline with local models.

    Args:
        stt_model: STT model to use ("parakeet" or "nemotron")
        ollama_model: Ollama model name for LLM
        ollama_url: Ollama server URL
        tts_voice: Kokoro voice ID
        tts_speed: TTS speed multiplier
        tts_streaming: Use streaming TTS for lower TTFA
        device: Device for models ("cuda" or "cpu")
        system_prompt: Optional system prompt for LLM

    Returns:
        Tuple of (Pipeline, PipelineTask)
    """
    logger.info("Creating Pipecat voice pipeline")
    logger.info("  STT: %s", stt_model)
    logger.info("  LLM: %s @ %s", ollama_model, ollama_url)
    logger.info("  TTS: Kokoro (voice=%s, speed=%.2f, streaming=%s)",
                tts_voice, tts_speed, tts_streaming)

    # Create STT service
    if stt_model == "parakeet":
        stt = ParakeetSTTService(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device=device,
        )
    else:
        stt = NemotronSTTService(
            model_name="nvidia/nemotron-speech-streaming-en-0.6b",
            device=device,
        )

    # Create LLM service (Ollama)
    llm = OLLamaLLMService(
        model=ollama_model,
        base_url=f"{ollama_url}/v1",
    )

    # Create TTS service
    if tts_streaming:
        tts = StreamingKokoroTTSService(
            voice=tts_voice,
            speed=tts_speed,
            device=device,
        )
    else:
        tts = KokoroTTSService(
            voice=tts_voice,
            speed=tts_speed,
            device=device,
        )

    # Create message aggregators
    user_aggregator = UserResponseAggregator()
    assistant_aggregator = LLMAssistantResponseAggregator()

    # Create transcript logger for debugging
    transcript_logger = TranscriptLogger()

    # Default system prompt
    if system_prompt is None:
        system_prompt = """You are Atlas, a helpful voice assistant.
Keep your responses concise and conversational - you're speaking, not writing.
Respond naturally as if having a conversation.
If asked about the time, weather, or other real-time information, acknowledge that you would need tools to answer accurately."""

    # Initial messages for LLM context
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Build the pipeline
    # Flow: Audio -> STT -> User Aggregator -> LLM -> Assistant Aggregator -> TTS -> Audio
    pipeline = Pipeline([
        stt,
        transcript_logger,
        user_aggregator,
        llm,
        assistant_aggregator,
        tts,
    ])

    # Create task with initial context
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    return pipeline, task


async def run_test_pipeline():
    """
    Test the pipeline with a simple text input.

    This doesn't use audio - just tests the LLM and TTS parts.
    """
    logger.info("Running Pipecat pipeline test")

    # Create a minimal pipeline for testing
    from pipecat.services.ollama import OLLamaLLMService

    llm = OLLamaLLMService(
        model="qwen3-coder-tools:latest",
        base_url="http://localhost:11434/v1",
    )

    tts = KokoroTTSService(
        voice="am_michael",
        speed=1.15,
        device="cuda",
    )

    pipeline = Pipeline([llm, tts])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    # Queue a test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses very brief."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    await task.queue_frame(LLMMessagesFrame(messages))
    await task.queue_frame(EndFrame())

    await runner.run(task)

    logger.info("Pipeline test complete")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test_pipeline())
