"""
Pipecat Voice Pipeline for Atlas.

Creates a complete voice pipeline using:
- SileroVAD for voice activity detection
- Nemotron STT (local NVIDIA model)
- Ollama LLM (local)
- Kokoro TTS (local)
- LocalAudioTransport for mic/speaker

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
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TextFrame,
    LLMMessagesFrame,
    EndFrame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.processors.filters.wake_check_filter import WakeCheckFilter
from pipecat.audio.utils import create_stream_resampler
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from .tts import KokoroTTSService, StreamingKokoroTTSService
from .stt import ParakeetSTTService, NemotronSTTService
from .agent_processor import AtlasAgentProcessor

logger = logging.getLogger("atlas.pipecat.pipeline")


def find_audio_device_by_name(name_substring: str, input_device: bool = True) -> int | None:
    """Find audio device index by name substring.

    Args:
        name_substring: Substring to search for in device name (case-insensitive)
        input_device: If True, search input devices; if False, search output devices

    Returns:
        Device index if found, None otherwise
    """
    import pyaudio

    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            has_channels = info['maxInputChannels'] > 0 if input_device else info['maxOutputChannels'] > 0
            if has_channels and name_substring.lower() in info['name'].lower():
                logger.info("Found audio device '%s' at index %d: %s",
                           name_substring, i, info['name'])
                return i
        logger.warning("Audio device '%s' not found", name_substring)
        return None
    finally:
        p.terminate()


class InputAudioResampler(FrameProcessor):
    """Resamples input audio from mic sample rate to pipeline sample rate.

    Required when microphone only supports 44100/48000 Hz but VAD needs 16000 Hz.
    """

    def __init__(self, input_rate: int = 44100, output_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self._input_rate = input_rate
        self._output_rate = output_rate
        self._resampler = create_stream_resampler()
        self._frame_count = 0

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._frame_count += 1
            # Log first frames and then every 500 frames (about every 10 seconds)
            if self._frame_count <= 3 or self._frame_count % 500 == 0:
                logger.info("Audio frame #%d received: %d bytes @ %d Hz",
                            self._frame_count, len(frame.audio), frame.sample_rate)
            # Resample audio from mic rate to pipeline rate
            if frame.sample_rate != self._output_rate:
                resampled_audio = await self._resampler.resample(
                    frame.audio,
                    frame.sample_rate,
                    self._output_rate,
                )
                # Create new frame with resampled audio
                frame = InputAudioRawFrame(
                    audio=resampled_audio,
                    sample_rate=self._output_rate,
                    num_channels=frame.num_channels,
                )

        await self.push_frame(frame, direction)


class VADProcessor(FrameProcessor):
    """Voice Activity Detection processor.

    Runs VAD on audio frames and emits speaking events.
    Gates VAD during bot speech to prevent echo feedback.
    Must be placed after resampler (expects 16000 Hz audio).
    """

    def __init__(self, sample_rate: int = 16000, min_volume: float = 0.01, gate_delay_ms: float = 300.0, **kwargs):
        super().__init__(**kwargs)
        vad_params = VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.8,
            min_volume=min_volume,
        )
        self._vad = SileroVADAnalyzer(sample_rate=sample_rate, params=vad_params)
        self._vad.set_sample_rate(sample_rate)
        self._is_speaking = False
        self._frame_count = 0
        self._bot_speaking = False
        self._bot_stop_time = 0.0
        self._gate_delay_s = gate_delay_ms / 1000.0
        logger.info("VAD initialized: min_volume=%.2f, gate_delay=%.0fms", min_volume, gate_delay_ms)

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        import time

        # Track bot speaking state (frames come upstream from transport)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            logger.info("VAD: Bot started speaking - gating VAD")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stop_time = time.time()
            logger.info("VAD: Bot stopped speaking - gating for %.0fms", self._gate_delay_s * 1000)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            self._frame_count += 1

            # Gate VAD during bot speech and shortly after
            if self._bot_speaking or time.time() < self._bot_stop_time + self._gate_delay_s:
                await self.push_frame(frame, direction)
                return

            if self._frame_count <= 5 or self._frame_count % 100 == 0:
                logger.debug("VAD received audio frame #%d: %d bytes", self._frame_count, len(frame.audio))

            vad_state = await self._vad.analyze_audio(frame.audio)

            if vad_state in (VADState.STARTING, VADState.SPEAKING) and not self._is_speaking:
                self._is_speaking = True
                logger.info("=" * 40)
                logger.info("VAD: User STARTED speaking (state=%s)", vad_state.name)
                logger.info("=" * 40)
                await self.push_frame(VADUserStartedSpeakingFrame(), direction)
            elif vad_state == VADState.QUIET and self._is_speaking:
                self._is_speaking = False
                logger.info("VAD: User stopped speaking - triggering STT")
                await self.push_frame(VADUserStoppedSpeakingFrame(), direction)

        await self.push_frame(frame, direction)


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
    from pipecat.services.ollama.llm import OLLamaLLMService

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


async def run_voice_pipeline(
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    input_device_name: Optional[str] = None,
    output_device_name: Optional[str] = None,
    input_sample_rate: int = 44100,
    stt_model: str = "nemotron",
    ollama_model: str = "gpt-oss:20b",
    ollama_url: str = "http://localhost:11434",
    tts_voice: str = "am_michael",
    tts_speed: float = 1.15,
    device: str = "cuda",
    session_id: Optional[str] = None,
    wake_word_enabled: bool = True,
    wake_phrases: Optional[list[str]] = None,
    wake_keepalive_secs: float = 15.0,
):
    """
    Run the full Pipecat voice pipeline with local audio.

    Uses SileroVAD for voice activity detection.

    Args:
        input_device: Audio input device index (deprecated, use input_device_name)
        output_device: Audio output device index (deprecated, use output_device_name)
        input_device_name: Audio input device name substring (e.g. "SoloCast")
        output_device_name: Audio output device name substring
        input_sample_rate: Mic native sample rate (44100 for SoloCast, 16000 if mic supports it)
    """
    # Resolve device names to indices (name takes priority over index)
    if input_device_name:
        input_device = find_audio_device_by_name(input_device_name, input_device=True)
        if input_device is None:
            logger.error("Input device '%s' not found, using default", input_device_name)
    if output_device_name:
        output_device = find_audio_device_by_name(output_device_name, input_device=False)
        if output_device is None:
            logger.error("Output device '%s' not found, using default", output_device_name)

    logger.info("Starting Pipecat voice pipeline")
    logger.info("  Input device: %s (name=%s)", input_device, input_device_name)
    logger.info("  Output device: %s (name=%s)", output_device, output_device_name)
    logger.info("  Input sample rate: %d Hz", input_sample_rate)
    logger.info("  STT: %s", stt_model)
    logger.info("  LLM: %s", ollama_model)

    # Create transport WITHOUT VAD at transport level
    # Many USB mics (like SoloCast) only support 44100/48000 Hz, but VAD needs 16000 Hz
    # We'll capture at native rate, resample if needed, then apply VAD in pipeline
    transport_params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_channels=1,
        audio_out_channels=1,
        audio_in_sample_rate=input_sample_rate,
        audio_out_sample_rate=24000,  # Kokoro TTS output rate
        vad_enabled=False,  # VAD will be handled after resampling
        input_device_index=input_device,
        output_device_index=output_device,
    )
    logger.info("Transport params: audio_in=%s, audio_out=%s, vad=%s, in_rate=%s, out_rate=%s",
                transport_params.audio_in_enabled, transport_params.audio_out_enabled,
                transport_params.vad_enabled, transport_params.audio_in_sample_rate,
                transport_params.audio_out_sample_rate)
    logger.info("Transport params: in_device=%s, out_device=%s",
                transport_params.input_device_index, transport_params.output_device_index)
    transport = LocalAudioTransport(transport_params)
    logger.info("LocalAudioTransport created")

    # Resampler to convert mic rate to 16000 Hz for STT/VAD (skipped if already 16kHz)
    audio_resampler = InputAudioResampler(
        input_rate=input_sample_rate,
        output_rate=16000,
    )

    # VAD processor (runs on resampled 16000 Hz audio)
    vad_processor = VADProcessor()

    # Create STT based on stt_model parameter
    if stt_model == "parakeet":
        stt = ParakeetSTTService(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device=device,
        )
    else:
        # Default to Nemotron (streaming-optimized)
        stt = NemotronSTTService(
            model_name="nvidia/nemotron-speech-streaming-en-0.6b",
            device=device,
        )

    # Create Agent processor (replaces LLM + context aggregators)
    # Routes through AtlasAgent for full capabilities: tools, devices, memory
    agent_processor = AtlasAgentProcessor(session_id=session_id)
    logger.info("  Agent: AtlasAgentProcessor (session=%s)", session_id)

    # Create TTS
    tts = StreamingKokoroTTSService(
        voice=tts_voice,
        speed=tts_speed,
        device=device,
    )

    # Create wake word filter (Pipecat's built-in WakeCheckFilter)
    # Only passes transcriptions through after wake phrase detected
    if wake_phrases is None:
        wake_phrases = ["hey atlas", "atlas", "hey assistant"]

    wake_filter = None
    if wake_word_enabled:
        wake_filter = WakeCheckFilter(
            wake_phrases=wake_phrases,
            keepalive_timeout=wake_keepalive_secs,
        )
        logger.info("  Wake word: enabled (phrases=%s, keepalive=%.1fs)",
                   wake_phrases, wake_keepalive_secs)
    else:
        logger.info("  Wake word: disabled (always listening)")

    # Build pipeline
    # Audio flow: Mic -> Resampler -> VAD -> STT -> [WakeFilter] -> Agent -> TTS -> Speaker
    pipeline_components = [
        transport.input(),
        audio_resampler,  # 44100 Hz -> 16000 Hz
        vad_processor,    # Speech detection at 16kHz
        stt,
    ]

    # Add wake filter if enabled
    if wake_filter:
        pipeline_components.append(wake_filter)

    pipeline_components.extend([
        agent_processor,  # Routes through AtlasAgent for tools, devices, memory
        tts,
        transport.output(),
    ])

    pipeline = Pipeline(pipeline_components)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            audio_in_sample_rate=16000,   # After resampling
            audio_out_sample_rate=24000,  # Kokoro TTS rate
        ),
        idle_timeout_secs=None,  # Disable idle timeout - always listen
    )

    runner = PipelineRunner()
    logger.info("=" * 60)
    logger.info("PIPECAT VOICE PIPELINE ACTIVE")
    logger.info("=" * 60)
    if wake_word_enabled:
        logger.info("Say '%s' to activate, then give your command.", wake_phrases[0])
        logger.info("Keepalive: %.0f seconds after last interaction.", wake_keepalive_secs)
    else:
        logger.info("Wake word disabled - always listening for commands.")
    await runner.run(task)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_test_pipeline())
