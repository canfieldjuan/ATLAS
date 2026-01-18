#!/usr/bin/env python3
"""
Atlas Voice Runner - Always-on voice interface daemon.

Continuously captures audio from the microphone and streams it to
Atlas Brain via WebSocket. Handles wake word detection, speech
recognition, and TTS response playback.

Usage:
    python -m atlas_voice.runner              # Start daemon
    python -m atlas_voice.runner --list       # List audio devices
    python -m atlas_voice.runner --test       # Test audio devices
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .audio_capture import AudioCapture, CaptureConfig, list_devices as list_input_devices
from .audio_output import AudioOutput, OutputConfig, list_devices as list_output_devices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("atlas.voice.runner")


@dataclass
class RunnerConfig:
    """Configuration for the voice runner."""

    # Server connection
    atlas_url: str = "ws://localhost:8000/api/v1/ws/orchestrated"

    # Audio settings
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    sample_rate: int = 16000
    input_gain: float = 1.0  # Software gain multiplier for quiet mics

    # Behavior
    require_wake_word: bool = True
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0

    # Feedback
    play_responses: bool = True
    show_transcripts: bool = True


class VoiceRunner:
    """
    Always-on voice capture and playback daemon.

    Connects to Atlas Brain via WebSocket, streams microphone audio,
    and plays TTS responses through speakers.
    """

    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()

        # Audio components
        self._capture = AudioCapture(CaptureConfig(
            sample_rate=self.config.sample_rate,
            device=self.config.input_device,
            gain=self.config.input_gain,
        ))
        self._output = AudioOutput(OutputConfig(
            device=self.config.output_device,
        ))

        # State
        self._running = False
        self._connected = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_state = "idle"
        self._muted = False  # Mute mic during TTS playback

    async def run(self) -> None:
        """
        Main run loop.

        Connects to WebSocket and processes audio continuously.
        """
        self._running = True
        logger.info("Atlas Voice Runner starting...")
        logger.info("Server: %s", self.config.atlas_url)
        logger.info("Wake word required: %s", self.config.require_wake_word)

        while self._running:
            try:
                await self._connect_and_run()
            except ConnectionClosed as e:
                logger.warning("WebSocket connection closed: %s", e)
            except Exception as e:
                logger.error("Connection error: %s", e)

            if self._running and self.config.auto_reconnect:
                logger.info("Reconnecting in %.1f seconds...", self.config.reconnect_delay)
                await asyncio.sleep(self.config.reconnect_delay)
            else:
                break

        logger.info("Atlas Voice Runner stopped")

    async def _connect_and_run(self) -> None:
        """Connect to WebSocket and run the audio loop."""
        logger.info("Connecting to Atlas Brain...")

        async with websockets.connect(
            self.config.atlas_url,
            ping_interval=30,  # Send ping every 30 seconds
            ping_timeout=120,  # Wait up to 120 seconds for pong
            max_size=10 * 1024 * 1024,  # 10MB max frame size for TTS audio
        ) as ws:
            self._ws = ws
            self._connected = True
            logger.info("Connected!")

            # Configure orchestrator
            await ws.send(json.dumps({
                "command": "config",
                "wake_word_enabled": self.config.require_wake_word,
            }))

            # Start audio capture
            self._capture.start()

            # Create tasks for sending and receiving
            send_task = asyncio.create_task(self._send_audio_loop())
            recv_task = asyncio.create_task(self._receive_loop())

            try:
                # Wait for either task to complete (or fail)
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            finally:
                self._capture.stop()
                self._connected = False
                self._ws = None

    async def _send_audio_loop(self) -> None:
        """Continuously send audio frames to WebSocket."""
        logger.debug("Audio send loop started")
        loop = asyncio.get_event_loop()

        while self._running and self._connected:
            # Skip sending while TTS is playing to prevent self-triggering
            if self._muted:
                await asyncio.sleep(0.05)
                continue

            # Get audio frame in thread pool to avoid blocking event loop
            frame = await loop.run_in_executor(
                None, lambda: self._capture.get_frame(timeout=0.1)
            )
            if frame:
                try:
                    await self._ws.send(frame)
                except Exception as e:
                    logger.error("Send error: %s", e)
                    break

        logger.debug("Audio send loop stopped")

    async def _receive_loop(self) -> None:
        """Handle incoming WebSocket messages."""
        logger.debug("Receive loop started")

        try:
            async for message in self._ws:
                await self._handle_message(message)
        except ConnectionClosed:
            pass

        logger.debug("Receive loop stopped")

    async def _handle_message(self, message: str) -> None:
        """Handle a WebSocket message from Atlas Brain."""
        try:
            data = json.loads(message)
            state = data.get("state", "")
            if state == "response":
                logger.info("GOT RESPONSE! text=%s, has_audio=%s",
                           data.get("text", "")[:50] if data.get("text") else None,
                           "audio_base64" in data)
            logger.debug("Received message: state=%s", state)

            # Track state changes
            if state and state != self._current_state:
                self._current_state = state
                self._log_state_change(state, data)

            # Handle specific message types
            if state == "transcript" and self.config.show_transcripts:
                text = data.get("text", "")
                print(f"\n[You] {text}")

            elif state == "response":
                # Show response text
                text = data.get("text", "")
                if text and self.config.show_transcripts:
                    print(f"[Atlas] {text}")

                # Play audio response (mute mic to prevent self-triggering)
                audio_b64 = data.get("audio_base64")
                if audio_b64 and self.config.play_responses:
                    audio_bytes = base64.b64decode(audio_b64)
                    # Mute before playback to prevent wake word self-trigger
                    self._muted = True
                    logger.info("PLAYBACK: Muted mic, starting TTS playback (%d bytes)", len(audio_bytes))
                    try:
                        loop = asyncio.get_event_loop()
                        logger.info("PLAYBACK: Playing audio...")
                        await loop.run_in_executor(
                            None, lambda: self._output.play(audio_bytes, blocking=True)
                        )
                        logger.info("PLAYBACK: Audio done, waiting for reverb decay")
                        # Wait for residual audio to decay (room reverb, speaker resonance)
                        await asyncio.sleep(0.6)
                        logger.info("PLAYBACK: Draining audio buffer")
                        # Drain buffered audio that may contain TTS echo
                        # Limit to prevent infinite loop if capture keeps producing
                        drained = 0
                        max_drain = 500  # ~5 seconds of audio at typical frame rate
                        for _ in range(3):  # Multiple drain passes
                            pass_count = 0
                            while pass_count < max_drain // 3:
                                frame = self._capture.get_frame(timeout=0.01)
                                if not frame:
                                    break
                                drained += 1
                                pass_count += 1
                            await asyncio.sleep(0.05)
                        logger.info("PLAYBACK: Drained %d frames", drained)
                    finally:
                        self._muted = False
                        logger.info("PLAYBACK: Unmuted mic, ready for next command")

            elif state == "error":
                error_msg = data.get("message", "Unknown error")
                logger.error("Pipeline error: %s", error_msg)

        except json.JSONDecodeError:
            logger.warning("Invalid JSON message: %s", message[:100])

    def _log_state_change(self, state: str, data: dict) -> None:
        """Log state transitions for debugging."""
        if state == "listening":
            if data.get("wake_word_enabled"):
                logger.info("Listening for wake word...")
            else:
                logger.info("Listening...")
        elif state == "wake_detected":
            logger.info("Wake word detected!")
        elif state == "recording":
            logger.debug("Recording speech...")
        elif state == "transcribing":
            logger.debug("Transcribing...")
        elif state == "processing":
            logger.debug("Processing...")
        elif state == "executing":
            logger.debug("Executing action...")

    def stop(self) -> None:
        """Stop the runner."""
        logger.info("Stopping...")
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())


def list_devices():
    """List all audio devices."""
    print("\n=== Audio Input Devices (Microphones) ===")
    for d in list_input_devices():
        default = " [DEFAULT]" if d["is_default"] else ""
        print(f"  {d['index']}: {d['name']}{default}")

    print("\n=== Audio Output Devices (Speakers) ===")
    for d in list_output_devices():
        default = " [DEFAULT]" if d["is_default"] else ""
        print(f"  {d['index']}: {d['name']}{default}")
    print()


def test_audio(input_device: Optional[int], output_device: Optional[int]):
    """Test audio devices with a quick recording and playback."""
    import time

    print("\n=== Audio Test ===")

    # Test capture
    print("Recording 3 seconds of audio...")
    capture = AudioCapture(CaptureConfig(device=input_device))
    capture.start()

    frames = []
    start = time.time()
    while time.time() - start < 3:
        frame = capture.get_frame(timeout=0.1)
        if frame:
            frames.append(frame)
    capture.stop()

    audio_data = b"".join(frames)
    print(f"Captured {len(audio_data)} bytes ({len(frames)} frames)")

    # Create WAV for playback
    import io
    import wave
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    wav_bytes = wav_buffer.getvalue()

    # Test playback
    print("Playing back...")
    output = AudioOutput(OutputConfig(device=output_device))
    output.play(wav_bytes)
    print("Done!")


async def main_async(config: RunnerConfig):
    """Async main entry point."""
    runner = VoiceRunner(config)

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        runner.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await runner.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Atlas Voice Runner - Always-on voice interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m atlas_voice.runner              # Start with defaults
  python -m atlas_voice.runner --no-wake    # No wake word required
  python -m atlas_voice.runner --list       # List audio devices
  python -m atlas_voice.runner --test       # Test audio devices
  python -m atlas_voice.runner --url ws://192.168.1.100:8000/api/v1/ws/orchestrated
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("ATLAS_URL", "ws://localhost:8000/api/v1/ws/orchestrated"),
        help="Atlas Brain WebSocket URL",
    )
    parser.add_argument(
        "--input",
        type=int,
        default=None,
        help="Input device index (microphone)",
    )
    parser.add_argument(
        "--output",
        type=int,
        default=None,
        help="Output device index (speakers)",
    )
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="Don't require wake word (always listening)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Don't play TTS responses",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test audio devices and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger("atlas").setLevel(logging.DEBUG)

    if args.list:
        list_devices()
        return

    if args.test:
        test_audio(args.input, args.output)
        return

    # Build configuration
    input_gain = float(os.environ.get("ATLAS_VOICE_INPUT_GAIN", "1.0"))
    config = RunnerConfig(
        atlas_url=args.url,
        input_device=args.input,
        output_device=args.output,
        require_wake_word=not args.no_wake,
        play_responses=not args.no_audio,
        input_gain=input_gain,
    )

    # Get wake word from environment for display
    wake_word = os.environ.get("ATLAS_WAKE_WORD", "Hey Jarvis")

    # Run
    print(f"""
    ============================================
              Atlas Voice Runner

       Say "{wake_word}" to activate
       (or use --no-wake for always-on)

       Press Ctrl+C to stop
    ============================================
    """)

    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
