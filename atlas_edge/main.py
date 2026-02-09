"""
Atlas Edge - Main entry point for edge devices.

Runs the voice pipeline on Jetson edge devices with:
- Local device control
- Brain escalation for complex queries
- Offline fallback
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Load environment variables
# .env.local overrides .env for machine-specific settings (API keys, ports, etc.)
from dotenv import load_dotenv

_env_root = Path(__file__).parent.parent
load_dotenv(_env_root / ".env", override=True)
load_dotenv(_env_root / ".env.local", override=True)

from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("atlas.edge.main")


class EdgeApplication:
    """
    Main edge device application.

    Manages:
    - Voice pipeline
    - Brain connectivity
    - Home Assistant connection
    - Graceful shutdown
    """

    def __init__(self):
        """Initialize the edge application."""
        self._pipeline = None
        self._brain_connection = None
        self._ha_client = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def startup(self) -> None:
        """Initialize all components."""
        logger.info("Atlas Edge starting up...")
        logger.info("Location: %s (%s)", settings.location_name, settings.location_id)

        # Initialize Home Assistant connection
        if settings.homeassistant.enabled and settings.homeassistant.token:
            try:
                from .capabilities.homeassistant import get_homeassistant

                self._ha_client = await get_homeassistant()
                logger.info("Home Assistant connected")
            except Exception as e:
                logger.error("Failed to connect to Home Assistant: %s", e)

        # Initialize brain connection
        if settings.brain.enabled:
            try:
                from .brain.connection import get_brain_connection

                self._brain_connection = await get_brain_connection()

                # Set up callbacks
                self._brain_connection.on_connected(self._on_brain_connected)
                self._brain_connection.on_disconnected(self._on_brain_disconnected)

                # Try to connect
                await self._brain_connection.connect()
            except Exception as e:
                logger.warning("Failed to connect to brain: %s", e)
                logger.info("Running in offline mode")

        # Initialize voice pipeline
        try:
            from .pipeline.voice_pipeline import get_voice_pipeline

            self._pipeline = get_voice_pipeline()

            # Set up callbacks
            self._pipeline.on_wake_word(self._on_wake_word)
            self._pipeline.on_transcript(self._on_transcript)
            self._pipeline.on_response(self._on_response)

            logger.info("Voice pipeline initialized")
        except Exception as e:
            logger.error("Failed to initialize voice pipeline: %s", e)
            raise

        logger.info("Atlas Edge ready")
        self._running = True

    async def shutdown(self) -> None:
        """Clean up all components."""
        logger.info("Atlas Edge shutting down...")
        self._running = False

        # Clean up skills (cancel active timers, etc.)
        from .skills import shutdown_skills
        shutdown_skills()

        # Disconnect from brain
        if self._brain_connection:
            await self._brain_connection.disconnect()

        # Disconnect from Home Assistant
        if self._ha_client:
            await self._ha_client.disconnect()

        logger.info("Atlas Edge stopped")

    def _on_wake_word(self) -> None:
        """Handle wake word detection."""
        logger.info("Wake word detected!")

    def _on_transcript(self, text: str) -> None:
        """Handle transcript ready."""
        logger.info("Transcript: %s", text)

    def _on_response(self, text: str) -> None:
        """Handle response ready."""
        logger.info("Response: %s", text)

    def _on_brain_connected(self) -> None:
        """Handle brain connection established."""
        logger.info("Brain connected - full capabilities available")

    def _on_brain_disconnected(self) -> None:
        """Handle brain disconnection."""
        logger.warning("Brain disconnected - running in offline mode")

    async def run_interactive(self) -> None:
        """Run in interactive text mode (for testing)."""
        await self.startup()

        print("\nAtlas Edge Interactive Mode")
        print("Type commands to test, or 'quit' to exit\n")

        try:
            while self._running:
                try:
                    # Get input
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, input, "You: "
                    )

                    if text.lower() in ("quit", "exit", "q"):
                        break

                    if not text.strip():
                        continue

                    # Process through pipeline
                    result = await self._pipeline.process_text(
                        text,
                        speak_response=False,
                    )

                    print(f"Atlas: {result.response_text}")
                    print(
                        f"  (local={result.handled_locally}, "
                        f"escalated={result.escalated}, "
                        f"time={result.total_ms:.0f}ms)"
                    )
                    print()

                except EOFError:
                    break

        except KeyboardInterrupt:
            print("\nInterrupted")

        finally:
            await self.shutdown()

    async def run_voice(self) -> None:
        """Run with voice input (production mode)."""
        await self.startup()

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        logger.info("Voice pipeline running. Say 'Hey Atlas' to activate.")

        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()

        finally:
            await self.shutdown()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Atlas Edge Device")
    parser.add_argument(
        "--mode",
        choices=["voice", "interactive"],
        default="interactive",
        help="Running mode (default: interactive)",
    )
    args = parser.parse_args()

    app = EdgeApplication()

    if args.mode == "interactive":
        asyncio.run(app.run_interactive())
    else:
        asyncio.run(app.run_voice())


if __name__ == "__main__":
    main()
