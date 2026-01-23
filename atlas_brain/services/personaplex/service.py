"""
PersonaPlex WebSocket client service.

Connects to PersonaPlex server for speech-to-speech conversation.
"""

import asyncio
import logging
import ssl
import time
from typing import Callable, Optional
from urllib.parse import urlencode

import aiohttp

from .audio import AudioConverter
from .config import PersonaPlexConfig, get_personaplex_config

logger = logging.getLogger("atlas.personaplex")

MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02


class PersonaPlexService:
    """WebSocket client for PersonaPlex speech-to-speech service."""

    def __init__(self, config: Optional[PersonaPlexConfig] = None):
        self._config = config or get_personaplex_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._audio_converter = AudioConverter()
        self._connected = False
        self._text_callback: Optional[Callable[[str], None]] = None
        self._audio_callback: Optional[Callable[[bytes], None]] = None
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to PersonaPlex server."""
        return self._connected and self._ws is not None

    @property
    def audio_converter(self) -> AudioConverter:
        """Get the audio converter instance."""
        return self._audio_converter

    def set_text_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for received text tokens."""
        self._text_callback = callback

    def set_audio_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for received audio (Opus format)."""
        self._audio_callback = callback

    def _build_ws_url(
        self,
        text_prompt: Optional[str] = None,
        voice_prompt: Optional[str] = None,
    ) -> str:
        """Build WebSocket URL with query parameters."""
        scheme = "wss" if self._config.use_ssl else "ws"
        base = f"{scheme}://{self._config.host}:{self._config.port}/api/chat"

        # Ensure voice prompt has .pt extension
        vp = voice_prompt or self._config.voice_prompt
        if not vp.endswith(".pt"):
            vp = f"{vp}.pt"

        params = {
            "voice_prompt": vp,
            "text_seed": str(self._config.seed),
            "audio_seed": str(self._config.seed),
        }
        prompt = text_prompt or self._config.text_prompt
        if prompt:
            params["text_prompt"] = prompt
        return f"{base}?{urlencode(params)}"

    async def connect(
        self,
        text_prompt: Optional[str] = None,
        voice_prompt: Optional[str] = None,
    ) -> bool:
        """Connect to PersonaPlex server."""
        if self._connected:
            logger.warning("Already connected to PersonaPlex")
            return True

        url = self._build_ws_url(text_prompt, voice_prompt)
        logger.info("Connecting to PersonaPlex: %s", url.split("?")[0])

        try:
            ssl_ctx = None
            if self._config.use_ssl:
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE

            timeout = aiohttp.ClientTimeout(
                total=self._config.connect_timeout
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._ws = await self._session.ws_connect(url, ssl=ssl_ctx)

            # Wait for binary handshake, skip any text messages
            while True:
                msg = await self._ws.receive()
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if len(msg.data) > 0 and msg.data[0] == MSG_HANDSHAKE:
                        break
                    logger.error("Invalid handshake byte: %s", msg.data[0] if msg.data else None)
                    await self.disconnect()
                    return False
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    logger.debug("Received text during handshake: %s", msg.data[:100])
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.error("WebSocket closed during handshake")
                    await self.disconnect()
                    return False

            self._connected = True
            self._audio_converter.reset()
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("Connected to PersonaPlex at %.3f", time.time())
            return True

        except Exception as e:
            logger.error("Failed to connect to PersonaPlex: %s", e)
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from PersonaPlex server."""
        self._connected = False

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        logger.info("Disconnected from PersonaPlex")

    async def _receive_loop(self) -> None:
        """Background task to receive messages from PersonaPlex."""
        print("[PERSONAPLEX] Receive loop started", flush=True)
        logger.info("PersonaPlex receive loop started")
        try:
            async for msg in self._ws:
                print(f"[PERSONAPLEX] WS msg type: {msg.type}", flush=True)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", self._ws.exception())
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket closed by server")
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error in receive loop: %s", e)
        finally:
            self._connected = False

    async def _handle_message(self, data: bytes) -> None:
        """Handle a binary message from PersonaPlex."""
        if len(data) < 1:
            return

        msg_type = data[0]
        payload = data[1:]

        if msg_type == MSG_AUDIO:
            print(f"[PERSONAPLEX] RX audio: {len(payload)} bytes", flush=True)
            logger.info(
                "RX audio from PersonaPlex: %d bytes at %.3f",
                len(payload),
                time.time(),
            )
            if self._audio_callback is not None:
                try:
                    self._audio_callback(payload)
                except Exception as e:
                    logger.error("Audio callback error: %s", e, exc_info=True)
            else:
                logger.warning("Audio callback not set, dropping audio")
        elif msg_type == MSG_TEXT:
            text = payload.decode("utf-8", errors="replace")
            logger.info("RX text from PersonaPlex: %s", text[:50])
            if self._text_callback is not None:
                self._text_callback(text)

    async def send_audio(self, opus_data: bytes) -> bool:
        """Send Opus audio to PersonaPlex."""
        if not self._connected or self._ws is None:
            logger.warning("Cannot send audio: not connected")
            return False

        try:
            message = bytes([MSG_AUDIO]) + opus_data
            await self._ws.send_bytes(message)
            logger.info("TX %d bytes opus at %.3f", len(opus_data), time.time())
            return True
        except Exception as e:
            logger.error("Failed to send audio: %s", e)
            return False

    async def send_audio_mulaw(self, mulaw_data: bytes) -> bool:
        """Send mulaw audio (converted to Opus) to PersonaPlex."""
        try:
            opus_data = self._audio_converter.signalwire_to_personaplex(mulaw_data)
            if not opus_data:
                return True  # Buffering, no data to send yet
            return await self.send_audio(opus_data)
        except RuntimeError as e:
            logger.error("Audio conversion failed: %s", e)
            return False
