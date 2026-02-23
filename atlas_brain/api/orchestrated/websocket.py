"""
Orchestrated WebSocket endpoint for remote voice conversations.

Handles the full voice pipeline over WebSocket:
  Audio in -> ASR -> Agent -> TTS -> Audio out (base64 WAV)
  Text in  -> Agent -> optional TTS -> Text + Audio out

Designed for the Atlas web UI and mobile app to connect
over LAN/Tailscale without running a local voice pipeline.

Commands:
  stop_recording  - Finalize ASR and process the utterance
  send_text       - Process a text message (skips ASR)
  set_privacy     - Toggle privacy mode (text-only responses, no TTS)
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ...agents.interface import process_with_fallback
from ...config import settings
from .asr_bridge import AsrBridge
from .tts_bridge import synthesize_to_wav_bytes

logger = logging.getLogger("atlas.api.orchestrated.websocket")

router = APIRouter(prefix="/ws/orchestrated", tags=["orchestrated"])

# Track active sessions for concurrency limiting
_active_sessions: dict[str, "OrchestratedConnection"] = {}


async def _broadcast(state: str, **kwargs) -> None:
    """Broadcast a state message to all connected UI sessions."""
    if not _active_sessions:
        return
    msg = {"state": state, **kwargs}
    for conn in list(_active_sessions.values()):
        try:
            await conn.send(msg)
        except Exception:
            pass


def broadcast_from_thread(loop: asyncio.AbstractEventLoop, state: str, **kwargs) -> None:
    """Thread-safe broadcast called from the voice pipeline thread."""
    if not _active_sessions:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(state, **kwargs), loop)


class OrchestratedConnection:
    """Represents a single orchestrated voice session."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.session_id = str(uuid4())
        self.connected_at = time.time()
        self._send_lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task] = set()
        self._asr: Optional[AsrBridge] = None
        self._closed = False
        self.privacy_mode: bool = False

    async def send(self, message: dict[str, Any]) -> None:
        """Send JSON message, protected by lock for Starlette safety."""
        if self._closed:
            return
        async with self._send_lock:
            try:
                await self.websocket.send_json(message)
            except Exception:
                self._closed = True

    async def send_state(self, state: str, **kwargs) -> None:
        """Send a state update message."""
        msg = {"state": state, **kwargs}
        await self.send(msg)

    def _spawn_task(self, coro, *, name: str | None = None) -> asyncio.Task:
        """Spawn a tracked background task."""
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def cancel_pending(self) -> None:
        """Cancel all pending tasks on disconnect."""
        tasks = list(self._pending_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_tasks.clear()

    async def close_asr(self) -> None:
        """Close ASR bridge connection."""
        if self._asr:
            await self._asr.close()
            self._asr = None


@router.websocket("")
async def orchestrated_websocket(websocket: WebSocket):
    """
    Orchestrated voice WebSocket endpoint.

    Protocol:
      Client -> Server:
        Binary:  Raw PCM (16kHz, 16-bit int, mono)
        JSON:    {"command": "stop_recording"}
        JSON:    {"command": "send_text", "text": "..."}
        JSON:    {"command": "set_privacy", "enabled": true/false}

      Server -> Client:
        {"state": "idle"}
        {"state": "recording"}
        {"state": "transcript", "text": "..."}
        {"state": "transcribing"}
        {"state": "processing"}
        {"state": "responding"}
        {"state": "response", "text": "...", "audio_base64": "..."}
        {"state": "error", "message": "..."}
    """
    # Check concurrency limit
    max_sessions = settings.orchestrated.max_concurrent_sessions
    if len(_active_sessions) >= max_sessions:
        await websocket.close(code=1013, reason="Too many concurrent sessions")
        return

    await websocket.accept()
    conn = OrchestratedConnection(websocket)
    _active_sessions[conn.session_id] = conn

    logger.info(
        "Orchestrated session started: %s (active: %d)",
        conn.session_id[:8], len(_active_sessions),
    )

    try:
        # Set up ASR bridge with partial transcript callback
        async def on_partial(text: str):
            await conn.send_state("transcript", text=text)

        conn._asr = AsrBridge(on_partial=on_partial)

        try:
            await conn._asr.connect()
        except ConnectionError as e:
            logger.error("ASR connection failed: %s", e)
            await conn.send_state("error", message=f"ASR unavailable: {e}")
            return

        # Signal ready
        await conn.send_state("idle")

        # Send a welcome system_event so the feed is non-empty immediately
        import uuid as _uuid
        from datetime import datetime as _dt, timezone as _tz
        await conn.send({
            "state": "system_event",
            "id": str(_uuid.uuid4()),
            "ts": _dt.now(_tz.utc).isoformat(),
            "category": "llm",
            "level": "info",
            "message": "Atlas connected",
        })

        # Main message loop
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Binary frame: PCM audio
            if "bytes" in message and message["bytes"]:
                pcm_data = message["bytes"]
                await conn._asr.send_audio(pcm_data)
                continue

            # Text frame: JSON command
            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await conn.send_state("error", message="Invalid JSON")
                    continue

                command = data.get("command", "")

                if command == "stop_recording":
                    # Process the completed utterance
                    conn._spawn_task(
                        _handle_stop_recording(conn),
                        name=f"process-{conn.session_id[:8]}",
                    )

                elif command == "set_privacy":
                    conn.privacy_mode = bool(data.get("enabled", False))
                    logger.info(
                        "Session %s privacy mode: %s",
                        conn.session_id[:8], conn.privacy_mode,
                    )

                elif command == "send_text":
                    text = (data.get("text") or "").strip()
                    if not text:
                        await conn.send_state("error", message="Empty text")
                    else:
                        await conn.send_state("transcript", text=text)
                        conn._spawn_task(
                            _handle_send_text(conn, text),
                            name=f"text-{conn.session_id[:8]}",
                        )

                else:
                    await conn.send_state(
                        "error", message=f"Unknown command: {command}"
                    )

    except WebSocketDisconnect:
        logger.info("Orchestrated session disconnected: %s", conn.session_id[:8])
    except Exception as e:
        logger.exception("Orchestrated session error: %s", e)
    finally:
        await conn.cancel_pending()
        await conn.close_asr()
        _active_sessions.pop(conn.session_id, None)
        logger.info(
            "Orchestrated session ended: %s (active: %d)",
            conn.session_id[:8], len(_active_sessions),
        )


async def _process_and_respond(conn: OrchestratedConnection, transcript: str) -> None:
    """Shared pipeline: agent -> optional TTS -> send response."""
    await conn.send_state("processing")

    try:
        result = await asyncio.wait_for(
            process_with_fallback(
                input_text=transcript,
                session_id=conn.session_id,
                input_type="text" if conn.privacy_mode else "voice",
            ),
            timeout=settings.orchestrated.agent_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Agent timed out for session %s", conn.session_id[:8])
        await conn.send_state("error", message="Processing timed out")
        return

    response_text = result.response_text or "I couldn't generate a response."
    logger.info(
        "Session %s response: %s",
        conn.session_id[:8], response_text[:80],
    )

    # TTS -- skip in privacy mode
    audio_base64 = ""
    if not conn.privacy_mode:
        await conn.send_state("responding")
        try:
            wav_bytes = await asyncio.wait_for(
                synthesize_to_wav_bytes(response_text),
                timeout=settings.orchestrated.tts_timeout,
            )
            audio_base64 = base64.b64encode(wav_bytes).decode("ascii")
        except asyncio.TimeoutError:
            logger.warning("TTS timed out for session %s", conn.session_id[:8])
        except Exception as e:
            logger.error("TTS failed for session %s: %s", conn.session_id[:8], e)

    await conn.send_state("response", text=response_text, audio_base64=audio_base64)


async def _handle_stop_recording(conn: OrchestratedConnection) -> None:
    """Handle the full pipeline after user stops recording."""
    try:
        # 1. Finalize ASR
        await conn.send_state("transcribing")
        transcript = await conn._asr.finalize()

        if not transcript or not transcript.strip():
            logger.info("Empty transcript, returning to idle")
            await conn._asr.reset()
            await conn.send_state("idle")
            return

        # Send final transcript
        await conn.send_state("transcript", text=transcript)
        logger.info(
            "Session %s transcript: %s",
            conn.session_id[:8], transcript[:80],
        )

        # 2. Process and respond
        await _process_and_respond(conn, transcript)

        # 3. Reset ASR for next utterance
        await conn._asr.reset()
        await conn.send_state("idle")

    except asyncio.CancelledError:
        logger.debug("Processing cancelled for session %s", conn.session_id[:8])
    except Exception as e:
        logger.exception("Pipeline error for session %s: %s", conn.session_id[:8], e)
        try:
            await conn.send_state("error", message=str(e))
            if conn._asr:
                await conn._asr.reset()
            await conn.send_state("idle")
        except Exception:
            pass


async def _handle_send_text(conn: OrchestratedConnection, text: str) -> None:
    """Handle a text message (no ASR involved)."""
    try:
        await _process_and_respond(conn, text)
        await conn.send_state("idle")
    except asyncio.CancelledError:
        logger.debug("Text processing cancelled for session %s", conn.session_id[:8])
    except Exception as e:
        logger.exception(
            "Text pipeline error for session %s: %s", conn.session_id[:8], e
        )
        try:
            await conn.send_state("error", message=str(e))
            await conn.send_state("idle")
        except Exception:
            pass
