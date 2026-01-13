"""
Orchestrated audio pipeline API endpoint.

Provides a WebSocket endpoint that handles the full voice pipeline:
Wake Word → VAD → STT → Intent/LLM → Action → TTS
"""

import asyncio
import base64
import json
import logging
from typing import Optional, Set
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..orchestration import Orchestrator, PipelineState
from ..orchestration.context import get_context
from ..agents import create_atlas_agent
from ..storage import db_settings
from ..storage.database import get_db_pool
from ..storage.repositories.session import get_session_repo

logger = logging.getLogger("atlas.api.orchestration")

router = APIRouter(tags=["orchestration"])


class ConnectionManager:
    """Manages WebSocket connections and broadcasts status to all clients."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Add a new connection."""
        async with self._lock:
            self.active_connections.add(websocket)
        logger.debug("Client connected, total: %d", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket):
        """Remove a connection."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.debug("Client disconnected, total: %d", len(self.active_connections))

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        if not self.active_connections:
            logger.debug("No active connections for broadcast")
            return
        state = message.get("state", "unknown")
        disconnected = []
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning("Broadcast failed for client: %s", e)
                    disconnected.append(connection)
            for conn in disconnected:
                self.active_connections.discard(conn)


# Global connection manager for broadcasting state updates
connection_manager = ConnectionManager()


@router.websocket("/ws/orchestrated")
async def orchestrated_audio_stream(websocket: WebSocket):
    """
    Full voice pipeline WebSocket endpoint.

    Accepts streaming audio and processes through:
    1. Wake word detection (if enabled)
    2. Voice activity detection
    3. Speech-to-text
    4. Intent parsing / LLM reasoning
    5. Action execution
    6. TTS response generation

    Message Protocol:
    - Client sends: Binary audio frames (16-bit, 16kHz, mono PCM)
    - Client sends: JSON commands {"command": "reset|cancel|config", ...}
    - Server sends: JSON status updates {"state": "...", "data": {...}}

    Status Updates:
    - {"state": "listening"}
    - {"state": "recording", "duration_ms": 1500}
    - {"state": "transcribing"}
    - {"state": "transcript", "text": "turn on the lights"}
    - {"state": "processing"}
    - {"state": "intent", "action": "turn_on", "target": "lights"}
    - {"state": "executing"}
    - {"state": "response", "text": "Done.", "audio_base64": "..."}
    - {"state": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("Orchestrated WebSocket connected")

    # Register with connection manager for broadcasts
    await connection_manager.connect(websocket)

    # Keepalive task to prevent WebSocket timeout
    keepalive_interval = 15  # Send keepalive every 15 seconds
    keepalive_task: Optional[asyncio.Task] = None

    async def keepalive_loop():
        """Send periodic keepalive pings to prevent timeout."""
        logger.debug("Keepalive loop started")
        try:
            while True:
                await asyncio.sleep(keepalive_interval)
                try:
                    await websocket.send_json({"type": "ping"})
                    logger.debug("Keepalive ping sent")
                except Exception as e:
                    logger.debug("Keepalive send failed: %s", e)
                    break
        except asyncio.CancelledError:
            logger.debug("Keepalive loop cancelled")
            pass

    # Session management - get from query params or create new
    session_id: Optional[str] = None
    user_id_param: Optional[str] = websocket.query_params.get("user_id")
    terminal_id: Optional[str] = websocket.query_params.get("terminal_id")

    # Initialize session if database is enabled
    if db_settings.enabled:
        pool = get_db_pool()
        if pool.is_initialized:
            try:
                session_repo = get_session_repo()
                user_uuid = UUID(user_id_param) if user_id_param else None
                session = await session_repo.get_or_create_session(
                    user_id=user_uuid,
                    terminal_id=terminal_id,
                )
                session_id = str(session.id)
                logger.info(
                    "Voice session: %s (user=%s, terminal=%s)",
                    session_id, user_id_param, terminal_id
                )
            except Exception as e:
                logger.warning("Failed to create session: %s", e)

    # Create agent for reasoning and orchestrator for audio pipeline
    agent = create_atlas_agent(session_id=session_id)
    orchestrator = Orchestrator(session_id=session_id, agent=agent)
    context = get_context()

    # Load conversation history if session exists
    if session_id:
        try:
            history = await orchestrator.load_session_history(limit=10)
            for turn in history:
                context.add_conversation_turn(
                    turn.role,
                    turn.content,
                    speaker_id=turn.speaker_id,
                )
            if history:
                logger.info("Loaded %d turns from session history", len(history))
        except Exception as e:
            logger.warning("Failed to load session history: %s", e)

    # Track state for status updates
    last_state: Optional[PipelineState] = None
    is_connected = True

    async def send_status(state: str, **kwargs):
        """Send a status update to this client only (no broadcast)."""
        if not is_connected:
            return
        try:
            message = {"state": state, **kwargs}
            # Send only to this specific WebSocket connection
            await websocket.send_json(message)
        except Exception:
            pass

    async def on_state_change(event, old_state, new_state, ctx):
        """Callback for pipeline state changes."""
        nonlocal last_state
        if new_state and new_state != last_state:
            last_state = new_state
            await send_status(new_state.name.lower())

    async def on_transcript(text: str):
        """Callback when transcription is complete."""
        await send_status("transcript", text=text)
        # Add to conversation context
        context.add_conversation_turn("user", text)

    async def on_response(result):
        """Callback when response is ready."""
        logger.info("on_response callback: text=%s, has_audio=%s",
                    result.response_text[:50] if result.response_text else None,
                    result.response_audio is not None)
        response_data = {
            "text": result.response_text,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "follow_up_mode": orchestrator.in_follow_up_mode,
        }
        if result.response_audio:
            response_data["audio_base64"] = base64.b64encode(result.response_audio).decode()
        if result.intent:
            response_data["intent"] = {
                "action": result.intent.action,
                "target_type": result.intent.target_type,
                "target_name": result.intent.target_name,
                "confidence": result.intent.confidence,
            }
        if result.action_results:
            response_data["actions"] = result.action_results

        await send_status("response", **response_data)
        logger.info("Sent response status with %d bytes audio", len(response_data.get("audio_base64", "")))

        # Add to conversation context
        if result.response_text:
            context.add_conversation_turn("assistant", result.response_text)

    # Set up callbacks
    orchestrator.set_callbacks(
        on_state_change=on_state_change,
        on_transcript=on_transcript,
        on_response=on_response,
    )

    # Send initial status
    await send_status(
        "listening",
        follow_up_mode=orchestrator.in_follow_up_mode,
        context=context.build_context_dict(),
    )

    # Start keepalive task to prevent WebSocket timeout
    keepalive_task = asyncio.create_task(keepalive_loop())
    logger.info("Keepalive task started (interval=%ds)", keepalive_interval)

    try:
        # Create async generator for audio chunks
        async def audio_generator():
            nonlocal is_connected
            while is_connected:
                try:
                    message = await websocket.receive()

                    if message.get("type") == "websocket.disconnect":
                        is_connected = False
                        break

                    if "bytes" in message:
                        # Audio data
                        yield message["bytes"]

                    elif "text" in message:
                        # JSON command
                        try:
                            cmd = json.loads(message["text"])
                            command = cmd.get("command", "")

                            if command == "reset":
                                orchestrator.reset()
                                await send_status("reset")
                                context.clear_conversation()

                            elif command == "cancel":
                                orchestrator.reset()
                                await send_status("cancelled")

                            elif command == "config":
                                # Update configuration
                                if "auto_execute" in cmd:
                                    orchestrator.config.auto_execute = cmd["auto_execute"]
                                await send_status("config_updated", config={
                                    "auto_execute": orchestrator.config.auto_execute,
                                })

                            elif command == "text":
                                # Direct text input (bypass audio)
                                text = cmd.get("text", "")
                                if text:
                                    result = await orchestrator.process_text(text)
                                    await on_response(result)

                            elif command == "context":
                                # Return current context
                                await send_status("context", **context.build_context_dict())

                            elif command == "stop_recording":
                                # Force process collected audio
                                audio_bytes = orchestrator._audio_buffer.get_utterance()
                                if audio_bytes:
                                    orchestrator._state_machine.context.audio_bytes = audio_bytes
                                    await orchestrator._process_utterance()
                                    # on_response callback is called by _process_utterance
                                else:
                                    await send_status("idle", message="No speech detected")

                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON command: %s", message["text"])

                except WebSocketDisconnect:
                    is_connected = False
                    break
                except RuntimeError:
                    # WebSocket already closed
                    is_connected = False
                    break

        # Process audio stream continuously
        while is_connected:
            try:
                result = await orchestrator.process_audio_stream(audio_generator())

                # Send final result if not already sent via callback
                if result and not result.success and result.error:
                    await send_status("error", message=result.error)

                # Calculate cooldown based on TTS audio duration
                cooldown_seconds = 0.5  # Minimum cooldown
                if result and result.response_audio:
                    # WAV format: 44 byte header + PCM data
                    # Kokoro outputs 24kHz, 16-bit mono = 48000 bytes/sec
                    # Subtract header and calculate duration
                    pcm_bytes = max(0, len(result.response_audio) - 44)
                    audio_duration = pcm_bytes / 48000.0
                    cooldown_seconds = audio_duration + 0.8  # Buffer for room reverb

                # Wait for TTS to finish, discarding incoming audio
                if cooldown_seconds > 0:
                    logger.debug("Cooldown for %.1f seconds", cooldown_seconds)
                    await send_status("responding")
                    cooldown_end = asyncio.get_event_loop().time() + cooldown_seconds
                    # Drain audio during cooldown to prevent buffer buildup
                    async for _ in audio_generator():
                        if asyncio.get_event_loop().time() >= cooldown_end:
                            break
                        if not is_connected:
                            break

                # Reset for next utterance
                orchestrator.reset()
                await send_status(
                    "listening",
                    follow_up_mode=orchestrator.in_follow_up_mode,
                )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning("Utterance processing error: %s", e)
                orchestrator.reset()
                await send_status("error", message=str(e))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        is_connected = False
    except Exception as e:
        logger.exception("Orchestration error")
        await send_status("error", message=str(e))
    finally:
        is_connected = False
        # Unregister from connection manager
        await connection_manager.disconnect(websocket)
        # Cancel keepalive task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
        orchestrator.reset()


@router.get("/orchestration/status")
async def get_orchestration_status():
    """Get current orchestration status and context."""
    from ..orchestration.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    context = get_context()

    return {
        "state": orchestrator.state.name.lower(),
        "config": {
            "auto_execute": orchestrator.config.auto_execute,
        },
        "context": context.build_context_dict(),
    }


@router.post("/orchestration/text")
async def process_text_command(body: dict):
    """
    Process a text command through the orchestrator.

    Bypasses audio processing - useful for testing or text interfaces.

    Request body: {"text": "turn on the lights", "session_id": "optional-uuid"}
    """
    from ..orchestration.orchestrator import Orchestrator
    from ..storage import db_settings
    from ..storage.database import get_db_pool
    from ..storage.repositories.session import get_session_repo

    text = body.get("text", "")
    if not text:
        return {"error": "No text provided"}

    session_id = body.get("session_id")

    # Create or get session for persistence
    if not session_id and db_settings.enabled:
        pool = get_db_pool()
        if pool.is_initialized:
            try:
                session_repo = get_session_repo()
                session = await session_repo.get_or_create_session(
                    terminal_id="rest-api",
                )
                session_id = str(session.id)
                logger.info("REST text command using session: %s", session_id)
            except Exception as e:
                logger.warning("Failed to create session: %s", e)

    # Create agent for reasoning and orchestrator for text processing
    agent = create_atlas_agent(session_id=session_id)
    orchestrator = Orchestrator(session_id=session_id, agent=agent)

    # Load conversation history for context
    if session_id:
        try:
            from ..orchestration.context import get_context
            context = get_context()
            history = await orchestrator.load_session_history(limit=10)
            for turn in history:
                context.add_conversation_turn(
                    turn.role,
                    turn.content,
                    speaker_id=turn.speaker_id,
                )
            if history:
                logger.info("Loaded %d turns from session history", len(history))
        except Exception as e:
            logger.warning("Failed to load session history: %s", e)

    result = await orchestrator.process_text(text)

    return {
        "success": result.success,
        "transcript": result.transcript,
        "intent": {
            "action": result.intent.action if result.intent else None,
            "target_type": result.intent.target_type if result.intent else None,
            "target_name": result.intent.target_name if result.intent else None,
            "confidence": result.intent.confidence if result.intent else 0,
        } if result.intent else None,
        "actions": result.action_results,
        "response": result.response_text,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "session_id": session_id,
    }


@router.post("/orchestration/streaming/text")
async def streaming_text_command(body: dict):
    """
    Process text through the streaming orchestrator with multi-model routing.

    Returns routing decision and response. Useful for testing.

    Request: {"text": "what time is it?"}
    """
    from ..orchestration.streaming_orchestrator import get_streaming_orchestrator

    text = body.get("text", "")
    if not text:
        return {"error": "No text provided"}

    orchestrator = get_streaming_orchestrator()

    # Collect all events
    events = []
    response_chunks = []
    routing_info = None

    async for event in orchestrator.process_text_streaming(text):
        events.append(event)

        if event.get("type") == "routing":
            routing_info = {
                "tier": event.get("tier"),
                "confidence": event.get("confidence"),
                "reason": event.get("reason"),
            }
        elif event.get("type") == "response_chunk":
            response_chunks.append(event.get("text", ""))
        elif event.get("type") == "complete":
            pass

    return {
        "input": text,
        "routing": routing_info,
        "response": " ".join(response_chunks),
        "events": [e for e in events if e.get("type") != "audio_chunk"],  # Exclude binary
    }


@router.get("/orchestration/routing/test")
async def test_routing():
    """
    Test the intent router with various query types.

    Returns routing decisions for predefined test queries.
    """
    from ..services.intent_router import get_intent_router
    from ..services.model_pool import get_model_pool

    router = get_intent_router()
    pool = get_model_pool()
    available_tiers = pool.get_available_tiers()

    test_queries = [
        # ACTION tier (device commands)
        "turn on the living room lights",
        "turn off the TV",
        "open the garage door",

        # FAST tier (simple)
        "what time is it?",
        "hello",
        "thanks",

        # BALANCED tier (reasoning)
        "why is the sky blue?",
        "explain how photosynthesis works",
        "what should I have for dinner?",

        # CLOUD tier (complex/coding)
        "write a python function to sort a list",
        "help me debug this javascript code",
    ]

    results = []
    for query in test_queries:
        decision = router.route_with_fallback(query, available_tiers)
        results.append({
            "query": query,
            "tier": decision.tier.name,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "is_device_command": decision.is_device_command,
        })

    return {
        "available_tiers": [t.name for t in available_tiers],
        "results": results,
    }


@router.websocket("/ws/streaming")
async def streaming_voice_pipeline(websocket: WebSocket):
    """
    Low-latency streaming voice pipeline.

    Optimized for human-like response times:
    - Streams LLM responses token by token
    - Sentence-level TTS chunking
    - First audio chunk plays while still generating

    Message Protocol:
    - Client sends: Binary audio frames OR JSON {"command": "text", "text": "..."}
    - Server sends: JSON events as they happen

    Events:
    - {"type": "transcript", "text": "..."}
    - {"type": "intent", "action": "...", ...}
    - {"type": "token", "text": "..."}  (streaming LLM tokens)
    - {"type": "response_chunk", "text": "..."}  (complete sentence)
    - {"type": "audio_chunk", "data": "<base64>"}
    - {"type": "complete", "total_ms": ..., "first_audio_ms": ...}
    """
    await websocket.accept()
    logger.info("Streaming WebSocket connected")

    from ..orchestration.streaming_orchestrator import get_streaming_orchestrator

    orchestrator = get_streaming_orchestrator()
    is_connected = True

    async def send_event(event: dict):
        """Send event to client."""
        if not is_connected:
            return
        try:
            # Encode audio data as base64
            if event.get("type") == "audio_chunk" and "data" in event:
                event = event.copy()
                event["data"] = base64.b64encode(event["data"]).decode()
            await websocket.send_json(event)
        except Exception:
            pass

    try:
        while is_connected:
            try:
                message = await websocket.receive()

                if "bytes" in message:
                    # Audio data - process through full pipeline
                    audio_bytes = message["bytes"]

                    # Stream events back to client
                    async for event in orchestrator.process_utterance_streaming(audio_bytes):
                        await send_event(event)

                elif "text" in message:
                    # JSON command
                    try:
                        cmd = json.loads(message["text"])
                        command = cmd.get("command", "")

                        if command == "text":
                            # Direct text input
                            text = cmd.get("text", "")
                            if text:
                                async for event in orchestrator.process_text_streaming(text):
                                    await send_event(event)

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON: %s", message["text"][:100])

            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info("Streaming WebSocket disconnected")
    except Exception as e:
        logger.exception("Streaming error")
        await send_event({"type": "error", "message": str(e)})
    finally:
        is_connected = False
