"""
Orchestrated audio pipeline API endpoint.

Provides a WebSocket endpoint that handles the full voice pipeline:
Wake Word → VAD → STT → Intent/LLM → Action → TTS
"""

import asyncio
import base64
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..orchestration import Orchestrator, PipelineState
from ..orchestration.context import get_context

logger = logging.getLogger("atlas.api.orchestration")

router = APIRouter(tags=["orchestration"])


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
    - {"state": "listening", "wake_word_enabled": true}
    - {"state": "wake_detected", "wake_word": "hey_jarvis"}
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

    # Create orchestrator for this session
    orchestrator = Orchestrator()
    context = get_context()

    # Track state for status updates
    last_state: Optional[PipelineState] = None
    is_connected = True

    async def send_status(state: str, **kwargs):
        """Send a status update to the client."""
        if not is_connected:
            return
        try:
            message = {"state": state, **kwargs}
            await websocket.send_json(message)
            logger.debug("Status: %s", state)
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
        response_data = {
            "text": result.response_text,
            "success": result.success,
            "latency_ms": result.latency_ms,
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
        wake_word_enabled=orchestrator.config.require_wake_word,
        context=context.build_context_dict(),
    )

    try:
        # Create async generator for audio chunks
        async def audio_generator():
            while True:
                try:
                    message = await websocket.receive()

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
                                if "wake_word_enabled" in cmd:
                                    orchestrator.config.require_wake_word = cmd["wake_word_enabled"]
                                if "auto_execute" in cmd:
                                    orchestrator.config.auto_execute = cmd["auto_execute"]
                                await send_status("config_updated", config={
                                    "wake_word_enabled": orchestrator.config.require_wake_word,
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
                    break

        # Process audio stream
        result = await orchestrator.process_audio_stream(audio_generator())

        # Send final result if not already sent via callback
        if result and not result.success and result.error:
            await send_status("error", message=result.error)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        is_connected = False
    except Exception as e:
        logger.exception("Orchestration error")
        await send_status("error", message=str(e))
    finally:
        is_connected = False
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
            "wake_word_enabled": orchestrator.config.wake_word_enabled,
            "require_wake_word": orchestrator.config.require_wake_word,
            "auto_execute": orchestrator.config.auto_execute,
        },
        "context": context.build_context_dict(),
    }


@router.post("/orchestration/text")
async def process_text_command(body: dict):
    """
    Process a text command through the orchestrator.

    Bypasses audio processing - useful for testing or text interfaces.

    Request body: {"text": "turn on the lights"}
    """
    from ..orchestration.orchestrator import get_orchestrator

    text = body.get("text", "")
    if not text:
        return {"error": "No text provided"}

    orchestrator = get_orchestrator()
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
    }
