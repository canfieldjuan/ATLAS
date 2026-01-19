"""
Audio query endpoints for speech-to-text.
"""

from fastapi import APIRouter, Depends, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from ...services import stt_registry, omni_registry
from ...services.protocols import STTService
from ..dependencies import get_stt

router = APIRouter()


@router.post("/audio")
async def query_audio(
    audio_file: UploadFile = File(...),
    stt: STTService = Depends(get_stt),
):
    """
    Transcribe an uploaded audio file using the active STT model.
    """
    contents = await audio_file.read()
    filename = audio_file.filename or "audio.wav"
    return await stt.transcribe(audio_bytes=contents, filename=filename)


@router.websocket("/ws/audio")
async def stream_audio(websocket: WebSocket):
    """
    Accept audio chunks over WebSocket for near real-time transcription.

    Protocol:
    - Send binary frames with raw/WAV audio data
    - Send text 'END' to transcribe buffered audio
    - Send text 'RESET' to clear buffer
    """
    await websocket.accept()
    buffer = bytearray()

    await websocket.send_json({
        "status": "ready",
        "message": "Send binary audio frames; send 'END' to transcribe or 'RESET' to clear.",
    })

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                buffer.extend(message["bytes"])
                await websocket.send_json({
                    "status": "receiving",
                    "received_bytes": len(buffer),
                })

            elif "text" in message and message["text"] is not None:
                command = message["text"].strip().lower()

                if command == "end":
                    if not buffer:
                        await websocket.send_json({
                            "status": "error",
                            "message": "No audio received yet.",
                        })
                        continue

                    # Get STT service (may not be loaded)
                    stt = stt_registry.get_active()
                    if stt is None:
                        await websocket.send_json({
                            "status": "error",
                            "message": "No STT model loaded.",
                        })
                        continue

                    result = await stt.transcribe(bytes(buffer), filename="stream.wav")
                    await websocket.send_json({
                        "status": "completed",
                        **result,
                    })
                    buffer.clear()

                elif command == "reset":
                    buffer.clear()
                    await websocket.send_json({
                        "status": "reset",
                        "message": "Audio buffer cleared.",
                    })

                else:
                    await websocket.send_json({
                        "status": "info",
                        "message": "Unrecognized command. Send binary audio or 'END'/'RESET'.",
                    })

    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011, reason="Internal error")
        raise


@router.post("/audio/omni")
async def query_audio_omni(
    audio_file: UploadFile = File(...),
):
    """
    Process audio through the Omni (unified speech-to-speech) model.

    Returns both text response and audio response.

    Requires omni model to be active: POST /api/v1/models/omni/activate
    """
    omni = omni_registry.get_active()
    if omni is None:
        return {
            "error": "Omni model not loaded. Activate with POST /api/v1/models/omni/activate",
            "available_models": omni_registry.list_available(),
        }

    contents = await audio_file.read()

    # Use speech-to-speech for full audio in/out
    response = await omni.speech_to_speech(contents)

    return {
        "text": response.text,
        "audio_duration_sec": response.audio_duration_sec,
        "has_audio": response.audio_bytes is not None,
        "audio_size_bytes": len(response.audio_bytes) if response.audio_bytes else 0,
        "metrics": response.metrics,
    }


@router.post("/audio/omni/full")
async def query_audio_omni_full(
    audio_file: UploadFile = File(...),
):
    """
    Process audio through Omni and return audio response directly.

    Returns WAV audio file.
    """
    omni = omni_registry.get_active()
    if omni is None:
        return {"error": "Omni model not loaded"}

    contents = await audio_file.read()
    response = await omni.speech_to_speech(contents)

    if response.audio_bytes:
        return Response(
            content=response.audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Text-Response": response.text[:200],
                "X-Audio-Duration": str(response.audio_duration_sec),
            }
        )
    else:
        return {"error": "No audio generated", "text": response.text}
