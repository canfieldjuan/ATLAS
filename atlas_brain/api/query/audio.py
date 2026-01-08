"""
Audio query endpoints for speech-to-text.
"""

from fastapi import APIRouter, Depends, File, UploadFile, WebSocket, WebSocketDisconnect

from ...services import stt_registry
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
