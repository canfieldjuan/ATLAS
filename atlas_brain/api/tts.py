"""
Text-to-Speech API endpoints.

Provides REST API for:
- Text to speech synthesis
- Voice management
- Audio streaming
"""

import io
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from ..services import tts_registry

router = APIRouter(prefix="/tts", tags=["tts"])


class ActivateRequest(BaseModel):
    name: str = "piper"
    voice: Optional[str] = None


class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0


class VoiceInfo(BaseModel):
    name: str
    language: str
    quality: str


# Common Piper voices
PIPER_VOICES = {
    "en_US-amy-medium": {"language": "en-US", "quality": "medium"},
    "en_US-lessac-medium": {"language": "en-US", "quality": "medium"},
    "en_US-libritts-high": {"language": "en-US", "quality": "high"},
    "en_GB-alan-medium": {"language": "en-GB", "quality": "medium"},
    "en_GB-alba-medium": {"language": "en-GB", "quality": "medium"},
    "de_DE-thorsten-medium": {"language": "de-DE", "quality": "medium"},
    "fr_FR-upmc-medium": {"language": "fr-FR", "quality": "medium"},
    "es_ES-carlfm-medium": {"language": "es-ES", "quality": "medium"},
}


@router.get("/available")
async def list_available():
    """List available TTS implementations."""
    return {
        "available": tts_registry.list_available(),
        "active": tts_registry.get_active_name(),
    }


@router.post("/activate")
async def activate_tts(request: ActivateRequest):
    """Activate a TTS implementation."""
    try:
        kwargs = {}
        if request.voice:
            kwargs["voice"] = request.voice

        service = tts_registry.activate(request.name, **kwargs)
        return {
            "success": True,
            "message": f"Activated TTS: {request.name}",
            "model_info": service.model_info.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate: {e}")


@router.post("/deactivate")
async def deactivate_tts():
    """Deactivate TTS to free resources."""
    tts_registry.deactivate()
    return {"success": True, "message": "TTS deactivated"}


@router.post("/synthesize")
async def synthesize_speech(request: SynthesizeRequest):
    """
    Synthesize speech from text.

    Returns WAV audio as binary response.
    """
    service = tts_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No TTS service active. Call /tts/activate first.",
        )

    try:
        audio_bytes = await service.synthesize(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Content-Length": str(len(audio_bytes)),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")


@router.get("/synthesize")
async def synthesize_speech_get(
    text: str = Query(..., description="Text to synthesize"),
    voice: Optional[str] = Query(None, description="Voice model to use"),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="Speech speed"),
):
    """
    Synthesize speech from text (GET for easy testing).

    Returns WAV audio as binary response.
    """
    service = tts_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No TTS service active. Call /tts/activate first.",
        )

    try:
        audio_bytes = await service.synthesize(
            text=text,
            voice=voice,
            speed=speed,
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Content-Length": str(len(audio_bytes)),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")


@router.get("/voices")
async def list_voices():
    """List available voices for the active TTS."""
    service = tts_registry.get_active()
    active_name = tts_registry.get_active_name()

    voices = []

    if active_name == "piper":
        for name, info in PIPER_VOICES.items():
            voices.append(VoiceInfo(
                name=name,
                language=info["language"],
                quality=info["quality"],
            ))
    elif active_name == "espeak":
        # Common eSpeak voices
        voices = [
            VoiceInfo(name="en", language="en", quality="low"),
            VoiceInfo(name="en-us", language="en-US", quality="low"),
            VoiceInfo(name="en-gb", language="en-GB", quality="low"),
            VoiceInfo(name="de", language="de", quality="low"),
            VoiceInfo(name="fr", language="fr", quality="low"),
            VoiceInfo(name="es", language="es", quality="low"),
        ]

    return {
        "active_tts": active_name,
        "voices": [v.model_dump() for v in voices],
        "count": len(voices),
    }


@router.get("/status")
async def get_status():
    """Get TTS service status."""
    service = tts_registry.get_active()

    if service is None:
        return {
            "active": False,
            "message": "No TTS service active",
        }

    return {
        "active": True,
        "model_info": service.model_info.to_dict(),
    }
