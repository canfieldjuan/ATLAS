"""
LLM (Large Language Model) API endpoints.

Provides REST API for:
- LLM activation and management
- Text generation and chat
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..agents.interface import get_agent
from ..services import llm_registry

logger = logging.getLogger("atlas.api.llm")

router = APIRouter(prefix="/llm", tags=["llm"])


class ActivateRequest(BaseModel):
    name: str = "llama-cpp"
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = all layers on GPU


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=100_000)
    system_prompt: Optional[str] = Field(None, max_length=50_000)
    max_tokens: int = Field(512, ge=1, le=16384)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class ChatMessage(BaseModel):
    role: str = Field(..., max_length=20)
    content: str = Field(..., max_length=100_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., max_length=200)
    max_tokens: int = Field(512, ge=1, le=16384)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    terminal_id: Optional[str] = None


@router.get("/available")
async def list_available():
    """List available LLM implementations."""
    return {
        "available": llm_registry.list_available(),
        "active": llm_registry.get_active_name(),
    }


@router.post("/activate")
async def activate_llm(request: ActivateRequest):
    """
    Activate an LLM implementation.

    For llama-cpp, provide model_path to the GGUF file.
    """
    try:
        kwargs = {
            "n_ctx": request.n_ctx,
            "n_gpu_layers": request.n_gpu_layers,
        }

        if request.model_path:
            kwargs["model_path"] = Path(request.model_path)
        if request.model_id:
            kwargs["model_id"] = request.model_id

        service = llm_registry.activate(request.name, **kwargs)
        return {
            "success": True,
            "message": f"Activated LLM: {request.name}",
            "model_info": service.model_info.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to activate LLM: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to activate LLM")


@router.post("/deactivate")
async def deactivate_llm():
    """Deactivate LLM to free VRAM."""
    llm_registry.deactivate()
    return {"success": True, "message": "LLM deactivated"}


@router.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text from a prompt."""
    service = llm_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No LLM active. Call /llm/activate first.",
        )

    try:
        result = service.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return result
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat using the Atlas Agent.

    Routes through the unified Agent for full capabilities:
    tools, device commands, and conversation memory.

    Optionally provide session_id for conversation persistence.
    """
    # Extract user message from request
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="No user message found in messages list.",
        )

    # Normalize session_id to a valid UUID (or generate one)
    from ..utils.session_id import normalize_session_id, ensure_session_row

    session_id = normalize_session_id(request.session_id)
    await ensure_session_row(session_id)

    logger.info("Chat request: %s (session=%s)", user_message[:50], session_id)

    agent = get_agent("atlas")

    try:
        result = await agent.process(
            input_text=user_message,
            session_id=session_id,
            input_type="text",
        )

        logger.info(
            "Agent result: action_type=%s, response_len=%d",
            result.action_type,
            len(result.response_text or ""),
        )

        # Return in same format as before for compatibility with voice_client.py
        return {
            "response": result.response_text or "",
        }
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail="Chat failed")


@router.get("/status")
async def get_status():
    """Get LLM status."""
    service = llm_registry.get_active()

    if service is None:
        return {
            "active": False,
            "message": "No LLM active",
        }

    return {
        "active": True,
        "model_info": service.model_info.to_dict(),
    }


