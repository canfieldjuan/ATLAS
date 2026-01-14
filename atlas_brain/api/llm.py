"""
LLM (Large Language Model) API endpoints.

Provides REST API for:
- LLM activation and management
- Text generation and chat
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services import llm_registry
from ..storage import db_settings
from ..storage.database import get_db_pool
from ..storage.repositories.session import get_session_repo
from ..storage.repositories.conversation import get_conversation_repo

logger = logging.getLogger("atlas.api.llm")

router = APIRouter(prefix="/llm", tags=["llm"])


class ActivateRequest(BaseModel):
    name: str = "llama-cpp"
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = all layers on GPU


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
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
        raise HTTPException(status_code=500, detail=f"Failed to activate: {e}")


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
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with the LLM.

    Optionally provide session_id to load conversation history
    and persist new turns for multi-location continuity.
    """
    service = llm_registry.get_active()
    if service is None:
        raise HTTPException(
            status_code=400,
            detail="No LLM active. Call /llm/activate first.",
        )

    try:
        from ..services.protocols import Message

        messages = [Message(role=m.role, content=m.content) for m in request.messages]
        session_uuid = None

        # Load conversation history if session_id provided and DB enabled
        if request.session_id and db_settings.enabled:
            pool = get_db_pool()
            if pool.is_initialized:
                try:
                    session_uuid = UUID(request.session_id)
                    conv_repo = get_conversation_repo()
                    history = await conv_repo.get_history(session_uuid, limit=20)

                    if history:
                        history_messages = [
                            Message(role=t.role, content=t.content)
                            for t in history
                        ]
                        messages = history_messages + messages
                        logger.debug(
                            "Loaded %d turns from session %s",
                            len(history),
                            request.session_id
                        )
                except Exception as e:
                    logger.warning("Failed to load history: %s", e)

        result = service.chat(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Persist new turns if session_id provided and DB enabled
        if session_uuid and db_settings.enabled:
            pool = get_db_pool()
            if pool.is_initialized:
                try:
                    conv_repo = get_conversation_repo()
                    user_content = request.messages[-1].content if request.messages else ""

                    await conv_repo.add_turn(
                        session_id=session_uuid,
                        role="user",
                        content=user_content,
                    )

                    response_content = result.get("response", "")
                    await conv_repo.add_turn(
                        session_id=session_uuid,
                        role="assistant",
                        content=response_content,
                    )
                    logger.debug("Persisted turns to session %s", request.session_id)
                except Exception as e:
                    logger.warning("Failed to persist turns: %s", e)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


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


