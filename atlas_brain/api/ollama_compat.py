"""Ollama-compatible API endpoints.

Allows Home Assistant's built-in Ollama integration to use Atlas
as its conversation agent by mimicking the Ollama API surface.

HA's ollama Python client calls:
  GET  /                — root health check ("Ollama is running")
  GET  /api/version     — version info
  GET  /api/tags        — list available models (setup/discovery)
  POST /api/chat        — chat completions (conversation)
"""

import hashlib
import logging
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..agents.interface import get_agent
from ..config import settings

logger = logging.getLogger("atlas.api.ollama_compat")

router = APIRouter(tags=["ollama-compat"])


# --- Models for /api/tags ---

class OllamaModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "gguf"
    family: str = "atlas"
    families: list[str] = ["atlas"]
    parameter_size: str = "30B"
    quantization_level: str = "Q4_K_S"


class OllamaModel(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails


class OllamaTagsResponse(BaseModel):
    models: list[OllamaModel]


# --- Models for /api/chat ---

class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str = "atlas"
    messages: list[OllamaChatMessage]
    stream: bool = False
    options: dict | None = None
    keep_alive: str | None = None


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaChatMessage
    done: bool = True
    done_reason: str = "stop"
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


@router.get("/", response_class=PlainTextResponse)
async def root():
    """Root health check — ollama client checks this on connect."""
    return "Ollama is running"


@router.get("/api/version")
async def version():
    """Version endpoint — ollama client may check this."""
    return {"version": "0.13.2"}


@router.get("/api/tags")
async def list_models():
    """List available models — HA calls this during integration setup."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())
    return OllamaTagsResponse(
        models=[
            OllamaModel(
                name="atlas:latest",
                model="atlas:latest",
                modified_at=now,
                size=0,
                digest="sha256:atlas",
                details=OllamaModelDetails(),
            )
        ]
    )


@router.post("/api/chat")
async def chat(
    request: OllamaChatRequest,
    x_session_id: Optional[str] = Header(None),
):
    """Ollama-compatible chat endpoint — routes through full Atlas agent."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")

    # Extract last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # Derive session_id: prefer header, else hash system prompt for stability
    session_id = x_session_id
    if not session_id:
        system_msgs = [m.content for m in request.messages if m.role == "system"]
        seed = system_msgs[0] if system_msgs else "ha-default"
        session_id = hashlib.sha256(seed.encode()).hexdigest()[:16]

    # Process through Atlas agent
    t0 = time.monotonic_ns()
    agent = get_agent("atlas")
    result = await agent.process(
        input_text=user_message,
        session_id=session_id,
        input_type="text",
    )
    duration_ns = time.monotonic_ns() - t0

    response_text = result.response_text or "I couldn't process that request."

    return OllamaChatResponse(
        model=request.model,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),
        message=OllamaChatMessage(role="assistant", content=response_text),
        total_duration=duration_ns,
    )
