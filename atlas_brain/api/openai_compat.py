"""OpenAI-compatible chat completions endpoint.

Allows Home Assistant's Extended OpenAI Conversation integration
to use Atlas as its LLM/conversation agent.
"""

import hashlib
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..agents.interface import get_agent
from ..config import settings

logger = logging.getLogger("atlas.api.openai_compat")

router = APIRouter(tags=["openai-compat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "atlas"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
):
    # Auth check
    api_key = settings.openai_compat.api_key
    if api_key:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

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
    agent = get_agent("atlas")
    result = await agent.process(
        input_text=user_message,
        session_id=session_id,
        input_type="text",
    )

    response_text = result.response_text or "I couldn't process that request."

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
            )
        ],
        usage=UsageInfo(),
    )
