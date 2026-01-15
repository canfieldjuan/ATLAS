"""
Text query endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException

from ...schemas.query import TextQueryRequest
from ...services import llm_registry
from ...services.protocols import Message

router = APIRouter()


@router.post("/text")
async def query_text(request: TextQueryRequest):
    """
    Process a text query using the active LLM.

    The LLM will generate a response based on the input text.
    """
    llm = llm_registry.get_active()
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM model is currently loaded. Use /models/llm/activate to load one.",
        )

    messages = [
        Message(
            role="system",
            content="You are Atlas, a capable personal assistant. You can control smart home devices, answer questions, and help with various tasks. Be conversational and concise.",
        ),
        Message(role="user", content=request.query_text),
    ]

    result = llm.chat(messages=messages, max_tokens=256, temperature=0.7)

    return {
        "response": result.get("response", ""),
        "query": request.query_text,
    }
