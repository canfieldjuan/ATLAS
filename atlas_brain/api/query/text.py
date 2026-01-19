"""
Text query endpoint with LLM tool calling support.
"""

import logging
from fastapi import APIRouter, HTTPException

from ...schemas.query import TextQueryRequest
from ...services import llm_registry
from ...services.protocols import Message
from ...services.tool_executor import execute_with_tools

router = APIRouter()
logger = logging.getLogger("atlas.api.query.text")


@router.post("/text")
async def query_text(request: TextQueryRequest):
    """
    Process a text query using the active LLM with tool calling.

    The LLM decides which tools to use based on the query.
    """
    llm = llm_registry.get_active()
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM model is currently loaded. Use /models/llm/activate to load one.",
        )

    system_msg = (
        "You are Atlas, a capable personal assistant. "
        "You can control smart home devices, answer questions, "
        "send emails, check weather, and help with various tasks. "
        "Be conversational and concise. "
        "Use the available tools when appropriate to help the user."
    )

    messages = [
        Message(role="system", content=system_msg),
        Message(role="user", content=request.query_text),
    ]

    logger.info("Calling execute_with_tools with LLM: %s, has chat_with_tools: %s",
                type(llm).__name__, hasattr(llm, "chat_with_tools"))

    result = await execute_with_tools(
        llm=llm,
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )

    logger.info("execute_with_tools result: tools=%s, response_len=%d",
                result.get("tools_executed", []), len(result.get("response", "")))

    tools_executed = result.get("tools_executed", [])
    if tools_executed:
        logger.info("Tools executed for query: %s", tools_executed)

    return {
        "response": result.get("response", ""),
        "query": request.query_text,
        "tools_executed": tools_executed,
    }
