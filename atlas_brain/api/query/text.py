"""
Text query endpoint using Atlas Agent.

Routes through the unified Agent for full capabilities:
tools, device commands, and conversation memory.
"""

import logging

from fastapi import APIRouter

from ...agents import get_atlas_agent, AgentContext
from ...schemas.query import TextQueryRequest

router = APIRouter()
logger = logging.getLogger("atlas.api.query.text")


@router.post("/text")
async def query_text(request: TextQueryRequest):
    """
    Process a text query using the Atlas Agent.

    The Agent handles intent parsing, tool execution,
    device commands, and LLM response generation.

    Args:
        request: TextQueryRequest with query_text and optional session_id

    Returns:
        Response with text, query echo, tools executed, and action type
    """
    logger.info("Text query: %s (session=%s)", request.query_text[:50], request.session_id)

    agent = get_atlas_agent(session_id=request.session_id)

    context = AgentContext(
        input_text=request.query_text,
        input_type="text",
        session_id=request.session_id,
    )

    result = await agent.run(context)

    # Build tools_executed list from action_results
    tools_executed = []
    for action_result in result.action_results:
        if action_result.get("tool"):
            tools_executed.append(action_result.get("tool"))
        elif action_result.get("action"):
            tools_executed.append(action_result.get("action"))

    logger.info(
        "Agent result: action_type=%s, tools=%s, response_len=%d",
        result.action_type,
        tools_executed,
        len(result.response_text or ""),
    )

    return {
        "response": result.response_text or "",
        "query": request.query_text,
        "tools_executed": tools_executed,
        "action_type": result.action_type,
    }
