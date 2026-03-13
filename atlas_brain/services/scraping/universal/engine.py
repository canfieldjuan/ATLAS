"""
LLM-powered data extraction engine.

Takes cleaned page text + an ExtractionSchema, sends to the LLM,
and parses the response into a JSON array of extracted items.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ....pipelines.llm import clean_llm_output, get_pipeline_llm
from ....services.protocols import Message
from .schemas import ExtractionSchema

logger = logging.getLogger("atlas.services.scraping.universal.engine")

_SYSTEM_PROMPT = """\
You are a precise data extraction assistant. You extract structured data from web page text.

Rules:
- Output ONLY a JSON array of objects. No explanation, no markdown fences.
- Each object represents one item found on the page.
- If no items match, output an empty array: []
- Use null for missing fields.
- Preserve exact text values (don't summarize or paraphrase).
- Numbers should be actual numbers, not strings (e.g., 4.5 not "4.5").
- Dates should be ISO 8601 when possible."""


async def extract_from_text(
    page_text: str,
    schema: ExtractionSchema,
    *,
    workload: str = "triage",
    max_tokens: int = 4096,
) -> tuple[list[dict[str, Any]], str]:
    """Extract structured data from page text using the LLM.

    Returns ``(items, raw_llm_output)``.

    Raises ``RuntimeError`` if no LLM is available.
    """
    llm = get_pipeline_llm(workload=workload)
    if llm is None:
        raise RuntimeError("No LLM available for universal scraper extraction")

    schema_fragment = schema.to_prompt_fragment()
    user_prompt = f"{schema_fragment}\n\nPage content:\n---\n{page_text}\n---"

    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]

    if hasattr(llm, "chat_async"):
        raw = await llm.chat_async(messages, max_tokens=max_tokens, temperature=0.1)
    else:
        import asyncio

        result = await asyncio.to_thread(
            llm.chat, messages, max_tokens=max_tokens, temperature=0.1
        )
        raw = result.get("response", "").strip()

    cleaned = clean_llm_output(raw)
    items = _parse_items(cleaned)
    return items, raw


def _parse_items(text: str) -> list[dict[str, Any]]:
    """Parse LLM output as a JSON array of dicts."""
    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Find JSON array in text
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if arr_match:
        try:
            result = json.loads(arr_match.group())
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict)]
        except json.JSONDecodeError:
            pass

    # Find single JSON object
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        try:
            obj = json.loads(obj_match.group())
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM output as JSON, returning empty list")
    return []
