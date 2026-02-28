"""
Shared LLM utilities for pipeline tasks.

Extracts duplicated LLM resolution, output cleaning, and JSON parsing
from article_enrichment, daily_intelligence, complaint_analysis,
complaint_enrichment, and complaint_content_generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("atlas.pipelines.llm")


# ------------------------------------------------------------------
# LLM resolution
# ------------------------------------------------------------------


def get_pipeline_llm(
    *,
    prefer_cloud: bool = True,
    try_openrouter: bool = True,
    auto_activate_ollama: bool = True,
):
    """Resolve an LLM instance using a configurable fallback chain.

    1. Triage LLM (Anthropic) if ``prefer_cloud``
    2. OpenRouter (DeepSeek) if ``try_openrouter`` and OPENROUTER_API_KEY set
    3. Active LLM from registry
    4. Auto-activate Ollama if ``auto_activate_ollama``

    Returns the LLM instance or None.
    """
    from ..services import llm_registry

    # 1. Triage (Anthropic)
    if prefer_cloud:
        from ..services.llm_router import get_triage_llm

        llm = get_triage_llm()
        if llm is not None:
            logger.debug("Using triage LLM (Anthropic)")
            return llm

    # 2. OpenRouter
    if try_openrouter:
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if or_key:
            try:
                llm_registry.activate(
                    "openrouter",
                    model="deepseek/deepseek-chat-v3-0324",
                    api_key=or_key,
                )
                llm = llm_registry.get_active()
                if llm is not None:
                    logger.info("Using OpenRouter LLM")
                    return llm
            except Exception as e:
                logger.debug("OpenRouter fallback failed: %s", e)

    # 3. Active LLM
    llm = llm_registry.get_active()
    if llm is not None:
        return llm

    # 4. Auto-activate Ollama
    if auto_activate_ollama:
        try:
            from ..config import settings

            llm_registry.activate(
                "ollama",
                model=settings.llm.ollama_model,
                base_url=settings.llm.ollama_url,
                timeout=settings.llm.ollama_timeout,
            )
            llm = llm_registry.get_active()
            if llm is not None:
                logger.info("Auto-activated Ollama LLM")
                return llm
        except Exception as e:
            logger.warning("Could not auto-activate Ollama LLM: %s", e)

    return None


# ------------------------------------------------------------------
# Output cleaning
# ------------------------------------------------------------------


def clean_llm_output(text: str) -> str:
    """Strip ``<think>`` tags and markdown fences from LLM output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    json_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    return text


# ------------------------------------------------------------------
# JSON parsing
# ------------------------------------------------------------------


def parse_json_response(
    text: str,
    *,
    recover_truncated: bool = False,
) -> dict[str, Any]:
    """Progressive JSON extraction from LLM response.

    Tries in order:
    1. ```json``` fenced block
    2. Entire response as JSON
    3. First ``{...}`` match
    4. Truncation recovery (if ``recover_truncated``)
    5. Fallback ``{"analysis_text": text}``
    """
    if not text:
        logger.warning("parse_json_response received empty/None input")
        return {"analysis_text": "", "_parse_fallback": True}

    # 1. Fenced JSON block
    json_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Entire response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. First {..} object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # 4. Truncation recovery
    if recover_truncated:
        recovered = _recover_truncated_json(text)
        if recovered:
            return recovered

    # 5. Fallback -- all structured fields lost, only raw text preserved
    logger.warning(
        "parse_json_response fell back to raw text (%d chars). "
        "LLM output was not valid JSON.",
        len(text),
    )
    return {"analysis_text": text, "_parse_fallback": True}


def _recover_truncated_json(raw_text: str) -> dict[str, Any] | None:
    """Attempt to recover a JSON object from truncated LLM output.

    When max_tokens cuts off output mid-JSON, we try closing open
    structures progressively to salvage whatever fields were complete.
    """
    start = raw_text.find("{")
    if start < 0:
        return None

    text = raw_text[start:]

    for trim in range(0, min(len(text), 500), 1):
        candidate = text if trim == 0 else text[:-trim]
        opens = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if opens <= 0 and open_brackets <= 0:
            continue
        suffix = "]" * max(open_brackets, 0) + "}" * max(opens, 0)
        try:
            result = json.loads(candidate + suffix)
            if isinstance(result, dict) and result.get("analysis_text"):
                logger.info(
                    "Recovered truncated JSON (trimmed %d chars, closed %d braces)",
                    trim, opens + open_brackets,
                )
                return result
        except json.JSONDecodeError:
            continue

    return None


# ------------------------------------------------------------------
# Full LLM call pattern
# ------------------------------------------------------------------


def call_llm_with_skill(
    skill_name: str,
    payload: dict[str, Any],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.4,
    prefer_cloud: bool = True,
    try_openrouter: bool = True,
    auto_activate_ollama: bool = True,
) -> str | None:
    """Load a skill, resolve an LLM, call it, clean the output.

    Returns the raw cleaned text, or None on failure.
    Caller is responsible for further parsing (JSON, etc.).
    """
    from ..skills import get_skill_registry
    from ..services.protocols import Message

    skill = get_skill_registry().get(skill_name)
    if skill is None:
        logger.warning("Skill '%s' not found", skill_name)
        return None

    llm = get_pipeline_llm(
        prefer_cloud=prefer_cloud,
        try_openrouter=try_openrouter,
        auto_activate_ollama=auto_activate_ollama,
    )
    if llm is None:
        logger.warning("No LLM available for skill '%s'", skill_name)
        return None

    messages = [
        Message(role="system", content=skill.content),
        Message(
            role="user",
            content=json.dumps(payload, indent=2, default=str),
        ),
    ]

    try:
        result = llm.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = result.get("response", "").strip()
        if not text:
            logger.warning("LLM returned empty response for skill '%s'", skill_name)
            return None

        # Clean think tags (Qwen3 models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    except Exception:
        logger.exception("LLM call failed for skill '%s'", skill_name)
        return None
