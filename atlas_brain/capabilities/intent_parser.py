"""
Intent parser for natural language to action conversion.

Uses the LLM (preferred) or VLM to extract structured intents from user queries.
"""

import json
import logging
import re
from typing import Optional

from .actions import Intent

logger = logging.getLogger("atlas.capabilities.intent_parser")

INTENT_EXTRACTION_PROMPT = """You are an intent parser for a smart home system called Atlas. Extract the user's intent from their query.

Respond ONLY with a JSON object (no other text):
{{
  "action": "turn_on|turn_off|toggle|set_brightness|read|none",
  "target_type": "light|switch|sensor|camera|thermostat|null",
  "target_name": "location name or null",
  "parameters": {{}},
  "confidence": 0.0-1.0
}}

IMPORTANT: If the query is NOT a device command (questions, greetings, help requests), return:
{{"action":"none","target_type":null,"target_name":null,"parameters":{{}},"confidence":0.0}}

Examples:
- "turn on the living room lights" -> {{"action":"turn_on","target_type":"light","target_name":"living room","parameters":{{}},"confidence":0.95}}
- "set bedroom lights to 50%" -> {{"action":"set_brightness","target_type":"light","target_name":"bedroom","parameters":{{"brightness":127}},"confidence":0.9}}
- "dim kitchen to 30%" -> {{"action":"set_brightness","target_type":"light","target_name":"kitchen","parameters":{{"brightness":76}},"confidence":0.9}}
- "turn off all lights" -> {{"action":"turn_off","target_type":"light","target_name":null,"parameters":{{}},"confidence":0.95}}
- "what can you do?" -> {{"action":"none","target_type":null,"target_name":null,"parameters":{{}},"confidence":0.0}}
- "hello" -> {{"action":"none","target_type":null,"target_name":null,"parameters":{{}},"confidence":0.0}}

User query: {query}

JSON:"""


class IntentParser:
    """
    Parses natural language queries into structured intents.

    Prefers LLM (Xortron) for better understanding, falls back to VLM.
    """

    def __init__(self):
        self._llm = None
        self._vlm = None

    def _get_llm(self):
        """Lazy import LLM."""
        if self._llm is None:
            from ..services import llm_registry
            self._llm = llm_registry.get_active()
        return self._llm

    def _get_vlm(self):
        """Lazy import VLM as fallback."""
        if self._vlm is None:
            from ..services import vlm_registry
            self._vlm = vlm_registry.get_active()
        return self._vlm

    async def parse(self, query: str) -> Optional[Intent]:
        """
        Parse a natural language query into an Intent.

        Uses LLM (preferred) or VLM to extract structured intent information.

        Args:
            query: Natural language query from the user

        Returns:
            Intent object if parsing succeeds, None otherwise
        """
        prompt = INTENT_EXTRACTION_PROMPT.format(query=query)

        # Try LLM first (better reasoning)
        llm = self._get_llm()
        if llm is not None:
            try:
                result = llm.generate(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.1,  # Low temp for structured output
                )
                response_text = result.get("response", "")
                intent = self._parse_response(response_text, query)
                if intent:
                    logger.info("LLM parsed intent: action=%s, target=%s", intent.action, intent.target_name)
                    return intent
            except Exception as e:
                logger.warning("LLM intent parsing failed: %s", e)

        # Fallback to VLM
        vlm = self._get_vlm()
        if vlm is not None:
            try:
                result = vlm.process_text(prompt)
                response_text = result.get("response", "")
                intent = self._parse_response(response_text, query)
                if intent:
                    logger.info("VLM parsed intent: action=%s, target=%s", intent.action, intent.target_name)
                    return intent
            except Exception as e:
                logger.warning("VLM intent parsing failed: %s", e)

        logger.warning("No model available for intent parsing")
        return None

    def _parse_response(self, response_text: str, query: str) -> Optional[Intent]:
        """Parse the model response into an Intent."""
        intent_data = self._extract_json(response_text)
        if not intent_data:
            logger.warning("Could not extract intent JSON from: %s", response_text[:200])
            return None

        action = intent_data.get("action", "")

        # Filter out non-action intents
        if action == "none" or not action:
            logger.debug("No action intent for query: %s", query)
            return None

        # Only allow valid device actions
        valid_actions = {"turn_on", "turn_off", "toggle", "set_brightness", "read", "set_color", "set_temperature"}
        if action not in valid_actions:
            logger.debug("Non-device action '%s' for query: %s", action, query)
            return None

        return Intent(
            action=action,
            target_type=intent_data.get("target_type"),
            target_name=intent_data.get("target_name"),
            parameters=intent_data.get("parameters", {}),
            confidence=float(intent_data.get("confidence", 0.0)),
            raw_query=query,
        )

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON object from text response."""
        # Try parsing the whole response as JSON first (most common case)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response (handles nested braces)
        # Match from first { to last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None


# Global parser instance
intent_parser = IntentParser()
