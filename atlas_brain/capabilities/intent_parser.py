"""
Intent parser for natural language to action conversion.

Uses the VLM to extract structured intents from user queries.
"""

import json
import logging
import re
from typing import Optional

from .actions import Intent

logger = logging.getLogger("atlas.capabilities.intent_parser")

INTENT_EXTRACTION_PROMPT = """You are an intent parser for a smart home system. Extract the user's intent from their query.

Respond ONLY with a JSON object in this exact format (no other text):
{{
  "action": "the action verb (turn_on, turn_off, set_brightness, toggle, read, etc.)",
  "target_type": "device type (light, switch, sensor, camera, thermostat) or null",
  "target_name": "location or device name mentioned (living room, bedroom, kitchen) or null",
  "parameters": {{}},
  "confidence": 0.0
}}

Examples:
- "turn on the living room lights" -> {{"action":"turn_on","target_type":"light","target_name":"living room","parameters":{{}},"confidence":0.95}}
- "set bedroom lights to 50%" -> {{"action":"set_brightness","target_type":"light","target_name":"bedroom","parameters":{{"brightness":127}},"confidence":0.9}}
- "what's the temperature" -> {{"action":"read","target_type":"sensor","target_name":null,"parameters":{{}},"confidence":0.85}}
- "turn off all lights" -> {{"action":"turn_off","target_type":"light","target_name":null,"parameters":{{}},"confidence":0.95}}

User query: {query}

JSON response:"""


class IntentParser:
    """
    Parses natural language queries into structured intents using the VLM.
    """

    def __init__(self):
        self._vlm = None

    def _get_vlm(self):
        """Lazy import to avoid circular dependencies."""
        if self._vlm is None:
            from ..services import vlm_registry
            self._vlm = vlm_registry.get_active()
        return self._vlm

    async def parse(self, query: str) -> Optional[Intent]:
        """
        Parse a natural language query into an Intent.

        Uses the VLM to extract structured intent information.

        Args:
            query: Natural language query from the user

        Returns:
            Intent object if parsing succeeds, None otherwise
        """
        vlm = self._get_vlm()
        if vlm is None:
            logger.warning("No VLM loaded, cannot parse intent")
            return None

        prompt = INTENT_EXTRACTION_PROMPT.format(query=query)

        try:
            result = vlm.process_text(prompt)
            response_text = result.get("response", "")

            intent_data = self._extract_json(response_text)
            if not intent_data:
                logger.warning("Could not extract intent JSON from: %s", response_text[:200])
                return None

            intent = Intent(
                action=intent_data.get("action", ""),
                target_type=intent_data.get("target_type"),
                target_name=intent_data.get("target_name"),
                parameters=intent_data.get("parameters", {}),
                confidence=float(intent_data.get("confidence", 0.0)),
                raw_query=query,
            )

            logger.info("Parsed intent: action=%s, target=%s", intent.action, intent.target_name)
            return intent

        except Exception as e:
            logger.exception("Intent parsing failed for query: %s", query)
            return None

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
