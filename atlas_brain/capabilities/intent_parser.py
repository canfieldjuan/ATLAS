"""
Intent parser for natural language to action conversion.

Uses the VLM for fast structured intent extraction (~500ms).
LLM is reserved for reasoning/conversation, not classification tasks.
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
  "target_type": "light|switch|sensor|camera|thermostat|tv|media_player|null",
  "target_name": "location name or device name or null",
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
- "turn off the tv" -> {{"action":"turn_off","target_type":"tv","target_name":"tv","parameters":{{}},"confidence":0.95}}
- "turn on the roku" -> {{"action":"turn_on","target_type":"tv","target_name":"roku","parameters":{{}},"confidence":0.95}}
- "put the television to sleep" -> {{"action":"turn_off","target_type":"tv","target_name":"television","parameters":{{}},"confidence":0.9}}
- "what can you do?" -> {{"action":"none","target_type":null,"target_name":null,"parameters":{{}},"confidence":0.0}}
- "hello" -> {{"action":"none","target_type":null,"target_name":null,"parameters":{{}},"confidence":0.0}}

User query: {query}

JSON:"""


class IntentParser:
    """
    Parses natural language queries into structured intents.

    Uses VLM (Moondream) for fast structured extraction (~500ms).
    Intent parsing is a classification task - doesn't need heavy reasoning.
    """

    def __init__(self):
        self._vlm = None

    def _get_vlm(self):
        """Lazy import VLM for intent extraction."""
        if self._vlm is None:
            from ..services import vlm_registry
            self._vlm = vlm_registry.get_active()
        return self._vlm

    async def parse(self, query: str) -> Optional[Intent]:
        """
        Parse a natural language query into an Intent.

        Uses VLM for fast structured intent extraction.

        Args:
            query: Natural language query from the user

        Returns:
            Intent object if a device action is detected, None otherwise
        """
        vlm = self._get_vlm()
        if vlm is None:
            logger.warning("VLM not available for intent parsing")
            return None

        prompt = INTENT_EXTRACTION_PROMPT.format(query=query)

        try:
            result = vlm.process_text(prompt)
            response_text = result.get("response", "")
            logger.info("VLM response for '%s': %s", query[:30], response_text[:200])
            intent = self._parse_response(response_text, query)
            if intent:
                logger.info("Parsed intent: action=%s, target=%s", intent.action, intent.target_name)
                return intent
        except Exception as e:
            logger.warning("Intent parsing failed: %s", e)

        return None

    def _normalize_action(self, raw_action: str) -> str:
        """Normalize VLM action output to valid action types."""
        action = raw_action.lower().strip()

        # Direct match
        valid_actions = {"turn_on", "turn_off", "toggle", "set_brightness", "read", "set_color", "set_temperature"}
        if action in valid_actions:
            return action

        # Pattern matching for common variations
        if any(x in action for x in ["turn on", "switch on", "power on"]):
            return "turn_on"
        if any(x in action for x in ["turn off", "switch off", "power off"]):
            return "turn_off"
        # "dim" always means brightness control
        if "dim" in action:
            return "set_brightness"
        # Percentage patterns for brightness
        if any(x in action for x in ["%", "percent", "brightness"]):
            return "set_brightness"
        if "toggle" in action:
            return "toggle"
        if "read" in action or "temperature" in action or "status" in action:
            return "read"

        return action  # Return as-is if no match

    def _extract_brightness_from_query(self, query: str) -> int | None:
        """
        Extract brightness value from natural language query.

        Handles patterns like:
        - "set to 50%", "to 50 percent", "50%"
        - "dim to 30", "brightness 80"

        Returns brightness as 0-255 value, or None if not found.
        """
        query_lower = query.lower()

        # Match percentage patterns: "50%", "50 percent", "50 %"
        match = re.search(r'(\d+)\s*(?:%|percent)', query_lower)
        if match:
            percent = int(match.group(1))
            # Clamp to 0-100 and convert to 0-255
            percent = max(0, min(100, percent))
            return int(percent * 255 / 100)

        # Match "to X" or "at X" patterns (assume percentage if no unit)
        match = re.search(r'(?:to|at)\s+(\d+)(?!\s*(?:degrees|celsius|fahrenheit|f|c))', query_lower)
        if match:
            value = int(match.group(1))
            if value <= 100:
                # Likely a percentage
                return int(value * 255 / 100)

        return None

    def _parse_response(self, response_text: str, query: str) -> Optional[Intent]:
        """Parse the model response into an Intent."""
        intent_data = self._extract_json(response_text)
        if not intent_data:
            logger.warning("Could not extract intent JSON from: %s", response_text[:200])
            return None

        raw_action = intent_data.get("action", "")
        action = self._normalize_action(raw_action)

        # Filter out non-action intents
        if action == "none" or not action:
            logger.info("No action intent (action=%s) for query: %s", action, query)
            return None

        # Only allow valid device actions
        valid_actions = {"turn_on", "turn_off", "toggle", "set_brightness", "read", "set_color", "set_temperature"}
        if action not in valid_actions:
            logger.info("Non-device action '%s' (raw: %s) for query: %s", action, raw_action[:30], query)
            return None

        # Get parameters from VLM response
        parameters = intent_data.get("parameters", {})

        # Extract brightness from query if VLM didn't provide it
        if action == "set_brightness" and "brightness" not in parameters:
            brightness = self._extract_brightness_from_query(query)
            if brightness is not None:
                parameters["brightness"] = brightness
                logger.info("Extracted brightness=%d from query", brightness)

        return Intent(
            action=action,
            target_type=intent_data.get("target_type"),
            target_name=intent_data.get("target_name"),
            parameters=parameters,
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
