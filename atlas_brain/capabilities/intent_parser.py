"""
Intent parser for natural language to action conversion.

Uses LLM for unified intent extraction (devices, tools, conversation).
"""

import json
import logging
import time
from typing import Any, Optional

from .actions import Intent

logger = logging.getLogger("atlas.capabilities.intent_parser")

# Compact prompt for fast intent extraction
UNIFIED_INTENT_PROMPT = """Parse intent. Output JSON only.
DEVICES: {devices}
TOOLS: time,weather,traffic,location,calendar,reminder,reminders
ACTIONS: turn_on,turn_off,toggle,set_brightness,query,conversation
Format: {{"action":"X","target_type":"Y","target_name":"Z","parameters":{{}},"confidence":0.95}}
"turn on kitchen light"->{{"action":"turn_on","target_type":"light","target_name":"kitchen","parameters":{{}},"confidence":0.95}}
"what time"->{{"action":"query","target_type":"tool","target_name":"time","parameters":{{}},"confidence":0.95}}
"dim to 50%"->{{"action":"set_brightness","target_type":"light","target_name":null,"parameters":{{"brightness":50}},"confidence":0.95}}
"what's on my calendar"->{{"action":"query","target_type":"tool","target_name":"calendar","parameters":{{}},"confidence":0.95}}
"remind me to call mom at 5pm"->{{"action":"query","target_type":"tool","target_name":"reminder","parameters":{{"message":"call mom","when":"at 5pm"}},"confidence":0.95}}
"list my reminders"->{{"action":"query","target_type":"tool","target_name":"reminders","parameters":{{}},"confidence":0.95}}
"hello"->{{"action":"conversation","target_type":null,"target_name":null,"parameters":{{}},"confidence":0.9}}
User: {query}
JSON:"""


class IntentParser:
    """
    Parses natural language queries into structured intents.

    Uses LLM for all intent extraction - unified system for devices and tools.
    """

    def __init__(self) -> None:
        self._llm = None
        self._device_cache: Optional[str] = None
        self._device_cache_time: float = 0.0
        self._cache_ttl: int = 60

    def _get_llm(self) -> Any:
        """Get LLM for intent extraction."""
        from ..services import llm_registry
        return llm_registry.get_active()

    def _get_config(self) -> Any:
        """Get intent config."""
        from ..config import settings
        return settings.intent

    def _get_device_list(self) -> str:
        """
        Build device list for prompt from CapabilityRegistry.

        Caches the result for performance.
        """
        config = self._get_config()
        now = time.time()

        # Return cached if still valid
        if self._device_cache and (now - self._device_cache_time) < config.device_cache_ttl:
            return self._device_cache

        try:
            from .registry import capability_registry

            devices_by_type: dict[str, list[str]] = {}
            for cap in capability_registry.list_all():
                cap_type = cap.capability_type.value if hasattr(cap.capability_type, "value") else str(cap.capability_type)
                if cap_type not in devices_by_type:
                    devices_by_type[cap_type] = []
                devices_by_type[cap_type].append(cap.name)

            if not devices_by_type:
                self._device_cache = "No devices registered"
            else:
                lines = []
                for device_type, names in devices_by_type.items():
                    lines.append(f"- {device_type}: {', '.join(names)}")
                self._device_cache = "\n".join(lines)

            self._device_cache_time = now
            logger.debug("Device cache refreshed: %s", self._device_cache)

        except Exception as e:
            logger.warning("Failed to get device list: %s", e)
            self._device_cache = "No devices available"
            self._device_cache_time = now

        return self._device_cache

    def invalidate_cache(self) -> None:
        """Force cache refresh on next call."""
        self._device_cache = None
        self._device_cache_time = 0.0

    async def parse(self, query: str) -> Optional[Intent]:
        """
        Parse a natural language query into an Intent.

        Uses LLM for all intent extraction.

        Args:
            query: Natural language query from the user

        Returns:
            Intent object with action, target_type, target_name, parameters
        """
        query = query.strip()
        if not query:
            return None

        # Strip wake word prefixes
        query = self._strip_wake_word(query)
        if not query:
            return None

        # Filter out very short queries (likely garbage from mic feedback)
        if len(query) < 3 or len(query.split()) < 2:
            logger.debug("Query too short, likely garbage: '%s'", query)
            return None

        return await self._parse_with_llm(query)

    def _strip_wake_word(self, query: str) -> str:
        """Strip wake word prefix from query if present."""
        import re
        pattern = r"^(?:hey\s+)?(?:jarvis|atlas|computer|assistant)[,.]?\s*"
        stripped = re.sub(pattern, "", query, flags=re.IGNORECASE).strip()
        if stripped != query:
            logger.debug("Stripped wake word: '%s' -> '%s'", query[:30], stripped[:30])
        return stripped

    async def _parse_with_llm(self, query: str) -> Optional[Intent]:
        """Parse intent using LLM."""
        llm = self._get_llm()
        if llm is None:
            logger.warning("No LLM available for intent parsing")
            return None

        config = self._get_config()
        device_list = self._get_device_list()
        logger.info("Intent parsing with LLM: %s", llm.model if hasattr(llm, 'model') else 'unknown')
        logger.debug("Device list for intent: %s", device_list)
        prompt = UNIFIED_INTENT_PROMPT.format(devices=device_list, query=query)

        try:
            from ..services.protocols import Message

            messages = [
                Message(role="system", content="You parse intents. Output ONLY valid JSON."),
                Message(role="user", content=prompt),
            ]

            start_time = time.perf_counter()
            result = llm.chat(
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            response_text = result.get("response", "")
            logger.info("LLM intent (%.0fms): %s", duration_ms, response_text[:100])

            intent = self._parse_response(response_text, query)
            if intent:
                logger.info(
                    "Intent: action=%s, target_type=%s, target_name=%s",
                    intent.action,
                    intent.target_type,
                    intent.target_name,
                )
            return intent

        except Exception as e:
            logger.warning("LLM intent parsing failed: %s", e)
            return None

    def _parse_response(self, response_text: str, query: str) -> Optional[Intent]:
        """Parse the LLM response into an Intent."""
        intent_data = self._extract_json(response_text)
        if not intent_data:
            logger.warning("Could not extract intent JSON from: %s", response_text[:200])
            return None

        action = intent_data.get("action", "")

        # Filter out non-action intents
        if not action or action == "none":
            logger.debug("No action intent for query: %s", query)
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
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None


# Global parser instance
intent_parser = IntentParser()
