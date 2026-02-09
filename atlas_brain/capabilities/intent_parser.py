"""
Intent parser for natural language to action conversion.

Uses LLM for unified intent extraction (devices, tools, conversation).
"""

import json
import logging
import re
import time
from typing import Any, Optional

from .actions import Intent

logger = logging.getLogger("atlas.capabilities.intent_parser")


def _normalize_spoken_numbers(text: str) -> str:
    """Convert spoken number words to digits for phone number recognition."""
    singles = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    }
    teens = {
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19',
    }
    tens = {
        'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
    }
    all_nums = {**singles, **teens, **tens}

    # "twenty fifty" -> "2050"
    p1 = r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+'
    p1 += r'(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
    p1 += r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b'
    text = re.sub(p1, lambda m: all_nums[m.group(1).lower()] + all_nums[m.group(2).lower()],
                  text, flags=re.IGNORECASE)

    # "two seventeen" -> "217"
    p2 = r'\b(one|two|three|four|five|six|seven|eight|nine)\s+'
    p2 += r'(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b'
    text = re.sub(p2, lambda m: singles[m.group(1).lower()] + teens[m.group(2).lower()],
                  text, flags=re.IGNORECASE)

    # "twenty one" -> "21"
    p3 = r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+'
    p3 += r'(one|two|three|four|five|six|seven|eight|nine)\b'
    text = re.sub(p3, lambda m: str(int(tens[m.group(1).lower()]) + int(singles[m.group(2).lower()])),
                  text, flags=re.IGNORECASE)

    # Standalone numbers
    for word, digit in sorted(all_nums.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)

    # Format digit sequences as phone numbers
    def fmt(m: re.Match) -> str:
        d = re.sub(r'\s', '', m.group(1))
        if len(d) == 7:
            return f"{d[:3]}-{d[3:]}"
        if len(d) == 10:
            return f"{d[:3]}-{d[3:6]}-{d[6:]}"
        return m.group(0)

    text = re.sub(r'\b(\d[\d\s]{5,12}\d)\b', fmt, text)
    return text

# Compact prompt template for fast intent extraction
# NOTE: {tools} is populated dynamically from ToolRegistry
# NOTE: Booking, reminder, email, calendar, security, presence queries are
# routed to dedicated workflows BEFORE reaching this parser. Parameterless
# tools (time, weather, traffic, location, calendar read, list reminders)
# execute directly via fast path. This prompt only handles:
# - Device commands that bypass the device resolver (pronouns, ambiguous)
# - Parameterized tool calls (notification, camera feed)
# - Low-confidence fallthrough from the semantic router
UNIFIED_INTENT_PROMPT = """Parse intent. Output JSON only.
DEVICES: {devices}
TOOLS: {tools}
ACTIONS: turn_on,turn_off,toggle,set_brightness,query,conversation
Format: {{"action":"X","target_type":"Y","target_name":"Z","parameters":{{}},"confidence":0.95}}
"turn on kitchen light"->{{"action":"turn_on","target_type":"light","target_name":"kitchen","parameters":{{}},"confidence":0.95}}
"turn it on"->{{"action":"turn_on","target_type":"device","target_name":null,"parameters":{{}},"confidence":0.95}}
"turn them off"->{{"action":"turn_off","target_type":"device","target_name":null,"parameters":{{}},"confidence":0.95}}
"dim to 50%"->{{"action":"set_brightness","target_type":"light","target_name":null,"parameters":{{"brightness":50}},"confidence":0.95}}
"send a notification saying dinner is ready"->{{"action":"query","target_type":"tool","target_name":"notification","parameters":{{"message":"dinner is ready"}},"confidence":0.95}}
"show the front door camera on the left monitor"->{{"action":"query","target_type":"tool","target_name":"show camera","parameters":{{"camera_name":"front door","display":"left"}},"confidence":0.95}}
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
        self._tools_cache: Optional[str] = None
        self._tools_cache_time: float = 0.0
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

    def _get_tools_list(self) -> str:
        """
        Build tools list for prompt from ToolRegistry.

        Dynamically discovers all registered tools and their aliases.
        Caches the result for performance.
        """
        now = time.time()

        # Return cached if still valid
        if self._tools_cache and (now - self._tools_cache_time) < self._cache_ttl:
            return self._tools_cache

        try:
            from ..tools import tool_registry

            aliases = tool_registry.get_all_aliases()
            if aliases:
                self._tools_cache = ",".join(sorted(set(aliases)))
            else:
                # Fallback to tool names if no aliases defined
                names = tool_registry.list_names()
                self._tools_cache = ",".join(sorted(names)) if names else "time,weather,calendar"

            self._tools_cache_time = now
            logger.debug("Tools cache refreshed: %s", self._tools_cache)

        except Exception as e:
            logger.warning("Failed to get tools list: %s", e)
            # Fallback to basic tools
            self._tools_cache = "time,weather,calendar,reminder,reminders"
            self._tools_cache_time = now

        return self._tools_cache

    def invalidate_cache(self) -> None:
        """Force cache refresh on next call."""
        self._device_cache = None
        self._device_cache_time = 0.0
        self._tools_cache = None
        self._tools_cache_time = 0.0
        try:
            from .device_resolver import get_device_resolver
            get_device_resolver().invalidate()
        except Exception:
            pass

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

        # Normalize spoken numbers to digits (e.g., "two seventeen" -> "217")
        query = _normalize_spoken_numbers(query)

        # Filter out very short queries (likely garbage from mic feedback)
        if len(query) < 3 or len(query.split()) < 2:
            logger.debug("Query too short, likely garbage: '%s'", query)
            return None

        # Fast path: embedding-based device resolution
        from ..config import settings
        if settings.device_resolver.enabled:
            try:
                from .device_resolver import get_device_resolver
                resolver = get_device_resolver()
                result = await resolver.resolve(query)
                if result is not None:
                    logger.info(
                        "Fast device resolve: %s -> %s (%.0fms)",
                        query[:40], result.device_name, result.resolve_time_ms,
                    )
                    return result.intent
            except Exception as e:
                logger.warning("Device resolver failed, falling through to LLM: %s", e)

        # Slow path: LLM-based parsing
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
        """Parse intent using LLM with retry on truncated responses."""
        llm = self._get_llm()
        if llm is None:
            logger.warning("No LLM available for intent parsing")
            return None

        config = self._get_config()
        device_list = self._get_device_list()
        tools_list = self._get_tools_list()
        logger.info("Intent parsing with LLM: %s", llm.model if hasattr(llm, 'model') else 'unknown')
        logger.debug("Device list for intent: %s", device_list)
        logger.debug("Tools list for intent: %s", tools_list)
        prompt = UNIFIED_INTENT_PROMPT.format(devices=device_list, tools=tools_list, query=query)

        from ..services.protocols import Message

        messages = [
            Message(role="system", content="You parse intents. Output ONLY valid JSON."),
            Message(role="user", content=prompt),
        ]

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                start_time = time.perf_counter()
                result = llm.chat(
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                duration_ms = (time.perf_counter() - start_time) * 1000

                response_text = result.get("response", "")
                logger.info("LLM intent (%.0fms): '%s'", duration_ms, response_text)

                intent = self._parse_response(response_text, query)
                if intent:
                    logger.info(
                        "Intent: action=%s, target_type=%s, target_name=%s",
                        intent.action,
                        intent.target_type,
                        intent.target_name,
                    )
                    return intent

                # Retry if JSON was truncated (response has { but not valid JSON)
                if "{" in response_text and attempt < max_retries:
                    logger.warning("Truncated JSON response, retrying (%d/%d)", attempt + 1, max_retries)
                    continue

                return None

            except Exception as e:
                logger.warning("LLM intent parsing failed: %s", e)
                if attempt < max_retries:
                    continue
                return None

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
