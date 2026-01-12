"""
Intent parser for natural language to action conversion.

Uses regex-first approach for fast intent extraction (~0ms).
Falls back to LLM only for ambiguous queries.
"""

import json
import logging
import re
from typing import Optional

from .actions import Intent

logger = logging.getLogger("atlas.capabilities.intent_parser")

# LLM prompt for ambiguous queries only
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

    Uses regex-first for speed (~0ms), LLM fallback for ambiguous queries.
    """

    # Device type keywords and their canonical types
    DEVICE_TYPES = {
        # Lights
        "light": "light",
        "lights": "light",
        "lamp": "light",
        "lamps": "light",
        "bulb": "light",
        "bulbs": "light",
        # Switches
        "switch": "switch",
        "switches": "switch",
        "outlet": "switch",
        "plug": "switch",
        # TV/Media
        "tv": "tv",
        "television": "tv",
        "roku": "tv",
        "firestick": "tv",
        "chromecast": "tv",
        "media": "media_player",
        "speaker": "media_player",
        "speakers": "media_player",
        # Climate
        "thermostat": "thermostat",
        "ac": "thermostat",
        "heat": "thermostat",
        "heater": "thermostat",
        # Other
        "fan": "fan",
        "fans": "fan",
        "camera": "camera",
        "sensor": "sensor",
    }

    # Common room/location names
    LOCATIONS = [
        "living room", "livingroom", "lounge",
        "bedroom", "master bedroom", "guest bedroom",
        "kitchen", "dining room", "dining",
        "bathroom", "bath", "restroom",
        "office", "study", "den",
        "garage", "basement", "attic",
        "porch", "patio", "deck", "backyard", "front yard",
        "hallway", "hall", "entryway", "foyer",
        "laundry", "laundry room", "utility",
        "nursery", "kids room", "playroom",
        "guest room", "spare room",
    ]

    def __init__(self):
        self._llm = None
        # Pre-compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for fast matching."""
        # Build device type pattern
        device_words = "|".join(re.escape(d) for d in self.DEVICE_TYPES.keys())

        # Build location pattern (sort by length to match longer first)
        sorted_locations = sorted(self.LOCATIONS, key=len, reverse=True)
        location_pattern = "|".join(re.escape(loc) for loc in sorted_locations)

        # Common action verb patterns
        turn_on_verbs = r"turn\s*on|switch\s*on|power\s*on|light\s*up|enable|activate"
        turn_off_verbs = r"turn\s*off|switch\s*off|power\s*off|shut\s*off|kill|disable|deactivate"

        # Pattern: turn on/off [the] [location] [device]
        # Examples: "turn on the living room lights", "shut off kitchen light", "kill the lights"
        self._turn_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?(?:go\s+)?"
            rf"({turn_on_verbs}|{turn_off_verbs})"
            rf"(?:\s+the)?"
            rf"(?:\s+({location_pattern}))?"
            rf"(?:\s+({device_words}))?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: turn on/off [the] [device] [in/at the] [location]
        # Examples: "turn on the lights in the kitchen"
        self._turn_reverse_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"({turn_on_verbs}|{turn_off_verbs})"
            rf"(?:\s+the)?"
            rf"(?:\s+({device_words}))?"
            rf"(?:\s+(?:in|at|for)(?:\s+the)?)?"
            rf"(?:\s+({location_pattern}))?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern for TV-specific commands
        # Examples: "turn off the tv", "turn on roku", "kill the tv", "shut off television"
        self._tv_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"({turn_on_verbs}|{turn_off_verbs})"
            rf"(?:\s+the)?"
            rf"\s*(tv|television|roku|firestick|chromecast|apple\s*tv|shield)"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: put TV to sleep / wake up TV
        # Examples: "put the tv to sleep", "wake up the roku"
        self._tv_sleep_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"(put|send)"
            rf"(?:\s+the)?"
            rf"\s*(tv|television|roku|firestick|chromecast|apple\s*tv|shield)"
            rf"\s+to\s+sleep"
            rf"\s*$",
            re.IGNORECASE
        )

        self._tv_wake_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"wake\s*(?:up)?"
            rf"(?:\s+the)?"
            rf"\s*(tv|television|roku|firestick|chromecast|apple\s*tv|shield)"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: dim/brighten [the] [location] [lights] [to X%]
        # Examples: "dim the living room to 50%", "brighten kitchen lights"
        self._dim_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"(dim|brighten|lower|raise)"
            rf"(?:\s+the)?"
            rf"(?:\s+({location_pattern}))?"
            rf"(?:\s+(?:lights?|lamps?))?"
            rf"(?:\s+(?:to|at|by))?"
            rf"(?:\s+(\d+)\s*%?)?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: "make it brighter/darker" or "make the lights brighter"
        self._make_bright_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"make"
            rf"(?:\s+the)?"
            rf"(?:\s+({location_pattern}))?"
            rf"(?:\s+(?:lights?|room|it))?"
            rf"\s+(brighter|darker|dimmer|lighter)"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: brightness presets - "lights to full", "set to max", "low brightness"
        self._brightness_preset_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?(?:set\s+)?"
            rf"(?:the\s+)?"
            rf"(?:({location_pattern})\s+)?"
            rf"(?:lights?\s+)?"
            rf"(?:to\s+)?"
            rf"(full|max|maximum|high|low|minimum|min|half|medium|mid)"
            rf"(?:\s+brightness)?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: set [the] [location] [device] to X%
        # Examples: "set living room lights to 50%", "set bedroom to 30%"
        self._set_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"set"
            rf"(?:\s+the)?"
            rf"(?:\s+({location_pattern}))?"
            rf"(?:\s+({device_words}))?"
            rf"(?:\s+(?:to|at))?"
            rf"\s+(\d+)\s*%?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: toggle [the] [location] [device]
        self._toggle_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"toggle"
            rf"(?:\s+the)?"
            rf"(?:\s+({location_pattern}))?"
            rf"(?:\s+({device_words}))?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern for "all" commands
        # Examples: "turn off all lights", "kill all the lights", "shut off all lights"
        self._all_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"({turn_on_verbs}|{turn_off_verbs})"
            rf"(?:\s+all)(?:\s+the)?"
            rf"\s+({device_words})"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: lights off/on (shorthand)
        self._shorthand_pattern = re.compile(
            rf"^(?:({location_pattern})\s+)?"
            rf"({device_words})"
            rf"\s+(on|off)"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: "I need light" / "it's too dark" / "it's too bright"
        self._contextual_pattern = re.compile(
            rf"^(?:it'?s?\s+)?(?:too\s+)?(dark|bright)"
            rf"(?:\s+in\s+(?:here|the\s+({location_pattern})))?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: media controls - play, pause, stop, mute
        self._media_control_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"(play|pause|stop|resume|mute|unmute)"
            rf"(?:\s+the)?"
            rf"(?:\s+(tv|television|roku|music|video|media|speaker|speakers))?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: volume control
        self._volume_pattern = re.compile(
            rf"^(?:please\s+)?(?:can\s+you\s+)?"
            rf"(?:turn\s+)?"
            rf"(volume\s+up|volume\s+down|louder|quieter)"
            rf"(?:\s+(?:the\s+)?(tv|television|roku|speaker|speakers))?"
            rf"\s*$",
            re.IGNORECASE
        )

        # Pattern: "lights" or "light" alone (often means toggle or turn on)
        self._bare_device_pattern = re.compile(
            rf"^({device_words})$",
            re.IGNORECASE
        )

    def _get_llm(self):
        """Lazy import LLM as fallback for intent extraction."""
        if self._llm is None:
            from ..services import llm_registry
            self._llm = llm_registry.get_active()
        return self._llm

    async def parse(self, query: str) -> Optional[Intent]:
        """
        Parse a natural language query into an Intent.

        Uses regex-first (~0ms), falls back to LLM for ambiguous queries.

        Args:
            query: Natural language query from the user

        Returns:
            Intent object if a device action is detected, None otherwise
        """
        query = query.strip()
        if not query:
            return None

        # Try regex-first (fast path ~0ms)
        intent = self._parse_regex(query)
        if intent:
            logger.info(
                "Regex intent for '%s': action=%s, target=%s (0ms)",
                query[:30], intent.action, intent.target_name
            )
            return intent

        # Check if this looks like a device command but regex didn't match
        if self._looks_like_device_command(query):
            # Fall back to LLM for ambiguous device commands
            logger.info("Ambiguous command, trying LLM: '%s'", query[:50])
            return await self._parse_with_llm(query)

        # Not a device command
        logger.debug("Not a device command: '%s'", query[:50])
        return None

    def _looks_like_device_command(self, query: str) -> bool:
        """Check if query looks like it might be a device command."""
        query_lower = query.lower()

        # Action keywords
        action_words = [
            # Power control
            "turn", "switch", "toggle", "power", "kill", "shut",
            "enable", "disable", "activate", "deactivate",
            # Lighting
            "dim", "brighten", "darker", "brighter", "lights", "light", "lamp",
            # Devices
            "tv", "television", "roku", "fan", "thermostat",
            # State
            "on", "off",
            # Brightness
            "set", "brightness", "full", "max", "low", "half",
            # Media
            "play", "pause", "stop", "mute", "volume", "louder", "quieter",
            # Contextual
            "dark", "bright", "sleep", "wake",
        ]

        # Check for action keywords
        return any(word in query_lower for word in action_words)

    def _determine_on_off_action(self, action_str: str) -> str:
        """Determine if action string means turn_on or turn_off."""
        action_lower = action_str.lower()
        # Keywords that mean "turn on"
        if any(kw in action_lower for kw in ["on", "enable", "activate", "light up"]):
            return "turn_on"
        # Keywords that mean "turn off"
        if any(kw in action_lower for kw in ["off", "disable", "deactivate", "kill", "shut"]):
            return "turn_off"
        # Default to turn_on if unclear
        return "turn_on"

    def _parse_regex(self, query: str) -> Optional[Intent]:
        """Try to parse intent using regex patterns."""
        query_lower = query.lower().strip()

        # Try TV-specific pattern first (highest priority for TV commands)
        match = self._tv_pattern.match(query_lower)
        if match:
            action_str, device = match.groups()
            action = self._determine_on_off_action(action_str)
            return Intent(
                action=action,
                target_type="tv",
                target_name=device.replace(" ", ""),  # "apple tv" -> "appletv"
                parameters={},
                confidence=0.95,
                raw_query=query,
            )

        # Try TV sleep pattern ("put the tv to sleep")
        match = self._tv_sleep_pattern.match(query_lower)
        if match:
            _, device = match.groups()
            return Intent(
                action="turn_off",
                target_type="tv",
                target_name=device.replace(" ", ""),
                parameters={},
                confidence=0.95,
                raw_query=query,
            )

        # Try TV wake pattern ("wake up the tv")
        match = self._tv_wake_pattern.match(query_lower)
        if match:
            device = match.group(1)
            return Intent(
                action="turn_on",
                target_type="tv",
                target_name=device.replace(" ", ""),
                parameters={},
                confidence=0.95,
                raw_query=query,
            )

        # Try "all" pattern
        match = self._all_pattern.match(query_lower)
        if match:
            action_str, device = match.groups()
            action = self._determine_on_off_action(action_str)
            device_type = self.DEVICE_TYPES.get(device, "light")
            return Intent(
                action=action,
                target_type=device_type,
                target_name=None,  # None means "all"
                parameters={},
                confidence=0.95,
                raw_query=query,
            )

        # Try turn on/off pattern
        match = self._turn_pattern.match(query_lower)
        if match:
            action_str, location, device = match.groups()
            action = self._determine_on_off_action(action_str)
            device_type = self.DEVICE_TYPES.get(device, "light") if device else "light"
            return Intent(
                action=action,
                target_type=device_type,
                target_name=location,
                parameters={},
                confidence=0.90,
                raw_query=query,
            )

        # Try reverse pattern (lights in the kitchen)
        match = self._turn_reverse_pattern.match(query_lower)
        if match:
            action_str, device, location = match.groups()
            action = self._determine_on_off_action(action_str)
            device_type = self.DEVICE_TYPES.get(device, "light") if device else "light"
            return Intent(
                action=action,
                target_type=device_type,
                target_name=location,
                parameters={},
                confidence=0.90,
                raw_query=query,
            )

        # Try dim/brighten pattern
        match = self._dim_pattern.match(query_lower)
        if match:
            action_str, location, brightness = match.groups()

            # Default brightness for dim/brighten without percentage
            if brightness:
                brightness_value = int(int(brightness) * 255 / 100)
            elif action_str in ("dim", "lower"):
                brightness_value = int(30 * 255 / 100)  # Default dim to 30%
            else:
                brightness_value = int(100 * 255 / 100)  # Default brighten to 100%

            return Intent(
                action="set_brightness",
                target_type="light",
                target_name=location,
                parameters={"brightness": brightness_value},
                confidence=0.90,
                raw_query=query,
            )

        # Try "make it brighter/darker" pattern
        match = self._make_bright_pattern.match(query_lower)
        if match:
            location, direction = match.groups()
            if direction in ("brighter", "lighter"):
                brightness_value = int(80 * 255 / 100)  # Increase to 80%
            else:  # darker, dimmer
                brightness_value = int(30 * 255 / 100)  # Decrease to 30%

            return Intent(
                action="set_brightness",
                target_type="light",
                target_name=location,
                parameters={"brightness": brightness_value},
                confidence=0.85,
                raw_query=query,
            )

        # Try brightness preset pattern ("full brightness", "lights to max")
        match = self._brightness_preset_pattern.match(query_lower)
        if match:
            location, preset = match.groups()
            preset_values = {
                "full": 100, "max": 100, "maximum": 100, "high": 100,
                "half": 50, "medium": 50, "mid": 50,
                "low": 20, "min": 10, "minimum": 10,
            }
            percent = preset_values.get(preset.lower(), 100)
            brightness_value = int(percent * 255 / 100)

            return Intent(
                action="set_brightness",
                target_type="light",
                target_name=location,
                parameters={"brightness": brightness_value},
                confidence=0.85,
                raw_query=query,
            )

        # Try set pattern
        match = self._set_pattern.match(query_lower)
        if match:
            location, device, brightness = match.groups()
            device_type = self.DEVICE_TYPES.get(device, "light") if device else "light"
            brightness_value = int(int(brightness) * 255 / 100)

            return Intent(
                action="set_brightness",
                target_type=device_type,
                target_name=location,
                parameters={"brightness": brightness_value},
                confidence=0.90,
                raw_query=query,
            )

        # Try toggle pattern
        match = self._toggle_pattern.match(query_lower)
        if match:
            location, device = match.groups()
            device_type = self.DEVICE_TYPES.get(device, "light") if device else "light"
            return Intent(
                action="toggle",
                target_type=device_type,
                target_name=location,
                parameters={},
                confidence=0.90,
                raw_query=query,
            )

        # Try contextual pattern ("it's too dark", "too bright")
        match = self._contextual_pattern.match(query_lower)
        if match:
            condition, location = match.groups()
            if condition == "dark":
                action = "turn_on"
            else:  # bright
                action = "turn_off"

            return Intent(
                action=action,
                target_type="light",
                target_name=location,
                parameters={},
                confidence=0.80,
                raw_query=query,
            )

        # Try media control pattern (play, pause, stop, mute)
        match = self._media_control_pattern.match(query_lower)
        if match:
            control, device = match.groups()
            device_type = "tv" if device in ("tv", "television", "roku") else "media_player"
            return Intent(
                action=control,  # play, pause, stop, mute, unmute, resume
                target_type=device_type,
                target_name=device,
                parameters={},
                confidence=0.90,
                raw_query=query,
            )

        # Try volume pattern
        match = self._volume_pattern.match(query_lower)
        if match:
            direction, device = match.groups()
            action = "volume_up" if direction in ("volume up", "louder") else "volume_down"
            device_type = "tv" if device in ("tv", "television", "roku") else "media_player"
            return Intent(
                action=action,
                target_type=device_type,
                target_name=device,
                parameters={},
                confidence=0.90,
                raw_query=query,
            )

        # Try shorthand pattern (e.g., "lights off", "kitchen lights on")
        match = self._shorthand_pattern.match(query_lower)
        if match:
            location, device, action_str = match.groups()
            action = "turn_on" if action_str == "on" else "turn_off"
            device_type = self.DEVICE_TYPES.get(device, "light")
            return Intent(
                action=action,
                target_type=device_type,
                target_name=location,
                parameters={},
                confidence=0.85,
                raw_query=query,
            )

        return None

    async def _parse_with_llm(self, query: str) -> Optional[Intent]:
        """Fall back to LLM for ambiguous queries."""
        llm = self._get_llm()
        if llm is None:
            logger.warning("No LLM available for intent parsing")
            return None

        prompt = INTENT_EXTRACTION_PROMPT.format(query=query)

        try:
            from ..services.protocols import Message
            messages = [
                Message(role="system", content="You parse commands. Output ONLY valid JSON."),
                Message(role="user", content=prompt),
            ]
            result = llm.chat(messages=messages, max_tokens=150, temperature=0.1)
            response_text = result.get("response", "")
            logger.info("LLM intent for '%s': %s", query[:30], response_text[:100])
            intent = self._parse_response(response_text, query)
            if intent:
                logger.info("Intent: action=%s, target=%s", intent.action, intent.target_name)
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

        # Get parameters from LLM response
        parameters = intent_data.get("parameters", {})

        # Extract brightness from query if LLM didn't provide it
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

    def _normalize_action(self, raw_action: str) -> str:
        """Normalize LLM action output to valid action types."""
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
        if "dim" in action:
            return "set_brightness"
        if any(x in action for x in ["%", "percent", "brightness"]):
            return "set_brightness"
        if "toggle" in action:
            return "toggle"
        if "read" in action or "temperature" in action or "status" in action:
            return "read"

        return action

    def _extract_brightness_from_query(self, query: str) -> int | None:
        """Extract brightness value from natural language query."""
        query_lower = query.lower()

        # Match percentage patterns: "50%", "50 percent", "50 %"
        match = re.search(r'(\d+)\s*(?:%|percent)', query_lower)
        if match:
            percent = int(match.group(1))
            percent = max(0, min(100, percent))
            return int(percent * 255 / 100)

        # Match "to X" or "at X" patterns (assume percentage if no unit)
        match = re.search(r'(?:to|at)\s+(\d+)(?!\s*(?:degrees|celsius|fahrenheit|f|c))', query_lower)
        if match:
            value = int(match.group(1))
            if value <= 100:
                return int(value * 255 / 100)

        return None

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON object from text response."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

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
