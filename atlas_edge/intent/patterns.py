"""
Pattern-based device intent parsing for edge devices.

Uses regex patterns for fast, reliable device command parsing
without requiring LLM inference.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("atlas.edge.intent.patterns")


@dataclass
class ParsedDeviceIntent:
    """Parsed device intent from pattern matching."""

    action: str  # turn_on, turn_off, toggle, set_brightness, etc.
    target_type: str  # light, switch, fan, etc.
    target_name: Optional[str] = None  # kitchen, living room, etc.
    parameters: dict = None
    confidence: float = 0.9
    pattern_matched: str = ""

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# Device patterns with named groups
# Format: (compiled_regex, action, target_type, extract_params_func)
# NOTE: Order matters! More specific patterns must come before generic ones.
DEVICE_PATTERNS = [
    # Fan patterns (before generic light pattern)
    (
        re.compile(
            r"turn\s+(on|off)\s+(?:the\s+)?(.+?)\s+fan$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(1).lower() == "on" else "turn_off"),
        "fan",
        lambda m: {"target_name": m.group(2).strip()},
    ),
    (
        re.compile(
            r"(?:the\s+)?(.+?)\s+fan\s+(on|off)$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(2).lower() == "on" else "turn_off"),
        "fan",
        lambda m: {"target_name": m.group(1).strip()},
    ),
    # Switch patterns (before generic light pattern)
    (
        re.compile(
            r"turn\s+(on|off)\s+(?:the\s+)?(.+?)\s+switch$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(1).lower() == "on" else "turn_off"),
        "switch",
        lambda m: {"target_name": m.group(2).strip()},
    ),
    # TV patterns (before generic light pattern)
    (
        re.compile(
            r"turn\s+(on|off)\s+(?:the\s+)?(?:tv|television)(?:\s+in\s+(?:the\s+)?(.+))?$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(1).lower() == "on" else "turn_off"),
        "media_player",
        lambda m: {"target_name": m.group(2).strip() if m.group(2) else "tv"},
    ),
    # Light patterns - turn on/off (requires "light" or "lights" in the phrase)
    (
        re.compile(
            r"turn\s+(on|off)\s+(?:the\s+)?(.+?)\s+lights?$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(1).lower() == "on" else "turn_off"),
        "light",
        lambda m: {"target_name": m.group(2).strip()},
    ),
    (
        re.compile(
            r"(?:switch|flip)\s+(on|off)\s+(?:the\s+)?(.+?)(?:\s+light)?(?:\s+lights)?$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(1).lower() == "on" else "turn_off"),
        "light",
        lambda m: {"target_name": m.group(2).strip()},
    ),
    # Lights on/off (reverse word order)
    (
        re.compile(
            r"(?:the\s+)?(.+?)\s+lights?\s+(on|off)$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(2).lower() == "on" else "turn_off"),
        "light",
        lambda m: {"target_name": m.group(1).strip()},
    ),
    # Dim patterns
    (
        re.compile(
            r"dim\s+(?:the\s+)?(.+?)\s+to\s+(\d+)(?:\s*%)?$",
            re.IGNORECASE,
        ),
        lambda m: "set_brightness",
        "light",
        lambda m: {"target_name": m.group(1).strip(), "brightness": int(m.group(2))},
    ),
    (
        re.compile(
            r"set\s+(?:the\s+)?(.+?)\s+(?:brightness\s+)?to\s+(\d+)(?:\s*%)?$",
            re.IGNORECASE,
        ),
        lambda m: "set_brightness",
        "light",
        lambda m: {"target_name": m.group(1).strip(), "brightness": int(m.group(2))},
    ),
    # Brighten/dim by amount
    (
        re.compile(
            r"(brighten|dim)\s+(?:the\s+)?(.+?)(?:\s+by\s+(\d+)(?:\s*%)?)?$",
            re.IGNORECASE,
        ),
        lambda m: "brighten" if m.group(1).lower() == "brighten" else "dim",
        "light",
        lambda m: {
            "target_name": m.group(2).strip(),
            "amount": int(m.group(3)) if m.group(3) else 10,
        },
    ),
    # Toggle patterns
    (
        re.compile(
            r"toggle\s+(?:the\s+)?(.+?)(?:\s+light)?(?:\s+lights)?$",
            re.IGNORECASE,
        ),
        lambda m: "toggle",
        "light",
        lambda m: {"target_name": m.group(1).strip()},
    ),
    # Scene patterns
    (
        re.compile(
            r"(?:activate|set|turn on)\s+(?:the\s+)?(.+?)\s+scene$",
            re.IGNORECASE,
        ),
        lambda m: "turn_on",
        "scene",
        lambda m: {"target_name": m.group(1).strip()},
    ),
    (
        re.compile(
            r"(?:set\s+)?scene\s+(?:to\s+)?(.+)$",
            re.IGNORECASE,
        ),
        lambda m: "turn_on",
        "scene",
        lambda m: {"target_name": m.group(1).strip()},
    ),
    # Media player patterns
    (
        re.compile(
            r"(pause|play|stop|mute|unmute)\s+(?:the\s+)?(.+)$",
            re.IGNORECASE,
        ),
        lambda m: m.group(1).lower(),
        "media_player",
        lambda m: {"target_name": m.group(2).strip()},
    ),
    # Volume patterns
    (
        re.compile(
            r"(?:set\s+)?volume\s+(?:to\s+)?(\d+)(?:\s*%)?(?:\s+(?:on|for)\s+(?:the\s+)?(.+))?$",
            re.IGNORECASE,
        ),
        lambda m: "set_volume",
        "media_player",
        lambda m: {
            "target_name": m.group(2).strip() if m.group(2) else None,
            "volume": int(m.group(1)),
        },
    ),
    (
        re.compile(
            r"(turn\s+)?(volume\s+)?(up|down)(?:\s+(?:on|for)\s+(?:the\s+)?(.+))?$",
            re.IGNORECASE,
        ),
        lambda m: ("volume_up" if m.group(3).lower() == "up" else "volume_down"),
        "media_player",
        lambda m: {"target_name": m.group(4).strip() if m.group(4) else None},
    ),
    # Generic "turn on/off the X" pattern (fallback)
    (
        re.compile(
            r"turn\s+(on|off)\s+(?:the\s+)?(.+)$",
            re.IGNORECASE,
        ),
        lambda m: ("turn_on" if m.group(1).lower() == "on" else "turn_off"),
        "device",
        lambda m: {"target_name": m.group(2).strip()},
    ),
]


class DevicePatternParser:
    """
    Pattern-based parser for device commands.

    Faster than LLM-based parsing for simple device commands.
    """

    def __init__(self, patterns: list = None):
        """
        Initialize the pattern parser.

        Args:
            patterns: List of pattern tuples (uses DEVICE_PATTERNS if None)
        """
        self._patterns = patterns or DEVICE_PATTERNS

    def parse(self, query: str) -> Optional[ParsedDeviceIntent]:
        """
        Parse a query into a device intent using patterns.

        Args:
            query: User query text

        Returns:
            ParsedDeviceIntent if matched, None otherwise
        """
        query = query.strip()
        if not query:
            return None

        for pattern, action_fn, target_type, params_fn in self._patterns:
            match = pattern.match(query)
            if match:
                params = params_fn(match)
                action = action_fn(match)
                target_name = params.pop("target_name", None)

                intent = ParsedDeviceIntent(
                    action=action,
                    target_type=target_type,
                    target_name=target_name,
                    parameters=params,
                    confidence=0.9,
                    pattern_matched=pattern.pattern,
                )

                logger.debug(
                    "Pattern matched: '%s' -> %s %s/%s",
                    query[:30],
                    action,
                    target_type,
                    target_name,
                )

                return intent

        logger.debug("No pattern matched for: '%s'", query[:30])
        return None

    def can_parse(self, query: str) -> bool:
        """
        Check if query can be parsed by patterns.

        Args:
            query: User query text

        Returns:
            True if a pattern matches
        """
        query = query.strip()
        if not query:
            return False

        for pattern, _, _, _ in self._patterns:
            if pattern.match(query):
                return True

        return False


# Singleton instance
_parser: Optional[DevicePatternParser] = None


def get_pattern_parser() -> DevicePatternParser:
    """Get or create global pattern parser."""
    global _parser
    if _parser is None:
        _parser = DevicePatternParser()
    return _parser
