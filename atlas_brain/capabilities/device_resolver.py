"""
Embedding-based device resolver for fast device command parsing.

Uses the same all-MiniLM-L6-v2 model loaded by SemanticIntentRouter to match
natural language queries against HA-discovered device names in ~20ms,
bypassing the LLM intent parser for simple on/off/toggle commands.

LLM remains as fallback for complex queries, ambiguous matches, and
pronoun resolution.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .actions import Intent

logger = logging.getLogger("atlas.capabilities.device_resolver")


@dataclass
class DeviceResolveResult:
    """Result from embedding-based device resolution."""

    intent: Intent
    device_name: str
    confidence: float
    resolve_time_ms: float


# Action phrases checked via `phrase in query_lower`.
# Longest phrases first for correct matching within each group.
ACTION_PHRASES: list[tuple[list[str], str, Optional[str]]] = [
    (["turn on", "switch on", "power on"], "turn_on", None),
    (["turn off", "switch off", "power off"], "turn_off", None),
    (["toggle"], "toggle", None),
    (["set brightness to", "dim to", "brightness to", "dim", "brightness"], "set_brightness", "brightness"),
    (["unmute"], "unmute", None),
    (["mute"], "mute", None),
    (["volume up", "louder"], "volume_up", None),
    (["volume down", "quieter", "lower the volume"], "volume_down", None),
    (["set volume to", "volume to", "volume"], "set_volume", "volume"),
    (["play"], "play", None),
    (["pause"], "pause", None),
    (["stop"], "stop", None),
]

# Pronouns that require entity tracking (LLM fallback)
_PRONOUN_PATTERN = re.compile(
    r"\b(it|them|that|this|those|these|the one|the ones)\b", re.IGNORECASE
)


def _has_pronoun(query: str) -> bool:
    """Check if query contains pronouns that need entity tracking."""
    return bool(_PRONOUN_PATTERN.search(query))


# Matches ASR-clipped "off the TV", "on the lights", "off kitchen light"
# Requires "on/off" at start + at least one more word (not a stop word)
_BARE_ON_OFF = re.compile(r"^(on|off)\s+(.+)", re.IGNORECASE)
_BARE_ON_OFF_STOP = {"my way", "the way", "a roll", "a break", "top of", "board", "hold", "fire"}


def _extract_action(query: str) -> Optional[tuple[str, Optional[str]]]:
    """
    Extract action from query using keyword matching.

    Returns (action_name, param_key) or None if no action found.
    """
    q = query.lower()
    for phrases, action, param_key in ACTION_PHRASES:
        for phrase in phrases:
            if phrase in q:
                return action, param_key

    # Handle ASR-clipped commands: "off the TV" / "on the lights"
    m = _BARE_ON_OFF.match(q)
    if m:
        rest = m.group(2).strip().lower()
        if rest not in _BARE_ON_OFF_STOP and not any(rest.startswith(s) for s in _BARE_ON_OFF_STOP):
            return ("turn_on" if m.group(1).lower() == "on" else "turn_off"), None

    return None


def _extract_number(query: str) -> Optional[int]:
    """Extract a number from the query for brightness/volume parameters."""
    match = re.search(r"\b(\d{1,3})\s*%?\b", query)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return val
    return None


def _generate_aliases(name: str) -> list[str]:
    """
    Generate shorter alias variants of a device name.

    "32 Philips Roku TV" -> ["32 Philips Roku TV", "Philips Roku TV", "Roku TV", "TV"]
    "Kitchen Light" -> ["Kitchen Light", "Kitchen"]
    """
    aliases = [name]
    words = name.split()
    if len(words) <= 1:
        return aliases

    # Progressive trimming from the left
    for i in range(1, len(words)):
        alias = " ".join(words[i:])
        if len(alias) >= 2:
            aliases.append(alias)

    return aliases


class DeviceResolver:
    """
    Fast device resolver using embedding similarity.

    Shares the embedding model with SemanticIntentRouter to avoid
    loading a second copy. Builds a centroid index of device names
    on first use and invalidates when HA devices change.
    """

    def __init__(self) -> None:
        self._device_centroids: dict[str, np.ndarray] = {}
        # Maps centroid key -> capability object
        self._device_caps: dict[str, object] = {}
        self._index_built = False
        self._build_lock = asyncio.Lock()

    def _get_embedder(self):
        """Get the shared embedding model from the intent router."""
        try:
            from ..services.intent_router import get_intent_router
            router = get_intent_router()
            return router.get_embedder()
        except Exception:
            return None

    async def _build_index(self) -> None:
        """Build embedding centroids for all registered devices."""
        async with self._build_lock:
            # Double-check after acquiring lock
            if self._index_built:
                return

            embedder = self._get_embedder()
            if embedder is None or not embedder.is_loaded:
                logger.debug("Embedder not loaded, skipping device index build")
                return

            from .registry import capability_registry

            caps = capability_registry.list_all()
            if not caps:
                logger.debug("No devices registered, skipping index build")
                return

            start = time.time()
            new_centroids: dict[str, np.ndarray] = {}
            new_caps: dict[str, object] = {}

            loop = asyncio.get_running_loop()

            for cap in caps:
                aliases = _generate_aliases(cap.name)
                embeddings = await loop.run_in_executor(
                    None, embedder.embed_batch, aliases,
                )
                # Centroid = mean of normalized alias embeddings, re-normalized
                centroid = embeddings.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm

                new_centroids[cap.id] = centroid
                new_caps[cap.id] = cap

            # Atomic swap — readers see old or new, never partial
            self._device_centroids = new_centroids
            self._device_caps = new_caps
            self._index_built = True

            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                "Device resolver index built: %d devices in %.0fms",
                len(self._device_centroids), elapsed_ms,
            )

    def invalidate(self) -> None:
        """Clear index so it rebuilds on next resolve call."""
        self._device_centroids.clear()
        self._device_caps.clear()
        self._index_built = False
        logger.debug("Device resolver index invalidated")

    async def resolve(self, query: str) -> Optional[DeviceResolveResult]:
        """
        Resolve a device command query to an Intent.

        Returns DeviceResolveResult if a confident match is found,
        None to fall through to LLM.
        """
        from ..config import settings
        config = settings.device_resolver

        if not config.enabled:
            return None

        start = time.time()

        # 1. Extract action -- if no action keyword, skip embedding entirely
        action_result = _extract_action(query)
        if action_result is None:
            return None

        action, param_key = action_result

        # 2. Check for pronouns -- LLM + entity tracker handles these
        if _has_pronoun(query):
            return None

        # 3. Get embedder
        embedder = self._get_embedder()
        if embedder is None or not embedder.is_loaded:
            return None

        # 4. Build index lazily
        if not self._index_built:
            await self._build_index()

        if not self._device_centroids:
            return None

        # 5. Embed query and find best match
        loop = asyncio.get_running_loop()
        query_vec = await loop.run_in_executor(None, embedder.embed, query)

        scores = []
        for cap_id, centroid in self._device_centroids.items():
            sim = float(np.dot(query_vec, centroid))
            scores.append((cap_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            return None

        best_id, best_score = scores[0]

        # 6. Check confidence threshold
        if best_score < config.confidence_threshold:
            elapsed = (time.time() - start) * 1000
            logger.debug(
                "Device resolve below threshold: best=%.3f < %.3f (%.0fms)",
                best_score, config.confidence_threshold, elapsed,
            )
            return None

        # 7. Check ambiguity gap (if 2+ devices)
        if len(scores) >= 2:
            second_score = scores[1][1]
            gap = best_score - second_score
            if gap < config.ambiguity_gap:
                elapsed = (time.time() - start) * 1000
                logger.debug(
                    "Device resolve ambiguous: top=%.3f, second=%.3f, gap=%.3f < %.3f (%.0fms)",
                    best_score, second_score, gap, config.ambiguity_gap, elapsed,
                )
                return None

        # 8. Validate action is supported by this device
        cap = self._device_caps[best_id]
        if action not in cap.supported_actions:
            logger.debug(
                "Action '%s' not supported by %s (supported: %s)",
                action, cap.name, cap.supported_actions,
            )
            return None

        # 9. Extract parameters if needed
        parameters = {}
        if param_key:
            number = _extract_number(query)
            if number is not None:
                parameters[param_key] = number

        elapsed = (time.time() - start) * 1000

        intent = Intent(
            action=action,
            target_type=cap.capability_type.value if hasattr(cap.capability_type, "value") else str(cap.capability_type),
            target_name=cap.name,
            target_id=cap.id,
            parameters=parameters,
            confidence=best_score,
            raw_query=query,
        )

        return DeviceResolveResult(
            intent=intent,
            device_name=cap.name,
            confidence=best_score,
            resolve_time_ms=elapsed,
        )


# Module-level singleton
_resolver: Optional[DeviceResolver] = None


def get_device_resolver() -> DeviceResolver:
    """Get or create the global device resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = DeviceResolver()
    return _resolver
