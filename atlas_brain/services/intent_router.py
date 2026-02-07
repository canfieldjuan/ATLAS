"""
Intent Router for fast query classification.

Uses semantic embeddings (sentence-transformers) for fast cosine-similarity
classification with optional LLM fallback for low-confidence queries.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import settings

logger = logging.getLogger("atlas.services.intent_router")


@dataclass
class IntentRouteResult:
    """Result from intent routing."""

    action_category: str  # "device_command", "tool_use", "conversation"
    raw_label: str  # Route name (e.g., "reminder", "device_command")
    confidence: float
    route_time_ms: float = 0.0
    tool_name: Optional[str] = None  # Mapped tool name if applicable
    fast_path_ok: bool = False  # True if tool can execute without params


# Tools that can execute without parameters (fast path OK)
PARAMETERLESS_TOOLS = {
    "get_time",
    "get_weather",
    "get_calendar",
    "list_reminders",
    "get_traffic",
    "get_location",
}


# ── Route definitions: exemplar utterances per route ──

ROUTE_DEFINITIONS: dict[str, list[str]] = {
    "device_command": [
        "turn on the living room lights", "turn off the kitchen light",
        "dim the bedroom lamp to 50 percent", "switch off the TV",
        "turn on the fan", "set the thermostat to 72", "toggle the porch light",
        "turn the volume up", "mute the speakers", "play some music",
        "set the lights to blue", "turn off all the lights",
        "start the robot vacuum", "turn on the coffee maker",
    ],
    "reminder": [
        "remind me to call the dentist tomorrow", "set a reminder for 3pm",
        "set an alarm for 6 in the morning", "wake me up at 7am",
        "don't let me forget to buy groceries", "add a reminder to pick up the kids",
        "create an alarm for monday morning", "alert me at noon",
        "remember to water the plants tonight",
        "delete the reminder about groceries", "remove my alarm",
        "complete the first reminder", "mark reminder done",
        "delete the reminder", "cancel my alarm", "remove the reminder",
    ],
    "email": [
        "send an email to John about the meeting", "draft an email to the client",
        "compose an email about the project update", "email Sarah regarding the invoice",
        "write an email to the team about Friday", "send a message to the contractor",
    ],
    "calendar_write": [
        "add a meeting to my calendar for Thursday", "create a calendar event for Tuesday",
        "schedule a meeting with the team on Friday", "put a dentist appointment on my calendar",
        "create an event called team standup", "add lunch with Maria to my calendar",
    ],
    "booking": [
        "book an appointment for next Monday", "schedule an appointment with the barber",
        "I need to book an appointment", "set up an appointment for a haircut",
        "make an appointment for next week", "I want to schedule a visit",
    ],
    "get_time": [
        "what time is it", "what's the current time", "tell me the time",
        "what's the date today", "what day is it",
    ],
    "get_weather": [
        "what's the weather like", "how's the weather today", "is it going to rain",
        "what's the temperature outside", "weather forecast for today",
    ],
    "get_calendar": [
        "what's on my calendar today", "do I have any meetings today",
        "show me my schedule", "what events do I have this week",
        "am I free this afternoon", "any appointments today",
    ],
    "list_reminders": [
        "show my reminders", "what reminders do I have", "list all my alarms",
        "what are my active reminders", "do I have any reminders",
    ],
    "get_traffic": [
        "how's the traffic", "what's the traffic like to work",
        "how long is my commute", "traffic conditions to downtown",
    ],
    "get_location": [
        "where am I", "where are we", "where are we located",
        "what is my location", "what's my current location",
        "where am I right now", "my location", "GPS location",
        "what's my position", "where is this place",
    ],
    "conversation": [
        "hello", "how are you", "tell me a joke", "what is the capital of France",
        "explain quantum physics", "who wrote Romeo and Juliet",
        "thank you", "goodbye", "what is the meaning of life",
        "recommend a good movie", "what's two plus two", "how do I make pancakes",
    ],
}

# Single-hop mapping: route name → (action_category, tool_name | None)
ROUTE_TO_ACTION: dict[str, tuple[str, Optional[str]]] = {
    "device_command": ("device_command", None),
    "reminder":       ("tool_use", "set_reminder"),
    "email":          ("tool_use", "send_email"),
    "calendar_write": ("tool_use", "set_calendar_event"),
    "booking":        ("tool_use", "book_appointment"),
    "get_time":       ("tool_use", "get_time"),
    "get_weather":    ("tool_use", "get_weather"),
    "get_calendar":   ("tool_use", "get_calendar"),
    "list_reminders": ("tool_use", "list_reminders"),
    "get_traffic":    ("tool_use", "get_traffic"),
    "get_location":   ("tool_use", "get_location"),
    "conversation":   ("conversation", None),
}

# Routes that trigger multi-turn workflows
ROUTE_TO_WORKFLOW: dict[str, str] = {
    "reminder": "reminder",
    "email": "email",
    "calendar_write": "calendar",
    "booking": "booking",
}

# Valid route names for LLM fallback validation
_VALID_ROUTES = set(ROUTE_TO_ACTION.keys())


class SemanticIntentRouter:
    """
    Hybrid semantic embedding + LLM fallback intent router.

    Fast path (~5-10ms): embed query, dot-product vs route centroids.
    Slow path (~200-500ms): LLM classification when semantic confidence is low.
    """

    def __init__(self) -> None:
        self._config = settings.intent_router
        self._embedder = None
        self._route_centroids: dict[str, np.ndarray] = {}

    @property
    def is_loaded(self) -> bool:
        return len(self._route_centroids) > 0

    async def load(self) -> None:
        """Load embedding model and compute route centroids."""
        if self._route_centroids:
            logger.info("Semantic intent router already loaded")
            return

        from .embedding.sentence_transformer import SentenceTransformerEmbedding

        logger.info("Loading semantic intent router (model=%s)", self._config.embedding_model)
        start = time.time()

        self._embedder = SentenceTransformerEmbedding(
            model_name=self._config.embedding_model,
            device=self._config.embedding_device,
        )

        # Load in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._embedder.load)

        # Compute centroids for each route
        for route_name, utterances in ROUTE_DEFINITIONS.items():
            embeddings = await loop.run_in_executor(
                None, self._embedder.embed_batch, utterances,
            )
            # Centroid = mean of normalized vectors, re-normalized
            centroid = embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self._route_centroids[route_name] = centroid

        elapsed = time.time() - start
        logger.info(
            "Semantic intent router loaded in %.2fs (%d routes, dim=%d)",
            elapsed, len(self._route_centroids), self._embedder.dimension,
        )

    def unload(self) -> None:
        """Unload model and free memory."""
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None
        self._route_centroids.clear()
        logger.info("Semantic intent router unloaded")

    async def route(self, query: str) -> IntentRouteResult:
        """
        Classify a query into a route.

        1. Semantic classification (fast path)
        2. If below threshold and LLM fallback enabled, try LLM
        3. Otherwise fall back to conversation
        """
        if not self._config.enabled:
            return IntentRouteResult(
                action_category="conversation",
                raw_label="disabled",
                confidence=0.0,
            )

        if not self._route_centroids:
            await self.load()

        start = time.time()

        # Semantic classification
        route_name, similarity = await self._semantic_classify(query)

        threshold = self._config.confidence_threshold

        # If above threshold, use semantic result
        if similarity >= threshold:
            route_time = (time.time() - start) * 1000
            action_category, tool_name = ROUTE_TO_ACTION.get(
                route_name, ("conversation", None)
            )
            logger.info(
                "Route: '%s' -> %s (semantic, conf=%.2f, %.0fms)",
                query[:40], route_name, similarity, route_time,
            )
            return IntentRouteResult(
                action_category=action_category,
                raw_label=route_name,
                confidence=similarity,
                route_time_ms=route_time,
                tool_name=tool_name,
                fast_path_ok=tool_name in PARAMETERLESS_TOOLS if tool_name else False,
            )

        # LLM fallback
        if self._config.llm_fallback_enabled:
            llm_result = await self._llm_classify(query)
            if llm_result is not None:
                llm_route, llm_conf = llm_result
                route_time = (time.time() - start) * 1000
                action_category, tool_name = ROUTE_TO_ACTION.get(
                    llm_route, ("conversation", None)
                )
                logger.info(
                    "Route: '%s' -> %s (llm_fallback, conf=%.2f, %.0fms)",
                    query[:40], llm_route, llm_conf, route_time,
                )
                return IntentRouteResult(
                    action_category=action_category,
                    raw_label=llm_route,
                    confidence=llm_conf,
                    route_time_ms=route_time,
                    tool_name=tool_name,
                    fast_path_ok=tool_name in PARAMETERLESS_TOOLS if tool_name else False,
                )

        # Fall back to conversation
        route_time = (time.time() - start) * 1000
        logger.info(
            "Route: '%s' -> conversation (fallback, semantic_conf=%.2f, %.0fms)",
            query[:40], similarity, route_time,
        )
        return IntentRouteResult(
            action_category="conversation",
            raw_label="conversation",
            confidence=similarity,
            route_time_ms=route_time,
        )

    async def _semantic_classify(self, query: str) -> tuple[str, float]:
        """Embed query and find best matching route centroid."""
        loop = asyncio.get_event_loop()
        query_vec = await loop.run_in_executor(None, self._embedder.embed, query)

        best_route = "conversation"
        best_sim = -1.0

        for route_name, centroid in self._route_centroids.items():
            # Dot product of normalized vectors = cosine similarity
            sim = float(np.dot(query_vec, centroid))
            if sim > best_sim:
                best_sim = sim
                best_route = route_name

        return best_route, best_sim

    async def _llm_classify(self, query: str) -> Optional[tuple[str, float]]:
        """Use LLM to classify query when semantic confidence is low."""
        try:
            from . import llm_registry
            from .protocols import Message

            llm = llm_registry.get_active()
            if llm is None:
                return None

            route_list = ", ".join(sorted(_VALID_ROUTES))
            prompt = (
                f"Classify this user query into exactly one route.\n"
                f"Routes: {route_list}\n"
                f'User query: "{query}"\n'
                f'Respond with ONLY JSON: {{"route": "<name>", "confidence": <0.0-1.0>}}'
            )

            messages = [Message(role="user", content=prompt)]

            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: llm.chat(
                        messages=messages,
                        max_tokens=self._config.llm_fallback_max_tokens,
                        temperature=self._config.llm_fallback_temperature,
                    ),
                ),
                timeout=self._config.llm_fallback_timeout,
            )

            response_text = result.get("response", "").strip()
            # Extract JSON from response (handle possible markdown wrapping)
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            # Strip any <think> tags from reasoning models
            if "<think>" in response_text:
                think_end = response_text.rfind("</think>")
                if think_end >= 0:
                    response_text = response_text[think_end + 8:].strip()

            parsed = json.loads(response_text)
            route = parsed.get("route", "")
            confidence = float(parsed.get("confidence", 0.5))

            if route in _VALID_ROUTES:
                return route, confidence

            logger.warning("LLM returned invalid route: %s", route)
            return None

        except asyncio.TimeoutError:
            logger.warning("LLM fallback timed out (%.1fs)", self._config.llm_fallback_timeout)
            return None
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("LLM fallback parse error: %s", e)
            return None
        except Exception as e:
            logger.warning("LLM fallback failed: %s", e)
            return None


# Module-level singleton
_router: Optional[SemanticIntentRouter] = None


def get_intent_router() -> SemanticIntentRouter:
    """Get or create the global intent router instance."""
    global _router
    if _router is None:
        _router = SemanticIntentRouter()
    return _router


async def route_query(query: str) -> IntentRouteResult:
    """Convenience function to route a query."""
    router = get_intent_router()
    return await router.route(query)
