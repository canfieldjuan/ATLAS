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
from datetime import datetime, timezone
from pathlib import Path
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
    "where_am_i",
    "who_is_here",
}


# -- Route definitions: exemplar utterances per route --

ROUTE_DEFINITIONS: dict[str, list[str]] = {
    "device_command": [
        "turn on the living room lights", "turn off the kitchen light",
        "dim the bedroom lamp to 50 percent", "switch off the TV",
        "turn on the fan", "set the thermostat to 72", "toggle the porch light",
        "turn the volume up", "mute the speakers", "play some music",
        "set the lights to blue", "turn off all the lights",
        "start the robot vacuum", "turn on the coffee maker",
        "turn on the camera", "turn off the camera",
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
        "send an estimate email to Sarah Johnson", "email the proposal to the client",
        "send the cleaning estimate to the customer", "what emails did I send today",
        "show me my email history", "check what emails were sent this week",
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
    "cancel_booking": [
        "cancel my appointment", "I need to cancel my booking",
        "cancel the appointment for Thursday", "I want to cancel",
        "remove my appointment", "cancel the booking",
    ],
    "reschedule_booking": [
        "reschedule my appointment", "move my appointment to Friday",
        "change my booking to next week", "I need to reschedule",
        "can we move the appointment", "reschedule the booking",
    ],
    "get_time": [
        "what time is it", "what's the current time right now",
        "tell me the time right now", "what's today's date",
        "what day of the week is it",
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
        "what is my GPS location", "what are my coordinates",
        "what city am I in", "what is my address",
        "track my phone location", "find my phone",
        "what is my geo location", "show my location on a map",
    ],
    "where_am_i": [
        "where am I", "what room am I in", "which room is this",
        "what room am I in right now", "what space am I in",
        "which room does the system think I am in",
        "detect my room",
    ],
    "who_is_here": [
        "who is here", "who is in the office", "is anyone home",
        "is anyone here", "who is home", "who is in the building",
        "how many people are here", "is the office occupied",
        "is somebody here", "who is at home",
        "who has been here today", "how long has someone been here",
    ],
    "notification": [
        "send me a notification", "send a push notification to my phone",
        "send a notification saying the laundry is done",
        "push a notification to my phone about the groceries",
        "notify my phone that dinner is ready",
        "send an alert to my phone saying I need to leave",
    ],
    "show_camera": [
        "show me the camera feed on the left monitor",
        "pull up the camera feed on screen",
        "display the front door camera on the right monitor",
        "show the office webcam on the left display",
        "put the backyard camera on screen",
        "close the camera viewer", "hide the camera viewer window",
        "close all camera viewer windows",
    ],
    "security": [
        "list my cameras", "show me all the cameras",
        "how many cameras do I have", "which cameras are online",
        "check the front door camera status", "is the driveway camera online",
        "start recording on the driveway camera",
        "stop recording on the backyard camera",
        "what are the cameras seeing right now",
        "is anybody at the front door camera",
        "check if someone is on the security camera",
        "any motion on the cameras", "check for motion activity",
        "show me the security zones", "list all security zones",
        "arm the home security", "disarm the home security",
        "activate the security system", "deactivate the security system",
    ],
    "presence": [
        "set it to movie mode", "movie time", "cinema mode",
        "make it cozy in here", "set a cozy atmosphere",
        "switch to relax mode", "relax mode",
        "bright mode", "dim mode", "focus mode",
        "set the scene to focus", "set the mood",
        "dim the lights in here", "brighten it up in here",
        "set the room to movie lighting",
        "turn off everything near me",
    ],
    "conversation": [
        "hello", "hey there", "how are you", "how's it going",
        "tell me a joke", "what is the capital of France",
        "explain quantum physics", "who wrote Romeo and Juliet",
        "thank you", "thanks I appreciate it", "goodbye", "see you later",
        "what is the meaning of life", "recommend a good movie",
        "what's two plus two", "how do I make pancakes",
        "sure", "okay", "alright", "sounds good", "got it",
        "oh my god that's crazy", "no way are you serious",
        "yeah I was just talking about that", "I don't know what happened",
        "that's hilarious", "you won't believe what happened today",
        "hey atlas", "hey", "good morning", "good night",
        "never mind", "forget about it", "what do you think",
        "what time do I usually wake up", "when do we normally get home",
        "what's my typical morning routine", "when does she usually leave",
        "what time do the kids usually go to bed",
        "what are my usual habits", "do I have a routine",
        "what time do we usually eat dinner",
        "when do I normally go to sleep", "what's my schedule like",
        "what time does he usually get here",
        "when do they typically arrive",
    ],
}

# Single-hop mapping: route name -> (action_category, tool_name | None)
ROUTE_TO_ACTION: dict[str, tuple[str, Optional[str]]] = {
    "device_command": ("device_command", None),
    "reminder":       ("tool_use", "set_reminder"),
    "email":          ("tool_use", "send_email"),
    "calendar_write": ("tool_use", "create_calendar_event"),
    "booking":            ("tool_use", "book_appointment"),
    "cancel_booking":     ("tool_use", "cancel_appointment"),
    "reschedule_booking": ("tool_use", "reschedule_appointment"),
    "get_time":       ("tool_use", "get_time"),
    "get_weather":    ("tool_use", "get_weather"),
    "get_calendar":   ("tool_use", "get_calendar"),
    "list_reminders": ("tool_use", "list_reminders"),
    "get_traffic":    ("tool_use", "get_traffic"),
    "get_location":   ("tool_use", "get_location"),
    "where_am_i":    ("tool_use", "where_am_i"),
    "who_is_here":   ("tool_use", "who_is_here"),
    "notification":  ("tool_use", "send_notification"),
    "show_camera":   ("tool_use", "show_camera_feed"),
    "security":      ("device_command", None),
    "presence":      ("device_command", None),
    "conversation":   ("conversation", None),
}

# Routes that trigger multi-turn workflows
ROUTE_TO_WORKFLOW: dict[str, str] = {
    "reminder": "reminder",
    "email": "email",
    "calendar_write": "calendar",
    "booking": "booking",
    "cancel_booking": "booking",
    "reschedule_booking": "booking",
    "security": "security",
    "presence": "presence",
}

# Valid route names for LLM fallback validation
_VALID_ROUTES = set(ROUTE_TO_ACTION.keys()) | set(ROUTE_TO_WORKFLOW.keys())


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
        self._fallback_llm = None  # Dedicated lightweight model for classification
        self._fallback_log_path: Optional[Path] = None
        if self._config.llm_fallback_log:
            self._fallback_log_path = Path(self._config.llm_fallback_log)
            self._fallback_log_path.parent.mkdir(parents=True, exist_ok=True)

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

    def load_sync(self) -> None:
        """Load embedding model and compute route centroids synchronously.

        Use this when calling from a context that blocks the event loop
        (e.g., synchronous startup code within async lifespan).
        """
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
        self._embedder.load()

        for route_name, utterances in ROUTE_DEFINITIONS.items():
            embeddings = self._embedder.embed_batch(utterances)
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
        if self._fallback_llm is not None:
            self._fallback_llm.unload()
            self._fallback_llm = None
        self._route_centroids.clear()
        logger.info("Semantic intent router unloaded")

    def get_embedder(self):
        """Get the loaded embedding model for shared use by other components."""
        return self._embedder

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

        # Guard: habitual/routine queries that mention time should go to
        # conversation, not get_time, even if the embedding model scores
        # get_time higher (the phrase "what time" has strong affinity).
        if route_name == "get_time" and similarity < 0.70:
            _q = query.lower()
            _habit_words = ("usually", "normally", "typically", "routine",
                            "habit", "pattern", "schedule")
            if any(w in _q for w in _habit_words):
                route_time = (time.time() - start) * 1000
                logger.info(
                    "Route: '%s' -> conversation (habit guard, was %s conf=%.2f, %.0fms)",
                    query[:40], route_name, similarity, route_time,
                )
                return IntentRouteResult(
                    action_category="conversation",
                    raw_label="conversation",
                    confidence=similarity,
                    route_time_ms=route_time,
                )

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

        # LLM fallback — skip for very short queries (1-2 words) where the
        # LLM won't have enough context to do better than conversation default,
        # and the 2s timeout wastes latency competing with prefill on the GPU.
        word_count = len(query.split())
        if self._config.llm_fallback_enabled and word_count >= 3:
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
                self._log_fallback(
                    query, route_name, similarity,
                    llm_route, llm_conf, llm_route, route_time,
                )
                return IntentRouteResult(
                    action_category=action_category,
                    raw_label=llm_route,
                    confidence=llm_conf,
                    route_time_ms=route_time,
                    tool_name=tool_name,
                    fast_path_ok=tool_name in PARAMETERLESS_TOOLS if tool_name else False,
                )
            else:
                # LLM fallback failed (timeout/parse error) — log anyway
                route_time = (time.time() - start) * 1000
                self._log_fallback(
                    query, route_name, similarity,
                    None, None, "conversation", route_time,
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

    def _log_fallback(
        self,
        query: str,
        semantic_route: str,
        semantic_conf: float,
        llm_route: Optional[str],
        llm_conf: Optional[float],
        final_route: str,
        route_time_ms: float,
    ) -> None:
        """Append a JSONL entry for queries that triggered LLM fallback."""
        if not self._fallback_log_path:
            return
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "semantic_route": semantic_route,
                "semantic_conf": round(semantic_conf, 4),
                "llm_route": llm_route,
                "llm_conf": round(llm_conf, 4) if llm_conf is not None else None,
                "final_route": final_route,
                "route_time_ms": round(route_time_ms, 1),
            }
            with open(self._fallback_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.debug("Failed to write fallback log entry", exc_info=True)

    def _get_fallback_llm(self):
        """Get or create the dedicated lightweight LLM for classification."""
        if self._fallback_llm is not None:
            return self._fallback_llm

        model_name = self._config.llm_fallback_model
        if not model_name:
            # No dedicated model configured — use the main LLM
            from . import llm_registry
            return llm_registry.get_active()

        from .llm.ollama import OllamaLLM

        base_url = settings.llm.ollama_url
        logger.info("Initializing dedicated fallback classifier: %s @ %s", model_name, base_url)
        self._fallback_llm = OllamaLLM(model=model_name, base_url=base_url)
        self._fallback_llm.load()
        return self._fallback_llm

    async def _llm_classify(self, query: str) -> Optional[tuple[str, float]]:
        """Use LLM to classify query when semantic confidence is low."""
        try:
            from .protocols import Message

            llm = self._get_fallback_llm()
            if llm is None:
                return None

            prompt = (
                "Classify this user query into exactly one route.\n"
                "Routes:\n"
                "- device_command: control a physical device (lights, TV, thermostat)\n"
                "- reminder: set a reminder or alarm\n"
                "- email: send an email\n"
                "- calendar_write: create a calendar event\n"
                "- booking/cancel_booking/reschedule_booking: manage appointments\n"
                "- get_time: ask for current time\n"
                "- get_weather: ask for current weather\n"
                "- get_calendar: ask what is on the calendar\n"
                "- list_reminders: ask what reminders are set\n"
                "- get_traffic: ask about traffic conditions\n"
                "- get_location/where_am_i: ask what room or location the system detects\n"
                "- who_is_here: ask who is physically present in the room right now\n"
                "- notification: send a push notification\n"
                "- show_camera: show a camera feed\n"
                "- security/presence: arm security or check presence sensors\n"
                "- conversation: general chat, personal questions, opinions, "
                "knowledge recall, or anything not matching above\n"
                f'User query: "{query}"\n'
                'Respond with ONLY JSON: {"route": "<name>", "confidence": <0.0-1.0>}'
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
