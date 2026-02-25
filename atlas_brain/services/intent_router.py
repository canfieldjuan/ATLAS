"""
Intent Router for fast query classification.

Uses semantic embeddings (sentence-transformers) for fast cosine-similarity
classification with optional LLM fallback for low-confidence queries.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
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
    entity_name: Optional[str] = None  # Extracted entity for graph traversal
    tool_params: dict = field(default_factory=dict)  # Temporal/spatial params extracted from query


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
    "run_digest",
    "get_motion_events",
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
    "email_query": [
        "what was my last email about", "check my inbox",
        "any new emails today", "read my latest email",
        "what did John say in his email", "was that email angry",
        "what's the tone of the last email", "summarize my recent emails",
        "did I get a reply from Sarah", "any emails from the contractor",
        "what emails came in today", "show me my unread emails",
        "reply to John's last email", "respond to that email",
        "let's reply to the last email", "send a reply to Sarah",
    ],
    "call_person": [
        "call John", "call the customer", "give them a call",
        "phone Sarah", "ring the client", "make a call to the plumber",
        "call back the customer who emailed", "dial the office",
        "I need to call someone", "can you call this person for me",
    ],
    "calendar_write": [
        "add a meeting to my calendar for Thursday", "create a calendar event for Tuesday",
        "schedule a meeting with the team on Friday", "put a dentist appointment on my calendar",
        "create an event called team standup", "add lunch with Maria to my calendar",
        "block out Thursday afternoon for the contractor", "mark my calendar for the 5th",
        "set up a meeting with John tomorrow at 2", "log a call with the client for Monday",
        "add my son's game to the calendar Saturday morning", "put a block on my calendar for project work",
        "create an appointment at 3pm Friday", "add an event for the team lunch next week",
    ],
    "booking": [
        "book an appointment for next Monday", "schedule an appointment with the barber",
        "I need to book an appointment", "set up an appointment for a haircut",
        "make an appointment for next week", "I want to schedule a visit",
        "can you get me in for a cleaning next Tuesday", "find me an opening this week",
        "get me in with the dentist", "put me down for an appointment Thursday",
        "can you book me in for Friday afternoon", "I need to get on the schedule",
        "book me a slot tomorrow", "get me an appointment as soon as possible",
    ],
    "cancel_booking": [
        "cancel my appointment", "I need to cancel my booking",
        "cancel the appointment for Thursday", "I want to cancel",
        "remove my appointment", "cancel the booking",
        "drop the appointment tomorrow", "I can't make my appointment",
        "clear my Thursday appointment", "call off the meeting",
        "delete the appointment on Friday", "scratch that booking",
        "I won't be able to make it to my appointment", "get rid of my appointment this week",
    ],
    "reschedule_booking": [
        "reschedule my appointment", "move my appointment to Friday",
        "change my booking to next week", "I need to reschedule",
        "can we move the appointment", "reschedule the booking",
        "push my appointment back to next week", "bump the appointment to Wednesday",
        "can you move the appointment to a later time", "shift my booking earlier",
        "postpone my appointment", "change the time of my appointment",
        "I need to push back my appointment", "move the meeting up an hour",
    ],
    "get_time": [
        "what time is it", "what's the current time right now",
        "tell me the time right now", "what's today's date",
        "what day of the week is it",
        "what's the time", "do you have the time",
        "how late is it", "what month are we in",
        "check the time for me", "what time is it currently",
        "what's the current date", "is it morning or afternoon",
        "what year is it", "tell me today's date",
    ],
    "get_weather": [
        "what's the weather like", "how's the weather today", "is it going to rain",
        "what's the temperature outside", "weather forecast for today",
        "what's the weather tomorrow", "what will the weather be like tomorrow",
        "will it rain tomorrow", "forecast for tomorrow",
        "what's the weather this weekend", "what will the weather be like on Friday",
        "is it going to snow next week", "what's the temperature tomorrow",
        "give me the weather forecast", "what's the forecast for the next few days",
    ],
    "get_calendar": [
        "what's on my calendar today", "do I have any meetings today",
        "show me my schedule", "what events do I have this week",
        "am I free this afternoon", "any appointments today",
        "what do I have coming up", "what's my day look like",
        "am I busy tomorrow", "do I have anything scheduled this week",
        "what are my plans for Friday", "check my calendar",
        "what meetings do I have tomorrow", "do I have anything going on tonight",
        "what's on the schedule for this week",
    ],
    "list_reminders": [
        "show my reminders", "what reminders do I have", "list all my alarms",
        "what are my active reminders", "do I have any reminders",
        "read me my reminders", "check my reminders",
        "any upcoming reminders", "what alarms do I have set",
        "are there any reminders pending", "what's on my reminder list",
        "how many reminders do I have", "tell me what reminders are set",
        "show me all pending reminders",
    ],
    "get_traffic": [
        "how's the traffic", "what's the traffic like to work",
        "how long is my commute", "traffic conditions to downtown",
        "is the highway clear", "how bad is traffic right now",
        "how long will it take to get home", "is there a lot of traffic",
        "how's the drive to work today", "any accidents on the road",
        "should I leave now to beat traffic", "traffic report",
        "is it a good time to drive", "how long to get downtown from here",
        "what's the drive time looking like",
    ],
    "get_location": [
        "what is my GPS location", "what are my coordinates",
        "what city am I in", "what is my address",
        "track my phone location", "find my phone",
        "what is my geo location", "show my location on a map",
        "where am I located right now", "give me my current location",
        "what neighborhood am I in", "can you find my location",
        "ping my location", "what's my exact location",
    ],
    "where_am_i": [
        "where am I", "what room am I in", "which room is this",
        "what room am I in right now", "what space am I in",
        "which room does the system think I am in",
        "detect my room",
        "what part of the house am I in", "what area am I in",
        "where does atlas think I am", "am I in the bedroom",
        "tell me what room I'm in", "what zone am I in",
        "can you detect where I am in the house", "identify my current room",
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
        "ping my phone", "flash a notification saying the meeting starts soon",
        "buzz my phone with a reminder", "pop up a notification about dinner",
        "shoot me a notification", "send a heads up to my phone",
        "alert me on my phone that the package arrived", "notify me right now",
    ],
    "show_camera": [
        "show me the camera feed on the left monitor",
        "pull up the camera feed on screen",
        "display the front door camera on the right monitor",
        "show the office webcam on the left display",
        "put the backyard camera on screen",
        "close the camera viewer", "hide the camera viewer window",
        "close all camera viewer windows",
        "bring up the front door camera", "let me see the driveway camera",
        "switch to the backyard camera", "open the camera view",
        "show me what the front camera sees", "pull up the live feed",
        "display the security camera on screen", "view the garage camera",
    ],
    "security": [
        "list my cameras", "show me all the cameras",
        "how many cameras do I have", "which cameras are online",
        "check the front door camera status", "is the driveway camera online",
        "start recording on the driveway camera",
        "stop recording on the backyard camera",
        "what are the cameras seeing right now",
        "show me the security zones", "list all security zones",
        "arm the home security", "disarm the home security",
        "activate the security system", "deactivate the security system",
    ],
    "crm_query": [
        "look up John", "is John in the CRM", "find customer Smith",
        "search for a contact named Mary", "look up the customer",
        "do we have a customer named Garcia", "find the contact",
        "who is John Smith in our system", "search contacts for Sarah",
        "check if we have a record for them", "look up that customer",
        "pull up the customer file", "find the client",
    ],
    "detection_query": [
        "who was at the front door", "who is at the back door",
        "is anyone outside", "what did the cameras see", "who came by today",
        "was there anyone at the driveway", "any people detected recently",
        "check the front door", "is anybody at the front door camera",
        "check if someone is on the security camera",
    ],
    "motion_query": [
        "any motion on the cameras", "check for motion activity",
        "was there any motion", "any movement outside",
        "motion events today", "any motion at the back door",
        "has there been any motion", "what motion was detected",
        "any activity on the cameras", "motion history",
    ],
    "digest": [
        "give me my morning briefing", "morning briefing",
        "daily summary", "security summary", "email digest",
        "how are my devices", "device status", "catch me up",
        "what did I miss", "give me a rundown",
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
    "email_query":    ("tool_use", "search_inbox"),
    "call_person":    ("tool_use", "make_call"),
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
    "security":         ("device_command", None),
    "presence":         ("device_command", None),
    "crm_query":        ("tool_use", "search_contacts"),
    "detection_query":  ("tool_use", "get_person_at_location"),
    "motion_query":     ("tool_use", "get_motion_events"),
    "digest":           ("tool_use", "run_digest"),
    "conversation":     ("conversation", None),
}

# Routes that trigger multi-turn workflows
ROUTE_TO_WORKFLOW: dict[str, str] = {
    "reminder": "reminder",
    "email": "email",
    "email_query": "email_query",
    "calendar_write": "calendar",
    "booking": "booking",
    "cancel_booking": "booking",
    "reschedule_booking": "booking",
    "security": "security",
    "presence": "presence",
    "call_person": "call",
}

# Valid route names for LLM fallback validation
_VALID_ROUTES = set(ROUTE_TO_ACTION.keys()) | set(ROUTE_TO_WORKFLOW.keys())

# Routes where we should attempt lightweight entity extraction on the
# semantic path (no LLM call — just strip known verb/preposition prefixes).
_ENTITY_ROUTES = {
    "crm_query", "call_person", "email", "email_query", "detection_query",
}
_ENTITY_STRIP_RE = re.compile(
    r"^(?:look\s*up|find|search\s*(?:for)?|check|pull\s*up|call|phone|ring"
    r"|email|send\s*(?:an?\s*)?(?:email|message)\s*to|reply\s*to"
    r"|who\s*(?:was|is)|is\s*there|any)\s+",
    re.IGNORECASE,
)
_ENTITY_SUFFIX_RE = re.compile(
    r"\s+(?:in\s+(?:the\s+)?(?:crm|system|database|contacts?|our\s+records?)"
    r"|on\s+(?:the\s+)?(?:camera|phone|door)"
    r"|(?:at|from)\s+(?:the\s+)?(?:front|back|side)\s*(?:door|gate)?)\.?\s*$",
    re.IGNORECASE,
)


def _extract_entity_from_query(query: str, route: str) -> Optional[str]:
    """Best-effort entity extraction without LLM — strip known prefixes/suffixes."""
    if route not in _ENTITY_ROUTES:
        return None
    text = _ENTITY_STRIP_RE.sub("", query).strip()
    text = _ENTITY_SUFFIX_RE.sub("", text).strip().rstrip("?.!")
    # Only return if we actually extracted something meaningful
    if text and text.lower() != query.lower() and len(text) < 60:
        return text
    return None


def _word_to_num(word: str) -> Optional[int]:
    """Convert a word or digit string to an integer, or return None."""
    _MAP = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    if word.isdigit():
        return int(word)
    return _MAP.get(word.lower())


def _extract_digest_params(query: str) -> dict:
    """Infer digest_type from keywords in the user query."""
    q = query.lower()
    if "security" in q:
        return {"digest_type": "security_summary"}
    if "device" in q or "health" in q:
        return {"digest_type": "device_health"}
    if "email" in q or "gmail" in q or "inbox" in q:
        return {"digest_type": "email_digest"}
    # Default: morning briefing covers the general case
    return {"digest_type": "morning_briefing"}


def _extract_detection_params(query: str) -> dict:
    """Infer location from keywords in a detection query."""
    q = query.lower()
    _LOCATIONS = [
        "front door", "back door", "side door", "garage door",
        "driveway", "backyard", "back yard", "front yard",
        "garage", "porch", "patio", "deck",
        "living room", "kitchen", "bedroom", "office",
        "hallway", "basement", "attic",
    ]
    for loc in _LOCATIONS:
        if loc in q:
            return {"location": loc}
    if "outside" in q:
        return {"location": "outside"}
    return {}


def _extract_entity_tool_params(query: str, route_name: str) -> dict:
    """Map extracted entity to tool params for entity-bearing routes.

    Makes routes like crm_query fast-path eligible by populating the
    tool's required params from the entity name.
    """
    entity = _extract_entity_from_query(query, route_name)
    if not entity:
        return {}
    if route_name == "crm_query":
        return {"query": entity}
    if route_name == "email_query":
        return {"query": entity}
    return {}


def _extract_temporal_params(query: str, route_name: str) -> dict:
    """Extract route-specific params from the query text.

    Weather/traffic: days_ahead for forecast queries.
    Digest: digest_type keyword inference (security, device, email).
    Detection: location keyword extraction.
    CRM/email queries: entity name to search query.
    """
    if route_name == "digest":
        return _extract_digest_params(query)
    if route_name == "detection_query":
        return _extract_detection_params(query)
    if route_name in ("crm_query", "email_query"):
        return _extract_entity_tool_params(query, route_name)
    if route_name not in ("get_weather", "get_traffic"):
        return {}
    q = query.lower()

    # "day after tomorrow" before "tomorrow" check
    if "day after tomorrow" in q:
        return {"days_ahead": 2}

    if "tomorrow" in q:
        return {"days_ahead": 1}

    # "in N days"
    m = re.search(r"in\s+(\w+)\s+days?", q)
    if m:
        num = _word_to_num(m.group(1))
        if num is not None:
            return {"days_ahead": num}

    if "next week" in q:
        return {"days_ahead": 7}

    if "weekend" in q:
        today = datetime.now(timezone.utc).weekday()  # Mon=0, Sun=6
        days_to_sat = (5 - today) % 7 or 7
        return {"days_ahead": days_to_sat}

    _DAY_NUMS = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for day_name, day_num in _DAY_NUMS.items():
        if day_name in q:
            today = datetime.now(timezone.utc).weekday()
            days_ahead = (day_num - today) % 7 or 7
            return {"days_ahead": days_ahead}

    return {}


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

        # Batch all route phrases into a single embed call (1 progress bar, not 26)
        all_phrases: list[str] = []
        route_slices: list[tuple[str, int, int]] = []
        for route_name, utterances in ROUTE_DEFINITIONS.items():
            start_idx = len(all_phrases)
            all_phrases.extend(utterances)
            route_slices.append((route_name, start_idx, len(all_phrases)))

        all_embeddings = await loop.run_in_executor(
            None, self._embedder.embed_batch, all_phrases,
        )
        for route_name, s, e in route_slices:
            centroid = all_embeddings[s:e].mean(axis=0)
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

        # Batch all route phrases into a single embed call (1 progress bar, not 26)
        all_phrases: list[str] = []
        route_slices: list[tuple[str, int, int]] = []
        for route_name, utterances in ROUTE_DEFINITIONS.items():
            start_idx = len(all_phrases)
            all_phrases.extend(utterances)
            route_slices.append((route_name, start_idx, len(all_phrases)))

        all_embeddings = self._embedder.embed_batch(all_phrases)
        for route_name, s, e in route_slices:
            centroid = all_embeddings[s:e].mean(axis=0)
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

        # If above threshold, use semantic result.
        if similarity >= threshold:
            route_time = (time.time() - start) * 1000
            action_category, tool_name = ROUTE_TO_ACTION.get(
                route_name, ("conversation", None)
            )
            logger.info(
                "Route: '%s' -> %s (semantic, conf=%.2f, %.0fms)",
                query[:40], route_name, similarity, route_time,
            )
            tool_params = _extract_temporal_params(query, route_name)
            fast_ok = (tool_name in PARAMETERLESS_TOOLS if tool_name else False) or bool(tool_params)
            entity = _extract_entity_from_query(query, route_name)
            return IntentRouteResult(
                action_category=action_category,
                raw_label=route_name,
                confidence=similarity,
                route_time_ms=route_time,
                tool_name=tool_name,
                fast_path_ok=fast_ok,
                tool_params=tool_params,
                entity_name=entity,
            )

        # LLM fallback -- skip for very short queries (1-2 words) where the
        # LLM won't have enough context to do better than conversation default,
        # and the 2s timeout wastes latency competing with prefill on the GPU.
        word_count = len(query.split())
        if self._config.llm_fallback_enabled and word_count >= 3:
            llm_result = await self._llm_classify(query)
            if llm_result is not None:
                llm_route, llm_conf, llm_entity = llm_result
                route_time = (time.time() - start) * 1000
                action_category, tool_name = ROUTE_TO_ACTION.get(
                    llm_route, ("conversation", None)
                )
                logger.info(
                    "Route: '%s' -> %s (llm_fallback, conf=%.2f, %.0fms, entity=%s)",
                    query[:40], llm_route, llm_conf, route_time, llm_entity,
                )
                self._log_fallback(
                    query, route_name, similarity,
                    llm_route, llm_conf, llm_route, route_time,
                )
                tool_params = _extract_temporal_params(query, llm_route)
                fast_ok = (tool_name in PARAMETERLESS_TOOLS if tool_name else False) or bool(tool_params)
                return IntentRouteResult(
                    action_category=action_category,
                    raw_label=llm_route,
                    confidence=llm_conf,
                    route_time_ms=route_time,
                    tool_name=tool_name,
                    fast_path_ok=fast_ok,
                    entity_name=llm_entity,
                    tool_params=tool_params,
                )
            else:
                # LLM fallback failed (timeout/parse error) -- log anyway
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
            # No dedicated model configured -- use the main LLM
            from . import llm_registry
            return llm_registry.get_active()

        from .llm.ollama import OllamaLLM

        base_url = settings.llm.ollama_url
        logger.info("Initializing dedicated fallback classifier: %s @ %s", model_name, base_url)
        self._fallback_llm = OllamaLLM(model=model_name, base_url=base_url)
        self._fallback_llm.load()
        return self._fallback_llm

    async def _llm_classify(self, query: str) -> Optional[tuple[str, float, Optional[str]]]:
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
                "- email_query: check inbox, read emails, ask about email content or tone, reply to received email\n"
                "- call_person: call or phone a person\n"
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
                "- crm_query: look up a customer or contact by name in the database, check if someone is in the CRM, find a contact's phone or email\n"
                "- detection_query: ask who was physically seen by a security camera or at a door, check camera footage for people\n"
                "- motion_query: ask about motion or movement on cameras\n"
                "- digest: request a briefing, summary, or status report\n"
                "- conversation: general chat, personal questions, opinions, "
                "knowledge recall, or anything not matching above\n"
                f'User query: "{query}"\n'
                'Respond with ONLY a single JSON object, nothing else:\n'
                '{"route": "<name>", "confidence": <0.0-1.0>, '
                '"entity": "<main subject name or null>"}\n'
                'Do NOT add explanations, examples, or markdown fencing.'
            )

            messages = [
                Message(role="system", content="You are a JSON classifier. Output raw JSON only. Never add commentary."),
                Message(role="user", content=prompt),
            ]

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

            # Extract first JSON object (some models add explanation after)
            brace_start = response_text.find("{")
            if brace_start >= 0:
                brace_end = response_text.find("}", brace_start)
                if brace_end >= 0:
                    response_text = response_text[brace_start:brace_end + 1]

            parsed = json.loads(response_text)
            route = parsed.get("route", "")
            confidence = float(parsed.get("confidence", 0.5))
            entity = parsed.get("entity") or None
            if isinstance(entity, str):
                entity = entity.strip() if entity.strip() else None

            if route in _VALID_ROUTES:
                return route, confidence, entity

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
