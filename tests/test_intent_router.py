"""Tests for SemanticIntentRouter.

Covers semantic classification with the real embedding model,
habit guard logic, LLM fallback behavior, disabled router,
PARAMETERLESS_TOOLS fast-path, and route mapping consistency.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.services.intent_router import (
    IntentRouteResult,
    PARAMETERLESS_TOOLS,
    ROUTE_DEFINITIONS,
    ROUTE_TO_ACTION,
    ROUTE_TO_WORKFLOW,
    SemanticIntentRouter,
    WORKFLOW_ONLY_TOOLS,
    _VALID_ROUTES,
)


# ---------------------------------------------------------------------------
# Module-scoped fixture: load the real model once (~2s) and reuse
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_router():
    """Load the real SemanticIntentRouter with all-MiniLM-L6-v2."""
    router = SemanticIntentRouter()
    router.load_sync()
    yield router
    router.unload()


# ---------------------------------------------------------------------------
# Semantic classification (real model)
# ---------------------------------------------------------------------------


class TestSemanticClassification:
    """Tests that use the real embedding model for classification."""

    def test_router_loaded_has_centroids_for_all_routes(self, loaded_router):
        """All ROUTE_DEFINITIONS should have a computed centroid."""
        assert loaded_router.is_loaded
        for route_name in ROUTE_DEFINITIONS:
            assert route_name in loaded_router._route_centroids, (
                f"Missing centroid for route: {route_name}"
            )

    @pytest.mark.asyncio
    async def test_device_command_routing(self, loaded_router):
        """'turn on the lights' should route to device_command."""
        result = await loaded_router.route("turn on the lights")
        assert result.action_category == "device_command"
        assert result.raw_label == "device_command"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_conversation_routing(self, loaded_router):
        """'tell me a joke' should route to conversation."""
        result = await loaded_router.route("tell me a joke")
        assert result.action_category == "conversation"
        assert result.raw_label == "conversation"
        # Conversation is the fallback route, so confidence may be below
        # the semantic threshold -- the important thing is the category.

    @pytest.mark.asyncio
    async def test_get_time_routing(self, loaded_router):
        """'what time is it' should route to get_time tool."""
        result = await loaded_router.route("what time is it")
        assert result.action_category == "tool_use"
        assert result.raw_label == "get_time"
        assert result.tool_name == "get_time"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_reminder_routing(self, loaded_router):
        """'remind me to call the dentist' should route to reminder."""
        result = await loaded_router.route("remind me to call the dentist")
        assert result.action_category == "tool_use"
        assert result.raw_label == "reminder"
        assert result.tool_name == "set_reminder"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_calendar_write_routing(self, loaded_router):
        """'add a meeting to my calendar' should route to calendar_write."""
        result = await loaded_router.route("add a meeting to my calendar for Thursday")
        assert result.action_category == "tool_use"
        assert result.raw_label == "calendar_write"
        assert result.tool_name == "create_calendar_event"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_booking_routing(self, loaded_router):
        """'book an appointment' should route to booking."""
        result = await loaded_router.route("book an appointment for next Monday")
        assert result.action_category == "tool_use"
        assert result.raw_label == "booking"
        assert result.tool_name == "book_appointment"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_security_routing(self, loaded_router):
        """'arm the home security' should route to device_command."""
        result = await loaded_router.route("arm the home security system")
        assert result.action_category == "device_command"
        assert result.raw_label == "security"
        assert result.tool_name is None
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_presence_routing(self, loaded_router):
        """'set it to movie mode' should route to device_command."""
        result = await loaded_router.route("set it to movie mode")
        assert result.action_category == "device_command"
        assert result.raw_label == "presence"
        assert result.tool_name is None
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_route_time_ms_is_populated(self, loaded_router):
        """route_time_ms should be a positive number."""
        result = await loaded_router.route("hello there")
        assert result.route_time_ms > 0


# ---------------------------------------------------------------------------
# Habit guard
# ---------------------------------------------------------------------------


class TestHabitGuard:
    """Tests for the habit-word redirect from get_time to conversation."""

    @pytest.mark.asyncio
    async def test_habit_query_redirects_to_conversation(self, loaded_router):
        """'what time do I usually wake up' should go to conversation, not get_time."""
        result = await loaded_router.route("what time do I usually wake up")
        assert result.action_category == "conversation"
        assert result.raw_label == "conversation"

    @pytest.mark.asyncio
    async def test_normal_time_query_stays_get_time(self, loaded_router):
        """'what time is it' (no habit words) should stay as get_time."""
        result = await loaded_router.route("what time is it")
        assert result.raw_label == "get_time"
        assert result.action_category == "tool_use"

    @pytest.mark.asyncio
    async def test_habit_word_normally(self, loaded_router):
        """'when do we normally get home' should go to conversation."""
        result = await loaded_router.route("when do we normally get home")
        assert result.action_category == "conversation"

    @pytest.mark.asyncio
    async def test_habit_word_routine(self, loaded_router):
        """'what's my typical morning routine' should go to conversation."""
        result = await loaded_router.route("what's my typical morning routine")
        assert result.action_category == "conversation"


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------


class TestLLMFallback:
    """Tests for LLM fallback behavior (mocked -- no real LLM call)."""

    @pytest.mark.asyncio
    async def test_llm_fallback_used_when_semantic_low(self, loaded_router):
        """When semantic confidence is below threshold, LLM fallback should be tried."""
        # Use a gibberish-ish query that will score low on all routes
        query = "xylophone zamboni fruitcake please"

        with patch.object(
            loaded_router, "_llm_classify", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = ("device_command", 0.85, None)

            # Temporarily enable LLM fallback
            original_enabled = loaded_router._config.llm_fallback_enabled
            loaded_router._config.llm_fallback_enabled = True
            try:
                result = await loaded_router.route(query)
            finally:
                loaded_router._config.llm_fallback_enabled = original_enabled

            mock_llm.assert_called_once_with(query)
            assert result.action_category == "device_command"
            assert result.raw_label == "device_command"
            assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_llm_fallback_returns_none_falls_to_conversation(self, loaded_router):
        """When LLM fallback returns None (timeout/error), fall back to conversation."""
        query = "xylophone zamboni fruitcake please"

        with patch.object(
            loaded_router, "_llm_classify", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = None

            original_enabled = loaded_router._config.llm_fallback_enabled
            loaded_router._config.llm_fallback_enabled = True
            try:
                result = await loaded_router.route(query)
            finally:
                loaded_router._config.llm_fallback_enabled = original_enabled

            mock_llm.assert_called_once_with(query)
            assert result.action_category == "conversation"
            assert result.raw_label == "conversation"

    @pytest.mark.asyncio
    async def test_short_query_skips_llm_fallback(self, loaded_router):
        """Queries with fewer than 3 words should skip LLM fallback."""
        query = "hi"  # 1 word -- should skip LLM

        with patch.object(
            loaded_router, "_llm_classify", new_callable=AsyncMock
        ) as mock_llm:
            original_enabled = loaded_router._config.llm_fallback_enabled
            original_threshold = loaded_router._config.confidence_threshold
            loaded_router._config.llm_fallback_enabled = True
            # Set threshold very high so semantic always falls below
            loaded_router._config.confidence_threshold = 0.99
            try:
                result = await loaded_router.route(query)
            finally:
                loaded_router._config.llm_fallback_enabled = original_enabled
                loaded_router._config.confidence_threshold = original_threshold

            # LLM should NOT have been called for a 1-word query
            mock_llm.assert_not_called()
            assert result.action_category == "conversation"

    @pytest.mark.asyncio
    async def test_two_word_query_skips_llm_fallback(self, loaded_router):
        """Queries with exactly 2 words should also skip LLM fallback."""
        query = "ok bye"

        with patch.object(
            loaded_router, "_llm_classify", new_callable=AsyncMock
        ) as mock_llm:
            original_enabled = loaded_router._config.llm_fallback_enabled
            original_threshold = loaded_router._config.confidence_threshold
            loaded_router._config.llm_fallback_enabled = True
            loaded_router._config.confidence_threshold = 0.99
            try:
                result = await loaded_router.route(query)
            finally:
                loaded_router._config.llm_fallback_enabled = original_enabled
                loaded_router._config.confidence_threshold = original_threshold

            mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Disabled router
# ---------------------------------------------------------------------------


class TestDisabledRouter:
    """Tests for when the intent router is disabled via config."""

    @pytest.mark.asyncio
    async def test_disabled_returns_conversation(self, loaded_router):
        """When router is disabled, always return conversation with confidence 0."""
        original_enabled = loaded_router._config.enabled
        loaded_router._config.enabled = False
        try:
            result = await loaded_router.route("turn on the lights")
            assert result.action_category == "conversation"
            assert result.raw_label == "disabled"
            assert result.confidence == 0.0
        finally:
            loaded_router._config.enabled = original_enabled

    @pytest.mark.asyncio
    async def test_disabled_for_tool_query(self, loaded_router):
        """Even tool queries return conversation when disabled."""
        original_enabled = loaded_router._config.enabled
        loaded_router._config.enabled = False
        try:
            result = await loaded_router.route("what time is it")
            assert result.action_category == "conversation"
            assert result.raw_label == "disabled"
            assert result.confidence == 0.0
        finally:
            loaded_router._config.enabled = original_enabled


# ---------------------------------------------------------------------------
# PARAMETERLESS_TOOLS and fast_path_ok
# ---------------------------------------------------------------------------


class TestFastPathOk:
    """Tests for fast_path_ok flag based on PARAMETERLESS_TOOLS."""

    @pytest.mark.asyncio
    async def test_parameterless_tool_has_fast_path(self, loaded_router):
        """get_time is in PARAMETERLESS_TOOLS, so fast_path_ok should be True."""
        result = await loaded_router.route("what time is it")
        assert result.raw_label == "get_time"
        assert result.fast_path_ok is True

    @pytest.mark.asyncio
    async def test_parameterized_tool_no_fast_path(self, loaded_router):
        """reminder requires params, so fast_path_ok should be False."""
        result = await loaded_router.route("remind me to call the dentist tomorrow")
        assert result.raw_label == "reminder"
        assert result.tool_name == "set_reminder"
        assert result.fast_path_ok is False

    @pytest.mark.asyncio
    async def test_device_command_no_fast_path(self, loaded_router):
        """device_command has tool_name=None, so fast_path_ok should be False."""
        result = await loaded_router.route("turn on the kitchen light")
        assert result.raw_label == "device_command"
        assert result.tool_name is None
        assert result.fast_path_ok is False

    @pytest.mark.asyncio
    async def test_conversation_no_fast_path(self, loaded_router):
        """conversation has tool_name=None, so fast_path_ok should be False."""
        result = await loaded_router.route("how are you doing today")
        assert result.fast_path_ok is False

    def test_parameterless_tools_are_known_routes(self):
        """Every tool in PARAMETERLESS_TOOLS should map to a known route."""
        all_tool_names = {
            tool_name
            for _, tool_name in ROUTE_TO_ACTION.values()
            if tool_name is not None
        }
        for tool in PARAMETERLESS_TOOLS:
            assert tool in all_tool_names, (
                f"PARAMETERLESS_TOOLS contains '{tool}' which is not in any "
                f"ROUTE_TO_ACTION tool_name mapping"
            )


# ---------------------------------------------------------------------------
# Route mapping consistency
# ---------------------------------------------------------------------------


class TestRouteMappings:
    """Tests for consistency between route definitions, actions, and workflows."""

    def test_all_route_definitions_have_action_or_workflow_entries(self):
        """Every key in ROUTE_DEFINITIONS should have an entry in
        ROUTE_TO_ACTION or ROUTE_TO_WORKFLOW (or both)."""
        for route_name in ROUTE_DEFINITIONS:
            has_action = route_name in ROUTE_TO_ACTION
            has_workflow = route_name in ROUTE_TO_WORKFLOW
            assert has_action or has_workflow, (
                f"ROUTE_DEFINITIONS has '{route_name}' but it is in neither "
                f"ROUTE_TO_ACTION nor ROUTE_TO_WORKFLOW"
            )

    def test_all_workflow_keys_are_valid_routes(self):
        """Every key in ROUTE_TO_WORKFLOW should be in _VALID_ROUTES."""
        for route_name in ROUTE_TO_WORKFLOW:
            assert route_name in _VALID_ROUTES, (
                f"ROUTE_TO_WORKFLOW has '{route_name}' but it is not in _VALID_ROUTES"
            )

    def test_valid_routes_is_union_of_action_and_workflow(self):
        """_VALID_ROUTES should be the union of action and workflow keys."""
        expected = set(ROUTE_TO_ACTION.keys()) | set(ROUTE_TO_WORKFLOW.keys())
        assert _VALID_ROUTES == expected

    def test_route_to_action_values_are_tuples(self):
        """Every ROUTE_TO_ACTION value should be a (category, tool_name) tuple."""
        for route_name, value in ROUTE_TO_ACTION.items():
            assert isinstance(value, tuple), (
                f"ROUTE_TO_ACTION['{route_name}'] should be a tuple"
            )
            assert len(value) == 2, (
                f"ROUTE_TO_ACTION['{route_name}'] should have exactly 2 elements"
            )
            category, tool_name = value
            assert isinstance(category, str)
            assert tool_name is None or isinstance(tool_name, str)

    def test_conversation_route_has_no_tool(self):
        """The conversation route should have tool_name=None."""
        category, tool_name = ROUTE_TO_ACTION["conversation"]
        assert category == "conversation"
        assert tool_name is None

    def test_device_command_route_has_no_tool(self):
        """The device_command route should have tool_name=None."""
        category, tool_name = ROUTE_TO_ACTION["device_command"]
        assert category == "device_command"
        assert tool_name is None

    def test_route_definitions_not_empty(self):
        """Each route in ROUTE_DEFINITIONS should have at least one exemplar."""
        for route_name, utterances in ROUTE_DEFINITIONS.items():
            assert len(utterances) > 0, (
                f"ROUTE_DEFINITIONS['{route_name}'] has no exemplar utterances"
            )


# ---------------------------------------------------------------------------
# Standalone tool routability
# ---------------------------------------------------------------------------


# Canonical set of standalone tool names: every tool that is NOT workflow-only
# and therefore MUST have its name appear in ROUTE_TO_ACTION values.
# This list is derived from WORKFLOW_ONLY_TOOLS (services/intent_router.py) and
# the registered tool registry. It is duplicated here intentionally so the test
# does NOT import the heavy tool modules (which have external deps like httpx).
# NOTE: If you add a new standalone tool to atlas_brain/tools/ and the tool
# registry but forget to add it here, the invariant test won't catch it. Keep
# this set in sync with tool registrations in atlas_brain/tools/__init__.py.
_EXPECTED_STANDALONE_TOOL_NAMES: frozenset[str] = frozenset({
    # Utility / query tools (direct routes)
    "get_time",
    "get_weather",
    "get_traffic",
    "get_location",
    "get_calendar",
    "list_reminders",
    "run_digest",
    # Notification
    "send_notification",
    # Presence (room detection, occupancy)
    "where_am_i",
    "who_is_here",
    # Security: standalone detection queries (direct routes)
    "get_person_at_location",
    "get_motion_events",
    # Display
    "show_camera_feed",
    "close_camera_feed",
})


class TestStandaloneToolRoutability:
    """
    Regression tests that prevent standalone tools from becoming unreachable.

    Root cause of the original tool-unreachability bug (2026-02-07 audit):
    20 of 26 registered tools had no entry in ROUTE_DEFINITIONS, ROUTE_TO_ACTION,
    or ROUTE_TO_WORKFLOW.  The SemanticIntentRouter therefore never assigned user
    queries to those tools and they silently fell through to 'conversation'.

    How reachability works (three required entries):
      1. ROUTE_DEFINITIONS  — exemplar utterances that train the route centroid
      2. ROUTE_TO_ACTION    — maps route name → (action_category, tool_name)
      3. ROUTE_TO_WORKFLOW  — for multi-turn workflows, maps route → workflow type

    A tool NOT in WORKFLOW_ONLY_TOOLS (tools/__init__.py) is "standalone" and
    MUST appear in ROUTE_TO_ACTION values.  Failing to add ROUTE_DEFINITIONS +
    ROUTE_TO_ACTION entries for a new standalone tool silently makes it
    unreachable from voice/text.
    """

    def test_all_standalone_tools_appear_in_route_to_action_values(self):
        """Every expected standalone tool name must be in ROUTE_TO_ACTION values."""
        routable_names = {
            tool_name
            for _, tool_name in ROUTE_TO_ACTION.values()
            if tool_name is not None
        }
        for tool_name in _EXPECTED_STANDALONE_TOOL_NAMES:
            assert tool_name in routable_names, (
                f"Standalone tool '{tool_name}' has no ROUTE_TO_ACTION entry. "
                f"Add an exemplar route in ROUTE_DEFINITIONS and map it in "
                f"ROUTE_TO_ACTION, otherwise voice/text commands will silently "
                f"fall through to conversation instead of executing the tool."
            )

    def test_route_to_action_tool_names_are_standalone_or_workflow_only(self):
        """Every tool_name in ROUTE_TO_ACTION should be a known standalone tool.

        This catches accidental typos in ROUTE_TO_ACTION tool names and also
        flags if a workflow-only tool was incorrectly given a direct route entry
        (which is harmless but indicates a mapping inconsistency).
        """
        routable_names = {
            tool_name
            for _, tool_name in ROUTE_TO_ACTION.values()
            if tool_name is not None
        }
        # Every tool name in ROUTE_TO_ACTION must be either standalone or workflow-only.
        # Unknown names indicate a typo or a tool that was renamed without updating the router.
        all_known_names = _EXPECTED_STANDALONE_TOOL_NAMES | WORKFLOW_ONLY_TOOLS
        for tool_name in routable_names:
            assert tool_name in all_known_names, (
                f"ROUTE_TO_ACTION references tool '{tool_name}' which is not in "
                f"_EXPECTED_STANDALONE_TOOL_NAMES or WORKFLOW_ONLY_TOOLS. "
                f"Either add it to the appropriate set or correct the tool name."
            )

    def test_workflow_only_tools_are_not_standalone_tools(self):
        """WORKFLOW_ONLY_TOOLS and _EXPECTED_STANDALONE_TOOL_NAMES must be disjoint."""
        overlap = WORKFLOW_ONLY_TOOLS & _EXPECTED_STANDALONE_TOOL_NAMES
        assert not overlap, (
            f"Tools appear in both WORKFLOW_ONLY_TOOLS and "
            f"_EXPECTED_STANDALONE_TOOL_NAMES (they should be disjoint): {overlap}"
        )


# ---------------------------------------------------------------------------
# IntentRouteResult dataclass
# ---------------------------------------------------------------------------


class TestIntentRouteResult:
    """Tests for the IntentRouteResult dataclass."""

    def test_defaults(self):
        result = IntentRouteResult(
            action_category="conversation",
            raw_label="conversation",
            confidence=0.75,
        )
        assert result.route_time_ms == 0.0
        assert result.tool_name is None
        assert result.fast_path_ok is False

    def test_full_construction(self):
        result = IntentRouteResult(
            action_category="tool_use",
            raw_label="get_time",
            confidence=0.92,
            route_time_ms=5.3,
            tool_name="get_time",
            fast_path_ok=True,
        )
        assert result.action_category == "tool_use"
        assert result.raw_label == "get_time"
        assert result.confidence == 0.92
        assert result.route_time_ms == 5.3
        assert result.tool_name == "get_time"
        assert result.fast_path_ok is True


# ---------------------------------------------------------------------------
# Router lifecycle
# ---------------------------------------------------------------------------


class TestRouterLifecycle:
    """Tests for load/unload behavior."""

    def test_unloaded_router_is_not_loaded(self):
        """A fresh router should not be loaded."""
        router = SemanticIntentRouter()
        assert not router.is_loaded
        assert len(router._route_centroids) == 0

    def test_load_sync_makes_router_loaded(self):
        """load_sync() should populate centroids."""
        router = SemanticIntentRouter()
        router.load_sync()
        try:
            assert router.is_loaded
            assert len(router._route_centroids) == len(ROUTE_DEFINITIONS)
        finally:
            router.unload()

    def test_unload_clears_centroids(self):
        """unload() should clear centroids and embedder."""
        router = SemanticIntentRouter()
        router.load_sync()
        router.unload()
        assert not router.is_loaded
        assert router._embedder is None
        assert len(router._route_centroids) == 0
