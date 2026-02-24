"""
Tests for Call Person voice workflow.

Covers:
1. Intent routing: call_person route in ROUTE_DEFINITIONS / ROUTE_TO_ACTION / ROUTE_TO_WORKFLOW
2. LLM routing: "call" workflow in TRIAGE_WORKFLOWS
3. Workflow constants and system prompt construction
4. Entity context: "call" topic in _TOPIC_KEYWORDS
5. Atlas.py dispatch wiring: call workflow in _WORKFLOW_NAMES and imports
6. Reasoning event type: CALL_COMPLETED exists
7. Semantic routing: "call John" routes to call_person (real model)
"""

import inspect

import pytest

from atlas_brain.services.intent_router import (
    ROUTE_DEFINITIONS,
    ROUTE_TO_ACTION,
    ROUTE_TO_WORKFLOW,
    SemanticIntentRouter,
    _VALID_ROUTES,
)
from atlas_brain.services.llm_router import TRIAGE_WORKFLOWS
from atlas_brain.agents.graphs.call import (
    CALL_WORKFLOW_TYPE,
    _build_call_system_prompt,
    run_call_workflow,
)
from atlas_brain.voice.entity_context import (
    _TOPIC_KEYWORDS,
    extract_topic_from_text,
)
from atlas_brain.reasoning.events import EventType


# ------------------------------------------------------------------ #
# Intent Router wiring
# ------------------------------------------------------------------ #


class TestCallPersonIntentRoute:
    """call_person route is wired in all four intent router data structures."""

    def test_route_definitions_has_call_person(self):
        assert "call_person" in ROUTE_DEFINITIONS
        exemplars = ROUTE_DEFINITIONS["call_person"]
        assert len(exemplars) >= 5, "Need enough exemplars for good centroid"

    def test_route_to_action_has_call_person(self):
        assert "call_person" in ROUTE_TO_ACTION
        category, tool = ROUTE_TO_ACTION["call_person"]
        assert category == "tool_use"
        assert tool == "make_call"

    def test_route_to_workflow_has_call_person(self):
        assert "call_person" in ROUTE_TO_WORKFLOW
        assert ROUTE_TO_WORKFLOW["call_person"] == "call"

    def test_valid_routes_includes_call_person(self):
        assert "call_person" in _VALID_ROUTES

    def test_exemplars_are_strings(self):
        for utterance in ROUTE_DEFINITIONS["call_person"]:
            assert isinstance(utterance, str)
            assert len(utterance) > 0


# ------------------------------------------------------------------ #
# LLM Router wiring
# ------------------------------------------------------------------ #


class TestCallLLMRouting:
    """call workflow routed to triage (Haiku) LLM."""

    def test_call_in_triage_workflows(self):
        assert "call" in TRIAGE_WORKFLOWS


# ------------------------------------------------------------------ #
# Workflow constants and prompt
# ------------------------------------------------------------------ #


class TestCallWorkflowModule:
    """call.py module constants and system prompt."""

    def test_workflow_type_constant(self):
        assert CALL_WORKFLOW_TYPE == "call"

    def test_system_prompt_without_speaker(self):
        prompt = _build_call_system_prompt()
        assert "search_contacts" in prompt
        assert "confirm" in prompt.lower()
        assert "voice" in prompt.lower()

    def test_system_prompt_with_speaker(self):
        prompt = _build_call_system_prompt(speaker_id="Juan")
        assert "Juan" in prompt
        assert "business owner" in prompt

    def test_run_call_workflow_is_async(self):
        assert inspect.iscoroutinefunction(run_call_workflow)


# ------------------------------------------------------------------ #
# Entity context: _TOPIC_KEYWORDS
# ------------------------------------------------------------------ #


class TestCallTopicKeywords:
    """call topic in entity_context._TOPIC_KEYWORDS."""

    def test_call_topic_exists(self):
        topic_names = [name for name, _ in _TOPIC_KEYWORDS]
        assert "call" in topic_names

    def test_call_keywords_not_empty(self):
        for name, keywords in _TOPIC_KEYWORDS:
            if name == "call":
                assert len(keywords) >= 3
                return
        pytest.fail("call topic not found in _TOPIC_KEYWORDS")

    def test_extract_topic_call_someone(self):
        assert extract_topic_from_text("call someone for me") == "call"

    def test_extract_topic_make_a_call(self):
        assert extract_topic_from_text("make a call to the plumber") == "call"

    def test_extract_topic_phone(self):
        assert extract_topic_from_text("phone Sarah") == "call"

    def test_extract_topic_give_them_a_call(self):
        assert extract_topic_from_text("give them a call") == "call"

    def test_extract_topic_dial(self):
        assert extract_topic_from_text("dial the office number") == "call"

    def test_extract_topic_unrelated_returns_none(self):
        assert extract_topic_from_text("turn on the lights") is None


# ------------------------------------------------------------------ #
# Atlas.py dispatch wiring
# ------------------------------------------------------------------ #


class TestAtlasDispatchWiring:
    """call workflow is wired in atlas.py _WORKFLOW_NAMES and dispatch."""

    def test_workflow_names_has_call(self):
        from atlas_brain.agents.graphs.atlas import _WORKFLOW_NAMES
        assert "call" in _WORKFLOW_NAMES
        assert _WORKFLOW_NAMES["call"] == "phone call"

    def test_start_workflow_has_call_branch(self):
        from atlas_brain.agents.graphs.atlas import start_workflow
        source = inspect.getsource(start_workflow)
        assert "CALL_WORKFLOW_TYPE" in source
        assert "run_call_workflow" in source

    def test_continue_workflow_has_call_branch(self):
        from atlas_brain.agents.graphs.atlas import continue_workflow
        source = inspect.getsource(continue_workflow)
        assert "CALL_WORKFLOW_TYPE" in source
        assert "run_call_workflow" in source

    def test_atlas_imports_call_workflow(self):
        from atlas_brain.agents.graphs import atlas
        assert hasattr(atlas, "CALL_WORKFLOW_TYPE")
        assert hasattr(atlas, "run_call_workflow")


# ------------------------------------------------------------------ #
# Package exports
# ------------------------------------------------------------------ #


class TestPackageExports:
    """call workflow exported from agents.graphs package."""

    def test_init_exports_run_call_workflow(self):
        from atlas_brain.agents.graphs import run_call_workflow as fn
        assert callable(fn)

    def test_init_exports_call_workflow_type(self):
        from atlas_brain.agents.graphs import CALL_WORKFLOW_TYPE as wt
        assert wt == "call"


# ------------------------------------------------------------------ #
# Reasoning event type
# ------------------------------------------------------------------ #


class TestReasoningEventType:
    """call.completed event type exists in EventType registry."""

    def test_call_completed_event_exists(self):
        assert hasattr(EventType, "CALL_COMPLETED")
        assert EventType.CALL_COMPLETED == "call.completed"

    def test_call_py_emits_event(self):
        source = inspect.getsource(run_call_workflow)
        assert "call.completed" in source
        assert "emit_if_enabled" in source


# ------------------------------------------------------------------ #
# Semantic routing (real embedding model)
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def loaded_router():
    """Load the real SemanticIntentRouter with all-MiniLM-L6-v2."""
    router = SemanticIntentRouter()
    router.load_sync()
    yield router
    router.unload()


class TestCallPersonSemanticRouting:
    """Semantic classification of call-related queries (real model)."""

    @pytest.mark.asyncio
    async def test_call_john_routes_to_call_person(self, loaded_router):
        result = await loaded_router.route("call John")
        assert result.raw_label == "call_person"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_phone_sarah_routes_to_call_person(self, loaded_router):
        result = await loaded_router.route("phone Sarah")
        assert result.raw_label == "call_person"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_make_a_call_routes_to_call_person(self, loaded_router):
        result = await loaded_router.route("make a call to the plumber")
        assert result.raw_label == "call_person"
        assert result.confidence >= 0.50

    @pytest.mark.asyncio
    async def test_give_them_a_call_routes_to_call_person(self, loaded_router):
        result = await loaded_router.route("give them a call")
        assert result.raw_label == "call_person"
        assert result.confidence >= 0.45

    @pytest.mark.asyncio
    async def test_call_does_not_steal_reminder_route(self, loaded_router):
        """'remind me to call the dentist' must NOT route to call_person."""
        result = await loaded_router.route("remind me to call the dentist")
        assert result.raw_label != "call_person"
