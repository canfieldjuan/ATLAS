"""
Tests for Voice Entity Context Between Turns feature.

Covers:
1. Unit: entity_context.py extraction and formatting functions
2. Unit: MemoryContext.recent_entities field and defaults
3. Integration: _load_recent_entities reads entity metadata from DB
4. Integration: gather_context populates recent_entities from all turn types
5. Wiring: launcher.py and atlas.py source inspection for injection points
"""

import inspect
import json
import pytest
import pytest_asyncio

from atlas_brain.voice.entity_context import (
    EntityRef,
    collect_recent_entities,
    extract_location_from_text,
    format_entity_context,
)
from atlas_brain.memory.service import MemoryContext, MemoryService, get_memory_service


# ------------------------------------------------------------------ #
# Unit: entity_context.py
# ------------------------------------------------------------------ #

class TestExtractLocationFromText:
    """extract_location_from_text: regex-based location extraction."""

    def test_weather_in_pattern(self):
        loc = extract_location_from_text("The weather in New York is sunny today.")
        assert loc == "New York"

    def test_in_pattern(self):
        loc = extract_location_from_text("It is 72 degrees in Chicago right now.")
        assert loc == "Chicago"

    def test_no_match_returns_none(self):
        assert extract_location_from_text("turn on the lights") is None

    def test_empty_string_returns_none(self):
        assert extract_location_from_text("") is None

    def test_too_short_location_skipped(self):
        # Single-char groups should not match (len > 2 required)
        result = extract_location_from_text("weather in LA today")
        # "LA" has len 2, which fails > 2 check
        assert result is None

    def test_does_not_bleed_past_punctuation(self):
        loc = extract_location_from_text("The weather in Dallas, Texas is warm.")
        # Should stop at comma
        assert loc is not None
        assert "," not in loc


class TestCollectRecentEntities:
    """collect_recent_entities: dedup and cap logic."""

    def test_basic_device_entity(self):
        dicts = [{"type": "device", "name": "kitchen light", "action": "turn_on"}]
        refs = collect_recent_entities(dicts)
        assert len(refs) == 1
        assert refs[0].type == "device"
        assert refs[0].name == "kitchen light"
        assert refs[0].action == "turn_on"
        assert refs[0].source == ""

    def test_source_field_preserved(self):
        dicts = [{"type": "person", "name": "Juan", "source": "speaker"}]
        refs = collect_recent_entities(dicts)
        assert refs[0].source == "speaker"

    def test_empty_name_skipped(self):
        dicts = [{"type": "device", "name": ""}, {"type": "person", "name": "Juan"}]
        refs = collect_recent_entities(dicts)
        assert len(refs) == 1
        assert refs[0].name == "Juan"

    def test_deduplication_same_type_and_name(self):
        dicts = [
            {"type": "device", "name": "kitchen light", "action": "turn_on"},
            {"type": "device", "name": "kitchen light", "action": "turn_off"},
        ]
        refs = collect_recent_entities(dicts)
        assert len(refs) == 1
        # First occurrence wins
        assert refs[0].action == "turn_on"

    def test_dedup_case_insensitive(self):
        dicts = [
            {"type": "device", "name": "Kitchen Light"},
            {"type": "device", "name": "kitchen light"},
        ]
        refs = collect_recent_entities(dicts)
        assert len(refs) == 1

    def test_same_name_different_type_not_deduped(self):
        dicts = [
            {"type": "device", "name": "kitchen"},
            {"type": "location", "name": "kitchen"},
        ]
        refs = collect_recent_entities(dicts)
        assert len(refs) == 2

    def test_cap_enforced(self):
        # limit=3 -> cap = 3 * 4 = 12 entities max
        dicts = [{"type": "device", "name": f"light_{i}"} for i in range(20)]
        refs = collect_recent_entities(dicts, limit=3)
        assert len(refs) == 12

    def test_empty_input(self):
        assert collect_recent_entities([]) == []

    def test_missing_type_gets_unknown(self):
        dicts = [{"name": "thing"}]
        refs = collect_recent_entities(dicts)
        assert len(refs) == 1
        assert refs[0].type == "unknown"


class TestFormatEntityContext:
    """format_entity_context: output structure and ordering."""

    def test_empty_returns_none(self):
        assert format_entity_context([]) is None

    def test_basic_device(self):
        refs = [EntityRef(type="device", name="kitchen light", action="turn_on")]
        out = format_entity_context(refs)
        assert out is not None
        assert out.startswith("Recently mentioned:")
        assert "- device: kitchen light (turn_on)" in out

    def test_no_action_no_parens(self):
        refs = [EntityRef(type="person", name="Juan")]
        out = format_entity_context(refs)
        assert "Juan ()" not in out
        assert "- person: Juan" in out

    def test_type_ordering(self):
        refs = [
            EntityRef(type="topic", name="weather"),
            EntityRef(type="location", name="Dallas"),
            EntityRef(type="person", name="Juan"),
            EntityRef(type="device", name="lamp", action="turn_on"),
        ]
        out = format_entity_context(refs)
        lines = out.split("\n")
        type_order = [l.split(":")[0].strip("- ") for l in lines if l.startswith("- ")]
        assert type_order == ["device", "person", "location", "topic"]

    def test_multiple_same_type_on_one_line(self):
        refs = [
            EntityRef(type="device", name="kitchen light", action="turn_on"),
            EntityRef(type="device", name="bedroom lamp"),
        ]
        out = format_entity_context(refs)
        device_line = [l for l in out.split("\n") if "device" in l][0]
        assert "kitchen light" in device_line
        assert "bedroom lamp" in device_line

    def test_unknown_type_not_in_output(self):
        # unknown type is not in the ordered list (device/person/location/topic)
        refs = [EntityRef(type="unknown", name="thing")]
        out = format_entity_context(refs)
        assert out is None  # no listed types matched -> only header -> None


class TestMemoryContextRecentEntitiesField:
    """MemoryContext has recent_entities field with correct default."""

    def test_default_is_empty_list(self):
        ctx = MemoryContext()
        assert ctx.recent_entities == []

    def test_field_accepts_entity_dicts(self):
        ctx = MemoryContext(recent_entities=[{"type": "device", "name": "lamp"}])
        assert len(ctx.recent_entities) == 1

    def test_independent_default_per_instance(self):
        a = MemoryContext()
        b = MemoryContext()
        a.recent_entities.append({"type": "device", "name": "x"})
        assert b.recent_entities == []


# ------------------------------------------------------------------ #
# Integration: DB round-trip for entity metadata
# ------------------------------------------------------------------ #

@pytest.mark.integration
class TestLoadRecentEntitiesIntegration:
    """_load_recent_entities reads entities from metadata across all turn types."""

    @pytest.mark.asyncio
    async def test_command_turn_entities_loaded(self, db_pool, test_session, conversation_repo):
        """Entities stored in a command turn's metadata are returned."""
        sid = str(test_session)
        entities = [{"type": "device", "name": "kitchen light", "action": "turn_on", "source": "command"}]

        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Turning on kitchen light.",
            turn_type="command",
            metadata={"entities": entities},
        )

        svc = get_memory_service()
        result = await svc._load_recent_entities(sid, limit=3)
        assert len(result) == 1
        assert result[0]["type"] == "device"
        assert result[0]["name"] == "kitchen light"

    @pytest.mark.asyncio
    async def test_conversation_turn_entities_loaded(self, db_pool, test_session, conversation_repo):
        """Entities stored in a conversation turn's metadata are returned."""
        sid = str(test_session)
        entities = [{"type": "person", "name": "Juan", "source": "speaker"}]

        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Hi Juan!",
            turn_type="conversation",
            metadata={"entities": entities},
        )

        svc = get_memory_service()
        result = await svc._load_recent_entities(sid, limit=3)
        assert any(e["name"] == "Juan" for e in result)

    @pytest.mark.asyncio
    async def test_entities_from_both_turn_types(self, db_pool, test_session, conversation_repo):
        """Entities from both command and conversation turns are returned together."""
        sid = str(test_session)

        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Turning on kitchen light.",
            turn_type="command",
            metadata={"entities": [{"type": "device", "name": "kitchen light"}]},
        )
        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Sure Juan!",
            turn_type="conversation",
            metadata={"entities": [{"type": "person", "name": "Juan"}]},
        )

        svc = get_memory_service()
        result = await svc._load_recent_entities(sid, limit=3)
        names = [e["name"] for e in result]
        assert "kitchen light" in names
        assert "Juan" in names

    @pytest.mark.asyncio
    async def test_turns_without_entities_skipped(self, db_pool, test_session, conversation_repo):
        """Turns with no entities in metadata return empty list."""
        sid = str(test_session)

        await conversation_repo.add_turn(
            session_id=test_session,
            role="user",
            content="Hello",
            turn_type="conversation",
            metadata={},
        )

        svc = get_memory_service()
        result = await svc._load_recent_entities(sid, limit=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_limit_respected(self, db_pool, test_session, conversation_repo):
        """Only the last `limit` turns are queried."""
        sid = str(test_session)

        for i in range(5):
            await conversation_repo.add_turn(
                session_id=test_session,
                role="assistant",
                content=f"Response {i}",
                turn_type="command",
                metadata={"entities": [{"type": "device", "name": f"light_{i}"}]},
            )

        svc = get_memory_service()
        # limit=2 -> only last 2 turns
        result = await svc._load_recent_entities(sid, limit=2)
        # Should have entities from at most 2 turns
        names = [e["name"] for e in result]
        assert "light_0" not in names  # oldest turn excluded


@pytest.mark.integration
class TestGatherContextEntityIntegration:
    """gather_context populates recent_entities from DB."""

    @pytest.mark.asyncio
    async def test_gather_context_populates_recent_entities(
        self, db_pool, test_session, conversation_repo
    ):
        """gather_context with include_history=True populates recent_entities."""
        sid = str(test_session)

        await conversation_repo.add_turn(
            session_id=test_session,
            role="assistant",
            content="Turning on kitchen light.",
            turn_type="command",
            metadata={"entities": [{"type": "device", "name": "kitchen light", "action": "turn_on"}]},
        )

        svc = get_memory_service()
        ctx = await svc.gather_context(
            query="dim them",
            session_id=sid,
            include_rag=False,
            include_history=True,
            include_physical=False,
        )

        assert len(ctx.recent_entities) >= 1
        assert any(e["name"] == "kitchen light" for e in ctx.recent_entities)

    @pytest.mark.asyncio
    async def test_gather_context_no_history_skips_entities(self, db_pool, test_session):
        """gather_context with include_history=False leaves recent_entities empty."""
        sid = str(test_session)

        svc = get_memory_service()
        ctx = await svc.gather_context(
            query="test",
            session_id=sid,
            include_rag=False,
            include_history=False,
            include_physical=False,
        )

        assert ctx.recent_entities == []

    @pytest.mark.asyncio
    async def test_gather_context_no_session_skips_entities(self, db_pool):
        """gather_context with no session_id leaves recent_entities empty."""
        svc = get_memory_service()
        ctx = await svc.gather_context(
            query="test",
            session_id=None,
            include_rag=False,
            include_history=True,
            include_physical=False,
        )

        assert ctx.recent_entities == []


# ------------------------------------------------------------------ #
# Wiring: source inspection
# ------------------------------------------------------------------ #

class TestAtlasWiring:
    """atlas.py _store_turn injects entities into assistant_metadata."""

    def _source(self):
        import atlas_brain.agents.graphs.atlas as mod
        return inspect.getsource(mod.AtlasAgentGraph._store_turn)

    def test_entity_extraction_block_present(self):
        assert "Entity extraction from intent and state" in self._source()

    def test_device_entity_extracted(self):
        src = self._source()
        assert "type.*device" in src or '"device"' in src

    def test_uses_tool_use_not_tool_call(self):
        src = self._source()
        assert '"tool_use"' in src
        assert '"tool_call"' not in src

    def test_location_extraction_called(self):
        assert "extract_location_from_text" in self._source()

    def test_person_entity_extracted(self):
        src = self._source()
        assert '"person"' in src

    def test_entities_merged_into_assistant_metadata(self):
        src = self._source()
        assert "assistant_metadata" in src
        assert "entities" in src


class TestLauncherWiring:
    """launcher.py injects entity context into prompt and persists person entity."""

    def _stream_source(self):
        import atlas_brain.voice.launcher as mod
        return inspect.getsource(mod._stream_llm_response)

    def _persist_source(self):
        import atlas_brain.voice.launcher as mod
        return inspect.getsource(mod._persist_streaming_turns)

    def test_entity_injection_in_stream(self):
        assert "recent_entities" in self._stream_source()

    def test_entity_context_imported_in_stream(self):
        src = self._stream_source()
        assert "entity_context" in src
        assert "format_entity_context" in src

    def test_prompt_parts_receives_entity_str(self):
        src = self._stream_source()
        assert "prompt_parts" in src
        assert "entity_str" in src

    def test_person_entity_persisted(self):
        src = self._persist_source()
        assert '"person"' in src
        assert "speaker_name" in src

    def test_person_entity_uses_setdefault(self):
        src = self._persist_source()
        assert "setdefault" in src

    def test_location_extracted_from_transcript(self):
        src = self._persist_source()
        assert "extract_location_from_text" in src
        assert "user_text" in src

    def test_agent_path_injects_entity_context(self):
        import atlas_brain.agents.graphs.atlas as mod
        src = inspect.getsource(mod._generate_llm_response)
        assert "recent_entities" in src
        assert "format_entity_context" in src
        assert "system_parts" in src


class TestEntityContextModule:
    """entity_context.py module is importable and has expected public API."""

    def test_module_importable(self):
        import atlas_brain.voice.entity_context as mod
        assert hasattr(mod, "EntityRef")
        assert hasattr(mod, "extract_location_from_text")
        assert hasattr(mod, "collect_recent_entities")
        assert hasattr(mod, "format_entity_context")

    def test_entity_ref_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(EntityRef)

    def test_entity_ref_fields(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(EntityRef)}
        assert field_names == {"type", "name", "action", "source"}
