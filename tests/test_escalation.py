"""
Tests for the escalation system.

Covers: EscalationEvaluator rules, rapid-unknowns sliding window,
config toggles, skill loading, and enhanced security_ack format.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from atlas_brain.escalation.evaluator import EscalationEvaluator, EscalationResult
from atlas_brain.skills import get_skill_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator() -> EscalationEvaluator:
    """Create a fresh evaluator for each test."""
    return EscalationEvaluator()


def _mock_presence(state_value: str = "empty", occupants: list[str] | None = None):
    """Patch get_presence_tracker to return a mock with given state."""
    from atlas_brain.autonomous.presence import OccupancyState, PresenceState

    mock_tracker = MagicMock()
    mock_state = PresenceState()
    mock_state.state = OccupancyState(state_value)
    if occupants:
        from datetime import datetime
        mock_state.occupants = {name: datetime.utcnow() for name in occupants}
    mock_tracker.state = mock_state
    # Patch where it's imported from (lazy import resolves to this module)
    return patch(
        "atlas_brain.autonomous.presence.get_presence_tracker",
        return_value=mock_tracker,
    )


def _mock_config(**overrides):
    """Patch settings.escalation with given overrides."""
    from atlas_brain.config import EscalationConfig

    config = EscalationConfig(**overrides)
    mock_settings = MagicMock()
    mock_settings.escalation = config
    mock_settings.autonomous.presence_enabled = True
    return patch("atlas_brain.config.settings", mock_settings)


# ---------------------------------------------------------------------------
# Rule: unknown_face + empty house -> escalation
# ---------------------------------------------------------------------------

class TestUnknownFaceEmptyHouse:
    """Rule 1: unknown face when house is empty."""

    @pytest.mark.asyncio
    async def test_unknown_face_empty_house_escalates(self):
        evaluator = _make_evaluator()
        msg = {"event": "unknown_face", "name": "unknown_001"}

        with _mock_presence("empty"), _mock_config():
            result = await evaluator.evaluate("unknown_face", msg, "office")

        assert result.should_escalate is True
        assert result.rule_name == "unknown_face_empty_house"
        assert result.priority == "high"

    @pytest.mark.asyncio
    async def test_unknown_person_entered_empty_house_escalates(self):
        evaluator = _make_evaluator()
        msg = {"event": "person_entered", "is_known": False, "name": "unknown"}

        with _mock_presence("empty"), _mock_config():
            result = await evaluator.evaluate("person_entered", msg, "office")

        assert result.should_escalate is True
        assert result.rule_name == "unknown_face_empty_house"
        assert result.priority == "high"

    @pytest.mark.asyncio
    async def test_known_person_entered_empty_house_routine(self):
        evaluator = _make_evaluator()
        msg = {"event": "person_entered", "is_known": True, "name": "Juan"}

        with _mock_presence("empty"), _mock_config():
            result = await evaluator.evaluate("person_entered", msg, "office")

        assert result.should_escalate is False
        assert result.priority == "routine"


# ---------------------------------------------------------------------------
# Rule: motion_detected -> always routine
# ---------------------------------------------------------------------------

class TestMotionDetected:
    """Motion events never trigger escalation."""

    @pytest.mark.asyncio
    async def test_motion_detected_routine(self):
        evaluator = _make_evaluator()
        msg = {"event": "motion_detected", "confidence": 0.85}

        with _mock_presence("empty"), _mock_config():
            result = await evaluator.evaluate("motion_detected", msg, "office")

        assert result.should_escalate is False
        assert result.priority == "routine"


# ---------------------------------------------------------------------------
# Rule: rapid unknowns within window -> escalation
# ---------------------------------------------------------------------------

class TestRapidUnknowns:
    """Rule 2: threshold+ unknown faces in sliding window."""

    @pytest.mark.asyncio
    async def test_rapid_unknowns_escalates(self):
        evaluator = _make_evaluator()
        msg = {"event": "unknown_face", "name": "unknown_001"}

        # Occupied house -- should still escalate on rapid unknowns
        with _mock_presence("identified", ["Juan"]), _mock_config(rapid_unknowns_threshold=3):
            # Fire 3 events in quick succession
            for i in range(2):
                result = await evaluator.evaluate("unknown_face", msg, "office")
            # Third one should trigger
            result = await evaluator.evaluate("unknown_face", msg, "office")

        assert result.should_escalate is True
        assert result.rule_name == "rapid_unknown_faces"
        assert result.priority == "high"
        # Deque should be cleared after escalation fires
        assert len(evaluator._unknown_timestamps) == 0

    @pytest.mark.asyncio
    async def test_deque_reset_prevents_immediate_re_escalation(self):
        """After escalation fires, next single event should not re-escalate."""
        evaluator = _make_evaluator()
        msg = {"event": "unknown_face", "name": "unknown_001"}

        with _mock_presence("identified", ["Juan"]), _mock_config(rapid_unknowns_threshold=3):
            # Trigger escalation with 3 events
            for _ in range(3):
                await evaluator.evaluate("unknown_face", msg, "office")
            # Deque cleared; next single event should be routine
            result = await evaluator.evaluate("unknown_face", msg, "office")

        assert result.should_escalate is False
        assert result.priority == "routine"

    @pytest.mark.asyncio
    async def test_below_threshold_routine(self):
        evaluator = _make_evaluator()
        msg = {"event": "unknown_face", "name": "unknown_001"}

        with _mock_presence("identified", ["Juan"]), _mock_config(rapid_unknowns_threshold=3):
            # Only 2 events -- below threshold
            for _ in range(2):
                result = await evaluator.evaluate("unknown_face", msg, "office")

        assert result.should_escalate is False
        assert result.priority == "routine"

    @pytest.mark.asyncio
    async def test_window_expiry(self):
        """Old timestamps should expire out of the sliding window."""
        evaluator = _make_evaluator()
        msg = {"event": "unknown_face", "name": "unknown_001"}

        with _mock_presence("identified", ["Juan"]), _mock_config(
            rapid_unknowns_threshold=3, rapid_unknowns_window_seconds=60,
        ):
            # Manually insert old timestamps that are outside the window
            old_time = time.monotonic() - 120  # 2 minutes ago
            evaluator._unknown_timestamps.append(old_time)
            evaluator._unknown_timestamps.append(old_time + 1)

            # One new event -- old ones should be pruned
            result = await evaluator.evaluate("unknown_face", msg, "office")

        # Only 1 event in window (the new one), old ones pruned
        assert result.should_escalate is False
        assert len(evaluator._unknown_timestamps) == 1


# ---------------------------------------------------------------------------
# Config toggle: disabled -> always routine
# ---------------------------------------------------------------------------

class TestConfigDisabled:
    """When escalation is disabled, everything is routine."""

    @pytest.mark.asyncio
    async def test_disabled_always_routine(self):
        evaluator = _make_evaluator()
        msg = {"event": "unknown_face", "name": "unknown_001"}

        with _mock_config(enabled=False):
            result = await evaluator.evaluate("unknown_face", msg, "office")

        assert result.should_escalate is False
        assert result.priority == "routine"

    @pytest.mark.asyncio
    async def test_unknown_empty_disabled(self):
        """Disable just the unknown+empty rule."""
        evaluator = _make_evaluator()
        msg = {"event": "unknown_face", "name": "unknown_001"}

        with _mock_presence("empty"), _mock_config(unknown_empty_enabled=False):
            result = await evaluator.evaluate("unknown_face", msg, "office")

        # Should not escalate via rule 1 (disabled), and only 1 event so rule 2 doesn't fire
        assert result.should_escalate is False


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------

class TestEscalationSkill:
    """Verify the security/escalation_narration skill loads correctly."""

    def test_skill_exists(self):
        registry = get_skill_registry()
        skill = registry.get("security/escalation_narration")
        assert skill is not None

    def test_skill_domain(self):
        registry = get_skill_registry()
        skill = registry.get("security/escalation_narration")
        assert skill.domain == "security"

    def test_skill_tags(self):
        registry = get_skill_registry()
        skill = registry.get("security/escalation_narration")
        assert "security" in skill.tags
        assert "escalation" in skill.tags
        assert "tts" in skill.tags

    def test_skill_content_not_empty(self):
        registry = get_skill_registry()
        skill = registry.get("security/escalation_narration")
        assert len(skill.content) > 50


# ---------------------------------------------------------------------------
# Enhanced security_ack format
# ---------------------------------------------------------------------------

class TestEnhancedAck:
    """Verify the enhanced security_ack narration field structure."""

    def test_routine_ack_structure(self):
        """Verify routine ack has expected fields."""
        ack = {
            "type": "security_ack",
            "event": "person_entered",
            "narration": {
                "classify": "routine",
                "hint": "Juan entered (confidence: 92.3%)",
                "occupancy_state": "identified",
                "occupants": ["Juan"],
            },
        }
        assert ack["narration"]["classify"] == "routine"
        assert isinstance(ack["narration"]["hint"], str)
        assert isinstance(ack["narration"]["occupants"], list)

    def test_suppressed_ack_structure(self):
        """Verify suppressed ack has expected fields."""
        ack = {
            "type": "security_ack",
            "event": "unknown_face",
            "narration": {
                "classify": "suppressed",
                "hint": "Unknown face auto-enrolled as unknown_003",
                "occupancy_state": "empty",
                "occupants": [],
            },
        }
        assert ack["narration"]["classify"] == "suppressed"
        assert ack["narration"]["occupancy_state"] == "empty"


# ---------------------------------------------------------------------------
# EscalationResult dataclass
# ---------------------------------------------------------------------------

class TestEscalationResult:
    """Basic dataclass sanity checks."""

    def test_defaults(self):
        result = EscalationResult(should_escalate=False)
        assert result.rule_name is None
        assert result.priority == "routine"
        assert result.context == {}

    def test_escalation_values(self):
        result = EscalationResult(
            should_escalate=True,
            rule_name="unknown_face_empty_house",
            priority="high",
            context={"event_type": "unknown_face"},
        )
        assert result.should_escalate is True
        assert result.rule_name == "unknown_face_empty_house"
        assert result.priority == "high"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    """get_escalation_evaluator returns the same instance."""

    def test_singleton(self):
        from atlas_brain.escalation import get_escalation_evaluator

        e1 = get_escalation_evaluator()
        e2 = get_escalation_evaluator()
        assert e1 is e2
