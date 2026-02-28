"""Tests for Behavioral Risk Sensors.

Covers the three sensor tools:
  - AlignmentSensorTool       (collaborative vs adversarial language)
  - OperationalUrgencySensorTool (planning vs reactionary temporal language)
  - NegotiationRigiditySensorTool (absolutist vs flexible negotiation language)

All tests are pure-unit (no DB, no network) and run without any external
service or async infrastructure.
"""

import pytest

from atlas_brain.tools.risk_sensors import (
    AlignmentSensorTool,
    OperationalUrgencySensorTool,
    NegotiationRigiditySensorTool,
    alignment_sensor_tool,
    operational_urgency_tool,
    negotiation_rigidity_tool,
    correlate,
    _tokenize,
    _count_terms,
    _COLLABORATIVE_TERMS,
    _ADVERSARIAL_TERMS,
    _ABSOLUTIST_TERMS,
    _FLEXIBILITY_TERMS,
    _URGENCY_TERMS,
    _PLANNING_TERMS,
)


# ---------------------------------------------------------------------------
# Helpers / tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_lowercase(self):
        tokens = _tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_handles_hyphenated_words(self):
        tokens = _tokenize("non-negotiable is bad")
        assert "non-negotiable" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = _tokenize("stop! now.")
        assert "stop" in tokens
        assert "now" in tokens


class TestCountTerms:
    def test_basic_unigram_hit(self):
        tokens = _tokenize("we work together as a team")
        count = _count_terms(tokens, _COLLABORATIVE_TERMS)
        assert count >= 2  # "we", "team"

    def test_no_match(self):
        tokens = _tokenize("the quick brown fox")
        count = _count_terms(tokens, _COLLABORATIVE_TERMS)
        assert count == 0

    def test_phrase_match(self):
        from atlas_brain.tools.risk_sensors import _ADVERSARIAL_PHRASES
        tokens = _tokenize("it is the suits who decide")
        count = _count_terms(tokens, _ADVERSARIAL_TERMS, _ADVERSARIAL_PHRASES)
        # "suits" matches as a unigram in _ADVERSARIAL_TERMS (via "the suits" phrase)
        # and "the suits" also matches as a bigram in _ADVERSARIAL_PHRASES
        assert count >= 1


# ---------------------------------------------------------------------------
# AlignmentSensorTool
# ---------------------------------------------------------------------------


class TestAlignmentSensor:
    """Tests for AlignmentSensorTool.analyze() and .execute()."""

    COLLAB_TEXT = (
        "Our team has built a strong partnership with leadership. "
        "We are aligned on our shared goals and working together collaboratively."
    )
    ADV_TEXT = (
        "They said management will never listen. "
        "The corporation only cares about money. "
        "Those executives and the board are completely out of touch."
    )

    def test_tool_name(self):
        assert alignment_sensor_tool.name == "analyze_alignment"

    def test_tool_category(self):
        assert alignment_sensor_tool.category == "analytics"

    def test_collaborative_text_not_triggered(self):
        result = alignment_sensor_tool.analyze(self.COLLAB_TEXT)
        assert result["triggered"] is False
        assert result["sensor"] == "alignment"
        assert result["collaborative_count"] > result["adversarial_count"]

    def test_adversarial_text_triggered(self):
        result = alignment_sensor_tool.analyze(self.ADV_TEXT)
        assert result["triggered"] is True
        assert result["adversarial_count"] > 0
        assert result["adversarial_share"] >= 0.30

    def test_adversarial_share_between_0_and_1(self):
        result = alignment_sensor_tool.analyze(self.ADV_TEXT)
        assert 0.0 <= result["adversarial_share"] <= 1.0

    def test_custom_threshold_lowers_bar(self):
        # With a threshold of 0.01 even mild adversarial text should trigger
        mildly_adv = "they said the company has concerns"
        result = alignment_sensor_tool.analyze(mildly_adv, threshold=0.01)
        assert result["triggered"] is True

    def test_custom_threshold_raises_bar(self):
        # With threshold=0.99 even a moderately adversarial text should not trigger
        # (equal parts collaborative and adversarial -> share ~ 0.50)
        mixed_text = "We are a strong team but they said management makes all decisions here"
        result = alignment_sensor_tool.analyze(mixed_text, threshold=0.99)
        assert result["triggered"] is False

    def test_empty_text_returns_zero_counts(self):
        # Empty text should not crash; combined == 0 -> share == 0
        result = alignment_sensor_tool.analyze("   ")
        assert result["triggered"] is False
        assert result["adversarial_share"] == 0.0

    def test_result_has_required_keys(self):
        result = alignment_sensor_tool.analyze(self.COLLAB_TEXT)
        for key in (
            "sensor", "total_words", "collaborative_count",
            "adversarial_count", "adversarial_share", "threshold",
            "triggered", "summary",
        ):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = AlignmentSensorTool()
        tr = await tool.execute({"text": self.ADV_TEXT})
        assert tr.success is True
        assert tr.data["sensor"] == "alignment"

    @pytest.mark.asyncio
    async def test_execute_missing_text(self):
        tool = AlignmentSensorTool()
        tr = await tool.execute({})
        assert tr.success is False
        assert tr.error == "MISSING_PARAMETER"

    @pytest.mark.asyncio
    async def test_execute_invalid_threshold(self):
        tool = AlignmentSensorTool()
        tr = await tool.execute({"text": self.ADV_TEXT, "threshold": -0.1})
        assert tr.success is False
        assert tr.error == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_execute_empty_text(self):
        tool = AlignmentSensorTool()
        tr = await tool.execute({"text": ""})
        assert tr.success is False
        assert tr.error == "MISSING_PARAMETER"


# ---------------------------------------------------------------------------
# OperationalUrgencySensorTool
# ---------------------------------------------------------------------------


class TestOperationalUrgencySensor:
    """Tests for OperationalUrgencySensorTool.analyze() and .execute()."""

    PLANNING_TEXT = (
        "The quarterly roadmap will focus on the long-term strategy. "
        "We have scheduled a proposal review for next quarter and plan to "
        "roll out the pipeline incrementally."
    )
    URGENCY_TEXT = (
        "Stop everything immediately! This is a critical emergency. "
        "We are completely blocked right now and the situation is urgent. "
        "Halt all shipments today -- the hub is in crisis."
    )

    def test_tool_name(self):
        assert operational_urgency_tool.name == "analyze_operational_urgency"

    def test_tool_category(self):
        assert operational_urgency_tool.category == "analytics"

    def test_planning_text_not_triggered(self):
        result = operational_urgency_tool.analyze(self.PLANNING_TEXT)
        assert result["triggered"] is False
        assert result["sensor"] == "operational_urgency"

    def test_urgency_text_triggered(self):
        result = operational_urgency_tool.analyze(self.URGENCY_TEXT)
        assert result["triggered"] is True
        assert result["density_ratio"] >= 2.0

    def test_densities_are_non_negative(self):
        result = operational_urgency_tool.analyze(self.URGENCY_TEXT)
        assert result["planning_density"] >= 0.0
        assert result["urgency_density"] >= 0.0

    def test_custom_density_factor(self):
        # Require urgency to be 10x more dense -- should not trigger on text
        # that still has planning terms to compare against.
        # (The original test used pure urgency text with no planning terms,
        # which always triggers regardless of density_factor because the
        # code deliberately fires when urgency > 0 and planning == 0.)
        mixed_text = (
            "We have scheduled a proposal for next quarter on the roadmap. "
            "There is some urgency today but the situation is not critical."
        )
        result = operational_urgency_tool.analyze(mixed_text, density_factor=10.0)
        assert result["triggered"] is False

    def test_no_planning_terms_with_urgency_triggers(self):
        # Text that has only urgency terms and no planning terms should trigger.
        pure_urgency = "stop blocked halt urgent crisis emergency now asap"
        result = operational_urgency_tool.analyze(pure_urgency, density_factor=2.0)
        assert result["triggered"] is True

    def test_result_has_required_keys(self):
        result = operational_urgency_tool.analyze(self.PLANNING_TEXT)
        for key in (
            "sensor", "total_words", "planning_count", "urgency_count",
            "planning_density", "urgency_density", "density_ratio",
            "density_factor_threshold", "triggered", "summary",
        ):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = OperationalUrgencySensorTool()
        tr = await tool.execute({"text": self.URGENCY_TEXT})
        assert tr.success is True
        assert tr.data["sensor"] == "operational_urgency"

    @pytest.mark.asyncio
    async def test_execute_missing_text(self):
        tool = OperationalUrgencySensorTool()
        tr = await tool.execute({})
        assert tr.success is False
        assert tr.error == "MISSING_PARAMETER"

    @pytest.mark.asyncio
    async def test_execute_invalid_density_factor(self):
        tool = OperationalUrgencySensorTool()
        tr = await tool.execute({"text": self.URGENCY_TEXT, "density_factor": 0})
        assert tr.success is False
        assert tr.error == "INVALID_PARAMETER"


# ---------------------------------------------------------------------------
# NegotiationRigiditySensorTool
# ---------------------------------------------------------------------------


class TestNegotiationRigiditySensor:
    """Tests for NegotiationRigiditySensorTool.analyze() and .execute()."""

    FLEXIBLE_TEXT = (
        "We are considering alternative options and exploring a creative "
        "middle ground. The team is open to suggestions and willing to "
        "revisit the proposal. Progress looks likely with flexible adjustments."
    )
    RIGID_TEXT = (
        "This is our final offer -- non-negotiable. Management must accept "
        "these demands immediately. It is absolutely impossible to consider "
        "any alternatives. Never and zero tolerance on this issue."
    )

    def test_tool_name(self):
        assert negotiation_rigidity_tool.name == "analyze_negotiation_rigidity"

    def test_tool_category(self):
        assert negotiation_rigidity_tool.category == "analytics"

    def test_flexible_text_not_triggered(self):
        result = negotiation_rigidity_tool.analyze(self.FLEXIBLE_TEXT)
        assert result["triggered"] is False
        assert result["sensor"] == "negotiation_rigidity"

    def test_rigid_text_triggered(self):
        result = negotiation_rigidity_tool.analyze(self.RIGID_TEXT)
        assert result["triggered"] is True
        assert result["absolutist_share"] >= 0.50

    def test_absolutist_share_between_0_and_1(self):
        result = negotiation_rigidity_tool.analyze(self.RIGID_TEXT)
        assert 0.0 <= result["absolutist_share"] <= 1.0

    def test_custom_threshold_lowers_bar(self):
        mildly_rigid = "must accept the final terms"
        result = negotiation_rigidity_tool.analyze(mildly_rigid, threshold=0.20)
        assert result["triggered"] is True

    def test_no_terms_returns_not_triggered(self):
        result = negotiation_rigidity_tool.analyze("the weather is fine today")
        assert result["triggered"] is False
        assert result["absolutist_share"] == 0.0

    def test_result_has_required_keys(self):
        result = negotiation_rigidity_tool.analyze(self.FLEXIBLE_TEXT)
        for key in (
            "sensor", "total_words", "absolutist_count", "flexibility_count",
            "absolutist_share", "threshold", "triggered", "summary",
        ):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = NegotiationRigiditySensorTool()
        tr = await tool.execute({"text": self.RIGID_TEXT})
        assert tr.success is True
        assert tr.data["sensor"] == "negotiation_rigidity"

    @pytest.mark.asyncio
    async def test_execute_missing_text(self):
        tool = NegotiationRigiditySensorTool()
        tr = await tool.execute({})
        assert tr.success is False
        assert tr.error == "MISSING_PARAMETER"

    @pytest.mark.asyncio
    async def test_execute_invalid_threshold(self):
        tool = NegotiationRigiditySensorTool()
        tr = await tool.execute({"text": self.RIGID_TEXT, "threshold": 1.5})
        assert tr.success is False
        assert tr.error == "INVALID_PARAMETER"

    @pytest.mark.asyncio
    async def test_phrase_detection(self):
        """Non-negotiable as a phrase (bigram) should be detected."""
        tool = NegotiationRigiditySensorTool()
        tr = await tool.execute({"text": "This position is non-negotiable."})
        assert tr.success is True
        # "non-negotiable" is in _ABSOLUTIST_TERMS (as a hyphenated unigram)
        assert tr.data["absolutist_count"] >= 1


# ---------------------------------------------------------------------------
# Registration in tool_registry
# ---------------------------------------------------------------------------


class TestRegistration:
    """Verify the sensors are registered in the global tool registry."""

    def test_alignment_sensor_registered(self):
        from atlas_brain.tools.registry import tool_registry
        tool = tool_registry.get("analyze_alignment")
        assert tool is not None
        assert tool.name == "analyze_alignment"

    def test_operational_urgency_registered(self):
        from atlas_brain.tools.registry import tool_registry
        tool = tool_registry.get("analyze_operational_urgency")
        assert tool is not None

    def test_negotiation_rigidity_registered(self):
        from atlas_brain.tools.registry import tool_registry
        tool = tool_registry.get("analyze_negotiation_rigidity")
        assert tool is not None

    def test_all_sensors_have_parameters(self):
        for sensor in (
            alignment_sensor_tool,
            operational_urgency_tool,
            negotiation_rigidity_tool,
        ):
            assert len(sensor.parameters) >= 1
            assert any(p.name == "text" for p in sensor.parameters)


# ---------------------------------------------------------------------------
# correlate() -- cross-sensor relationship detection
# ---------------------------------------------------------------------------


class TestCorrelate:
    """Tests for the correlate() function that connects sensor outputs."""

    # Pre-built result stubs -- real sensor output shapes, values chosen to
    # control triggered=True/False without running full text analysis.

    def _make(self, sensor: str, triggered: bool) -> dict:
        return {"sensor": sensor, "triggered": triggered, "summary": "stub"}

    def test_no_sensors_triggered_low_risk(self):
        result = correlate(
            self._make("alignment", False),
            self._make("operational_urgency", False),
            self._make("negotiation_rigidity", False),
        )
        assert result["composite_risk_level"] == "LOW"
        assert result["sensor_count"] == 0
        assert result["triggered_sensors"] == []
        assert result["relationship_count"] == 0

    def test_one_sensor_triggered_medium_risk(self):
        result = correlate(
            self._make("alignment", True),
            self._make("operational_urgency", False),
            self._make("negotiation_rigidity", False),
        )
        assert result["composite_risk_level"] == "MEDIUM"
        assert result["sensor_count"] == 1
        assert result["triggered_sensors"] == ["alignment"]
        assert result["relationship_count"] == 0

    def test_two_sensors_triggered_high_risk(self):
        result = correlate(
            self._make("alignment", True),
            self._make("operational_urgency", False),
            self._make("negotiation_rigidity", True),
        )
        assert result["composite_risk_level"] == "HIGH"
        assert result["sensor_count"] == 2
        assert "negotiation_rigidity" in result["triggered_sensors"]
        assert "alignment" in result["triggered_sensors"]

    def test_alignment_rigidity_pattern_detected(self):
        result = correlate(
            self._make("alignment", True),
            self._make("operational_urgency", False),
            self._make("negotiation_rigidity", True),
        )
        labels = [r["label"] for r in result["relationships"]]
        assert "adversarial_rigidity" in labels

    def test_alignment_urgency_pattern_detected(self):
        result = correlate(
            self._make("alignment", True),
            self._make("operational_urgency", True),
            self._make("negotiation_rigidity", False),
        )
        labels = [r["label"] for r in result["relationships"]]
        assert "adversarial_reactivity" in labels

    def test_urgency_rigidity_pattern_detected(self):
        result = correlate(
            self._make("alignment", False),
            self._make("operational_urgency", True),
            self._make("negotiation_rigidity", True),
        )
        labels = [r["label"] for r in result["relationships"]]
        assert "reactive_lock" in labels

    def test_all_three_triggered_critical_risk(self):
        result = correlate(
            self._make("alignment", True),
            self._make("operational_urgency", True),
            self._make("negotiation_rigidity", True),
        )
        assert result["composite_risk_level"] == "CRITICAL"
        assert result["sensor_count"] == 3
        labels = [r["label"] for r in result["relationships"]]
        assert "full_friction_cascade" in labels
        # All pair patterns also present
        assert "adversarial_rigidity" in labels
        assert "adversarial_reactivity" in labels
        assert "reactive_lock" in labels

    def test_relationship_has_required_keys(self):
        result = correlate(
            self._make("alignment", True),
            self._make("operational_urgency", True),
            self._make("negotiation_rigidity", True),
        )
        for rel in result["relationships"]:
            assert "label" in rel
            assert "sensors" in rel
            assert "insight" in rel
            assert isinstance(rel["sensors"], list)
            assert len(rel["insight"]) > 0

    def test_result_has_required_keys(self):
        result = correlate(
            self._make("alignment", False),
            self._make("operational_urgency", False),
            self._make("negotiation_rigidity", False),
        )
        for key in (
            "triggered_sensors", "sensor_count", "composite_risk_level",
            "relationships", "relationship_count", "summary",
        ):
            assert key in result, f"Missing key: {key}"

    def test_end_to_end_with_real_sensors(self):
        """Run all three real sensors on high-friction text and correlate."""
        text = (
            "They are demanding we stop everything immediately -- "
            "non-negotiable, final offer. Management must accept now. "
            "The corporation will never budge. Crisis. Halt all operations."
        )
        a = alignment_sensor_tool.analyze(text)
        u = operational_urgency_tool.analyze(text)
        r = negotiation_rigidity_tool.analyze(text)
        result = correlate(a, u, r)
        # All three sensors should fire on this extreme text
        assert result["composite_risk_level"] == "CRITICAL"
        assert result["sensor_count"] == 3
        assert "full_friction_cascade" in [
            rel["label"] for rel in result["relationships"]
        ]

    def test_end_to_end_low_friction_text(self):
        """Run all three real sensors on calm collaborative text and correlate."""
        text = (
            "Our team is planning for next quarter. We will propose a roadmap "
            "and explore alternative options together. Partners are aligned."
        )
        a = alignment_sensor_tool.analyze(text)
        u = operational_urgency_tool.analyze(text)
        r = negotiation_rigidity_tool.analyze(text)
        result = correlate(a, u, r)
        assert result["composite_risk_level"] in ("LOW", "MEDIUM")
        assert result["relationship_count"] == 0
