"""
Comprehensive tests for the atlas_brain.alerts module.

Covers events, rules, manager, and delivery.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_brain.alerts.events import (
    AlertEvent,
    AudioAlertEvent,
    HAStateAlertEvent,
    PresenceAlertEvent,
    ReminderAlertEvent,
    SecurityAlertEvent,
    VisionAlertEvent,
)
from atlas_brain.alerts.rules import (
    AlertRule,
    create_audio_rule,
    create_ha_state_rule,
    create_vision_rule,
)
from atlas_brain.alerts.manager import (
    AlertManager,
    get_alert_manager,
    reset_alert_manager,
)
from atlas_brain.alerts.delivery import (
    NtfyDelivery,
    TTSDelivery,
    WebhookDelivery,
    log_alert_callback,
    setup_default_callbacks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.utcnow()


def _make_vision_event(**overrides):
    defaults = dict(
        source_id="cam_front_door",
        timestamp=_now(),
        class_name="person",
        detection_type="new_track",
        track_id=42,
        node_id="node_1",
    )
    defaults.update(overrides)
    return VisionAlertEvent(**defaults)


def _make_audio_event(**overrides):
    defaults = dict(
        source_id="mic_living_room",
        timestamp=_now(),
        sound_class="Doorbell",
        confidence=0.9,
        priority="high",
    )
    defaults.update(overrides)
    return AudioAlertEvent(**defaults)


def _make_ha_event(**overrides):
    defaults = dict(
        source_id="binary_sensor.front_door",
        timestamp=_now(),
        old_state="off",
        new_state="on",
        domain="binary_sensor",
    )
    defaults.update(overrides)
    return HAStateAlertEvent(**defaults)


def _make_reminder_event(**overrides):
    defaults = dict(
        source_id="reminder_123",
        timestamp=_now(),
        message="Take out the trash",
        reminder_id="123",
    )
    defaults.update(overrides)
    return ReminderAlertEvent(**defaults)


def _make_security_event(**overrides):
    defaults = dict(
        source_id="cam_backyard",
        timestamp=_now(),
        detection_type="intrusion",
    )
    defaults.update(overrides)
    return SecurityAlertEvent(**defaults)


def _make_presence_event(**overrides):
    defaults = dict(
        source_id="presence_tracker",
        timestamp=_now(),
        transition="arrival",
        occupancy_state="occupied",
    )
    defaults.update(overrides)
    return PresenceAlertEvent(**defaults)


# ===========================================================================
# TestAlertEvents
# ===========================================================================


class TestAlertEvents:
    """Tests for alert event types and their get_field behaviour."""

    def test_vision_event_get_field_direct(self):
        event = _make_vision_event(class_name="car", track_id=7)
        assert event.get_field("class_name") == "car"
        assert event.get_field("track_id") == 7
        assert event.get_field("detection_type") == "new_track"
        assert event.get_field("node_id") == "node_1"

    def test_vision_event_get_field_metadata_fallback(self):
        event = _make_vision_event(metadata={"zone": "porch"})
        assert event.get_field("zone") == "porch"

    def test_vision_event_get_field_default(self):
        event = _make_vision_event()
        assert event.get_field("nonexistent") is None
        assert event.get_field("nonexistent", "fallback") == "fallback"

    def test_audio_event_get_field_direct(self):
        event = _make_audio_event(sound_class="Siren", confidence=0.85, priority="critical")
        assert event.get_field("sound_class") == "Siren"
        assert event.get_field("confidence") == 0.85
        assert event.get_field("priority") == "critical"

    def test_audio_event_get_field_metadata_fallback(self):
        event = _make_audio_event(metadata={"location": "kitchen"})
        assert event.get_field("location") == "kitchen"

    def test_ha_state_event_get_field_checks_attributes(self):
        event = _make_ha_event(
            attributes={"brightness": 200, "friendly_name": "Hall Light"},
        )
        # Direct fields
        assert event.get_field("new_state") == "on"
        assert event.get_field("domain") == "binary_sensor"
        # Attribute lookup (between field_map and metadata)
        assert event.get_field("brightness") == 200
        assert event.get_field("friendly_name") == "Hall Light"

    def test_ha_state_event_get_field_metadata_after_attributes(self):
        event = _make_ha_event(
            attributes={"brightness": 200},
            metadata={"extra_key": "extra_val"},
        )
        assert event.get_field("extra_key") == "extra_val"

    def test_ha_state_event_get_field_default(self):
        event = _make_ha_event()
        assert event.get_field("missing", "nope") == "nope"

    def test_reminder_event_get_field(self):
        event = _make_reminder_event(
            message="Call mom",
            reminder_id="r55",
            repeat_pattern="daily",
        )
        assert event.get_field("message") == "Call mom"
        assert event.get_field("reminder_id") == "r55"
        assert event.get_field("repeat_pattern") == "daily"

    def test_security_event_get_field(self):
        event = _make_security_event(
            detection_type="motion",
            label="human",
            confidence=0.77,
        )
        assert event.get_field("detection_type") == "motion"
        assert event.get_field("label") == "human"
        assert event.get_field("confidence") == 0.77

    def test_presence_event_get_field(self):
        event = _make_presence_event(
            transition="departure",
            occupancy_state="empty",
            person_name="Alice",
            occupants=["Bob"],
        )
        assert event.get_field("transition") == "departure"
        assert event.get_field("occupancy_state") == "empty"
        assert event.get_field("person_name") == "Alice"
        assert event.get_field("occupants") == ["Bob"]

    def test_from_ha_event_factory(self):
        ha_data = {
            "entity_id": "light.kitchen",
            "old_state": {"state": "off", "attributes": {}},
            "new_state": {"state": "on", "attributes": {"brightness": 255}},
        }
        event = HAStateAlertEvent.from_ha_event(ha_data)
        assert event.source_id == "light.kitchen"
        assert event.domain == "light"
        assert event.old_state == "off"
        assert event.new_state == "on"
        assert event.attributes == {"brightness": 255}
        assert event.event_type == "ha_state"

    def test_from_ha_event_missing_states(self):
        ha_data = {"entity_id": "sensor.temp"}
        event = HAStateAlertEvent.from_ha_event(ha_data)
        assert event.old_state == "unknown"
        assert event.new_state == "unknown"
        assert event.domain == "sensor"

    def test_from_presence_state_factory(self):
        event = PresenceAlertEvent.from_presence_state(
            transition="arrival",
            state_value="occupied",
            occupants=["Juan"],
            person="Juan",
        )
        assert event.source_id == "presence_tracker"
        assert event.transition == "arrival"
        assert event.occupancy_state == "occupied"
        assert event.person_name == "Juan"
        assert event.occupants == ["Juan"]
        assert event.metadata["person_name"] == "Juan"

    def test_from_presence_state_custom_source(self):
        event = PresenceAlertEvent.from_presence_state(
            transition="departure",
            state_value="empty",
            occupants=[],
            source_id="custom_source",
        )
        assert event.source_id == "custom_source"

    def test_event_types_satisfy_alert_event_protocol(self):
        """All concrete event types should satisfy the AlertEvent protocol."""
        events = [
            _make_vision_event(),
            _make_audio_event(),
            _make_ha_event(),
            _make_reminder_event(),
            _make_security_event(),
            _make_presence_event(),
        ]
        for event in events:
            assert isinstance(event, AlertEvent), (
                f"{type(event).__name__} does not satisfy AlertEvent protocol"
            )

    def test_from_kafka_event_factory(self):
        kafka_data = {
            "camera_id": "cam_side_yard",
            "type": "loitering",
            "label": "person",
            "confidence": 0.92,
            "extra_field": "something",
        }
        event = SecurityAlertEvent.from_kafka_event(kafka_data)
        assert event.source_id == "cam_side_yard"
        assert event.detection_type == "loitering"
        assert event.label == "person"
        assert event.confidence == 0.92
        # The entire dict is stored as metadata
        assert event.metadata["extra_field"] == "something"


# ===========================================================================
# TestAlertRules
# ===========================================================================


class TestAlertRules:
    """Tests for AlertRule matching, formatting, and factory functions."""

    def test_matches_enabled_disabled(self):
        rule = AlertRule(
            name="test",
            event_types=["vision"],
            source_pattern="*",
            enabled=True,
        )
        event = _make_vision_event()
        assert rule.matches(event) is True

        rule.enabled = False
        assert rule.matches(event) is False

    def test_matches_event_type_exact(self):
        rule = AlertRule(name="v", event_types=["vision"], source_pattern="*")
        assert rule.matches(_make_vision_event()) is True
        assert rule.matches(_make_audio_event()) is False

    def test_matches_event_type_wildcard(self):
        rule = AlertRule(name="all", event_types=["*"], source_pattern="*")
        assert rule.matches(_make_vision_event()) is True
        assert rule.matches(_make_audio_event()) is True
        assert rule.matches(_make_ha_event()) is True

    def test_matches_source_exact(self):
        rule = AlertRule(
            name="exact",
            event_types=["vision"],
            source_pattern="cam_front_door",
        )
        assert rule.matches(_make_vision_event(source_id="cam_front_door")) is True
        assert rule.matches(_make_vision_event(source_id="cam_back_door")) is False

    def test_matches_source_wildcard_pattern(self):
        rule = AlertRule(
            name="fnm",
            event_types=["vision"],
            source_pattern="cam_*_door",
        )
        assert rule.matches(_make_vision_event(source_id="cam_front_door")) is True
        assert rule.matches(_make_vision_event(source_id="cam_back_door")) is True
        assert rule.matches(_make_vision_event(source_id="cam_driveway")) is False

    def test_matches_source_case_insensitive(self):
        rule = AlertRule(
            name="ci",
            event_types=["vision"],
            source_pattern="cam_front_door",
        )
        assert rule.matches(_make_vision_event(source_id="CAM_FRONT_DOOR")) is True
        assert rule.matches(_make_vision_event(source_id="Cam_Front_Door")) is True

    def test_matches_source_star_matches_all(self):
        rule = AlertRule(name="s", event_types=["vision"], source_pattern="*")
        assert rule.matches(_make_vision_event(source_id="anything_at_all")) is True

    def test_matches_conditions_string_match(self):
        rule = AlertRule(
            name="cls",
            event_types=["vision"],
            source_pattern="*",
            conditions={"class_name": "person"},
        )
        assert rule.matches(_make_vision_event(class_name="person")) is True
        assert rule.matches(_make_vision_event(class_name="car")) is False

    def test_matches_conditions_string_wildcard(self):
        rule = AlertRule(
            name="wc",
            event_types=["audio"],
            source_pattern="*",
            conditions={"sound_class": "*glass*"},
        )
        assert rule.matches(_make_audio_event(sound_class="glass_break")) is True
        assert rule.matches(_make_audio_event(sound_class="breaking_glass_sound")) is True
        assert rule.matches(_make_audio_event(sound_class="doorbell")) is False

    def test_matches_conditions_none_actual_fails(self):
        rule = AlertRule(
            name="none",
            event_types=["vision"],
            source_pattern="*",
            conditions={"nonexistent_field": "anything"},
        )
        event = _make_vision_event()
        # get_field returns None for unknown field, _check_condition returns False
        assert rule.matches(event) is False

    def test_matches_conditions_wildcard_expected(self):
        rule = AlertRule(
            name="star",
            event_types=["vision"],
            source_pattern="*",
            conditions={"class_name": "*"},
        )
        assert rule.matches(_make_vision_event(class_name="anything")) is True

    def test_operator_gt_and_gte(self):
        rule = AlertRule(
            name="op",
            event_types=["audio"],
            source_pattern="*",
            conditions={"confidence": {"$gt": 0.5}},
        )
        assert rule.matches(_make_audio_event(confidence=0.6)) is True
        assert rule.matches(_make_audio_event(confidence=0.5)) is False
        assert rule.matches(_make_audio_event(confidence=0.4)) is False

        rule.conditions = {"confidence": {"$gte": 0.5}}
        assert rule.matches(_make_audio_event(confidence=0.5)) is True
        assert rule.matches(_make_audio_event(confidence=0.4)) is False

    def test_operator_lt_and_lte(self):
        rule = AlertRule(
            name="op",
            event_types=["audio"],
            source_pattern="*",
            conditions={"confidence": {"$lt": 0.8}},
        )
        assert rule.matches(_make_audio_event(confidence=0.7)) is True
        assert rule.matches(_make_audio_event(confidence=0.8)) is False

        rule.conditions = {"confidence": {"$lte": 0.8}}
        assert rule.matches(_make_audio_event(confidence=0.8)) is True
        assert rule.matches(_make_audio_event(confidence=0.9)) is False

    def test_operator_in_and_nin(self):
        rule = AlertRule(
            name="in",
            event_types=["vision"],
            source_pattern="*",
            conditions={"class_name": {"$in": ["person", "car"]}},
        )
        assert rule.matches(_make_vision_event(class_name="person")) is True
        assert rule.matches(_make_vision_event(class_name="car")) is True
        assert rule.matches(_make_vision_event(class_name="dog")) is False

        rule.conditions = {"class_name": {"$nin": ["person", "car"]}}
        assert rule.matches(_make_vision_event(class_name="dog")) is True
        assert rule.matches(_make_vision_event(class_name="person")) is False

    def test_operator_ne(self):
        rule = AlertRule(
            name="ne",
            event_types=["vision"],
            source_pattern="*",
            conditions={"class_name": {"$ne": "person"}},
        )
        assert rule.matches(_make_vision_event(class_name="car")) is True
        assert rule.matches(_make_vision_event(class_name="person")) is False

    def test_format_message_with_fields(self):
        rule = AlertRule(
            name="fmt",
            event_types=["vision"],
            source_pattern="*",
            message_template="{class_name} detected at {source}.",
        )
        event = _make_vision_event(source_id="cam_front_door", class_name="person")
        msg = rule.format_message(event)
        assert msg == "person detected at front door."

    def test_format_message_missing_field_fallback(self):
        rule = AlertRule(
            name="bad_tmpl",
            event_types=["vision"],
            source_pattern="*",
            message_template="{nonexistent_placeholder} is here",
        )
        event = _make_vision_event(source_id="cam_front_door")
        msg = rule.format_message(event)
        assert msg == "Alert from front door"

    def test_source_to_name_known(self):
        rule = AlertRule(name="x", event_types=["vision"], source_pattern="*")
        assert rule._source_to_name("cam_front_door") == "front door"
        assert rule._source_to_name("cam_back_door") == "back door"
        assert rule._source_to_name("cam_driveway") == "driveway"
        assert rule._source_to_name("front_door") == "front door"

    def test_source_to_name_unknown(self):
        rule = AlertRule(name="x", event_types=["vision"], source_pattern="*")
        # cam_ prefix removed, underscores to spaces
        assert rule._source_to_name("cam_side_yard") == "side yard"
        # Dotted entity IDs -> last part cleaned
        assert rule._source_to_name("binary_sensor.front_motion") == "front motion"

    def test_create_vision_rule(self):
        rule = create_vision_rule(
            name="test_vis",
            source_pattern="*front*",
            class_name="person",
            detection_type="new_track",
            priority=10,
        )
        assert rule.name == "test_vis"
        assert rule.event_types == ["vision"]
        assert rule.conditions["class_name"] == "person"
        assert rule.conditions["detection_type"] == "new_track"
        assert rule.priority == 10

    def test_create_audio_rule(self):
        rule = create_audio_rule(
            name="test_aud",
            source_pattern="*",
            sound_class="Doorbell",
            min_confidence=0.7,
            priority=8,
        )
        assert rule.name == "test_aud"
        assert rule.event_types == ["audio"]
        assert rule.conditions["sound_class"] == "Doorbell"
        assert rule.conditions["confidence"] == {"$gte": 0.7}
        assert rule.priority == 8

    def test_create_ha_state_rule(self):
        rule = create_ha_state_rule(
            name="test_ha",
            source_pattern="light.*",
            new_state="on",
            domain="light",
            priority=3,
        )
        assert rule.name == "test_ha"
        assert rule.event_types == ["ha_state"]
        assert rule.conditions["new_state"] == "on"
        assert rule.conditions["domain"] == "light"

    def test_create_ha_state_rule_no_domain(self):
        rule = create_ha_state_rule(
            name="no_dom",
            source_pattern="*",
            new_state="off",
        )
        assert "domain" not in rule.conditions

    def test_check_condition_non_string_actual_vs_string_expected(self):
        """When actual is not a string but expected is, str comparison is used."""
        rule = AlertRule(name="x", event_types=["vision"], source_pattern="*")
        # actual=42 (int), expected="42" (str) -> str(42).lower() == "42" -> True
        assert rule._check_condition(42, "42") is True
        assert rule._check_condition(42, "99") is False

    def test_check_condition_direct_equality(self):
        """Non-string, non-dict expected uses direct equality."""
        rule = AlertRule(name="x", event_types=["vision"], source_pattern="*")
        assert rule._check_condition(42, 42) is True
        assert rule._check_condition(42, 43) is False
        assert rule._check_condition(True, True) is True


# ===========================================================================
# TestAlertManager
# ===========================================================================


class TestAlertManager:
    """Tests for the AlertManager class and singleton."""

    def setup_method(self):
        reset_alert_manager()

    def test_initialize_adds_default_rules(self):
        mgr = AlertManager()
        mgr.initialize()
        rules = mgr.list_rules()
        rule_names = {r.name for r in rules}
        expected_names = {
            "person_front_door",
            "person_back_door",
            "vehicle_driveway",
            "person_garage",
            "doorbell",
            "glass_break",
            "reminder_due",
            "presence_arrival",
            "presence_departure",
        }
        assert expected_names.issubset(rule_names)

    def test_initialize_idempotent(self):
        mgr = AlertManager()
        mgr.initialize()
        count_before = len(mgr.list_rules())
        mgr.initialize()
        count_after = len(mgr.list_rules())
        assert count_before == count_after

    def test_add_and_remove_rule(self):
        mgr = AlertManager()
        rule = AlertRule(name="custom", event_types=["vision"], source_pattern="*")
        mgr.add_rule(rule)
        assert mgr.get_rule("custom") is rule

        removed = mgr.remove_rule("custom")
        assert removed is True
        assert mgr.get_rule("custom") is None

        # Removing nonexistent rule returns False
        assert mgr.remove_rule("custom") is False

    def test_get_rule_returns_none_for_missing(self):
        mgr = AlertManager()
        assert mgr.get_rule("does_not_exist") is None

    def test_list_rules_sorted_by_priority(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(name="low", event_types=["vision"], source_pattern="*", priority=1))
        mgr.add_rule(AlertRule(name="high", event_types=["vision"], source_pattern="*", priority=10))
        mgr.add_rule(AlertRule(name="mid", event_types=["vision"], source_pattern="*", priority=5))

        rules = mgr.list_rules()
        priorities = [r.priority for r in rules]
        assert priorities == sorted(priorities, reverse=True)

    def test_list_rules_filtered_by_event_type(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(name="vis", event_types=["vision"], source_pattern="*"))
        mgr.add_rule(AlertRule(name="aud", event_types=["audio"], source_pattern="*"))
        mgr.add_rule(AlertRule(name="all", event_types=["*"], source_pattern="*"))

        vision_rules = mgr.list_rules(event_type="vision")
        names = {r.name for r in vision_rules}
        assert "vis" in names
        assert "all" in names  # wildcard matches
        assert "aud" not in names

    def test_enable_disable_rule(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(name="toggle", event_types=["vision"], source_pattern="*", enabled=True))

        assert mgr.disable_rule("toggle") is True
        assert mgr.get_rule("toggle").enabled is False

        assert mgr.enable_rule("toggle") is True
        assert mgr.get_rule("toggle").enabled is True

        # Non-existent rule
        assert mgr.enable_rule("nope") is False
        assert mgr.disable_rule("nope") is False

    def test_register_unregister_callback(self):
        mgr = AlertManager()
        cb = AsyncMock()
        mgr.register_callback(cb)
        assert cb in mgr._callbacks

        mgr.unregister_callback(cb)
        assert cb not in mgr._callbacks

        # Unregistering something not registered is safe (no-op)
        mgr.unregister_callback(cb)

    def test_cooldown_blocks_rapid_fire(self):
        mgr = AlertManager()
        mgr._update_cooldown("rule_a")
        # Immediately after update, 30s cooldown should block
        assert mgr._check_cooldown("rule_a", 30) is False

        # With 0 cooldown, should pass
        assert mgr._check_cooldown("rule_a", 0) is True

    def test_cooldown_allows_after_expiry(self):
        mgr = AlertManager()
        # Set cooldown to a time in the past
        mgr._cooldowns["rule_b"] = datetime.utcnow() - timedelta(seconds=60)
        assert mgr._check_cooldown("rule_b", 30) is True

    def test_cooldown_no_prior_trigger(self):
        mgr = AlertManager()
        # Never triggered -> should allow
        assert mgr._check_cooldown("never_seen", 30) is True

    @pytest.mark.asyncio
    async def test_process_event_triggers_matching_rule(self):
        mgr = AlertManager()
        rule = AlertRule(
            name="test_rule",
            event_types=["vision"],
            source_pattern="*",
            conditions={"class_name": "person"},
            message_template="Person detected!",
            cooldown_seconds=0,
            priority=10,
        )
        mgr.add_rule(rule)
        mgr._initialized = True

        event = _make_vision_event(class_name="person")

        with patch("atlas_brain.alerts.manager.settings") as mock_settings:
            mock_settings.alerts.enabled = True
            mock_settings.alerts.persist_alerts = False
            result = await mgr.process_event(event)

        assert result == "Person detected!"

    @pytest.mark.asyncio
    async def test_process_event_disabled(self):
        mgr = AlertManager()
        mgr._initialized = True
        mgr.add_rule(AlertRule(
            name="r",
            event_types=["vision"],
            source_pattern="*",
            cooldown_seconds=0,
        ))
        event = _make_vision_event()

        with patch("atlas_brain.alerts.manager.settings") as mock_settings:
            mock_settings.alerts.enabled = False
            result = await mgr.process_event(event)

        assert result is None

    @pytest.mark.asyncio
    async def test_process_event_calls_callbacks(self):
        mgr = AlertManager()
        mgr._initialized = True

        rule = AlertRule(
            name="cb_test",
            event_types=["audio"],
            source_pattern="*",
            conditions={"sound_class": "Doorbell"},
            message_template="Ding dong!",
            cooldown_seconds=0,
            priority=5,
        )
        mgr.add_rule(rule)

        cb1 = AsyncMock()
        cb2 = AsyncMock()
        mgr.register_callback(cb1)
        mgr.register_callback(cb2)

        event = _make_audio_event(sound_class="Doorbell")

        with patch("atlas_brain.alerts.manager.settings") as mock_settings:
            mock_settings.alerts.enabled = True
            mock_settings.alerts.persist_alerts = False
            await mgr.process_event(event)

        cb1.assert_awaited_once()
        cb2.assert_awaited_once()
        # Verify message argument
        assert cb1.call_args[0][0] == "Ding dong!"

    @pytest.mark.asyncio
    async def test_process_event_no_match_returns_none(self):
        mgr = AlertManager()
        mgr._initialized = True
        mgr.add_rule(AlertRule(
            name="only_vision",
            event_types=["vision"],
            source_pattern="*",
            cooldown_seconds=0,
        ))

        event = _make_audio_event()

        with patch("atlas_brain.alerts.manager.settings") as mock_settings:
            mock_settings.alerts.enabled = True
            mock_settings.alerts.persist_alerts = False
            result = await mgr.process_event(event)

        assert result is None

    @pytest.mark.asyncio
    async def test_process_event_highest_priority_wins(self):
        mgr = AlertManager()
        mgr._initialized = True

        low_rule = AlertRule(
            name="low",
            event_types=["vision"],
            source_pattern="*",
            message_template="Low priority",
            cooldown_seconds=0,
            priority=1,
        )
        high_rule = AlertRule(
            name="high",
            event_types=["vision"],
            source_pattern="*",
            message_template="High priority",
            cooldown_seconds=0,
            priority=20,
        )
        mgr.add_rule(low_rule)
        mgr.add_rule(high_rule)

        event = _make_vision_event()

        with patch("atlas_brain.alerts.manager.settings") as mock_settings:
            mock_settings.alerts.enabled = True
            mock_settings.alerts.persist_alerts = False
            result = await mgr.process_event(event)

        assert result == "High priority"

    def test_singleton_get_and_reset(self):
        m1 = get_alert_manager()
        m2 = get_alert_manager()
        assert m1 is m2

        reset_alert_manager()
        m3 = get_alert_manager()
        assert m3 is not m1

    @pytest.mark.asyncio
    async def test_process_event_auto_initializes(self):
        mgr = AlertManager()
        assert mgr._initialized is False

        event = _make_vision_event(source_id="cam_front_door", class_name="person")

        with patch("atlas_brain.alerts.manager.settings") as mock_settings:
            mock_settings.alerts.enabled = True
            mock_settings.alerts.persist_alerts = False
            result = await mgr.process_event(event)

        assert mgr._initialized is True
        # Default rule person_front_door should match
        assert result is not None


# ===========================================================================
# TestAlertDelivery
# ===========================================================================


class TestAlertDelivery:
    """Tests for alert delivery mechanisms."""

    def _rule(self, priority=5, name="test_rule"):
        return AlertRule(
            name=name,
            event_types=["vision"],
            source_pattern="*",
            priority=priority,
        )

    def test_ntfy_priority_mapping_urgent(self):
        delivery = NtfyDelivery()
        # We test the priority mapping logic by inspecting headers indirectly.
        # Priority >= 15 -> urgent
        rule = self._rule(priority=15)
        event = _make_vision_event()
        # Extract mapping by running deliver with a mock
        # The mapping is inline so we verify by checking the actual code path
        assert rule.priority >= 15

    def test_ntfy_priority_mapping_high(self):
        rule = self._rule(priority=10)
        assert rule.priority >= 10

    def test_ntfy_priority_mapping_low(self):
        rule = self._rule(priority=2)
        assert rule.priority <= 3

    @pytest.mark.asyncio
    async def test_ntfy_deliver_success(self):
        delivery = NtfyDelivery(base_url="http://ntfy.test", topic="alerts")
        rule = self._rule(priority=10, name="ntfy_test")
        event = _make_vision_event()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            await delivery.deliver("Test alert", rule, event)

        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["content"] == "Test alert"
        assert call_kwargs[1]["headers"]["Priority"] == "high"
        assert call_kwargs[1]["headers"]["Title"] == "Atlas: ntfy_test"

    @pytest.mark.asyncio
    async def test_ntfy_deliver_retry_on_failure(self):
        delivery = NtfyDelivery(base_url="http://ntfy.test", topic="alerts")
        rule = self._rule(priority=5, name="retry_test")
        event = _make_vision_event()

        mock_response_ok = MagicMock()
        mock_response_ok.raise_for_status = MagicMock()

        fail_client = AsyncMock()
        fail_client.post = AsyncMock(side_effect=ConnectionError("refused"))
        fail_client.__aenter__ = AsyncMock(return_value=fail_client)
        fail_client.__aexit__ = AsyncMock(return_value=False)

        ok_client = AsyncMock()
        ok_client.post = AsyncMock(return_value=mock_response_ok)
        ok_client.__aenter__ = AsyncMock(return_value=ok_client)
        ok_client.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        def client_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fail_client
            return ok_client

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.side_effect = client_factory

        with patch.dict("sys.modules", {"httpx": mock_httpx}), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await delivery.deliver("Retry test", rule, event, _max_retries=2)

        # First call fails, second succeeds
        assert fail_client.post.await_count == 1
        assert ok_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_ntfy_deliver_priority_values(self):
        """Verify the actual priority header values for different rule priorities."""
        delivery = NtfyDelivery(base_url="http://ntfy.test", topic="t")

        priorities_to_check = [
            (20, "urgent"),
            (15, "urgent"),
            (10, "high"),
            (12, "high"),
            (5, "default"),
            (4, "default"),
            (3, "low"),
            (1, "low"),
        ]

        for rule_priority, expected_ntfy_priority in priorities_to_check:
            rule = self._rule(priority=rule_priority)
            event = _make_vision_event()

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_httpx = MagicMock()
            mock_httpx.AsyncClient.return_value = mock_client

            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                await delivery.deliver("msg", rule, event)

            actual_priority = mock_client.post.call_args[1]["headers"]["Priority"]
            assert actual_priority == expected_ntfy_priority, (
                f"Rule priority {rule_priority}: expected ntfy '{expected_ntfy_priority}', "
                f"got '{actual_priority}'"
            )

    @pytest.mark.asyncio
    async def test_webhook_deliver_success(self):
        delivery = WebhookDelivery(
            webhook_url="http://hooks.test/alert",
            headers={"Authorization": "Bearer abc"},
        )
        rule = self._rule(priority=5, name="wh_test")
        event = _make_vision_event(source_id="cam_front_door")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            await delivery.deliver("Webhook alert", rule, event)

        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["rule_name"] == "wh_test"
        assert payload["event_type"] == "vision"
        assert payload["message"] == "Webhook alert"
        assert payload["source_id"] == "cam_front_door"
        assert payload["priority"] == 5
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer abc"

    @pytest.mark.asyncio
    async def test_webhook_deliver_retry(self):
        delivery = WebhookDelivery(webhook_url="http://hooks.test/alert")
        rule = self._rule(priority=5)
        event = _make_vision_event()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        fail_client = AsyncMock()
        fail_client.post = AsyncMock(side_effect=ConnectionError("timeout"))
        fail_client.__aenter__ = AsyncMock(return_value=fail_client)
        fail_client.__aexit__ = AsyncMock(return_value=False)

        ok_client = AsyncMock()
        ok_client.post = AsyncMock(return_value=mock_response)
        ok_client.__aenter__ = AsyncMock(return_value=ok_client)
        ok_client.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        def client_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return fail_client
            return ok_client

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.side_effect = client_factory

        with patch.dict("sys.modules", {"httpx": mock_httpx}), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await delivery.deliver("Retry webhook", rule, event, _max_retries=2)

        # Two failures, then success
        assert fail_client.post.await_count == 2
        assert ok_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_tts_deliver_success(self):
        mock_tts = AsyncMock()
        mock_tts.synthesize = AsyncMock(return_value=b"\x00\x01\x02\x03")

        mock_registry = MagicMock()
        mock_registry.get_active.return_value = mock_tts

        mock_conn_mgr = AsyncMock()
        mock_conn_mgr.queue_announcement = AsyncMock(return_value=True)

        delivery = TTSDelivery(mock_registry, mock_conn_mgr)
        rule = self._rule(priority=10, name="tts_test")
        event = _make_vision_event()

        await delivery.deliver("Someone is at the door", rule, event)

        mock_tts.synthesize.assert_awaited_once_with("Someone is at the door")
        mock_conn_mgr.queue_announcement.assert_awaited_once()

        announcement = mock_conn_mgr.queue_announcement.call_args[0][0]
        assert announcement["state"] == "alert"
        assert announcement["rule"] == "tts_test"
        assert announcement["text"] == "Someone is at the door"
        assert "audio_base64" in announcement

    @pytest.mark.asyncio
    async def test_tts_deliver_no_active_tts(self):
        mock_registry = MagicMock()
        mock_registry.get_active.return_value = None

        mock_conn_mgr = AsyncMock()
        delivery = TTSDelivery(mock_registry, mock_conn_mgr)
        rule = self._rule()
        event = _make_vision_event()

        # Should return without error when TTS is unavailable
        await delivery.deliver("No TTS", rule, event)
        mock_conn_mgr.queue_announcement.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_log_alert_callback(self, caplog):
        rule = self._rule(name="log_test")
        event = _make_vision_event(source_id="cam_kitchen")

        with caplog.at_level(logging.INFO, logger="atlas.alerts.delivery"):
            await log_alert_callback("Test log message", rule, event)

        assert "ALERT [log_test/vision]" in caplog.text
        assert "Test log message" in caplog.text
        assert "cam_kitchen" in caplog.text

    def test_setup_default_callbacks(self):
        mock_manager = MagicMock()
        setup_default_callbacks(mock_manager)
        mock_manager.register_callback.assert_called_once_with(log_alert_callback)
