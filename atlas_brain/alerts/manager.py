"""
Centralized alert manager for all event sources.

Processes events against rules, manages cooldowns, triggers callbacks,
and persists alerts to the database.
"""

import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from ..config import settings
from .events import AlertEvent
from .rules import AlertRule, create_vision_rule, create_audio_rule

logger = logging.getLogger("atlas.alerts.manager")

AlertCallback = Callable[[str, AlertRule, AlertEvent], Awaitable[None]]


class AlertManager:
    """
    Centralized alert manager for all event sources.

    Supports:
    - Multiple rules with priority ordering
    - Cooldown per rule to prevent spam
    - Callback-based actions (TTS, notifications, etc.)
    - Database persistence
    """

    def __init__(self):
        self._rules: dict[str, AlertRule] = {}
        self._cooldowns: dict[str, datetime] = {}
        self._callbacks: list[AlertCallback] = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the manager with default rules."""
        if self._initialized:
            return
        self._add_default_rules()
        self._initialized = True
        logger.info("AlertManager initialized with %d default rules", len(self._rules))

    def _add_default_rules(self) -> None:
        """Add default alert rules for common scenarios."""
        self.add_rule(create_vision_rule(
            name="person_front_door",
            source_pattern="*front_door*",
            class_name="person",
            detection_type="new_track",
            message_template="Someone is at the front door.",
            cooldown_seconds=30,
            priority=10,
        ))

        self.add_rule(create_vision_rule(
            name="person_back_door",
            source_pattern="*back_door*",
            class_name="person",
            detection_type="new_track",
            message_template="Someone is at the back door.",
            cooldown_seconds=30,
            priority=9,
        ))

        self.add_rule(create_vision_rule(
            name="vehicle_driveway",
            source_pattern="*driveway*",
            class_name="car",
            detection_type="new_track",
            message_template="Vehicle detected in the driveway.",
            cooldown_seconds=60,
            priority=5,
        ))

        self.add_rule(create_vision_rule(
            name="person_garage",
            source_pattern="*garage*",
            class_name="person",
            detection_type="new_track",
            message_template="Someone is in the garage.",
            cooldown_seconds=30,
            priority=8,
        ))

        self.add_rule(create_audio_rule(
            name="doorbell",
            source_pattern="*",
            sound_class="Doorbell",
            min_confidence=0.5,
            message_template="Doorbell is ringing.",
            cooldown_seconds=15,
            priority=10,
        ))

        self.add_rule(create_audio_rule(
            name="glass_break",
            source_pattern="*",
            sound_class="*glass*",
            min_confidence=0.6,
            message_template="Glass breaking detected!",
            cooldown_seconds=5,
            priority=15,
        ))

        # Reminder alerts - always trigger for due reminders
        self.add_rule(AlertRule(
            name="reminder_due",
            event_types=["reminder"],
            source_pattern="*",
            conditions={},
            message_template="Reminder: {message}",
            cooldown_seconds=0,  # No cooldown for reminders
            priority=20,  # High priority
        ))

        # Presence transition rules — hook tasks can bind to these
        self.add_rule(AlertRule(
            name="presence_arrival",
            event_types=["presence"],
            source_pattern="*",
            conditions={"transition": "arrival"},
            message_template="Someone arrived home.",
            cooldown_seconds=300,
            priority=5,
        ))

        self.add_rule(AlertRule(
            name="presence_departure",
            event_types=["presence"],
            source_pattern="*",
            conditions={"transition": "departure"},
            message_template="House appears empty.",
            cooldown_seconds=300,
            priority=5,
        ))

    def add_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule."""
        self._rules[rule.name] = rule
        logger.debug("Added alert rule: %s (priority=%d)", rule.name, rule.priority)

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule."""
        if name in self._rules:
            del self._rules[name]
            logger.info("Removed alert rule: %s", name)
            return True
        return False

    def get_rule(self, name: str) -> Optional[AlertRule]:
        """Get a rule by name."""
        return self._rules.get(name)

    def list_rules(self, event_type: Optional[str] = None) -> list[AlertRule]:
        """List all rules sorted by priority (highest first)."""
        rules = list(self._rules.values())
        if event_type:
            rules = [r for r in rules if event_type in r.event_types or "*" in r.event_types]
        return sorted(rules, key=lambda r: -r.priority)

    def enable_rule(self, name: str) -> bool:
        """Enable a rule."""
        if name in self._rules:
            self._rules[name].enabled = True
            return True
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a rule."""
        if name in self._rules:
            self._rules[name].enabled = False
            return True
        return False

    def register_callback(self, callback: AlertCallback) -> None:
        """Register a callback for when alerts trigger."""
        self._callbacks.append(callback)
        logger.debug("Registered alert callback")

    def unregister_callback(self, callback: AlertCallback) -> None:
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _check_cooldown(self, rule_name: str, cooldown_seconds: int) -> bool:
        """Check if rule is in cooldown period."""
        last_triggered = self._cooldowns.get(rule_name)
        if last_triggered:
            elapsed = (datetime.utcnow() - last_triggered).total_seconds()
            if elapsed < cooldown_seconds:
                return False
        return True

    def _update_cooldown(self, rule_name: str) -> None:
        """Update cooldown timestamp for rule."""
        self._cooldowns[rule_name] = datetime.utcnow()

    async def process_event(self, event: AlertEvent) -> Optional[str]:
        """
        Process an event against all rules.

        Returns the alert message if a rule triggered, None otherwise.
        Only the highest priority matching rule triggers.
        """
        if not settings.alerts.enabled:
            return None

        if not self._initialized:
            self.initialize()

        for rule in self.list_rules(event.event_type):
            if not rule.matches(event):
                continue

            if not self._check_cooldown(rule.name, rule.cooldown_seconds):
                logger.debug("Rule %s in cooldown, skipping", rule.name)
                continue

            message = rule.format_message(event)
            self._update_cooldown(rule.name)

            logger.info("Alert triggered [%s]: %s", rule.name, message)

            if settings.alerts.persist_alerts:
                await self._persist_alert(rule, message, event)

            for callback in self._callbacks:
                try:
                    await callback(message, rule, event)
                except Exception as e:
                    logger.warning("Alert callback error: %s", e)

            return message

        return None

    async def _persist_alert(
        self,
        rule: AlertRule,
        message: str,
        event: AlertEvent,
    ) -> None:
        """Persist triggered alert to database."""
        try:
            from ..storage.repositories import get_unified_alert_repo

            repo = get_unified_alert_repo()
            await repo.save_alert(
                rule_name=rule.name,
                event_type=event.event_type,
                message=message,
                source_id=event.source_id,
                event_data={
                    "timestamp": event.timestamp.isoformat(),
                    "metadata": event.metadata,
                },
            )
        except ImportError:
            logger.debug("Unified alert repository not available yet")
        except Exception as e:
            logger.warning("Failed to persist alert: %s", e)


_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
        _alert_manager.initialize()
    return _alert_manager


def reset_alert_manager() -> None:
    """Reset the global alert manager (for testing)."""
    global _alert_manager
    _alert_manager = None
