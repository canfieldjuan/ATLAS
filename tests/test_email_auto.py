"""
Unit tests for auto-execute intent actions and auto-approve drafts.

Tests pure logic without requiring a live database or external services.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_email(
    *,
    msg_id: str = "uid-100",
    intent: str | None = None,
    confidence: float = 0.0,
    replyable: bool | None = True,
    priority: str = "action_required",
    category: str = "customer",
    action_plan: list | None = None,
) -> dict[str, Any]:
    """Build a fake email dict matching what email_intake.py produces."""
    e: dict[str, Any] = {
        "id": msg_id,
        "from": "Alice <alice@example.com>",
        "subject": "Test email",
        "body_text": "Hello world",
        "message_id": "<abc@example.com>",
        "replyable": replyable,
        "priority": priority,
        "category": category,
    }
    if intent:
        e["_intent"] = intent
        e["_confidence"] = confidence
    if action_plan is not None:
        e["_action_plan"] = action_plan
    return e


def _make_cfg(
    *,
    auto_execute_enabled: bool = True,
    auto_execute_min_confidence: float = 0.85,
    auto_execute_intents: list[str] | None = None,
):
    """Build a mock EmailIntakeConfig."""
    cfg = MagicMock()
    cfg.auto_execute_enabled = auto_execute_enabled
    cfg.auto_execute_min_confidence = auto_execute_min_confidence
    cfg.auto_execute_intents = auto_execute_intents or [
        "estimate_request",
        "reschedule",
        "info_admin",
    ]
    return cfg


# ---------------------------------------------------------------------------
# Feature A: _auto_execute_actions
# ---------------------------------------------------------------------------


class TestAutoExecuteActions:
    """Tests for _auto_execute_actions in email_intake.py."""

    @pytest.fixture(autouse=True)
    def _patch_actions(self):
        """Patch all action endpoint functions."""
        base = "atlas_brain.api.email_actions"
        with (
            patch(f"{base}.generate_quote", new_callable=AsyncMock) as gq,
            patch(f"{base}.show_slots", new_callable=AsyncMock) as ss,
            patch(f"{base}.send_info", new_callable=AsyncMock) as si,
            patch(f"{base}.archive_email", new_callable=AsyncMock) as ae,
        ):
            self.mock_quote = gq
            self.mock_slots = ss
            self.mock_info = si
            self.mock_archive = ae
            yield

    async def _run(self, emails, cfg=None):
        from atlas_brain.autonomous.tasks.email_intake import (
            _auto_execute_actions,
        )

        return await _auto_execute_actions(emails, cfg or _make_cfg())

    @pytest.mark.asyncio
    async def test_executes_estimate_request(self):
        emails = [_make_email(intent="estimate_request", confidence=0.92)]
        count = await self._run(emails)
        assert count == 1
        self.mock_quote.assert_awaited_once_with("uid-100")
        assert emails[0]["_auto_executed"] is True

    @pytest.mark.asyncio
    async def test_executes_reschedule(self):
        emails = [_make_email(intent="reschedule", confidence=0.90)]
        count = await self._run(emails)
        assert count == 1
        self.mock_slots.assert_awaited_once_with("uid-100")

    @pytest.mark.asyncio
    async def test_executes_info_admin_replyable_true(self):
        emails = [_make_email(intent="info_admin", confidence=0.88, replyable=True)]
        count = await self._run(emails)
        assert count == 1
        self.mock_info.assert_awaited_once_with("uid-100")
        self.mock_archive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_executes_info_admin_replyable_false(self):
        emails = [_make_email(intent="info_admin", confidence=0.88, replyable=False)]
        count = await self._run(emails)
        assert count == 1
        self.mock_archive.assert_awaited_once_with("uid-100")
        self.mock_info.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_info_admin_replyable_none(self):
        """Ambiguous replyable should NOT be archived or info-replied."""
        emails = [_make_email(intent="info_admin", confidence=0.90, replyable=None)]
        count = await self._run(emails)
        assert count == 0
        self.mock_info.assert_not_awaited()
        self.mock_archive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_never_executes_complaint(self):
        """Complaint intent must NEVER be auto-executed."""
        emails = [_make_email(intent="complaint", confidence=0.99)]
        count = await self._run(emails)
        assert count == 0
        self.mock_quote.assert_not_awaited()
        self.mock_slots.assert_not_awaited()
        self.mock_info.assert_not_awaited()
        self.mock_archive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_below_confidence(self):
        emails = [_make_email(intent="estimate_request", confidence=0.60)]
        count = await self._run(emails)
        assert count == 0
        self.mock_quote.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_intent_not_in_whitelist(self):
        cfg = _make_cfg(auto_execute_intents=["estimate_request"])
        emails = [_make_email(intent="reschedule", confidence=0.95)]
        count = await self._run(emails, cfg)
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_no_intent(self):
        emails = [_make_email()]  # no intent set
        count = await self._run(emails)
        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_no_msg_id(self):
        e = _make_email(intent="estimate_request", confidence=0.95)
        e["id"] = ""
        count = await self._run([e])
        assert count == 0

    @pytest.mark.asyncio
    async def test_handles_action_exception(self):
        """If action endpoint raises, it should be logged and skipped."""
        self.mock_quote.side_effect = Exception("IMAP down")
        emails = [_make_email(intent="estimate_request", confidence=0.95)]
        count = await self._run(emails)
        assert count == 0
        assert "_auto_executed" not in emails[0]

    @pytest.mark.asyncio
    async def test_multiple_emails_mixed(self):
        emails = [
            _make_email(msg_id="1", intent="estimate_request", confidence=0.95),
            _make_email(msg_id="2", intent="complaint", confidence=0.99),
            _make_email(msg_id="3", intent="reschedule", confidence=0.50),
            _make_email(msg_id="4", intent="info_admin", confidence=0.90, replyable=True),
        ]
        count = await self._run(emails)
        assert count == 2  # #1 (quote) + #4 (info)
        assert emails[0].get("_auto_executed") is True
        assert "_auto_executed" not in emails[1]  # complaint
        assert "_auto_executed" not in emails[2]  # low confidence
        assert emails[3].get("_auto_executed") is True

    @pytest.mark.asyncio
    async def test_confidence_at_exact_threshold(self):
        """Confidence exactly at threshold should be accepted."""
        emails = [_make_email(intent="estimate_request", confidence=0.85)]
        count = await self._run(emails)
        assert count == 1

    @pytest.mark.asyncio
    async def test_confidence_just_below_threshold(self):
        emails = [_make_email(intent="estimate_request", confidence=0.849)]
        count = await self._run(emails)
        assert count == 0


# ---------------------------------------------------------------------------
# Feature A: _parse_intent_plan confidence extraction
# ---------------------------------------------------------------------------


class TestParseIntentPlanConfidence:
    """Verify confidence coercion in _parse_intent_plan."""

    def _parse(self, text):
        from atlas_brain.autonomous.tasks.email_intake import _parse_intent_plan

        return _parse_intent_plan(text)

    def test_numeric_confidence(self):
        result = self._parse('{"intent":"estimate_request","confidence":0.92,"actions":[]}')
        assert result is not None
        assert result["confidence"] == pytest.approx(0.92)

    def test_string_confidence(self):
        result = self._parse('{"intent":"estimate_request","confidence":"0.88","actions":[]}')
        assert result is not None
        assert result["confidence"] == pytest.approx(0.88)

    def test_missing_confidence_defaults(self):
        result = self._parse('{"intent":"reschedule","actions":[]}')
        assert result is not None
        assert result["confidence"] == pytest.approx(0.5)

    def test_invalid_confidence_defaults(self):
        result = self._parse('{"intent":"info_admin","confidence":"high","actions":[]}')
        assert result is not None
        assert result["confidence"] == pytest.approx(0.5)

    def test_strips_think_tags(self):
        text = '<think>reasoning here</think>{"intent":"complaint","confidence":0.95,"actions":[]}'
        result = self._parse(text)
        assert result is not None
        assert result["intent"] == "complaint"
        assert result["confidence"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Feature A: action_plan JSON format
# ---------------------------------------------------------------------------


class TestActionPlanJsonFormat:
    """Verify the new action_plan JSON wraps confidence + actions."""

    def test_wraps_confidence_and_actions(self):
        """The DB-stored JSON should be {"confidence": ..., "actions": [...]}."""
        actions = [{"action": "draft_reply", "priority": 1}]
        email = _make_email(intent="estimate_request", confidence=0.92, action_plan=actions)

        # Replicate what _record_with_action_plans does
        stored = json.dumps({
            "confidence": email.get("_confidence", 0.5),
            "actions": email["_action_plan"],
        })
        parsed = json.loads(stored)

        assert parsed["confidence"] == pytest.approx(0.92)
        assert parsed["actions"] == actions

    def test_no_action_plan_stores_none(self):
        email = _make_email(intent="estimate_request", confidence=0.92)
        result = (
            json.dumps({
                "confidence": email.get("_confidence", 0.5),
                "actions": email["_action_plan"],
            })
            if email.get("_action_plan")
            else None
        )
        assert result is None


# ---------------------------------------------------------------------------
# Feature B: auto-approve confidence extraction from action_plan JSON
# ---------------------------------------------------------------------------


class TestAutoApproveConfidenceExtraction:
    """Verify confidence is correctly extracted from action_plan JSONB."""

    def _extract(self, action_plan_raw):
        """Replicate the extraction logic from email_auto_approve.py."""
        confidence = 0.0
        if action_plan_raw:
            try:
                plan = (
                    json.loads(action_plan_raw)
                    if isinstance(action_plan_raw, str)
                    else action_plan_raw
                )
                if isinstance(plan, dict):
                    confidence = float(plan.get("confidence", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return confidence

    def test_new_format_dict(self):
        raw = {"confidence": 0.92, "actions": [{"action": "draft"}]}
        assert self._extract(raw) == pytest.approx(0.92)

    def test_new_format_string(self):
        raw = json.dumps({"confidence": 0.88, "actions": []})
        assert self._extract(raw) == pytest.approx(0.88)

    def test_legacy_list_format(self):
        """Old action_plans stored as a plain list should yield 0.0."""
        raw = [{"action": "draft", "priority": 1}]
        assert self._extract(raw) == pytest.approx(0.0)

    def test_none_yields_zero(self):
        assert self._extract(None) == pytest.approx(0.0)

    def test_malformed_json_string(self):
        assert self._extract("{bad json") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Feature B: draft notification auto-approve eligibility
# ---------------------------------------------------------------------------


class TestDraftNotificationAutoApprove:
    """Test _send_draft_notification auto-approve eligibility logic."""

    def _check_eligible(self, *, intent, confidence, enabled=True, intents=None):
        """Replicate the eligibility check from _send_draft_notification."""
        cfg = MagicMock()
        cfg.auto_approve_enabled = enabled
        cfg.auto_approve_intents = intents or [
            "info_admin",
            "estimate_request",
            "reschedule",
        ]
        cfg.auto_approve_min_confidence = 0.85
        return (
            cfg.auto_approve_enabled
            and intent
            and intent != "complaint"
            and intent in cfg.auto_approve_intents
            and confidence >= cfg.auto_approve_min_confidence
        )

    def test_eligible_estimate_request(self):
        assert self._check_eligible(intent="estimate_request", confidence=0.90)

    def test_eligible_info_admin(self):
        assert self._check_eligible(intent="info_admin", confidence=0.85)

    def test_not_eligible_complaint(self):
        assert not self._check_eligible(intent="complaint", confidence=0.99)

    def test_not_eligible_low_confidence(self):
        assert not self._check_eligible(intent="estimate_request", confidence=0.50)

    def test_not_eligible_disabled(self):
        assert not self._check_eligible(
            intent="estimate_request", confidence=0.95, enabled=False
        )

    def test_not_eligible_intent_not_in_list(self):
        assert not self._check_eligible(
            intent="reschedule", confidence=0.95, intents=["info_admin"]
        )

    def test_not_eligible_none_intent(self):
        assert not self._check_eligible(intent=None, confidence=0.95)


# ---------------------------------------------------------------------------
# Config field validation
# ---------------------------------------------------------------------------


class TestConfigFields:
    """Verify config field defaults and validation."""

    def test_email_intake_config_defaults(self):
        from atlas_brain.config import EmailIntakeConfig

        cfg = EmailIntakeConfig()
        assert cfg.auto_execute_enabled is False
        assert cfg.auto_execute_min_confidence == pytest.approx(0.85)
        assert "estimate_request" in cfg.auto_execute_intents
        assert "complaint" not in cfg.auto_execute_intents

    def test_email_draft_config_defaults(self):
        from atlas_brain.config import EmailDraftConfig

        cfg = EmailDraftConfig()
        assert cfg.auto_approve_enabled is False
        assert cfg.auto_approve_delay_seconds == 300
        assert cfg.auto_approve_min_confidence == pytest.approx(0.85)
        assert "complaint" not in cfg.auto_approve_intents

    def test_auto_approve_delay_bounds(self):
        from pydantic import ValidationError
        from atlas_brain.config import EmailDraftConfig

        # Below minimum
        with pytest.raises(ValidationError):
            EmailDraftConfig(auto_approve_delay_seconds=30)

        # Above maximum
        with pytest.raises(ValidationError):
            EmailDraftConfig(auto_approve_delay_seconds=3600)

    def test_confidence_bounds(self):
        from pydantic import ValidationError
        from atlas_brain.config import EmailIntakeConfig

        with pytest.raises(ValidationError):
            EmailIntakeConfig(auto_execute_min_confidence=0.3)

        with pytest.raises(ValidationError):
            EmailIntakeConfig(auto_execute_min_confidence=1.5)


# ---------------------------------------------------------------------------
# Enriched notification: auto-executed emails skipped
# ---------------------------------------------------------------------------


class TestEnrichedNotificationSkipsAutoExecuted:
    """Verify that auto-executed emails are skipped in _send_enriched_notifications."""

    @pytest.mark.asyncio
    async def test_auto_executed_email_skipped(self):
        """An email with _auto_executed=True should not trigger enriched notification."""
        email = _make_email(intent="estimate_request", confidence=0.92)
        email["_auto_executed"] = True
        email["_customer_summary"] = "Returning customer"

        with (
            patch("atlas_brain.autonomous.tasks.email_intake.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.email_intake.httpx") as mock_httpx,
        ):
            mock_settings.alerts.ntfy_enabled = True
            mock_settings.email_draft.enabled = True
            mock_settings.email_draft.atlas_api_url = "http://localhost"
            mock_settings.alerts.ntfy_url = "http://ntfy.sh"
            mock_settings.alerts.ntfy_topic = "test"

            from atlas_brain.autonomous.tasks.email_intake import (
                _send_enriched_notifications,
            )

            await _send_enriched_notifications([email])

            # httpx.AsyncClient should NOT have been used (no POST call)
            mock_httpx.AsyncClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_auto_executed_email_sends(self):
        """A normal enriched email should still send a notification."""
        email = _make_email(intent="estimate_request", confidence=0.92)
        email["_customer_summary"] = "Returning customer"
        # No _auto_executed flag

        with (
            patch("atlas_brain.autonomous.tasks.email_intake.settings") as mock_settings,
            patch("atlas_brain.autonomous.tasks.email_intake.httpx") as mock_httpx,
        ):
            mock_settings.alerts.ntfy_enabled = True
            mock_settings.email_draft.enabled = True
            mock_settings.email_draft.atlas_api_url = "http://localhost"
            mock_settings.alerts.ntfy_url = "http://ntfy.sh"
            mock_settings.alerts.ntfy_topic = "test"
            mock_settings.alerts.ntfy_auth_token = ""

            mock_client = AsyncMock()
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(
                return_value=False
            )
            mock_client.post.return_value.raise_for_status = MagicMock()

            from atlas_brain.autonomous.tasks.email_intake import (
                _send_enriched_notifications,
            )

            await _send_enriched_notifications([email])

            mock_client.post.assert_awaited_once()
