from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock
from uuid import uuid4

import pytest

from atlas_brain.autonomous.tasks import campaign_send as mod


class _Pool:
    is_initialized = True

    def __init__(self):
        self.execute_calls = []

    async def fetch(self, query, *args):
        return [{
            "id": uuid4(),
            "sequence_id": uuid4(),
            "step_number": 1,
            "recipient_email": "owner@example.com",
            "from_email": "atlas@example.com",
            "subject": "Renewal pressure",
            "body": "<p>There is broad pressure across the market.</p>",
            "cta": "Book time",
            "company_name": "Acme Co",
            "channel": "email_followup",
            "target_mode": "vendor_retention",
            "competitors_considering": [{"name": "Freshdesk", "reason": "pricing"}],
            "metadata": {
                "reasoning_anchor_examples": {
                    "outlier_or_named_account": [
                        {
                            "witness_id": "witness:r1:0",
                            "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                            "time_anchor": "Q2 renewal",
                            "numeric_literals": {"currency_mentions": ["$200k/year"]},
                            "competitor": "Freshdesk",
                            "pain_category": "pricing",
                        },
                    ],
                },
                "reasoning_witness_highlights": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    },
                ],
                "reasoning_reference_ids": {"witness_ids": ["witness:r1:0"]},
            },
            "company_context": {},
        }]

    async def fetchval(self, query, *args):
        if "SELECT count(*) FROM b2b_campaigns WHERE recipient_email = $1 AND status = 'sent'" in query:
            return 0
        return 0

    async def fetchrow(self, query, *args):
        return None

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"


class _FallbackReasoningView:
    def consumer_context(self, consumer):
        assert consumer == "campaign"
        return {
            "anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "witness:r1:0",
                        "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "Freshdesk",
                        "pain_category": "pricing",
                    },
                ],
            },
            "witness_highlights": [
                {
                    "witness_id": "witness:r1:0",
                    "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "Freshdesk",
                    "pain_category": "pricing",
                },
            ],
            "reference_ids": {"witness_ids": ["witness:r1:0"]},
        }


@pytest.mark.asyncio
async def test_campaign_send_reverts_generic_campaign_to_draft(monkeypatch):
    pool = _Pool()
    sender = SimpleNamespace(send=AsyncMock(return_value={"id": "esp-1"}))

    async def _not_suppressed(pool, email):
        return None

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_in_send_window", lambda cfg: True)
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", True)
    monkeypatch.setattr(mod.settings.campaign_sequence, "auto_send_enabled", True)
    monkeypatch.setattr(mod.settings.b2b_campaign, "revalidate_before_send", True)
    monkeypatch.setattr("atlas_brain.services.campaign_sender.get_campaign_sender", lambda: sender)
    monkeypatch.setattr("atlas_brain.autonomous.tasks.campaign_suppression.is_suppressed", _not_suppressed)
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())
    monkeypatch.setattr(mod, "record_attempt", AsyncMock())
    monkeypatch.setattr(mod, "emit_event", AsyncMock())

    result = await mod.run(SimpleNamespace())

    assert result["sent"] == 0
    assert result["failed"] == 1
    sender.send.assert_not_awaited()
    assert any("SET metadata = $1::jsonb" in call[0] for call in pool.execute_calls)
    assert any("SET status = 'draft'" in call[0] for call in pool.execute_calls)


@pytest.mark.asyncio
async def test_campaign_send_uses_synthesis_fallback_and_records_success(monkeypatch):
    pool = _Pool()
    pool.fetch = AsyncMock(return_value=[{
        "id": uuid4(),
        "sequence_id": uuid4(),
        "step_number": 1,
        "recipient_email": "owner@example.com",
        "from_email": "atlas@example.com",
        "subject": "Renewal pressure",
        "body": "<p>The Q2 renewal now carries a $200k/year issue and Freshdesk is showing up.</p>",
        "cta": "Book time",
        "company_name": "Slack",
        "vendor_name": "Slack",
        "channel": "email_followup",
        "target_mode": "vendor_retention",
        "competitors_considering": [{"name": "Freshdesk", "reason": "pricing"}],
        "metadata": {"generation_audit": {"status": "succeeded"}},
        "company_context": None,
    }])
    sender = SimpleNamespace(send=AsyncMock(return_value={"id": "esp-1"}))
    record_attempt = AsyncMock()

    async def _not_suppressed(pool, email):
        return None

    async def _fake_reasoning_view(pool, vendor_name):
        assert vendor_name == "Slack"
        return _FallbackReasoningView()

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_in_send_window", lambda cfg: True)
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", True)
    monkeypatch.setattr(mod.settings.campaign_sequence, "auto_send_enabled", True)
    monkeypatch.setattr(mod.settings.b2b_campaign, "revalidate_before_send", True)
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks._b2b_synthesis_reader.load_best_reasoning_view",
        _fake_reasoning_view,
    )
    monkeypatch.setattr("atlas_brain.services.campaign_sender.get_campaign_sender", lambda: sender)
    monkeypatch.setattr("atlas_brain.autonomous.tasks.campaign_suppression.is_suppressed", _not_suppressed)
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())
    monkeypatch.setattr(mod, "record_attempt", record_attempt)
    monkeypatch.setattr(mod, "emit_event", AsyncMock())

    result = await mod.run(SimpleNamespace())

    assert result["sent"] == 1
    assert result["failed"] == 0
    sender.send.assert_awaited_once()
    assert any("SET metadata = $1::jsonb" in call[0] for call in pool.execute_calls)
    record_attempt.assert_any_await(
        pool,
        artifact_type="campaign",
        artifact_id=ANY,
        attempt_no=1,
        stage="send_validation",
        status="succeeded",
        blocker_count=0,
        warning_count=0,
        warnings=[],
    )


@pytest.mark.asyncio
async def test_campaign_send_blocks_vendor_cold_email_with_competitor_name(monkeypatch):
    pool = _Pool()
    pool.fetch = AsyncMock(return_value=[{
        "id": uuid4(),
        "sequence_id": uuid4(),
        "step_number": 1,
        "recipient_email": "owner@example.com",
        "from_email": "atlas@example.com",
        "subject": "Renewal pressure",
        "body": "<p>Freshdesk is showing up in the active comparison before renewal, and that is the signal we can unpack for you.</p>",
        "cta": "Book time",
        "company_name": "Slack",
        "vendor_name": "Slack",
        "channel": "email_cold",
        "target_mode": "vendor_retention",
        "competitors_considering": [{"name": "Freshdesk", "reason": "pricing"}],
        "metadata": {"tier": "report"},
        "company_context": None,
    }])
    sender = SimpleNamespace(send=AsyncMock(return_value={"id": "esp-1"}))

    async def _not_suppressed(pool, email):
        return None

    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)
    monkeypatch.setattr(mod, "_in_send_window", lambda cfg: True)
    monkeypatch.setattr(mod.settings.campaign_sequence, "enabled", True)
    monkeypatch.setattr(mod.settings.campaign_sequence, "auto_send_enabled", True)
    monkeypatch.setattr(mod.settings.b2b_campaign, "revalidate_before_send", True)
    monkeypatch.setattr("atlas_brain.services.campaign_sender.get_campaign_sender", lambda: sender)
    monkeypatch.setattr("atlas_brain.autonomous.tasks.campaign_suppression.is_suppressed", _not_suppressed)
    monkeypatch.setattr(mod, "log_campaign_event", AsyncMock())
    monkeypatch.setattr(mod, "record_attempt", AsyncMock())
    monkeypatch.setattr(mod, "emit_event", AsyncMock())

    result = await mod.run(SimpleNamespace())

    assert result["sent"] == 0
    assert result["failed"] == 1
    sender.send.assert_not_awaited()
    assert any("SET status = 'draft'" in call[0] for call in pool.execute_calls)
