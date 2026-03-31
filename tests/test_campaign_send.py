from types import SimpleNamespace
from unittest.mock import AsyncMock
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
            "company_name": "Acme Co",
            "channel": "email_followup",
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
