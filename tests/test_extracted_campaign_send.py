from __future__ import annotations

from datetime import datetime, timezone

import pytest

from extracted_content_pipeline.campaign_ports import SendResult
from extracted_content_pipeline.campaign_send import (
    CampaignSendConfig,
    CampaignSendService,
    build_unsubscribe_url,
    unsubscribe_headers,
    wrap_with_footer,
)
from extracted_content_pipeline.campaign_suppression import CampaignSuppressionService


class _Clock:
    def __init__(self):
        self.value = datetime(2026, 5, 1, 15, 30, tzinfo=timezone.utc)

    def now(self):
        return self.value


class _CampaignRepo:
    def __init__(self, rows):
        self.rows = rows
        self.list_calls = []
        self.sent = []
        self.cancelled = []
        self.failed = []

    async def list_due_sends(self, *, limit, now):
        self.list_calls.append({"limit": limit, "now": now})
        return self.rows

    async def mark_sent(self, *, campaign_id, result, sent_at):
        self.sent.append({"campaign_id": campaign_id, "result": result, "sent_at": sent_at})

    async def mark_cancelled(self, *, campaign_id, reason, metadata=None):
        self.cancelled.append({"campaign_id": campaign_id, "reason": reason, "metadata": metadata})

    async def mark_send_failed(self, *, campaign_id, error, metadata=None):
        self.failed.append({"campaign_id": campaign_id, "error": error, "metadata": metadata})

    async def save_drafts(self, drafts, *, scope):  # pragma: no cover - protocol filler
        raise AssertionError("not used")

    async def record_webhook_event(self, event):  # pragma: no cover - protocol filler
        raise AssertionError("not used")

    async def refresh_analytics(self):  # pragma: no cover - protocol filler
        raise AssertionError("not used")


class _SuppressionRepo:
    def __init__(self, suppressed=None):
        self.suppressed = suppressed or set()
        self.calls = []

    async def is_suppressed(self, *, email=None, domain=None):
        self.calls.append({"email": email, "domain": domain})
        return (email, domain) in self.suppressed

    async def add_suppression(self, **kwargs):  # pragma: no cover - protocol filler
        raise AssertionError("not used")


class _Sender:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.requests = []

    async def send(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return SendResult(provider="test", message_id=f"msg-{request.campaign_id}")


class _Audit:
    def __init__(self):
        self.events = []

    async def record(self, event_type, *, campaign_id=None, sequence_id=None, metadata=None):
        self.events.append({
            "event_type": event_type,
            "campaign_id": campaign_id,
            "sequence_id": sequence_id,
            "metadata": metadata,
        })


def _service(rows, *, suppressed=None, sender=None, config=None):
    repo = _CampaignRepo(rows)
    suppression_repo = _SuppressionRepo(suppressed=suppressed)
    audit = _Audit()
    clock = _Clock()
    service = CampaignSendService(
        campaigns=repo,
        suppression=CampaignSuppressionService(suppression_repo),
        sender=sender or _Sender(),
        audit=audit,
        clock=clock,
        config=config or CampaignSendConfig(
            default_from_email="sender@example.com",
            default_reply_to="reply@example.com",
            unsubscribe_base_url="https://example.test/unsub",
            company_address="123 Market St",
            limit=10,
        ),
    )
    return service, repo, suppression_repo, service._sender, audit, clock


def _row(**overrides):
    data = {
        "id": "campaign-1",
        "sequence_id": "sequence-1",
        "recipient_email": " Buyer@Example.COM ",
        "from_email": "",
        "subject": "Pricing signal",
        "body": "<p>Hello</p>",
        "company_name": "Acme",
        "step_number": 2,
        "metadata": {"source": "test"},
    }
    data.update(overrides)
    return data


def test_build_unsubscribe_url_handles_existing_query_string():
    assert (
        build_unsubscribe_url("https://example.test/unsub?source=email", "a+b@example.com")
        == "https://example.test/unsub?source=email&email=a%2Bb%40example.com"
    )


def test_unsubscribe_headers_and_footer_are_optional():
    assert unsubscribe_headers("", "person@example.com") == {}
    assert wrap_with_footer(
        "<p>Hello</p>",
        recipient_email="person@example.com",
        config=CampaignSendConfig(),
    ) == "<p>Hello</p>"


@pytest.mark.asyncio
async def test_send_due_sends_campaign_and_records_audit():
    service, repo, suppression_repo, sender, audit, clock = _service([_row()])

    summary = await service.send_due()

    assert summary.as_dict() == {"sent": 1, "failed": 0, "suppressed": 0, "skipped": 0}
    assert repo.list_calls == [{"limit": 10, "now": clock.value}]
    assert suppression_repo.calls == [
        {"email": "buyer@example.com", "domain": None},
        {"email": None, "domain": "example.com"},
    ]
    assert len(sender.requests) == 1
    request = sender.requests[0]
    assert request.to_email == "buyer@example.com"
    assert request.from_email == "sender@example.com"
    assert request.reply_to == "reply@example.com"
    assert "Unsubscribe" in request.html_body
    assert request.headers["List-Unsubscribe"] == "<https://example.test/unsub?email=buyer%40example.com>"
    assert {"name": "company", "value": "Acme"} in request.tags
    assert repo.sent == [{
        "campaign_id": "campaign-1",
        "result": SendResult(provider="test", message_id="msg-campaign-1"),
        "sent_at": clock.value,
    }]
    assert audit.events == [{
        "event_type": "sent",
        "campaign_id": "campaign-1",
        "sequence_id": "sequence-1",
        "metadata": {
            "provider": "test",
            "message_id": "msg-campaign-1",
            "recipient_email": "buyer@example.com",
        },
    }]


@pytest.mark.asyncio
async def test_send_due_cancels_suppressed_campaign_before_sender_call():
    service, repo, _, sender, audit, _ = _service(
        [_row()],
        suppressed={("buyer@example.com", None)},
    )

    summary = await service.send_due()

    assert summary.as_dict() == {"sent": 0, "failed": 0, "suppressed": 1, "skipped": 0}
    assert sender.requests == []
    assert repo.cancelled == [{
        "campaign_id": "campaign-1",
        "reason": "suppressed",
        "metadata": {"recipient_email": "buyer@example.com", "domain": "example.com"},
    }]
    assert audit.events[0]["event_type"] == "suppressed"


@pytest.mark.asyncio
async def test_send_due_skips_missing_recipient_and_marks_failed():
    service, repo, _, sender, audit, _ = _service([_row(recipient_email=" ")])

    summary = await service.send_due()

    assert summary.as_dict() == {"sent": 0, "failed": 0, "suppressed": 0, "skipped": 1}
    assert sender.requests == []
    assert repo.failed == [{
        "campaign_id": "campaign-1",
        "error": "recipient_email_missing",
        "metadata": {"reason": "recipient_email_missing"},
    }]
    assert audit.events[0]["event_type"] == "send_skipped"


@pytest.mark.asyncio
async def test_send_due_skips_missing_from_email_and_marks_failed():
    service, repo, _, sender, audit, _ = _service(
        [_row(from_email=" ")],
        config=CampaignSendConfig(default_from_email=""),
    )

    summary = await service.send_due()

    assert summary.as_dict() == {"sent": 0, "failed": 0, "suppressed": 0, "skipped": 1}
    assert sender.requests == []
    assert repo.failed == [{
        "campaign_id": "campaign-1",
        "error": "from_email_missing",
        "metadata": {"reason": "from_email_missing"},
    }]
    assert audit.events[0]["metadata"] == {"reason": "from_email_missing"}


@pytest.mark.asyncio
async def test_send_due_records_sender_failure():
    service, repo, _, sender, audit, _ = _service(
        [_row()],
        sender=_Sender(error=RuntimeError("provider down")),
    )

    summary = await service.send_due()

    assert summary.as_dict() == {"sent": 0, "failed": 1, "suppressed": 0, "skipped": 0}
    assert len(sender.requests) == 1
    assert repo.failed == [{
        "campaign_id": "campaign-1",
        "error": "RuntimeError: provider down",
        "metadata": {"recipient_email": "buyer@example.com"},
    }]
    assert audit.events[0]["event_type"] == "send_failed"


@pytest.mark.asyncio
async def test_send_due_skips_rows_without_campaign_id_without_touching_ports():
    service, repo, suppression_repo, sender, audit, _ = _service([_row(id=" ")])

    summary = await service.send_due()

    assert summary.as_dict() == {"sent": 0, "failed": 0, "suppressed": 0, "skipped": 1}
    assert suppression_repo.calls == []
    assert sender.requests == []
    assert repo.sent == []
    assert repo.failed == []
    assert audit.events == []
