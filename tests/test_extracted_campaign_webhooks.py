from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest

from extracted_content_pipeline.campaign_ports import WebhookEvent
from extracted_content_pipeline.campaign_suppression import CampaignSuppressionService
from extracted_content_pipeline.campaign_webhooks import (
    CampaignWebhookIngestionConfig,
    CampaignWebhookIngestionService,
    ResendWebhookConfig,
    ResendWebhookVerifier,
    WebhookPayloadError,
    WebhookVerificationError,
    normalize_resend_payload,
    verify_svix_signature,
)


def _secret() -> str:
    return "whsec_" + base64.b64encode(b"secret").decode("utf-8")


def _headers(body: bytes, *, secret: str | None = None, msg_id: str = "msg_1"):
    secret_text = secret or _secret()
    raw_secret = secret_text[6:] if secret_text.startswith("whsec_") else secret_text
    secret_bytes = base64.b64decode(raw_secret)
    timestamp = "1714550400"
    to_sign = f"{msg_id}.{timestamp}.".encode("utf-8") + body
    signature = base64.b64encode(
        hmac.new(secret_bytes, to_sign, hashlib.sha256).digest()
    ).decode("utf-8")
    return {
        "svix-id": msg_id,
        "svix-timestamp": timestamp,
        "svix-signature": f"v1,{signature}",
    }


def test_verify_svix_signature_accepts_valid_signature():
    body = b'{"type":"email.delivered","data":{"email_id":"email_1"}}'

    assert verify_svix_signature(body, _headers(body), _secret()) is True


def test_verify_svix_signature_rejects_missing_or_wrong_signature():
    body = b'{"type":"email.delivered","data":{"email_id":"email_1"}}'

    assert verify_svix_signature(body, {}, _secret()) is False
    assert verify_svix_signature(body, _headers(body, msg_id="msg_1"), _secret()) is True
    bad_headers = dict(_headers(body))
    bad_headers["svix-signature"] = "v1,bad"
    assert verify_svix_signature(body, bad_headers, _secret()) is False


def test_verify_svix_signature_can_be_disabled_for_local_dev():
    assert verify_svix_signature(b"{}", {}, "", verify_signatures=False) is True
    assert verify_svix_signature(b"{}", {}, "") is True


def test_normalize_resend_payload_maps_known_events_and_metadata():
    event = normalize_resend_payload({
        "type": "email.clicked",
        "created_at": "2026-05-01T12:00:00Z",
        "data": {
            "email_id": "email_1",
            "to": "buyer@example.com",
            "click": {"link": "https://example.test/demo"},
        },
    })

    assert event.provider == "resend"
    assert event.event_type == "clicked"
    assert event.message_id == "email_1"
    assert event.email == "buyer@example.com"
    assert event.occurred_at.isoformat() == "2026-05-01T12:00:00+00:00"
    assert event.payload["normalized"] == {
        "raw_event_type": "email.clicked",
        "click_url": "https://example.test/demo",
        "bounce_type": None,
    }


def test_normalize_resend_payload_preserves_unknown_event_type():
    event = normalize_resend_payload({
        "type": "email.rendered",
        "data": {"email_id": "email_1"},
    })

    assert event.event_type == "email.rendered"
    assert event.message_id == "email_1"


def test_normalize_resend_payload_extracts_bounce_type():
    event = normalize_resend_payload({
        "type": "email.bounced",
        "data": {
            "email_id": "email_1",
            "email": "buyer@example.com",
            "bounce": {"type": "hard"},
        },
    })

    assert event.event_type == "bounced"
    assert event.email == "buyer@example.com"
    assert event.payload["normalized"]["bounce_type"] == "hard"


def test_resend_verifier_returns_normalized_event_for_valid_payload():
    body = json.dumps({
        "type": "email.opened",
        "data": {"email_id": "email_1", "to": "buyer@example.com"},
    }).encode("utf-8")
    verifier = ResendWebhookVerifier(ResendWebhookConfig(signing_secret=_secret()))

    event = verifier.verify_and_parse(body=body, headers=_headers(body))

    assert event.event_type == "opened"
    assert event.message_id == "email_1"


def test_resend_verifier_rejects_invalid_signature():
    verifier = ResendWebhookVerifier(ResendWebhookConfig(signing_secret=_secret()))

    with pytest.raises(WebhookVerificationError, match="Invalid webhook signature"):
        verifier.verify_and_parse(body=b"{}", headers={})


def test_resend_verifier_rejects_invalid_json():
    verifier = ResendWebhookVerifier(
        ResendWebhookConfig(signing_secret="", verify_signatures=False)
    )

    with pytest.raises(WebhookPayloadError, match="Invalid JSON"):
        verifier.verify_and_parse(body=b"{not-json", headers={})


def test_resend_verifier_rejects_non_object_json():
    verifier = ResendWebhookVerifier(
        ResendWebhookConfig(signing_secret="", verify_signatures=False)
    )

    with pytest.raises(WebhookPayloadError, match="must be a JSON object"):
        verifier.verify_and_parse(body=b"[]", headers={})


class _StaticVerifier:
    def __init__(self, event: WebhookEvent):
        self.event = event
        self.calls = []

    def verify_and_parse(self, *, body, headers):
        self.calls.append({"body": body, "headers": headers})
        return self.event


class _CampaignRepo:
    def __init__(self):
        self.events = []

    async def record_webhook_event(self, event):
        self.events.append(event)

    async def save_drafts(self, drafts, *, scope):  # pragma: no cover - protocol filler
        raise AssertionError("not used")

    async def list_due_sends(self, *, limit, now):  # pragma: no cover - protocol filler
        raise AssertionError("not used")

    async def mark_sent(self, *, campaign_id, result, sent_at):  # pragma: no cover
        raise AssertionError("not used")

    async def mark_cancelled(self, *, campaign_id, reason, metadata=None):  # pragma: no cover
        raise AssertionError("not used")

    async def mark_send_failed(self, *, campaign_id, error, metadata=None):  # pragma: no cover
        raise AssertionError("not used")

    async def refresh_analytics(self):  # pragma: no cover - protocol filler
        raise AssertionError("not used")


class _SuppressionRepo:
    def __init__(self):
        self.writes = []

    async def is_suppressed(self, *, email=None, domain=None):  # pragma: no cover
        raise AssertionError("not used")

    async def add_suppression(self, **kwargs):
        self.writes.append(kwargs)


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


class _Clock:
    def __init__(self):
        self.value = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    def now(self):
        return self.value


def _event(event_type="delivered", **overrides):
    data = {
        "provider": "resend",
        "event_type": event_type,
        "message_id": "email_1",
        "email": "Buyer@Example.com",
        "payload": {"type": f"email.{event_type}", "data": {"email_id": "email_1"}},
    }
    data.update(overrides)
    return WebhookEvent(**data)


def _ingestion_service(event, *, suppression=True, audit=None, config=None):
    repo = _CampaignRepo()
    suppression_repo = _SuppressionRepo()
    service = CampaignWebhookIngestionService(
        verifier=_StaticVerifier(event),
        campaigns=repo,
        suppression=CampaignSuppressionService(suppression_repo) if suppression else None,
        audit=audit,
        clock=_Clock(),
        config=config,
    )
    return service, repo, suppression_repo


@pytest.mark.asyncio
async def test_ingestion_records_known_event_and_audit():
    audit = _Audit()
    service, repo, suppression_repo = _ingestion_service(_event("delivered"), audit=audit)

    result = await service.ingest(body=b"{}", headers={"x-test": "1"})

    assert result.as_dict() == {
        "status": "ok",
        "event_type": "delivered",
        "message_id": "email_1",
        "reason": None,
        "suppressed": False,
    }
    assert len(repo.events) == 1
    assert repo.events[0].event_type == "delivered"
    assert suppression_repo.writes == []
    assert audit.events == [{
        "event_type": "webhook_delivered",
        "campaign_id": None,
        "sequence_id": None,
        "metadata": {
            "provider": "resend",
            "message_id": "email_1",
            "email": "Buyer@Example.com",
            "suppressed": False,
        },
    }]


@pytest.mark.asyncio
async def test_ingestion_ignores_event_without_message_id():
    service, repo, suppression_repo = _ingestion_service(
        _event("delivered", message_id=None)
    )

    result = await service.ingest(body=b"{}", headers={})

    assert result.status == "ignored"
    assert result.reason == "no_message_id"
    assert repo.events == []
    assert suppression_repo.writes == []


@pytest.mark.asyncio
async def test_ingestion_ignores_unknown_event_by_default():
    service, repo, suppression_repo = _ingestion_service(_event("rendered"))

    result = await service.ingest(body=b"{}", headers={})

    assert result.status == "ignored"
    assert result.reason == "unhandled_event_type"
    assert repo.events == []
    assert suppression_repo.writes == []


@pytest.mark.asyncio
async def test_ingestion_can_record_unknown_events_when_configured():
    service, repo, _ = _ingestion_service(
        _event("rendered"),
        config=CampaignWebhookIngestionConfig(record_unknown_events=True),
    )

    result = await service.ingest(body=b"{}", headers={})

    assert result.status == "ok"
    assert repo.events[0].event_type == "rendered"


@pytest.mark.asyncio
async def test_ingestion_adds_complaint_suppression():
    service, repo, suppression_repo = _ingestion_service(_event("complained"))

    result = await service.ingest(body=b"{}", headers={})

    assert result.suppressed is True
    assert len(repo.events) == 1
    assert suppression_repo.writes == [{
        "reason": "complaint",
        "email": "buyer@example.com",
        "domain": None,
        "source": "webhook",
        "campaign_id": None,
        "notes": None,
        "expires_at": None,
        "metadata": {"provider_message_id": "email_1"},
    }]


@pytest.mark.asyncio
async def test_ingestion_adds_permanent_hard_bounce_suppression():
    event = _event(
        "bounced",
        payload={
            "type": "email.bounced",
            "data": {"email_id": "email_1", "bounce": {"type": "hard"}},
            "normalized": {"bounce_type": "hard"},
        },
    )
    service, _, suppression_repo = _ingestion_service(event)

    await service.ingest(body=b"{}", headers={})

    assert suppression_repo.writes[0]["reason"] == "bounce_hard"
    assert suppression_repo.writes[0]["expires_at"] is None
    assert suppression_repo.writes[0]["metadata"] == {
        "provider_message_id": "email_1",
        "bounce_type": "hard",
    }


@pytest.mark.asyncio
async def test_ingestion_adds_temporary_soft_bounce_suppression():
    event = _event(
        "bounced",
        payload={
            "type": "email.bounced",
            "data": {"email_id": "email_1", "bounce": {"type": "soft"}},
            "normalized": {"bounce_type": "soft"},
        },
    )
    service, _, suppression_repo = _ingestion_service(
        event,
        config=CampaignWebhookIngestionConfig(soft_bounce_suppression_days=7),
    )

    await service.ingest(body=b"{}", headers={})

    assert suppression_repo.writes[0]["reason"] == "bounce_soft"
    assert suppression_repo.writes[0]["expires_at"].isoformat() == "2026-05-08T12:00:00+00:00"


@pytest.mark.asyncio
async def test_ingestion_records_bounce_even_without_suppression_email():
    service, repo, suppression_repo = _ingestion_service(
        _event("bounced", email=None)
    )

    result = await service.ingest(body=b"{}", headers={})

    assert result.status == "ok"
    assert result.suppressed is False
    assert len(repo.events) == 1
    assert suppression_repo.writes == []
