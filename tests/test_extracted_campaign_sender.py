from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import SendRequest
from extracted_content_pipeline.campaign_sender import (
    RESEND_API_URL,
    ResendCampaignSender,
    ResendSenderConfig,
    SESCampaignSender,
    SESSenderConfig,
    create_campaign_sender,
    normalize_tags,
    sanitize_tag_value,
)


class _Response:
    def __init__(self, payload, *, status_error: Exception | None = None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._payload


class _HTTPClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, *, json, headers):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response


class _SESClient:
    def __init__(self):
        self.calls = []

    def send_email(self, **kwargs):
        self.calls.append(kwargs)
        return {"MessageId": "ses-1"}


def _request(**overrides):
    data = {
        "campaign_id": "campaign-1",
        "to_email": "buyer@example.com",
        "from_email": "seller@example.com",
        "reply_to": "reply@example.com",
        "subject": "Pricing signal",
        "html_body": "<p>Hello</p>",
        "text_body": "Hello",
        "headers": {"List-Unsubscribe": "<https://example.test/unsub>"},
        "tags": (
            {"name": "campaign", "value": "campaign/1"},
            {"name": "mode", "value": "vendor retention"},
        ),
    }
    data.update(overrides)
    return SendRequest(**data)


def test_sanitize_tag_value_replaces_provider_unsafe_characters():
    assert sanitize_tag_value("vendor retention/step 1") == "vendor_retention_step_1"


def test_normalize_tags_drops_blank_and_non_mapping_values():
    assert normalize_tags([
        {"name": "campaign", "value": "campaign/1"},
        {"name": "", "value": "drop"},
        {"name": "drop", "value": ""},
        "bad",
    ]) == [{"name": "campaign", "value": "campaign_1"}]


@pytest.mark.asyncio
async def test_resend_sender_builds_payload_and_returns_message_id():
    client = _HTTPClient(_Response({"id": "resend-1"}))
    sender = ResendCampaignSender(ResendSenderConfig(api_key="re_key"), http_client=client)

    result = await sender.send(_request())

    assert result.provider == "resend"
    assert result.message_id == "resend-1"
    assert result.raw == {"id": "resend-1"}
    assert client.calls == [{
        "url": RESEND_API_URL,
        "json": {
            "from": "seller@example.com",
            "to": ["buyer@example.com"],
            "subject": "Pricing signal",
            "html": "<p>Hello</p>",
            "text": "Hello",
            "reply_to": "reply@example.com",
            "headers": {"List-Unsubscribe": "<https://example.test/unsub>"},
            "tags": [
                {"name": "campaign", "value": "campaign_1"},
                {"name": "mode", "value": "vendor_retention"},
            ],
        },
        "headers": {
            "Authorization": "Bearer re_key",
            "Content-Type": "application/json",
        },
    }]


def test_resend_sender_requires_api_key():
    with pytest.raises(ValueError, match="api_key is required"):
        ResendCampaignSender(ResendSenderConfig(api_key=""))


@pytest.mark.asyncio
async def test_ses_sender_builds_payload_and_returns_message_id():
    client = _SESClient()
    sender = SESCampaignSender(
        SESSenderConfig(
            from_email="default@example.com",
            configuration_set="tracking",
        ),
        client=client,
    )

    result = await sender.send(_request(from_email=""))

    assert result.provider == "ses"
    assert result.message_id == "ses-1"
    assert client.calls == [{
        "FromEmailAddress": "default@example.com",
        "Destination": {"ToAddresses": ["buyer@example.com"]},
        "Content": {
            "Simple": {
                "Subject": {"Data": "Pricing signal", "Charset": "UTF-8"},
                "Body": {
                    "Html": {"Data": "<p>Hello</p>", "Charset": "UTF-8"},
                    "Text": {"Data": "Hello", "Charset": "UTF-8"},
                },
                "Headers": [
                    {"Name": "List-Unsubscribe", "Value": "<https://example.test/unsub>"},
                ],
            },
        },
        "ReplyToAddresses": ["reply@example.com"],
        "ConfigurationSetName": "tracking",
        "EmailTags": [
            {"Name": "campaign", "Value": "campaign_1"},
            {"Name": "mode", "Value": "vendor_retention"},
        ],
    }]


def test_ses_sender_requires_from_email():
    with pytest.raises(ValueError, match="from_email is required"):
        SESCampaignSender(SESSenderConfig(from_email=""), client=_SESClient())


def test_create_campaign_sender_builds_resend_sender():
    sender = create_campaign_sender(
        "resend",
        {"api_key": "re_key", "api_url": "https://example.test/send"},
        http_client=_HTTPClient(_Response({"id": "msg"})),
    )

    assert isinstance(sender, ResendCampaignSender)


def test_create_campaign_sender_builds_ses_sender():
    sender = create_campaign_sender(
        "ses",
        {"from_email": "sender@example.com"},
        ses_client=_SESClient(),
    )

    assert isinstance(sender, SESCampaignSender)


def test_create_campaign_sender_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported campaign sender provider"):
        create_campaign_sender("smtp", {})
