"""Atlas-compatible sender seam backed by standalone campaign sender ports."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..campaign_ports import SendRequest
from ..campaign_sender import create_campaign_sender
from ..config import settings


class CampaignSenderAdapter:
    """Adapt the product sender port to Atlas' legacy keyword-call shape."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def send(
        self,
        *,
        to: str,
        from_email: str | None,
        subject: str,
        body: str,
        tags: Sequence[Mapping[str, str]] | None = None,
        text: str | None = None,
        reply_to: str | None = None,
        headers: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        request = SendRequest(
            campaign_id=str((metadata or {}).get("campaign_id") or "vendor_briefing"),
            to_email=str(to),
            from_email=from_email,
            subject=str(subject),
            html_body=str(body),
            text_body=text,
            reply_to=reply_to,
            headers=dict(headers or {}),
            tags=tuple(tags or ()),
            metadata=dict(metadata or {}),
        )
        result = await self._inner.send(request)
        return {
            "id": result.message_id,
            "provider": result.provider,
            "raw": result.raw,
        }


_sender_instance: CampaignSenderAdapter | None = None


def _sender_config_from_settings() -> tuple[str, dict[str, Any]]:
    cfg = settings.campaign_sequence
    sender_type = str(getattr(cfg, "sender_type", "resend") or "resend").lower()
    if sender_type == "ses":
        return (
            "ses",
            {
                "from_email": getattr(cfg, "ses_from_email", "") or getattr(
                    cfg,
                    "resend_from_email",
                    "",
                ),
                "region": getattr(cfg, "ses_region", "us-east-1"),
                "access_key_id": getattr(cfg, "ses_access_key_id", "") or None,
                "secret_access_key": getattr(cfg, "ses_secret_access_key", "") or None,
                "configuration_set": getattr(cfg, "ses_configuration_set", "") or None,
            },
        )

    api_key = getattr(cfg, "resend_api_key", "") or ""
    if not api_key:
        raise RuntimeError(
            "settings.campaign_sequence.resend_api_key is required when "
            "sender_type=resend; set EXTRACTED_RESEND_API_KEY, "
            "EXTRACTED_CAMPAIGN_RESEND_API_KEY, or "
            "EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY"
        )
    return (
        "resend",
        {
            "api_key": api_key,
            "api_url": getattr(cfg, "resend_api_url", "") or None,
            "timeout_seconds": getattr(cfg, "sender_timeout_seconds", 30.0),
        },
    )


def get_campaign_sender() -> CampaignSenderAdapter:
    """Return the configured standalone sender behind the legacy API."""
    global _sender_instance
    if _sender_instance is None:
        provider, config = _sender_config_from_settings()
        _sender_instance = CampaignSenderAdapter(create_campaign_sender(provider, config))
    return _sender_instance


def reset_campaign_sender_for_tests() -> None:
    """Clear the cached singleton between tests."""
    global _sender_instance
    _sender_instance = None
