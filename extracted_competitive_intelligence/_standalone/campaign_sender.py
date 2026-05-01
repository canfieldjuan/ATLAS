"""Campaign sender port for standalone competitive intelligence."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class CampaignSenderNotConfigured(RuntimeError):
    pass


@runtime_checkable
class CampaignSender(Protocol):
    async def send(
        self,
        *,
        to: str,
        from_email: str,
        subject: str,
        body: str,
        tags: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        ...


_sender: CampaignSender | None = None


def configure_campaign_sender(sender: CampaignSender | None) -> None:
    global _sender
    _sender = sender


def get_campaign_sender() -> CampaignSender:
    if _sender is None:
        raise CampaignSenderNotConfigured(
            "Standalone campaign sender adapter is not configured"
        )
    return _sender

