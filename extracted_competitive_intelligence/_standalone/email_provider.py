"""Email provider port for standalone competitive intelligence."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class EmailProviderNotConfigured(RuntimeError):
    pass


@runtime_checkable
class EmailProvider(Protocol):
    async def send(
        self,
        to: list[str],
        subject: str,
        body: str,
        from_email: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        reply_to: str | None = None,
        html: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        thread_id: str | None = None,
        in_reply_to: str | None = None,
        references: str | None = None,
    ) -> dict[str, Any]:
        ...


_provider: EmailProvider | None = None


def configure_email_provider(provider: EmailProvider | None) -> None:
    global _provider
    _provider = provider


def get_email_provider() -> EmailProvider:
    if _provider is None:
        raise EmailProviderNotConfigured(
            "Standalone email provider adapter is not configured"
        )
    return _provider

