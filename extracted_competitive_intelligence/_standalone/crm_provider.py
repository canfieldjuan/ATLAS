"""CRM provider port for standalone competitive intelligence."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class CRMProviderNotConfigured(RuntimeError):
    pass


@runtime_checkable
class CRMProvider(Protocol):
    async def health_check(self) -> bool:
        ...

    async def create_contact(self, data: dict[str, Any]) -> dict[str, Any]:
        ...

    async def find_or_create_contact(
        self,
        full_name: str,
        phone: str | None = None,
        email: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        ...

    async def search_contacts(
        self,
        query: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        business_context_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        ...

    async def log_interaction(
        self,
        contact_id: str,
        interaction_type: str,
        summary: str,
        occurred_at: str | None = None,
        intent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


_provider: CRMProvider | None = None


def configure_crm_provider(provider: CRMProvider | None) -> None:
    global _provider
    _provider = provider


def get_crm_provider() -> CRMProvider:
    if _provider is None:
        raise CRMProviderNotConfigured(
            "Standalone CRM provider adapter is not configured"
        )
    return _provider

