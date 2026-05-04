from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from extracted_competitive_intelligence._standalone.crm_provider import (
    CRMProvider,
    CRMProviderNotConfigured,
    configure_crm_provider,
    get_crm_provider,
)


class CRMAdapter:
    async def health_check(self) -> bool:
        return True

    async def create_contact(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"id": "contact-1", **data}

    async def find_or_create_contact(
        self,
        full_name: str,
        phone: str | None = None,
        email: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "id": "contact-1",
            "full_name": full_name,
            "phone": phone,
            "email": email,
            **extra,
        }

    async def search_contacts(
        self,
        query: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        business_context_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return [{
            "query": query,
            "phone": phone,
            "email": email,
            "business_context_id": business_context_id,
            "limit": limit,
        }]

    async def log_interaction(
        self,
        contact_id: str,
        interaction_type: str,
        summary: str,
        occurred_at: str | None = None,
        intent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "contact_id": contact_id,
            "interaction_type": interaction_type,
            "summary": summary,
            "occurred_at": occurred_at,
            "intent": intent,
            "metadata": metadata or {},
        }


def teardown_function() -> None:
    configure_crm_provider(None)


def test_standalone_crm_provider_fails_closed_until_configured() -> None:
    configure_crm_provider(None)

    with pytest.raises(CRMProviderNotConfigured):
        get_crm_provider()


def test_standalone_crm_provider_returns_configured_adapter() -> None:
    adapter = CRMAdapter()

    configure_crm_provider(adapter)

    assert isinstance(adapter, CRMProvider)
    assert get_crm_provider() is adapter


def test_service_module_uses_standalone_port_without_atlas(monkeypatch) -> None:
    module_name = "extracted_competitive_intelligence.services.crm_provider"
    atlas_module_name = "atlas_brain.services.crm_provider"
    monkeypatch.setenv("EXTRACTED_COMP_INTEL_STANDALONE", "1")
    sys.modules.pop(module_name, None)
    sys.modules.pop(atlas_module_name, None)

    module = importlib.import_module(module_name)

    try:
        assert module.CRMProvider.__module__ == (
            "extracted_competitive_intelligence._standalone.crm_provider"
        )
        with pytest.raises(module.CRMProviderNotConfigured):
            module.get_crm_provider()

        adapter = CRMAdapter()
        module.configure_crm_provider(adapter)
        assert module.get_crm_provider() is adapter
        assert atlas_module_name not in sys.modules
    finally:
        module.configure_crm_provider(None)
        sys.modules.pop(module_name, None)

