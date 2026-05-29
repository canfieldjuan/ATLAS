from __future__ import annotations

from dataclasses import dataclass

import pytest

from atlas_brain.config import B2BCampaignConfig
from atlas_brain._content_ops_macro_writeback import (
    ConfigZendeskMacroCredentialsProvider,
    TenantZendeskMacroCredentialsProvider,
    build_content_ops_macro_publish_provider,
    zendesk_macro_credentials_from_config,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback_postgres import (
    PostgresFAQMacroWritebackMappingRepository,
)
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZendeskMacroCredentials,
    ZendeskMacroPublishProvider,
)


class _Pool:
    is_initialized = True

    async def fetchrow(self, query, *args):
        return None

    async def execute(self, query, *args):
        return None


@dataclass(frozen=True)
class _Config:
    content_ops_zendesk_email: str = ""
    content_ops_zendesk_api_token: str = ""
    content_ops_zendesk_subdomain: str = ""
    content_ops_zendesk_base_url: str = ""


def test_zendesk_macro_credentials_from_config() -> None:
    credentials = zendesk_macro_credentials_from_config(_Config(
        content_ops_zendesk_email=" agent@example.com ",
        content_ops_zendesk_api_token=" token ",
        content_ops_zendesk_subdomain="acme",
    ))

    assert credentials is not None
    assert credentials.email == "agent@example.com"
    assert credentials.normalized_base_url() == "https://acme.zendesk.com"
    assert "token" not in repr(credentials)


def test_b2b_campaign_config_accepts_content_ops_zendesk_env_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ATLAS_CONTENT_OPS_ZENDESK_EMAIL",
        "ATLAS_CONTENT_OPS_ZENDESK_API_TOKEN",
        "ATLAS_CONTENT_OPS_ZENDESK_SUBDOMAIN",
        "ATLAS_CONTENT_OPS_ZENDESK_BASE_URL",
        "ZENDESK_EMAIL",
        "ZENDESK_API_TOKEN",
        "ZENDESK_SUBDOMAIN",
        "ZENDESK_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZENDESK_EMAIL", "agent@example.com")
    monkeypatch.setenv("ZENDESK_API_TOKEN", "token")
    monkeypatch.setenv("ZENDESK_BASE_URL", "https://acme.zendesk.com/")

    config = B2BCampaignConfig(_env_file=None)
    credentials = zendesk_macro_credentials_from_config(config)

    assert credentials is not None
    assert credentials.normalized_base_url() == "https://acme.zendesk.com"


@pytest.mark.asyncio
async def test_config_zendesk_credentials_provider_returns_none_when_incomplete() -> None:
    provider = ConfigZendeskMacroCredentialsProvider(_Config(
        content_ops_zendesk_email="agent@example.com",
    ))

    assert await provider.credentials_for_scope(TenantScope(account_id="acct-1")) is None


@pytest.mark.asyncio
async def test_build_content_ops_macro_publish_provider_uses_pool_mapping_repo() -> None:
    pool = _Pool()

    async def pool_provider():
        return pool

    provider = await build_content_ops_macro_publish_provider(
        pool_provider=pool_provider,
        config_provider=lambda: _Config(
            content_ops_zendesk_email="agent@example.com",
            content_ops_zendesk_api_token="token",
            content_ops_zendesk_subdomain="acme",
        ),
    )

    assert isinstance(provider, ZendeskMacroPublishProvider)
    assert isinstance(provider.credentials_provider, TenantZendeskMacroCredentialsProvider)
    assert isinstance(
        provider.mapping_repository,
        PostgresFAQMacroWritebackMappingRepository,
    )
    assert provider.mapping_repository.pool is pool
    credentials = await provider.credentials_provider.credentials_for_scope(
        TenantScope(account_id="acct-1")
    )
    assert credentials is not None
    assert credentials.email == "agent@example.com"


@pytest.mark.asyncio
async def test_build_content_ops_macro_publish_provider_returns_none_without_pool() -> None:
    async def pool_provider():
        return None

    assert await build_content_ops_macro_publish_provider(
        pool_provider=pool_provider,
        config_provider=lambda: _Config(),
    ) is None


@pytest.mark.asyncio
async def test_build_content_ops_macro_publish_provider_returns_none_for_uninitialized_pool() -> None:
    class _UninitializedPool:
        is_initialized = False

    assert await build_content_ops_macro_publish_provider(
        pool_provider=lambda: _UninitializedPool(),
        config_provider=lambda: _Config(),
    ) is None


@pytest.mark.asyncio
async def test_tenant_zendesk_credentials_provider_prefers_tenant_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.services import content_ops_zendesk_credentials

    tenant_credentials = ZendeskMacroCredentials(
        email="tenant@example.com",
        api_token="tenant-token",
        subdomain="tenant",
    )
    calls: list[dict] = []

    async def lookup(pool, *, account_id):
        calls.append({"pool": pool, "account_id": account_id})
        return tenant_credentials

    monkeypatch.setattr(content_ops_zendesk_credentials, "lookup_zendesk_credentials", lookup)
    pool = _Pool()
    fallback = ConfigZendeskMacroCredentialsProvider(_Config(
        content_ops_zendesk_email="fallback@example.com",
        content_ops_zendesk_api_token="fallback-token",
        content_ops_zendesk_subdomain="fallback",
    ))
    provider = TenantZendeskMacroCredentialsProvider(
        pool=pool,
        fallback_provider=fallback,
    )

    credentials = await provider.credentials_for_scope(
        TenantScope(account_id="11111111-1111-1111-1111-111111111111")
    )

    assert credentials is tenant_credentials
    assert calls == [{
        "pool": pool,
        "account_id": "11111111-1111-1111-1111-111111111111",
    }]


@pytest.mark.asyncio
async def test_tenant_zendesk_credentials_provider_falls_back_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.services import content_ops_zendesk_credentials

    async def lookup(pool, *, account_id):
        return None

    monkeypatch.setattr(content_ops_zendesk_credentials, "lookup_zendesk_credentials", lookup)
    provider = TenantZendeskMacroCredentialsProvider(
        pool=_Pool(),
        fallback_provider=ConfigZendeskMacroCredentialsProvider(_Config(
            content_ops_zendesk_email="fallback@example.com",
            content_ops_zendesk_api_token="fallback-token",
            content_ops_zendesk_subdomain="fallback",
        )),
    )

    credentials = await provider.credentials_for_scope(
        TenantScope(account_id="11111111-1111-1111-1111-111111111111")
    )

    assert credentials is not None
    assert credentials.email == "fallback@example.com"
    assert credentials.normalized_base_url() == "https://fallback.zendesk.com"
