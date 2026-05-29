"""Host wiring for Content Ops FAQ macro writeback providers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import MacroPublishProvider
from extracted_content_pipeline.faq_macro_writeback_postgres import (
    PostgresFAQMacroWritebackMappingRepository,
)
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZendeskMacroCredentials,
    ZendeskMacroPublishProvider,
)


PoolProvider = Callable[[], Any | Awaitable[Any]]
ConfigProvider = Callable[[], Any | Awaitable[Any]]


@dataclass(frozen=True)
class ConfigZendeskMacroCredentialsProvider:
    """Zendesk credential source backed by centralized host config."""

    config: Any

    async def credentials_for_scope(
        self,
        scope: TenantScope,
    ) -> ZendeskMacroCredentials | None:
        del scope
        return zendesk_macro_credentials_from_config(self.config)


async def build_content_ops_macro_publish_provider(
    *,
    pool_provider: PoolProvider,
    config_provider: ConfigProvider | None = None,
) -> MacroPublishProvider | None:
    """Build the host-configured FAQ macro publish provider."""

    pool = await _maybe_await(pool_provider())
    if pool is None:
        return None
    if getattr(pool, "is_initialized", True) is False:
        return None
    config = await _resolve_config(config_provider)
    return ZendeskMacroPublishProvider(
        credentials_provider=ConfigZendeskMacroCredentialsProvider(config),
        mapping_repository=PostgresFAQMacroWritebackMappingRepository(pool),
    )


def zendesk_macro_credentials_from_config(
    config: Any,
) -> ZendeskMacroCredentials | None:
    """Return complete Zendesk macro credentials from centralized config."""

    credentials = ZendeskMacroCredentials(
        email=_config_value(config, "content_ops_zendesk_email"),
        api_token=_config_value(config, "content_ops_zendesk_api_token"),
        subdomain=_config_value(config, "content_ops_zendesk_subdomain"),
        base_url=_config_value(config, "content_ops_zendesk_base_url"),
    )
    return credentials if credentials.is_complete() else None


async def _resolve_config(config_provider: ConfigProvider | None) -> Any:
    if config_provider is not None:
        return await _maybe_await(config_provider())
    from .config import settings

    return settings.b2b_campaign


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _config_value(config: Any, name: str) -> str:
    return str(getattr(config, name, "") or "").strip()


__all__ = [
    "ConfigZendeskMacroCredentialsProvider",
    "build_content_ops_macro_publish_provider",
    "zendesk_macro_credentials_from_config",
]
