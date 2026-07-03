"""Host wiring for Content Ops FAQ macro writeback providers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import MacroPublishProvider
from extracted_content_pipeline.faq_macro_writeback_postgres import (
    PostgresFAQMacroPublishAttemptRepository,
    PostgresFAQMacroWritebackMappingRepository,
)
from extracted_content_pipeline.faq_macro_writeback_publish import (
    FAQMacroWritebackPublishService,
)
from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZendeskMacroCredentials,
    ZendeskMacroCredentialsProvider,
    ZendeskMacroPublishProvider,
)
from extracted_content_pipeline.ticket_faq_postgres import PostgresTicketFAQRepository


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


@dataclass(frozen=True)
class TenantZendeskMacroCredentialsProvider:
    """Zendesk credential source backed by tenant storage with unscoped fallback."""

    pool: Any
    fallback_provider: ZendeskMacroCredentialsProvider

    async def credentials_for_scope(
        self,
        scope: TenantScope,
    ) -> ZendeskMacroCredentials | None:
        account_id = _scope_text(scope.account_id)
        if account_id:
            from ._content_ops_zendesk_credentials import (
                lookup_zendesk_credentials,
            )

            return await lookup_zendesk_credentials(
                self.pool,
                account_id=account_id,
            )
        if _has_tenant_markers(scope):
            return None
        return await self.fallback_provider.credentials_for_scope(scope)


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
    config_provider_instance = ConfigZendeskMacroCredentialsProvider(config)
    return ZendeskMacroPublishProvider(
        credentials_provider=TenantZendeskMacroCredentialsProvider(
            pool=pool,
            fallback_provider=config_provider_instance,
        ),
        mapping_repository=PostgresFAQMacroWritebackMappingRepository(pool),
    )


async def build_content_ops_faq_macro_publish_service(
    *,
    pool_provider: PoolProvider,
    config_provider: ConfigProvider | None = None,
) -> FAQMacroWritebackPublishService | None:
    """Build the host FAQ macro publish service for paid Resolution Audit reports.

    Composes the existing macro-writeback building blocks (tenant-scoped FAQ
    repository, the host Zendesk publish provider, and the attempt-history
    repository) so paid-report publishing reuses the same credential handling,
    idempotency mappings, and attempt records as the Generated Asset Review
    path. Returns None when the database pool or publish provider is
    unavailable so callers fail closed.
    """

    pool = await _maybe_await(pool_provider())
    if pool is None:
        return None
    if getattr(pool, "is_initialized", True) is False:
        return None
    provider = await build_content_ops_macro_publish_provider(
        pool_provider=lambda: pool,
        config_provider=config_provider,
    )
    if provider is None:
        return None
    return FAQMacroWritebackPublishService(
        faq_repository=PostgresTicketFAQRepository(pool=pool),
        provider=provider,
        attempt_repository=PostgresFAQMacroPublishAttemptRepository(pool),
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


def _scope_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _has_tenant_markers(scope: TenantScope) -> bool:
    if _scope_text(scope.user_id):
        return True
    return any(
        _scope_text(value)
        for values in (scope.allowed_vendors or (), scope.roles or ())
        for value in values
    )


__all__ = [
    "ConfigZendeskMacroCredentialsProvider",
    "TenantZendeskMacroCredentialsProvider",
    "build_content_ops_faq_macro_publish_service",
    "build_content_ops_macro_publish_provider",
    "zendesk_macro_credentials_from_config",
]
