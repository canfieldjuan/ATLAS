"""Zendesk macro writeback adapter for approved FAQ macro drafts."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import httpx

from .campaign_ports import JsonDict, TenantScope
from .faq_macro_writeback import (
    MacroPublishResult,
    MacroWritebackMapping,
    MacroWritebackMappingRepository,
    SupportMacroDraft,
)


ZENDESK_PLATFORM = "zendesk"


@dataclass(frozen=True)
class ZendeskMacroCredentials:
    """Tenant Zendesk API credentials."""

    email: str
    api_token: str = field(repr=False)
    subdomain: str = ""
    base_url: str = ""

    def normalized_base_url(self) -> str:
        if cleaned := _clean(self.base_url).rstrip("/"):
            return cleaned
        subdomain = _clean(self.subdomain).removesuffix(".zendesk.com")
        if not subdomain:
            return ""
        return f"https://{subdomain}.zendesk.com"

    def is_complete(self) -> bool:
        return bool(self.normalized_base_url() and _clean(self.email) and _clean(self.api_token))

    def authorization_header(self) -> str:
        token = f"{_clean(self.email)}/token:{_clean(self.api_token)}".encode("utf-8")
        return "Basic " + base64.b64encode(token).decode("ascii")


class ZendeskMacroCredentialsProvider(Protocol):
    """Host-provided scoped Zendesk credential source."""

    async def credentials_for_scope(
        self,
        scope: TenantScope,
    ) -> ZendeskMacroCredentials | None:
        """Return Zendesk credentials for one tenant scope."""


@dataclass(frozen=True)
class StaticZendeskMacroCredentialsProvider:
    """Simple credential provider for tests and single-tenant hosts."""

    credentials: ZendeskMacroCredentials | None

    async def credentials_for_scope(
        self,
        scope: TenantScope,
    ) -> ZendeskMacroCredentials | None:
        _ = scope
        return self.credentials


@dataclass(frozen=True)
class ZendeskMacroTransportError(Exception):
    """Sanitized Zendesk transport failure."""

    status_code: int
    message: str = "zendesk_request_failed"

    def __str__(self) -> str:
        return f"{self.message}: status={self.status_code}"


class ZendeskMacroTransport(Protocol):
    """Transport boundary for Zendesk macro API calls."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Send one Zendesk macro request and return a JSON object."""


class ZendeskHTTPMacroTransport:
    """HTTP transport for Zendesk macro API calls."""

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.request(
                method,
                url,
                headers=dict(headers),
                json=dict(json),
            )
        if response.status_code >= 400:
            raise ZendeskMacroTransportError(status_code=response.status_code)
        payload = response.json()
        return payload if isinstance(payload, Mapping) else {}


@dataclass(frozen=True)
class ZendeskMacroPublishProvider:
    """Zendesk implementation of macro writeback publishing."""

    credentials_provider: ZendeskMacroCredentialsProvider
    mapping_repository: MacroWritebackMappingRepository
    transport: ZendeskMacroTransport = field(default_factory=ZendeskHTTPMacroTransport)

    async def publish(
        self,
        macros: Sequence[SupportMacroDraft],
        *,
        scope: TenantScope,
    ) -> Sequence[MacroPublishResult]:
        credentials = await self.credentials_provider.credentials_for_scope(scope)
        if credentials is None or not credentials.is_complete():
            return tuple(
                MacroPublishResult(
                    macro=macro,
                    status="failed",
                    error="zendesk_credentials_missing",
                )
                for macro in macros
            )
        return tuple([
            await self._publish_one(macro, credentials=credentials, scope=scope)
            for macro in macros
        ])

    async def _publish_one(
        self,
        macro: SupportMacroDraft,
        *,
        credentials: ZendeskMacroCredentials,
        scope: TenantScope,
    ) -> MacroPublishResult:
        if not macro.faq_draft_id or not macro.faq_item_id:
            return MacroPublishResult(
                macro=macro,
                status="failed",
                error="zendesk_macro_missing_faq_identity",
            )
        try:
            existing = await self.mapping_repository.get_mapping(
                platform=ZENDESK_PLATFORM,
                faq_draft_id=macro.faq_draft_id,
                faq_item_id=macro.faq_item_id,
                scope=scope,
            )
            payload = _zendesk_macro_payload(macro)
            if existing is None:
                data = await self.transport.request(
                    "POST",
                    f"{credentials.normalized_base_url()}/api/v2/macros",
                    headers=_headers(credentials),
                    json=payload,
                )
                status = "published"
            else:
                data = await self.transport.request(
                    "PUT",
                    f"{credentials.normalized_base_url()}/api/v2/macros/{existing.external_id}",
                    headers=_headers(credentials),
                    json=payload,
                )
                status = "updated"
            external_id = _external_id(data, fallback=existing.external_id if existing else "")
            if not external_id:
                return MacroPublishResult(
                    macro=macro,
                    status="failed",
                    error="zendesk_macro_id_missing",
                )
            mapping = await self.mapping_repository.upsert_mapping(
                MacroWritebackMapping(
                    platform=ZENDESK_PLATFORM,
                    faq_draft_id=macro.faq_draft_id,
                    faq_item_id=macro.faq_item_id,
                    external_id=external_id,
                    external_url=_external_url(data),
                    metadata={
                        "title": macro.title,
                        "category": macro.category,
                    },
                ),
                scope=scope,
            )
            return MacroPublishResult(
                macro=macro,
                status=status,
                external_id=mapping.external_id,
            )
        except Exception as exc:
            return MacroPublishResult(
                macro=macro,
                status="failed",
                error=_safe_error(exc),
            )


def _zendesk_macro_payload(macro: SupportMacroDraft) -> JsonDict:
    return {
        "macro": {
            "title": macro.title,
            "active": True,
            "description": _description(macro),
            "actions": [
                {"field": "comment_value", "value": macro.body},
                {"field": "comment_mode_is_public", "value": True},
            ],
        }
    }


def _description(macro: SupportMacroDraft) -> str:
    if macro.category:
        return f"FAQ macro generated from {macro.category} support tickets."
    return "FAQ macro generated from support tickets."


def _headers(credentials: ZendeskMacroCredentials) -> dict[str, str]:
    return {
        "Authorization": credentials.authorization_header(),
        "Content-Type": "application/json",
    }


def _external_id(data: Mapping[str, Any], *, fallback: str = "") -> str:
    macro = data.get("macro")
    if isinstance(macro, Mapping):
        return _clean(macro.get("id")) or _clean(fallback)
    return _clean(fallback)


def _external_url(data: Mapping[str, Any]) -> str:
    macro = data.get("macro")
    if isinstance(macro, Mapping):
        return _clean(macro.get("url"))
    return ""


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ZendeskMacroTransportError):
        return str(exc)
    return exc.__class__.__name__


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "StaticZendeskMacroCredentialsProvider",
    "ZENDESK_PLATFORM",
    "ZendeskHTTPMacroTransport",
    "ZendeskMacroCredentials",
    "ZendeskMacroCredentialsProvider",
    "ZendeskMacroPublishProvider",
    "ZendeskMacroTransport",
    "ZendeskMacroTransportError",
]
