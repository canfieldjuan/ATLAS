"""Intercom saved-reply writeback adapter for approved FAQ macro drafts."""

from __future__ import annotations

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


INTERCOM_PLATFORM = "intercom"
DEFAULT_INTERCOM_BASE_URL = "https://api.intercom.io"
DEFAULT_INTERCOM_VERSION = "Unstable"


@dataclass(frozen=True)
class IntercomMacroCredentials:
    access_token: str = field(repr=False)
    base_url: str = DEFAULT_INTERCOM_BASE_URL
    intercom_version: str = DEFAULT_INTERCOM_VERSION

    def normalized_base_url(self) -> str:
        return _clean(self.base_url).rstrip("/")

    def is_complete(self) -> bool:
        return bool(self.normalized_base_url() and _clean(self.access_token))


class IntercomMacroCredentialsProvider(Protocol):
    async def credentials_for_scope(self, scope: TenantScope) -> IntercomMacroCredentials | None:
        """Return Intercom credentials for one tenant scope."""


@dataclass(frozen=True)
class StaticIntercomMacroCredentialsProvider:
    credentials: IntercomMacroCredentials | None

    async def credentials_for_scope(self, scope: TenantScope) -> IntercomMacroCredentials | None:
        _ = scope
        return self.credentials


@dataclass(frozen=True)
class IntercomMacroTransportError(Exception):
    status_code: int
    message: str = "intercom_request_failed"

    def __str__(self) -> str:
        return f"{self.message}: status={self.status_code}"


class IntercomMacroTransport(Protocol):
    async def request(self, method: str, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send one Intercom saved-reply request and return a JSON object."""


class IntercomHTTPMacroTransport:
    async def request(self, method: str, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]) -> Mapping[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.request(method, url, headers=dict(headers), json=dict(json))
        if response.status_code >= 400:
            raise IntercomMacroTransportError(status_code=response.status_code)
        payload = response.json()
        return payload if isinstance(payload, Mapping) else {}


@dataclass(frozen=True)
class IntercomMacroPublishProvider:
    credentials_provider: IntercomMacroCredentialsProvider
    mapping_repository: MacroWritebackMappingRepository
    transport: IntercomMacroTransport = field(default_factory=IntercomHTTPMacroTransport)

    async def publish(self, macros: Sequence[SupportMacroDraft], *, scope: TenantScope) -> Sequence[MacroPublishResult]:
        credentials = await self.credentials_provider.credentials_for_scope(scope)
        if credentials is None or not credentials.is_complete():
            return tuple(MacroPublishResult(macro=m, status="failed", error="intercom_credentials_missing") for m in macros)
        return tuple([await self._publish_one(m, credentials=credentials, scope=scope) for m in macros])

    async def _publish_one(self, macro: SupportMacroDraft, *, credentials: IntercomMacroCredentials, scope: TenantScope) -> MacroPublishResult:
        if not macro.faq_draft_id or not macro.faq_item_id:
            return MacroPublishResult(macro=macro, status="failed", error="intercom_macro_missing_faq_identity")
        try:
            existing = await self.mapping_repository.get_mapping(
                platform=INTERCOM_PLATFORM,
                faq_draft_id=macro.faq_draft_id,
                faq_item_id=macro.faq_item_id,
                scope=scope,
            )
            if existing is not None and not existing.external_id:
                return MacroPublishResult(macro=macro, status="failed", error="intercom_macro_mapping_pending_reconcile")
            payload = _payload(macro)
            headers = _headers(credentials, macro=macro, scope=scope)
            if existing is None:
                await self.mapping_repository.reserve_mapping(_mapping(macro, external_id="", publish_status="pending"), scope=scope)
                data = await self.transport.request("POST", f"{credentials.normalized_base_url()}/macros", headers=headers, json=payload)
                status = "published"
            else:
                data = await self.transport.request(
                    "PUT",
                    f"{credentials.normalized_base_url()}/macros/{existing.external_id}",
                    headers=headers,
                    json=payload,
                )
                status = "updated"
            external_id = _external_id(data, fallback=existing.external_id if existing else "")
            if not external_id:
                return MacroPublishResult(macro=macro, status="failed", error="intercom_macro_response_invalid")
            saved = await self.mapping_repository.upsert_mapping(_mapping(macro, external_id=external_id, publish_status="published"), scope=scope)
            return MacroPublishResult(macro=macro, status=status, external_id=saved.external_id)
        except Exception as exc:
            return MacroPublishResult(macro=macro, status="failed", error=_safe_error(exc))


def _payload(macro: SupportMacroDraft) -> JsonDict:
    return {"name": macro.title, "body_text": macro.body, "visible_to": "everyone", "available_on": ["inbox", "messenger"]}


def _headers(credentials: IntercomMacroCredentials, *, macro: SupportMacroDraft, scope: TenantScope) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_clean(credentials.access_token)}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Intercom-Version": _clean(credentials.intercom_version) or DEFAULT_INTERCOM_VERSION,
        "Idempotency-Key": ":".join((
            "atlas-faq-macro-writeback",
            _clean(scope.account_id) or "unscoped",
            _clean(macro.faq_draft_id),
            _clean(macro.faq_item_id),
        )),
    }


def _external_id(data: Mapping[str, Any], *, fallback: str = "") -> str:
    return (_clean(data.get("id")) or _clean(fallback)) if data.get("type") == "macro" else ""


def _mapping(macro: SupportMacroDraft, *, external_id: str, publish_status: str) -> MacroWritebackMapping:
    return MacroWritebackMapping(
        platform=INTERCOM_PLATFORM,
        faq_draft_id=macro.faq_draft_id,
        faq_item_id=macro.faq_item_id,
        external_id=external_id,
        publish_status=publish_status,
        metadata={"title": macro.title, "category": macro.category},
    )


def _safe_error(exc: Exception) -> str:
    return str(exc) if isinstance(exc, IntercomMacroTransportError) else exc.__class__.__name__


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
