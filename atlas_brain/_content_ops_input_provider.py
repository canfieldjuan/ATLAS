"""Atlas host input provider for Content Ops support-ticket requests."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_input_provider import (
    ContentOpsInputPackage,
    RequestPayload,
)
from extracted_content_pipeline.faq_output_ingestion import (
    FAQ_OUTPUT_SOURCE_TYPE,
    is_faq_output_bundle,
)
from extracted_content_pipeline.support_ticket_input_provider import (
    SupportTicketInputProvider,
)


_SUPPORT_TICKET_BUNDLE_KEYS = frozenset({
    "support_tickets",
    "tickets",
    "cases",
    "conversations",
})
_SUPPORT_TICKET_ROW_KEYS = frozenset({
    "source_id",
    "ticket_id",
    "id",
    "case_id",
    "conversation_id",
    "requester_email",
})
_SOURCE_TITLE_KEYS = frozenset({
    "source_title",
    "subject",
    "title",
    "summary",
})
_SOURCE_TEXT_KEYS = frozenset({
    "text",
    "body",
    "description",
    "message",
    "content",
    "complaint",
    "notes",
    "summary",
})
_SUPPORT_TICKET_SOURCE_TYPES = frozenset({
    "ticket",
    "support_ticket",
    "support-ticket",
    "case",
    "chat",
    "chat_transcript",
    "conversation",
    "transcript",
})


@dataclass(frozen=True)
class _AtlasSupportTicketInputProvider:
    """Request-aware provider used by the Atlas Content Ops router mount."""

    provider_name: str = "atlas_support_ticket_request"

    def build_content_ops_input_package(
        self,
        *,
        scope: TenantScope,
        request: RequestPayload | None = None,
    ) -> ContentOpsInputPackage:
        source_material = _request_source_material(request)
        if _is_empty_source_material(source_material) or not _is_support_ticket_material(
            source_material
        ):
            return _noop_package(self.provider_name)
        return SupportTicketInputProvider(
            source_material=source_material,
            provider=self.provider_name,
        ).build_content_ops_input_package(
            scope=scope,
            request=request,
        )


def build_content_ops_input_provider() -> _AtlasSupportTicketInputProvider:
    """Return the host's request-aware Content Ops input provider."""

    return _AtlasSupportTicketInputProvider()


def _request_source_material(request: RequestPayload | None) -> Any:
    if not isinstance(request, Mapping):
        return None
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    return inputs.get("source_material")


def _is_empty_source_material(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, (list, tuple, set)):
        return not value or all(_is_empty_source_material(item) for item in value)
    return False


def _is_support_ticket_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return (
            is_faq_output_bundle(value)
            or _is_support_ticket_bundle(value)
            or _is_support_ticket_row(value)
        )
    if isinstance(value, (list, tuple)):
        return any(_is_support_ticket_material(item) for item in value)
    return False


def _is_support_ticket_bundle(value: Mapping[str, Any]) -> bool:
    for value_key, item in value.items():
        if _key(value_key) not in _SUPPORT_TICKET_BUNDLE_KEYS:
            continue
        if _is_empty_source_material(item):
            continue
        if isinstance(item, Mapping):
            return _is_support_ticket_row(item)
        if isinstance(item, (list, tuple)):
            return any(
                isinstance(row, Mapping) and _is_support_ticket_row(row)
                for row in item
            )
    return False


def _is_support_ticket_row(value: Mapping[str, Any]) -> bool:
    keys = {_key(item) for item in value}
    source_type = _key(value.get("source_type") or value.get("type"))
    if source_type == FAQ_OUTPUT_SOURCE_TYPE:
        return _has_any_text(value, _SOURCE_TITLE_KEYS) or _has_any_text(
            value,
            _SOURCE_TEXT_KEYS,
        )
    if source_type and source_type not in _SUPPORT_TICKET_SOURCE_TYPES:
        return False
    has_title = _has_any_text(value, _SOURCE_TITLE_KEYS)
    has_body = _has_any_text(value, _SOURCE_TEXT_KEYS)
    if source_type in _SUPPORT_TICKET_SOURCE_TYPES:
        return has_title or has_body
    if keys & _SUPPORT_TICKET_ROW_KEYS:
        return has_title or has_body
    return has_title and has_body


def _has_any_text(value: Mapping[str, Any], keys: frozenset[str]) -> bool:
    return any(
        _key(row_key) in keys and _clean(row_value)
        for row_key, row_value in value.items()
    )


def _key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _noop_package(provider: str) -> ContentOpsInputPackage:
    return ContentOpsInputPackage(
        provider=provider,
        inputs={},
        outputs=(),
        target_mode="",
        ingestion_profile="",
        metadata={"source": "atlas_content_ops_input_provider", "mode": "noop"},
    )


__all__ = ["build_content_ops_input_provider"]
