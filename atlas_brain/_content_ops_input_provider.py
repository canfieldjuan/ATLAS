"""Atlas host input provider for Content Ops support-ticket requests."""

from __future__ import annotations
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from extracted_content_pipeline.campaign_postgres import PostgresIntelligenceRepository
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
from extracted_content_pipeline.ticket_faq_postgres import PostgresTicketFAQRepository


PoolProvider = Callable[[], Any | Awaitable[Any]]
FAQRepositoryFactory = Callable[[Any], Any]
OpportunityRepositoryFactory = Callable[[Any], Any]

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
_SOURCE_FAQ_ID_KEYS = (
    "source_faq_ids",
    "source_faq_draft_ids",
    "selected_faq_draft_ids",
)
_SOURCE_IMPORT_TARGET_ID_KEYS = (
    "source_import_target_ids",
    "source_target_ids",
    "import_target_ids",
)


@dataclass(frozen=True)
class _AtlasSupportTicketInputProvider:
    """Request-aware provider used by the Atlas Content Ops router mount."""

    provider_name: str = "atlas_support_ticket_request"
    pool_provider: PoolProvider | None = None
    faq_repository_factory: FAQRepositoryFactory = PostgresTicketFAQRepository
    opportunity_repository_factory: OpportunityRepositoryFactory = (
        PostgresIntelligenceRepository
    )

    def build_content_ops_input_package(
        self,
        *,
        scope: TenantScope,
        request: RequestPayload | None = None,
    ) -> ContentOpsInputPackage | Awaitable[ContentOpsInputPackage]:
        source_material = _request_source_material(request)
        source_faq_ids = _request_source_faq_ids(request)
        source_target_ids = _request_source_target_ids(request)
        if source_faq_ids or source_target_ids:
            return self._build_from_selected_sources(
                source_faq_ids,
                source_target_ids,
                source_material=source_material,
                scope=scope,
                request=request,
            )
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

    async def _build_from_selected_sources(
        self,
        source_faq_ids: Sequence[str],
        source_target_ids: Sequence[str],
        *,
        source_material: Any,
        scope: TenantScope,
        request: RequestPayload | None,
    ) -> ContentOpsInputPackage:
        valid_faq_ids, invalid_faq_ids = _partition_source_faq_ids(source_faq_ids)
        loaded_faqs, missing_faqs = await self._load_selected_faq_drafts(
            valid_faq_ids,
            scope=scope,
        )
        (
            loaded_targets,
            missing_targets,
            ambiguous_targets,
            target_skip_reason,
        ) = await self._load_selected_import_targets(
            source_target_ids,
            scope=scope,
            request=request,
        )
        combined_source_material = _combine_source_material(
            source_material,
            [*loaded_targets, *loaded_faqs],
        )
        metadata = {
            "selected_faq_id_count": len(source_faq_ids),
            "selected_faq_loaded_count": len(loaded_faqs),
            "selected_faq_missing_id_count": len(missing_faqs),
            "selected_faq_invalid_id_count": len(invalid_faq_ids),
            "source_target_id_count": len(source_target_ids),
            "source_target_loaded_count": len(loaded_targets),
            "source_target_missing_id_count": len(missing_targets),
            "source_target_ambiguous_id_count": len(ambiguous_targets),
        }
        warnings = (
            *_selected_faq_warnings(
                invalid=invalid_faq_ids,
                missing=missing_faqs,
                repository_configured=self.pool_provider is not None,
            ),
            *_selected_source_target_warnings(
                missing=missing_targets,
                ambiguous=ambiguous_targets,
                skip_reason=target_skip_reason,
            ),
        )
        if _is_empty_source_material(combined_source_material) or not _is_support_ticket_material(
            combined_source_material
        ):
            return _noop_package(
                self.provider_name,
                metadata=metadata,
                warnings=warnings,
            )
        package = SupportTicketInputProvider(
            source_material=combined_source_material,
            provider=self.provider_name,
            metadata=metadata,
        ).build_content_ops_input_package(
            scope=scope,
            request=request,
        )
        if hasattr(package, "__await__"):
            package = await package
        if not warnings:
            return package
        return ContentOpsInputPackage(
            provider=package.provider,
            inputs=package.inputs,
            outputs=package.outputs,
            target_mode=package.target_mode,
            ingestion_profile=package.ingestion_profile,
            metadata=package.metadata,
            warnings=tuple(package.warnings) + tuple(warnings),
        )

    async def _load_selected_faq_drafts(
        self,
        source_faq_ids: Sequence[str],
        *,
        scope: TenantScope,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if self.pool_provider is None:
            return [], list(source_faq_ids)
        pool = self.pool_provider()
        if hasattr(pool, "__await__"):
            pool = await pool
        repository = self.faq_repository_factory(pool)
        loaded: list[dict[str, Any]] = []
        missing: list[str] = []
        for faq_id in source_faq_ids:
            draft = await repository.get_draft(faq_id, scope=scope)
            if draft is None:
                missing.append(faq_id)
                continue
            loaded.append(draft.as_dict())
        return loaded, missing

    async def _load_selected_import_targets(
        self,
        source_target_ids: Sequence[str],
        *,
        scope: TenantScope,
        request: RequestPayload | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str], str | None]:
        if not source_target_ids:
            return [], [], [], None
        if not _clean(getattr(scope, "account_id", None)):
            return [], list(source_target_ids), [], "missing_account_scope"
        if self.pool_provider is None:
            return [], list(source_target_ids), [], "repository_unconfigured"
        pool = self.pool_provider()
        if hasattr(pool, "__await__"):
            pool = await pool
        repository = self.opportunity_repository_factory(pool)
        target_mode = _request_target_mode(request)
        loaded: list[dict[str, Any]] = []
        missing: list[str] = []
        ambiguous: list[str] = []
        for target_id in source_target_ids:
            rows = await repository.read_campaign_opportunities(
                scope=scope,
                target_mode=target_mode,
                limit=2,
                filters={"target_id": target_id},
            )
            if len(rows) == 1:
                loaded.append(dict(rows[0]))
            elif len(rows) > 1:
                ambiguous.append(target_id)
            else:
                missing.append(target_id)
        return loaded, missing, ambiguous, None


def build_content_ops_input_provider(
    *,
    pool_provider: PoolProvider | None = None,
    faq_repository_factory: FAQRepositoryFactory = PostgresTicketFAQRepository,
    opportunity_repository_factory: OpportunityRepositoryFactory = (
        PostgresIntelligenceRepository
    ),
) -> _AtlasSupportTicketInputProvider:
    """Return the host's request-aware Content Ops input provider."""

    return _AtlasSupportTicketInputProvider(
        pool_provider=pool_provider,
        faq_repository_factory=faq_repository_factory,
        opportunity_repository_factory=opportunity_repository_factory,
    )


def _request_source_material(request: RequestPayload | None) -> Any:
    if not isinstance(request, Mapping):
        return None
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    return inputs.get("source_material")


def _request_source_faq_ids(request: RequestPayload | None) -> tuple[str, ...]:
    if not isinstance(request, Mapping):
        return ()
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        return ()
    for key in _SOURCE_FAQ_ID_KEYS:
        values = _string_values(inputs.get(key))
        if values:
            return tuple(values)
    return ()


def _request_source_target_ids(request: RequestPayload | None) -> tuple[str, ...]:
    if not isinstance(request, Mapping):
        return ()
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        return ()
    for key in _SOURCE_IMPORT_TARGET_ID_KEYS:
        values = _string_values(inputs.get(key))
        if values:
            return tuple(values)
    return ()


def _request_target_mode(request: RequestPayload | None) -> str:
    if not isinstance(request, Mapping):
        return "vendor_retention"
    return _clean(request.get("target_mode")) or "vendor_retention"


def _string_values(value: Any) -> tuple[str, ...]:
    if value in (None, "", [], {}):
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    out: list[str] = []
    for item in values:
        text = _clean(item)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _partition_source_faq_ids(values: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    valid: list[str] = []
    invalid: list[str] = []
    for value in values:
        try:
            UUID(value)
        except ValueError:
            invalid.append(value)
        else:
            valid.append(value)
    return tuple(valid), tuple(invalid)


def _combine_source_material(source_material: Any, selected_faq_outputs: Sequence[Any]) -> Any:
    if not selected_faq_outputs:
        return source_material
    if _is_empty_source_material(source_material):
        return list(selected_faq_outputs)
    if isinstance(source_material, (list, tuple)):
        return [*source_material, *selected_faq_outputs]
    return [source_material, *selected_faq_outputs]


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


def _selected_faq_warnings(
    *,
    invalid: Sequence[str],
    missing: Sequence[str],
    repository_configured: bool,
) -> tuple[dict[str, Any], ...]:
    warnings: list[dict[str, Any]] = []
    if invalid:
        warnings.append({
            "code": "source_faq_ids_invalid",
            "message": "One or more selected FAQ report IDs are invalid.",
            "invalid_count": len(invalid),
        })
    if not repository_configured:
        warnings.append({
            "code": "source_faq_repository_unconfigured",
            "message": "Saved FAQ source selection is not configured for this route.",
        })
    elif missing:
        warnings.append({
            "code": "source_faq_drafts_not_found",
            "message": "One or more selected FAQ reports were not found for this account.",
            "missing_count": len(missing),
        })
    return tuple(warnings)


def _selected_source_target_warnings(
    *,
    missing: Sequence[str],
    ambiguous: Sequence[str],
    skip_reason: str | None,
) -> tuple[dict[str, Any], ...]:
    warnings: list[dict[str, Any]] = []
    if skip_reason == "missing_account_scope":
        warnings.append({
            "code": "source_import_targets_unscoped",
            "message": (
                "Persisted import target IDs require an account-scoped Content "
                "Ops request."
            ),
            "missing_count": len(missing),
        })
        return tuple(warnings)
    if skip_reason == "repository_unconfigured":
        warnings.append({
            "code": "source_import_target_repository_unconfigured",
            "message": "Persisted import target selection is not configured for this route.",
            "missing_count": len(missing),
        })
        return tuple(warnings)
    if missing:
        warnings.append({
            "code": "source_import_targets_not_found",
            "message": "One or more persisted import target IDs were not found for this account.",
            "missing_count": len(missing),
        })
    if ambiguous:
        warnings.append({
            "code": "source_import_targets_ambiguous",
            "message": "One or more persisted import target IDs matched multiple rows.",
            "ambiguous_count": len(ambiguous),
        })
    return tuple(warnings)


def _noop_package(
    provider: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    warnings: Sequence[Mapping[str, Any]] = (),
) -> ContentOpsInputPackage:
    return ContentOpsInputPackage(
        provider=provider,
        inputs={},
        outputs=(),
        target_mode="",
        ingestion_profile="",
        metadata={
            "source": "atlas_content_ops_input_provider",
            "mode": "noop",
            **dict(metadata or {}),
        },
        warnings=tuple(dict(warning) for warning in warnings),
    )


__all__ = ["build_content_ops_input_provider"]
