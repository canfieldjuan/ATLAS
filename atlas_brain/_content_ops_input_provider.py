"""Atlas host input provider for Content Ops support-ticket requests."""

from __future__ import annotations
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any
from uuid import UUID

from extracted_content_pipeline.campaign_postgres import PostgresIntelligenceRepository
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_source_adapters import (
    source_material_to_source_rows,
    source_rows_to_campaign_opportunities,
)
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
_B2B_DISPLACEMENT_VENDOR_KEYS = (
    "b2b_displacement_vendors",
    "b2b_displacement_vendor_names",
    "competitive_vendor_names",
    "source_b2b_displacement_vendors",
)
_B2B_DISPLACEMENT_LIMIT_KEYS = (
    "b2b_displacement_limit",
    "competitive_displacement_limit",
)
_B2B_DISPLACEMENT_DEFAULT_LIMIT = 10
_B2B_DISPLACEMENT_MAX_LIMIT = 25
_SOURCE_TYPE_KEYS = ("source_type", "source_material_type")
_SUPPORT_TICKET_SOURCE_TYPE_ALIASES = frozenset({
    "support",
    "support_ticket",
    "support_tickets",
    "ticket",
    "tickets",
})
_REVIEW_SOURCE_TYPE_ALIASES = frozenset({
    "review",
    "reviews",
    "complaint",
    "complaints",
})
_COMPETITIVE_SOURCE_TYPE_ALIASES = frozenset({
    "competitive",
    "competition",
    "competitor",
    "competitors",
    "competitive_displacement",
    "competitive_signal",
    "competitive_signals",
    "displacement",
    "displacement_edge",
    "displacement_edges",
})
_COMPETITIVE_BUNDLE_KEYS = frozenset({
    "competitive_inputs",
    "competitive_rows",
    "competitive_signal",
    "competitive_signals",
    "displacement_edge",
    "displacement_edges",
    "displacements",
    "switching_evidence",
})
_COMPETITIVE_MARKER_KEYS = frozenset({
    "alternative",
    "alternative_name",
    "alternative_product",
    "alternative_vendor",
    "alternatives",
    "competitive_alternative",
    "competitive_alternatives",
    "competitive_displacement",
    "competitor",
    "competitor_name",
    "competitor_overlap",
    "competitor_vendor",
    "competitor_vendor_name",
    "competitors",
    "competitors_mentioned",
    "displacement_confidence",
    "displacement_direction",
    "displacement_map",
    "displacement_mention_count",
    "from_vendor",
    "losing_vendor",
    "new_vendor",
    "switched_from",
    "switched_to",
    "switching_from",
    "switching_to",
    "to_vendor",
    "top_displacement_targets",
    "winning_vendor",
})
_REVIEW_CAMPAIGN_OUTPUTS = (
    "landing_page",
    "blog_post",
    "sales_brief",
    "social_post",
    "ad_copy",
    "quote_card",
)
_REVIEW_CAMPAIGN_NAME = "Review-Signal Campaign"
_REVIEW_TOPIC = "Customer review themes worth turning into content"
_REVIEW_AUDIENCE = "Marketing teams turning customer reviews into grounded content"
_REVIEW_OFFER = (
    "Turn customer review themes into buyer-facing landing pages, blog posts, "
    "sales briefs, social posts, ad copy, and quote cards"
)
_REVIEW_TARGET_KEYWORD = "customer review content marketing"
_COMPETITIVE_CAMPAIGN_OUTPUTS = (
    "landing_page",
    "blog_post",
    "sales_brief",
    "social_post",
    "ad_copy",
    "quote_card",
)
_COMPETITIVE_CAMPAIGN_NAME = "Competitive Displacement Campaign"
_COMPETITIVE_TOPIC = "Competitive displacement themes worth turning into content"
_COMPETITIVE_AUDIENCE = (
    "Marketing teams turning competitive evidence into grounded content"
)
_COMPETITIVE_OFFER = (
    "Turn competitor and switching evidence into buyer-facing landing pages, "
    "blog posts, sales briefs, social posts, ad copy, and quote cards"
)
_COMPETITIVE_TARGET_KEYWORD = "competitive displacement content marketing"


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
        source_type = _request_source_type(request)
        source_faq_ids = _request_source_faq_ids(request)
        source_target_ids = _request_source_target_ids(request)
        b2b_displacement_vendors = _request_b2b_displacement_vendors(request)
        if source_type in _REVIEW_SOURCE_TYPE_ALIASES:
            if source_faq_ids:
                return _noop_package(
                    "atlas_review_request",
                    metadata={"requested_source_type": source_type},
                    warnings=({
                        "code": "review_source_faq_ids_unsupported",
                        "message": "Saved FAQ source selection is only supported for support-ticket runs.",
                    },),
                )
            if source_target_ids:
                return self._build_reviews_from_selected_sources(
                    source_target_ids,
                    source_material=source_material,
                    scope=scope,
                    request=request,
                )
            return _build_review_input_package(
                source_material,
                metadata={"requested_source_type": source_type},
                default_source_type=source_type,
            )
        if source_type in _COMPETITIVE_SOURCE_TYPE_ALIASES:
            if source_faq_ids:
                return _noop_package(
                    "atlas_competitive_request",
                    metadata={"requested_source_type": source_type},
                    warnings=({
                        "code": "competitive_source_faq_ids_unsupported",
                        "message": "Saved FAQ source selection is only supported for support-ticket runs.",
                    },),
                )
            if source_target_ids or b2b_displacement_vendors:
                return self._build_competitive_from_selected_sources(
                    source_target_ids,
                    b2b_displacement_vendors=b2b_displacement_vendors,
                    source_material=source_material,
                    scope=scope,
                    request=request,
                )
            return _build_competitive_input_package(
                source_material,
                metadata={"requested_source_type": source_type},
                default_source_type=source_type,
            )
        if source_type and source_type not in _SUPPORT_TICKET_SOURCE_TYPE_ALIASES:
            return _noop_package(
                self.provider_name,
                metadata={"requested_source_type": source_type},
                warnings=({
                    "code": "content_ops_source_type_unsupported",
                    "message": "Unsupported Content Ops source type.",
                    "source_type": source_type,
                },),
            )
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

    async def _build_reviews_from_selected_sources(
        self,
        source_target_ids: Sequence[str],
        *,
        source_material: Any,
        scope: TenantScope,
        request: RequestPayload | None,
    ) -> ContentOpsInputPackage:
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
            loaded_targets,
        )
        requested_source_type = _request_source_type(request) or "reviews"
        metadata = {
            "requested_source_type": requested_source_type,
            "source_target_id_count": len(source_target_ids),
            "source_target_loaded_count": len(loaded_targets),
            "source_target_missing_id_count": len(missing_targets),
            "source_target_ambiguous_id_count": len(ambiguous_targets),
        }
        warnings = _selected_source_target_warnings(
            missing=missing_targets,
            ambiguous=ambiguous_targets,
            skip_reason=target_skip_reason,
        )
        return _build_review_input_package(
            combined_source_material,
            metadata=metadata,
            warnings=warnings,
            default_source_type=requested_source_type,
        )

    async def _build_competitive_from_selected_sources(
        self,
        source_target_ids: Sequence[str],
        *,
        b2b_displacement_vendors: Sequence[str] = (),
        source_material: Any,
        scope: TenantScope,
        request: RequestPayload | None,
    ) -> ContentOpsInputPackage:
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
        (
            loaded_b2b_displacement,
            missing_b2b_displacement,
            b2b_displacement_skip_reason,
        ) = await self._load_b2b_displacement_dynamics(
            b2b_displacement_vendors,
            scope=scope,
            limit=_request_b2b_displacement_limit(request),
        )
        combined_source_material = _combine_source_material(
            source_material,
            [*loaded_targets, *loaded_b2b_displacement],
        )
        requested_source_type = _request_source_type(request) or "competitive"
        metadata = {
            "requested_source_type": requested_source_type,
            "source_target_id_count": len(source_target_ids),
            "source_target_loaded_count": len(loaded_targets),
            "source_target_missing_id_count": len(missing_targets),
            "source_target_ambiguous_id_count": len(ambiguous_targets),
            "b2b_displacement_vendor_count": len(b2b_displacement_vendors),
            "b2b_displacement_loaded_count": len(loaded_b2b_displacement),
            "b2b_displacement_missing_vendor_count": len(missing_b2b_displacement),
        }
        warnings = (
            *_selected_source_target_warnings(
                missing=missing_targets,
                ambiguous=ambiguous_targets,
                skip_reason=target_skip_reason,
            ),
            *_selected_b2b_displacement_warnings(
                missing=missing_b2b_displacement,
                skip_reason=b2b_displacement_skip_reason,
            ),
        )
        return _build_competitive_input_package(
            combined_source_material,
            metadata=metadata,
            warnings=warnings,
            default_source_type=requested_source_type,
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

    async def _load_b2b_displacement_dynamics(
        self,
        vendors: Sequence[str],
        *,
        scope: TenantScope,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[str], str | None]:
        vendor_names = tuple(_string_values(vendors))
        if not vendor_names:
            return [], [], None
        account_id = _clean(getattr(scope, "account_id", None))
        if not account_id:
            return [], list(vendor_names), "missing_account_scope"
        if self.pool_provider is None:
            return [], list(vendor_names), "repository_unconfigured"
        pool = self.pool_provider()
        if hasattr(pool, "__await__"):
            pool = await pool
        normalized_vendors = [_lookup_text(vendor) for vendor in vendor_names]
        try:
            rows = await pool.fetch(
                """
                SELECT DISTINCT ON (d.from_vendor, d.to_vendor)
                       d.from_vendor,
                       d.to_vendor,
                       d.as_of_date,
                       d.analysis_window_days,
                       d.schema_version,
                       d.dynamics,
                       d.created_at
                FROM b2b_displacement_dynamics d
                WHERE lower(BTRIM(d.from_vendor)) = ANY($2::text[])
                  AND EXISTS (
                      SELECT 1
                      FROM tracked_vendors tv
                      WHERE tv.account_id = $1
                        AND lower(BTRIM(tv.vendor_name)) = lower(BTRIM(d.from_vendor))
                  )
                ORDER BY d.from_vendor, d.to_vendor, d.as_of_date DESC, d.created_at DESC
                LIMIT $3
                """,
                account_id,
                normalized_vendors,
                max(1, min(limit, _B2B_DISPLACEMENT_MAX_LIMIT)),
            )
        except Exception:
            return [], list(vendor_names), "query_failed"

        source_rows: list[dict[str, Any]] = []
        loaded_vendors: set[str] = set()
        for raw in rows:
            row = _b2b_displacement_dynamics_source_row(dict(raw))
            if not row:
                continue
            source_rows.append(row)
            loaded_vendors.add(_lookup_text(row.get("from_vendor")))
        missing = [
            vendor
            for vendor in vendor_names
            if _lookup_text(vendor) not in loaded_vendors
        ]
        return source_rows, missing, None


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


def _request_source_type(request: RequestPayload | None) -> str:
    if not isinstance(request, Mapping):
        return ""
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        return ""
    for key in _SOURCE_TYPE_KEYS:
        value = _key(inputs.get(key))
        if value:
            return value
    return ""


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


def _request_b2b_displacement_vendors(request: RequestPayload | None) -> tuple[str, ...]:
    if not isinstance(request, Mapping):
        return ()
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        return ()
    for key in _B2B_DISPLACEMENT_VENDOR_KEYS:
        values = _string_values(inputs.get(key))
        if values:
            return tuple(values)
    return ()


def _request_b2b_displacement_limit(request: RequestPayload | None) -> int:
    if not isinstance(request, Mapping):
        return _B2B_DISPLACEMENT_DEFAULT_LIMIT
    inputs = request.get("inputs")
    if not isinstance(inputs, Mapping):
        return _B2B_DISPLACEMENT_DEFAULT_LIMIT
    for key in _B2B_DISPLACEMENT_LIMIT_KEYS:
        raw = inputs.get(key)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _B2B_DISPLACEMENT_DEFAULT_LIMIT
        return max(1, min(value, _B2B_DISPLACEMENT_MAX_LIMIT))
    return _B2B_DISPLACEMENT_DEFAULT_LIMIT


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
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _lookup_text(value: Any) -> str:
    return _clean(value).lower()


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _build_review_input_package(
    source_material: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    warnings: Sequence[Mapping[str, Any]] = (),
    default_source_type: str = "reviews",
) -> ContentOpsInputPackage:
    source_rows = source_material_to_source_rows(source_material)
    load_result = source_rows_to_campaign_opportunities(
        source_rows,
        default_fields={"source_type": default_source_type},
    )
    review_opportunities = [
        dict(row)
        for row in load_result.opportunities
        if _key(row.get("source_type")) in _REVIEW_SOURCE_TYPE_ALIASES
    ]
    adapter_warnings = [
        {
            "code": warning.code,
            "message": warning.message,
            **({"row_index": warning.row_index} if warning.row_index is not None else {}),
            **({"field": warning.field} if warning.field else {}),
        }
        for warning in load_result.warnings
    ]
    package_metadata = {
        "source": "review_input_package",
        "source_row_count": len(source_rows),
        "opportunity_count": len(load_result.opportunities),
        "included_row_count": len(review_opportunities),
        "skipped_row_count": max(0, len(source_rows) - len(review_opportunities)),
        **dict(metadata or {}),
    }
    package_warnings = [dict(warning) for warning in warnings]
    package_warnings.extend(adapter_warnings)
    if not review_opportunities:
        warning_code = (
            "review_source_rows_unrecognized"
            if source_rows
            else "review_source_material_empty"
        )
        warning_message = (
            "Review source rows were provided, but none were recognized as review or complaint rows."
            if source_rows
            else "No review or complaint source rows were provided."
        )
        package_warnings.append({
            "code": warning_code,
            "message": warning_message,
        })
        return _noop_package(
            "atlas_review_request",
            metadata=package_metadata,
            warnings=package_warnings,
        )
    primary_entity = _review_primary_entity(review_opportunities)
    inputs = {
        "source_material": review_opportunities,
        "review_source_material": review_opportunities,
        "source_type": "reviews",
        "campaign_name": _REVIEW_CAMPAIGN_NAME,
        "target_account": primary_entity,
        "topic": _REVIEW_TOPIC,
        "offer": _REVIEW_OFFER,
        "audience": _REVIEW_AUDIENCE,
        "target_keyword": _REVIEW_TARGET_KEYWORD,
        "brief_type": "discovery",
        "search_intent": (
            "Marketing teams looking for review-grounded themes for landing "
            "pages and blog posts."
        ),
        "primary_entity": primary_entity,
        "audience_entity": _REVIEW_AUDIENCE,
        "competitors": _review_competitors(review_opportunities),
        "objections": [
            "Will this sound generic?",
            "Can we trace the claims back to customer language?",
        ],
        "source_period": "Recent customer reviews",
        "review_source_count": len(review_opportunities),
        "internal_links": ["/systems/ai-content-ops/intake"],
        "cta_label": "Turn Reviews Into Content",
        "cta_url": "/systems/ai-content-ops/intake",
    }
    return ContentOpsInputPackage(
        provider="atlas_review_request",
        inputs=inputs,
        outputs=_REVIEW_CAMPAIGN_OUTPUTS,
        target_mode="vendor_retention",
        ingestion_profile="existing_evidence",
        metadata=package_metadata,
        warnings=tuple(package_warnings),
    )


def _review_primary_entity(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        for key in ("vendor_name", "product_name", "company_name"):
            value = _clean(row.get(key))
            if value:
                return value
    return _REVIEW_CAMPAIGN_NAME


def _review_competitors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = _clean(row.get("vendor_name") or row.get("product_name"))
        if value and value not in values:
            values.append(value)
        if len(values) == 3:
            break
    return values or ["generic AI copy", "manual review mining"]


def _build_competitive_input_package(
    source_material: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    warnings: Sequence[Mapping[str, Any]] = (),
    default_source_type: str = "competitive",
) -> ContentOpsInputPackage:
    source_rows = source_material_to_source_rows(
        _competitive_source_material_rows(source_material)
    )
    load_result = source_rows_to_campaign_opportunities(
        source_rows,
        default_fields={"source_type": default_source_type},
    )
    competitive_opportunities = [
        dict(row)
        for row in load_result.opportunities
        if _is_competitive_opportunity(row)
    ]
    adapter_warnings = [
        {
            "code": warning.code,
            "message": warning.message,
            **({"row_index": warning.row_index} if warning.row_index is not None else {}),
            **({"field": warning.field} if warning.field else {}),
        }
        for warning in load_result.warnings
    ]
    package_metadata = {
        "source": "competitive_input_package",
        "source_row_count": len(source_rows),
        "opportunity_count": len(load_result.opportunities),
        "included_row_count": len(competitive_opportunities),
        "skipped_row_count": max(0, len(source_rows) - len(competitive_opportunities)),
        **dict(metadata or {}),
    }
    package_warnings = [dict(warning) for warning in warnings]
    package_warnings.extend(adapter_warnings)
    if not competitive_opportunities:
        warning_code = (
            "competitive_source_rows_unrecognized"
            if source_rows
            else "competitive_source_material_empty"
        )
        warning_message = (
            "Competitive source rows were provided, but none carried competitor or displacement markers."
            if source_rows
            else "No competitive or displacement source rows were provided."
        )
        package_warnings.append({
            "code": warning_code,
            "message": warning_message,
        })
        return _noop_package(
            "atlas_competitive_request",
            metadata=package_metadata,
            warnings=package_warnings,
        )
    primary_entity = _competitive_primary_entity(competitive_opportunities)
    competitors = _competitive_competitors(competitive_opportunities)
    inputs = {
        "source_material": competitive_opportunities,
        "competitive_source_material": competitive_opportunities,
        "source_type": "competitive",
        "campaign_name": _COMPETITIVE_CAMPAIGN_NAME,
        "target_account": primary_entity,
        "topic": _COMPETITIVE_TOPIC,
        "offer": _COMPETITIVE_OFFER,
        "audience": _COMPETITIVE_AUDIENCE,
        "target_keyword": _COMPETITIVE_TARGET_KEYWORD,
        "brief_type": "displacement",
        "search_intent": (
            "Marketing teams looking for competitor and switching evidence "
            "they can turn into grounded buyer-facing content."
        ),
        "primary_entity": primary_entity,
        "audience_entity": _COMPETITIVE_AUDIENCE,
        "competitors": competitors,
        "competitive_alternatives": competitors,
        "objections": [
            "Will this make unsupported competitive claims?",
            "Can we trace switching claims back to source evidence?",
        ],
        "source_period": "Recent competitive and displacement evidence",
        "competitive_source_count": len(competitive_opportunities),
        "displacement_source_count": len(competitive_opportunities),
        "internal_links": ["/systems/ai-content-ops/intake"],
        "cta_label": "Turn Competitive Evidence Into Content",
        "cta_url": "/systems/ai-content-ops/intake",
    }
    return ContentOpsInputPackage(
        provider="atlas_competitive_request",
        inputs=inputs,
        outputs=_COMPETITIVE_CAMPAIGN_OUTPUTS,
        target_mode="vendor_retention",
        ingestion_profile="existing_evidence",
        metadata=package_metadata,
        warnings=tuple(package_warnings),
    )


def _is_competitive_opportunity(row: Mapping[str, Any]) -> bool:
    if _key(row.get("source_type")) in _COMPETITIVE_SOURCE_TYPE_ALIASES:
        return _has_competitive_marker(row)
    return _has_competitive_marker(row)


def _has_competitive_marker(row: Mapping[str, Any]) -> bool:
    return any(
        _key(key) in _COMPETITIVE_MARKER_KEYS and value not in (None, "", [], {})
        for key, value in row.items()
    )


def _competitive_primary_entity(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        for key in (
            "from_vendor",
            "vendor_name",
            "current_vendor",
            "incumbent_vendor",
            "product_name",
            "company_name",
        ):
            value = _clean(row.get(key))
            if value:
                return value
    return _COMPETITIVE_CAMPAIGN_NAME


def _competitive_competitors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        for key in (
            "to_vendor",
            "competitor",
            "competitor_name",
            "competitor_vendor",
            "alternative_name",
            "alternative_vendor",
            "winning_vendor",
            "new_vendor",
            "competitive_alternatives",
            "alternatives",
            "competitors",
            "competitors_mentioned",
            "top_displacement_targets",
            "competitor_overlap",
        ):
            _append_competitive_names(values, row.get(key))
            if len(values) >= 5:
                return values[:5]
    return values[:5] or ["manual competitive research", "generic AI copy"]


def _competitive_source_material_rows(source_material: Any) -> Any:
    if not isinstance(source_material, Mapping):
        return source_material
    rows: list[Any] = []
    for key, value in source_material.items():
        if _key(key) not in _COMPETITIVE_BUNDLE_KEYS:
            continue
        if _is_empty_source_material(value):
            continue
        if isinstance(value, (list, tuple)):
            rows.extend(value)
        else:
            rows.append(value)
    return rows if rows else source_material


def _append_competitive_names(values: list[str], raw: Any) -> None:
    if raw in (None, "", [], {}):
        return
    if isinstance(raw, str):
        items: Sequence[Any] = raw.split(",")
    elif isinstance(raw, Mapping):
        items = (
            raw.get("name")
            or raw.get("competitor")
            or raw.get("vendor")
            or raw.get("to_vendor")
            or raw.get("alternative")
            or raw.get("alternative_name"),
        )
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        items = raw
    else:
        items = (raw,)
    for item in items:
        if isinstance(item, Mapping):
            _append_competitive_names(values, item)
            continue
        value = _clean(item)
        if value and value not in values:
            values.append(value)


def _b2b_displacement_dynamics_source_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    from_vendor = _clean(row.get("from_vendor"))
    to_vendor = _clean(row.get("to_vendor"))
    if not from_vendor or not to_vendor:
        return {}
    dynamics = _json_mapping(row.get("dynamics"))
    edge_metrics = _json_mapping(dynamics.get("edge_metrics"))
    mention_count = _safe_positive_int(
        edge_metrics.get("mention_count")
        or dynamics.get("mention_count")
        or dynamics.get("displacement_mention_count")
    )
    primary_driver = _clean(
        edge_metrics.get("primary_driver")
        or dynamics.get("primary_driver")
        or dynamics.get("driver")
        or dynamics.get("displacement_driver")
    )
    as_of_date = _clean(row.get("as_of_date"))
    source_id = _b2b_displacement_source_id(from_vendor, to_vendor, as_of_date)
    text = _b2b_displacement_summary_text(
        dynamics,
        from_vendor=from_vendor,
        to_vendor=to_vendor,
        mention_count=mention_count,
        primary_driver=primary_driver,
    )
    out: dict[str, Any] = {
        "source_type": "competitive_displacement",
        "source_id": source_id,
        "target_id": source_id,
        "from_vendor": from_vendor,
        "to_vendor": to_vendor,
        "vendor_name": from_vendor,
        "competitor": to_vendor,
        "competitive_alternatives": [to_vendor],
        "displacement_direction": "from_vendor_to_to_vendor",
        "text": text,
        "content": text,
    }
    if mention_count is not None:
        out["displacement_mention_count"] = mention_count
    if primary_driver:
        out["primary_driver"] = primary_driver
    if as_of_date:
        out["source_date"] = as_of_date
    if _clean(row.get("analysis_window_days")):
        out["analysis_window_days"] = row.get("analysis_window_days")
    if _clean(row.get("schema_version")):
        out["schema_version"] = row.get("schema_version")
    return out


def _b2b_displacement_source_id(
    from_vendor: str,
    to_vendor: str,
    as_of_date: str,
) -> str:
    date_part = as_of_date or "latest"
    return f"b2b_displacement:{from_vendor}->{to_vendor}:{date_part}"


def _b2b_displacement_summary_text(
    dynamics: Mapping[str, Any],
    *,
    from_vendor: str,
    to_vendor: str,
    mention_count: int | None,
    primary_driver: str,
) -> str:
    parts: list[str] = []
    for value in (
        dynamics.get("battle_summary"),
        dynamics.get("migration_proof"),
        dynamics.get("competitive_reframes"),
        dynamics.get("market_context"),
    ):
        _append_dynamics_text(parts, value)
    if primary_driver:
        _append_unique_text(parts, f"Primary driver: {primary_driver}.")
    if mention_count is not None:
        _append_unique_text(parts, f"Mentions observed: {mention_count}.")
    if not parts:
        parts.append(
            f"Canonical B2B displacement dynamics track {from_vendor} to {to_vendor}."
        )
    return " ".join(parts[:4])


def _append_dynamics_text(parts: list[str], value: Any) -> None:
    if value in (None, "", [], {}):
        return
    if isinstance(value, str):
        _append_unique_text(parts, value)
        return
    if isinstance(value, Mapping):
        for key in (
            "conclusion",
            "summary",
            "evidence_summary",
            "why_this_matters",
            "migration_pattern",
            "primary_driver",
            "quote",
            "text",
        ):
            _append_dynamics_text(parts, value.get(key))
            if len(parts) >= 4:
                return
        for key in ("proof_points", "evidence", "signals"):
            _append_dynamics_text(parts, value.get(key))
            if len(parts) >= 4:
                return
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _append_dynamics_text(parts, item)
            if len(parts) >= 4:
                return


def _append_unique_text(parts: list[str], raw: Any) -> None:
    value = _clean(raw)
    if value and value not in parts:
        parts.append(value)


def _safe_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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


def _selected_b2b_displacement_warnings(
    *,
    missing: Sequence[str],
    skip_reason: str | None,
) -> tuple[dict[str, Any], ...]:
    warnings: list[dict[str, Any]] = []
    if skip_reason == "missing_account_scope":
        warnings.append({
            "code": "b2b_displacement_missing_account_scope",
            "message": (
                "B2B displacement source selection requires an account-scoped "
                "Content Ops request."
            ),
            "missing_count": len(missing),
        })
        return tuple(warnings)
    if skip_reason == "repository_unconfigured":
        warnings.append({
            "code": "b2b_displacement_repository_unconfigured",
            "message": "B2B displacement source selection is not configured for this route.",
            "missing_count": len(missing),
        })
        return tuple(warnings)
    if skip_reason == "query_failed":
        warnings.append({
            "code": "b2b_displacement_query_failed",
            "message": "B2B displacement source selection failed closed.",
            "missing_count": len(missing),
        })
        return tuple(warnings)
    if missing:
        warnings.append({
            "code": "b2b_displacement_vendors_not_found",
            "message": (
                "One or more selected B2B displacement vendors had no scoped "
                "canonical displacement dynamics."
            ),
            "missing_count": len(missing),
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
