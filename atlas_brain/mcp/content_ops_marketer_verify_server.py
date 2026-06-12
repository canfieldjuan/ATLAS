"""Verify-only Content Ops marketer MCP server."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from extracted_content_pipeline.adversarial_pass import (
    AdversarialFinding,
    AdversarialFindingCategory,
    AdversarialPass,
)
from extracted_content_pipeline.calibration_library import (
    CalibrationExample,
    CalibrationLabel,
)
from extracted_content_pipeline.claims_map import ExtractedClaim
from extracted_content_pipeline.content_pr import (
    CommentCategory,
    CoverageRow,
    CoverageStatus,
    ReviewComment,
    RulePacketVersions,
)

from .._content_ops_claim_registry import ContentOpsClaimRegistryRepository
from .._content_ops_review_workflow import (
    ContentOpsReviewRequest,
    ContentOpsAccountResolver,
    TenantCalibrationLibraryReader,
    TenantClaimRegistryReader,
    run_content_ops_review_for_bound_tenant,
)
from ..config_defaults import (
    DEFAULT_CONTENT_OPS_MARKETER_VERIFY_PORT,
    DEFAULT_MCP_HOST,
)

logger = logging.getLogger("atlas.mcp.content_ops.marketer_verify")

_AUTH_MODE_BEARER = "bearer"
_AUTH_MODE_OAUTH = "oauth"
_MIN_HTTP_AUTH_TOKEN_LENGTH = 24
_OAUTH_AUTHORIZATION_METADATA_PATH = "/.well-known/oauth-authorization-server"
_PUBLIC_CLIENT_TOKEN_AUTH_METHODS = ("none", "client_secret_post", "client_secret_basic")
_PLACEHOLDER_HTTP_AUTH_TOKENS = {
    "<token>",
    "changeme",
    "change-me",
    "password",
    "secret",
    "test-token",
    "token",
}
_MALFORMED_COVERAGE_RULE_PREFIX = "MALFORMED-COVERAGE"
_registry_reader_override: TenantClaimRegistryReader | None = None
_calibration_reader_override: TenantCalibrationLibraryReader | None = None
_account_resolver_override: ContentOpsAccountResolver | None = None
_oauth_provider = None
_QUALITY_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "severity": {
            "type": "string",
            "enum": ["blocker", "warning", "info"],
        },
        "field_name": {"type": "string"},
    },
}
_QUALITY_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": _QUALITY_FINDING_SCHEMA,
        },
    },
}
VERIFY_DRAFT_PARAMETER_SCHEMA: dict[str, dict[str, Any]] = {
    "asset_id": {
        "type": "string",
        "examples": ["landing-page-hero-v3"],
    },
    "rule_packet": {
        "type": "object",
        "properties": {
            "brief": {"type": "string"},
            "brand_voice": {"type": "string"},
            "claim_registry": {"type": "string"},
            "compliance": {"type": "string"},
            "channel_schema": {"type": "string"},
        },
    },
    "coverage": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string"},
                "requirement": {"type": "string"},
                "required": {"type": "boolean", "default": True},
                "status": {
                    "type": "string",
                    "enum": ["pass", "fail", "not_applicable", "unresolved"],
                },
                "evidence": {"type": "string"},
            },
        },
    },
    "extracted_claims": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "location": {"type": "string"},
                "registry_id": {"type": "string"},
            },
        },
    },
    "quality_reports": {
        "anyOf": [
            _QUALITY_REPORT_SCHEMA,
            {
                "type": "array",
                "items": _QUALITY_REPORT_SCHEMA,
            },
        ],
    },
    "brand_voice_payload": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
            "banned_terms": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    "comments": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "brief",
                        "brand_rule",
                        "claim_registry",
                        "compliance",
                        "channel_constraint",
                        "performance_hypothesis",
                        "editorial_judgment",
                        "nit",
                    ],
                },
                "message": {"type": "string"},
                "evidence": {"type": "string"},
                "blocking": {"type": "boolean", "default": False},
            },
        },
    },
    "adversarial_passes": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "pass_id": {"type": "string"},
                "source": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": [
                                    "overclaim",
                                    "ambiguity",
                                    "reader_objection",
                                    "promise_cta_mismatch",
                                    "generic_stretch",
                                    "missing_proof",
                                    "voice_slip",
                                ],
                            },
                            "message": {"type": "string"},
                            "evidence": {"type": "string"},
                            "location": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
    "calibration_library": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "example_id": {"type": "string"},
                "label": {
                    "type": "string",
                    "enum": [
                        "approved",
                        "rejected",
                        "borderline",
                        "known_defect",
                        "good_voice",
                        "voice_drift",
                        "overclaim",
                        "weak_persuasion",
                        "strong_persuasion",
                    ],
                },
                "excerpt": {"type": "string"},
                "reasoning": {"type": "string"},
                "source": {"type": "string"},
            },
        },
    },
    "as_of": {
        "type": "string",
        "format": "date",
        "examples": ["2026-06-09"],
    },
}
_VERIFY_DRAFT_PARAMETER_DESCRIPTIONS = {
    "asset_id": "Stable identifier for the draft or asset being reviewed.",
    "rule_packet": "Pinned rule-packet version ids used for this review.",
    "coverage": "Requirement coverage rows extracted from the draft.",
    "extracted_claims": "Claims extracted from the draft and mapped to the tenant registry.",
    "quality_reports": "Deterministic quality-gate reports for the draft.",
    "brand_voice_payload": "Brand-voice audit result for the draft.",
    "comments": "Reviewer comments or blocking findings to include in the verdict.",
    "adversarial_passes": (
        "Independent adversarial-review passes whose substantiated findings are "
        "folded into the verdict as never-blocking editor evidence."
    ),
    "calibration_library": (
        "Curated review examples; the verdict surfaces the teachable anchors "
        "matching the failure modes the adversarial passes raised."
    ),
    "as_of": "ISO review date used for registry expiration checks.",
}


def _schema_field(name: str):
    return Field(
        description=_VERIFY_DRAFT_PARAMETER_DESCRIPTIONS[name],
        json_schema_extra=VERIFY_DRAFT_PARAMETER_SCHEMA[name],
    )


AssetIdArg = Annotated[Any, _schema_field("asset_id")]
RulePacketArg = Annotated[Any, _schema_field("rule_packet")]
CoverageArg = Annotated[Any, _schema_field("coverage")]
ExtractedClaimsArg = Annotated[Any, _schema_field("extracted_claims")]
QualityReportsArg = Annotated[Any, _schema_field("quality_reports")]
BrandVoicePayloadArg = Annotated[Any, _schema_field("brand_voice_payload")]
CommentsArg = Annotated[Any, _schema_field("comments")]
AdversarialPassesArg = Annotated[Any, _schema_field("adversarial_passes")]
CalibrationLibraryArg = Annotated[Any, _schema_field("calibration_library")]
AsOfArg = Annotated[Any, _schema_field("as_of")]


@dataclass(frozen=True)
class StaticContentOpsMarketerAccountResolver:
    """Direct/test resolver used before OAuth token binding lands."""

    account_id: str

    async def resolve_account_id(self) -> str | None:
        return _clean(self.account_id) or None


class ConfiguredContentOpsMarketerAccountResolver:
    """Resolve direct/test account binding from Atlas settings."""

    async def resolve_account_id(self) -> str | None:
        from ..config import settings

        return _clean(settings.mcp.content_ops_marketer_verify_account_id) or None


@dataclass(frozen=True)
class OAuthContentOpsMarketerAccountResolver:
    """Resolve the bound tenant account from the authenticated OAuth token."""

    provider: Any
    access_token: Any | None = None

    async def resolve_account_id(self) -> str | None:
        token = self.access_token
        if token is None:
            from mcp.server.auth.middleware.auth_context import get_access_token

            token = get_access_token()
        token_value = _clean(getattr(token, "token", ""))
        if not token_value:
            return None
        return _clean(self.provider.account_id_for_access_token(token_value)) or None


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import close_database, init_database

    await init_database()
    logger.info("Content Ops marketer verify MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-content-ops-marketer-verify",
    instructions=(
        "Verify-only Content Ops review server for Atlas marketers. "
        "Submit structured draft evidence and receive deterministic review "
        "verdicts for one bound tenant. This server cannot generate, publish, "
        "approve, unlock, or mutate claim-registry rows."
    ),
    lifespan=_lifespan,
)


@mcp.custom_route("/oauth/approve", methods=["GET", "POST"], include_in_schema=False)
async def _oauth_approve(request):
    """Operator approval page for remote OAuth connectors."""
    if _oauth_provider is None:
        from starlette.responses import HTMLResponse

        return HTMLResponse("<h1>OAuth mode is not enabled</h1>", status_code=404)
    from .content_ops_marketer_verify_oauth import handle_approval_request

    return await handle_approval_request(_oauth_provider, request)


@mcp.tool(structured_output=True)
async def verify_draft(
    asset_id: AssetIdArg = "",
    rule_packet: RulePacketArg = None,
    coverage: CoverageArg = None,
    extracted_claims: ExtractedClaimsArg = None,
    quality_reports: QualityReportsArg = None,
    brand_voice_payload: BrandVoicePayloadArg = None,
    comments: CommentsArg = None,
    adversarial_passes: AdversarialPassesArg = None,
    calibration_library: CalibrationLibraryArg = None,
    as_of: AsOfArg = "",
) -> dict[str, Any]:
    """Verify structured draft evidence for the bound tenant."""

    result = await run_content_ops_review_for_bound_tenant(
        _review_request_from_tool_args(
            asset_id=asset_id,
            rule_packet=rule_packet,
            coverage=coverage,
            extracted_claims=extracted_claims,
            quality_reports=quality_reports,
            brand_voice_payload=brand_voice_payload,
            comments=comments,
            adversarial_passes=adversarial_passes,
            calibration_library=calibration_library,
            as_of=as_of,
        ),
        account_resolver=_get_account_resolver(),
        registry_reader=_get_registry_reader(),
        calibration_reader=_get_calibration_reader(),
    )
    return result.as_dict()


def _review_request_from_tool_args(
    *,
    asset_id: Any,
    rule_packet: Any,
    coverage: Any,
    extracted_claims: Any,
    quality_reports: Any,
    brand_voice_payload: Any,
    comments: Any,
    adversarial_passes: Any = None,
    calibration_library: Any = None,
    as_of: Any,
) -> ContentOpsReviewRequest:
    return ContentOpsReviewRequest(
        asset_id=_clean(asset_id),
        rule_packet=_rule_packet(rule_packet),
        coverage=_coverage_rows(coverage),
        quality_reports=_items(quality_reports),
        brand_voice_payload=brand_voice_payload if isinstance(brand_voice_payload, Mapping) else None,
        extracted_claims=_claims(extracted_claims),
        comments=_comments(comments),
        adversarial_passes=_adversarial_passes(adversarial_passes),
        calibration_examples=_calibration_examples(calibration_library),
        as_of=_date(as_of),
    )


def _streamable_http_app():
    """Build the authenticated streamable HTTP app for verify-only tools."""
    if _http_auth_mode() == _AUTH_MODE_OAUTH:
        _configure_oauth_auth()
        return _apply_content_ops_public_client_metadata(mcp.streamable_http_app())

    from .auth import BearerAuthMiddleware

    return BearerAuthMiddleware(
        mcp.streamable_http_app(),
        token=_require_http_auth_token(),
    )


def _http_auth_mode() -> str:
    from ..config import settings

    mode = _clean(settings.mcp.content_ops_marketer_verify_auth_mode).lower() or _AUTH_MODE_BEARER
    if mode not in {_AUTH_MODE_BEARER, _AUTH_MODE_OAUTH}:
        raise RuntimeError(
            "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_AUTH_MODE must be either "
            "'bearer' or 'oauth'"
        )
    return mode


def _require_http_auth_token() -> str:
    """Return a production-shaped auth token or fail before serving HTTP."""
    from ..config import settings

    token = _clean(settings.mcp.auth_token)
    if not token:
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN is required for Content Ops marketer verify "
            "HTTP mode; this tool exposes tenant review verdicts."
        )
    if token.lower() in _PLACEHOLDER_HTTP_AUTH_TOKENS or token.startswith("<"):
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must not be a placeholder value for Content "
            "Ops marketer verify HTTP mode."
        )
    if len(token) < _MIN_HTTP_AUTH_TOKEN_LENGTH:
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must be at least "
            f"{_MIN_HTTP_AUTH_TOKEN_LENGTH} characters for Content Ops marketer "
            "verify HTTP mode."
        )
    return token


def _configure_oauth_auth(target_mcp: Any | None = None):
    """Configure FastMCP OAuth auth for remote connector clients."""
    global _oauth_provider

    from mcp.server.auth.provider import ProviderTokenVerifier
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions

    from ..config import settings
    from .content_ops_marketer_verify_oauth import (
        DEFAULT_CONTENT_OPS_VERIFY_SCOPE,
        ContentOpsMarketerVerifyOAuthProvider,
        as_any_http_url,
        validate_oauth_settings,
    )

    issuer_url = _clean(settings.mcp.content_ops_marketer_verify_oauth_issuer_url)
    resource_url = _clean(settings.mcp.content_ops_marketer_verify_oauth_resource_url)
    approval_token = _clean(settings.mcp.content_ops_marketer_verify_oauth_approval_token)
    account_id = _clean(settings.mcp.content_ops_marketer_verify_account_id)
    state_file = _clean(settings.mcp.content_ops_marketer_verify_oauth_state_file) or None
    validate_oauth_settings(
        issuer_url=issuer_url,
        resource_server_url=resource_url,
        approval_token=approval_token,
    )
    if not account_id:
        raise RuntimeError(
            "ATLAS_MCP_CONTENT_OPS_MARKETER_VERIFY_ACCOUNT_ID is required in oauth "
            "mode to issue tenant-bound tokens"
        )
    configured_mcp = target_mcp or mcp
    configured_mcp.settings.transport_security = _oauth_transport_security_settings(
        issuer_url=issuer_url,
        resource_url=resource_url,
    )

    if _oauth_provider is None:
        _oauth_provider = ContentOpsMarketerVerifyOAuthProvider(
            issuer_url=issuer_url,
            approval_token=approval_token,
            account_id=account_id,
            scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
            state_file=state_file,
        )
    configured_mcp.settings.auth = AuthSettings(
        issuer_url=as_any_http_url(issuer_url),
        resource_server_url=as_any_http_url(resource_url),
        required_scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
            default_scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
        ),
    )
    configured_mcp._auth_server_provider = _oauth_provider
    configured_mcp._token_verifier = ProviderTokenVerifier(_oauth_provider)
    return _oauth_provider


def _content_ops_oauth_metadata(*, issuer_url: str, scopes: list[str]) -> dict[str, Any]:
    """Return OAuth metadata matching FastMCP's routes plus public clients."""
    issuer = issuer_url.strip().rstrip("/")
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token",
        "registration_endpoint": f"{issuer}/register",
        "scopes_supported": scopes,
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": list(_PUBLIC_CLIENT_TOKEN_AUTH_METHODS),
        "code_challenge_methods_supported": ["S256"],
    }


def _oauth_metadata_cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "mcp-protocol-version",
    }


def _content_ops_oauth_metadata_endpoint(*, issuer_url: str, scopes: list[str]):
    async def _metadata(request):
        from starlette.responses import JSONResponse, Response

        if request.method == "OPTIONS":
            return Response(status_code=204, headers=_oauth_metadata_cors_headers())
        return JSONResponse(
            _content_ops_oauth_metadata(issuer_url=issuer_url, scopes=scopes),
            headers=_oauth_metadata_cors_headers(),
        )

    return _metadata


def _apply_content_ops_public_client_metadata(app):
    """Replace FastMCP's auth metadata route with Content Ops public-client metadata."""
    router = getattr(app, "router", None)
    routes = list(getattr(router, "routes", []) or [])
    if router is None or not routes:
        return app

    from starlette.routing import Route

    from ..config import settings
    from .content_ops_marketer_verify_oauth import DEFAULT_CONTENT_OPS_VERIFY_SCOPE

    issuer_url = _clean(settings.mcp.content_ops_marketer_verify_oauth_issuer_url)
    metadata_route = Route(
        _OAUTH_AUTHORIZATION_METADATA_PATH,
        endpoint=_content_ops_oauth_metadata_endpoint(
            issuer_url=issuer_url,
            scopes=[DEFAULT_CONTENT_OPS_VERIFY_SCOPE],
        ),
        methods=["GET", "OPTIONS"],
        include_in_schema=False,
    )
    replacement = []
    replaced = False
    for route in routes:
        if getattr(route, "path", "") == _OAUTH_AUTHORIZATION_METADATA_PATH:
            replacement.append(metadata_route)
            replaced = True
        else:
            replacement.append(route)
    if not replaced:
        replacement.insert(0, metadata_route)
    router.routes = replacement
    return app


def _oauth_transport_security_settings(*, issuer_url: str, resource_url: str):
    """Allow configured OAuth hosts while keeping DNS rebinding protection on."""
    from mcp.server.transport_security import TransportSecuritySettings

    allowed_hosts = {
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
    }
    for url in (issuer_url, resource_url):
        allowed_hosts.update(_host_header_variants(url))
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=sorted(allowed_hosts),
    )


def _host_header_variants(url: str) -> set[str]:
    parsed = urlparse(url.strip())
    if not parsed.hostname:
        return set()
    variants = {parsed.netloc}
    if parsed.port is None:
        variants.add(parsed.hostname)
        if parsed.scheme == "https":
            variants.add(f"{parsed.hostname}:443")
        elif parsed.scheme == "http":
            variants.add(f"{parsed.hostname}:80")
    elif (parsed.scheme == "https" and parsed.port == 443) or (
        parsed.scheme == "http" and parsed.port == 80
    ):
        variants.add(parsed.hostname)
    return {variant for variant in variants if variant}


def _get_account_resolver():
    if _account_resolver_override is not None:
        return _account_resolver_override
    if _http_auth_mode() == _AUTH_MODE_OAUTH:
        provider = _oauth_provider or _configure_oauth_auth()
        return OAuthContentOpsMarketerAccountResolver(provider=provider)
    return ConfiguredContentOpsMarketerAccountResolver()


def _get_registry_reader() -> TenantClaimRegistryReader:
    if _registry_reader_override is not None:
        return _registry_reader_override
    from ..storage.database import get_db_pool

    return ContentOpsClaimRegistryRepository(pool=get_db_pool())


class _EmptyCalibrationLibraryReader:
    """No-op calibration reader: no server-side anchors until persistence lands.

    Slice A wires the read seam end to end; the Postgres-backed repository that
    actually returns tenant anchors is the next slice. Until then this default
    returns nothing, so verify behaves exactly as the request-supplied path.
    """

    async def list_calibration_examples(self, *, scope: Any) -> tuple[Any, ...]:
        return ()


_empty_calibration_reader = _EmptyCalibrationLibraryReader()


def _get_calibration_reader() -> TenantCalibrationLibraryReader:
    if _calibration_reader_override is not None:
        return _calibration_reader_override
    return _empty_calibration_reader


def _rule_packet(value: Any) -> RulePacketVersions:
    if not isinstance(value, Mapping):
        value = {}
    return RulePacketVersions(
        brief=_clean(value.get("brief")),
        brand_voice=_clean(value.get("brand_voice")),
        claim_registry=_clean(value.get("claim_registry")),
        compliance=_clean(value.get("compliance")),
        channel_schema=_clean(value.get("channel_schema")),
    )


def _coverage_rows(value: Any) -> tuple[CoverageRow, ...]:
    rows: list[CoverageRow] = []
    for index, item in enumerate(_row_items(value), start=1):
        if not isinstance(item, Mapping):
            rows.append(_malformed_coverage_row(index))
            continue
        rule_id = _clean(item.get("rule_id"))
        if not rule_id:
            rows.append(_malformed_coverage_row(index))
            continue
        rows.append(
            CoverageRow(
                rule_id=rule_id,
                requirement=_clean(item.get("requirement")),
                required=_bool(item.get("required"), default=True),
                status=_coverage_status(item.get("status")),
                evidence=_clean(item.get("evidence")),
            )
        )
    return tuple(rows)


def _malformed_coverage_row(index: int) -> CoverageRow:
    return CoverageRow(
        rule_id=f"{_MALFORMED_COVERAGE_RULE_PREFIX}-{index}",
        requirement="Malformed decoded coverage row",
        required=True,
        status=CoverageStatus.UNRESOLVED,
        evidence="",
    )


def _claims(value: Any) -> tuple[ExtractedClaim, ...]:
    claims: list[ExtractedClaim] = []
    for item in _dict_rows(value):
        registry_id = _clean(item.get("registry_id")) or None
        text = item.get("text") if isinstance(item.get("text"), str) else None
        claims.append(
            ExtractedClaim(
                text=text,
                location=_clean(item.get("location")),
                registry_id=registry_id,
            )
        )
    return tuple(claims)


def _comments(value: Any) -> tuple[ReviewComment, ...]:
    comments: list[ReviewComment] = []
    for item in _dict_rows(value):
        blocking = _bool(item.get("blocking"), default=False)
        category = _comment_category(item.get("category"))
        if category == CommentCategory.NIT and blocking:
            category = CommentCategory.EDITORIAL_JUDGMENT
        comments.append(
            ReviewComment(
                category=category,
                message=_clean(item.get("message")),
                evidence=_clean(item.get("evidence")),
                blocking=blocking,
            )
        )
    return tuple(comments)


_ADVERSARIAL_CATEGORY_VALUES = frozenset(c.value for c in AdversarialFindingCategory)


def _adversarial_category(value: Any) -> Any:
    """Coerce a known category string to the enum; keep an unknown one as text.

    ``comment_from_finding`` tolerates a raw-string category (its lane lookup is
    value-based and falls back to editorial judgment), so an unrecognized
    category is preserved rather than rejected.
    """

    cleaned = _clean(value)
    if cleaned in _ADVERSARIAL_CATEGORY_VALUES:
        return AdversarialFindingCategory(cleaned)
    return cleaned


def _adversarial_findings(value: Any) -> tuple[AdversarialFinding, ...]:
    findings: list[AdversarialFinding] = []
    for item in _dict_rows(value):
        findings.append(
            AdversarialFinding(
                category=_adversarial_category(item.get("category")),
                message=_clean(item.get("message")),
                evidence=_clean(item.get("evidence")),
                location=_clean(item.get("location")),
            )
        )
    return tuple(findings)


def _adversarial_passes(value: Any) -> tuple[AdversarialPass, ...]:
    passes: list[AdversarialPass] = []
    for index, item in enumerate(_dict_rows(value)):
        passes.append(
            AdversarialPass(
                pass_id=_clean(item.get("pass_id")) or f"pass-{index}",
                source=_clean(item.get("source")),
                findings=_adversarial_findings(item.get("findings")),
            )
        )
    return tuple(passes)


_CALIBRATION_LABEL_VALUES = frozenset(label.value for label in CalibrationLabel)


def _calibration_label(value: Any) -> Any:
    """Coerce a known label string to the enum; keep an unknown one as text.

    The library queries compare labels by value, so an unrecognized label is
    preserved rather than rejected -- it simply matches no finding category.
    """

    cleaned = _clean(value)
    if cleaned in _CALIBRATION_LABEL_VALUES:
        return CalibrationLabel(cleaned)
    return cleaned


def _calibration_examples(value: Any) -> tuple[CalibrationExample, ...]:
    examples: list[CalibrationExample] = []
    for index, item in enumerate(_dict_rows(value)):
        examples.append(
            CalibrationExample(
                example_id=_clean(item.get("example_id")) or f"anchor-{index}",
                excerpt=_clean(item.get("excerpt")),
                label=_calibration_label(item.get("label")),
                reasoning=_clean(item.get("reasoning")),
                source=_clean(item.get("source")) or "curated",
            )
        )
    return tuple(examples)


def _dict_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(item for item in _row_items(value) if isinstance(item, Mapping))


def _row_items(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(value)


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return (value,)


def _coverage_status(value: Any) -> CoverageStatus:
    try:
        return CoverageStatus(_clean(value))
    except ValueError:
        return CoverageStatus.UNRESOLVED


def _comment_category(value: Any) -> CommentCategory:
    try:
        return CommentCategory(_clean(value))
    except ValueError:
        return CommentCategory.EDITORIAL_JUDGMENT


def _date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return default


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


if __name__ == "__main__":
    if "--sse" in sys.argv:
        import anyio
        import uvicorn
        from mcp.server.transport_security import TransportSecuritySettings
        from ..config import settings

        host = settings.mcp.host or DEFAULT_MCP_HOST
        port = settings.mcp.content_ops_marketer_verify_port or DEFAULT_CONTENT_OPS_MARKETER_VERIFY_PORT

        mcp.settings.host = host
        mcp.settings.port = port
        if _http_auth_mode() == _AUTH_MODE_BEARER:
            mcp.settings.transport_security = TransportSecuritySettings(
                enable_dns_rebinding_protection=False,
            )

        async def _serve():
            config = uvicorn.Config(
                _streamable_http_app(),
                host=host,
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            await server.serve()

        anyio.run(_serve)
    else:
        mcp.run(transport="stdio")
