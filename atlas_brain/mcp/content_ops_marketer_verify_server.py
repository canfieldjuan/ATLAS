"""Verify-only Content Ops marketer MCP server."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

from mcp.server.fastmcp import FastMCP

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
    TenantClaimRegistryReader,
    run_content_ops_review_for_bound_tenant,
)
from ..config_defaults import (
    DEFAULT_CONTENT_OPS_MARKETER_VERIFY_PORT,
    DEFAULT_MCP_HOST,
)

logger = logging.getLogger("atlas.mcp.content_ops.marketer_verify")

_MIN_HTTP_AUTH_TOKEN_LENGTH = 24
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
_account_resolver_override: "StaticContentOpsMarketerAccountResolver | None" = None


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


@mcp.tool(structured_output=True)
async def verify_draft(
    asset_id: Any = "",
    rule_packet: Any = None,
    coverage: Any = None,
    extracted_claims: Any = None,
    quality_reports: Any = None,
    brand_voice_payload: Any = None,
    comments: Any = None,
    as_of: Any = "",
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
            as_of=as_of,
        ),
        account_resolver=_get_account_resolver(),
        registry_reader=_get_registry_reader(),
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
        as_of=_date(as_of),
    )


def _streamable_http_app():
    """Build the authenticated streamable HTTP app for verify-only tools."""
    from .auth import BearerAuthMiddleware

    return BearerAuthMiddleware(
        mcp.streamable_http_app(),
        token=_require_http_auth_token(),
    )


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


def _get_account_resolver():
    return _account_resolver_override or ConfiguredContentOpsMarketerAccountResolver()


def _get_registry_reader() -> TenantClaimRegistryReader:
    if _registry_reader_override is not None:
        return _registry_reader_override
    from ..storage.database import get_db_pool

    return ContentOpsClaimRegistryRepository(pool=get_db_pool())


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
