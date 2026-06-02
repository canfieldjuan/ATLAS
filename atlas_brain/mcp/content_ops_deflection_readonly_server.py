"""
Read-only Content Ops FAQ deflection MCP server.

This server exposes only ChatGPT-compatible search/fetch tools for free
deflection report snapshots. It never accepts an account ID as a tool argument;
every tool resolves the tenant binding before touching storage.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

from extracted_content_pipeline.deflection_report_access import (
    DeflectionReportArtifactStore,
    DeflectionReportListRecord,
    PostgresDeflectionReportArtifactStore,
)
from extracted_content_pipeline.faq_deflection_report import (
    DEFAULT_DEFLECTION_SNAPSHOT_TOP_N,
    deflection_snapshot_content_opportunities,
)

from ..config_defaults import (
    DEFAULT_CONTENT_OPS_DEFLECTION_READONLY_PORT,
    DEFAULT_MCP_HOST,
)

logger = logging.getLogger("atlas.mcp.content_ops.deflection.readonly")

_MIN_HTTP_AUTH_TOKEN_LENGTH = 24
_PLACEHOLDER_HTTP_AUTH_TOKENS = {
    "<token>",
    "changeme",
    "change-me",
    "password",
    "secret",
    "test-readonly-token",
    "test-token",
    "token",
}
_store_override: DeflectionReportArtifactStore | None = None
_account_resolver_override: "ContentOpsDeflectionAccountResolver | None" = None


class ContentOpsDeflectionAccountResolver(Protocol):
    """Resolve the single tenant account visible to this MCP session."""

    async def resolve_account_id(self) -> str | None:
        """Return the bound account ID, or None when binding is unavailable."""


@dataclass(frozen=True)
class StaticContentOpsDeflectionAccountResolver:
    """Direct/test resolver used before OAuth token binding lands."""

    account_id: str

    async def resolve_account_id(self) -> str | None:
        return _clean(self.account_id) or None


class ConfiguredContentOpsDeflectionAccountResolver:
    """Resolve direct/test account binding from Atlas settings."""

    async def resolve_account_id(self) -> str | None:
        from ..config import settings

        return _clean(settings.mcp.content_ops_deflection_readonly_account_id) or None


@asynccontextmanager
async def _lifespan(server):
    """Initialize DB pool on startup, close on shutdown."""
    from ..storage.database import close_database, init_database

    await init_database()
    logger.info("Read-only Content Ops deflection MCP: DB pool initialized")
    yield
    await close_database()


mcp = FastMCP(
    "atlas-content-ops-deflection-readonly",
    instructions=(
        "Read-only FAQ deflection report server for Atlas Content Ops. "
        "Search and fetch unpaid-safe report snapshots for one bound tenant. "
        "This server cannot generate, publish, unlock, or expose full report "
        "artifacts."
    ),
    lifespan=_lifespan,
)


@mcp.tool(structured_output=True)
async def search(query: str = "", limit: int = 10, paid: bool | None = None) -> dict[str, Any]:
    """Search unpaid-safe deflection report snapshots for the bound tenant."""
    account_id = await _resolve_account_id()
    if account_id is None:
        return _failure_payload(
            "account_binding_required",
            "A single Content Ops account binding is required before reports can be searched.",
            results=[],
        )

    store = _get_store()
    bounded_limit = _bounded_limit(limit, default=10, upper=100)
    cleaned_query = _clean(query)
    records = await store.list_reports(
        account_id=account_id,
        limit=None if cleaned_query else bounded_limit,
        paid=paid,
    )
    filtered = [
        record
        for record in records
        if _matches_query(record, query)
    ][:bounded_limit]
    return {
        "results": [
            {
                "id": record.request_id,
                "title": _snapshot_title(record.snapshot, record.request_id),
                "url": _report_url(record.request_id),
            }
            for record in filtered
        ],
        "metadata": {
            "ok": True,
            "query": cleaned_query,
            "count": len(filtered),
            "paid_filter": paid,
        },
    }


@mcp.tool(structured_output=True)
async def fetch(id: str) -> dict[str, Any]:
    """Fetch one unpaid-safe deflection report document by request ID."""
    account_id = await _resolve_account_id()
    if account_id is None:
        return _failure_payload(
            "account_binding_required",
            "A single Content Ops account binding is required before reports can be fetched.",
        )

    request_id = _clean(id)
    if not request_id:
        return _failure_payload(
            "request_id_required",
            "fetch requires a deflection report request ID.",
        )

    record = await _get_store().get_artifact_record(
        account_id=account_id,
        request_id=request_id,
    )
    if record is None:
        return {
            "id": request_id,
            "title": "Deflection report not found",
            "text": "No deflection report was found for this request in the bound account.",
            "url": _report_url(request_id),
            "metadata": {
                "ok": True,
                "found": False,
            },
        }

    snapshot = dict(record.snapshot)
    title = _snapshot_title(snapshot, request_id)
    top_questions = _safe_top_questions(snapshot)
    opportunities = [
        dict(item)
        for item in deflection_snapshot_content_opportunities(
            snapshot,
            limit=DEFAULT_DEFLECTION_SNAPSHOT_TOP_N,
        )
    ]
    metadata = {
        "ok": True,
        "found": True,
        "summary": _safe_summary(snapshot),
        "top_questions": top_questions,
        "content_opportunities": opportunities,
        "unlock_status": {
            "paid": bool(record.paid),
            "full_report_locked": not bool(record.paid),
        },
    }
    return {
        "id": record.request_id,
        "title": title,
        "text": _document_text(
            title=title,
            summary=metadata["summary"],
            top_questions=top_questions,
            opportunities=opportunities,
            paid=bool(record.paid),
        ),
        "url": _report_url(record.request_id),
        "metadata": metadata,
    }


def _streamable_http_app():
    """Build the authenticated streamable HTTP app for read-only tools."""
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
            "ATLAS_MCP_AUTH_TOKEN is required for read-only Content Ops "
            "deflection HTTP mode; these tools expose tenant report snapshots."
        )
    if token.lower() in _PLACEHOLDER_HTTP_AUTH_TOKENS or token.startswith("<"):
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must not be a placeholder value for read-only "
            "Content Ops deflection HTTP mode."
        )
    if len(token) < _MIN_HTTP_AUTH_TOKEN_LENGTH:
        raise RuntimeError(
            "ATLAS_MCP_AUTH_TOKEN must be at least "
            f"{_MIN_HTTP_AUTH_TOKEN_LENGTH} characters for read-only Content Ops "
            "deflection HTTP mode."
        )
    return token


def _get_store() -> DeflectionReportArtifactStore:
    if _store_override is not None:
        return _store_override
    from ..storage.database import get_db_pool

    return PostgresDeflectionReportArtifactStore(pool=get_db_pool())


async def _resolve_account_id() -> str | None:
    resolver = _account_resolver_override or ConfiguredContentOpsDeflectionAccountResolver()
    return _clean(await resolver.resolve_account_id()) or None


def _failure_payload(
    code: str,
    message: str,
    *,
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {
            "ok": False,
            "error": code,
            "message": message,
        },
    }
    if results is not None:
        payload["results"] = results
    return payload


def _matches_query(record: DeflectionReportListRecord, query: str) -> bool:
    needle = _clean(query).lower()
    if not needle:
        return True
    haystack = " ".join(
        [
            record.request_id,
            _snapshot_title(record.snapshot, record.request_id),
            *_question_texts(record.snapshot),
        ]
    ).lower()
    return needle in haystack


def _snapshot_title(snapshot: Mapping[str, Any], request_id: str) -> str:
    explicit = _clean(snapshot.get("title") or snapshot.get("report_title"))
    if explicit:
        return explicit
    questions = _question_texts(snapshot)
    if questions:
        return f"Deflection report: {questions[0]}"
    return f"Deflection report {request_id}"


def _safe_summary(snapshot: Mapping[str, Any]) -> dict[str, int]:
    summary = snapshot.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "generated": _int(summary.get("generated")),
        "drafted_answer_count": _int(summary.get("drafted_answer_count")),
        "no_proven_answer_count": _int(summary.get("no_proven_answer_count")),
    }


def _safe_top_questions(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    top_questions = snapshot.get("top_questions")
    if not isinstance(top_questions, Sequence) or isinstance(
        top_questions,
        (str, bytes, bytearray),
    ):
        return []
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(top_questions, start=1):
        if len(out) >= DEFAULT_DEFLECTION_SNAPSHOT_TOP_N:
            break
        if not isinstance(raw, Mapping):
            continue
        question = _clean(raw.get("question"))
        if not question:
            continue
        out.append(
            {
                "rank": _int(raw.get("rank")) or index,
                "question": question,
                "weighted_frequency": _int(raw.get("weighted_frequency") or raw.get("frequency")),
                "customer_wording": _clean(raw.get("customer_wording")),
            }
        )
    return out


def _document_text(
    *,
    title: str,
    summary: Mapping[str, int],
    top_questions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    paid: bool,
) -> str:
    lines = [
        title,
        "",
        "Summary",
        f"- Generated opportunities: {_int(summary.get('generated'))}",
        f"- Drafted answer count: {_int(summary.get('drafted_answer_count'))}",
        f"- No proven answer count: {_int(summary.get('no_proven_answer_count'))}",
        "",
        "Top questions",
    ]
    if not top_questions:
        lines.append("- No top questions are available in the free snapshot.")
    for item in top_questions:
        lines.append(
            "- "
            f"{_int(item.get('rank'))}. {_clean(item.get('question'))} "
            f"(frequency {_int(item.get('weighted_frequency'))})"
        )
    lines.extend(["", "Content opportunities"])
    if not opportunities:
        lines.append("- No structured opportunities are available in the free snapshot.")
    for item in opportunities:
        lines.append(
            "- "
            f"{_clean(item.get('question'))}: "
            f"{_clean(item.get('recommended_content_action'))}"
        )
    lines.extend([
        "",
        "Unlock status",
        "- Full report is available after unlock." if paid else "- Full report remains locked.",
    ])
    return "\n".join(lines)


def _question_texts(snapshot: Mapping[str, Any]) -> list[str]:
    top_questions = snapshot.get("top_questions")
    if not isinstance(top_questions, Sequence) or isinstance(
        top_questions,
        (str, bytes, bytearray),
    ):
        return []
    out: list[str] = []
    for raw in top_questions:
        if isinstance(raw, Mapping):
            question = _clean(raw.get("question"))
            if question:
                out.append(question)
    return out


def _report_url(request_id: str) -> str:
    from ..config import settings

    base_url = _clean(settings.mcp.content_ops_deflection_readonly_report_base_url)
    if not base_url:
        base_url = "https://atlas.local/content-ops/deflection-reports"
    return f"{base_url.rstrip('/')}/{quote(_clean(request_id), safe='')}"


def _bounded_limit(value: Any, *, default: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, upper))


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    if "--sse" in sys.argv:
        import anyio
        import uvicorn
        from mcp.server.transport_security import TransportSecuritySettings
        from ..config import settings

        host = settings.mcp.host or DEFAULT_MCP_HOST
        port = (
            settings.mcp.content_ops_deflection_readonly_port
            or DEFAULT_CONTENT_OPS_DEFLECTION_READONLY_PORT
        )

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
