"""ChatGPT-compatible search/fetch adapter for Content Ops draft verification."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import content_ops_marketer_verify_server as verify_server


CONTRACT_ID = "content-ops-verify-draft-contract"
_RESULT_PREFIX = "content-ops-verdict:"
_ACCEPTED_FIELDS = (
    "asset_id",
    "rule_packet",
    "coverage",
    "extracted_claims",
    "quality_reports",
    "brand_voice_payload",
    "comments",
    "as_of",
)
# Optional submission fields -- accepted and documented, but not required.
_OPTIONAL_FIELDS = ("adversarial_passes", "calibration_library")


@dataclass(frozen=True)
class _CachedVerdict:
    account_id: str
    payload: dict[str, Any]


_verdict_cache: dict[str, _CachedVerdict] = {}

mcp = FastMCP(
    "atlas-content-ops-marketer-verify-chatgpt",
    instructions=(
        "ChatGPT-compatible Content Ops verification adapter. Search accepts a "
        "structured JSON review request and returns a tenant-bound verdict ID; "
        "fetch returns the cached verdict document. This adapter cannot "
        "generate, publish, approve, unlock, or mutate claim-registry rows."
    ),
    lifespan=verify_server._lifespan,
)


@mcp.custom_route("/oauth/approve", methods=["GET", "POST"], include_in_schema=False)
async def _oauth_approve(request):
    """Operator approval page for remote OAuth connectors."""
    return await verify_server._oauth_approve(request)


@mcp.tool(structured_output=True)
async def search(query: str = "", limit: int = 10) -> dict[str, Any]:
    """Search or submit one structured Content Ops review request."""
    request_payload = _json_object(query)
    if request_payload is None:
        return _contract_search_result(query=query, limit=limit)

    account_id = await _resolve_account_id()
    if account_id is None:
        return _failure_payload(
            "account_binding_required",
            "A single Content Ops account binding is required before drafts can be verified.",
            results=[],
        )

    result = await verify_server.run_content_ops_review_for_bound_tenant(
        verify_server._review_request_from_tool_args(
            asset_id=request_payload.get("asset_id"),
            rule_packet=request_payload.get("rule_packet"),
            coverage=request_payload.get("coverage"),
            extracted_claims=request_payload.get("extracted_claims"),
            quality_reports=request_payload.get("quality_reports"),
            brand_voice_payload=request_payload.get("brand_voice_payload"),
            comments=request_payload.get("comments"),
            adversarial_passes=request_payload.get("adversarial_passes"),
            calibration_library=request_payload.get("calibration_library"),
            as_of=request_payload.get("as_of"),
        ),
        account_resolver=verify_server.StaticContentOpsMarketerAccountResolver(account_id),
        registry_reader=verify_server._get_registry_reader(),
    )
    verdict = result.as_dict()
    verdict_id = _verdict_id(account_id=account_id, request_payload=request_payload)
    _verdict_cache[verdict_id] = _CachedVerdict(account_id=account_id, payload=verdict)
    decision = _clean(verdict.get("decision")) or "unknown"
    return {
        "results": [
            {
                "id": verdict_id,
                "title": f"Content Ops verification: {decision}",
                "url": _result_url(verdict_id),
            }
        ][:_bounded_limit(limit)],
        "metadata": {
            "ok": True,
            "mode": "verification",
            "decision": decision,
            "count": 1,
        },
    }


@mcp.tool(structured_output=True)
async def fetch(id: str) -> dict[str, Any]:
    """Fetch the adapter contract or a cached tenant-bound verdict document."""
    document_id = _clean(id)
    if not document_id:
        return _failure_payload("id_required", "fetch requires a contract or verdict ID.")
    if document_id == CONTRACT_ID:
        return _contract_document()

    account_id = await _resolve_account_id()
    if account_id is None:
        return _failure_payload(
            "account_binding_required",
            "A single Content Ops account binding is required before verdicts can be fetched.",
        )

    cached = _verdict_cache.get(document_id)
    if cached is None or cached.account_id != account_id:
        return _failure_payload(
            "verdict_not_found",
            "No cached Content Ops verdict was found for this ID in the bound account.",
        )

    return {
        "id": document_id,
        "title": _verdict_title(cached.payload),
        "text": _verdict_text(cached.payload),
        "url": _result_url(document_id),
        "metadata": {
            "ok": True,
            "found": True,
            "verdict": cached.payload,
        },
    }


async def _resolve_account_id() -> str | None:
    resolver = verify_server._get_account_resolver()
    return _clean(await resolver.resolve_account_id()) or None


def _json_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return dict(decoded) if isinstance(decoded, dict) else None


def _contract_search_result(*, query: Any, limit: Any) -> dict[str, Any]:
    return {
        "results": [
            {
                "id": CONTRACT_ID,
                "title": "Content Ops verify draft JSON contract",
                "url": _result_url(CONTRACT_ID),
            }
        ][:_bounded_limit(limit)],
        "metadata": {
            "ok": True,
            "mode": "contract",
            "query": _clean(query),
            "count": 1,
        },
    }


def _contract_document() -> dict[str, Any]:
    text = (
        "Empty or non-JSON search query values return this contract. A "
        "JSON-encoded string whose decoded value is an object is treated as one "
        "Content Ops review submission with these fields: asset_id, rule_packet, "
        "coverage, extracted_claims, quality_reports, brand_voice_payload, "
        "comments, and as_of, plus the optional adversarial_passes and "
        "calibration_library. Pass submissions as query=json.dumps(example); "
        "query is a string in the tool schema, not an object. The adapter "
        "returns a tenant-bound verdict ID. Call fetch with that ID to read the "
        "full Content Ops review verdict."
    )
    return {
        "id": CONTRACT_ID,
        "title": "Content Ops verify draft JSON contract",
        "text": text,
        "url": _result_url(CONTRACT_ID),
        "metadata": {
            "ok": True,
            "found": True,
            "type": "adapter_contract",
            "accepted_fields": list(_ACCEPTED_FIELDS) + list(_OPTIONAL_FIELDS),
            "dispatch": _contract_dispatch(),
            "schema": _contract_schema(),
            "example": _contract_example(),
        },
    }


def _contract_dispatch() -> dict[str, Any]:
    example = _contract_example()
    return {
        "contract_shape": {"query": ""},
        "submit_shape": {"query": "<JSON-encoded string of an object matching schema/example>"},
        "submit_example_query": json.dumps(example, sort_keys=True),
        "rule": (
            "empty or non-string-typed query returns this contract; a JSON-encoded "
            "string whose decoded value is an object submits a review"
        ),
    }


def _contract_example() -> dict[str, Any]:
    return {
        "asset_id": "draft-123",
        "rule_packet": {
            "brief": "rules/brief.v3",
            "brand_voice": "rules/brand_voice.v2",
            "claim_registry": "rules/claims.v5",
            "compliance": "rules/compliance.v1",
            "channel_schema": "rules/channel_schema.v2",
        },
        "coverage": [
            {
                "rule_id": "CLAIM-001",
                "requirement": "Every claim must map to approved registry wording",
                "required": True,
                "status": "pass",
                "evidence": "Claim mapped to approved registry entry feature.sso",
            }
        ],
        "quality_reports": {
            "passed": True,
            "findings": [],
        },
        "brand_voice_payload": {
            "passed": True,
            "warnings": [],
            "banned_terms": [],
        },
        "extracted_claims": [
            {
                "text": "SSO is included on every plan",
                "location": "draft:section-2",
                "registry_id": "feature.sso",
            }
        ],
        "comments": [
            {
                "category": "nit",
                "message": "Optional reviewer note",
                "evidence": "",
                "blocking": False,
            }
        ],
        "adversarial_passes": [
            {
                "pass_id": "pass-b",
                "source": "adversarial-prompt@v1 / model-b",
                "findings": [
                    {
                        "category": "overclaim",
                        "message": "The 40% claim has no cited source",
                        "evidence": "cuts support tickets by 40%",
                        "location": "draft:section-2",
                    }
                ],
            }
        ],
        "calibration_library": [
            {
                "example_id": "overclaim-001",
                "label": "overclaim",
                "excerpt": "guaranteed 99.99% uptime",
                "reasoning": "No SLA backs this number; reads as an overclaim.",
                "source": "curated",
            }
        ],
        "as_of": "2026-06-09",
    }


def _contract_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": list(_ACCEPTED_FIELDS),
        "properties": {
            "asset_id": {"type": "string"},
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
                        "required": {"type": "boolean"},
                        "status": {"enum": ["pass", "fail", "not_applicable", "unresolved"]},
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
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "findings": {"type": "array"},
                },
            },
            "brand_voice_payload": {
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                    "banned_terms": {"type": "array", "items": {"type": "string"}},
                },
            },
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "message": {"type": "string"},
                        "evidence": {"type": "string"},
                        "blocking": {"type": "boolean"},
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
                                        "enum": [
                                            "overclaim",
                                            "ambiguity",
                                            "reader_objection",
                                            "promise_cta_mismatch",
                                            "generic_stretch",
                                            "missing_proof",
                                            "voice_slip",
                                        ]
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
                            ]
                        },
                        "excerpt": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "source": {"type": "string"},
                    },
                },
            },
            "as_of": {"type": "string", "format": "date"},
        },
    }


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


def _verdict_id(*, account_id: str, request_payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"account_id": account_id, "request": request_payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{_RESULT_PREFIX}{digest}"


def _verdict_title(payload: dict[str, Any]) -> str:
    decision = _clean(payload.get("decision")) or "unknown"
    content_pr = payload.get("content_pr")
    asset_id = _clean(content_pr.get("asset_id")) if isinstance(content_pr, dict) else ""
    return f"Content Ops verification for {asset_id or 'draft'}: {decision}"


def _verdict_text(payload: dict[str, Any]) -> str:
    decision = _clean(payload.get("decision")) or "unknown"
    reasons = payload.get("reasons")
    reason_lines = [str(reason) for reason in reasons] if isinstance(reasons, list) else []
    if not reason_lines:
        reason_lines = ["No blocking reasons returned."]
    sections = [
        "Decision: " + decision + "\nReasons:\n- " + "\n- ".join(reason_lines),
    ]
    objections = _comment_lines(payload)
    if objections:
        sections.append("Objections:\n- " + "\n- ".join(objections))
    anchors = _anchor_lines(payload)
    if anchors:
        sections.append("Calibration anchors:\n- " + "\n- ".join(anchors))
    return "\n".join(sections)


def _comment_lines(payload: dict[str, Any]) -> list[str]:
    content_pr = payload.get("content_pr")
    comments = content_pr.get("comments") if isinstance(content_pr, dict) else None
    if not isinstance(comments, list):
        return []
    lines: list[str] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        message = _clean(comment.get("message"))
        if not message:
            continue
        category = _clean(comment.get("category")) or "comment"
        marker = " [BLOCKING]" if comment.get("blocking") is True else ""
        evidence = _clean(comment.get("evidence"))
        suffix = f" (evidence: {evidence})" if evidence else ""
        lines.append(f"[{category}]{marker} {message}{suffix}")
    return lines


def _anchor_lines(payload: dict[str, Any]) -> list[str]:
    anchors = payload.get("calibration_anchors")
    if not isinstance(anchors, list):
        return []
    lines: list[str] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        excerpt = _clean(anchor.get("excerpt"))
        if not excerpt:
            continue
        label = _clean(anchor.get("label")) or "anchor"
        reasoning = _clean(anchor.get("reasoning"))
        tail = f" -- {reasoning}" if reasoning else ""
        lines.append(f"{label}: {excerpt}{tail}")
    return lines


def _result_url(document_id: str) -> str:
    return f"atlas://content-ops/marketer-verify/{document_id}"


def _bounded_limit(value: Any) -> int:
    return max(1, min(value if isinstance(value, int) else 10, 10))


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _streamable_http_app():
    """Build the authenticated streamable HTTP app for ChatGPT adapter tools."""
    if verify_server._http_auth_mode() == verify_server._AUTH_MODE_OAUTH:
        verify_server._configure_oauth_auth(target_mcp=mcp)
        return verify_server._apply_content_ops_public_client_metadata(mcp.streamable_http_app())

    from .auth import BearerAuthMiddleware

    return BearerAuthMiddleware(
        mcp.streamable_http_app(),
        token=verify_server._require_http_auth_token(),
    )


if __name__ == "__main__":
    if "--sse" in sys.argv:
        import anyio
        import uvicorn
        from mcp.server.transport_security import TransportSecuritySettings

        from ..config import settings
        from ..config_defaults import (
            DEFAULT_CONTENT_OPS_MARKETER_VERIFY_PORT,
            DEFAULT_MCP_HOST,
        )

        host = settings.mcp.host or DEFAULT_MCP_HOST
        port = settings.mcp.content_ops_marketer_verify_port or DEFAULT_CONTENT_OPS_MARKETER_VERIFY_PORT

        mcp.settings.host = host
        mcp.settings.port = port
        if verify_server._http_auth_mode() == verify_server._AUTH_MODE_BEARER:
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
