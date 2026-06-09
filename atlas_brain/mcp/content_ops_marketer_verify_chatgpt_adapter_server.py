"""ChatGPT-compatible search/fetch adapter for Content Ops draft verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import content_ops_marketer_verify_server as verify_server


CONTRACT_ID = "content-ops-verify-draft-contract"
_RESULT_PREFIX = "content-ops-verdict:"


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
        "Submit one JSON object through search with these fields: asset_id, "
        "rule_packet, coverage, extracted_claims, quality_reports, "
        "brand_voice_payload, comments, and as_of. The adapter returns a "
        "tenant-bound verdict ID. Call fetch with that ID to read the full "
        "Content Ops review verdict."
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
            "accepted_fields": [
                "asset_id",
                "rule_packet",
                "coverage",
                "extracted_claims",
                "quality_reports",
                "brand_voice_payload",
                "comments",
                "as_of",
            ],
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
    asset_id = _clean(payload.get("asset_id")) or "draft"
    return f"Content Ops verification for {asset_id}: {decision}"


def _verdict_text(payload: dict[str, Any]) -> str:
    decision = _clean(payload.get("decision")) or "unknown"
    reasons = payload.get("reasons")
    reason_lines = [str(reason) for reason in reasons] if isinstance(reasons, list) else []
    if not reason_lines:
        reason_lines = ["No blocking reasons returned."]
    return "Decision: " + decision + "\nReasons:\n- " + "\n- ".join(reason_lines)


def _result_url(document_id: str) -> str:
    return f"atlas://content-ops/marketer-verify/{document_id}"


def _bounded_limit(value: Any) -> int:
    return max(1, min(value if isinstance(value, int) else 10, 10))


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


if __name__ == "__main__":
    mcp.run(transport="stdio")
