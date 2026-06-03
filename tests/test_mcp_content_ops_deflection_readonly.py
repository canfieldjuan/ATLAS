from __future__ import annotations

import json
from typing import Any

import pytest

from atlas_brain.config import settings
from atlas_brain.mcp.auth import BearerAuthMiddleware
from atlas_brain.mcp import content_ops_deflection_readonly_server as readonly
from extracted_content_pipeline.deflection_report_access import (
    InMemoryDeflectionReportArtifactStore,
)


READONLY_TOOLS = {"search", "fetch"}
MUTATING_TOOLS = {
    "generate",
    "generate_blog_post",
    "generate_faq_package",
    "generate_landing_page",
    "mark_paid",
    "publish",
    "unlock",
}


def _tool_names() -> set[str]:
    return set(readonly.mcp._tool_manager._tools)


@pytest.fixture(autouse=True)
def _reset_readonly_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readonly, "_store_override", None)
    monkeypatch.setattr(readonly, "_account_resolver_override", None)


def _snapshot(question: str, *, generated: int = 1) -> dict[str, Any]:
    return {
        "summary": {
            "generated": generated,
            "drafted_answer_count": 0,
            "no_proven_answer_count": generated,
        },
        "top_questions": [
            {
                "rank": 1,
                "question": question,
                "ticket_count": 4,
                "weighted_frequency": 7,
                "customer_wording": question,
                "answer": "Open Analytics and export the report.",
                "source_ids": ["ticket-1"],
                "evidence_quotes": ["ticket-1 says export is blocked"],
            }
        ],
    }


def test_content_ops_deflection_readonly_exposes_exact_chatgpt_tool_surface() -> None:
    assert _tool_names() == READONLY_TOOLS
    assert _tool_names().isdisjoint(MUTATING_TOOLS)


@pytest.mark.asyncio
async def test_search_fails_closed_without_account_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingStore:
        async def list_reports(self, **kwargs: Any) -> tuple[()]:
            raise AssertionError("store must not be called without account binding")

    monkeypatch.setattr(readonly, "_store_override", _FailingStore())
    monkeypatch.setattr(
        readonly,
        "_account_resolver_override",
        readonly.StaticContentOpsDeflectionAccountResolver(" "),
    )

    payload = await readonly.search(query="export")

    assert payload["results"] == []
    assert payload["metadata"]["ok"] is False
    assert payload["metadata"]["error"] == "account_binding_required"


@pytest.mark.asyncio
async def test_fetch_fails_closed_without_account_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingStore:
        async def get_artifact_record(self, **kwargs: Any) -> object:
            raise AssertionError("store must not be called without account binding")

    monkeypatch.setattr(readonly, "_store_override", _FailingStore())
    monkeypatch.setattr(
        readonly,
        "_account_resolver_override",
        readonly.StaticContentOpsDeflectionAccountResolver(""),
    )

    payload = await readonly.fetch(id="request-1")

    assert payload["metadata"]["ok"] is False
    assert payload["metadata"]["error"] == "account_binding_required"


@pytest.mark.asyncio
async def test_search_lists_only_bound_account_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await store.save_report(
        account_id="acct-1",
        request_id="request-export",
        snapshot=_snapshot("How do I export attribution reports?"),
        artifact={"markdown": "# Full report"},
    )
    await store.save_report(
        account_id="acct-2",
        request_id="request-sso",
        snapshot=_snapshot("How do I enable SSO?"),
        artifact={"markdown": "# Other tenant"},
    )
    monkeypatch.setattr(readonly, "_store_override", store)
    monkeypatch.setattr(
        readonly,
        "_account_resolver_override",
        readonly.StaticContentOpsDeflectionAccountResolver("acct-1"),
    )

    payload = await readonly.search(query="export", limit=10)

    assert payload["metadata"]["ok"] is True
    assert payload["metadata"]["count"] == 1
    assert payload["results"] == [
        {
            "id": "request-export",
            "title": "Deflection report: How do I export attribution reports?",
            "url": "https://atlas.local/content-ops/deflection-reports/request-export",
        }
    ]


@pytest.mark.asyncio
async def test_search_applies_limit_after_query_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await store.save_report(
        account_id="acct-1",
        request_id="request-old-export",
        snapshot=_snapshot("How do I export attribution reports?"),
        artifact={},
    )
    await store.save_report(
        account_id="acct-1",
        request_id="request-new-sso",
        snapshot=_snapshot("How do I enable SSO?"),
        artifact={},
    )
    await store.save_report(
        account_id="acct-1",
        request_id="request-new-permissions",
        snapshot=_snapshot("How do I update permissions?"),
        artifact={},
    )
    monkeypatch.setattr(readonly, "_store_override", store)
    monkeypatch.setattr(
        readonly,
        "_account_resolver_override",
        readonly.StaticContentOpsDeflectionAccountResolver("acct-1"),
    )

    payload = await readonly.search(query="export", limit=1)

    assert payload["metadata"]["count"] == 1
    assert payload["results"] == [
        {
            "id": "request-old-export",
            "title": "Deflection report: How do I export attribution reports?",
            "url": "https://atlas.local/content-ops/deflection-reports/request-old-export",
        }
    ]


@pytest.mark.asyncio
async def test_fetch_uses_bound_account_and_excludes_paid_artifact_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryDeflectionReportArtifactStore()
    await store.save_report(
        account_id="acct-1",
        request_id="request-export",
        snapshot=_snapshot("How do I export attribution reports?"),
        artifact={
            "markdown": "# Full report\n\nOpen Analytics and export the report.",
            "faq_result": {
                "items": [
                    {
                        "answer": "Open Analytics.",
                        "source_ids": ["ticket-1"],
                        "evidence_quotes": ["ticket-1 quote"],
                    }
                ]
            },
        },
    )
    await store.save_report(
        account_id="acct-2",
        request_id="request-other",
        snapshot=_snapshot("How do I enable SSO?"),
        artifact={"markdown": "# Other tenant"},
    )
    monkeypatch.setattr(readonly, "_store_override", store)
    monkeypatch.setattr(
        readonly,
        "_account_resolver_override",
        readonly.StaticContentOpsDeflectionAccountResolver("acct-1"),
    )

    payload = await readonly.fetch(id="request-export")
    missing = await readonly.fetch(id="request-other")
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["metadata"]["found"] is True
    assert payload["metadata"]["unlock_status"] == {
        "paid": False,
        "full_report_locked": True,
    }
    assert payload["metadata"]["content_opportunities"] == [
        {
            "rank": 1,
            "question": "How do I export attribution reports?",
            "ticket_count": 4,
            "weighted_frequency": 7,
            "customer_wording": "How do I export attribution reports?",
            "opportunity_score": 7,
            "coverage_status": "locked_snapshot",
            "recommended_content_action": (
                "Create or improve an FAQ entry for this repeated customer question."
            ),
            "unlock_hint": "Unlock the full report for detailed source-backed guidance.",
        }
    ]
    assert missing["metadata"]["found"] is False
    assert "Open Analytics" not in encoded
    assert "ticket-1" not in encoded
    assert "markdown" not in encoded
    assert "faq_result" not in encoded
    assert "source_ids" not in encoded
    assert "evidence_quotes" not in encoded


def test_content_ops_deflection_readonly_http_requires_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", "")

    with pytest.raises(RuntimeError, match="ATLAS_MCP_AUTH_TOKEN is required"):
        readonly._streamable_http_app()


def test_content_ops_deflection_readonly_http_wraps_with_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", "test-token-with-enough-entropy")

    app = readonly._streamable_http_app()

    assert isinstance(app, BearerAuthMiddleware)


@pytest.mark.parametrize("token", ["<token>", "token", "test-token", "test-readonly-token"])
def test_content_ops_deflection_readonly_http_rejects_placeholder_tokens(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", token)

    with pytest.raises(RuntimeError, match="must not be a placeholder"):
        readonly._streamable_http_app()


def test_content_ops_deflection_readonly_http_rejects_short_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.mcp, "auth_token", "short-token-value")

    with pytest.raises(RuntimeError, match="at least 24 characters"):
        readonly._streamable_http_app()
