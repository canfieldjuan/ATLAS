from __future__ import annotations

import csv
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from atlas_brain._content_ops_input_provider import build_content_ops_input_provider
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.blog_generation import BlogPostGenerationService
from extracted_content_pipeline.campaign_ports import LLMMessage, LLMResponse
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices
from extracted_content_pipeline.content_ops_input_provider import (
    merge_content_ops_input_package,
)
from extracted_content_pipeline.control_surfaces import (
    preview_control_surface,
    request_from_mapping,
)
from extracted_content_pipeline.support_ticket_input_provider import (
    SupportTicketInputProvider,
)
from extracted_content_pipeline.landing_page_generation import (
    LandingPageGenerationService,
)
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownService
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft
from tests.content_ops_live_execute_harness import MemoryDraftRepository, MemorySkillStore


ROOT = Path(__file__).resolve().parents[1]
FAQ_DRAFT_ID = "11111111-1111-4111-8111-111111111111"
MISSING_FAQ_DRAFT_ID = "22222222-2222-4222-8222-222222222222"


def _fresh_api_package():
    original = sys.modules.pop("atlas_brain.api", None)
    try:
        return importlib.import_module("atlas_brain.api")
    finally:
        if original is not None:
            sys.modules["atlas_brain.api"] = original


def _route(api_pkg, path: str):
    route = next((route for route in api_pkg.router.routes if route.path == path), None)
    assert route is not None, f"Route {path!r} not mounted"
    return route


def _router_route(router, path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) == path and method.upper() in getattr(
            route,
            "methods",
            set(),
        ):
            return route
    raise AssertionError(f"Route {method.upper()} {path!r} not mounted")


def _ticket_payload() -> dict[str, object]:
    return {
        "inputs": {
            "source_material": [
                {
                    "subject": "Billing question",
                    "body": "Why was I charged twice this month exactly?",
                }
            ]
        }
    }


def _support_ticket_csv_rows() -> list[dict[str, str]]:
    path = ROOT / "extracted_content_pipeline" / "examples" / "support_ticket_sources.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _saved_faq_draft(faq_id: str = FAQ_DRAFT_ID) -> TicketFAQDraft:
    return TicketFAQDraft(
        id=faq_id,
        target_id="ticket-faq-report",
        target_mode="support_ticket_faq",
        title="Saved FAQ report",
        markdown="# Saved FAQ report",
        items=(
            {
                "topic": "billing confusion",
                "question": "Why was I charged twice?",
                "summary": "Customers ask why duplicate-looking invoices appear.",
                "steps": ("Check invoice history.",),
                "answer_evidence_status": "resolution_evidence",
            },
        ),
        source_count=2,
        ticket_source_count=2,
        output_checks={"has_action_items": True},
        status="draft",
    )


class _FAQRepo:
    def __init__(self, pool):
        self.pool = pool

    async def get_draft(self, faq_id: str, *, scope: TenantScope):
        self.pool["calls"].append((faq_id, scope.account_id))
        return self.pool["drafts"].get(faq_id)


class _OpportunityRepo:
    def __init__(self, pool):
        self.pool = pool

    async def read_campaign_opportunities(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters=None,
    ):
        filters = dict(filters or {})
        target_id = filters.get("target_id")
        self.pool["opportunity_calls"].append({
            "account_id": scope.account_id,
            "target_mode": target_mode,
            "limit": limit,
            "target_id": target_id,
        })
        rows = self.pool["opportunities"].get(target_id, ())
        return tuple(rows)[:limit]


class _BlueprintRepo:
    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.calls: list[dict[str, object]] = []

    async def read_blog_blueprints(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters=None,
    ):
        self.calls.append({
            "account_id": scope.account_id,
            "target_mode": target_mode,
            "limit": limit,
            "filters": dict(filters or {}),
        })
        return self.rows[:limit]


class _ContentLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        del max_tokens, temperature
        meta = dict(metadata or {})
        self.calls.append({
            "messages": tuple(messages),
            "metadata": meta,
        })
        return LLMResponse(
            content=json.dumps(_content_response_for_metadata(meta), separators=(",", ":")),
            model="persisted-source-target-test",
            usage={"input_tokens": 3, "output_tokens": 7},
        )


def _content_response_for_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    asset_type = str(metadata.get("asset_type") or "")
    if asset_type == "landing_page":
        return {
            "title": "FAQ Report From Imported Tickets",
            "slug": "faq-report-from-imported-tickets",
            "hero": {
                "headline": "Turn imported support tickets into FAQ answers",
                "subheadline": "Use real support-ticket questions to build pages.",
            },
            "sections": [{
                "id": "proof",
                "title": "Imported Tickets Show Repeat Questions",
                "body_markdown": "Imported support tickets show billing and login questions.",
            }],
            "cta": {"label": "Upload Ticket CSV", "url": "/systems/ai-content-ops/intake"},
            "meta": {
                "title_tag": "FAQ Report From Imported Tickets",
                "description": "Build public answers from imported support-ticket rows.",
            },
            "reference_ids": ["ticket-1", "ticket-2"],
        }
    if asset_type == "blog_post":
        return {
            "slug": "imported-support-ticket-faq-report",
            "title": "Imported Support Ticket FAQ Report",
            "content": (
                "## What do imported support tickets show?\n\n"
                "Imported support tickets show repeat billing and account questions."
            ),
            "topic_type": "content_ops_support_ticket_faq",
            "description": "How imported tickets become FAQ content.",
            "seo_title": "Imported Support Ticket FAQ Report",
            "seo_description": "Use imported support tickets to find repeat questions.",
            "target_keyword": "support ticket FAQ report",
            "secondary_keywords": ["customer support FAQ"],
            "faq": [{
                "question": "What do imported support tickets show?",
                "answer": "They show repeat questions customers already asked.",
            }],
        }
    raise AssertionError(f"unexpected LLM metadata: {metadata!r}")


def _persisted_ticket_row(target_id: str, *, subject: str | None = None) -> dict[str, object]:
    return {
        "target_id": target_id,
        "ticket_id": target_id,
        "source_type": "support_ticket",
        "subject": subject or "Billing renewal question",
        "message": "How do I confirm my renewal invoice before payment?",
        "pain_category": "billing",
    }


def test_atlas_content_ops_input_provider_noops_without_source_material() -> None:
    provider = build_content_ops_input_provider()

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Revenue audit",
            },
        },
    )
    payload = merge_content_ops_input_package(
        {
            "outputs": ["email_campaign"],
            "inputs": {
                "target_account": "Acme",
                "offer": "Revenue audit",
            },
        },
        package,
    )
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert package.metadata == {
        "source": "atlas_content_ops_input_provider",
        "mode": "noop",
    }
    assert payload["outputs"] == ["email_campaign"]
    assert payload["inputs"] == {
        "target_account": "Acme",
        "offer": "Revenue audit",
    }
    assert request.ingestion_profile == "domain_specific"
    assert preview.can_run is True


def test_atlas_content_ops_input_provider_noops_for_blank_source_material() -> None:
    provider = build_content_ops_input_provider()

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "outputs": ["faq_markdown"],
            "inputs": {"source_material": ["  ", "\n"]},
        },
    )
    payload = merge_content_ops_input_package(
        {
            "outputs": ["faq_markdown"],
            "inputs": {"source_material": ["  ", "\n"]},
        },
        package,
    )

    assert package.metadata["mode"] == "noop"
    assert payload["outputs"] == ["faq_markdown"]
    assert payload["inputs"] == {"source_material": ["  ", "\n"]}


def test_atlas_content_ops_input_provider_noops_for_generic_source_material() -> None:
    provider = build_content_ops_input_provider()

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "outputs": ["signal_extraction"],
            "inputs": {
                "source_material": [
                    {
                        "source_id": "review-1",
                        "source_type": "review",
                        "text": "Export settings are hard to find.",
                    }
                ]
            },
        },
    )
    payload = merge_content_ops_input_package(
        {
            "outputs": ["signal_extraction"],
            "inputs": {
                "source_material": [
                    {
                        "source_id": "review-1",
                        "source_type": "review",
                        "text": "Export settings are hard to find.",
                    }
                ]
            },
        },
        package,
    )
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert package.metadata["mode"] == "noop"
    assert request.outputs == ("signal_extraction",)
    assert request.inputs["source_material"][0]["source_type"] == "review"
    assert preview.can_run is True


def test_atlas_content_ops_input_provider_expands_support_ticket_source_material() -> None:
    provider = build_content_ops_input_provider()

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_material": [
                    {
                        "ticket_id": "ticket-1",
                        "subject": "How do I change my login email?",
                        "description": "I cannot find the account email setting.",
                    }
                ]
            }
        },
    )
    payload = merge_content_ops_input_package({"inputs": {}}, package)
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert request.outputs == ("faq_markdown", "landing_page", "blog_post")
    assert request.ingestion_profile == "existing_evidence"
    assert request.inputs["faq_questions"] == ["How do I change my login email?"]
    assert request.inputs["source_material"][0]["source_id"] == "ticket-1"
    assert preview.can_run is True


def test_atlas_content_ops_input_provider_expands_selected_faq_output() -> None:
    provider = build_content_ops_input_provider()

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_material": {
                    "id": "faq-draft-42",
                    "title": "Saved FAQ report",
                    "markdown": "# Saved FAQ report",
                    "ticket_source_count": 3,
                    "output_checks": {"has_action_items": True},
                    "items": [
                        {
                            "topic": "billing confusion",
                            "question": "Why was I charged twice?",
                            "summary": (
                                "Customers ask why duplicate-looking invoices "
                                "appear."
                            ),
                            "steps": [
                                "Check whether the second charge is pending.",
                                "Confirm the invoice workspace.",
                            ],
                            "answer_evidence_status": "resolution_evidence",
                            "source_ids": ["ticket-1", "ticket-2"],
                        },
                        {
                            "topic": "export setup",
                            "question": "How do I export the report?",
                            "summary": "Customers ask where report exports live.",
                            "steps": [
                                "Draft answer - support team should add the "
                                "verified resolution."
                            ],
                            "answer_evidence_status": "draft_needs_review",
                            "source_ids": ["ticket-3"],
                        },
                    ],
                }
            }
        },
    )
    payload = merge_content_ops_input_package({"inputs": {}}, package)
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert package.provider == "atlas_support_ticket_request"
    assert request.outputs == ("faq_markdown", "landing_page", "blog_post")
    assert request.inputs["source_material"][0]["source_type"] == "faq_output"
    assert request.inputs["source_material"][0]["source_id"] == "faq-draft-42:item-1"
    assert request.inputs["source_material"][0]["faq_draft_id"] == "faq-draft-42"
    assert request.inputs["source_material"][0]["resolution_text"] == (
        "Check whether the second charge is pending. Confirm the invoice workspace."
    )
    assert "resolution_text" not in request.inputs["source_material"][1]
    assert request.inputs["support_ticket_resolution_evidence_present"] is True
    assert request.inputs["support_ticket_resolution_evidence_count"] == 1
    assert request.inputs["support_ticket_resolution_examples"][0] == {
        "source_id": "faq-draft-42:item-1",
        "source_title": "Why was I charged twice?",
        "text": (
            "Check whether the second charge is pending. Confirm the invoice "
            "workspace."
        ),
    }
    assert preview.can_run is True


def test_atlas_content_ops_input_provider_expands_faq_output_inside_source_lists() -> None:
    provider = build_content_ops_input_provider()

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_material": [
                    {
                        "ticket_id": "ticket-1",
                        "subject": "How do I update billing?",
                        "description": "I cannot find the billing page.",
                    },
                    {
                        "id": "faq-draft-42",
                        "markdown": "# Saved FAQ report",
                        "ticket_source_count": 2,
                        "items": [
                            {
                                "topic": "billing confusion",
                                "question": "Why was I charged twice?",
                                "summary": (
                                    "Customers ask why duplicate-looking invoices "
                                    "appear."
                                ),
                                "steps": ("Check invoice history.",),
                                "answer_evidence_status": "resolution_evidence",
                            },
                        ],
                    },
                ]
            }
        },
    )

    assert package.metadata["included_row_count"] == 2
    assert package.inputs["source_material"][0]["source_type"] == "support_ticket"
    assert package.inputs["source_material"][1]["source_type"] == "faq_output"
    assert package.inputs["source_material"][1]["source_id"] == "faq-draft-42"
    assert package.inputs["source_material"][1]["faq_draft_id"] == "faq-draft-42"
    assert package.inputs["support_ticket_resolution_evidence_present"] is True
    assert package.inputs["support_ticket_resolution_evidence_count"] == 1


@pytest.mark.asyncio
async def test_atlas_content_ops_input_provider_fetches_selected_faq_ids_by_scope() -> None:
    pool = {
        "drafts": {FAQ_DRAFT_ID: _saved_faq_draft()},
        "calls": [],
    }
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        faq_repository_factory=_FAQRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={"inputs": {"source_faq_ids": [FAQ_DRAFT_ID]}},
    )

    assert pool["calls"] == [(FAQ_DRAFT_ID, "acct-1")]
    assert package.provider == "atlas_support_ticket_request"
    assert package.metadata["selected_faq_id_count"] == 1
    assert package.metadata["selected_faq_loaded_count"] == 1
    assert package.metadata["selected_faq_missing_id_count"] == 0
    assert package.warnings == ()
    assert package.inputs["source_material"][0]["source_type"] == "faq_output"
    assert package.inputs["source_material"][0]["source_id"] == FAQ_DRAFT_ID
    assert package.inputs["source_material"][0]["faq_draft_id"] == FAQ_DRAFT_ID
    assert package.inputs["support_ticket_resolution_evidence_present"] is True
    assert package.inputs["support_ticket_resolution_evidence_count"] == 1


@pytest.mark.asyncio
async def test_atlas_content_ops_input_provider_fetches_persisted_source_targets_by_scope() -> None:
    pool = {
        "opportunities": {
            "ticket-1": [_persisted_ticket_row("ticket-1")],
            "ticket-2": [_persisted_ticket_row("ticket-2", subject="Login email change")],
        },
        "opportunity_calls": [],
    }
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        opportunity_repository_factory=_OpportunityRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "target_mode": "support_ticket_faq",
            "inputs": {
                "source_import_target_ids": ["ticket-1", "ticket-2", "ticket-1"],
            },
        },
    )

    assert pool["opportunity_calls"] == [
        {
            "account_id": "acct-1",
            "target_mode": "support_ticket_faq",
            "limit": 2,
            "target_id": "ticket-1",
        },
        {
            "account_id": "acct-1",
            "target_mode": "support_ticket_faq",
            "limit": 2,
            "target_id": "ticket-2",
        },
    ]
    assert package.provider == "atlas_support_ticket_request"
    assert package.metadata["source_target_id_count"] == 2
    assert package.metadata["source_target_loaded_count"] == 2
    assert package.metadata["source_target_missing_id_count"] == 0
    assert package.metadata["source_target_ambiguous_id_count"] == 0
    assert package.warnings == ()
    assert [row["source_id"] for row in package.inputs["source_material"]] == [
        "ticket-1",
        "ticket-2",
    ]
    assert package.inputs["faq_questions"] == [
        "How do I confirm my renewal invoice before payment?"
    ]


@pytest.mark.asyncio
async def test_atlas_content_ops_input_provider_skips_persisted_targets_without_account_scope() -> None:
    pool = {
        "opportunities": {
            "ticket-1": [_persisted_ticket_row("ticket-1")],
        },
        "opportunity_calls": [],
    }
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        opportunity_repository_factory=_OpportunityRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(),
        request={"inputs": {"source_import_target_ids": ["ticket-1"]}},
    )

    assert pool["opportunity_calls"] == []
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["source_target_id_count"] == 1
    assert package.metadata["source_target_loaded_count"] == 0
    assert package.metadata["source_target_missing_id_count"] == 1
    assert package.warnings == ({
        "code": "source_import_targets_unscoped",
        "message": (
            "Persisted import target IDs require an account-scoped Content "
            "Ops request."
        ),
        "missing_count": 1,
    },)


@pytest.mark.asyncio
async def test_atlas_content_ops_input_provider_fails_safe_on_ambiguous_persisted_targets() -> None:
    pool = {
        "opportunities": {
            "ticket-1": [
                _persisted_ticket_row("ticket-1", subject="First duplicate"),
                _persisted_ticket_row("ticket-1", subject="Second duplicate"),
            ],
        },
        "opportunity_calls": [],
    }
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        opportunity_repository_factory=_OpportunityRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={"inputs": {"source_target_ids": ["ticket-1"]}},
    )

    assert pool["opportunity_calls"] == [
        {
            "account_id": "acct-1",
            "target_mode": "vendor_retention",
            "limit": 2,
            "target_id": "ticket-1",
        }
    ]
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["source_target_loaded_count"] == 0
    assert package.metadata["source_target_ambiguous_id_count"] == 1
    assert package.warnings == ({
        "code": "source_import_targets_ambiguous",
        "message": "One or more persisted import target IDs matched multiple rows.",
        "ambiguous_count": 1,
    },)


@pytest.mark.asyncio
async def test_atlas_content_ops_input_provider_warns_for_missing_selected_faq_ids() -> None:
    pool = {"drafts": {}, "calls": []}
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        faq_repository_factory=_FAQRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={"inputs": {"source_faq_ids": [MISSING_FAQ_DRAFT_ID]}},
    )

    assert pool["calls"] == [(MISSING_FAQ_DRAFT_ID, "acct-1")]
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["selected_faq_id_count"] == 1
    assert package.metadata["selected_faq_loaded_count"] == 0
    assert package.metadata["selected_faq_missing_id_count"] == 1
    assert package.warnings == ({
        "code": "source_faq_drafts_not_found",
        "message": "One or more selected FAQ reports were not found for this account.",
        "missing_count": 1,
    },)


@pytest.mark.asyncio
async def test_atlas_content_ops_input_provider_warns_for_invalid_selected_faq_ids() -> None:
    pool = {"drafts": {FAQ_DRAFT_ID: _saved_faq_draft()}, "calls": []}
    provider = build_content_ops_input_provider(
        pool_provider=lambda: pool,
        faq_repository_factory=_FAQRepo,
    )

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={"inputs": {"source_faq_ids": ["not-a-uuid"]}},
    )

    assert pool["calls"] == []
    assert package.inputs == {}
    assert package.metadata["mode"] == "noop"
    assert package.metadata["selected_faq_id_count"] == 1
    assert package.metadata["selected_faq_loaded_count"] == 0
    assert package.metadata["selected_faq_missing_id_count"] == 0
    assert package.metadata["selected_faq_invalid_id_count"] == 1
    assert package.warnings == ({
        "code": "source_faq_ids_invalid",
        "message": "One or more selected FAQ report IDs are invalid.",
        "invalid_count": 1,
    },)


def test_atlas_content_ops_input_provider_expands_subject_body_ticket_rows() -> None:
    provider = build_content_ops_input_provider()

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_material": [
                    {
                        "subject": "Billing question",
                        "body": "Why was I charged twice this month exactly?",
                    }
                ]
            }
        },
    )

    assert package.metadata["source_row_count"] == 1
    assert package.inputs["faq_questions"] == [
        "Why was I charged twice this month exactly?"
    ]
    assert package.inputs["source_material"][0]["source_id"] == "ticket-1"


def test_atlas_content_ops_input_provider_uses_same_list_and_bundle_gate() -> None:
    provider = build_content_ops_input_provider()
    request = {
        "inputs": {
            "source_material": [
                {
                    "subject": "Billing question",
                    "body": "Why was I charged twice this month exactly?",
                }
            ]
        }
    }
    bundled_request = {
        "inputs": {
            "source_material": {
                "support_tickets": request["inputs"]["source_material"],
            }
        }
    }

    list_package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request=request,
    )
    bundled_package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request=bundled_request,
    )

    assert list_package.metadata["included_row_count"] == 1
    assert bundled_package.metadata["included_row_count"] == 1
    assert list_package.outputs == bundled_package.outputs


@pytest.mark.skipif(
    importlib.util.find_spec("asyncpg") is None,
    reason="atlas_brain.api imports the host database module when asyncpg is present",
)
def test_api_aggregator_wires_content_ops_input_provider() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/preview")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )

    provider = closure["input_provider"]
    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={
            "inputs": {
                "source_material": [
                    {
                        "ticket_id": "ticket-1",
                        "subject": "Where is my invoice?",
                    }
                ]
            }
        },
    )

    assert provider.__class__.__name__ == "_AtlasSupportTicketInputProvider"
    assert package.inputs["faq_questions"] == ["Where is my invoice?"]


@pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)
@pytest.mark.asyncio
async def test_execute_route_generates_support_ticket_faq_at_inline_cap() -> None:
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/content-ops"),
        input_provider=build_content_ops_input_provider(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=TicketFAQMarkdownService()
        ),
        scope_provider=lambda: {"account_id": "acct-route-faq"},
    )
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "subject": "Billing renewal question",
            "message": "How do I confirm my renewal invoice before payment?",
            "pain_category": "billing",
        }
        for index in range(1000)
    ]

    route = _router_route(router, "/content-ops/execute", "POST")
    payload = await route.endpoint({
        "outputs": ["faq_markdown"],
        "inputs": {
            "source_material": {
                "support_tickets": rows,
            },
        },
    })

    step = payload["steps"][0]
    result = step["result"]
    assert payload["status"] == "completed"
    assert payload["errors"] == []
    assert step["output"] == "faq_markdown"
    assert step["status"] == "completed"
    assert result["source_count"] == 1000
    assert result["ticket_source_count"] == 1000
    assert result["generated"] == 1
    assert result["output_checks"] == {
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
        "uses_user_vocabulary": True,
    }
    assert result["saved_ids"] == []
    assert result["items"][0]["ticket_count"] == 1000
    assert len(result["items"][0]["source_ids"]) == 1000


@pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)
@pytest.mark.asyncio
async def test_execute_route_generates_faq_from_persisted_source_targets() -> None:
    pool = {
        "opportunities": {
            "ticket-1": [_persisted_ticket_row("ticket-1")],
            "ticket-2": [_persisted_ticket_row("ticket-2", subject="Login email change")],
        },
        "opportunity_calls": [],
    }
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/content-ops"),
        input_provider=build_content_ops_input_provider(
            pool_provider=lambda: pool,
            opportunity_repository_factory=_OpportunityRepo,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=TicketFAQMarkdownService()
        ),
        scope_provider=lambda: TenantScope(
            account_id="acct-persisted-source-execute",
            user_id="user-persisted-source-execute",
        ),
    )
    route = _router_route(router, "/content-ops/execute", "POST")

    payload = await route.endpoint({
        "outputs": ["faq_markdown"],
        "limit": 2,
        "require_quality_gates": False,
        "inputs": {
            "source_import_target_ids": ["ticket-1", "ticket-2"],
        },
    })

    assert [call["target_id"] for call in pool["opportunity_calls"]] == [
        "ticket-1",
        "ticket-2",
    ]
    assert {call["account_id"] for call in pool["opportunity_calls"]} == {
        "acct-persisted-source-execute"
    }
    assert payload["status"] == "completed"
    assert payload["input_provider"]["provider"] == "atlas_support_ticket_request"
    assert payload["input_provider"]["metadata"]["source_row_count"] == 2
    assert payload["input_provider"]["metadata"]["included_row_count"] == 2
    assert payload["input_provider"]["warnings"] == []
    step = payload["steps"][0]
    assert step["output"] == "faq_markdown"
    assert step["status"] == "completed"
    assert step["result"]["source_count"] == 2
    assert step["result"]["ticket_source_count"] == 2


@pytest.mark.skipif(
    api_module.APIRouter is None,
    reason="fastapi is not installed in this test environment",
)
@pytest.mark.asyncio
async def test_execute_route_generates_landing_and_blog_from_persisted_source_targets() -> None:
    pool = {
        "opportunities": {
            "ticket-1": [_persisted_ticket_row("ticket-1")],
            "ticket-2": [_persisted_ticket_row("ticket-2", subject="Login email change")],
        },
        "opportunity_calls": [],
    }
    llm = _ContentLLM()
    landing_pages = MemoryDraftRepository("landing")
    blog_posts = MemoryDraftRepository("blog")
    blueprints = _BlueprintRepo([{
        "id": "bp-imported-support-tickets",
        "slug": "imported-support-ticket-faq-report",
        "topic": "Imported support-ticket questions customers keep asking",
        "topic_type": "content_ops_support_ticket_faq",
    }])
    scope = TenantScope(
        account_id="acct-imported-source-public",
        user_id="user-imported-source-public",
    )
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/content-ops"),
        input_provider=build_content_ops_input_provider(
            pool_provider=lambda: pool,
            opportunity_repository_factory=_OpportunityRepo,
        ),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            landing_page=LandingPageGenerationService(
                landing_pages=landing_pages,
                llm=llm,
                skills=MemorySkillStore(),
            ),
            blog_post=BlogPostGenerationService(
                blueprints=blueprints,
                blog_posts=blog_posts,
                llm=llm,
                skills=MemorySkillStore(),
            ),
        ),
        scope_provider=lambda: scope,
    )
    route = _router_route(router, "/content-ops/execute", "POST")

    payload = await route.endpoint({
        "target_mode": "vendor_retention",
        "outputs": ["landing_page", "blog_post"],
        "limit": 1,
        "require_quality_gates": False,
        "inputs": {
            "source_import_target_ids": ["ticket-1", "ticket-2"],
        },
    })

    assert [call["target_id"] for call in pool["opportunity_calls"]] == [
        "ticket-1",
        "ticket-2",
    ]
    assert {call["account_id"] for call in pool["opportunity_calls"]} == {
        "acct-imported-source-public"
    }
    assert payload["status"] == "completed"
    assert payload["input_provider"]["metadata"]["source_row_count"] == 2
    assert payload["input_provider"]["metadata"]["included_row_count"] == 2
    assert payload["input_provider"]["warnings"] == []

    steps = {step["output"]: step for step in payload["steps"]}
    assert set(steps) == {"landing_page", "blog_post"}
    assert steps["landing_page"]["status"] == "completed"
    assert steps["landing_page"]["result"]["saved_ids"] == ["landing-1"]
    assert steps["blog_post"]["status"] == "completed"
    assert steps["blog_post"]["result"]["saved_ids"] == ["blog-1"]

    assert len(landing_pages.saved) == 1
    landing_scope, landing_draft = landing_pages.saved[0]
    assert landing_scope == scope
    assert landing_draft.slug == "faq-report-from-imported-tickets"
    assert landing_draft.reference_ids == ("ticket-1", "ticket-2")

    assert blueprints.calls == [{
        "account_id": "acct-imported-source-public",
        "target_mode": "vendor_retention",
        "limit": 1,
        "filters": {"topic_type": "content_ops_support_ticket_faq"},
    }]
    assert len(blog_posts.saved) == 1
    blog_scope, blog_draft = blog_posts.saved[0]
    assert blog_scope == scope
    assert blog_draft.slug == "imported-support-ticket-faq-report"
    assert blog_draft.data_context["source"] == "support_ticket_provider"
    assert blog_draft.data_context["included_ticket_row_count"] == 2
    assert blog_draft.data_context["faq_questions"] == [
        "How do I confirm my renewal invoice before payment?"
    ]

    assert [call["metadata"]["asset_type"] for call in llm.calls] == [
        "landing_page",
        "blog_post",
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("asyncpg") is None,
    reason="atlas_brain.api imports the host database module when asyncpg is present",
)
@pytest.mark.asyncio
async def test_api_preview_route_applies_support_ticket_input_provider() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/preview")

    payload = await route.endpoint(_ticket_payload())

    assert payload["can_run"] is True
    assert payload["outputs"] == ["faq_markdown", "landing_page", "blog_post"]
    assert payload["missing_inputs"] == []


@pytest.mark.asyncio
async def test_api_preview_route_fetches_selected_faq_ids() -> None:
    pool = {
        "drafts": {FAQ_DRAFT_ID: _saved_faq_draft()},
        "calls": [],
    }
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/content-ops"),
        input_provider=build_content_ops_input_provider(
            pool_provider=lambda: pool,
            faq_repository_factory=_FAQRepo,
        ),
        scope_provider=lambda: TenantScope(account_id="acct-preview"),
    )

    route = _router_route(router, "/content-ops/preview", "POST")
    payload = await route.endpoint({
        "outputs": ["landing_page"],
        "inputs": {"source_faq_ids": [FAQ_DRAFT_ID]},
    })

    assert pool["calls"] == [(FAQ_DRAFT_ID, "acct-preview")]
    assert payload["can_run"] is True
    assert payload["outputs"] == ["landing_page"]
    assert payload["input_provider"]["provider"] == "atlas_support_ticket_request"
    assert payload["input_provider"]["metadata"]["included_row_count"] == 1
    assert payload["input_provider"]["warnings"] == []


@pytest.mark.asyncio
async def test_preview_route_surfaces_support_ticket_loader_truncation_warning() -> None:
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "subject": "Billing renewal question",
            "body": "How do I confirm my renewal invoice before payment?",
        }
        for index in range(1005)
    ]
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/content-ops"),
        input_provider=SupportTicketInputProvider(source_material_loader=lambda scope, request: rows),
        scope_provider=lambda: {"account_id": "acct-truncation-proof"},
    )

    route = _router_route(router, "/content-ops/preview", "POST")
    payload = await route.endpoint({
        "outputs": ["landing_page"],
    })

    assert payload["can_run"] is True
    assert payload["outputs"] == ["landing_page"]
    assert payload["input_provider"] == {
        "provider": "support_ticket_input_provider",
        "metadata": {
            "included_row_count": 1000,
            "skipped_row_count": 0,
            "source": "support_ticket_input_package",
            "source_period": "Uploaded support tickets",
            "source_row_count": 1005,
            "support_ticket_resolution_evidence_count": 0,
            "support_ticket_resolution_evidence_present": False,
            "truncated_row_count": 5,
        },
        "warnings": [{
            "code": "ticket_rows_truncated",
            "message": "Used first 1000 ticket rows out of 1005.",
            "row_count": 1005,
            "max_rows": 1000,
            "truncated_row_count": 5,
        }],
    }


@pytest.mark.asyncio
async def test_atlas_preview_route_rejects_inline_source_material_over_request_cap() -> None:
    rows = [
        {
            "ticket_id": f"ticket-{index}",
            "subject": "Billing renewal question",
            "body": "How do I confirm my renewal invoice before payment?",
        }
        for index in range(1005)
    ]
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/content-ops"),
        input_provider=build_content_ops_input_provider(),
        scope_provider=lambda: {"account_id": "acct-inline-cap-proof"},
    )

    route = _router_route(router, "/content-ops/preview", "POST")
    with pytest.raises(api_module.HTTPException) as exc:
        await route.endpoint({
            "outputs": ["landing_page"],
            "inputs": {
                "source_material": rows,
            },
        })

    assert exc.value.status_code == 422
    assert "inputs arrays are too large" in str(exc.value.detail)


@pytest.mark.skipif(
    importlib.util.find_spec("asyncpg") is None,
    reason="atlas_brain.api imports the host database module when asyncpg is present",
)
@pytest.mark.asyncio
async def test_api_plan_route_applies_support_ticket_input_provider() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/plan")

    payload = await route.endpoint(_ticket_payload())

    assert payload["can_execute"] is True
    assert [step["output"] for step in payload["steps"]] == [
        "faq_markdown",
        "landing_page",
        "blog_post",
    ]
    assert payload["steps"][0]["runner"] == "TicketFAQMarkdownService.generate"
    assert payload["preview"]["outputs"] == [
        "faq_markdown",
        "landing_page",
        "blog_post",
    ]


@pytest.mark.asyncio
async def test_execute_route_generates_faq_from_support_ticket_input_provider() -> None:
    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(),
        input_provider=build_content_ops_input_provider(),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=TicketFAQMarkdownService(),
        ),
        scope_provider=lambda: TenantScope(
            account_id="acct-support-ticket-execute",
            user_id="user-support-ticket-execute",
        ),
    )
    route = _router_route(router, "/content-ops/execute", "POST")

    payload = await route.endpoint({
        "outputs": ["faq_markdown"],
        "limit": 3,
        "require_quality_gates": False,
        "inputs": {
            "source_material": _support_ticket_csv_rows(),
        },
    })

    assert payload["status"] == "completed"
    assert payload["plan"]["steps"][0]["runner"] == "TicketFAQMarkdownService.generate"

    step = payload["steps"][0]
    result = step["result"]

    assert step["output"] == "faq_markdown"
    assert step["status"] == "completed"
    assert result["source_count"] == 4
    assert result["ticket_source_count"] == 4
    assert result["generated"] == 2
    assert result["output_checks"] == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert "FAQ Report" in result["markdown"]
    assert result["items"][0]["source_ids"] == (
        "ticket-northstar-1",
        "ticket-northstar-2",
    )
