from __future__ import annotations

import csv
import importlib
import importlib.util
from pathlib import Path
import sys

import pytest

import extracted_content_pipeline.api.control_surfaces as api_module
from atlas_brain._content_ops_input_provider import build_content_ops_input_provider
from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
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
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownService


ROOT = Path(__file__).resolve().parents[1]


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
        "uses_user_vocabulary": True,
    }
    assert result["saved_ids"] == []
    assert result["items"][0]["ticket_count"] == 1000
    assert len(result["items"][0]["source_ids"]) == 1000


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
    }
    assert "FAQ Report" in result["markdown"]
    assert result["items"][0]["source_ids"] == (
        "ticket-northstar-1",
        "ticket-northstar-2",
    )
