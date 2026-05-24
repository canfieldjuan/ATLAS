from __future__ import annotations

import pytest

from extracted_content_pipeline.api.control_surfaces import (
    ContentOpsControlSurfaceApiConfig,
    create_content_ops_control_surface_router,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.control_surfaces import (
    preview_control_surface,
    request_from_mapping,
)
from extracted_content_pipeline.content_ops_execution import ContentOpsExecutionServices
from extracted_content_pipeline.content_ops_input_provider import (
    content_ops_payload_from_input_package,
)
from extracted_content_pipeline.support_ticket_input_provider import (
    SupportTicketInputProvider,
)
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownService


def _ticket_rows() -> list[dict[str, str]]:
    return [
        {
            "ticket_id": "ticket-1",
            "subject": "How do I change my login email?",
            "description": "I cannot find the account email setting.",
        }
    ]


def _route(router, path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


def test_support_ticket_input_provider_builds_package_from_direct_source_material() -> None:
    provider = SupportTicketInputProvider(
        source_material=_ticket_rows(),
        outputs="landing_page",
        window_days=30,
        metadata={"host_request_id": "req-1"},
    )

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-1"),
        request={"inputs": {"offer": "Operator offer"}},
    )
    payload = content_ops_payload_from_input_package(package)
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert package.provider == "support_ticket_input_provider"
    assert package.metadata["source_row_count"] == 1
    assert package.metadata["host_request_id"] == "req-1"
    assert request.outputs == ("landing_page",)
    assert request.inputs["source_period"] == "Uploaded support tickets"
    assert "faq_window_days" not in request.inputs
    assert request.inputs["faq_questions"] == ["How do I change my login email?"]
    assert preview.can_run is True


def test_support_ticket_input_provider_keeps_computed_metadata_authoritative() -> None:
    provider = SupportTicketInputProvider(
        source_material=_ticket_rows(),
        metadata={
            "host_request_id": "req-1",
            "source_row_count": 999,
            "included_row_count": 999,
        },
    )

    package = provider.build_content_ops_input_package(scope=TenantScope(account_id="acct-1"))

    assert package.metadata["host_request_id"] == "req-1"
    assert package.metadata["source_row_count"] == 1
    assert package.metadata["included_row_count"] == 1


def test_support_ticket_input_provider_uses_sync_loader_with_scope_and_request() -> None:
    calls: list[dict[str, object]] = []

    def loader(scope: TenantScope, request):
        calls.append({"scope": scope, "request": request})
        return {
            "company": scope.account_id,
            "support_tickets": _ticket_rows(),
        }

    provider = SupportTicketInputProvider(source_material_loader=loader)

    package = provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-sync"),
        request={"outputs": ["faq_markdown"]},
    )

    assert calls[0]["scope"] == TenantScope(account_id="acct-sync")
    assert calls[0]["request"] == {"outputs": ["faq_markdown"]}
    assert package.inputs["source_material"][0]["company_name"] == "acct-sync"


@pytest.mark.asyncio
async def test_support_ticket_input_provider_uses_async_loader() -> None:
    async def loader(scope: TenantScope, request):
        assert scope.account_id == "acct-async"
        assert request == {"outputs": ["landing_page"]}
        return _ticket_rows()

    provider = SupportTicketInputProvider(source_material_loader=loader)

    package = await provider.build_content_ops_input_package(
        scope=TenantScope(account_id="acct-async"),
        request={"outputs": ["landing_page"]},
    )

    assert package.inputs["source_material"][0]["source_id"] == "ticket-1"


def test_support_ticket_input_provider_rejects_direct_source_and_loader() -> None:
    def loader(scope: TenantScope, request):
        del scope, request
        return _ticket_rows()

    with pytest.raises(
        ValueError,
        match="provide either source_material or source_material_loader",
    ):
        SupportTicketInputProvider(source_material=_ticket_rows(), source_material_loader=loader)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_days": 0}, "window_days must be at least 1"),
        ({"max_rows": 0}, "max_rows must be at least 1"),
    ],
)
def test_support_ticket_input_provider_rejects_invalid_numeric_config(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SupportTicketInputProvider(source_material=_ticket_rows(), **kwargs)


def test_support_ticket_input_provider_without_source_reports_empty_material() -> None:
    provider = SupportTicketInputProvider()

    package = provider.build_content_ops_input_package(scope=TenantScope(account_id="acct-1"))

    assert package.metadata["source_row_count"] == 0
    assert package.warnings == (
        {
            "code": "source_material_empty",
            "message": "No support-ticket source rows were provided.",
        },
    )


@pytest.mark.asyncio
async def test_support_ticket_input_provider_wires_into_preview_route() -> None:
    def loader(scope: TenantScope, request):
        assert scope.account_id == "acct-route"
        assert request == {
            "outputs": ("landing_page",),
            "inputs": {
                "audience": "Operator audience",
                "offer": "Operator offer",
            },
        }
        return _ticket_rows()

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        input_provider=SupportTicketInputProvider(source_material_loader=loader),
        scope_provider=lambda: {"account_id": "acct-route"},
    )

    route = _route(router, "/ops/preview", "POST")
    payload = await route.endpoint({
        "outputs": ["landing_page"],
        "inputs": {
            "audience": "Operator audience",
            "offer": "Operator offer",
        },
    })

    assert payload["can_run"] is True
    assert payload["outputs"] == ["landing_page"]
    assert payload["missing_inputs"] == []


@pytest.mark.asyncio
async def test_support_ticket_input_provider_wires_into_plan_route() -> None:
    calls: list[dict[str, object]] = []

    def loader(scope: TenantScope, request):
        calls.append({"scope": scope, "request": request})
        return _ticket_rows()

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        input_provider=SupportTicketInputProvider(source_material_loader=loader),
        scope_provider=lambda: {"account_id": "acct-plan"},
    )

    route = _route(router, "/ops/plan", "POST")
    payload = await route.endpoint({"outputs": ["faq_markdown"]})

    assert payload["can_execute"] is True
    assert payload["steps"][0]["output"] == "faq_markdown"
    assert payload["steps"][0]["runner"] == "TicketFAQMarkdownService.generate"
    assert payload["preview"]["can_run"] is True
    assert calls[0]["scope"] == TenantScope(account_id="acct-plan")


@pytest.mark.asyncio
async def test_support_ticket_input_provider_wires_into_execute_route() -> None:
    def loader(scope: TenantScope, request):
        assert scope.account_id == "acct-execute"
        assert request == {"outputs": ("faq_markdown",)}
        return _ticket_rows()

    router = create_content_ops_control_surface_router(
        config=ContentOpsControlSurfaceApiConfig(prefix="/ops", tags=("ops",)),
        execution_services_provider=lambda: ContentOpsExecutionServices(
            faq_markdown=TicketFAQMarkdownService(),
        ),
        input_provider=SupportTicketInputProvider(source_material_loader=loader),
        scope_provider=lambda: {"account_id": "acct-execute"},
    )

    route = _route(router, "/ops/execute", "POST")
    payload = await route.endpoint({"outputs": ["faq_markdown"]})

    assert payload["status"] == "completed"
    assert payload["steps"][0]["output"] == "faq_markdown"
    assert payload["steps"][0]["status"] == "completed"
    assert payload["steps"][0]["result"]["generated"] == 1
    assert "How do I change my login email?" in payload["steps"][0]["result"]["markdown"]
