from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

from atlas_brain._content_ops_input_provider import build_content_ops_input_provider
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_ops_input_provider import (
    merge_content_ops_input_package,
)
from extracted_content_pipeline.control_surfaces import (
    preview_control_surface,
    request_from_mapping,
)


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
