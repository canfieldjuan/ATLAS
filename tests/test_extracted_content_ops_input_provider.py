from __future__ import annotations

import pytest

from extracted_content_pipeline.content_ops_input_provider import (
    ContentOpsInputPackage,
    content_ops_payload_from_input_package,
    merge_content_ops_input_package,
)
from extracted_content_pipeline.control_surfaces import (
    preview_control_surface,
    request_from_mapping,
)
from extracted_content_pipeline.generation_plan import build_generation_plan


def _source_material() -> list[dict[str, str]]:
    return [
        {
            "source_id": "ticket-1",
            "source_type": "support_ticket",
            "source_title": "Report export missing",
            "text": "How do I export the dashboard before renewal?",
        }
    ]


def test_input_package_builds_existing_content_ops_request_payload() -> None:
    package = ContentOpsInputPackage(
        provider="support_ticket_csv",
        outputs=("faq_markdown", "landing_page", "blog_post"),
        inputs={
            "source_material": _source_material(),
            "faq_window_days": 90,
            "faq_source_types": ["support_ticket"],
            "topic": "Support ticket FAQ gaps customers keep asking about",
            "filters": {"topic_type": "content_ops_ticket_faq"},
            "campaign_name": "FAQ Report",
            "offer": "Turn repeat support tickets into customer-ready FAQ answers",
            "audience": "10-50 person SaaS teams",
            "target_keyword": "support ticket FAQ report",
            "faq_questions": [
                "What happens after I upload support tickets?",
                "Does FAQ Report publish automatically?",
            ],
            "cta_label": "Upload Ticket CSV -- Free Analysis",
            "cta_url": "/systems/ai-content-ops/intake",
        },
        metadata={"source": "upload"},
        warnings=({"code": "sampled_rows", "message": "Using first 1,000 rows."},),
    )

    payload = content_ops_payload_from_input_package(package)
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)
    plan = build_generation_plan(request)

    assert preview.can_run is True
    assert preview.missing_inputs == ()
    assert [step.output for step in plan.steps] == [
        "faq_markdown",
        "landing_page",
        "blog_post",
    ]
    assert request.inputs["source_material"] == _source_material()
    assert request.inputs["topic"] == "Support ticket FAQ gaps customers keep asking about"
    assert request.inputs["offer"] == "Turn repeat support tickets into customer-ready FAQ answers"
    assert payload["input_provider"] == {
        "provider": "support_ticket_csv",
        "metadata": {"source": "upload"},
        "warnings": [{"code": "sampled_rows", "message": "Using first 1,000 rows."}],
    }


def test_merge_input_package_keeps_explicit_request_inputs_authoritative() -> None:
    package = ContentOpsInputPackage(
        provider="ticket_import",
        outputs=("faq_markdown", "landing_page"),
        inputs={
            "source_material": _source_material(),
            "offer": "Provider offer",
            "audience": "Provider audience",
            "cta_url": "/provider-cta",
        },
    )
    payload = merge_content_ops_input_package(
        {
            "outputs": ["landing_page"],
            "target_mode": "custom_target",
            "ingestion_profile": "manual",
            "inputs": {
                "offer": "Operator offer",
                "cta_url": "/operator-cta",
            },
        },
        package,
    )

    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert payload["outputs"] == ["landing_page"]
    assert payload["target_mode"] == "custom_target"
    assert payload["ingestion_profile"] == "manual"
    assert payload["input_provider"] == {
        "provider": "ticket_import",
        "metadata": {},
        "warnings": [],
    }
    assert request.inputs["source_material"] == _source_material()
    assert request.inputs["audience"] == "Provider audience"
    assert request.inputs["offer"] == "Operator offer"
    assert request.inputs["cta_url"] == "/operator-cta"
    assert preview.can_run is True


def test_merge_input_package_preserves_explicit_account_usage_budget() -> None:
    package = ContentOpsInputPackage(
        provider="ticket_import",
        outputs=("landing_page",),
        inputs={
            "source_material": _source_material(),
            "offer": "Provider offer",
            "audience": "Provider audience",
        },
    )

    payload = merge_content_ops_input_package(
        {
            "outputs": ["landing_page"],
            "account_usage_budget_usd": 1.5,
            "account_usage_budget_days": 14,
            "inputs": {"offer": "Operator offer"},
        },
        package,
    )
    request = request_from_mapping(payload)

    assert payload["account_usage_budget_usd"] == 1.5
    assert payload["account_usage_budget_days"] == 14
    assert request.account_usage_budget_usd == 1.5
    assert request.account_usage_budget_days == 14


def test_merge_input_package_preserves_explicit_cache_policy() -> None:
    package = ContentOpsInputPackage(
        provider="ticket_import",
        outputs=("landing_page",),
        inputs={
            "source_material": _source_material(),
            "offer": "Provider offer",
            "audience": "Provider audience",
        },
    )

    payload = merge_content_ops_input_package(
        {
            "outputs": ["landing_page"],
            "content_ops_cache_policy": "exact-cache",
            "inputs": {"offer": "Operator offer"},
        },
        package,
    )
    request = request_from_mapping(payload)

    assert payload["content_ops_cache_policy"] == "exact-cache"
    assert request.content_ops_cache_policy == "exact"


def test_merge_input_package_ignores_null_request_overrides() -> None:
    package = ContentOpsInputPackage(
        provider="ticket_import",
        outputs=("faq_markdown",),
        ingestion_profile="existing_evidence",
        inputs={"source_material": _source_material()},
    )

    payload = merge_content_ops_input_package(
        {
            "outputs": None,
            "target_mode": None,
            "ingestion_profile": None,
            "inputs": None,
        },
        package,
    )

    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert payload["outputs"] == ["faq_markdown"]
    assert payload["target_mode"] == "vendor_retention"
    assert payload["ingestion_profile"] == "existing_evidence"
    assert request.inputs["source_material"] == _source_material()
    assert preview.can_run is True


def test_mapping_package_is_accepted_for_host_adapters() -> None:
    payload = content_ops_payload_from_input_package({
        "provider": "json_adapter",
        "outputs": "faq_markdown",
        "inputs": {"source_material": _source_material()},
    })

    request = request_from_mapping(payload)

    assert request.outputs == ("faq_markdown",)
    assert request.ingestion_profile == "existing_evidence"
    assert request.inputs["source_material"] == _source_material()


def test_input_package_rejects_non_object_inputs() -> None:
    with pytest.raises(TypeError, match="inputs must be a mapping"):
        content_ops_payload_from_input_package({
            "provider": "bad_adapter",
            "inputs": ["source_material"],
        })
