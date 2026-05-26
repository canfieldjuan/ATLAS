from __future__ import annotations

from extracted_content_pipeline.content_ops_input_provider import (
    content_ops_payload_from_input_package,
)
from extracted_content_pipeline.control_surfaces import (
    preview_control_surface,
    request_from_mapping,
)
from extracted_content_pipeline.generation_plan import build_generation_plan
from extracted_content_pipeline.support_ticket_input_package import (
    DEFAULT_FAQ_REPORT_CTA_LABEL,
    build_support_ticket_input_package,
)


def test_support_ticket_input_package_feeds_existing_content_ops_plan() -> None:
    package = build_support_ticket_input_package([
        {
            "Ticket ID": "ticket-1",
            "Account Name": "Acme Logistics",
            "Vendor Name": "HelpDeskPro",
            "Subject": "How do I change my login email?",
            "Description": "I cannot find where to update the email on my account.",
            "Pain Category": "profile updates",
            "Created At": "2026-05-01",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Export dashboard",
            "description": "Where do I export the dashboard before renewal?",
            "source_url": "https://example.test/tickets/2",
            "created_at": "2026-05-02",
        },
    ])

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
    assert request.inputs["faq_window_days"] == 90
    assert request.inputs["faq_source_types"] == ["support_ticket"]
    assert request.inputs["source_period"] == "Last 90 days of support tickets"
    assert request.inputs["has_dated_window"] is True
    assert request.inputs["cta_label"] == DEFAULT_FAQ_REPORT_CTA_LABEL
    assert request.inputs["topic"] == "Support-ticket questions customers keep asking"
    assert request.inputs["filters"] == {"topic_type": "content_ops_support_ticket_faq"}
    assert request.inputs["source_row_count"] == 2
    assert request.inputs["included_ticket_row_count"] == 2
    assert request.inputs["skipped_ticket_row_count"] == 0
    assert request.inputs["truncated_ticket_row_count"] == 0
    assert request.inputs["question_like_ticket_count"] == 2
    assert request.inputs["support_ticket_resolution_evidence_present"] is False
    assert request.inputs["support_ticket_resolution_evidence_count"] == 0
    assert request.inputs["support_ticket_resolution_examples"] == []
    assert request.inputs["has_measured_outcomes"] is False
    assert request.inputs["measured_outcome_count"] == 0
    assert request.inputs["measured_outcome_examples"] == []
    assert request.inputs["top_ticket_clusters"] == [
        {"label": "profile updates", "count": 1},
        {"label": "Export dashboard", "count": 1},
    ]
    assert request.inputs["customer_wording_examples"][0] == {
        "source_id": "ticket-1",
        "source_title": "How do I change my login email?",
        "pain_category": "profile updates",
        "text": (
            "How do I change my login email? I cannot find where to update the "
            "email on my account."
        ),
    }
    assert "support_ticket_source_summary" not in request.inputs
    assert request.inputs["faq_questions"] == [
        "How do I change my login email?",
        "Where do I export the dashboard before renewal?",
    ]
    assert request.inputs["source_material"][0] == {
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "How do I change my login email?",
        "text": (
            "How do I change my login email? I cannot find where to update the "
            "email on my account."
        ),
        "company_name": "Acme Logistics",
        "vendor_name": "HelpDeskPro",
        "pain_category": "profile updates",
        "created_at": "2026-05-01",
    }
    assert package.metadata["included_row_count"] == 2


def test_support_ticket_bundle_inherits_parent_fields_and_comment_text() -> None:
    package = build_support_ticket_input_package({
        "company": "Riverbend Supply",
        "vendor": "LegacyCRM",
        "support_tickets": [
            {
                "ticket_id": "support-riverbend-1",
                "subject": "Manual sequence cleanup after demos",
                "comments": [
                    {"author": "Customer", "body": "Can I automate demo follow-up?"},
                    {"role": "Agent", "message": "Automation is not on this plan."},
                ],
            },
        ],
    })

    rows = package.inputs["source_material"]

    assert rows == [
        {
            "source_id": "support-riverbend-1",
            "source_type": "support_ticket",
            "source_title": "Manual sequence cleanup after demos",
            "text": (
                "Manual sequence cleanup after demos Can I automate demo follow-up? "
                "Automation is not on this plan."
            ),
            "company_name": "Riverbend Supply",
            "vendor_name": "LegacyCRM",
        }
    ]
    assert package.inputs["faq_questions"] == ["Can I automate demo follow-up?"]
    assert package.inputs["support_ticket_resolution_evidence_present"] is False
    assert package.inputs["has_dated_window"] is False


def test_support_ticket_input_package_surfaces_explicit_resolution_evidence() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "How do I export reports?",
            "description": "Where do I export the dashboard?",
            "resolution": "Open Reports, choose Export, then select CSV.",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Where do I update billing?",
            "description": "I cannot find the billing page.",
        },
    ])

    assert package.inputs["support_ticket_resolution_evidence_present"] is True
    assert package.inputs["support_ticket_resolution_evidence_count"] == 1
    assert package.inputs["support_ticket_resolution_examples"] == [
        {
            "source_id": "ticket-1",
            "source_title": "How do I export reports?",
            "text": "Open Reports, choose Export, then select CSV.",
        }
    ]
    assert package.inputs["source_material"][0]["resolution_text"] == (
        "Open Reports, choose Export, then select CSV."
    )
    assert package.metadata["support_ticket_resolution_evidence_present"] is True
    assert package.metadata["support_ticket_resolution_evidence_count"] == 1


def test_support_ticket_input_package_surfaces_measured_outcome_evidence() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Do FAQs reduce repeat tickets?",
            "description": "Can we tell if the billing FAQ helped?",
            "measured_outcome": "Repeat billing tickets fell from 18 to 11 after publishing.",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Zero deflections",
            "description": "What happened after the trial FAQ update?",
            "deflection_rate": 0,
        },
    ])

    assert package.inputs["has_measured_outcomes"] is True
    assert package.inputs["measured_outcome_count"] == 2
    assert package.inputs["measured_outcome_examples"] == [
        {
            "source_id": "ticket-1",
            "source_title": "Do FAQs reduce repeat tickets?",
            "text": "Repeat billing tickets fell from 18 to 11 after publishing.",
        },
        {
            "source_id": "ticket-2",
            "source_title": "Zero deflections",
            "text": "0",
        },
    ]
    assert package.inputs["source_material"][1]["measured_outcome"] == "0"
    assert package.metadata["has_measured_outcomes"] is True
    assert package.metadata["measured_outcome_count"] == 2


def test_support_ticket_input_package_prefers_measured_outcome_value_over_label() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Did the billing FAQ help?",
            "description": "Can we see what changed after publishing?",
            "outcome_metric": "deflection_rate",
            "outcome_value": "42%",
        },
    ])

    assert package.inputs["has_measured_outcomes"] is True
    assert package.inputs["measured_outcome_count"] == 1
    assert package.inputs["measured_outcome_examples"] == [
        {
            "source_id": "ticket-1",
            "source_title": "Did the billing FAQ help?",
            "text": "42%",
        }
    ]
    assert package.inputs["source_material"][0]["measured_outcome"] == "42%"


def test_support_ticket_input_package_ignores_boolean_measured_outcome_flags() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Did the billing FAQ help?",
            "description": "Can we see what changed after publishing?",
            "outcome_value": False,
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Was this measured?",
            "description": "Did we track the result?",
            "measured_outcome": True,
        },
    ])

    assert package.inputs["has_measured_outcomes"] is False
    assert package.inputs["measured_outcome_count"] == 0
    assert package.inputs["measured_outcome_examples"] == []
    assert "measured_outcome" not in package.inputs["source_material"][0]
    assert "measured_outcome" not in package.inputs["source_material"][1]


def test_support_ticket_input_package_derives_faq_source_types_from_rows() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "source_type": "ticket",
            "subject": "Login email change",
            "message": "How do I change my email address?",
        },
        {
            "ticket_id": "ticket-2",
            "source_type": "support_ticket",
            "subject": "Export dashboard",
            "message": "Where do I export the dashboard?",
        },
    ])

    assert package.inputs["faq_source_types"] == ["ticket", "support_ticket"]


def test_support_ticket_clusters_do_not_use_synthetic_ticket_ids() -> None:
    package = build_support_ticket_input_package([
        {"description": "How do I export data?"},
        {"description": "Where is the billing page?"},
        {"description": "Can I change my plan?"},
    ])

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "uncategorized", "count": 3}
    ]
    assert package.inputs["customer_wording_examples"][0] == {
        "source_id": "ticket-1",
        "text": "How do I export data?",
    }


def test_support_ticket_clusters_include_remaining_bucket() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": f"ticket-{index}",
            "description": f"How do I fix issue {index}?",
            "pain_category": f"category-{index}",
        }
        for index in range(1, 9)
    ])

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "category-1", "count": 1},
        {"label": "category-2", "count": 1},
        {"label": "category-3", "count": 1},
        {"label": "category-4", "count": 1},
        {"label": "category-5", "count": 1},
        {"label": "category-6", "count": 1},
        {"label": "remaining", "count": 2},
    ]


def test_support_ticket_input_package_omits_window_filter_without_row_dates() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Login email change",
            "message": "How do I change my email address?",
        }
    ])

    assert "faq_window_days" not in package.inputs
    assert package.inputs["source_period"] == "Uploaded support tickets"
    assert package.inputs["has_dated_window"] is False


def test_support_ticket_input_package_omits_window_filter_without_parseable_row_dates() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Login email change",
            "message": "How do I change my email address?",
            "created_at": "last week",
        }
    ])

    assert "faq_window_days" not in package.inputs
    assert package.inputs["source_period"] == "Uploaded support tickets"
    assert package.inputs["has_dated_window"] is False


def test_support_ticket_input_package_omits_window_filter_for_mixed_date_rows() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Login email change",
            "message": "How do I change my email address?",
            "created_at": "2026-05-02T12:00:00Z",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Export dashboard",
            "message": "Where do I export the dashboard?",
        },
    ])

    assert "faq_window_days" not in package.inputs
    assert package.inputs["has_dated_window"] is False


def test_support_ticket_input_package_accepts_single_mapping_comment_thread() -> None:
    package = build_support_ticket_input_package({
        "support_tickets": [
            {
                "ticket_id": "ticket-1",
                "subject": "Billing setup",
                "comments": {"body": "Where do I update billing details?"},
            },
        ],
    })

    assert package.inputs["source_material"][0]["text"] == (
        "Billing setup Where do I update billing details?"
    )
    assert package.inputs["faq_questions"] == ["Where do I update billing details?"]


def test_support_ticket_input_package_accepts_single_string_output() -> None:
    package = build_support_ticket_input_package(
        [{"ticket_id": "ticket-1", "subject": "How do I export reports?"}],
        outputs="landing_page",
    )
    payload = content_ops_payload_from_input_package(package)
    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert request.outputs == ("landing_page",)
    assert preview.can_run is True


def test_support_ticket_input_package_uses_stable_duplicate_key_precedence() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "How do I export reports?",
            "Account Name": "First Account",
            "account_name": "Second Account",
            "company": "Company Fallback",
        },
    ])

    assert package.inputs["source_material"][0]["company_name"] == "First Account"


def test_support_ticket_input_package_keeps_request_overrides_authoritative() -> None:
    package = build_support_ticket_input_package(
        [{"ticket_id": "ticket-1", "subject": "How do I export reports?"}],
        audience="Provider audience",
        offer="Provider offer",
    )
    payload = content_ops_payload_from_input_package(
        package,
        request_payload={
            "outputs": ["landing_page"],
            "inputs": {
                "audience": "Operator audience",
                "offer": "Operator offer",
                "cta_url": "/operator",
            },
        },
    )

    request = request_from_mapping(payload)
    preview = preview_control_surface(request)

    assert request.outputs == ("landing_page",)
    assert request.inputs["audience"] == "Operator audience"
    assert request.inputs["offer"] == "Operator offer"
    assert request.inputs["cta_url"] == "/operator"
    assert request.inputs["source_material"][0]["source_id"] == "ticket-1"
    assert preview.can_run is True


def test_support_ticket_input_package_reports_skipped_and_truncated_rows() -> None:
    package = build_support_ticket_input_package(
        [
            {"ticket_id": "ticket-1"},
            {"ticket_id": "ticket-2", "description": "How do I export data?"},
            {"ticket_id": "ticket-3", "description": "Where is the billing page?"},
        ],
        max_rows=2,
    )

    assert package.inputs["source_material"] == [
        {
            "source_id": "ticket-2",
            "source_type": "support_ticket",
            "source_title": "ticket-2",
            "text": "How do I export data?",
        }
    ]
    assert package.metadata["source_row_count"] == 3
    assert package.metadata["included_row_count"] == 1
    assert package.metadata["skipped_row_count"] == 1
    assert package.metadata["truncated_row_count"] == 1
    assert package.inputs["source_row_count"] == 3
    assert package.inputs["included_ticket_row_count"] == 1
    assert package.inputs["skipped_ticket_row_count"] == 1
    assert package.inputs["truncated_ticket_row_count"] == 1
    assert package.warnings == (
        {
            "code": "ticket_row_missing_text",
            "row_index": 1,
            "message": "Skipped ticket row because it did not include customer wording.",
        },
        {
            "code": "ticket_rows_truncated",
            "message": "Used first 2 ticket rows out of 3.",
            "row_count": 3,
            "max_rows": 2,
            "truncated_row_count": 1,
        },
    )


def test_support_ticket_input_package_reconciles_truncated_valid_row_counts() -> None:
    package = build_support_ticket_input_package(
        [
            {"ticket_id": "ticket-1", "description": "How do I export data?"},
            {"ticket_id": "ticket-2", "description": "Where is the billing page?"},
            {"ticket_id": "ticket-3", "description": "Can I change my plan?"},
            {"ticket_id": "ticket-4", "description": "Why was I charged?"},
        ],
        max_rows=2,
    )

    assert package.metadata["source_row_count"] == 4
    assert package.metadata["included_row_count"] == 2
    assert package.metadata["skipped_row_count"] == 0
    assert package.metadata["truncated_row_count"] == 2
    assert package.inputs["source_row_count"] == 4
    assert package.inputs["included_ticket_row_count"] == 2
    assert package.inputs["skipped_ticket_row_count"] == 0
    assert package.inputs["truncated_ticket_row_count"] == 2
    assert (
        package.metadata["included_row_count"]
        + package.metadata["skipped_row_count"]
        + package.metadata["truncated_row_count"]
        == package.metadata["source_row_count"]
    )
    assert package.warnings == (
        {
            "code": "ticket_rows_truncated",
            "message": "Used first 2 ticket rows out of 4.",
            "row_count": 4,
            "max_rows": 2,
            "truncated_row_count": 2,
        },
    )


def test_support_ticket_input_package_caps_default_generation_rows_at_1000() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": f"ticket-{index}",
            "description": f"How do I fix issue {index}?",
            "pain_category": "setup friction",
        }
        for index in range(1, 1006)
    ])

    assert package.inputs["source_row_count"] == 1005
    assert package.inputs["included_ticket_row_count"] == 1000
    assert package.inputs["skipped_ticket_row_count"] == 0
    assert package.inputs["truncated_ticket_row_count"] == 5
    assert len(package.inputs["source_material"]) == 1000
    assert package.metadata["source_row_count"] == 1005
    assert package.metadata["included_row_count"] == 1000
    assert package.metadata["skipped_row_count"] == 0
    assert package.metadata["truncated_row_count"] == 5
    assert package.warnings == (
        {
            "code": "ticket_rows_truncated",
            "message": "Used first 1000 ticket rows out of 1005.",
            "row_count": 1005,
            "max_rows": 1000,
            "truncated_row_count": 5,
        },
    )


def test_support_ticket_input_package_reports_non_mapping_rows() -> None:
    package = build_support_ticket_input_package([
        "bare string",
        42,
        {"ticket_id": "ticket-1", "description": "How do I export data?"},
    ])

    assert package.metadata["source_row_count"] == 3
    assert package.metadata["included_row_count"] == 1
    assert package.metadata["skipped_row_count"] == 2
    assert package.warnings == (
        {
            "code": "ticket_row_not_object",
            "row_index": 1,
            "message": "Skipped ticket row because it was not an object.",
        },
        {
            "code": "ticket_row_not_object",
            "row_index": 2,
            "message": "Skipped ticket row because it was not an object.",
        },
    )


def test_support_ticket_input_package_reports_empty_source_material() -> None:
    package = build_support_ticket_input_package(None)

    assert package.metadata["source_row_count"] == 0
    assert package.metadata["included_row_count"] == 0
    assert package.metadata["skipped_row_count"] == 0
    assert package.metadata["truncated_row_count"] == 0
    assert package.warnings == (
        {
            "code": "source_material_empty",
            "message": "No support-ticket source rows were provided.",
        },
    )


def test_support_ticket_input_package_rejects_invalid_window_and_row_limit() -> None:
    try:
        build_support_ticket_input_package([], window_days=0)
    except ValueError as exc:
        assert str(exc) == "window_days must be at least 1"
    else:
        raise AssertionError("expected window_days validation error")

    try:
        build_support_ticket_input_package([], max_rows=0)
    except ValueError as exc:
        assert str(exc) == "max_rows must be at least 1"
    else:
        raise AssertionError("expected max_rows validation error")
