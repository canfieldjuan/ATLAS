from __future__ import annotations

import json
from pathlib import Path

import pytest

from extracted_content_pipeline.content_ops_input_provider import (
    content_ops_payload_from_input_package,
)
from extracted_content_pipeline.control_surfaces import (
    preview_control_surface,
    request_from_mapping,
)
from extracted_content_pipeline.generation_plan import build_generation_plan
from extracted_content_pipeline.support_ticket_clustering import (
    _PHRASE_FOLDS,
    _TOKEN_FOLDS,
    support_ticket_plain_text,
    support_ticket_tokens,
)
from extracted_content_pipeline.support_ticket_dates import (
    parse_support_ticket_source_date,
)
from extracted_content_pipeline.support_ticket_input_package import (
    DEFAULT_FAQ_REPORT_CTA_LABEL,
    DEFAULT_FAQ_REPRESENTATIVE_TAXONOMY_TERMS,
    build_support_ticket_input_package,
)
from extracted_content_pipeline.support_ticket_zendesk_thread import (
    load_zendesk_full_thread_rows_from_json_bytes,
    load_zendesk_full_thread_rows_from_json_file,
    rows_from_zendesk_full_thread,
)


ROOT = Path(__file__).resolve().parents[1]
ZENDESK_THREAD_SAMPLE = ROOT / "tests/fixtures/zendesk_full_thread_seed_sample.json"


def test_support_ticket_fold_targets_survive_tokenization() -> None:
    token_fold_targets = {
        target for target in _TOKEN_FOLDS.values() if target and len(target) >= 2
    }
    phrase_fold_targets = {replacement for _pattern, replacement in _PHRASE_FOLDS}

    for target in sorted(token_fold_targets | phrase_fold_targets):
        assert target in support_ticket_tokens(f"customer mentioned {target} issue"), target


def test_support_ticket_input_package_feeds_existing_content_ops_plan() -> None:
    package = build_support_ticket_input_package([
        {
            "Ticket ID": "ticket-1",
            "Account Name": "Acme Logistics",
            "Vendor Name": "HelpDeskPro",
            "Subject": "How do I change my login email?",
            "Description": "I cannot find where to update the email on my account.",
            "Pain Category": "profile updates",
            "Product": "Account settings",
            "Issue": "Email profile updates",
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
    assert request.inputs["faq_representative_taxonomy_terms"] == list(
        DEFAULT_FAQ_REPRESENTATIVE_TAXONOMY_TERMS
    )
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
        {"label": "dashboard export renewal", "count": 1},
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
        "product": "Account settings",
        "issue": "Email profile updates",
        "created_at": "2026-05-01",
        "support_ticket_cluster": "profile updates",
        "support_ticket_cluster_key": "explicit:profile-updates",
        "support_ticket_cluster_source": "explicit",
        "support_ticket_evidence_tier": "csv_customer_text",
    }
    assert package.metadata["included_row_count"] == 2
    assert package.metadata["top_ticket_clusters"] == request.inputs["top_ticket_clusters"]
    assert package.metadata["cluster_quality"] == {
        "clustered_row_count": 2,
        "uncategorized_row_count": 0,
        "cluster_count": 2,
        "singleton_cluster_count": 2,
        "largest_cluster_count": 1,
    }



@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("2026-05-01", "2026-05-01"),
        ("2026-05-01T12:00:00Z", "2026-05-01"),
        ("05/01/2026", "2026-05-01"),
        ("5/1/2026", "2026-05-01"),
        ("05/01/26", "2026-05-01"),
        ("05-01-2026", "2026-05-01"),
    ),
)
def test_support_ticket_source_date_parser_accepts_us_export_dates(
    raw: str,
    expected: str,
) -> None:
    parsed = parse_support_ticket_source_date(raw)

    assert parsed is not None
    assert parsed.isoformat() == expected


def test_support_ticket_source_date_parser_rejects_natural_language() -> None:
    assert parse_support_ticket_source_date("last week") is None


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
            "organization": "Riverbend Supply",
            "company_name": "Riverbend Supply",
            "vendor_name": "LegacyCRM",
            "support_ticket_cluster": "automation cleanup demo follow",
            "support_ticket_cluster_key": "tokens:automation-cleanup-demo-follow",
            "support_ticket_cluster_source": "token_set",
            "support_ticket_evidence_tier": "csv_customer_text",
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


def test_support_ticket_input_package_rejects_generic_response_metadata_as_resolution() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "How do I reset MFA?",
            "description": "The customer cannot access MFA settings.",
            "first_response": "Thanks for contacting support; we received your ticket.",
        },
        {
            "ticket_id": "ticket-2",
            "subject": "Where is the export button?",
            "description": "The customer needs the report export button.",
            "last_response": "First response SLA met in 22 minutes.",
        },
        {
            "ticket_id": "ticket-3",
            "subject": "Can I update billing contacts?",
            "description": "The customer asks where billing contacts live.",
            "reply_text": "Auto-ack sent by routing workflow.",
        },
    ])

    assert package.inputs["support_ticket_resolution_evidence_present"] is False
    assert package.inputs["support_ticket_resolution_evidence_count"] == 0
    assert package.inputs["support_ticket_resolution_examples"] == []
    assert all("resolution_text" not in row for row in package.inputs["source_material"])
    assert package.metadata["support_ticket_resolution_evidence_present"] is False
    assert package.metadata["support_ticket_resolution_evidence_count"] == 0


def test_support_ticket_input_package_counts_faq_output_resolution_evidence() -> None:
    package = build_support_ticket_input_package({
        "generated": 2,
        "markdown": "# FAQ Report",
        "saved_ids": ["faq-draft-1"],
        "items": [
            {
                "topic": "billing confusion",
                "question": "Why was I charged twice?",
                "summary": "Customers ask why duplicate-looking invoices appear.",
                "steps": [
                    "Check whether the second charge is a pending authorization.",
                    "Confirm the invoice date and subscription workspace.",
                ],
                "answer_evidence_status": "resolution_evidence",
                "source_ids": ["ticket-1", "ticket-2"],
            },
            {
                "topic": "export setup",
                "question": "How do I export the report?",
                "summary": "Customers ask where report exports live.",
                "steps": ["Draft answer - support team should add the verified resolution."],
                "answer_evidence_status": "draft_needs_review",
                "source_ids": ["ticket-3"],
            },
        ],
    })

    assert package.inputs["source_material"][0]["source_type"] == "faq_output"
    assert package.inputs["source_material"][0]["source_id"] == "faq-draft-1:item-1"
    assert package.inputs["source_material"][0]["resolution_text"] == (
        "Check whether the second charge is a pending authorization. "
        "Confirm the invoice date and subscription workspace."
    )
    assert "resolution_text" not in package.inputs["source_material"][1]
    assert package.inputs["support_ticket_resolution_evidence_present"] is True
    assert package.inputs["support_ticket_resolution_evidence_count"] == 1
    assert package.inputs["support_ticket_resolution_examples"] == [
        {
            "source_id": "faq-draft-1:item-1",
            "source_title": "Why was I charged twice?",
            "text": (
                "Check whether the second charge is a pending authorization. "
                "Confirm the invoice date and subscription workspace."
            ),
        }
    ]
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


def test_support_ticket_input_package_accepts_common_platform_csv_shapes() -> None:
    package = build_support_ticket_input_package([
        {
            "Id": "zd-100",
            "Subject": "How do I reset MFA?",
            "Description": "I cannot get the login code on my new phone.",
            "Created at": "05/01/2026",
            "Requester email": "maya@example.test",
        },
        {
            "Ticket ID": "fd-200",
            "Ticket Subject": "Where do I update billing?",
            "Ticket Description": "Why was I charged twice this month?",
            "Created time": "5/2/2026",
            "Contact email address": "ops@example.test",
        },
        {
            "Conversation ID": "ic-300",
            "Conversation title": "Cancellation before renewal",
            "Conversation body": "How do I cancel my account before it renews?",
            "Conversation created at": "05-03-26",
            "User email": "founder@example.test",
        },
    ])

    rows = package.inputs["source_material"]

    assert package.inputs["has_dated_window"] is True
    assert package.inputs["source_period"] == "Last 90 days of support tickets"
    assert package.inputs["faq_questions"] == [
        "How do I reset MFA?",
        "Where do I update billing?",
        "How do I cancel my account before it renews?",
    ]
    assert rows[0] == {
        "source_id": "zd-100",
        "source_type": "support_ticket",
        "source_title": "How do I reset MFA?",
        "text": (
            "How do I reset MFA? I cannot get the login code on my new phone."
        ),
        "created_at": "05/01/2026",
        "contact_email": "maya@example.test",
        "support_ticket_cluster": "code login mfa new",
        "support_ticket_cluster_key": "tokens:code-login-mfa-new",
        "support_ticket_cluster_source": "token_set",
        "support_ticket_evidence_tier": "csv_customer_text",
    }
    assert rows[1]["source_id"] == "fd-200"
    assert rows[1]["source_title"] == "Where do I update billing?"
    assert rows[1]["text"] == (
        "Where do I update billing? Why was I charged twice this month?"
    )
    assert rows[1]["created_at"] == "5/2/2026"
    assert rows[1]["contact_email"] == "ops@example.test"
    assert rows[2]["source_id"] == "ic-300"
    assert rows[2]["source_title"] == "Cancellation before renewal"
    assert rows[2]["text"] == (
        "Cancellation before renewal How do I cancel my account before it renews?"
    )
    assert rows[2]["created_at"] == "05-03-26"
    assert rows[2]["contact_email"] == "founder@example.test"


def test_support_ticket_input_package_keeps_customer_message_before_latest_reply() -> None:
    package = build_support_ticket_input_package([
        {
            "Ticket ID": "fd-201",
            "Ticket Subject": "Report export",
            "Message": "Where do I export my monthly report?",
            "Latest message": "Agent reply: use the Export button.",
        },
    ])

    assert package.inputs["source_material"][0]["text"] == (
        "Report export Where do I export my monthly report?"
    )
    assert package.inputs["faq_questions"] == [
        "Where do I export my monthly report?"
    ]


def test_support_ticket_clusters_do_not_use_synthetic_ticket_ids() -> None:
    package = build_support_ticket_input_package([
        {"description": "How do I export data?"},
        {"description": "Where is the billing page?"},
        {"description": "Can I change my plan?"},
    ])

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "data export", "count": 1},
        {"label": "billing", "count": 1},
        {"label": "plan update", "count": 1},
    ]
    assert all(
        cluster["label"] not in {"ticket-1", "ticket-2", "ticket-3"}
        for cluster in package.inputs["top_ticket_clusters"]
    )
    assert package.inputs["customer_wording_examples"][0] == {
        "source_id": "ticket-1",
        "text": "How do I export data?",
    }


def test_support_ticket_clusters_group_messy_untagged_export_rows() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "zd-1",
            "subject": "Password reset help",
            "description": "<p>How do I reset my password?</p>",
        },
        {
            "ticket_id": "zd-2",
            "subject": "Password reset not working",
            "description": "I cannot reset password from the login screen",
        },
        {
            "ticket_id": "hs-1",
            "subject": "Change email address",
            "description": "Where do I update my email?",
        },
        {
            "ticket_id": "hs-2",
            "subject": "Update account email",
            "description": "Need to change email address",
        },
    ])

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "login password reset", "count": 2},
        {"label": "email update", "count": 2},
    ]
    assert package.metadata["cluster_quality"] == {
        "clustered_row_count": 4,
        "uncategorized_row_count": 0,
        "cluster_count": 2,
        "singleton_cluster_count": 0,
        "largest_cluster_count": 2,
    }
    assert package.inputs["source_material"][0]["text"] == (
        "Password reset help How do I reset my password?"
    )
    assert package.inputs["source_material"][0]["support_ticket_cluster"] == (
        "login password reset"
    )
    assert "<p>" not in package.inputs["customer_wording_examples"][0]["text"]


def test_support_ticket_input_package_strips_generic_provider_html_before_clustering() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "html-1",
            "subject": "<section>Reset password</section>",
            "description": (
                "<article><h2>How do I reset my password?</h2>"
                "<custom-widget>Use the login screen &amp; email link.</custom-widget>"
                "<script>ignore me</script></article>"
            ),
            "comments": [
                {
                    "body": (
                        "<message-fragment>Customer still sees the reset error.</message-fragment>"
                    )
                }
            ],
            "resolution_text": (
                "<answer-card>Open Login, choose Forgot password, then use the "
                "emailed reset link.</answer-card>"
            ),
        },
        {
            "ticket_id": "plain-1",
            "subject": "Export threshold comparison",
            "description": (
                "Why does the export fail when total < 10? "
                "Keep the literal &lt;manual_review&gt; marker in the note."
            ),
        },
        {
            "ticket_id": "xml-1",
            "subject": "API field text",
            "description": (
                "Why did <email>user@example.test</email> fail when "
                "<config>retries=3</config> was set?"
            ),
        },
    ])

    rows = package.inputs["source_material"]

    assert rows[0]["text"] == (
        "Reset password How do I reset my password? Use the login screen & "
        "email link. Customer still sees the reset error."
    )
    assert rows[0]["resolution_text"] == (
        "Open Login, choose Forgot password, then use the emailed reset link."
    )
    assert "section" not in rows[0]["support_ticket_cluster"]
    assert "article" not in rows[0]["support_ticket_cluster"]
    assert "custom" not in rows[0]["support_ticket_cluster"]
    assert rows[1]["text"] == (
        "Export threshold comparison Why does the export fail when total < 10? "
        "Keep the literal <manual_review> marker in the note."
    )
    assert rows[2]["text"] == (
        "API field text Why did <email>user@example.test</email> fail when "
        "<config>retries=3</config> was set?"
    )
    assert "<" not in rows[0]["text"]
    assert "<manual_review>" in rows[1]["text"]
    assert "<email>user@example.test</email>" in rows[2]["text"]
    assert "<config>retries=3</config>" in rows[2]["text"]
    assert "ignore me" not in rows[0]["text"]
    assert "answer-card" not in rows[0]["resolution_text"]
    assert "<" not in rows[0]["resolution_text"]
    assert package.inputs["customer_wording_examples"][0]["text"] == rows[0]["text"]
    assert package.inputs["faq_questions"][0] == "How do I reset my password?"


def test_support_ticket_inline_provider_html_strips_link_attribute_tokens() -> None:
    inline_html = (
        'Please <a href="https://ex.com/reset?token=abc123xyz" '
        'data-tracking-id="track-77">click here</a> to reset and '
        "<b>confirm</b> your <u>email</u> <small>Use step</small> "
        "<sup>2</sup><img src=\"https://cdn.example.test/screenshot.png\">"
    )

    assert support_ticket_plain_text(inline_html) == (
        "Please click here to reset and confirm your email Use step 2"
    )
    tokens = support_ticket_tokens(inline_html)
    assert {"click", "reset", "confirm", "email", "step"}.issubset(tokens)
    assert tokens.isdisjoint(
        {
            "href",
            "https",
            "ex",
            "com",
            "token",
            "abc123xyz",
            "data",
            "tracking",
            "track",
            "77",
            "src",
            "cdn",
            "example",
            "test",
            "screenshot",
            "png",
        }
    )
    assert support_ticket_plain_text(
        "Why did <email>user@example.test</email> fail when "
        "<config>retries=3</config> was set?"
    ) == (
        "Why did <email>user@example.test</email> fail when "
        "<config>retries=3</config> was set?"
    )


@pytest.mark.parametrize(
    ("text", "expected", "rejected"),
    (
        ("Atlas seed", {"atlas", "seed"}, {"atla"}),
        ("ticket status", {"status"}, {"statu"}),
        ("analysis report", {"analysis", "report"}, {"analysi"}),
        ("damaged devices", {"damaged", "device"}, {"devices"}),
    ),
)
def test_support_ticket_tokens_do_not_depluralize_non_plural_final_s_words(
    text: str,
    expected: set[str],
    rejected: set[str],
) -> None:
    tokens = support_ticket_tokens(text)

    assert expected.issubset(tokens)
    assert tokens.isdisjoint(rejected)
    assert support_ticket_plain_text("If a<b and c>d then fail") == (
        "If a<b and c>d then fail"
    )


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('Please <a href="https://example.test/reset">reset link</a>', "Please reset link"),
        ('Use <b class="agent-note">billing portal</b>', "Use billing portal"),
        ("Use <b>billing portal</b>", "Use billing portal"),
        ("Use <u>account email</u>", "Use account email"),
        ("Read the <small>agent note</small>", "Read the agent note"),
        ("Step <sub>2</sub> is missing", "Step 2 is missing"),
        ("Error <sup>TM</sup> appears", "Error TM appears"),
        ('Screenshot <img src="https://cdn.example.test/a.png"> attached', "Screenshot attached"),
        ("Old <s>legacy answer</s>", "Old legacy answer"),
        ("Old <strike>legacy answer</strike>", "Old legacy answer"),
    ],
)
def test_support_ticket_common_inline_provider_tags_are_html_signals(
    html: str,
    expected: str,
) -> None:
    assert support_ticket_plain_text(html) == expected


def test_support_ticket_clusters_group_topic_varied_anchor_rows() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "export-1",
            "subject": "Attribution export blocked",
            "description": "The attribution report export never starts.",
        },
        {
            "ticket_id": "export-2",
            "subject": "CSV export missing column",
            "description": "The downloaded CSV leaves out the campaign column.",
        },
        {
            "ticket_id": "export-3",
            "subject": "Export timeout",
            "description": "Report download times out before the file is ready.",
        },
        {
            "ticket_id": "sso-1",
            "subject": "SSO login loop",
            "description": "Users return to the SSO screen after Okta.",
        },
        {
            "ticket_id": "sso-2",
            "subject": "SAML single sign on failure",
            "description": "The identity provider rejects the SAML response.",
        },
        {
            "ticket_id": "billing-1",
            "subject": "Billing amount looks wrong",
            "description": "The billing total changed after renewal.",
        },
        {
            "ticket_id": "billing-2",
            "subject": "Card charged twice this month",
            "description": "The card was charged twice for the same workspace.",
        },
    ])

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "export", "count": 3},
        {"label": "sso", "count": 2},
        {"label": "billing", "count": 2},
    ]
    assert package.metadata["cluster_quality"] == {
        "clustered_row_count": 7,
        "uncategorized_row_count": 0,
        "cluster_count": 3,
        "singleton_cluster_count": 0,
        "largest_cluster_count": 3,
    }
    assert {
        row["support_ticket_cluster"]
        for row in package.inputs["source_material"]
    } == {"export", "sso", "billing"}
    assert all(
        row["support_ticket_cluster_source"] == "token_anchor"
        for row in package.inputs["source_material"]
    )


def test_support_ticket_clusters_group_login_access_synonyms_without_shared_raw_token() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "login-syn-1",
            "subject": "Locked out after MFA reset",
            "description": "The app refuses my session.",
        },
        {
            "ticket_id": "login-syn-2",
            "subject": "Account access denied",
            "description": "I cannot enter the workspace.",
        },
        {
            "ticket_id": "login-syn-3",
            "subject": "Sign-in rejected",
            "description": "Authentication bounces after Okta.",
        },
        {
            "ticket_id": "file-access-1",
            "subject": "File access request",
            "description": "Need S3 permissions for the export bucket.",
        },
        {
            "ticket_id": "api-auth-1",
            "subject": "API authentication header invalid",
            "description": "Webhook authentication failed during token rotation.",
        },
    ])

    assert package.inputs["top_ticket_clusters"][0] == {"label": "login", "count": 3}
    assert package.metadata["cluster_quality"] == {
        "clustered_row_count": 5,
        "uncategorized_row_count": 0,
        "cluster_count": 3,
        "singleton_cluster_count": 2,
        "largest_cluster_count": 3,
    }
    rows = package.inputs["source_material"]
    assert {
        row["support_ticket_cluster"]
        for row in rows[:3]
    } == {"login"}
    assert all(
        row["support_ticket_cluster_source"] == "token_anchor"
        for row in rows[:3]
    )
    assert rows[3]["support_ticket_cluster"] != "login"
    assert "login" in support_ticket_tokens(rows[0]["text"])
    assert "login" in support_ticket_tokens(rows[1]["text"])
    assert "login" in support_ticket_tokens(rows[2]["text"])
    assert "login" not in support_ticket_tokens(rows[3]["text"])
    assert rows[4]["support_ticket_cluster"] != "login"
    assert "login" not in support_ticket_tokens(rows[4]["text"])
    assert "api" in support_ticket_tokens(rows[4]["text"])


def test_support_ticket_clusters_derive_anchors_without_static_topic_list() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "login-1",
            "subject": "Cannot log in",
            "description": "Login fails after password reset.",
        },
        {
            "ticket_id": "login-2",
            "subject": "Login loop",
            "description": "Users return to the same login page.",
        },
        {
            "ticket_id": "login-3",
            "subject": "Forgot credentials",
            "description": "Login credentials reset email never arrives.",
        },
        {
            "ticket_id": "api-1",
            "subject": "Webhook not firing",
            "description": "API callback never arrives.",
        },
        {
            "ticket_id": "api-2",
            "subject": "API 500 errors",
            "description": "API request returns a 500 error.",
        },
        {
            "ticket_id": "api-3",
            "subject": "Integration broken",
            "description": "API integration cannot sync records.",
        },
        {
            "ticket_id": "dashboard-1",
            "subject": "Dashboard blank",
            "description": "Dashboard charts show a blank page.",
        },
        {
            "ticket_id": "dashboard-2",
            "subject": "Charts not loading",
            "description": "Dashboard charts do not load.",
        },
    ])

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "login", "count": 3},
        {"label": "api", "count": 3},
        {"label": "dashboard", "count": 2},
    ]
    assert package.metadata["cluster_quality"] == {
        "clustered_row_count": 8,
        "uncategorized_row_count": 0,
        "cluster_count": 3,
        "singleton_cluster_count": 0,
        "largest_cluster_count": 3,
    }
    assert {
        row["support_ticket_cluster"]
        for row in package.inputs["source_material"]
    } == {"login", "api", "dashboard"}


def test_support_ticket_clusters_include_remaining_bucket() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": f"ticket-{index}",
            "description": f"How do I fix issue {index}?",
            "pain_category": f"category-{index}",
        }
        for index in range(1, 15)
    ])

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "category-1", "count": 1},
        {"label": "category-2", "count": 1},
        {"label": "category-3", "count": 1},
        {"label": "category-4", "count": 1},
        {"label": "category-5", "count": 1},
        {"label": "category-6", "count": 1},
        {"label": "category-7", "count": 1},
        {"label": "category-8", "count": 1},
        {"label": "category-9", "count": 1},
        {"label": "category-10", "count": 1},
        {"label": "category-11", "count": 1},
        {"label": "category-12", "count": 1},
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
    assert package.warnings == ()


def test_support_ticket_input_package_warns_when_date_column_is_blank() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Login email change",
            "message": "How do I change my email address?",
            "created_at": "",
        }
    ])

    assert "faq_window_days" not in package.inputs
    assert package.inputs["source_period"] == "Uploaded support tickets"
    assert package.inputs["has_dated_window"] is False
    assert package.warnings == (
        {
            "code": "support_ticket_date_window_disabled",
            "message": (
                "Disabled the dated support-ticket source window because "
                "1 of 1 included ticket rows did not include a parseable "
                "source date."
            ),
            "included_row_count": 1,
            "dated_row_count": 0,
            "missing_or_unparseable_date_count": 1,
            "example_source_ids": ["ticket-1"],
        },
    )


def test_support_ticket_date_signal_is_carried_out_of_band_not_on_rows() -> None:
    # #1519 follow-up: the date-column-present signal used to be stamped onto
    # the shared row dict (_date_source_present) and stripped only at the
    # source_material egress. It is now computed during normalization and
    # handed to _source_date_diagnostics out-of-band. Behavior is preserved
    # (a blank date column still disables the window and warns), and no
    # internal marker rides on the exported rows.
    package = build_support_ticket_input_package([
        {
            "ticket_id": "ticket-1",
            "subject": "Login email change",
            "message": "How do I change my email address?",
            "created_at": "",
        }
    ])

    assert package.inputs["has_dated_window"] is False
    assert any(
        warning["code"] == "support_ticket_date_window_disabled"
        for warning in package.warnings
    )
    for row in package.inputs["source_material"]:
        assert "_date_source_present" not in row
        assert not any(str(key).startswith("_") for key in row)


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
    assert package.warnings == (
        {
            "code": "support_ticket_date_window_disabled",
            "message": (
                "Disabled the dated support-ticket source window because "
                "1 of 1 included ticket rows did not include a parseable "
                "source date."
            ),
            "included_row_count": 1,
            "dated_row_count": 0,
            "missing_or_unparseable_date_count": 1,
            "example_source_ids": ["ticket-1"],
        },
    )


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
    assert package.warnings == (
        {
            "code": "support_ticket_date_window_disabled",
            "message": (
                "Disabled the dated support-ticket source window because "
                "1 of 2 included ticket rows did not include a parseable "
                "source date."
            ),
            "included_row_count": 2,
            "dated_row_count": 1,
            "missing_or_unparseable_date_count": 1,
            "example_source_ids": ["ticket-2"],
        },
    )


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
            "support_ticket_cluster": "data export",
            "support_ticket_cluster_key": "tokens:data-export",
            "support_ticket_cluster_source": "token_set",
            "support_ticket_evidence_tier": "csv_customer_text",
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


def test_support_ticket_input_package_recognizes_status_and_csat_columns() -> None:
    package = build_support_ticket_input_package([
        {
            "Ticket ID": "zd-1",
            "Subject": "How do I reset my password?",
            "Description": "I cannot reset my password from the login screen.",
            "Ticket Status": "Closed",
            "Customer Satisfaction Rating": "5",
        },
        {
            "ticket_id": "zd-2",
            "subject": "Billing question",
            "description": "Why was I charged twice this month?",
            "status": "Open",
            "satisfaction_score": 2,
        },
    ])

    rows = package.inputs["source_material"]
    assert rows[0]["ticket_status"] == "Closed"
    assert rows[0]["ticket_status_state"] == "resolved"
    assert rows[0]["csat"] == "5"
    assert rows[0]["csat_score"] == 5.0
    assert rows[1]["ticket_status_state"] == "open"
    assert rows[1]["csat_score"] == 2.0

    assert package.metadata["ticket_status_present"] is True
    assert package.metadata["ticket_status_present_count"] == 2
    assert package.metadata["ticket_status_summary"] == {"resolved": 1, "open": 1}
    assert package.metadata["csat_present"] is True
    assert package.metadata["csat_present_count"] == 2
    assert package.metadata["csat_score_count"] == 2
    assert package.metadata["csat_score_average"] == 3.5


def test_support_ticket_input_package_preserves_support_platform_provenance() -> None:
    package = build_support_ticket_input_package([
        {
            "Ticket ID": "zd-1",
            "Subject": "Where is the login button?",
            "Requester Comment": "Where is the login button?",
            "Support Platform": "zendesk",
        },
        {
            "ticket_id": "hs-1",
            "subject": "How do I export reports?",
            "description": "How do I export reports?",
            "platform": "help_scout",
        },
        {
            "ticket_id": "ic-1",
            "subject": "How do I update my billing email?",
            "description": "How do I update my billing email?",
            "support_platform": "intercom",
        },
    ])

    rows = package.inputs["source_material"]

    assert rows[0]["support_platform"] == "zendesk"
    assert rows[1]["support_platform"] == "help_scout"
    assert rows[2]["support_platform"] == "intercom"


def test_support_ticket_status_normalizes_to_canonical_buckets() -> None:
    statuses = {
        "done": "resolved",
        "Solved": "resolved",
        "In Progress": "open",
        "Pending Customer": "open",
        "Pending Customer Approval": "open",
        "Pending Customer Response": "open",
        "Awaiting Customer": "open",
        "Customer Response": "open",
        "Waiting on Customer": "open",
        "reopened": "reopened",
        "Cancelled": "cancelled",
        "Escalated": "other",
        "Customer Escalation": "other",
        "Pending Vendor Approval": "other",
    }
    package = build_support_ticket_input_package([
        {
            "ticket_id": f"t-{index}",
            "description": f"How do I handle scenario {index} for my account export?",
            "issue_status": raw,
        }
        for index, raw in enumerate(statuses, start=1)
    ])

    got = {
        row["ticket_status"]: row["ticket_status_state"]
        for row in package.inputs["source_material"]
    }
    assert got == statuses
    # the reopened bucket is the churn signal #1419/#1466 consume; keep it distinct
    assert package.metadata["ticket_status_summary"]["reopened"] == 1
    assert package.metadata["ticket_status_summary"]["open"] == 7
    assert package.metadata["ticket_status_summary"]["other"] == 3


def test_support_ticket_csat_parses_numeric_only_and_averages_numeric() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "t-1",
            "description": "How do I export the dashboard before renewal?",
            "csat": "4",
        },
        {
            "ticket_id": "t-2",
            "description": "Where do I update my billing contact details?",
            "csat": "good",
        },
        {
            "ticket_id": "t-3",
            "description": "Why did my report fail to generate overnight?",
            "satisfaction_rating": 2,
        },
    ])

    rows = package.inputs["source_material"]
    assert rows[0]["csat"] == "4"
    assert rows[0]["csat_score"] == 4.0
    # textual ratings are kept raw but yield no numeric score (threshold is #1419's call)
    assert rows[1]["csat"] == "good"
    assert "csat_score" not in rows[1]
    assert rows[2]["csat_score"] == 2.0

    assert package.metadata["csat_present"] is True
    assert package.metadata["csat_present_count"] == 3
    assert package.metadata["csat_score_count"] == 2
    assert package.metadata["csat_score_average"] == 3.0


def test_support_ticket_input_package_marks_textual_csat_present_without_score() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "zd-1",
            "subject": "How do I reset my password?",
            "description": "I cannot reset my password from the login screen.",
            "Customer Satisfaction Rating": "good",
        },
        {
            "ticket_id": "zd-2",
            "subject": "Billing question",
            "description": "Why was I charged twice this month?",
            "satisfaction_rating": "bad",
        },
    ])

    rows = package.inputs["source_material"]
    assert rows[0]["csat"] == "good"
    assert "csat_score" not in rows[0]
    assert rows[1]["csat"] == "bad"
    assert "csat_score" not in rows[1]

    # all-textual CSAT must read as PRESENT even though no numeric score exists
    assert package.metadata["csat_present"] is True
    assert package.metadata["csat_present_count"] == 2
    assert package.metadata["csat_score_count"] == 0
    assert package.metadata["csat_score_average"] is None


def test_support_ticket_input_package_without_status_or_csat_is_unchanged() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "t-1",
            "subject": "How do I reset my password?",
            "description": "I cannot reset my password from the login screen.",
        },
    ])

    row = package.inputs["source_material"][0]
    assert "ticket_status" not in row
    assert "ticket_status_state" not in row
    assert "csat" not in row
    assert "csat_score" not in row
    assert package.metadata["ticket_status_present"] is False
    assert package.metadata["ticket_status_present_count"] == 0
    assert package.metadata["ticket_status_summary"] == {}
    assert package.metadata["csat_present"] is False
    assert package.metadata["csat_present_count"] == 0
    assert package.metadata["csat_score_count"] == 0
    assert package.metadata["csat_score_average"] is None


def test_zendesk_full_thread_rows_preserve_public_roles_and_drop_private_notes() -> None:
    result = load_zendesk_full_thread_rows_from_json_bytes(
        ZENDESK_THREAD_SAMPLE.read_bytes()
    )

    assert result.warnings == ()
    by_id = {row["ticket_id"]: row for row in result.rows}
    assert by_id["2"]["description"] == (
        "I was billed twice for this month. How do I get the duplicate charge refunded?"
    )
    assert by_id["2"]["resolution_text"] == (
        "We confirmed the duplicate billing event and refunded the extra charge. "
        "Refunds normally appear on the original payment method in 5-10 business days."
    )
    assert by_id["2"]["ticket_status"] == "solved"
    assert by_id["2"]["satisfaction_rating"] == "good"

    assert "resolution_text" not in by_id["28"]
    assert "A member of the support team will get back to you" not in str(by_id["28"])
    assert "resolution_text" not in by_id["34"]
    assert "Internal note" not in str(result.rows)

    assert by_id["41"]["resolution_text"] == (
        "We sent the standard resolution steps and marked this as solved. "
        "Please reply if the issue continues."
    )
    assert "This is still broken after trying the steps" in by_id["41"]["description"]
    assert "A member of the support team will get back to you" not in str(by_id["41"])


def test_zendesk_full_thread_rows_load_from_json_file() -> None:
    result = load_zendesk_full_thread_rows_from_json_file(ZENDESK_THREAD_SAMPLE)

    by_id = {row["ticket_id"]: row for row in result.rows}
    assert set(by_id) == {"2", "28", "34", "41"}
    assert by_id["2"]["resolution_text"].startswith(
        "We confirmed the duplicate billing event"
    )
    assert by_id["2"]["satisfaction_rating"] == "good"
    assert "Internal note" not in json.dumps(result.rows)
    assert result.warnings == ()


def test_zendesk_full_thread_rows_suppress_private_first_description() -> None:
    result = rows_from_zendesk_full_thread({
        "tickets": [{
            "ticket": {
                "id": "zd-private-first",
                "subject": "Internal migration workaround",
                "description": (
                    "Internal note: explain the real workaround only to the owner."
                ),
                "requester_id": "requester-1",
            },
            "comments": [
                {
                    "author_id": "agent-1",
                    "public": False,
                    "plain_body": (
                        "Internal note: explain the real workaround only to the owner."
                    ),
                },
                {
                    "author_id": "requester-1",
                    "public": True,
                    "plain_body": "What permission do I need for account exports?",
                },
            ],
        }],
    })

    assert result.warnings == ()
    assert result.rows == [{
        "ticket_id": "zd-private-first",
        "source_id": "zd-private-first",
        "source_type": "support_ticket",
        "subject": "Internal migration workaround",
        "description": "What permission do I need for account exports?",
    }]
    package = build_support_ticket_input_package(result.rows)
    assert "Internal note" not in json.dumps(package.as_dict())
    assert package.inputs["faq_questions"] == [
        "What permission do I need for account exports?"
    ]


def test_zendesk_full_thread_rows_keep_substantive_agent_reply_after_boilerplate() -> None:
    result = rows_from_zendesk_full_thread({
        "tickets": [{
            "ticket": {
                "id": "zd-boilerplate-answer",
                "subject": "How do I export invoices?",
                "description": "Where do invoice exports live?",
                "requester_id": "requester-1",
            },
            "comments": [
                {
                    "author_id": "agent-1",
                    "public": True,
                    "plain_body": (
                        "Thanks for reaching out. Open Billing > Invoices, "
                        "then choose Export CSV."
                    ),
                },
                {
                    "author_id": "agent-1",
                    "public": True,
                    "plain_body": (
                        "A member of the support team will get back to you within "
                        "the next 48 hours."
                    ),
                },
            ],
        }],
    })

    assert result.warnings == ()
    row = result.rows[0]
    assert row["resolution_text"] == (
        "Thanks for reaching out. Open Billing > Invoices, then choose Export CSV."
    )
    assert "A member of the support team" not in str(row)


def test_zendesk_full_thread_rows_feed_status_csat_and_resolution_package() -> None:
    result = load_zendesk_full_thread_rows_from_json_bytes(
        ZENDESK_THREAD_SAMPLE.read_bytes()
    )
    package = build_support_ticket_input_package(result.rows)

    assert package.inputs["source_row_count"] == 4
    assert package.inputs["included_ticket_row_count"] == 4
    assert package.inputs["support_ticket_resolution_evidence_present"] is True
    assert package.inputs["support_ticket_resolution_evidence_count"] == 2
    assert package.metadata["ticket_status_present"] is True
    assert package.metadata["ticket_status_present_count"] == 4
    assert package.metadata["ticket_status_summary"] == {"resolved": 1, "open": 3}
    assert package.metadata["csat_present"] is True
    assert package.metadata["csat_present_count"] == 1
    assert package.metadata["csat_score_count"] == 0
    assert package.metadata["csat_score_average"] is None
    assert "Internal note" not in json.dumps(package.as_dict())
    assert "A member of the support team will get back to you" not in json.dumps(
        package.as_dict()
    )


def test_zendesk_full_thread_rows_warn_on_malformed_entries() -> None:
    result = rows_from_zendesk_full_thread({
        "tickets": [
            {"comments": []},
            {
                "ticket": {
                    "id": "zd-valid",
                    "subject": "How do I export data?",
                    "description": "Where do I download the export?",
                },
                "comments": "not-a-list",
            },
        ],
    })

    assert result.rows == [{
        "ticket_id": "zd-valid",
        "source_id": "zd-valid",
        "source_type": "support_ticket",
        "subject": "How do I export data?",
        "description": "Where do I download the export?",
    }]
    assert result.warnings == (
        {
            "code": "zendesk_thread_ticket_missing",
            "row_index": 1,
            "message": "Skipped Zendesk thread row because ticket was missing.",
        },
        {
            "code": "zendesk_thread_comments_invalid",
            "row_index": 2,
            "source_id": "zd-valid",
            "message": "Ignored Zendesk comments because they were not a list.",
        },
    )
