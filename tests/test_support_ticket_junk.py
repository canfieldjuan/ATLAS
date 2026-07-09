"""S6E: junk/auto-reply admission gate (F2).

The F2 acceptance shape (from the audit's live run): repeated auto-replies
must not form a billed cluster or own the top recommendation. Both error
directions are pinned: machine-generated mail is excluded by positional
shape (subject generator prefixes, first-person whole-line assertions),
while customer tickets ABOUT auto-reply/out-of-office features pass.
"""

from __future__ import annotations

import pytest

from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)
from extracted_content_pipeline.support_ticket_junk import (
    JUNK_REASON_AUTO_REPLY,
    JUNK_REASON_BOUNCE,
    JUNK_REASON_NO_NEW_CONTENT,
    support_ticket_row_is_junk,
)


def _f2_rows() -> list[dict[str, str]]:
    real = [
        {
            "id": f"r{i}",
            "subject": "Cannot reset my password",
            "description": "I cannot reset my password from the login page.",
        }
        for i in range(5)
    ]
    junk = [
        {
            "id": f"j{i}",
            "subject": "Automatic reply: Out of Office",
            "description": (
                "I am out of the office until Monday with limited access "
                "to email."
            ),
        }
        for i in range(6)
    ]
    return real + junk


def test_f2_auto_replies_do_not_form_a_billed_cluster() -> None:
    package = build_support_ticket_input_package(_f2_rows())
    labels = " ".join(
        str(cluster.get("label", "")).lower()
        for cluster in package.inputs.get("top_ticket_clusters") or []
    )
    assert "office" not in labels
    assert "automatic" not in labels
    assert package.inputs["included_ticket_row_count"] == 5
    assert package.metadata["junk_excluded_count"] == 6
    assert package.metadata["junk_excluded_reasons"] == {"auto_reply": 6}


def test_f2_diagnostics_report_counts_not_content() -> None:
    package = build_support_ticket_input_package(_f2_rows())
    warning = next(
        w for w in package.warnings
        if w.get("code") == "support_ticket_junk_excluded"
    )
    assert warning["count"] == 6
    assert warning["reasons"] == {"auto_reply": 6}
    assert "Out of Office" not in str(warning)


def test_row_accounting_still_sums() -> None:
    package = build_support_ticket_input_package(_f2_rows())
    inputs = package.inputs
    assert (
        inputs["included_ticket_row_count"]
        + inputs["skipped_ticket_row_count"]
        == inputs["source_row_count"]
    )


def test_feature_questions_about_auto_reply_are_admitted() -> None:
    package = build_support_ticket_input_package([
        {
            "id": "a",
            "subject": "How do I set an out of office auto-reply?",
            "description": (
                "I want to configure the out of office auto-reply feature "
                "for my team."
            ),
        },
        {
            "id": "b",
            "subject": "Out of office not working",
            "description": (
                "My out of office replies are not sending to external "
                "contacts."
            ),
        },
    ])
    assert package.metadata["junk_excluded_count"] == 0
    assert package.inputs["included_ticket_row_count"] == 2


@pytest.mark.parametrize(
    ("subject", "reason"),
    [
        ("Automatic reply: Out of Office", JUNK_REASON_AUTO_REPLY),
        ("Auto-Reply: your ticket", JUNK_REASON_AUTO_REPLY),
        ("[EXT] Automatic reply: away", JUNK_REASON_AUTO_REPLY),
        ("Out of Office: Re: invoice question", JUNK_REASON_AUTO_REPLY),
        ("Undeliverable: your message", JUNK_REASON_BOUNCE),
        ("Delivery Status Notification (Failure)", JUNK_REASON_BOUNCE),
        ("Mail delivery failed: returning message", JUNK_REASON_BOUNCE),
    ],
)
def test_generator_subject_prefixes_are_junk(subject: str, reason: str) -> None:
    assert support_ticket_row_is_junk(subject, "some body") == reason


@pytest.mark.parametrize(
    "line",
    [
        "I am out of the office until Monday.",
        "I will be away from my desk this week",
        "I am currently on vacation and will respond on my return.",
        "I will have limited access to email until 7/20.",
        "This is an automated response to your inquiry.",
        "Please do not reply to this automated email.",
        "Your message could not be delivered to the recipient.",
    ],
)
def test_first_person_assertion_lines_are_junk(line: str) -> None:
    assert support_ticket_row_is_junk("Re: help", line) is not None


@pytest.mark.parametrize(
    ("subject", "body"),
    [
        (
            "How do I set an out of office auto-reply?",
            "I want the out of office feature to reply automatically.",
        ),
        (
            "Out of office not working",
            "My out of office replies are not sending.",
        ),
        (
            "Question about automated responses",
            "Can I customize what the automated response says?",
        ),
        (
            "Vacation policy",
            "Am I able to be out of the office next month?",
        ),
    ],
)
def test_customer_wording_about_the_features_passes(
    subject: str, body: str,
) -> None:
    assert support_ticket_row_is_junk(subject, body) is None


def test_all_quote_rows_count_as_no_new_content() -> None:
    assert support_ticket_row_is_junk(
        "", "", had_source_text=True
    ) == JUNK_REASON_NO_NEW_CONTENT
    # A subject-only row keeps its subject content and is not junk.
    assert support_ticket_row_is_junk(
        "Re: export question", "", had_source_text=True
    ) is None


# Round-1 review refinements: comment-only rows in scope, bounce regex,
# colon-only generator delimiters, question-line veto, body-only prefixes.


def test_comment_only_junk_rows_are_gated() -> None:
    package = build_support_ticket_input_package([
        {
            "id": f"c{i}",
            "subject": "Re: help",
            "comments": [
                "I am out of the office until Monday with limited access "
                "to email."
            ],
        }
        for i in range(3)
    ])
    assert package.metadata["junk_excluded_count"] == 3


def test_was_not_delivered_bounce_body_matches() -> None:
    assert support_ticket_row_is_junk(
        "Re: x", "Your message was not delivered to the recipient."
    ) == JUNK_REASON_BOUNCE


@pytest.mark.parametrize(
    ("subject", "body"),
    [
        ("Out of office - not working", "my replies are not sending"),
        ("Auto reply - setup question", "how to configure the auto reply"),
        (
            "Undeliverable emails are not reaching customers",
            "our outbound emails keep bouncing",
        ),
    ],
)
def test_customer_separators_are_not_generator_delimiters(
    subject: str, body: str,
) -> None:
    assert support_ticket_row_is_junk(subject, body) is None


def test_mixed_ticket_quoting_a_template_is_admitted() -> None:
    assert support_ticket_row_is_junk(
        "Template help",
        "Template we configured:\n"
        "I am out of the office until Monday.\n"
        "Why is it not sending?",
    ) is None


def test_generator_prefix_on_first_body_line_is_junk() -> None:
    assert support_ticket_row_is_junk(
        "", "Automatic reply: Out of Office\nI am away."
    ) == JUNK_REASON_AUTO_REPLY
    assert support_ticket_row_is_junk(
        "", "Undeliverable: your message\ndetails follow"
    ) == JUNK_REASON_BOUNCE


# Round-2 review refinements: mid-line questions veto, line-preserved
# comments, anchored delivery-failed subjects, labeled subject lines.


def test_question_before_trailing_text_vetoes_body_junk() -> None:
    assert support_ticket_row_is_junk(
        "Template help",
        "I am out of the office until Monday.\nWhy is it not sending? Thanks",
    ) is None


def test_multiline_comment_auto_reply_is_gated() -> None:
    package = build_support_ticket_input_package([
        {
            "id": "c1",
            "subject": "Re: help",
            "comments": ["Hello,\nI am out of the office until Monday.\nBest,\nBob"],
        }
    ])
    assert package.metadata["junk_excluded_count"] == 1


def test_delivery_failed_prose_subject_is_admitted() -> None:
    assert support_ticket_row_is_junk(
        "Delivery has failed to these recipients when sending invoices",
        "our invoices bounce",
    ) is None
    assert support_ticket_row_is_junk(
        "Delivery has failed to these recipients or groups:", "x"
    ) == JUNK_REASON_BOUNCE


def test_labeled_subject_line_in_text_export_is_junk() -> None:
    assert support_ticket_row_is_junk(
        "", "Subject: Automatic reply: Out of Office\nI am away."
    ) == JUNK_REASON_AUTO_REPLY


# Round-3 review refinements: contracted first-person forms, office-context
# return lines, and header-block subject scanning.


@pytest.mark.parametrize(
    "line",
    [
        "I'm out of the office until Monday with limited access to email",
        "I'll be away from my desk this week",
    ],
)
def test_contracted_first_person_assertions_are_junk(line: str) -> None:
    assert support_ticket_row_is_junk("Re: help", line) == JUNK_REASON_AUTO_REPLY


def test_return_lines_need_office_context() -> None:
    assert support_ticket_row_is_junk(
        "Update", "I will return on Tuesday with the debug logs."
    ) is None
    assert support_ticket_row_is_junk(
        "Re: help", "I will return to the office on Monday."
    ) == JUNK_REASON_AUTO_REPLY


def test_header_block_subject_lines_are_scanned() -> None:
    assert support_ticket_row_is_junk(
        "",
        "From: Mail Delivery Subsystem\n"
        "Subject: Undeliverable: your message\n"
        "Diagnostic-Code: smtp 550",
    ) == JUNK_REASON_BOUNCE


def test_deep_unlabeled_prefixes_do_not_junk_the_row() -> None:
    assert support_ticket_row_is_junk(
        "Setup question",
        "here is what my customers see\nsome context\nmore context\n"
        "extra line\nAutomatic reply: Out of Office",
    ) is None


# Round-4 refinement: bounce assertions classify before the question veto.


def test_bounce_quoting_an_interrogative_subject_is_still_a_bounce() -> None:
    assert support_ticket_row_is_junk(
        "Re: something",
        "Your message could not be delivered to the recipient.\n"
        "Subject: Why can't I login?",
    ) == JUNK_REASON_BOUNCE


def test_customer_question_about_bounces_still_passes() -> None:
    assert support_ticket_row_is_junk(
        "Bounce troubleshooting",
        "Why do my customers see Your message could not be delivered?",
    ) is None


# Round-5 refinements: first-line assertion precedence, temporal OOO tails,
# and reply-label narrowing.


def test_leading_assertion_beats_quoted_question() -> None:
    assert support_ticket_row_is_junk(
        "Re: something",
        "I am out of the office until Monday.\nSubject: Why can't I login?",
    ) == JUNK_REASON_AUTO_REPLY


def test_customer_context_before_bounce_line_is_admitted() -> None:
    assert support_ticket_row_is_junk(
        "Email issue",
        "Customer reports seeing this error:\n"
        "Your message could not be delivered to the recipient.\n"
        "How do we fix this?",
    ) is None


def test_descriptive_ooo_feature_report_is_admitted() -> None:
    assert support_ticket_row_is_junk(
        "Settings issue",
        "I am out of the office auto-reply setting is not working",
    ) is None


def test_customer_reply_to_auto_reply_subject_is_admitted() -> None:
    assert support_ticket_row_is_junk(
        "Re: Automatic reply: Out of Office",
        "Why is my auto-reply not sending?",
    ) is None


# Round-6 refinements: bounded OOO continuations, URL-safe question voice,
# subject-only assertions, and header-shape-guarded subject scanning.


def test_comma_and_feature_report_is_admitted() -> None:
    assert support_ticket_row_is_junk(
        "Settings", "I am out of the office, and the auto-reply is not sending"
    ) is None
    assert support_ticket_row_is_junk(
        "Re: help", "I am out of the office."
    ) == JUNK_REASON_AUTO_REPLY


def test_url_query_strings_are_not_customer_questions() -> None:
    assert support_ticket_row_is_junk(
        "Re: help",
        "I am out of the office until Monday.\n"
        "https://help.example.com/contact?id=12",
    ) == JUNK_REASON_AUTO_REPLY


def test_subject_only_machine_assertions_classify() -> None:
    assert support_ticket_row_is_junk(
        "Your message could not be delivered", ""
    ) == JUNK_REASON_BOUNCE
    assert support_ticket_row_is_junk(
        "I am out of the office until Monday", ""
    ) == JUNK_REASON_AUTO_REPLY


def test_prose_before_subject_label_does_not_junk() -> None:
    assert support_ticket_row_is_junk(
        "Auto reply help",
        "I need help with auto replies\n"
        "Subject: Automatic reply: Out of Office\n"
        "Why is this happening?",
    ) is None


# Round-7 refinements: compact headers, bounded limited-access tails,
# body-over-comment gating, and subject assertions deferring to body voice.


def test_compact_header_lines_are_header_shaped() -> None:
    assert support_ticket_row_is_junk(
        "", "From:Mail Delivery Subsystem\nSubject: Undeliverable: your message"
    ) == JUNK_REASON_BOUNCE


def test_limited_access_tail_is_bounded() -> None:
    assert support_ticket_row_is_junk(
        "Settings",
        "I will have limited access to email auto-reply setting is not working",
    ) is None
    assert support_ticket_row_is_junk(
        "Re: help", "I will have limited access to email until 7/20."
    ) == JUNK_REASON_AUTO_REPLY


def test_auto_ack_comment_does_not_junk_a_real_ticket() -> None:
    package = build_support_ticket_input_package([{
        "id": "r1",
        "subject": "Login broken",
        "description": "Login is broken for our team.",
        "comments": ["We received your ticket and will get back to you soon."],
    }])
    assert package.metadata["junk_excluded_count"] == 0
    assert package.inputs["included_ticket_row_count"] == 1


def test_machine_subject_defers_to_customer_body_voice() -> None:
    assert support_ticket_row_is_junk(
        "Your message could not be delivered",
        "Why are our invoices bouncing?",
    ) is None
    assert support_ticket_row_is_junk(
        "Your message could not be delivered", ""
    ) == JUNK_REASON_BOUNCE
