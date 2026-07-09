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
