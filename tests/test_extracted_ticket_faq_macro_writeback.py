from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.faq_macro_writeback import (
    DryRunMacroPublishProvider,
    build_macro_writeback_preview,
    macro_content_hash,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


def _draft(
    *,
    status: str = "approved",
    items: tuple[dict[str, object], ...],
    draft_id: str = "faq-draft-1",
) -> TicketFAQDraft:
    return TicketFAQDraft(
        id=draft_id,
        target_id="ticket-faq-report",
        target_mode="support_ticket_faq",
        title="Saved FAQ report",
        markdown="# Saved FAQ report",
        items=items,
        source_count=3,
        ticket_source_count=3,
        status=status,
    )


def test_macro_writeback_preview_maps_verified_approved_faq_item() -> None:
    preview = build_macro_writeback_preview([
        _draft(
            items=(
                {
                    "faq_item_id": "faq-item-1",
                    "topic": "billing confusion",
                    "question": "Why was I charged twice?",
                    "resolution_text": (
                        "Open Billing, review the invoice history, and compare "
                        "pending authorizations with settled charges."
                    ),
                    "answer_evidence_status": "resolution_evidence",
                    "source_ids": ("ticket-1", "ticket-2"),
                },
            )
        )
    ])

    assert preview.publishable_count == 1
    assert preview.skipped_count == 0
    macro = preview.macros[0]
    assert macro.title == "Why was I charged twice?"
    assert macro.body == (
        "Open Billing, review the invoice history, and compare pending "
        "authorizations with settled charges."
    )
    assert macro.category == "billing confusion"
    assert macro.faq_draft_id == "faq-draft-1"
    assert macro.faq_item_id == "faq-item-1"
    assert macro.source_ids == ("ticket-1", "ticket-2")
    assert macro.metadata == {
        "target_id": "ticket-faq-report",
        "target_mode": "support_ticket_faq",
        "draft_title": "Saved FAQ report",
    }
    assert preview.as_dict()["publishable_count"] == 1


def test_macro_writeback_preview_requires_verified_and_approved_double_gate() -> None:
    preview = build_macro_writeback_preview([
        _draft(
            status="draft",
            draft_id="faq-unapproved",
            items=(
                {
                    "question": "How do I update billing?",
                    "resolution_text": "Open Billing and update the payment method.",
                    "answer_evidence_status": "resolution_evidence",
                },
            ),
        ),
        _draft(
            draft_id="faq-unverified",
            items=(
                {
                    "question": "How do I export a report?",
                    "steps": ("Open Reports.", "Choose Export."),
                    "answer_evidence_status": "draft_needs_review",
                },
            ),
        ),
        _draft(
            draft_id="faq-approved",
            items=(
                {
                    "question": "Where do I find invoices?",
                    "answer": "Open Billing and choose Invoice history.",
                    "answer_evidence_status": "resolution_evidence",
                },
            ),
        ),
    ])

    assert [macro.title for macro in preview.macros] == ["Where do I find invoices?"]
    assert [(item.faq_draft_id, item.reason) for item in preview.skipped] == [
        ("faq-unapproved", "draft_not_approved"),
        ("faq-unverified", "answer_not_verified"),
    ]


def test_macro_writeback_preview_falls_back_to_numbered_steps() -> None:
    preview = build_macro_writeback_preview([
        _draft(
            items=(
                {
                    "question": "How do I reset my password?",
                    "answer": "Customers mention: reset access. Evidence comes from 1 ticket source(s).",
                    "steps": ("Open Settings.", "Choose Reset password."),
                    "answer_evidence_status": "resolution_evidence",
                },
            )
        )
    ])

    assert preview.macros[0].body == (
        "1. Open Settings.\n"
        "2. Choose Reset password."
    )


def test_macro_content_hash_tracks_customer_facing_body_changes() -> None:
    preview = build_macro_writeback_preview([
        _draft(
            items=(
                {
                    "faq_item_id": "faq-item-1",
                    "topic": "billing confusion",
                    "question": "How do I export?",
                    "resolution_text": "Open Reports and choose Export.",
                    "answer_evidence_status": "resolution_evidence",
                },
            )
        )
    ])
    original = preview.macros[0]
    same_content = type(original)(
        title="  How   do I export? ",
        body="Open Reports and choose Export.\n",
        category=" billing confusion ",
        faq_draft_id=original.faq_draft_id,
        faq_item_id=original.faq_item_id,
    )
    changed_body = type(original)(
        title=original.title,
        body="Open Reports, choose Export, then select CSV.",
        category=original.category,
        faq_draft_id=original.faq_draft_id,
        faq_item_id=original.faq_item_id,
    )

    assert macro_content_hash(original) == macro_content_hash(same_content)
    assert macro_content_hash(original) != macro_content_hash(changed_body)


def test_macro_writeback_preview_uses_real_faq_generator_resolution_steps() -> None:
    generated = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Double charge",
            "source_id": "ticket-1",
            "evidence": [{
                "text": "Why was I charged twice this month?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "Open Billing, review the invoice history, and compare "
                    "pending authorizations with settled charges."
                ),
            }],
        }]
    )
    preview = build_macro_writeback_preview([
        _draft(items=tuple(dict(item) for item in generated.items))
    ])

    macro = preview.macros[0]
    assert "Customers mention:" not in generated.items[0]["answer"]
    assert generated.items[0]["answer"] == (
        "To resolve this, open Billing, review the invoice history, and compare "
        "pending authorizations with settled charges."
    )
    assert "Customers mention:" not in macro.body
    assert macro.body.startswith("1. Open Billing, review the invoice history")
    assert "Use the uploaded resolution evidence" not in macro.body
    assert "pending authorizations with settled charges" in macro.body


def test_macro_writeback_preview_reports_missing_customer_facing_fields() -> None:
    preview = build_macro_writeback_preview([
        _draft(
            items=(
                {
                    "resolution_text": "Open Billing and review invoices.",
                    "answer_evidence_status": "resolution_evidence",
                },
                {
                    "question": "How do I change seats?",
                    "answer_evidence_status": "resolution_evidence",
                },
            )
        )
    ])

    assert preview.macros == ()
    assert [item.reason for item in preview.skipped] == [
        "missing_question",
        "missing_resolution_body",
    ]
    assert preview.skipped[1].question == "How do I change seats?"


@pytest.mark.asyncio
async def test_dry_run_macro_provider_returns_per_item_preview_results() -> None:
    preview = build_macro_writeback_preview([
        _draft(
            items=(
                {
                    "question": "Why was I charged twice?",
                    "answer": "Open Billing and compare pending and settled charges.",
                    "answer_evidence_status": "resolution_evidence",
                },
            )
        )
    ])

    results = await DryRunMacroPublishProvider().publish(
        preview.macros,
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
    )

    assert len(results) == 1
    assert results[0].status == "dry_run"
    assert results[0].external_id == ""
    assert results[0].error == ""
    assert results[0].macro.title == "Why was I charged twice?"
    assert results[0].as_dict()["macro"]["faq_item_id"] == "faq-draft-1:item-1"
