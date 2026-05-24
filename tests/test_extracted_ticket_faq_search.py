from __future__ import annotations

from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft
from extracted_content_pipeline.ticket_faq_search import (
    build_ticket_faq_search_documents,
    search_ticket_faq_documents,
)


def _draft(
    *,
    faq_id: str = "faq-1",
    account_id: str = "acct-1",
    corpus_id: str = "corpus-1",
    status: str = "approved",
    target_id: str = "support-account-1",
    items: list[dict[str, object]] | None = None,
) -> TicketFAQDraft:
    return TicketFAQDraft(
        id=faq_id,
        target_id=target_id,
        target_mode="support_account",
        title="Support FAQ",
        markdown="# Support FAQ",
        status=status,
        source_count=3,
        ticket_source_count=3,
        metadata={
            "scope": {"account_id": account_id},
            "corpus_id": corpus_id,
        },
        items=items or [
            {
                "topic": "password reset",
                "question": "How do I reset my password?",
                "summary": "Customers cannot find the password reset email.",
                "steps": [
                    "Open the login page.",
                    "Request a new password reset email.",
                ],
                "when_to_contact_support": "Contact support if the reset email never arrives.",
                "source_ids": ["ticket-1", "ticket-2", "ticket-2"],
                "ticket_count": 2,
            },
            {
                "topic": "billing invoice",
                "question": "Where can I download an invoice?",
                "answer": "Open billing settings and download the invoice PDF.",
                "source_ids": ["ticket-3"],
                "ticket_count": 1,
            },
        ],
    )


def test_build_ticket_faq_search_documents_projects_item_fields() -> None:
    documents = build_ticket_faq_search_documents(_draft())

    assert len(documents) == 2
    first = documents[0]
    assert first.account_id == "acct-1"
    assert first.corpus_id == "corpus-1"
    assert first.faq_id == "faq-1"
    assert first.rank == 1
    assert first.topic == "password reset"
    assert first.question == "How do I reset my password?"
    assert first.answer_summary == "Customers cannot find the password reset email."
    assert first.source_ids == ("ticket-1", "ticket-2")
    assert first.ticket_count == 2
    assert "reset email" in first.search_text
    assert "search_text" not in first.as_dict()


def test_search_ticket_faq_documents_returns_route_shaped_envelope() -> None:
    documents = build_ticket_faq_search_documents(_draft())

    response = search_ticket_faq_documents(
        documents,
        query="password reset email",
        account_id="acct-1",
        corpus_id="corpus-1",
    )

    assert response.as_dict() == {
        "query": "password reset email",
        "count": 1,
        "results": [{
            "account_id": "acct-1",
            "corpus_id": "corpus-1",
            "faq_id": "faq-1",
            "target_id": "support-account-1",
            "target_mode": "support_account",
            "status": "approved",
            "rank": 1,
            "topic": "password reset",
            "question": "How do I reset my password?",
            "answer_summary": "Customers cannot find the password reset email.",
            "source_ids": ["ticket-1", "ticket-2"],
            "ticket_count": 2,
            "score": 21,
        }],
    }


def test_search_ticket_faq_documents_filters_tenant_corpus_and_status() -> None:
    documents = (
        *build_ticket_faq_search_documents(_draft(faq_id="faq-1", account_id="acct-1", corpus_id="corpus-1")),
        *build_ticket_faq_search_documents(_draft(faq_id="faq-2", account_id="acct-2", corpus_id="corpus-1")),
        *build_ticket_faq_search_documents(_draft(faq_id="faq-3", account_id="acct-1", corpus_id="corpus-2")),
        *build_ticket_faq_search_documents(_draft(faq_id="faq-4", account_id="acct-1", corpus_id="corpus-1", status="draft")),
    )

    response = search_ticket_faq_documents(
        documents,
        query="reset",
        account_id="acct-1",
        corpus_id="corpus-1",
        status="approved",
    )

    result_ids = [row["faq_id"] for row in response.as_dict()["results"]]
    assert result_ids == ["faq-1"]


def test_search_ticket_faq_documents_fails_closed_for_blank_tenant() -> None:
    documents = build_ticket_faq_search_documents(
        _draft(faq_id="empty-tenant-faq", account_id="", corpus_id="corpus-1")
    )

    response = search_ticket_faq_documents(
        documents,
        query="reset",
        account_id="",
        corpus_id="corpus-1",
    )

    assert response.as_dict() == {"query": "reset", "results": [], "count": 0}


def test_search_ticket_faq_documents_ranks_by_score_then_rank() -> None:
    documents = build_ticket_faq_search_documents(
        _draft(
            items=[
                {
                    "topic": "account access",
                    "question": "How do I access the account?",
                    "summary": "Customers ask about account access.",
                    "source_ids": ["ticket-1"],
                    "ticket_count": 1,
                },
                {
                    "topic": "password reset",
                    "question": "How do I reset account password access?",
                    "summary": "Password reset access access.",
                    "source_ids": ["ticket-2"],
                    "ticket_count": 1,
                },
            ]
        )
    )

    response = search_ticket_faq_documents(
        documents,
        query="password access",
        account_id="acct-1",
        corpus_id="corpus-1",
    )

    rows = response.as_dict()["results"]
    assert [row["question"] for row in rows] == [
        "How do I reset account password access?",
        "How do I access the account?",
    ]
    assert rows[0]["score"] > rows[1]["score"]


def test_search_ticket_faq_documents_handles_blank_query_and_limit() -> None:
    documents = build_ticket_faq_search_documents(_draft())

    blank = search_ticket_faq_documents(
        documents,
        query="   ",
        account_id="acct-1",
    )
    limited = search_ticket_faq_documents(
        documents,
        query="password",
        account_id="acct-1",
        limit=0,
    )

    assert blank.as_dict() == {"query": "", "results": [], "count": 0}
    assert limited.as_dict() == {"query": "password", "results": [], "count": 0}
