from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_support_ticket_generated_content.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_support_ticket_generated_content",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def _landing_export(
    *,
    source_context: dict[str, Any] | None = None,
    text_override: str | None = None,
) -> dict[str, Any]:
    context = source_context if source_context is not None else _source_context()
    headline = text_override or (
        "Turn 2 support-ticket rows about account and reporting questions "
        "into FAQ answers customers can use."
    )
    return {
        "count": 1,
        "rows": [{
            "id": "lp-live-smoke-1",
            "title": "Support-ticket FAQ report",
            "hero": {
                "headline": headline,
                "subheadline": (
                    "Customers asked: How do I change my login email? "
                    "The answer should be visible before they email support."
                ),
            },
            "sections": [
                {
                    "heading": "Account questions keep coming back",
                    "body": "Reporting export questions show the same help-center gap.",
                }
            ],
            "cta": {"label": "Upload Ticket CSV -- Free Analysis"},
            "meta": {"description": "Support ticket FAQ answers for small teams."},
            "metadata": {"source_context": context},
        }],
    }


def _blog_export(
    *,
    data_context: dict[str, Any] | None = None,
    content_override: str | None = None,
) -> dict[str, Any]:
    context = data_context if data_context is not None else _blog_context()
    content = content_override or (
        "## What repeat support tickets reveal\n\n"
        "The uploaded 2 ticket rows show account and reporting questions. "
        "Those clusters should become FAQ answers before customers ask again."
    )
    return {
        "count": 1,
        "rows": [{
            "id": "blog-live-smoke-1",
            "title": "Support-ticket questions customers keep asking",
            "description": "A support-ticket FAQ article for small teams.",
            "content": content,
            "tags": ["support tickets", "FAQ"],
            "charts": [{"title": "Top support-ticket clusters"}],
            "data_context": context,
        }],
    }


def _source_context() -> dict[str, Any]:
    return {
        "source_row_count": 2,
        "included_ticket_row_count": 2,
        "skipped_ticket_row_count": 0,
        "truncated_ticket_row_count": 0,
        "question_like_ticket_count": 2,
        "source_period": "Uploaded support tickets",
        "top_ticket_clusters": [
            {"label": "account", "count": 1},
            {"label": "reporting", "count": 1},
        ],
        "faq_questions": [
            "How do I change my login email?",
            "How do we export campaign attribution data before renewal?",
        ],
        "customer_wording_examples": [
            "I cannot find where to update the email on my account.",
        ],
    }


def _blog_context() -> dict[str, Any]:
    return {
        "source_row_count": 2,
        "included_ticket_row_count": 2,
        "question_like_ticket_count": 2,
        "source_period": "Uploaded support tickets",
        "top_clusters": [
            {"label": "account", "count": 1},
            {"label": "reporting", "count": 1},
        ],
    }


def test_landing_export_passes_when_generated_text_uses_ticket_context() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(),
        output="landing_page",
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["source_context_summary"] == {
        "source_row_count": 2,
        "included_ticket_row_count": 2,
        "question_like_ticket_count": 2,
        "source_period": "Uploaded support tickets",
        "cluster_count": 2,
    }


def test_blog_export_passes_with_blog_data_context_shape() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _blog_export(),
        output="blog_post",
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert not result["warnings"]


def test_blog_export_fails_unsupported_uploaded_ticket_timeframe() -> None:
    export = _blog_export(
        content_override=(
            "Between May 2026 and the present, we analyzed 2 support tickets. "
            "The account and reporting clusters show what customers keep asking."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    assert any(
        "undated uploaded-ticket source" in error
        for error in result["errors"]
    )
    timeframe_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_timeframe_truthful"
    )
    assert timeframe_check["passed"] is False
    assert timeframe_check["details"] == {
        "unsupported_timeframes": ["Between May 2026 and"]
    }


def test_landing_export_fails_unsupported_uploaded_ticket_timeframe() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(
            text_override=(
                "Between May 2026 and today, the uploaded 2 support tickets "
                "show account and reporting questions."
            )
        ),
        output="landing_page",
    )

    assert result["ok"] is False
    assert any(
        "undated uploaded-ticket source" in error
        for error in result["errors"]
    )
    timeframe_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_timeframe_truthful"
    )
    assert timeframe_check["passed"] is False
    assert timeframe_check["details"] == {
        "unsupported_timeframes": ["Between May 2026 and"]
    }


def test_blog_export_allows_timeframe_when_source_period_supplies_it() -> None:
    context = _blog_context()
    context["source_period"] = "Last 90 days of support tickets"
    context["review_period"] = "last 90 days"
    export = _blog_export(
        data_context=context,
        content_override=(
            "In the last 90 days, the 2 support tickets show account and "
            "reporting questions customers keep asking."
        ),
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    timeframe_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_timeframe_truthful"
    )
    assert timeframe_check["passed"] is True
    assert timeframe_check["details"] == {"applicable": False}


def test_blog_export_fails_unsupported_uploaded_ticket_cadence() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Publishing both FAQ answers could eliminate 2 tickets per week."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    assert any(
        "recurring cadence for an undated uploaded-ticket source" in error
        for error in result["errors"]
    )
    cadence_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_cadence_truthful"
    )
    assert cadence_check["passed"] is False
    assert cadence_check["details"] == {"unsupported_cadences": ["per week"]}


def test_blog_export_allows_source_backed_uploaded_ticket_cadence() -> None:
    context = _blog_context()
    context["faq_questions"] = [
        "Can I schedule a weekly attribution report export for finance?"
    ]
    export = _blog_export(
        data_context=context,
        content_override=(
            "The uploaded tickets include the customer question "
            "\"Can I schedule a weekly attribution report export for finance?\" "
            "That question should be reviewed as customer wording, not a "
            "generated cadence claim."
        ),
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    cadence_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_cadence_truthful"
    )
    assert cadence_check["passed"] is True
    assert cadence_check["details"] == {"unsupported_cadences": []}


def test_blog_export_still_blocks_unquoted_cadence_with_source_cadence_present() -> None:
    context = _blog_context()
    context["faq_questions"] = [
        "Can I schedule a weekly attribution report export for finance?"
    ]
    export = _blog_export(
        data_context=context,
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Your team sees reporting questions weekly."
        ),
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    cadence_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_cadence_truthful"
    )
    assert cadence_check["passed"] is False
    assert cadence_check["details"] == {"unsupported_cadences": ["weekly"]}


def test_landing_export_fails_unsupported_uploaded_ticket_cadence() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(
            text_override=(
                "The uploaded 2 support tickets show account and reporting "
                "questions your team answers weekly."
            )
        ),
        output="landing_page",
    )

    assert result["ok"] is False
    cadence_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_cadence_truthful"
    )
    assert cadence_check["passed"] is False
    assert cadence_check["details"] == {"unsupported_cadences": ["weekly"]}


def test_blog_export_allows_cadence_when_source_period_supplies_it() -> None:
    context = _blog_context()
    context["source_period"] = "Last 90 days of support tickets"
    context["review_period"] = "last 90 days"
    export = _blog_export(
        data_context=context,
        content_override=(
            "In the last 90 days, the 2 support tickets show account and "
            "reporting questions your team sees weekly."
        ),
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    cadence_check = next(
        check for check in result["checks"]
        if check["name"] == "uploaded_ticket_cadence_truthful"
    )
    assert cadence_check["passed"] is True
    assert cadence_check["details"] == {"applicable": False}


def test_blog_export_fails_unsupported_percentage_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "These FAQ answers can reduce support volume by 20-40%."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    assert any(
        "percentage claims not backed" in error
        for error in result["errors"]
    )
    percent_check = next(
        check for check in result["checks"]
        if check["name"] == "percentage_claims_source_backed"
    )
    assert percent_check["passed"] is False
    assert percent_check["details"]["unsupported"] == ["20-40%"]


def test_blog_export_fails_unsupported_em_dash_percentage_claims() -> None:
    claim = "20\u201440%"
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            f"These FAQ answers can reduce support volume by {claim}."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    percent_check = next(
        check for check in result["checks"]
        if check["name"] == "percentage_claims_source_backed"
    )
    assert percent_check["passed"] is False
    assert percent_check["details"]["unsupported"] == [claim]


def test_landing_export_fails_unsupported_percentage_claims() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(
            text_override=(
                "The uploaded 2 support tickets show account and reporting "
                "questions, and the FAQ can cut repeat tickets by 30-50%."
            )
        ),
        output="landing_page",
    )

    assert result["ok"] is False
    percent_check = next(
        check for check in result["checks"]
        if check["name"] == "percentage_claims_source_backed"
    )
    assert percent_check["passed"] is False
    assert percent_check["details"]["unsupported"] == ["30-50%"]


def test_blog_export_allows_percentage_claim_derived_from_source_counts() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "That means 50% of included tickets mention account issues."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    percent_check = next(
        check for check in result["checks"]
        if check["name"] == "percentage_claims_source_backed"
    )
    assert percent_check["details"]["unsupported"] == []


def test_landing_export_fails_guaranteed_support_outcome_claims() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(
            text_override=(
                "The uploaded 2 support tickets show account and reporting "
                "questions. Support tickets for those questions drop after "
                "publishing. Answering these two questions will reduce incoming "
                "support tickets immediately."
            )
        ),
        output="landing_page",
    )

    assert result["ok"] is False
    assert any(
        "guarantees support-ticket or customer outcomes" in error
        for error in result["errors"]
    )
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "Support tickets for those questions drop after publishing.",
        (
            "Answering these two questions will reduce incoming support tickets "
            "immediately."
        ),
    ]


def test_blog_export_fails_future_support_prevention_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Writing one clear FAQ entry now prevents future support tickets. "
            "The support queue will shrink after publishing."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "Writing one clear FAQ entry now prevents future support tickets.",
        "The support queue will shrink after publishing.",
    ]


def test_blog_export_fails_customer_retention_outcome_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Customers who find answers in your FAQ stay longer and churn less."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "Customers who find answers in your FAQ stay longer and churn less."
    ]


def test_blog_export_fails_soft_customer_retention_outcome_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Customers who find answers quickly are more likely to stay."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "Customers who find answers quickly are more likely to stay."
    ]


def test_blog_export_fails_support_volume_outcome_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Reduce support volume: customers find answers without opening a ticket."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "Reduce support volume: customers find answers without opening a ticket."
    ]


def test_blog_export_fails_live_smoke_soft_outcome_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "This is the most efficient way to reduce repeat support tickets. "
            "Unresolved reporting issues may delay customer decisions or trigger "
            "churn at renewal time. "
            "If customers are finding the FAQ and solving their own problems, "
            "you'll see fewer tickets on that topic. "
            "For a small support team, this matters because it frees up capacity. "
            "Customers who find them should resolve their issues faster than "
            "customers who open support tickets. "
            "Your help content helps future customers find answers faster. "
            "When a customer can answer their own question in 30 seconds instead "
            "of waiting for support, they are more likely to stay, upgrade, "
            "and recommend you. "
            "These patterns indicate where FAQ entries would help customers find "
            "answers without opening a support ticket. "
            "Every FAQ entry that answers a question before a customer opens a "
            "ticket can help free the support team. "
            "A small team cannot scale by hiring more support staff; they scale "
            "by reducing the number of tickets that need human attention. "
            "Support teams spend countless hours answering the same questions. "
            "This process typically takes 2-3 minutes and requires no support "
            "intervention. "
            "The cluster is an opportunity to deflect future tickets with a "
            "single FAQ entry. "
            "This builds a help center that reduces support load. "
            "A single FAQ entry could resolve this issue for multiple users. "
            "The FAQ could have provided an immediate answer. "
            "Customers can find help without opening a support ticket. "
            "Future customers can find the solution without opening a support "
            "ticket. "
            "Customers can find the answer in your help center without opening "
            "a ticket. "
            "The workflow reduces the volume of repeat support interactions. "
            "Support teams spend thousands of hours answering repeat questions. "
            "Uploaded support tickets reveal these patterns instantly. "
            "The update takes effect immediately after verification. "
            "This frees up support capacity for more complex issues. "
            "This issue can create potential churn and affect account retention."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "This is the most efficient way to reduce repeat support tickets.",
        (
            "Unresolved reporting issues may delay customer decisions or trigger "
            "churn at renewal time."
        ),
        (
            "If customers are finding the FAQ and solving their own problems, "
            "you'll see fewer tickets on that topic."
        ),
        "For a small support team, this matters because it frees up capacity.",
        (
            "Customers who find them should resolve their issues faster than "
            "customers who open support tickets."
        ),
        "Your help content helps future customers find answers faster.",
        (
            "When a customer can answer their own question in 30 seconds instead "
            "of waiting for support, they are more likely to stay, upgrade, "
            "and recommend you."
        ),
        (
            "These patterns indicate where FAQ entries would help customers find "
            "answers without opening a support ticket."
        ),
        (
            "Every FAQ entry that answers a question before a customer opens a "
            "ticket can help free the support team."
        ),
        (
            "A small team cannot scale by hiring more support staff; they scale "
            "by reducing the number of tickets that need human attention."
        ),
        "Support teams spend countless hours answering the same questions.",
        (
            "This process typically takes 2-3 minutes and requires no support "
            "intervention."
        ),
        (
            "The cluster is an opportunity to deflect future tickets with a "
            "single FAQ entry."
        ),
        "This builds a help center that reduces support load.",
        "A single FAQ entry could resolve this issue for multiple users.",
        "The FAQ could have provided an immediate answer.",
        "Customers can find help without opening a support ticket.",
        (
            "Future customers can find the solution without opening a support "
            "ticket."
        ),
        (
            "Customers can find the answer in your help center without opening "
            "a ticket."
        ),
        "The workflow reduces the volume of repeat support interactions.",
        "Support teams spend thousands of hours answering repeat questions.",
        "Uploaded support tickets reveal these patterns instantly.",
        "The update takes effect immediately after verification.",
        "This frees up support capacity for more complex issues.",
    ]


def test_blog_export_fails_evidence_contract_live_false_green_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 4 support tickets show email and reporting clusters. "
            "The result: fewer repeat tickets, faster resolution for customers, "
            "and a documented knowledge base your team can trust. "
            "The clusters that appear most often in your support inbox are the "
            "ones that will reduce the most repeat work when you answer them "
            "in a help center. "
            "If 2 customers asked the same question in your uploaded tickets, "
            "more customers have asked it before, and more will ask it in the "
            "future. "
            "Each FAQ entry you publish is one fewer repeat ticket your team "
            "has to answer. "
            "Over time, that adds up to real time savings and a better customer "
            "experience. "
            "Document them, publish them, and watch your repeat ticket volume drop."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        (
            "The result: fewer repeat tickets, faster resolution for customers, "
            "and a documented knowledge base your team can trust."
        ),
        (
            "The clusters that appear most often in your support inbox are the "
            "ones that will reduce the most repeat work when you answer them "
            "in a help center."
        ),
        (
            "If 2 customers asked the same question in your uploaded tickets, "
            "more customers have asked it before, and more will ask it in the "
            "future."
        ),
        (
            "Each FAQ entry you publish is one fewer repeat ticket your team "
            "has to answer."
        ),
        (
            "Over time, that adds up to real time savings and a better customer "
            "experience."
        ),
        "Document them, publish them, and watch your repeat ticket volume drop.",
    ]


def test_blog_export_fails_second_live_false_green_outcome_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 4 support tickets show email and reporting clusters. "
            "A support ticket FAQ strategy turns these repeated questions into "
            "self-service answers that help customers and reduce support workload. "
            "Every support ticket that could have been answered by a self-service "
            "FAQ represents time your team is not spending on complex, one-off issues. "
            "Your customers will find answers faster. "
            "Your support team will focus on complex issues that require human judgment."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        (
            "A support ticket FAQ strategy turns these repeated questions into "
            "self-service answers that help customers and reduce support workload."
        ),
        (
            "Every support ticket that could have been answered by a self-service "
            "FAQ represents time your team is not spending on complex, one-off issues."
        ),
        "Your customers will find answers faster.",
        "Your support team will focus on complex issues that require human judgment.",
    ]


def test_blog_export_fails_third_live_false_green_benefit_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 4 support tickets show email and reporting clusters. "
            "New customers will likely ask the same question. "
            "Customers can find answers in the FAQ instead of opening a ticket, "
            "freeing the team to focus on complex issues. "
            "Customers get instant answers instead of waiting for support to respond. "
            "The goal is to make your help center so complete that customers can "
            "answer their own questions."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "New customers will likely ask the same question.",
        (
            "Customers can find answers in the FAQ instead of opening a ticket, "
            "freeing the team to focus on complex issues."
        ),
        "Customers get instant answers instead of waiting for support to respond.",
        (
            "The goal is to make your help center so complete that customers can "
            "answer their own questions."
        ),
    ]


def test_blog_export_fails_saas_demo_false_green_outcome_claims() -> None:
    claims = [
        (
            "Each repeated question represents a support ticket that could have "
            "been prevented by a clear FAQ entry."
        ),
        (
            "A small FAQ effort could address two-thirds of your incoming "
            "ticket volume."
        ),
        (
            "Each FAQ entry you publish has the potential to address multiple "
            "customer questions and reduce the number of tickets your team "
            "must handle manually."
        ),
        (
            "Each FAQ entry you publish represents a question your support team "
            "will no longer need to answer repeatedly."
        ),
        (
            "Each repeat question is an opportunity to move the answer into a "
            "customer-facing FAQ and reduce support ticket volume by enabling "
            "self-service."
        ),
        (
            "These gaps in your customer-facing documentation are driving the "
            "most inbound support volume."
        ),
        (
            "A step-by-step SSO setup guide becomes a self-service resource "
            "that reduces the need for support team intervention on "
            "configuration questions."
        ),
        (
            "A single FAQ entry can help your team handle future instances of "
            "the same question more efficiently."
        ),
        (
            "Addressing them will likely reduce your repeat-question volume "
            "significantly, freeing your team to focus on complex issues."
        ),
        (
            "A complete answer helps customers resolve their issue without "
            "follow-up questions."
        ),
        "Answering one export question in an FAQ could prevent four similar tickets from arriving.",
        "Addressing these issues will reduce your repeat-ticket volume.",
        "FAQ titles and descriptions help customers find answers using the same phrasing.",
        "Monitor your support queue to see whether the same questions stop appearing.",
        "After 2-4 weeks, compare repeat ticket volume before and after publication.",
        "Track FAQ page traffic and support ticket resolution time by cluster.",
        (
            "When you preserve the customer's own words in your FAQ entries, "
            "future customers will recognize their situation and find the "
            "answer faster."
        ),
        (
            "When the same support ticket FAQ question appears 4 times in a "
            "36-ticket sample, it signals the question is being asked much more "
            "frequently across your full support queue."
        ),
        (
            "These are configuration and integration questions typically asked "
            "once per new customer."
        ),
        "These setup questions are asked by every customer at some point.",
    ]
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            + " ".join(claims)
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == claims


def test_blog_export_fails_procedural_answer_steps_without_resolution_evidence() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Simple answers like Go to Settings > Email > Change Email are "
            "faster to document and publish."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    answer_step_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_answer_steps_grounded"
    )
    assert answer_step_check["passed"] is False
    assert answer_step_check["details"]["unsupported_answer_steps"] == [
        (
            "Simple answers like Go to Settings > Email > Change Email are "
            "faster to document and publish."
        )
    ]


def test_blog_export_fails_procedural_answer_step_lead_ins_without_resolution_evidence() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Then click the Settings menu. "
            "First, go to the billing dashboard. "
            "Next, select the Export button. "
            "To fix this, go to the Settings menu. "
            "1. Click the Export tab."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    answer_step_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_answer_steps_grounded"
    )
    assert answer_step_check["passed"] is False
    assert answer_step_check["details"]["unsupported_answer_steps"] == [
        "Then click the Settings menu.",
        "First, go to the billing dashboard.",
        "Next, select the Export button.",
        "To fix this, go to the Settings menu.",
        "Click the Export tab.",
    ]


def test_blog_export_allows_non_ui_greater_than_comparisons() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Revenue > Costs is a basic operating equation. "
            "G2 > Capterra is just a preference statement here."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    answer_step_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_answer_steps_grounded"
    )
    assert answer_step_check["passed"] is True
    assert answer_step_check["details"] == {"unsupported_answer_steps": []}


def test_blog_export_allows_descriptive_support_ticket_draft_language() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Teams with fewer tickets tend to onboard faster, but this upload "
            "does not prove that publishing an FAQ will create that result. "
            "Customers often go to billing questions first when they cannot "
            "find a plain-language answer. "
            "You can export your data from most analytics tools. "
            "That sentence is general background, not a verified answer for "
            "the uploaded product. "
            "Faster resolution for customers is a goal to track after publishing, "
            "not an outcome proven by these tickets."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is True
    answer_step_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_answer_steps_grounded"
    )
    assert answer_step_check["passed"] is True


def test_blog_export_fails_unverified_answer_capabilities_without_resolution_evidence() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Your support team has confirmed this is possible, but the exact "
            "steps depend on your account type and permissions. "
            "Campaign attribution data can be exported from the reporting dashboard."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    answer_step_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_answer_steps_grounded"
    )
    assert answer_step_check["passed"] is False
    assert answer_step_check["details"]["unsupported_answer_steps"] == [
        (
            "Your support team has confirmed this is possible, but the exact "
            "steps depend on your account type and permissions."
        ),
        "Campaign attribution data can be exported from the reporting dashboard.",
    ]


def test_blog_export_fails_likely_resolution_paths_without_resolution_evidence() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "For the email and profile updates cluster, the resolution is likely "
            "straightforward: navigate to account settings, locate the email field, "
            "update it, and confirm."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    answer_step_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_answer_steps_grounded"
    )
    assert answer_step_check["passed"] is False
    assert answer_step_check["details"]["unsupported_answer_steps"] == [
        (
            "For the email and profile updates cluster, the resolution is likely "
            "straightforward: navigate to account settings, locate the email field, "
            "update it, and confirm."
        )
    ]


def test_blog_export_allows_procedural_answer_steps_with_resolution_evidence() -> None:
    context = _blog_context()
    context["support_ticket_resolution_evidence_present"] = True
    context["support_ticket_resolution_evidence_count"] = 1
    export = _blog_export(
        data_context=context,
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "The verified resolution says to go to Settings > Email > Change Email."
        ),
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    answer_step_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_answer_steps_grounded"
    )
    assert answer_step_check["passed"] is True
    assert answer_step_check["details"] == {"applicable": False}


def test_blog_export_allows_churn_retention_context_and_disclaimers() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "We cannot promise this will reduce churn or improve retention. "
            "Account retention is influenced by many factors beyond docs. "
            "Slow support responses are a well-known churn driver in SaaS. "
            "This issue can create potential churn and affect account retention."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is True
    assert outcome_check["details"] == {"unsupported_claims": []}


def test_blog_export_fails_direct_retention_improvement_claims() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Publishing these answers improves account retention."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "Publishing these answers improves account retention."
    ]


def test_blog_export_fails_claims_with_contrastive_or_guidance_phrasing() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Publishing answers reduces support volume rather than increasing it. "
            "This will reduce churn rather than ignore it. "
            "Use cautious language: support tickets will drop after publishing."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        "Publishing answers reduces support volume rather than increasing it.",
        "This will reduce churn rather than ignore it.",
        "Use cautious language: support tickets will drop after publishing.",
    ]


def test_blog_export_fails_claims_after_disclaimer_contrast() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "We cannot promise this will reduce churn, but publishing answers "
            "reduces support volume. "
            "This is not a guarantee; support tickets will drop after publishing."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is False
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is False
    assert outcome_check["details"]["unsupported_claims"] == [
        (
            "We cannot promise this will reduce churn, but publishing answers "
            "reduces support volume."
        ),
        "This is not a guarantee; support tickets will drop after publishing.",
    ]


def test_blog_export_allows_cautious_support_outcome_language() -> None:
    export = _blog_export(
        content_override=(
            "The uploaded 2 support tickets show account and reporting clusters. "
            "Publishing FAQ answers can help reduce repeat support tickets. "
            "Track whether ticket volume for that topic drops after publication."
        )
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    outcome_check = next(
        check for check in result["checks"]
        if check["name"] == "support_ticket_outcome_claims_grounded"
    )
    assert outcome_check["passed"] is True
    assert outcome_check["details"] == {"unsupported_claims": []}


def test_blog_export_allows_whole_percent_rounding_from_source_counts() -> None:
    context = _blog_context()
    context["source_row_count"] = 3
    context["included_ticket_row_count"] = 3
    context["question_like_ticket_count"] = 3
    context["top_clusters"] = [{"label": "account", "count": 1}]
    export = _blog_export(
        data_context=context,
        content_override=(
            "The uploaded 3 support tickets show account questions. "
            "That means 33% of included tickets mention account issues."
        ),
    )

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="blog_post",
    )

    assert result["ok"] is True
    percent_check = next(
        check for check in result["checks"]
        if check["name"] == "percentage_claims_source_backed"
    )
    assert percent_check["details"]["source_backed_percentages"] == [33, 100, 300]
    assert percent_check["details"]["unsupported"] == []


def test_landing_export_fails_when_source_context_missing() -> None:
    export = _landing_export(source_context=None)
    export["rows"][0]["metadata"] = {}

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="landing_page",
    )

    assert result["ok"] is False
    assert result["errors"] == [
        "export row is missing support-ticket source context"
    ]


def test_landing_export_fails_when_generated_text_is_blank() -> None:
    export = _landing_export()
    row = export["rows"][0]
    row["title"] = ""
    row["hero"] = {"headline": " ", "subheadline": "\n"}
    row["sections"] = []
    row["cta"] = {"label": ""}
    row["meta"] = {"description": ""}

    result = evaluator.evaluate_support_ticket_generated_content(
        export,
        output="landing_page",
    )

    assert result["ok"] is False
    assert "export row did not contain generated text fields" in result["errors"]
    text_check = next(
        check for check in result["checks"]
        if check["name"] == "generated_text_present"
    )
    assert text_check["passed"] is False


def test_landing_export_fails_on_stale_benchmark_numbers() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(
            text_override=(
                "Across 186 support tickets, 78 repeat questions created a "
                "42% FAQ gap around account issues."
            )
        ),
        output="landing_page",
    )

    assert result["ok"] is False
    assert (
        "generated text contains stale benchmark numbers not present in "
        "source context: 186, 78, 42%"
    ) in result["errors"]


def test_landing_export_does_not_match_stale_number_inside_larger_number() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(
            text_override=(
                "Across 2186 support-ticket records, account questions still "
                "show where customers need FAQ answers."
            )
        ),
        output="landing_page",
    )

    assert result["ok"] is True
    assert not any("stale benchmark numbers" in error for error in result["errors"])


def test_landing_export_context_larger_number_does_not_authorize_stale_number() -> None:
    context = _source_context()
    context["source_row_count"] = 2186

    result = evaluator.evaluate_support_ticket_generated_content(
        _landing_export(
            source_context=context,
            text_override=(
                "Across 186 support tickets, account questions still show "
                "where customers need FAQ answers."
            ),
        ),
        output="landing_page",
    )

    assert result["ok"] is False
    assert (
        "generated text contains stale benchmark numbers not present in "
        "source context: 186"
    ) in result["errors"]


def test_blog_export_fails_when_no_source_signal_is_visible() -> None:
    result = evaluator.evaluate_support_ticket_generated_content(
        _blog_export(
            content_override=(
                "## Support ticket FAQ\n\n"
                "This article discusses answers in general without naming "
                "the observed ticket issues."
            )
        ),
        output="blog_post",
    )

    assert result["ok"] is False
    assert (
        "generated text does not mention any observed ticket cluster, "
        "customer question, or customer wording example"
    ) in result["errors"]


def test_cli_returns_nonzero_for_failing_export(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "landing-export.json"
    export = _landing_export(text_override="Generic marketing copy.")
    export["rows"][0]["title"] = "Generic marketing page"
    export["rows"][0]["hero"]["subheadline"] = "A broad page about operations."
    export["rows"][0]["sections"] = [{"heading": "Overview", "body": "General advice."}]
    export["rows"][0]["cta"] = {"label": "Learn more"}
    export["rows"][0]["meta"] = {"description": "General operations page."}
    path.write_text(json.dumps(export), encoding="utf-8")

    code = evaluator.main([str(path), "--output", "landing_page"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["ok"] is False
    assert any(
        "support-ticket, FAQ, help-center, or answer framing" in error
        for error in payload["errors"]
    )


def test_cli_returns_zero_for_passing_blog_export(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "blog-export.json"
    path.write_text(json.dumps(_blog_export()), encoding="utf-8")

    code = evaluator.main([str(path), "--output", "blog_post", "--pretty"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["ok"] is True
