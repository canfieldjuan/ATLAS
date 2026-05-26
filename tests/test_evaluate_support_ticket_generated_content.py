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
