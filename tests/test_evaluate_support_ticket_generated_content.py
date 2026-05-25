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
