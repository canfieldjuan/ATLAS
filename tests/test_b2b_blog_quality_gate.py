import atlas_brain.autonomous.tasks.b2b_blog_post_generation as blog_mod
from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
    ChartSpec,
    PostBlueprint,
    _apply_blog_quality_gate,
    _enforce_blog_quality,
)


def _build_blueprint(quotes: list[dict] | None = None) -> PostBlueprint:
    return PostBlueprint(
        topic_type="vendor_showdown",
        slug="mailchimp-vs-intercom-2026-03",
        suggested_title="Mailchimp vs Intercom",
        tags=["mailchimp", "intercom"],
        data_context={
            "vendor_a": "Mailchimp",
            "vendor_b": "Intercom",
            "review_period": "2026-02 to 2026-03",
        },
        sections=[],
        charts=[
            ChartSpec(
                chart_id="head2head-bar",
                chart_type="bar",
                title="Head to head",
                data=[],
            )
        ],
        quotable_phrases=quotes or [],
    )


def test_quality_gate_blocks_unknown_chart_placeholder():
    blueprint = _build_blueprint()
    content = {
        "title": "Mailchimp vs Intercom",
        "description": "desc",
        "content": "## Comparison\n{{chart:wrong-chart-id}}\n",
    }
    _, report = _apply_blog_quality_gate(blueprint, content)
    assert report["status"] == "fail"
    assert any("unknown_chart_placeholders" in issue for issue in report["blocking_issues"])


def test_quality_gate_sanitizes_answer_prefix_and_drops_unsourced_quotes():
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "I just canceled my Mailchimp account after hitting plan limits"},
            {"phrase": "Intercom support solved our issue in one day"},
        ]
    )
    content = {
        "title": "Mailchimp vs Intercom",
        "description": "desc",
        "content": """
## Introduction
Answer: This analysis is based on self-selected reviewers from 2026-02 to 2026-03.
Mailchimp and Intercom are compared using reviewer signals.
{{chart:head2head-bar}}

> "I just canceled my Mailchimp account after hitting plan limits" -- reviewer on Reddit
> "Intercom support solved our issue in one day" -- reviewer on G2
> "Been using pipedrive for years with DNS errors" -- reviewer on Reddit
""",
    }

    cleaned, report = _apply_blog_quality_gate(blueprint, content)
    assert report["status"] == "pass"
    assert "Answer:" not in cleaned["content"]
    assert "pipedrive" not in cleaned["content"].lower()
    assert any(item.startswith("removed_answer_prefix:") for item in report["fixes_applied"])
    assert any(item.startswith("removed_unmatched_quotes:") for item in report["fixes_applied"])


def test_quality_gate_blocks_when_required_vendor_missing():
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "Mailchimp pricing climbed too quickly"},
            {"phrase": "Intercom onboarding is strong"},
        ]
    )
    content = {
        "title": "Mailchimp vs Intercom",
        "description": "desc",
        "content": """
## Overview
This analysis is based on self-selected reviewers from 2026-02 to 2026-03.
{{chart:head2head-bar}}
> "Mailchimp pricing climbed too quickly" -- reviewer on Reddit
> "Intercom onboarding is strong" -- reviewer on G2
""",
    }
    # Remove one required vendor name to trigger the entity gate.
    content["content"] = content["content"].replace("Intercom", "Platform B")

    _, report = _apply_blog_quality_gate(blueprint, content)
    assert report["status"] == "fail"
    assert any(issue.startswith("missing_vendor_mentions:") for issue in report["blocking_issues"])


def _raw_content_missing_methodology() -> str:
    filler = (
        "Mailchimp and Intercom are compared using reviewer sentiment signals, "
        "urgency patterns, and switching context from public software reviews. "
    ) * 14
    return f"""
## Introduction
This comparison focuses on reviewer sentiment signals and churn intent patterns.
{filler}
{{{{chart:head2head-bar}}}}
> "Mailchimp pricing climbed too quickly for our team" -- reviewer on Reddit
> "Intercom onboarding helped us move faster" -- reviewer on G2
"""


def test_enforce_quality_retries_for_critical_warnings_and_passes(monkeypatch):
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "Mailchimp pricing climbed too quickly for our team"},
            {"phrase": "Intercom onboarding helped us move faster"},
        ]
    )
    raw = {
        "title": "Mailchimp vs Intercom",
        "description": "desc",
        "content": _raw_content_missing_methodology(),
    }

    calls = {"count": 0}

    def _fake_retry_generate(*args, **kwargs):
        calls["count"] += 1
        feedback = kwargs.get("quality_feedback") or []
        assert any("self-selected" in str(item).lower() for item in feedback)
        assert any("review period" in str(item).lower() for item in feedback)
        return {
            "title": raw["title"],
            "description": raw["description"],
            "content": (
                "## Introduction\n"
                "This analysis covers 2026-02 to 2026-03 and uses self-selected reviewer feedback, "
                "so findings reflect perception signals rather than universal product truth.\n"
                + raw["content"]
            ),
        }

    monkeypatch.setattr(blog_mod, "_generate_content", _fake_retry_generate)

    final, report = _enforce_blog_quality(
        llm=object(),
        blueprint=blueprint,
        content=raw,
        max_tokens=1200,
        related_posts=None,
    )

    assert calls["count"] == 1
    assert final is not None
    assert report["status"] == "pass"
    assert "review_period_not_explicitly_mentioned" not in (report.get("warnings") or [])
    assert "methodology_disclaimer_missing_self_selected" not in (report.get("warnings") or [])
    assert "2026-02 to 2026-03" in str(final.get("content") or "")
    assert "self-selected" in str(final.get("content") or "").lower()


def test_enforce_quality_fails_when_critical_warnings_persist_after_retry(monkeypatch):
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "Mailchimp pricing climbed too quickly for our team"},
            {"phrase": "Intercom onboarding helped us move faster"},
        ]
    )
    raw = {
        "title": "Mailchimp vs Intercom",
        "description": "desc",
        "content": _raw_content_missing_methodology(),
    }

    def _fake_retry_generate(*args, **kwargs):
        return {
            "title": raw["title"],
            "description": raw["description"],
            "content": raw["content"],
        }

    monkeypatch.setattr(blog_mod, "_generate_content", _fake_retry_generate)

    final, report = _enforce_blog_quality(
        llm=object(),
        blueprint=blueprint,
        content=raw,
        max_tokens=1200,
        related_posts=None,
    )

    assert final is None
    issues = report.get("blocking_issues") or []
    assert any(
        issue == "critical_warning_unresolved:review_period_not_explicitly_mentioned"
        for issue in issues
    )
    assert any(
        issue == "critical_warning_unresolved:methodology_disclaimer_missing_self_selected"
        for issue in issues
    )
