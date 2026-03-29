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
    _pad = (
        "Mailchimp and Intercom are compared using reviewer sentiment signals "
        "and switching context from public software reviews across platforms. "
    ) * 120
    content = {
        "title": "Mailchimp vs Intercom",
        "description": "desc",
        "content": f"""
## Introduction
Answer: This analysis is based on self-selected reviewers from 2026-02 to 2026-03.
Mailchimp and Intercom are compared using reviewer signals.
{_pad}
{{{{chart:head2head-bar}}}}

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
    ) * 120
    return f"""
## Introduction
This comparison focuses on reviewer sentiment signals and churn intent patterns.
{filler}
{{{{chart:head2head-bar}}}}
> "Mailchimp pricing climbed too quickly for our team" -- reviewer on Reddit
> "Intercom onboarding helped us move faster" -- reviewer on G2
"""


def test_enforce_quality_auto_injects_methodology(monkeypatch):
    """_ensure_methodology_context auto-adds the methodology note when missing,
    so the quality gate should pass without needing a retry."""
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
        return raw

    monkeypatch.setattr(blog_mod, "_generate_content", _fake_retry_generate)

    final, report = _enforce_blog_quality(
        llm=object(),
        blueprint=blueprint,
        content=raw,
        max_tokens=1200,
        related_posts=None,
    )

    # No retry needed -- methodology auto-injected
    assert calls["count"] == 0
    assert final is not None
    assert report["status"] == "pass"
    assert "self-selected" in str(final.get("content") or "").lower()
    assert "2026-02 to 2026-03" in str(final.get("content") or "")


def test_enforce_quality_fails_when_blocking_issues_persist_after_retry(monkeypatch):
    """When the retry still produces content with blocking issues, the gate
    should return None."""
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "Mailchimp pricing climbed too quickly for our team"},
            {"phrase": "Intercom onboarding helped us move faster"},
        ]
    )
    # Content that mentions Mailchimp but NOT Intercom -> blocking vendor miss
    _pad = (
        "Mailchimp is a popular email marketing platform used by many teams. "
    ) * 140
    raw = {
        "title": "Mailchimp vs Intercom",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_pad}\n"
            '{{chart:head2head-bar}}\n'
            '> "Mailchimp pricing climbed too quickly for our team" -- reviewer on Reddit\n'
            '> "Intercom onboarding helped us move faster" -- reviewer on G2\n'
        ).replace("Intercom", "Platform B"),
    }

    def _fake_retry_generate(*args, **kwargs):
        return dict(raw)

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
    assert any("missing_vendor_mentions" in issue for issue in issues)
