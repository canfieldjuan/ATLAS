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


def test_quality_gate_emits_and_enforces_score_threshold(monkeypatch):
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "Mailchimp pricing climbed too quickly for our team"},
            {"phrase": "Intercom onboarding helped us move faster"},
        ]
    )
    filler = (
        "Mailchimp and Intercom are compared using reviewer sentiment signals "
        "from public software reviews. "
    ) * 130
    content = {
        "title": "Short title",
        "description": "desc",
        "content": f"""
## Introduction
Mailchimp and Intercom are compared using reviewer sentiment signals.
{filler}
{{{{chart:head2head-bar}}}}
> "Mailchimp pricing climbed too quickly for our team" -- reviewer on Reddit
> "Intercom onboarding helped us move faster" -- reviewer on G2
""",
    }

    monkeypatch.setattr(blog_mod.settings.b2b_churn, "blog_quality_pass_score", 95, raising=False)

    _, report = _apply_blog_quality_gate(blueprint, content)

    assert report["threshold"] == 95
    assert report["score"] < 95
    assert report["status"] == "fail"


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


def test_quality_gate_drops_quotes_from_unexpected_vendors():
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "Mailchimp pricing climbed too quickly for our team", "vendor": "Mailchimp"},
            {"phrase": "Zendesk costs kept rising without warning", "vendor": "Zendesk"},
            {"phrase": "Intercom onboarding helped us move faster", "vendor": "Intercom"},
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
This analysis is based on self-selected reviewers from 2026-02 to 2026-03.
Mailchimp and Intercom are compared using reviewer signals.
{_pad}
{{{{chart:head2head-bar}}}}

> "Mailchimp pricing climbed too quickly for our team" -- reviewer on Reddit
> "Zendesk costs kept rising without warning" -- reviewer on G2
> "Intercom onboarding helped us move faster" -- reviewer on G2
""",
    }

    cleaned, report = _apply_blog_quality_gate(blueprint, content)
    assert report["status"] == "pass"
    assert "Zendesk costs kept rising without warning" not in cleaned["content"]
    assert any(item.startswith("removed_unmatched_quotes:") for item in report["fixes_applied"])


def test_quality_gate_blocks_unsupported_category_outcome_assertion():
    blueprint = _build_blueprint(
        quotes=[
            {"phrase": "Mailchimp pricing climbed too quickly for our team", "vendor": "Mailchimp"},
            {"phrase": "Intercom onboarding helped us move faster", "vendor": "Intercom"},
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
This analysis is based on self-selected reviewers from 2026-02 to 2026-03.
Mailchimp and Intercom are compared using reviewer signals.
{_pad}
{{{{chart:head2head-bar}}}}

> "Mailchimp pricing climbed too quickly for our team" -- reviewer on Reddit
> "Intercom onboarding helped us move faster" -- reviewer on G2

The category winner in the data is HubSpot.
""",
    }

    _, report = _apply_blog_quality_gate(blueprint, content)
    assert report["status"] == "fail"
    assert "unsupported_category_outcome_assertion" in (report["blocking_issues"] or [])


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


# ---------------------------------------------------------------------------
# Unsupported data claim tests
# ---------------------------------------------------------------------------

_CLAIM_PAD = (
    "Mailchimp and Intercom are compared using reviewer sentiment signals "
    "and switching context from public software reviews across platforms. "
) * 120


def _migration_blueprint() -> PostBlueprint:
    """Blueprint with migration chart grounding WooCommerce and BigCommerce."""
    return PostBlueprint(
        topic_type="migration_guide",
        slug="switch-to-shopify-2026-03",
        suggested_title="Migration Guide: Why Teams Are Switching to Shopify",
        tags=["shopify", "migration"],
        data_context={
            "vendor": "Shopify",
            "review_period": "2026-02 to 2026-03",
        },
        sections=[],
        charts=[
            ChartSpec(
                chart_id="sources-bar",
                chart_type="horizontal_bar",
                title="Where Shopify Users Come From",
                data=[
                    {"name": "WooCommerce", "migrations": 5},
                    {"name": "BigCommerce", "migrations": 3},
                ],
            ),
        ],
        quotable_phrases=[
            {"phrase": "We switched from WooCommerce after hitting scaling limits"},
            {"phrase": "BigCommerce pricing pushed us to Shopify"},
        ],
    )


def test_data_claim_flags_ungrounded_vendor():
    """Magento is not in chart data -- claiming it as a top source is unsupported.
    Single-word vendors are caught via known-vendor lookup, not regex."""
    bp = _migration_blueprint()
    bp.data_context["_known_vendors"] = ["Magento", "Squarespace", "Shopify"]
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "The top migration sources include Magento and Squarespace.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "BigCommerce pricing pushed us to Shopify" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert any("unsupported_data_claim" in w for w in report.get("warnings", []))


def test_data_claim_allows_contextual_mention_without_claim_marker():
    """BigCommerce mentioned without a claim marker should not flag."""
    bp = _migration_blueprint()
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "Shopify competes with Squarespace in the e-commerce space.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "BigCommerce pricing pushed us to Shopify" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert not any("unsupported_data_claim" in w for w in report.get("warnings", []))


def test_data_claim_allows_grounded_vendor_in_claim_sentence():
    """WooCommerce IS in chart data -- claiming it as top source is fine."""
    bp = _migration_blueprint()
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "The top migration source is WooCommerce with 5 stories analyzed.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "BigCommerce pricing pushed us to Shopify" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert not any("unsupported_data_claim" in w for w in report.get("warnings", []))


def test_data_claim_allows_topic_vendor_in_claim():
    """Shopify is the topic vendor -- mentioning it in data claims is fine."""
    bp = _migration_blueprint()
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "Data shows Shopify attracts 93 switching stories from competitors.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "BigCommerce pricing pushed us to Shopify" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert not any("unsupported_data_claim" in w for w in report.get("warnings", []))


def test_data_claim_flags_ungrounded_camelcase_vendor():
    """CamelCase vendors like BigCommerce should still be detected when ungrounded."""
    bp = _migration_blueprint()
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "The top migration source is QuickBooks POS for fast-moving retail teams.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "BigCommerce pricing pushed us to Shopify" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert any("unsupported_data_claim:QuickBooks POS" in w for w in report.get("warnings", []))


def test_data_claim_allows_to_vendor_grounding():
    """Destination-only topic vendor keys should still ground supported claims."""
    bp = PostBlueprint(
        topic_type="migration_guide",
        slug="switch-to-shopify-2026-03",
        suggested_title="Migration Guide: Why Teams Are Switching to Shopify",
        tags=["shopify", "migration"],
        data_context={
            "to_vendor": "Shopify",
            "review_period": "2026-02 to 2026-03",
        },
        sections=[],
        charts=[
            ChartSpec(
                chart_id="sources-bar",
                chart_type="horizontal_bar",
                title="Where Shopify Users Come From",
                data=[{"name": "WooCommerce", "migrations": 5}],
            )
        ],
        quotable_phrases=[
            {"phrase": "We switched from WooCommerce after hitting scaling limits"},
            {"phrase": "Shopify setup was faster than our old stack"},
        ],
    )
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "Data shows Shopify attracts 93 switching stories from competitors.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "Shopify setup was faster than our old stack" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert not any("unsupported_data_claim" in w for w in report.get("warnings", []))


# ---------------------------------------------------------------------------
# Known-vendor DB lookup path tests
# ---------------------------------------------------------------------------

from atlas_brain.autonomous.tasks.b2b_blog_post_generation import (
    _build_grounded_vendor_set,
    _find_unsupported_data_claims,
)


def test_known_vendor_lookup_flags_ungrounded_known_vendor():
    """When _known_vendors includes Magento, it should be flagged via exact
    name match even if the regex fallback would miss it."""
    bp = _migration_blueprint()
    grounded = _build_grounded_vendor_set(bp)
    body = (
        f"{_CLAIM_PAD}\n"
        "The top migration source is magento for mid-market teams.\n"
    )
    # Lowercase "magento" -- regex fallback would miss it (no capital),
    # but known-vendor lookup matches case-insensitively.
    flagged = _find_unsupported_data_claims(
        body, grounded, known_vendors=["Magento", "Salesforce"],
    )
    assert any("Magento" in f for f in flagged)


def test_known_vendor_lookup_skips_grounded_vendor():
    """A known vendor that IS in the grounded set should not be flagged."""
    bp = _migration_blueprint()
    grounded = _build_grounded_vendor_set(bp)
    body = (
        f"{_CLAIM_PAD}\n"
        "The top migration source is WooCommerce with 5 stories analyzed.\n"
    )
    flagged = _find_unsupported_data_claims(
        body, grounded, known_vendors=["WooCommerce", "Magento"],
    )
    assert not flagged


def test_known_vendor_lookup_empty_list_falls_back_to_regex_for_multi_word():
    """With no known vendors, the regex fallback catches multi-word ungrounded
    names. Single-word names require the known-vendor path."""
    bp = _migration_blueprint()
    grounded = _build_grounded_vendor_set(bp)
    body = (
        f"{_CLAIM_PAD}\n"
        "The top migration sources include Palo Alto Networks and Oracle Cloud.\n"
    )
    flagged = _find_unsupported_data_claims(body, grounded, known_vendors=[])
    assert any("Palo Alto" in f or "Oracle Cloud" in f for f in flagged)


# ---------------------------------------------------------------------------
# Chart-scope ambiguity tests
# ---------------------------------------------------------------------------


def test_chart_scope_ambiguity_flags_non_chart_vendor_in_strongest_claim():
    """BigCommerce is grounded (in chart) but Magento is not -- using
    'most common source' with Magento should trigger chart_scope_ambiguity."""
    bp = _migration_blueprint()
    bp.data_context["_known_vendors"] = ["Magento", "BigCommerce", "Shopify"]
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "The most common source of migrations is Magento among mid-market teams.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "BigCommerce pricing pushed us to Shopify" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert any("chart_scope_ambiguity" in w for w in report.get("warnings", []))


def test_chart_scope_ambiguity_allows_chart_vendor_in_strongest_claim():
    """WooCommerce IS in the chart -- 'most common source is WooCommerce' is fine."""
    bp = _migration_blueprint()
    bp.data_context["_known_vendors"] = ["Magento", "WooCommerce", "Shopify"]
    content = {
        "title": "Switch to Shopify",
        "description": "desc",
        "content": (
            "## Introduction\n"
            "This analysis is based on self-selected reviewers from 2026-02 to 2026-03.\n"
            f"{_CLAIM_PAD}\n"
            "The most common source of migrations is WooCommerce based on charted data.\n"
            "{{chart:sources-bar}}\n"
            '> "We switched from WooCommerce after hitting scaling limits" -- reviewer\n'
            '> "BigCommerce pricing pushed us to Shopify" -- reviewer\n'
        ),
    }
    _, report = _apply_blog_quality_gate(bp, content)
    assert not any("chart_scope_ambiguity" in w for w in report.get("warnings", []))
