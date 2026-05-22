from datetime import datetime, timezone

from atlas_brain.api.blog_admin import _write_blog_ts_file
from atlas_brain.autonomous.tasks._blog_ts import build_post_ts


def test_build_post_ts_serializes_cta():
    _, ts_content = build_post_ts(
        slug="switch-to-shopify-2026-03",
        title="Migration Guide: Why Teams Are Switching to Shopify",
        description="Shopify migration guide",
        date_str="2026-03-30",
        author="Churn Signals Team",
        tags=["shopify", "migration"],
        topic_type="migration_guide",
        charts_json=[],
        content="# Heading",
        data_context={"affiliate_url": "https://example.com/shopify"},
        cta={
            "headline": "See the full migration brief",
            "body": "Get the full report before renewal.",
            "button_text": "Book briefing",
            "report_type": "migration_guide",
        },
    )

    assert "author: 'Churn Signals Team'" in ts_content
    assert "cta:" in ts_content
    assert "See the full migration brief" in ts_content


def test_build_post_ts_converts_tight_bullet_list():
    # A bullet list tightly coupled to a label (no blank line) must render as
    # a real <ul>, not literal "- item" text inside a <p> (the D9 bug).
    _, ts_content = build_post_ts(
        slug="x-deep-dive-2026-04",
        title="X Deep Dive",
        description="d",
        date_str="2026-04-01",
        author="Churn Signals Team",
        tags=["x"],
        topic_type="vendor_deep_dive",
        charts_json=[],
        content="**Top pain points:**\n- Pricing\n- Support",
        data_context={},
    )
    assert "<ul>" in ts_content
    assert "<li>Pricing</li>" in ts_content
    assert "<li>Support</li>" in ts_content
    # The broken markdown-in-<p> shape must not appear.
    assert "<p><strong>Top pain points:</strong>\n- Pricing" not in ts_content


def test_write_blog_ts_file_uses_churn_signals_team_and_persists_cta(tmp_path):
    blog_dir = tmp_path / "blog"
    blog_dir.mkdir()
    (blog_dir / "index.ts").write_text(
        "import existingPost from './existing-post'\n\nexport const POSTS = [\n  existingPost,\n]\n",
        encoding="utf-8",
    )

    row = {
        "slug": "switch-to-shopify-2026-03",
        "title": "Migration Guide: Why Teams Are Switching to Shopify",
        "description": "Shopify migration guide",
        "charts": [],
        "tags": ["shopify", "migration"],
        "topic_type": "migration_guide",
        "content": "# Heading",
        "data_context": {"affiliate_url": "https://example.com/shopify"},
        "seo_title": "SEO title",
        "seo_description": "SEO desc",
        "target_keyword": "switch to shopify",
        "secondary_keywords": ["shopify migration"],
        "faq": [],
        "related_slugs": [],
        "cta": {
            "headline": "See the full migration brief",
            "body": "Get the full report before renewal.",
            "button_text": "Book briefing",
            "report_type": "migration_guide",
        },
    }

    file_path = _write_blog_ts_file(row, str(blog_dir), datetime(2026, 3, 30, tzinfo=timezone.utc))

    assert file_path is not None
    written = (blog_dir / "switch-to-shopify-2026-03.ts").read_text(encoding="utf-8")
    assert "author: 'Churn Signals Team'" in written
    assert "cta:" in written
    assert "See the full migration brief" in written
