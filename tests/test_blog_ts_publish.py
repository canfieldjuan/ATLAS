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
