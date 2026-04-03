"""Focused tests for B2B source governance filters."""

from types import SimpleNamespace

import atlas_brain.api.b2b_scrape as scrape_mod
import atlas_brain.autonomous.tasks._b2b_shared as shared_mod
import atlas_brain.autonomous.tasks.b2b_blog_post_generation as blog_mod
from atlas_brain.services.scraping.sources import filter_deprecated_sources


def test_filter_deprecated_sources_preserves_order():
    filtered = filter_deprecated_sources(
        ["g2", "capterra", "reddit", "software_advice", "g2"],
        "capterra,software_advice",
    )
    assert filtered == ["g2", "reddit"]


def test_scrape_current_allowed_sources_excludes_deprecated(monkeypatch):
    cfg = SimpleNamespace(
        source_allowlist="g2,capterra,software_advice,reddit",
        deprecated_sources="capterra,software_advice",
    )
    monkeypatch.setattr(scrape_mod, "settings", SimpleNamespace(b2b_scrape=cfg))

    assert scrape_mod._current_allowed_sources() == ["g2", "reddit"]


def test_intelligence_and_blog_allowlists_exclude_deprecated(monkeypatch):
    churn_cfg = SimpleNamespace(
        intelligence_source_allowlist="g2,capterra,trustradius,gartner,reddit",
        intelligence_executive_sources="g2,capterra,trustradius,gartner",
        blog_source_allowlist="g2,capterra,trustpilot,reddit",
        deprecated_review_sources="capterra,trustradius,trustpilot",
    )
    monkeypatch.setattr(shared_mod, "settings", SimpleNamespace(b2b_churn=churn_cfg))
    monkeypatch.setattr(blog_mod, "settings", SimpleNamespace(b2b_churn=churn_cfg))

    assert shared_mod._intelligence_source_allowlist() == ["g2", "gartner", "reddit"]
    assert shared_mod._executive_source_list() == ["g2", "gartner"]
    assert blog_mod._blog_source_allowlist() == ["g2", "reddit"]
