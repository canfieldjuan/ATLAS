"""Tests for tenant-report displacement driver backfill defaults."""

from __future__ import annotations

from typing import Any

from atlas_brain.autonomous.tasks.b2b_tenant_report import (
    _tenant_displacement_backfill_row,
)


def _row(
    edge: dict[str, Any],
    *,
    reason_lookup: dict[tuple[str, str], list[str]] | None = None,
) -> dict[str, Any]:
    return _tenant_displacement_backfill_row(
        {
            "vendor": "Notion",
            "competitor": "ClickUp",
            "mention_count": 4,
            **edge,
        },
        reason_lookup=reason_lookup or {},
    )


def test_backfill_returns_empty_driver_when_no_bucket_matches():
    row = _row({"reason_categories": {"miscellaneous": 3}})
    assert row["primary_driver"] == ""


def test_backfill_returns_empty_driver_for_missing_reasons():
    row = _row({"reason_categories": {}})
    assert row["primary_driver"] == ""


def test_backfill_uses_highest_count_reason_category_bucket():
    row = _row({
        "reason_categories": {
            "feature gaps": 2,
            "pricing pressure": 9,
        },
    })
    assert row["primary_driver"] == "pricing"


def test_backfill_falls_through_to_reason_lookup_when_categories_do_not_match():
    row = _row(
        {"reason_categories": {"miscellaneous": 3}},
        reason_lookup={
            ("notion", "clickup"): [
                "Sales teams mention better workflow automation in ClickUp",
            ],
        },
    )
    assert row["primary_driver"] == "features"


def test_backfill_preserves_other_fields():
    row = _row({
        "vendor": "Zendesk",
        "competitor": "Freshdesk",
        "mention_count": 12,
        "reason_categories": {"support response": 5},
    })
    assert row["from_vendor"] == "Zendesk"
    assert row["to_vendor"] == "Freshdesk"
    assert row["mention_count"] == 12
    assert row["primary_driver"] == "support"
    assert row["signal_strength"] == "moderate"
    assert row["key_quote"] is None


def test_backfill_signal_strength_boundaries():
    assert _row({"mention_count": 20})["signal_strength"] == "strong"
    assert _row({"mention_count": 5})["signal_strength"] == "moderate"
    assert _row({"mention_count": 4})["signal_strength"] == "light"
    assert _row({"mention_count": 1, "explicit_switches": 1})["signal_strength"] == "strong"
    assert _row({"mention_count": 1, "active_evaluations": 1})["signal_strength"] == "moderate"
