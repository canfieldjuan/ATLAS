import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4


sys.modules.setdefault("asyncpg", MagicMock())

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from backfill_cross_source_review_dedup import (  # noqa: E402
    _parse_vendor_filter,
    _plan_vendor_duplicate_updates,
)


def _make_row(**overrides):
    row = {
        "id": uuid4(),
        "vendor_name": "ActiveCampaign",
        "source": "g2",
        "source_review_id": "r1",
        "reviewer_name": "Igor K.",
        "reviewed_at": datetime(2026, 3, 30, tzinfo=timezone.utc),
        "rating": 2.0,
        "imported_at": datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
        "enrichment_status": "enriched",
        "source_weight": 1.0,
        "summary": "Great automation, but expensive.",
        "review_text": "Robust functionality and expensive pricing.",
        "pros": "Great automation",
        "cons": "Expensive pricing",
        "raw_metadata": {},
        "cross_source_content_hash": "hash-1",
        "cross_source_identity_key": "activecampaign|igork|2026-03-30|2.0",
        "duplicate_of_review_id": None,
    }
    row.update(overrides)
    return row


def test_parse_vendor_filter_normalizes_names():
    assert _parse_vendor_filter(" ActiveCampaign, HubSpot ,, ") == [
        "activecampaign",
        "hubspot",
    ]


def test_plan_vendor_duplicate_updates_marks_exact_content_duplicates():
    canonical = _make_row(
        source="g2",
        source_weight=1.0,
        imported_at=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
    )
    duplicate = _make_row(
        id=uuid4(),
        source="trustpilot",
        source_weight=0.5,
        imported_at=datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc),
    )

    updates = _plan_vendor_duplicate_updates(
        [canonical, duplicate],
        similarity_threshold=0.82,
        loose_similarity_threshold=0.9,
        reviewer_stem_length=5,
        review_date_tolerance_days=1,
        rating_tolerance=1.0,
    )

    assert len(updates) == 1
    assert updates[0]["review_id"] == str(duplicate["id"])
    assert updates[0]["survivor_review_id"] == str(canonical["id"])
    assert updates[0]["duplicate_reason"] == "cross_source_exact_content"
    assert updates[0]["metadata"]["prior_enrichment_status"] == "enriched"


def test_plan_vendor_duplicate_updates_marks_identity_similarity_duplicates():
    canonical = _make_row(
        source="g2",
        source_weight=1.0,
        imported_at=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
        cross_source_content_hash=None,
        summary="Robust automations and email tools.",
        review_text="ActiveCampaign was fantastic for 8 years and had robust automations.",
        pros="Fantastic automations",
        cons="Pricing became harder to justify",
    )
    duplicate = _make_row(
        id=uuid4(),
        source="trustpilot",
        source_weight=0.5,
        imported_at=datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc),
        cross_source_content_hash=None,
        summary="Fantastic automation platform",
        review_text="I used ActiveCampaign for years. Fantastic automation features, but pricing became hard to justify.",
        pros="Fantastic automation features",
        cons="Pricing became hard to justify",
    )

    updates = _plan_vendor_duplicate_updates(
        [canonical, duplicate],
        similarity_threshold=0.45,
        loose_similarity_threshold=0.9,
        reviewer_stem_length=5,
        review_date_tolerance_days=1,
        rating_tolerance=1.0,
    )

    assert len(updates) == 1
    assert updates[0]["review_id"] == str(duplicate["id"])
    assert updates[0]["survivor_review_id"] == str(canonical["id"])
    assert updates[0]["duplicate_reason"] == "cross_source_identity_similarity"
    assert updates[0]["duplicate_detail"]["identity_key_match"] is True


def test_plan_vendor_duplicate_updates_marks_reviewer_date_similarity_duplicates():
    canonical = _make_row(
        source="g2",
        reviewer_name="Igor K.",
        reviewed_at=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
        rating=0.0,
        cross_source_content_hash=None,
        cross_source_identity_key="activecampaign|igork|2026-03-30|0.0",
        summary="Robust functionality and automations",
        review_text="I used ActiveCampaign for years. Robust functionality and automations, but pricing got harder to justify.",
        pros="Robust functionality",
        cons="Pricing got harder to justify",
    )
    duplicate = _make_row(
        id=uuid4(),
        source="trustpilot",
        reviewer_name="Igor Klibanov",
        reviewed_at=datetime(2026, 3, 31, 2, 0, tzinfo=timezone.utc),
        rating=1.0,
        source_weight=0.5,
        imported_at=datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc),
        cross_source_content_hash=None,
        cross_source_identity_key="activecampaign|igorklibanov|2026-03-31|1.0",
        summary="Robust functionality and automations",
        review_text="I used ActiveCampaign for years. Robust functionality and automations, but pricing got harder to justify.",
        pros="Robust functionality",
        cons="Pricing got harder to justify",
    )

    updates = _plan_vendor_duplicate_updates(
        [canonical, duplicate],
        similarity_threshold=0.82,
        loose_similarity_threshold=0.9,
        reviewer_stem_length=5,
        review_date_tolerance_days=1,
        rating_tolerance=1.0,
    )

    assert len(updates) == 1
    assert updates[0]["review_id"] == str(duplicate["id"])
    assert updates[0]["survivor_review_id"] == str(canonical["id"])
    assert updates[0]["duplicate_reason"] == "cross_source_reviewer_date_similarity"
    assert updates[0]["duplicate_detail"]["reviewer_overlap_match"] is True
