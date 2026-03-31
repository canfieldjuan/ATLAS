from __future__ import annotations

from datetime import datetime, timezone

from atlas_brain.services.blog_visibility_backfill import (
    QUALITY_GATE_REJECTION_CODE,
    derive_blog_visibility_patch,
)


def test_derive_blog_visibility_patch_backfills_threshold_and_attempt_metadata():
    patch = derive_blog_visibility_patch(
        {
            "status": "draft",
            "latest_run_id": None,
            "latest_attempt_no": None,
            "latest_failure_step": None,
            "latest_error_code": None,
            "latest_error_summary": None,
            "quality_score": None,
            "quality_threshold": None,
            "blocker_count": 0,
            "warning_count": 0,
            "rejected_at": None,
            "rejection_reason": None,
            "attempt_run_id": "run-123",
            "attempt_attempt_no": 2,
            "attempt_failure_step": None,
            "attempt_error_message": None,
            "quality_attempt_no": 2,
            "quality_attempt_status": "succeeded",
            "quality_failure_step": None,
            "quality_score_attempt": 88,
            "quality_threshold_attempt": None,
            "quality_blocker_count": 0,
            "quality_warning_count": 3,
            "quality_blocking_issues": [],
            "quality_completed_at": datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
        },
        default_threshold=70,
    )

    assert patch == {
        "latest_run_id": "run-123",
        "latest_attempt_no": 2,
        "quality_score": 88,
        "quality_threshold": 70,
        "warning_count": 3,
    }


def test_derive_blog_visibility_patch_backfills_rejection_context():
    rejected_at = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    patch = derive_blog_visibility_patch(
        {
            "status": "rejected",
            "latest_run_id": None,
            "latest_attempt_no": None,
            "latest_failure_step": None,
            "latest_error_code": None,
            "latest_error_summary": None,
            "quality_score": None,
            "quality_threshold": None,
            "blocker_count": 0,
            "warning_count": 0,
            "rejected_at": None,
            "rejection_reason": None,
            "attempt_run_id": None,
            "attempt_attempt_no": 1,
            "attempt_failure_step": None,
            "attempt_error_message": None,
            "quality_attempt_no": 1,
            "quality_attempt_status": "rejected",
            "quality_failure_step": "quality_gate",
            "quality_score_attempt": 64,
            "quality_threshold_attempt": 70,
            "quality_blocker_count": 2,
            "quality_warning_count": 1,
            "quality_blocking_issues": [
                "unsupported_data_claim:Magento",
                "chart_scope_ambiguity:Magento",
            ],
            "quality_completed_at": rejected_at,
        },
        default_threshold=70,
    )

    assert patch["latest_attempt_no"] == 1
    assert patch["latest_failure_step"] == "quality_gate"
    assert patch["latest_error_code"] == QUALITY_GATE_REJECTION_CODE
    assert patch["latest_error_summary"] == (
        "unsupported_data_claim:Magento, chart_scope_ambiguity:Magento"
    )
    assert patch["quality_score"] == 64
    assert patch["quality_threshold"] == 70
    assert patch["blocker_count"] == 2
    assert patch["warning_count"] == 1
    assert patch["rejection_reason"] == (
        "unsupported_data_claim:Magento, chart_scope_ambiguity:Magento"
    )
    assert patch["rejected_at"] == rejected_at


def test_derive_blog_visibility_patch_keeps_existing_truth_fields():
    patch = derive_blog_visibility_patch(
        {
            "status": "draft",
            "latest_run_id": "run-existing",
            "latest_attempt_no": 3,
            "latest_failure_step": None,
            "latest_error_code": None,
            "latest_error_summary": None,
            "quality_score": 91,
            "quality_threshold": 70,
            "blocker_count": 0,
            "warning_count": 2,
            "rejected_at": None,
            "rejection_reason": None,
            "attempt_run_id": "run-new",
            "attempt_attempt_no": 4,
            "attempt_failure_step": None,
            "attempt_error_message": None,
            "quality_attempt_no": 4,
            "quality_attempt_status": "succeeded",
            "quality_failure_step": None,
            "quality_score_attempt": 95,
            "quality_threshold_attempt": 70,
            "quality_blocker_count": 0,
            "quality_warning_count": 2,
            "quality_blocking_issues": [],
            "quality_completed_at": datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
        },
        default_threshold=70,
    )

    assert patch == {}
