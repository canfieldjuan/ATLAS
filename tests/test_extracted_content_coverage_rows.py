from __future__ import annotations

from extracted_content_pipeline.content_pr import (
    ContentPR,
    CoverageStatus,
    RulePacketVersions,
    review_verdict,
)
from extracted_content_pipeline.coverage_rows import (
    brand_voice_coverage_rows,
    quality_gate_coverage_rows,
)
from extracted_content_pipeline.review_contract import ReviewDecision
from extracted_quality_gate.types import GateDecision, GateFinding, GateSeverity, QualityReport


_PINNED = RulePacketVersions(
    brief="brief-v1",
    brand_voice="voice-v1",
    claim_registry="claims-v1",
    compliance="compliance-v1",
    channel_schema="channel-v1",
)


def _verdict(rows):
    return review_verdict(ContentPR(rule_packet=_PINNED, coverage=tuple(rows)))


def test_quality_gate_pass_emits_required_pass_row() -> None:
    rows = quality_gate_coverage_rows(QualityReport(passed=True, decision=GateDecision.PASS))

    assert len(rows) == 1
    assert rows[0].rule_id == "QUALITY-GATE:report"
    assert rows[0].required is True
    assert rows[0].status == CoverageStatus.PASS
    assert rows[0].is_resolved() is True
    assert _verdict(rows) is ReviewDecision.APPROVED


def test_quality_gate_blocker_emits_required_fail_row() -> None:
    rows = quality_gate_coverage_rows(
        QualityReport(
            passed=False,
            decision=GateDecision.BLOCK,
            findings=(
                GateFinding(
                    code="no_cta",
                    message="CTA is missing",
                    severity=GateSeverity.BLOCKER,
                    field_name="cta",
                ),
            ),
        )
    )

    assert rows[0].rule_id == "QUALITY-GATE:no-cta"
    assert rows[0].required is True
    assert rows[0].status == CoverageStatus.FAIL
    assert "cta" in rows[0].evidence
    assert _verdict(rows) is ReviewDecision.REVISION_REQUIRED


def test_quality_gate_warnings_are_optional_resolved_rows() -> None:
    rows = quality_gate_coverage_rows(
        QualityReport(
            passed=True,
            decision=GateDecision.WARN,
            findings=(
                GateFinding("missing_confidence", "confidence missing", GateSeverity.WARNING),
                GateFinding("word_count", "word count noted", GateSeverity.INFO),
            ),
        )
    )

    assert rows[0].status == CoverageStatus.PASS
    assert [row.required for row in rows[1:]] == [False, False]
    assert all(row.is_resolved() for row in rows)
    assert _verdict(rows) is ReviewDecision.APPROVED


def test_quality_gate_decoded_failures_do_not_pass() -> None:
    failed_without_findings = quality_gate_coverage_rows(
        {"passed": False, "decision": "block", "findings": []}
    )
    decoded_blocker = quality_gate_coverage_rows(
        {
            "passed": False,
            "findings": [
                {
                    "code": "unsupported_claim",
                    "message": "unsupported claim",
                    "severity": "blocker",
                }
            ],
        }
    )

    assert failed_without_findings[0].status == CoverageStatus.FAIL
    assert decoded_blocker[0].status == CoverageStatus.FAIL
    assert _verdict(failed_without_findings) is ReviewDecision.REVISION_REQUIRED
    assert _verdict(decoded_blocker) is ReviewDecision.REVISION_REQUIRED


def test_quality_gate_missing_malformed_and_unknown_inputs_block() -> None:
    cases = (
        None,
        "not a report",
        {"passed": "yes"},
        {"passed": True, "findings": [{"code": "mystery", "severity": "maybe"}]},
        {"passed": True, "findings": ["bad"]},
    )

    for report in cases:
        rows = quality_gate_coverage_rows(report)
        assert any(row.status == CoverageStatus.UNRESOLVED for row in rows)
        assert _verdict(rows) is ReviewDecision.BLOCKED


def test_quality_gate_contradictory_blocking_envelopes_block() -> None:
    cases = (
        {"passed": True, "decision": "rejected"},
        {"passed": True, "decision": "blocked"},
        {"passed": True, "findings": [], "decision": "block"},
        {"passed": True, "verdict": "blocked"},
        {"passed": True, "outcome": "fail"},
        QualityReport(passed=True, decision=GateDecision.BLOCK),
    )

    for report in cases:
        rows = quality_gate_coverage_rows(report)
        assert rows[0].rule_id.startswith("QUALITY-GATE:contradictory-")
        assert rows[0].status == CoverageStatus.UNRESOLVED
        assert _verdict(rows) is ReviewDecision.BLOCKED


def test_quality_gate_nonblocking_decision_near_miss_still_passes() -> None:
    rows = quality_gate_coverage_rows(
        {"passed": True, "decision": "warn", "findings": []}
    )

    assert rows[0].rule_id == "QUALITY-GATE:report"
    assert rows[0].status == CoverageStatus.PASS
    assert _verdict(rows) is ReviewDecision.APPROVED


def test_brand_voice_pass_emits_required_pass_row() -> None:
    for key in ("_brand_voice_audit", "brand_voice_audit"):
        rows = brand_voice_coverage_rows({key: {"passed": True}})

        assert rows[0].rule_id == "BRAND-VOICE:audit"
        assert rows[0].status == CoverageStatus.PASS
        assert rows[0].required is True
        assert _verdict(rows) is ReviewDecision.APPROVED


def test_brand_voice_warnings_and_banned_terms_emit_fail_rows() -> None:
    rows = brand_voice_coverage_rows(
        {
            "_brand_voice_audit": {
                "passed": False,
                "warnings": ["preferred_pov_second_person_not_detected", None],
                "banned_terms": ["synergy", "synergy", 5],
            }
        }
    )

    assert [row.rule_id for row in rows] == [
        "BRAND-VOICE:warning-preferred-pov-second-person-not-detected",
        "BRAND-VOICE:banned-term-synergy",
    ]
    assert all(row.required for row in rows)
    assert all(row.status == CoverageStatus.FAIL for row in rows)
    assert _verdict(rows) is ReviewDecision.REVISION_REQUIRED


def test_brand_voice_missing_or_malformed_audit_blocks() -> None:
    for payload in ({}, {"_brand_voice_audit": "bad"}, None):
        rows = brand_voice_coverage_rows(payload)
        assert rows[0].status == CoverageStatus.UNRESOLVED
        assert _verdict(rows) is ReviewDecision.BLOCKED
