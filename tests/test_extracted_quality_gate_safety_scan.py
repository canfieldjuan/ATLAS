"""Tests for extracted_quality_gate.safety_gate.

The two functions under test (`check_content`, `assess_risk`) are pure:
no DB, no clock, no network. So these tests are pure unit tests --
no fixtures, no mocks, no async.
"""

from __future__ import annotations

import pytest

from extracted_quality_gate.safety_gate import (
    _PROHIBITED_PATTERNS,
    assess_risk,
    check_content,
)
from extracted_quality_gate.types import (
    ContentFlag,
    ContentScanResult,
    RiskAssessment,
    RiskLevel,
)


# ---- check_content ----


def test_check_content_empty_passes():
    result = check_content("")
    assert isinstance(result, ContentScanResult)
    assert result.passed is True
    assert result.blocked is False
    assert result.flags == ()


def test_check_content_none_safe_text_passes():
    result = check_content("Schedule a meeting next Tuesday at 2pm to review.")
    assert result.passed is True
    assert result.blocked is False
    assert result.flags == ()


def test_check_content_catches_each_known_pattern():
    # Every label in _PROHIBITED_PATTERNS should be detectable by SOME
    # input. Use the bare label-style word as a smoke check that the
    # regex covers the expected stem.
    samples = {
        "impersonation": "We will impersonate the CFO",
        "fabricated_facts": "instructed staff to fabricate facts about",
        "blackmail": "engaging in blackmail of the executive",
        "extortion": "an attempt at extortion",
        "threat_of_harm": "threaten to harm them physically",
        "evidence_manipulation": "manipulate evidence in the audit log",
        "doxxing": "publish a doxxing post",
        "phishing": "send a phishing email",
        "social_engineering": "social engineering the help desk",
    }
    expected_labels = {label for _, label in _PROHIBITED_PATTERNS}
    assert set(samples.keys()) == expected_labels, (
        "test sample must cover every pattern label"
    )
    for label, text in samples.items():
        result = check_content(text)
        flag_labels = {flag.pattern for flag in result.flags}
        assert label in flag_labels, f"pattern {label!r} did not flag {text!r}"


def test_check_content_returns_position_and_match_text():
    result = check_content("they tried to blackmail the witness")
    assert result.blocked is True
    assert len(result.flags) == 1
    flag = result.flags[0]
    assert isinstance(flag, ContentFlag)
    assert flag.pattern == "blackmail"
    assert flag.match.lower() == "blackmail"
    # "blackmail" starts at column 14 in the input above
    assert flag.position == 14


def test_check_content_multiple_flags_in_one_input():
    text = "We will phishing the team and use social engineering."
    result = check_content(text)
    assert result.blocked is True
    flag_labels = sorted(flag.pattern for flag in result.flags)
    assert "phishing" in flag_labels
    assert "social_engineering" in flag_labels


def test_check_content_is_case_insensitive():
    assert check_content("BLACKMAIL").blocked is True
    assert check_content("BlAcKmAiL").blocked is True


def test_content_scan_result_is_frozen():
    result = check_content("doxxing the witness")
    with pytest.raises(Exception):
        result.passed = True  # type: ignore[misc]
    with pytest.raises(Exception):
        result.flags = ()  # type: ignore[misc]


# ---- assess_risk ----


def test_assess_risk_no_signals_returns_low():
    assessment = assess_risk(sensor_summary={}, pressure={})
    assert isinstance(assessment, RiskAssessment)
    assert assessment.risk_level == RiskLevel.LOW
    assert assessment.risk_score == 0
    assert assessment.auto_approve_eligible is True
    assert assessment.factors == ()


def test_assess_risk_handles_none_inputs():
    # Calling code from the Atlas wrapper sometimes passes None when
    # upstream stage output is missing. Don't crash.
    assessment = assess_risk(sensor_summary=None, pressure=None)
    assert assessment.risk_level == RiskLevel.LOW
    assert assessment.risk_score == 0


def test_assess_risk_unknown_sensor_level_falls_back_to_low():
    assessment = assess_risk(
        sensor_summary={"dominant_risk_level": "BANANAS"},
        pressure={},
    )
    assert assessment.risk_level == RiskLevel.LOW


def test_assess_risk_high_sensor_adds_factor():
    assessment = assess_risk(
        sensor_summary={"dominant_risk_level": "HIGH"},
        pressure={},
    )
    # HIGH = +2 score -> MEDIUM band (1-2)
    assert assessment.risk_score == 2
    assert assessment.risk_level == RiskLevel.MEDIUM
    assert any("Sensor composite: HIGH" in factor for factor in assessment.factors)


def test_assess_risk_critical_pressure_pushes_to_critical():
    assessment = assess_risk(
        sensor_summary={"dominant_risk_level": "MEDIUM"},
        pressure={"pressure_score": 9},
    )
    # MEDIUM (+1) + critical pressure (+2) = 3 -> HIGH
    assert assessment.risk_score == 3
    assert assessment.risk_level == RiskLevel.HIGH
    assert any("Critical pressure: 9/10" in f for f in assessment.factors)


def test_assess_risk_content_block_adds_three_and_lists_flags():
    content = check_content("phishing attack")
    assert content.blocked
    assessment = assess_risk(
        sensor_summary={},
        pressure={},
        content_check=content,
    )
    # blocked content alone = +3 -> HIGH
    assert assessment.risk_score == 3
    assert assessment.risk_level == RiskLevel.HIGH
    assert any("Content flags: phishing" in f for f in assessment.factors)


def test_assess_risk_combines_signals():
    content = check_content("doxxing the witness")
    assessment = assess_risk(
        sensor_summary={"dominant_risk_level": "CRITICAL"},
        pressure={"pressure_score": 9},
        content_check=content,
    )
    # CRITICAL (+3) + critical pressure (+2) + blocked content (+3) = 8 -> CRITICAL
    assert assessment.risk_score == 8
    assert assessment.risk_level == RiskLevel.CRITICAL
    assert assessment.auto_approve_eligible is False


def test_assess_risk_auto_approve_threshold_is_configurable():
    # Default threshold is MEDIUM. Tighten to LOW: a MEDIUM input
    # should no longer auto-approve.
    assessment = assess_risk(
        sensor_summary={"dominant_risk_level": "MEDIUM"},
        pressure={},
        auto_approve_max_risk=RiskLevel.LOW,
    )
    assert assessment.risk_level == RiskLevel.MEDIUM
    assert assessment.auto_approve_eligible is False

    # Loosen to HIGH: HIGH input now auto-approves.
    assessment2 = assess_risk(
        sensor_summary={"dominant_risk_level": "HIGH"},
        pressure={},
        auto_approve_max_risk=RiskLevel.HIGH,
    )
    assert assessment2.risk_level == RiskLevel.MEDIUM  # +2 score = MEDIUM band
    assert assessment2.auto_approve_eligible is True


def test_assess_risk_non_numeric_pressure_score_is_ignored():
    assessment = assess_risk(
        sensor_summary={},
        pressure={"pressure_score": "high"},
    )
    # Non-numeric -> ignored, not crashed
    assert assessment.risk_score == 0
    assert assessment.risk_level == RiskLevel.LOW


def test_risk_assessment_is_frozen():
    assessment = assess_risk(sensor_summary={}, pressure={})
    with pytest.raises(Exception):
        assessment.risk_level = RiskLevel.CRITICAL  # type: ignore[misc]
    with pytest.raises(Exception):
        assessment.factors = ()  # type: ignore[misc]
