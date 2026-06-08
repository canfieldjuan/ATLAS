from __future__ import annotations

import pytest

from extracted_content_pipeline.brand_voice import (
    apply_brand_voice_to_system_prompt,
    brand_voice_profile_from_mapping,
    brand_voice_prompt_block,
    brand_voice_quality_blockers,
    brand_voice_result_metadata,
    generation_grounding_prompt_block,
)
from extracted_content_pipeline.campaign_ports import TenantScope


def test_brand_voice_profile_normalizes_prompt_block_and_audit_metadata() -> None:
    profile = brand_voice_profile_from_mapping(
        {
            "id": "acme-main",
            "account_id": "acct-1",
            "name": "Acme main voice",
            "descriptors": ["plainspoken", "operator-led", "plainspoken"],
            "exemplars": [
                {"text": "We explain tradeoffs in plain English."},
                "Use concrete proof before claims.",
                "Avoid theatrical language.",
                "Ignored fourth sample.",
            ],
            "banned_terms": ["synergy"],
            "preferred_pov": "you",
            "reading_level": "plain",
        },
        scope=TenantScope(account_id="acct-1"),
    )

    assert profile is not None
    assert profile.descriptors == ("plainspoken", "operator-led")
    assert profile.exemplars == (
        "We explain tradeoffs in plain English.",
        "Use concrete proof before claims.",
        "Avoid theatrical language.",
    )
    block = brand_voice_prompt_block(profile)
    assert "Use this profile as style guidance only" in block
    assert "grounding contract controls factual claims" in block
    assert "Do not omit or alter grounded claims" in block
    assert "Use `you` or `your` naturally" in block
    assert "operator-led" in block
    assert "Ignored fourth sample" not in block
    grounding = generation_grounding_prompt_block()
    assert "## Grounding contract" in grounding
    assert "Never introduce counts, percentages, statistics" in grounding
    assert "scan or research claims" in grounding
    assert "company or contact names" in grounding
    assert apply_brand_voice_to_system_prompt("Base\n{brand_voice}", profile) == (
        "Base\n" + grounding + "\n\n" + block
    )
    assert apply_brand_voice_to_system_prompt("Base", None) == (
        "Base\n\n" + grounding
    )

    parsed = brand_voice_result_metadata(
        {"subject": "No synergy pitch", "body": "We explain the tradeoff."},
        profile,
    )

    assert parsed["_brand_voice_profile"]["id"] == "acme-main"
    audit = parsed["_brand_voice_audit"]
    assert audit["passed"] is False
    assert audit["banned_terms"] == ["synergy"]
    assert "preferred_pov_second_person_not_detected" in audit["warnings"]


def test_brand_voice_profile_scope_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="account_id does not match"):
        brand_voice_profile_from_mapping(
            {"account_id": "acct-2", "name": "Wrong tenant", "descriptors": ["warm"]},
            scope=TenantScope(account_id="acct-1"),
        )


def test_brand_voice_profile_id_without_inline_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires inputs.brand_voice"):
        brand_voice_profile_from_mapping(
            None,
            scope=TenantScope(account_id="acct-1"),
            profile_id="acme-main",
        )


def test_brand_voice_profile_id_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="id does not match"):
        brand_voice_profile_from_mapping(
            {"id": "stale-profile", "descriptors": ["warm"]},
            scope=TenantScope(account_id="acct-1"),
            profile_id="acme-main",
        )


def test_brand_voice_profile_caps_descriptor_and_banned_term_lengths() -> None:
    long_descriptor = "d" * 500
    long_term = "b" * 500

    profile = brand_voice_profile_from_mapping(
        {
            "id": "acme-main",
            "descriptors": [long_descriptor],
            "banned_terms": [long_term],
        },
        scope=TenantScope(account_id="acct-1"),
    )

    assert profile is not None
    assert profile.descriptors == ("d" * 120,)
    assert profile.banned_terms == ("b" * 120,)


def test_brand_voice_audit_flags_reading_level_band() -> None:
    profile = brand_voice_profile_from_mapping(
        {"id": "acme-main", "reading_level": "concise"},
        scope=TenantScope(account_id="acct-1"),
    )

    parsed = brand_voice_result_metadata(
        {
            "body": (
                "This sentence intentionally keeps going with many extra words "
                "so the concise reading level detector has a clear breach."
            )
        },
        profile,
    )

    assert parsed["_brand_voice_audit"]["passed"] is False
    assert "reading_level_concise_exceeded" in parsed["_brand_voice_audit"]["warnings"]


def test_brand_voice_audit_ignores_private_generation_metadata() -> None:
    profile = brand_voice_profile_from_mapping(
        {"id": "acme-main", "preferred_pov": "second_person"},
        scope=TenantScope(account_id="acct-1"),
    )

    parsed = brand_voice_result_metadata(
        {
            "title": "Renewal risk brief",
            "body": "The renewal risk is visible in repeated pricing complaints.",
            "_variant_angle": "your renewal risk",
        },
        profile,
    )

    assert parsed["_brand_voice_audit"]["passed"] is False
    assert "preferred_pov_second_person_not_detected" in parsed["_brand_voice_audit"]["warnings"]


def test_brand_voice_quality_blockers_surface_failed_audit_fields() -> None:
    parsed = {
        "_brand_voice_audit": {
            "passed": False,
            "warnings": [
                "preferred_pov_second_person_not_detected",
                "preferred_pov_second_person_not_detected",
                "",
            ],
            "banned_terms": ["synergy", "synergy", ""],
        }
    }

    assert brand_voice_quality_blockers(parsed) == (
        "brand_voice:preferred_pov_second_person_not_detected",
        "brand_voice:banned_term:synergy",
    )


def test_brand_voice_quality_blockers_ignore_passed_or_missing_audits() -> None:
    assert brand_voice_quality_blockers({}) == ()
    assert brand_voice_quality_blockers({"_brand_voice_audit": {"passed": True}}) == ()
