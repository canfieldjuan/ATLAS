"""Failure-detection suite for BrandVoiceValidator (AGENTS.md 3i)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from atlas_brain.brand.voice_validator import BrandVoiceFinding, BrandVoiceValidator


REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG_PATH = REPO_ROOT / "atlas_brain" / "skills" / "brand" / "brand_voice.yml"
VALIDATOR_CLI = REPO_ROOT / "atlas_brain" / "brand" / "voice_validator.py"


def _write_config(tmp_path: Path, body: str, name: str = "brand_voice.yml") -> Path:
    """Write a small focused YAML config under tmp_path for unit isolation."""
    config_path = tmp_path / name
    config_path.write_text(textwrap.dedent(body))
    return config_path


def _real_validator() -> BrandVoiceValidator:
    """Build a validator backed by the real shipped brand_voice.yml."""
    return BrandVoiceValidator(config_path=REAL_CONFIG_PATH)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR_CLI), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _messages(findings: list[BrandVoiceFinding]) -> list[str]:
    return [finding.message for finding in findings]


def _only_finding(findings: list[BrandVoiceFinding]) -> BrandVoiceFinding:
    assert len(findings) == 1
    return findings[0]


# ---------------------------------------------------------------------------
# Sanity: the real config is present and loadable (it is the substrate for the
# "real shipped config" arm of this suite).
# ---------------------------------------------------------------------------


def test_real_shipped_brand_voice_yml_exists_and_loads():
    assert REAL_CONFIG_PATH.exists(), f"missing real config: {REAL_CONFIG_PATH}"
    validator = _real_validator()
    assert isinstance(validator.config, dict)
    assert "vocabulary" in validator.config


# ===========================================================================
# RULE TYPE 1: forbidden vocabulary (vocabulary.avoid)
# ===========================================================================


def test_forbidden_vocabulary_word_bites_on_real_config():
    """A shipped avoid-word used as a standalone word fires a violation."""
    validator = _real_validator()
    findings = validator.validate("This is a game-changer for the team.", "blog_post")
    assert "Contains forbidden word: 'game-changer'" in _messages(findings)


def test_forbidden_vocabulary_finding_has_structured_fields():
    validator = _real_validator()
    finding = _only_finding(
        validator.validate("This is a game-changer for the team.", "blog_post")
    )
    assert finding.rule_id == "vocabulary.avoid.game-changer"
    assert finding.severity == "BLOCKER"
    assert finding.category == "vocabulary"
    assert finding.message == "Contains forbidden word: 'game-changer'"


def test_suggested_vocabulary_finding_has_replacement_metadata():
    validator = _real_validator()
    finding = _only_finding(
        validator.validate("The result is predictable for operators.", "blog_post")
    )
    assert finding.rule_id == "vocabulary.use.predictable"
    assert finding.severity == "NIT"
    assert finding.category == "vocabulary"
    assert finding.message == "Prefer 'deterministic' over 'predictable'"
    assert finding.suggestion == "Use 'deterministic' instead of 'predictable'"


def test_preferred_vocabulary_terms_are_clean():
    validator = _real_validator()
    findings = validator.validate(
        "A deterministic capability dispatch system gives operators a clear path.",
        "blog_post",
    )
    assert findings == []


def test_clean_content_has_no_forbidden_vocabulary_false_positive():
    """On-brand prose containing no whole avoid-word yields no vocab violation."""
    validator = _real_validator()
    findings = validator.validate(
        "Our deterministic dispatch system ships a reliable migration path.",
        "blog_post",
    )
    assert findings == []


# ===========================================================================
# RULE TYPE 2: tone_rules -- excessive punctuation (pattern "!!|\?\?")
# Each ALTERNATION arm proven independently (dead-alternation lesson).
# ===========================================================================

_EXCLAMATION_TONE_MSG = (
    "Tone violation: Avoid using more than one exclamation point or "
    "question mark in a row."
)


def test_tone_excessive_punctuation_double_bang_branch_bites():
    """The '!!' alternation arm fires independently."""
    validator = _real_validator()
    findings = validator.validate("This release is incredible!!", "blog_post")
    assert _EXCLAMATION_TONE_MSG in _messages(findings)


def test_tone_rule_finding_uses_rule_id_and_default_major_severity():
    validator = _real_validator()
    finding = _only_finding(validator.validate("This release is incredible!!", "blog_post"))
    assert finding.rule_id == "no_excessive_punctuation"
    assert finding.severity == "MAJOR"
    assert finding.category == "tone"
    assert finding.message == _EXCLAMATION_TONE_MSG


def test_tone_excessive_punctuation_double_question_branch_bites():
    """The '\\?\\?' alternation arm fires independently."""
    validator = _real_validator()
    findings = validator.validate("Did the build pass??", "blog_post")
    assert _EXCLAMATION_TONE_MSG in _messages(findings)


def test_tone_excessive_punctuation_single_marks_are_clean():
    """A single ! and a single ? do not trip the excessive-punctuation rule."""
    validator = _real_validator()
    findings = validator.validate(
        "The build passed! Does that surprise you?", "blog_post"
    )
    assert findings == []


# ===========================================================================
# RULE TYPE 2: tone_rules -- casual phrases
# pattern "(so,|well,|you know,|like,|basically)"
# Each ALTERNATION arm proven independently. NOTE the shipped pattern's last
# arm is the bare token "basically" (no trailing comma), unlike the other
# four arms which require a trailing comma -- each test pins that exact arm.
# ===========================================================================

_CASUAL_TONE_MSG = "Tone violation: Avoid overly casual slang and filler words."


def test_tone_casual_phrase_so_comma_branch_bites():
    validator = _real_validator()
    findings = validator.validate("So, here is the summary.", "blog_post")
    assert _CASUAL_TONE_MSG in _messages(findings)


def test_tone_casual_phrase_well_comma_branch_bites():
    validator = _real_validator()
    findings = validator.validate("Well, that settles it.", "blog_post")
    assert _CASUAL_TONE_MSG in _messages(findings)


def test_tone_casual_phrase_you_know_comma_branch_bites():
    validator = _real_validator()
    findings = validator.validate("You know, it depends.", "blog_post")
    assert _CASUAL_TONE_MSG in _messages(findings)


def test_tone_casual_phrase_like_comma_branch_bites():
    validator = _real_validator()
    findings = validator.validate("Like, the latency dropped.", "blog_post")
    assert _CASUAL_TONE_MSG in _messages(findings)


def test_tone_casual_phrase_basically_bare_token_branch_bites():
    """The 'basically' arm has no trailing comma in the shipped pattern; it
    must still fire on the bare token."""
    validator = _real_validator()
    findings = validator.validate("Basically it just works.", "blog_post")
    assert _CASUAL_TONE_MSG in _messages(findings)


def test_clean_content_has_no_casual_phrase_false_positive():
    """Formal prose containing none of the five casual arms is clean.

    Deliberately avoids the bare token 'basically' as well as the
    comma-suffixed arms.
    """
    validator = _real_validator()
    findings = validator.validate(
        "The system dispatches each request deterministically and returns a "
        "structured result.",
        "blog_post",
    )
    assert findings == []


# ===========================================================================
# RULE TYPE 3 (content_rules): landing_page must-mention extensibility
# fail_on_match: false -> fails when the pattern is ABSENT.
# pattern "extensibility|extensible|pluggable" -- each synonym arm proven to
# SATISFY the rule independently (dead-alternation lesson on the positive side).
# ===========================================================================

_LANDING_MUST_MENTION_MSG = (
    "Content rule violation: Landing pages must mention the core value "
    "proposition of extensibility."
)


def test_landing_page_must_mention_bites_when_extensibility_absent():
    """A landing page that never mentions extensibility fails the must-mention
    rule (fail_on_match: false branch)."""
    validator = _real_validator()
    findings = validator.validate(
        "Our product helps teams move faster with a clean dashboard.",
        "landing_page",
    )
    assert _LANDING_MUST_MENTION_MSG in _messages(findings)


def test_content_rule_finding_uses_rule_id_and_default_blocker_severity():
    validator = _real_validator()
    finding = _only_finding(
        validator.validate(
            "Our product helps teams move faster with a clean dashboard.",
            "landing_page",
        )
    )
    assert finding.rule_id == "landing_page_extensibility_mention"
    assert finding.severity == "BLOCKER"
    assert finding.category == "content"
    assert finding.message == _LANDING_MUST_MENTION_MSG


def test_landing_page_must_mention_satisfied_by_extensibility_arm():
    validator = _real_validator()
    findings = validator.validate(
        "Atlas is built around extensibility from day one.", "landing_page"
    )
    assert _LANDING_MUST_MENTION_MSG not in _messages(findings)
    assert findings == []


def test_landing_page_must_mention_satisfied_by_extensible_arm():
    validator = _real_validator()
    findings = validator.validate(
        "Every component of the system is extensible.", "landing_page"
    )
    assert _LANDING_MUST_MENTION_MSG not in _messages(findings)
    assert findings == []


def test_landing_page_must_mention_satisfied_by_pluggable_arm():
    validator = _real_validator()
    findings = validator.validate(
        "Each capability is pluggable behind a typed port.", "landing_page"
    )
    assert _LANDING_MUST_MENTION_MSG not in _messages(findings)
    assert findings == []


# ===========================================================================
# RULE TYPE 3 (content_rules): release_notes no future tense
# fail_on_match: true -> fails when the pattern is PRESENT.
# pattern "(will be|coming soon|in the future)" -- each arm proven independently.
# ===========================================================================

_RELEASE_FUTURE_TENSE_MSG = (
    "Content rule violation: Release notes for shipped features should not "
    "use future-tense language."
)


def test_release_notes_future_tense_will_be_branch_bites():
    validator = _real_validator()
    findings = validator.validate(
        "Dark mode will be enabled for all tenants.", "release_notes"
    )
    assert _RELEASE_FUTURE_TENSE_MSG in _messages(findings)


def test_release_notes_future_tense_coming_soon_branch_bites():
    validator = _real_validator()
    findings = validator.validate(
        "A redesigned export view is coming soon.", "release_notes"
    )
    assert _RELEASE_FUTURE_TENSE_MSG in _messages(findings)


def test_release_notes_future_tense_in_the_future_branch_bites():
    validator = _real_validator()
    findings = validator.validate(
        "We may revisit this in the future.", "release_notes"
    )
    assert _RELEASE_FUTURE_TENSE_MSG in _messages(findings)


def test_release_notes_past_tense_is_clean():
    """Shipped-feature, past-tense release copy with no future-tense arm passes."""
    validator = _real_validator()
    findings = validator.validate(
        "We added a deterministic export and fixed the dispatch retry path.",
        "release_notes",
    )
    assert findings == []


# ===========================================================================
# ISOLATED-CONFIG mirrors of each rule type (unit isolation via tmp_path).
# These prove the branches against tiny configs the test owns, independent of
# the shipped YAML drifting.
# ===========================================================================


def test_isolated_forbidden_vocabulary_word_bites(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid:
            - "synergy"
        tone_rules: []
        content_rules: []
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    assert _messages(validator.validate("Pure synergy across teams.", "blog_post")) == [
        "Contains forbidden word: 'synergy'"
    ]


def test_isolated_forbidden_vocabulary_clean_has_no_violation(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid:
            - "synergy"
        tone_rules: []
        content_rules: []
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    assert validator.validate("Clear collaboration across teams.", "blog_post") == []


def test_isolated_tone_rule_bites_and_reports_description(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules:
          - id: "no_caps_yell"
            description: "Do not yell."
            pattern: "URGENT"
            fail_on_match: true
        content_rules: []
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    assert _messages(validator.validate("This is URGENT, act now.", "blog_post")) == [
        "Tone violation: Do not yell."
    ]


def test_tone_rule_fail_on_match_false_is_rejected_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules:
          - id: "must_say_trusted"
            description: "Tone rules cannot require a phrase."
            pattern: "trusted"
            fail_on_match: false
        content_rules: []
        """,
    )
    with pytest.raises(
        ValueError,
        match=r"tone_rules\[0\].*must_say_trusted.*fail_on_match",
    ):
        BrandVoiceValidator(config_path=config)


def test_isolated_content_rule_must_mention_bites_when_absent(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules: []
        content_rules:
          - id: "must_mention_pricing"
            description: "Landing pages must mention pricing."
            applies_to: "landing_page"
            pattern: "pricing"
            fail_on_match: false
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    assert _messages(
        validator.validate("A capability-rich landing page.", "landing_page")
    ) == [
        "Content rule violation: Landing pages must mention pricing."
    ]
    # Same rule satisfied when the term is present.
    assert validator.validate("See our pricing page.", "landing_page") == []


def test_isolated_content_rule_fail_on_match_true_bites_when_present(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules: []
        content_rules:
          - id: "no_tbd"
            description: "Release notes must not say TBD."
            applies_to: "release_notes"
            pattern: "tbd"
            fail_on_match: true
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    assert _messages(validator.validate("Ship date: TBD.", "release_notes")) == [
        "Content rule violation: Release notes must not say TBD."
    ]
    assert validator.validate("Ship date: shipped.", "release_notes") == []


def test_isolated_content_rule_custom_severity_is_normalized(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules: []
        content_rules:
          - id: "prefer_specific_copy"
            description: "Use specific copy."
            applies_to: "landing_page"
            pattern: "specific"
            fail_on_match: false
            severity: nit
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    finding = _only_finding(validator.validate("A generic landing page.", "landing_page"))
    assert finding.rule_id == "prefer_specific_copy"
    assert finding.severity == "NIT"
    assert finding.category == "content"
    assert finding.message == "Content rule violation: Use specific copy."


def test_rule_invalid_severity_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules:
          - id: "no_caps_yell"
            description: "Do not yell."
            pattern: "URGENT"
            severity: urgent
        content_rules: []
        """,
    )
    with pytest.raises(ValueError, match="no_caps_yell.*invalid severity"):
        BrandVoiceValidator(config_path=config)


def test_vocabulary_use_non_list_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          use: predictable
          avoid: []
        tone_rules: []
        content_rules: []
        """,
    )
    with pytest.raises(ValueError, match="vocabulary.use must be a list"):
        BrandVoiceValidator(config_path=config)


def test_vocabulary_use_empty_string_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          use: ""
          avoid: []
        tone_rules: []
        content_rules: []
        """,
    )
    with pytest.raises(ValueError, match="vocabulary.use must be a list"):
        BrandVoiceValidator(config_path=config)


def test_vocabulary_non_mapping_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary: []
        tone_rules: []
        content_rules: []
        """,
    )
    with pytest.raises(ValueError, match="vocabulary must be a mapping"):
        BrandVoiceValidator(config_path=config)


def test_vocabulary_use_requires_one_item_mapping_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          use:
            - deterministic: predictable
              capability: feature
          avoid: []
        tone_rules: []
        content_rules: []
        """,
    )
    with pytest.raises(ValueError, match=r"vocabulary.use\[0\].*one-item mapping"):
        BrandVoiceValidator(config_path=config)


def test_vocabulary_use_empty_discouraged_term_raises_value_error(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          use:
            - deterministic: ""
          avoid: []
        tone_rules: []
        content_rules: []
        """,
    )
    with pytest.raises(ValueError, match=r"vocabulary.use\[0\].*discouraged"):
        BrandVoiceValidator(config_path=config)


def test_rule_missing_id_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules:
          - description: "Do not yell."
            pattern: "URGENT"
        content_rules: []
        """,
    )
    with pytest.raises(ValueError, match="missing required.*id"):
        BrandVoiceValidator(config_path=config)


def test_content_rule_missing_applies_to_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules: []
        content_rules:
          - id: "missing_applies_to"
            description: "Needs an applies_to field."
            pattern: "pricing"
            fail_on_match: false
        """,
    )
    with pytest.raises(ValueError, match="missing_applies_to.*applies_to"):
        BrandVoiceValidator(config_path=config)


# ===========================================================================
# REGRESSION GUARD: substring footgun (the substring vs word-boundary FP).
#
# A shipped avoid-word embedded in a larger legitimate word must not raise a
# false positive. The old unanchored substring matcher flagged clean prose:
#   'transform' flags "Transformer", 'leverage' flags "deleverage",
#   'disrupt' flags "non-disruptive".
#
# Each test below asserts the locked-in whole-word behavior.
# ===========================================================================


def test_substring_footgun_transform_does_not_false_positive_on_transformer():
    validator = _real_validator()
    findings = validator.validate(
        "The Transformer architecture powers our embeddings.", "blog_post"
    )
    assert "Contains forbidden word: 'transform'" not in _messages(findings)


def test_substring_footgun_leverage_does_not_false_positive_on_deleverage():
    validator = _real_validator()
    findings = validator.validate(
        "We help teams deleverage their technical debt.", "blog_post"
    )
    assert "Contains forbidden word: 'leverage'" not in _messages(findings)


def test_substring_footgun_disrupt_does_not_false_positive_on_non_disruptive():
    validator = _real_validator()
    findings = validator.validate(
        "Our non-disruptive migration path keeps you live.", "blog_post"
    )
    assert "Contains forbidden word: 'disrupt'" not in _messages(findings)


def test_substring_word_boundary_still_catches_the_true_positive_standalone():
    """Guard the OTHER direction: the legitimate standalone avoid-word must
    still be caught. (Holds today AND after the word-boundary fix, so it is a
    plain green assertion -- it locks the fix from over-correcting into a
    miss.)"""
    validator = _real_validator()
    findings = validator.validate(
        "We will transform your workflow overnight.", "blog_post"
    )
    assert "Contains forbidden word: 'transform'" in _messages(findings)


# ===========================================================================
# REGRESSION GUARD: content_rule fail_on_match default.
#
# A content_rule author who omits fail_on_match gets the fail-closed default:
# the pattern is banned on match rather than inverted into a must-contain rule.
# ===========================================================================


def test_content_rule_missing_fail_on_match_key_bans_on_match_not_inverts(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules: []
        content_rules:
          - id: "no_future_tense_defaulted"
            description: "No future tense."
            applies_to: "blog_post"
            pattern: "will be"
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    msg = "Content rule violation: No future tense."
    # Intended: clean text (no banned phrase) must NOT be flagged.
    assert msg not in _messages(
        validator.validate("This is clean shipped copy.", "blog_post")
    )
    # Intended: text containing the banned phrase MUST be flagged.
    assert msg in _messages(validator.validate("This will be great.", "blog_post"))


# ===========================================================================
# RULE TYPE 4: robustness.
#
# Malformed config is caught at load time with clear errors. Empty text, empty
# sections, and unmatched content types return clean results without crashing.
# ===========================================================================


# --- Robust clean paths -----------------------------------------------------


def test_empty_text_yields_no_violations(tmp_path):
    """Empty input is on-brand by definition (audit: functionally correct)."""
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid:
            - "leverage"
        tone_rules:
          - id: "no_bang"
            description: "No double bang."
            pattern: "!!"
            fail_on_match: true
        content_rules:
          - id: "must_mention"
            description: "Landing must mention extensibility."
            applies_to: "landing_page"
            pattern: "extensible"
            fail_on_match: false
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    assert validator.validate("", "blog_post") == []


def test_empty_text_on_real_config_yields_no_violations():
    """Same, but exercised directly against the real shipped config."""
    validator = _real_validator()
    assert validator.validate("", "blog_post") == []


def test_content_type_with_no_matching_content_rules_yields_no_content_violation():
    """A content_type with no applies_to match (e.g. 'tweet' in the shipped
    config) gets zero content-specific rules and does not crash (audit:
    confirmed correct -- empty content-rule set)."""
    validator = _real_validator()
    # Clean tweet text: no avoid-word, no tone arm, and no tweet content_rule.
    assert validator.validate("A concise, on-brand product note.", "tweet") == []


def test_unknown_content_type_only_applies_global_rules(tmp_path):
    """An entirely unmapped content_type still applies vocab + tone rules and
    skips all content_rules without error."""
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid:
            - "synergy"
        tone_rules: []
        content_rules:
          - id: "landing_only"
            description: "Landing must mention pricing."
            applies_to: "landing_page"
            pattern: "pricing"
            fail_on_match: false
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    # The landing-only content_rule must NOT fire for an unrelated type, but
    # the global vocab rule still does.
    assert _messages(validator.validate("Pure synergy here.", "email_blast")) == [
        "Contains forbidden word: 'synergy'"
    ]


# --- Malformed config guards -----------------------------------------------


def test_tone_rule_missing_pattern_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules:
          - id: "broken_no_pattern"
            description: "This rule has no pattern."
        content_rules: []
        """,
    )
    with pytest.raises(ValueError):
        validator = BrandVoiceValidator(config_path=config)
        validator.validate("any text", "blog_post")


def test_tone_rule_missing_description_raises_value_error_at_load(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules:
          - id: "broken_no_description"
            pattern: "boom"
        content_rules: []
        """,
    )
    with pytest.raises(ValueError):
        validator = BrandVoiceValidator(config_path=config)
        validator.validate("this goes boom", "blog_post")


def test_none_tone_rules_section_does_not_crash(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid: []
        tone_rules:
        content_rules: []
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    # Intended: a None section is treated as empty, so this is just clean.
    assert validator.validate("ordinary text", "blog_post") == []


def test_none_avoid_section_does_not_crash(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid:
        tone_rules: []
        content_rules: []
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    assert validator.validate("ordinary text", "blog_post") == []


def test_empty_config_file_does_not_crash_validate(tmp_path):
    config = _write_config(tmp_path, "")
    validator = BrandVoiceValidator(config_path=config)
    # Intended: an empty config means no rules, so all text is on-brand.
    assert validator.validate("anything at all", "blog_post") == []


def test_non_string_text_raises_type_error(tmp_path):
    config = _write_config(
        tmp_path,
        """
        vocabulary:
          avoid:
            - "leverage"
        tone_rules: []
        content_rules: []
        """,
    )
    validator = BrandVoiceValidator(config_path=config)
    # Intended: a clear TypeError, not an opaque AttributeError on .lower().
    with pytest.raises(TypeError):
        validator.validate(None, "blog_post")


# --- Construction guard (confirmed-correct: missing file raises) -----------


def test_missing_config_file_raises_file_not_found(tmp_path):
    """__init__ already guards a missing config path (audit: correct)."""
    missing = tmp_path / "does_not_exist.yml"
    with pytest.raises(FileNotFoundError):
        BrandVoiceValidator(config_path=missing)


# --- CLI regression tests ---------------------------------------------------


def test_cli_returns_zero_for_clean_file(tmp_path):
    content = tmp_path / "clean_landing_page.md"
    content.write_text("Atlas is built around extensibility from day one.")

    result = _run_cli("--file", str(content), "--type", "landing_page")

    assert result.returncode == 0
    assert "PASS:" in result.stdout
    assert result.stderr == ""


def test_cli_returns_one_for_brand_voice_violations(tmp_path):
    content = tmp_path / "bad_landing_page.md"
    content.write_text("This is a game-changer!!")

    result = _run_cli("--file", str(content), "--type", "landing_page")

    assert result.returncode == 1
    assert "FAIL: Found 3 blocking brand voice violations" in result.stdout
    assert "[BLOCKER] vocabulary.avoid.game-changer" in result.stdout
    assert "Contains forbidden word: 'game-changer'" in result.stdout
    assert "[MAJOR] no_excessive_punctuation" in result.stdout
    assert _EXCLAMATION_TONE_MSG in result.stdout
    assert "[BLOCKER] landing_page_extensibility_mention" in result.stdout
    assert _LANDING_MUST_MENTION_MSG in result.stdout
    assert result.stderr == ""


def test_cli_prints_suggestion_for_discouraged_vocabulary(tmp_path):
    content = tmp_path / "discouraged_blog_post.md"
    content.write_text("The result is predictable for operators.")

    result = _run_cli("--file", str(content), "--type", "blog_post")

    assert result.returncode == 0
    assert "WARN: Found 1 advisory brand voice findings" in result.stdout
    assert "[NIT] vocabulary.use.predictable" in result.stdout
    assert "Prefer 'deterministic' over 'predictable'" in result.stdout
    assert "suggestion: Use 'deterministic' instead of 'predictable'" in result.stdout
    assert result.stderr == ""


def test_cli_returns_one_for_mixed_blocking_and_advisory_findings(tmp_path):
    content = tmp_path / "mixed_blog_post.md"
    content.write_text("The result is predictable and a game-changer.")

    result = _run_cli("--file", str(content), "--type", "blog_post")

    assert result.returncode == 1
    assert "FAIL: Found 1 blocking brand voice violations" in result.stdout
    assert "[BLOCKER] vocabulary.avoid.game-changer" in result.stdout
    assert "WARN: Found 1 advisory brand voice findings" in result.stdout
    assert "[NIT] vocabulary.use.predictable" in result.stdout
    assert "suggestion: Use 'deterministic' instead of 'predictable'" in result.stdout
    assert result.stderr == ""


def test_cli_returns_one_for_missing_file(tmp_path):
    missing = tmp_path / "missing.md"

    result = _run_cli("--file", str(missing), "--type", "blog_post")

    assert result.returncode == 1
    assert f"Error: File not found at {missing}" in result.stdout
    assert result.stderr == ""
