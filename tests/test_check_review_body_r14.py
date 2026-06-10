from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_review_body_r14.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_review_body_r14", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALID_LGTM = """**LGTM. All review items resolved; CI green.**

**Reviewed head:** `55ead4e25651b4f7dfe953e923b338dc46191510`

**Codebase verification (R14):**
- Changed code inspected: `docs/REVIEWER_RULES.md:153`, `AGENTS.md:161`.
- Caller/test/artifact spot-checks: `rg "Codebase verification" AGENTS.md`.
- Not verified: None.

**Rule results:** R1 Pass, R10 Pass, R14 Pass.
"""


def test_valid_lgtm_body_passes():
    checker = load_checker()
    assert checker.r14_errors(VALID_LGTM) == []


def test_at_head_label_qualifiers_pass():
    checker = load_checker()
    body = VALID_LGTM.replace(
        "- Changed code inspected:",
        "- Changed code inspected at HEAD:",
    ).replace(
        "- Caller/test/artifact spot-checks:",
        "- Caller/test/artifact spot-checks at HEAD:",
    )
    assert checker.r14_errors(body) == []


def test_spot_checks_shorthand_label_passes():
    checker = load_checker()
    body = VALID_LGTM.replace(
        "- Caller/test/artifact spot-checks:",
        "- Spot-checks:",
    )
    assert checker.r14_errors(body) == []


def test_forced_lgtm_body_passes_without_auto_lgtm_phrase():
    checker = load_checker()
    body = VALID_LGTM.replace("**LGTM. All review items resolved; CI green.**", "Ready.")
    assert checker.r14_errors(body, verdict="lgtm") == []


def test_non_lgtm_body_does_not_require_r14():
    checker = load_checker()
    body = "BLOCKER - missing tests. No LGTM until fixed.\n"
    assert checker.r14_errors(body) == []


def test_explicit_non_lgtm_bypasses_incidental_lgtm_word():
    checker = load_checker()
    body = "LGTM on the idea, but not mergeable yet.\n"
    assert checker.r14_errors(body, verdict="non-lgtm") == []


def test_missing_reviewed_head_fails():
    checker = load_checker()
    body = VALID_LGTM.replace(
        "**Reviewed head:** `55ead4e25651b4f7dfe953e923b338dc46191510`\n\n",
        "",
    )
    errors = checker.r14_errors(body)
    assert "missing reviewed head SHA" in errors


def test_placeholder_reviewed_head_fails():
    checker = load_checker()
    body = VALID_LGTM.replace("55ead4e25651b4f7dfe953e923b338dc46191510", "<sha>")
    errors = checker.r14_errors(body)
    assert "missing reviewed head SHA" in errors


def test_missing_codebase_section_fails():
    checker = load_checker()
    body = VALID_LGTM.replace("**Codebase verification (R14):**", "**Verification:**")
    errors = checker.r14_errors(body)
    assert "missing Codebase verification (R14) section" in errors


def test_missing_changed_code_evidence_fails():
    checker = load_checker()
    body = VALID_LGTM.replace(
        "- Changed code inspected: `docs/REVIEWER_RULES.md:153`, `AGENTS.md:161`.",
        "- Changed code inspected: TODO",
    )
    errors = checker.r14_errors(body)
    assert (
        "missing non-placeholder changed-code evidence "
        "(expected 'Changed code inspected:' or 'Changed code inspected at HEAD:')"
    ) in errors


def test_missing_spot_check_fails():
    checker = load_checker()
    body = VALID_LGTM.replace(
        '- Caller/test/artifact spot-checks: `rg "Codebase verification" AGENTS.md`.',
        "- Caller/test/artifact spot-checks: None",
    )
    errors = checker.r14_errors(body)
    assert (
        "missing non-placeholder caller/test/artifact spot-checks "
        "(expected 'Caller/test/artifact spot-checks:' or "
        "'Caller/test/artifact spot-checks at HEAD:' or 'Spot-checks:' or "
        "'Spot-checks at HEAD:')"
    ) in errors


def test_missing_label_error_names_expected_changed_code_label():
    checker = load_checker()
    body = VALID_LGTM.replace(
        "- Changed code inspected: `docs/REVIEWER_RULES.md:153`, `AGENTS.md:161`.",
        "- Changed code checked: `docs/REVIEWER_RULES.md:153`, `AGENTS.md:161`.",
    )
    errors = checker.r14_errors(body)
    assert any("expected 'Changed code inspected:'" in error for error in errors)


def test_missing_label_error_names_expected_spot_check_labels():
    checker = load_checker()
    body = VALID_LGTM.replace(
        '- Caller/test/artifact spot-checks: `rg "Codebase verification" AGENTS.md`.',
        '- Verification commands: `rg "Codebase verification" AGENTS.md`.',
    )
    errors = checker.r14_errors(body)
    assert any("expected 'Caller/test/artifact spot-checks:'" in error for error in errors)
    assert any("'Spot-checks:'" in error for error in errors)


def test_missing_not_verified_disclosure_fails():
    checker = load_checker()
    body = VALID_LGTM.replace("- Not verified: None.", "- Not verified: ")
    errors = checker.r14_errors(body)
    assert "missing not-verified disclosure (expected 'Not verified:')" in errors


def test_not_verified_none_is_allowed():
    checker = load_checker()
    body = VALID_LGTM.replace("- Not verified: None.", "- Not verified: None")
    assert checker.r14_errors(body) == []


def test_missing_r14_rule_result_fails():
    checker = load_checker()
    body = VALID_LGTM.replace("R1 Pass, R10 Pass, R14 Pass", "R1 Pass, R10 Pass")
    errors = checker.r14_errors(body)
    assert "missing passing R14 rule result" in errors


def test_non_pass_r14_rule_result_fails():
    checker = load_checker()
    for status in ("Fail", "N/A", "Not applicable"):
        body = VALID_LGTM.replace("R14 Pass", f"R14 {status}")
        errors = checker.r14_errors(body)
        assert "R14 rule result must pass for LGTM" in errors


def test_cli_exit_codes(tmp_path: Path):
    checker = load_checker()

    ok = tmp_path / "ok.md"
    ok.write_text(VALID_LGTM, encoding="utf-8")
    assert checker.main([str(ok)]) == 0

    bad = tmp_path / "bad.md"
    bad.write_text("LGTM.\n", encoding="utf-8")
    assert checker.main([str(bad)]) == 1

    missing = tmp_path / "missing.md"
    assert checker.main([str(missing)]) == 2


def test_cli_forced_lgtm_catches_non_lgtm_shape(tmp_path: Path):
    checker = load_checker()
    body = tmp_path / "body.md"
    body.write_text("Looks good after discussion, but no explicit LGTM marker.\n", encoding="utf-8")
    assert checker.main([str(body), "--verdict", "lgtm"]) == 1
