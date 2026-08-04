from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_ai_reconciliation.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("audit_ai_reconciliation", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- section extraction / false-positive rejection -------------------------

def test_extract_section_finds_heading():
    aud = load_auditor()
    body = "## Summary\nstuff\n\n## AI reconciliation\n- no-findings\n\n## Next\nx\n"
    section = aud.extract_section(body)
    assert section is not None
    assert "no-findings" in section
    assert "Next" not in section  # bounded by the next same-level heading


def test_subheadings_stay_inside_record():
    # A record using "### Codex" subheadings must not be truncated, so a later
    # unresolved marker is still detected (fail closed, not fail open).
    aud = load_auditor()
    body = (
        "## AI reconciliation\n"
        "### Codex\n- no-findings\n"
        "### Copilot\n- fixed or waived: No\n"
        "## Next\nunrelated\n"
    )
    section = aud.extract_section(body)
    assert "### Copilot" in section
    assert "Next" not in section
    errors = aud.reconciliation_errors(body, require=False)
    assert any("incomplete" in e for e in errors)


def test_prose_mention_is_not_treated_as_section():
    # A lookalike: "reconciliation" appears in prose, not as a heading.
    aud = load_auditor()
    body = "## Summary\nWe reconciled the ledger and reconciliation went fine.\n"
    assert aud.extract_section(body) is None
    # And with no section, a non-require run reports no errors.
    assert aud.reconciliation_errors(body, require=False) == []


# --- structured resolution markers -----------------------------------------

def test_global_all_fixed_or_waived_alone_fails_as_vague():
    aud = load_auditor()
    body = "## AI reconciliation\n- AI findings reviewed: Yes\n- All fixed or waived: Yes\n"
    errors = aud.reconciliation_errors(body, require=True)
    assert any("must include 'no-findings'" in e for e in errors)


def test_no_findings_disposition_passes():
    aud = load_auditor()
    body = "## AI reconciliation\n- no-findings\n"
    assert aud.reconciliation_errors(body, require=True) == []


def test_all_allowed_dispositions_pass():
    aud = load_auditor()
    body = (
        "## AI reconciliation\n"
        "- Auth boundary deadlock -- fixed-in: atlas_brain/eom_api/auth.py and tests/test_eom_render_profile.py\n"
        "- Duplicate auth import thread -- waived-duplicate: same root decision as auth boundary deadlock\n"
        "- Optional docs polish -- waived-out-of-scope: parked for follow-up issue #2260\n"
        "- Future env loader risk -- waived-speculative: no concrete failure path from this diff\n"
        "- Rename suggestion -- waived-nit: skip-worthy style-only comment\n"
        "- Legacy docs citation -- not-applicable: AGENTS.md section 4a excludes unrelated hardening\n"
    )
    assert aud.reconciliation_errors(body, require=True) == []


def test_inline_bold_label_record_requires_structured_disposition():
    # AGENTS.md section 2a template shape: a one-line bold-label record whose
    # resolution marker is on the anchor line itself. Global claims are no
    # longer enough for a builder-owned PR body record.
    aud = load_auditor()
    body = "## Summary\nx\n\n**AI reconciliation:** AI findings reviewed: Yes. All fixed or waived: Yes\n"
    errors = aud.reconciliation_errors(body, require=True)
    assert any("must include 'no-findings'" in e for e in errors)


def test_inline_bold_label_unresolved_on_anchor_line_fails():
    aud = load_auditor()
    body = "**AI reconciliation:** All fixed or waived: No\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("incomplete" in e for e in errors)


def test_yes_requires_word_boundary():
    # "yesterday" must not satisfy the "...: yes" resolution marker.
    aud = load_auditor()
    body = "## AI reconciliation\n- All fixed or waived: yesterday we discussed it\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("exactly one allowed disposition" in e for e in errors)


def test_no_findings_waived_alone_is_not_resolution():
    # Allowed near-miss: "no findings waived" only says nothing was waived, not
    # that findings were handled, so on its own it must NOT count as resolved.
    aud = load_auditor()
    body = "## AI reconciliation\n- AI findings reviewed: Yes\n- No findings waived.\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("must include 'no-findings'" in e for e in errors)


def test_fixed_then_no_findings_waived_still_requires_structured_item():
    aud = load_auditor()
    body = "## AI reconciliation\n- All fixed or waived: Yes\n- No findings waived.\n"
    errors = aud.reconciliation_errors(body, require=True)
    assert any("exactly one allowed disposition" in e for e in errors)


# --- detection branches (each negative fixture) ----------------------------

def test_unresolved_marker_fails():
    aud = load_auditor()
    body = "## AI reconciliation\n- All fixed or waived: No\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("incomplete" in e for e in errors)


def test_negative_findings_reviewed_summary_fails_even_with_resolution_marker():
    aud = load_auditor()
    body = "## AI reconciliation\n- AI findings reviewed: No\n- no-findings\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("incomplete" in e for e in errors)


def test_open_findings_phrase_fails():
    aud = load_auditor()
    body = "## AI reconciliation\nTwo findings still open pending discussion.\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("incomplete" in e for e in errors)


def test_waiver_without_reason_fails():
    aud = load_auditor()
    body = "## AI reconciliation\n- waived-out-of-scope:\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("lacks an allowed disposition" in e or "must include" in e for e in errors)


def test_vague_fixed_comments_fails():
    aud = load_auditor()
    body = "## AI reconciliation\n- fixed comments from the review round\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("exactly one allowed disposition" in e for e in errors)


def test_placeholder_disposition_detail_fails():
    aud = load_auditor()
    body = "## AI reconciliation\n- Review thread -- fixed-in: TBD\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("no usable evidence" in e for e in errors)


def test_disposition_requires_named_finding():
    aud = load_auditor()
    body = "## AI reconciliation\n- fixed-in: tests/test_audit_ai_reconciliation.py\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("must name the finding" in e for e in errors)


def test_bullet_cannot_carry_multiple_dispositions():
    aud = load_auditor()
    body = (
        "## AI reconciliation\n"
        "- Auth bug -- fixed-in: atlas_brain/eom_api/auth.py -- waived-nit: skip-worthy\n"
    )
    errors = aud.reconciliation_errors(body, require=False)
    assert any("exactly one allowed disposition" in e for e in errors)


def test_fixed_in_requires_file_commit_or_test_evidence():
    aud = load_auditor()
    body = "## AI reconciliation\n- Auth bug -- fixed-in: trust me\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("must cite a commit, file, or test path" in e for e in errors)


def test_no_findings_cannot_mix_with_dispositions():
    aud = load_auditor()
    body = (
        "## AI reconciliation\n"
        "- no-findings\n"
        "- Thread one -- fixed-in: tests/test_audit_ai_reconciliation.py\n"
    )
    errors = aud.reconciliation_errors(body, require=False)
    assert any("cannot be mixed" in e for e in errors)


# --- require-mode -----------------------------------------------------------

def test_missing_section_passes_without_require():
    aud = load_auditor()
    body = "## Summary\njust a normal PR body\n"
    assert aud.reconciliation_errors(body, require=False) == []


def test_missing_section_fails_with_require():
    aud = load_auditor()
    body = "## Summary\njust a normal PR body\n"
    errors = aud.reconciliation_errors(body, require=True)
    assert any("no 'AI reconciliation' section" in e for e in errors)


def test_read_body_surfaces_missing_file(tmp_path: Path):
    aud = load_auditor()

    with pytest.raises(FileNotFoundError):
        aud.read_body(str(tmp_path / "missing.md"))


# --- CLI exit-code contract -------------------------------------------------

def test_cli_exit_codes(tmp_path: Path):
    aud = load_auditor()

    ok = tmp_path / "ok.md"
    ok.write_text("## AI reconciliation\n- no-findings\n", encoding="utf-8")
    assert aud.main(["--current-pr-body-file", str(ok)]) == 0

    bad = tmp_path / "bad.md"
    bad.write_text("## AI reconciliation\n- All fixed or waived: No\n", encoding="utf-8")
    assert aud.main(["--current-pr-body-file", str(bad)]) == 1

    unreadable = tmp_path / "unreadable.md"
    unreadable.write_text("## AI reconciliation\n- no-findings\n", encoding="utf-8")
    unreadable.unlink()
    assert aud.main(["--current-pr-body-file", str(unreadable)]) == 2

    # No body file + --require is a usage error (exit 2).
    assert aud.main(["--require"]) == 2

    # Missing file path is a usage error (exit 2).
    assert aud.main(["--current-pr-body-file", str(tmp_path / "nope.md")]) == 2
