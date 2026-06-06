from __future__ import annotations

import importlib.util
from pathlib import Path

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
    body = "## Summary\nstuff\n\n## AI reconciliation\n- All fixed or waived: Yes\n\n## Next\nx\n"
    section = aud.extract_section(body)
    assert section is not None
    assert "All fixed or waived: Yes" in section
    assert "Next" not in section  # bounded by the next same-level heading


def test_subheadings_stay_inside_record():
    # A record using "### Codex" subheadings must not be truncated, so a later
    # unresolved marker is still detected (fail closed, not fail open).
    aud = load_auditor()
    body = (
        "## AI reconciliation\n"
        "### Codex\n- All fixed or waived: Yes\n"
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


# --- resolved (happy) markers ----------------------------------------------

def test_all_fixed_or_waived_yes_passes():
    aud = load_auditor()
    body = "## AI reconciliation\n- AI findings reviewed: Yes\n- All fixed or waived: Yes\n"
    assert aud.reconciliation_errors(body, require=True) == []


def test_no_findings_marker_passes():
    aud = load_auditor()
    body = "## AI reconciliation\nNo outstanding findings.\n"
    assert aud.reconciliation_errors(body, require=True) == []


def test_inline_bold_label_record_passes():
    # AGENTS.md section 2a template shape: a one-line bold-label record whose
    # resolution marker is on the anchor line itself.
    aud = load_auditor()
    body = "## Summary\nx\n\n**AI reconciliation:** AI findings reviewed: Yes. All fixed or waived: Yes\n"
    assert aud.reconciliation_errors(body, require=True) == []


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
    assert any("no resolution" in e for e in errors)


def test_no_findings_waived_alone_is_not_resolution():
    # Allowed near-miss: "no findings waived" only says nothing was waived, not
    # that findings were handled, so on its own it must NOT count as resolved.
    aud = load_auditor()
    body = "## AI reconciliation\n- AI findings reviewed: Yes\n- No findings waived.\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("no resolution" in e for e in errors)


def test_fixed_then_no_findings_waived_passes():
    # But a real resolution marker ("all fixed or waived: yes") still passes,
    # even when "no findings waived" is also present.
    aud = load_auditor()
    body = "## AI reconciliation\n- All fixed or waived: Yes\n- No findings waived.\n"
    assert aud.reconciliation_errors(body, require=True) == []


# --- detection branches (each negative fixture) ----------------------------

def test_unresolved_marker_fails():
    aud = load_auditor()
    body = "## AI reconciliation\n- All fixed or waived: No\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("incomplete" in e for e in errors)


def test_open_findings_phrase_fails():
    aud = load_auditor()
    body = "## AI reconciliation\nTwo findings still open pending discussion.\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("incomplete" in e for e in errors)


def test_waiver_without_reason_fails():
    aud = load_auditor()
    body = "## AI reconciliation\n- All fixed or waived: Yes\n- Waived:\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("no reason" in e for e in errors)


def test_present_but_no_resolution_marker_fails():
    aud = load_auditor()
    body = "## AI reconciliation\n- AI findings reviewed: Yes\n"
    errors = aud.reconciliation_errors(body, require=False)
    assert any("no resolution" in e for e in errors)


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


# --- CLI exit-code contract -------------------------------------------------

def test_cli_exit_codes(tmp_path: Path):
    aud = load_auditor()

    ok = tmp_path / "ok.md"
    ok.write_text("## AI reconciliation\n- All fixed or waived: Yes\n", encoding="utf-8")
    assert aud.main(["--current-pr-body-file", str(ok)]) == 0

    bad = tmp_path / "bad.md"
    bad.write_text("## AI reconciliation\n- All fixed or waived: No\n", encoding="utf-8")
    assert aud.main(["--current-pr-body-file", str(bad)]) == 1

    # No body file + --require is a usage error (exit 2).
    assert aud.main(["--require"]) == 2

    # Missing file path is a usage error (exit 2).
    assert aud.main(["--current-pr-body-file", str(tmp_path / "nope.md")]) == 2
