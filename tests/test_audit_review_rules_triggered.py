from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_review_rules_triggered.py"

RULES_DOC = """
# Reviewer Rules Pack v1

## Path-based rule triggers

| Changed path glob | Rules triggered |
|---|---|
| `db/migrations/**`, `*.sql` migrations | R4, R2 (migration test) |
| `atlas_brain/api/**`, `atlas_brain/mcp/**` | R1, R2, R5 |
| `scripts/audit_*.py`, `scripts/check_*.py` | R2, R10 |
| `atlas-*-ui/**`, `*.tsx` | R9, R12 |
| `extracted_*/` synced files | R1, R10 |
| invoicing / billing / payment code | R3, R8 |

## AI-finding reconciliation

Out of the trigger table now.
"""


def load_auditor():
    spec = importlib.util.spec_from_file_location("audit_review_rules_triggered", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- table parsing ----------------------------------------------------------

def test_parse_trigger_table_splits_glob_and_prose_rows():
    aud = load_auditor()
    glob_rows, prose_rows = aud.parse_trigger_table(RULES_DOC)
    globs = {g for g, _ in glob_rows}
    assert "scripts/audit_*.py" in globs
    assert "*.tsx" in globs
    assert "extracted_*/" in globs
    # The header/separator rows are not parsed as triggers.
    assert all("changed path" not in g.lower() for g in globs)
    # Prose-only row is surfaced, not dropped.
    descs = {d for d, _ in prose_rows}
    assert any("invoicing" in d for d in descs)


def test_mixed_row_surfaces_prose_portion():
    # A row with both a glob and prose must enforce the glob AND surface the
    # prose part (not silently drop the non-glob trigger).
    aud = load_auditor()
    doc = (
        "## Path-based rule triggers\n\n"
        "| Changed path glob | Rules triggered |\n"
        "|---|---|\n"
        "| `**/auth/**`, login/token/permission code | R3, R2 |\n"
        "| `db/migrations/**`, `*.sql` migrations | R4 |\n"
    )
    glob_rows, prose_rows = aud.parse_trigger_table(doc)
    assert ("**/auth/**", frozenset({"R3", "R2"})) in glob_rows
    # auth row's prose portion is surfaced...
    assert any("login/token/permission" in d for d, _ in prose_rows)
    # ...but a glob row whose only extra word is filler ("migrations") is not.
    assert not any("migration" in d.lower() for d, _ in prose_rows)


def test_parse_ignores_table_outside_the_section():
    aud = load_auditor()
    # The "Out of the trigger table now." line is past the next heading.
    glob_rows, _ = aud.parse_trigger_table(RULES_DOC)
    assert all("Out of the trigger" not in g for g, _ in glob_rows)


# --- glob matching ----------------------------------------------------------

def test_path_matches_double_star_and_dirs():
    aud = load_auditor()
    assert aud.path_matches("atlas_brain/api/**", "atlas_brain/api/health.py")
    assert aud.path_matches("extracted_*/", "extracted_content_pipeline/foo.py")
    assert aud.path_matches("scripts/audit_*.py", "scripts/audit_x.py")
    assert not aud.path_matches("atlas_brain/api/**", "atlas_brain/mcp/x.py")


def test_path_matches_extension_glob_on_basename():
    aud = load_auditor()
    assert aud.path_matches("*.tsx", "atlas-intel-ui/src/Page.tsx")
    assert aud.path_matches("*.sql", "db/migrations/001_x.sql")
    assert not aud.path_matches("*.tsx", "atlas-intel-ui/src/Page.ts")


# --- required vs declared ---------------------------------------------------

def test_required_rules_tracks_triggering_paths():
    aud = load_auditor()
    glob_rows, _ = aud.parse_trigger_table(RULES_DOC)
    triggered = aud.required_rules(["scripts/audit_x.py", "db/migrations/9.sql"], glob_rows)
    assert "R2" in triggered and "R10" in triggered and "R4" in triggered
    assert "scripts/audit_x.py" in triggered["R10"]


def test_declared_rules_parses_line():
    aud = load_auditor()
    plan = "## Scope\n- Reviewer rules triggered: R2 (test evidence), R10 (maintainability)\n"
    assert aud.declared_rules(plan) == {"R2", "R10"}


def test_declared_rules_empty_when_line_absent():
    aud = load_auditor()
    assert aud.declared_rules("## Scope\nno such line here\n") == set()


def test_declared_rules_spans_wrapped_lines():
    # The bullet wraps across 80-col lines; the continuation carries R10.
    aud = load_auditor()
    plan = (
        "- Reviewer rules triggered: R2 (the audit is a detector -- failure branch\n"
        "  proven), R10 (maintainable).\n"
        "\n"
        "### Files touched\n"
    )
    assert aud.declared_rules(plan) == {"R2", "R10"}


def test_declared_rules_ignores_unindented_later_text():
    # A non-indented paragraph after the bullet is not part of it; its stray
    # rule id must not be absorbed (which would mask an omission).
    aud = load_auditor()
    plan = (
        "- Reviewer rules triggered: R2\n"
        "Later prose mentioning R5 that is not part of the bullet.\n"
    )
    assert aud.declared_rules(plan) == {"R2"}


# --- audit() happy path + detection branch ----------------------------------

def test_audit_passes_when_plan_declares_triggered_rules():
    aud = load_auditor()
    plan = "- Reviewer rules triggered: R2, R10\n"
    _, missing, _, _ = aud.audit(plan, ["scripts/audit_x.py"], RULES_DOC)
    assert missing == {}


def test_audit_flags_missing_triggered_rule():
    aud = load_auditor()
    # Diff triggers R2+R10 (audit script) but the plan only declares R2.
    plan = "- Reviewer rules triggered: R2\n"
    _, missing, _, _ = aud.audit(plan, ["scripts/audit_x.py"], RULES_DOC)
    assert "R10" in missing
    assert "scripts/audit_x.py" in missing["R10"]


def test_audit_surfaces_prose_rows():
    aud = load_auditor()
    _, _, prose_rows, _ = aud.audit("- Reviewer rules triggered: R2, R10\n", ["scripts/audit_x.py"], RULES_DOC)
    assert any("invoicing" in d for d, _ in prose_rows)


# --- CLI ---------------------------------------------------------------------

def test_cli_missing_rules_doc_is_usage_error(tmp_path: Path):
    aud = load_auditor()
    assert aud.main(["--reviewer-rules", str(tmp_path / "nope.md")]) == 2
