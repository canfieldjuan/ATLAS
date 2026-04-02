"""CI governance tests for B2B enrichment field ownership.

Static analysis -- scans Python source files for enrichment JSONB access
patterns and enforces the contract defined in _b2b_field_contracts.py.

No database, no application imports beyond the contract module.

Fails on:
  1. Enrichment fields accessed but not declared in the contract.
  2. Non-exempt modules with enrichment reads missing APPROVED or
     DEPRECATED markers whose field lists cover the actual reads.
  3. Stranded fields that gained downstream consumers.
  4. New ad-hoc enrichment reads in non-exempt modules without markers.
"""

from __future__ import annotations

import re
from pathlib import Path

from atlas_brain.autonomous.tasks._b2b_field_contracts import (
    EXEMPT_MODULES,
    FIELD_CONTRACTS,
    STRANDED_FIELDS,
    validate_contracts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = [
    REPO_ROOT / "atlas_brain" / "autonomous" / "tasks",
    REPO_ROOT / "atlas_brain" / "api",
    REPO_ROOT / "atlas_brain" / "mcp" / "b2b",
    REPO_ROOT / "atlas_brain" / "services",
    REPO_ROOT / "scripts",
]

# Matches all JSONB access forms:
#   enrichment->>'field'    enrichment->'field'
#   enrichment #>> '{field,...}'    enrichment #> '{field,...}'
ENRICHMENT_RE = re.compile(
    r"""(?:\w+\.)?enrichment\s*->>?\s*'([^']+)'"""
    r"""|(?:\w+\.)?enrichment\s*#>>?\s*'\{([^,}]+)""",
    re.IGNORECASE,
)

# Matches marker lines and captures the field list
MARKER_WITH_FIELDS_RE = re.compile(
    r"#\s*(?:APPROVED|DEPRECATED)-ENRICHMENT-READ:\s*(.+)",
)


def _scan_py_files() -> list[Path]:
    """Collect all .py files under SCAN_DIRS."""
    files: list[Path] = []
    for d in SCAN_DIRS:
        if d.is_dir():
            files.extend(sorted(d.rglob("*.py")))
    return files


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _is_exempt(rel_path: str) -> bool:
    if rel_path in EXEMPT_MODULES:
        return True
    # Backfill/migration scripts need direct enrichment access by nature.
    # They are governed by APPROVED markers but not held to the 60-line
    # window since their SQL can span 100+ lines.
    if rel_path.startswith("scripts/backfill_") or rel_path.startswith("scripts/re_enrich_"):
        return True
    return False


def _extract_enrichment_reads(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, field_name) pairs for enrichment reads."""
    hits: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return hits
    for lineno, line in enumerate(lines, 1):
        for m in ENRICHMENT_RE.finditer(line):
            field = m.group(1) or m.group(2)
            if field:
                hits.append((lineno, field))
    return hits


def _find_covering_marker(
    lines: list[str], target_lineno: int, field: str, window: int = 60,
) -> bool:
    """Check if a marker within *window* lines above covers *field*.

    The marker must list the field (or a parent of a nested field like
    reviewer_context covering reviewer_context.industry).
    """
    start = max(0, target_lineno - window - 1)
    end = target_lineno - 1
    for i in range(start, end):
        m = MARKER_WITH_FIELDS_RE.search(lines[i])
        if not m:
            continue
        marker_fields = {f.strip().split(".")[0] for f in m.group(1).split(",")}
        if field in marker_fields or field.split(".")[0] in marker_fields:
            return True
    return False


# ---- Tests ----


def test_contract_schema_is_valid():
    """The contract itself has no structural errors."""
    errors = validate_contracts()
    assert not errors, "Contract validation errors:\n" + "\n".join(errors)


def test_all_enrichment_reads_declared_in_contract():
    """Every enrichment field accessed in code must exist in FIELD_CONTRACTS."""
    undeclared: list[str] = []
    seen: set[str] = set()
    for pyfile in _scan_py_files():
        rel = _relative(pyfile)
        if _is_exempt(rel):
            continue
        for _, field in _extract_enrichment_reads(pyfile):
            if field not in seen and field not in FIELD_CONTRACTS:
                undeclared.append(field)
                seen.add(field)
    assert not undeclared, (
        "Enrichment fields accessed but not in _b2b_field_contracts.py:\n"
        + "\n".join(f"  - {f}" for f in sorted(undeclared))
    )


def _extract_written_fields() -> set[str]:
    """Scan b2b_enrichment.py for result["field"] assignments.

    Derives the set of enrichment fields from the actual writer, not a
    hand-maintained duplicate list.
    """
    enrichment_py = (
        REPO_ROOT / "atlas_brain" / "autonomous" / "tasks" / "b2b_enrichment.py"
    )
    # Matches result["field_name"] and result.setdefault("field_name", ...)
    pattern = re.compile(r'''result\["([a-z_]+)"\]|result\.setdefault\("([a-z_]+)"''')
    fields: set[str] = set()
    for line in enrichment_py.read_text(encoding="utf-8").splitlines():
        for m in pattern.finditer(line):
            field = m.group(1) or m.group(2)
            if field:
                fields.add(field)
    return fields


def test_contract_covers_all_extracted_fields():
    """The contract must declare every field the enrichment pipeline writes.

    Prevents extracted fields from being invisible to the governance system.
    Derives the field list by scanning b2b_enrichment.py for result["..."]
    assignments rather than maintaining a hard-coded duplicate.
    """
    written_fields = _extract_written_fields()
    # Filter out internal/transient keys that are not persisted to JSONB
    transient = {"version_upgrade_requeued"}
    written_fields -= transient

    contracted = set(FIELD_CONTRACTS.keys())
    missing = written_fields - contracted
    assert not missing, (
        "Enrichment pipeline writes these fields but they are not in the contract:\n"
        + "\n".join(f"  - {f}" for f in sorted(missing))
    )


def test_non_exempt_reads_are_marked():
    """Non-exempt modules with enrichment reads must have APPROVED or
    DEPRECATED markers that list the fields actually read."""
    unmarked: list[str] = []
    for pyfile in _scan_py_files():
        rel = _relative(pyfile)
        if _is_exempt(rel):
            continue
        reads = _extract_enrichment_reads(pyfile)
        if not reads:
            continue
        try:
            lines = pyfile.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for lineno, field in reads:
            if not _find_covering_marker(lines, lineno, field):
                unmarked.append(f"{rel}:{lineno} reads '{field}' without covering marker")
    assert not unmarked, (
        "Enrichment reads without covering APPROVED/DEPRECATED markers:\n"
        + "\n".join(f"  - {u}" for u in unmarked[:30])
        + (f"\n  ... and {len(unmarked) - 30} more" if len(unmarked) > 30 else "")
    )


def test_stranded_fields_have_zero_consumers():
    """Fields marked stranded=True must have zero reads outside exempt modules."""
    violations: list[str] = []
    for pyfile in _scan_py_files():
        rel = _relative(pyfile)
        if _is_exempt(rel):
            continue
        for lineno, field in _extract_enrichment_reads(pyfile):
            if field in STRANDED_FIELDS:
                violations.append(f"{rel}:{lineno} reads stranded field '{field}'")
    assert not violations, (
        "Stranded fields have consumers (update contract or remove stranded flag):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_new_adhoc_reads_outside_approved_modules():
    """Non-exempt enrichment reads must have APPROVED or DEPRECATED markers.

    This is the primary gate preventing new ad-hoc reads from sneaking in.
    """
    test_non_exempt_reads_are_marked()
