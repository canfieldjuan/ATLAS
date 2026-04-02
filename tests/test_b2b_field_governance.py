"""CI governance tests for B2B enrichment field ownership.

Static analysis -- scans Python source files for enrichment JSONB access
patterns and enforces the contract defined in _b2b_field_contracts.py.

No database, no application imports beyond the contract module.

Fails on:
  1. Enrichment fields accessed but not declared in the contract.
  2. Non-exempt modules with enrichment reads missing APPROVED or
     DEPRECATED markers.
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
]

# Matches enrichment->>'field' and enrichment->'field' (with optional alias)
ENRICHMENT_RE = re.compile(
    r"""(?:\w+\.)?enrichment\s*->>?\s*'([^']+)'""",
    re.IGNORECASE,
)

# Matches either marker type on a comment line
MARKER_RE = re.compile(
    r"#\s*(?:APPROVED|DEPRECATED)-ENRICHMENT-READ:",
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
    return rel_path in EXEMPT_MODULES


def _extract_enrichment_reads(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, field_name) pairs for enrichment reads."""
    hits: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return hits
    for lineno, line in enumerate(lines, 1):
        for m in ENRICHMENT_RE.finditer(line):
            hits.append((lineno, m.group(1)))
    return hits


def _has_marker_near(lines: list[str], target_lineno: int, window: int = 60) -> bool:
    """Check if a DEPRECATED or APPROVED marker exists within window lines above."""
    start = max(0, target_lineno - window - 1)
    end = target_lineno - 1
    for i in range(start, end):
        if MARKER_RE.search(lines[i]):
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
        for _, field in _extract_enrichment_reads(pyfile):
            if field not in seen and field not in FIELD_CONTRACTS:
                undeclared.append(field)
                seen.add(field)
    assert not undeclared, (
        "Enrichment fields accessed but not in _b2b_field_contracts.py:\n"
        + "\n".join(f"  - {f}" for f in sorted(undeclared))
    )


def test_non_exempt_reads_are_marked():
    """Non-exempt modules with enrichment reads must have APPROVED or DEPRECATED markers."""
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
            if not _has_marker_near(lines, lineno):
                unmarked.append(f"{rel}:{lineno} reads '{field}' without marker")
    assert not unmarked, (
        "Enrichment reads without APPROVED/DEPRECATED markers:\n"
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
    Identical to test_non_exempt_reads_are_marked but kept as a separate
    named test for clarity in CI output.
    """
    # Delegates to the same logic -- both tests enforce the same rule.
    # Kept separate so CI output clearly shows "no new ad-hoc reads" as a
    # distinct gate from "existing reads are marked".
    test_non_exempt_reads_are_marked()
