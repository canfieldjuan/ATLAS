#!/usr/bin/env python3
"""Synthetic fixture oracle for Atlas Codex review scope.

This is not a Codex adapter. It is a deterministic policy checker that keeps
the intended review dispositions executable in tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Mapping


BLOCKER = "BLOCKER"
MAJOR = "MAJOR"
WAIVE_OUT_OF_SCOPE = "WAIVE_OUT_OF_SCOPE"
WAIVE_NIT = "WAIVE_NIT"
WAIVE_DUPLICATE = "WAIVE_DUPLICATE"
WAIVE_SPECULATIVE = "WAIVE_SPECULATIVE"
NO_FINDING = "NO_FINDING"

DISPOSITIONS = frozenset(
    {
        BLOCKER,
        MAJOR,
        WAIVE_OUT_OF_SCOPE,
        WAIVE_NIT,
        WAIVE_DUPLICATE,
        WAIVE_SPECULATIVE,
        NO_FINDING,
    }
)

MATERIAL_IMPACTS = frozenset(
    {
        "authorization",
        "billing",
        "ci",
        "correctness",
        "customer_output",
        "data_loss",
        "migration",
        "privacy",
        "security",
    }
)


@dataclass(frozen=True)
class Fixture:
    name: str
    finding: Mapping[str, object]
    expected: str


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "docs_process_noise",
        {
            "kind": "no_finding",
            "changed_path": "docs/process.md",
            "contract_met": True,
        },
        NO_FINDING,
    ),
    Fixture(
        "duplicate_instance",
        {
            "duplicate_of": "R2 missing regression proof for parser fallback",
            "impact": "correctness",
            "concrete_failure_path": True,
        },
        WAIVE_DUPLICATE,
    ),
    Fixture(
        "out_of_scope_hardening",
        {
            "in_scope": False,
            "impact": "observability",
            "hardening": True,
            "concrete_failure_path": False,
        },
        WAIVE_OUT_OF_SCOPE,
    ),
    Fixture(
        "missing_regression_test",
        {
            "in_scope": True,
            "missing_mandatory_proof": True,
            "rule": "R2",
        },
        BLOCKER,
    ),
    Fixture(
        "concrete_security_data_failure",
        {
            "in_scope": True,
            "impact": "security",
            "concrete_failure_path": True,
        },
        BLOCKER,
    ),
    Fixture(
        "speculative_risk_no_failure_path",
        {
            "in_scope": True,
            "impact": "performance",
            "speculative": True,
            "concrete_failure_path": False,
        },
        WAIVE_SPECULATIVE,
    ),
    Fixture(
        "nit_suppression",
        {
            "category": "nit",
            "in_scope": True,
            "one_line_changed_code_fix": False,
        },
        WAIVE_NIT,
    ),
    Fixture(
        "material_one_line_nit",
        {
            "category": "nit",
            "in_scope": True,
            "one_line_changed_code_fix": True,
            "materially_clarifies_changed_code": True,
        },
        MAJOR,
    ),
)


def classify_finding(finding: Mapping[str, object]) -> str:
    """Return the Atlas disposition for one synthetic review scenario."""

    if finding.get("kind") == "no_finding":
        return NO_FINDING
    if finding.get("duplicate_of"):
        return WAIVE_DUPLICATE
    if finding.get("category") == "nit":
        if finding.get("one_line_changed_code_fix") and finding.get(
            "materially_clarifies_changed_code"
        ):
            return MAJOR
        return WAIVE_NIT
    if finding.get("in_scope") is False:
        return WAIVE_OUT_OF_SCOPE
    if finding.get("missing_mandatory_proof"):
        return BLOCKER

    impact = str(finding.get("impact", ""))
    if finding.get("concrete_failure_path") and impact in MATERIAL_IMPACTS:
        return BLOCKER
    if finding.get("speculative") and not finding.get("concrete_failure_path"):
        return WAIVE_SPECULATIVE
    return MAJOR


def validate_fixtures(fixtures: tuple[Fixture, ...] = FIXTURES) -> list[str]:
    errors: list[str] = []
    for fixture in fixtures:
        if fixture.expected not in DISPOSITIONS:
            errors.append(f"{fixture.name}: unknown expected {fixture.expected}")
            continue
        actual = classify_finding(fixture.finding)
        if actual != fixture.expected:
            errors.append(f"{fixture.name}: expected {fixture.expected}, got {actual}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate built-in Codex review scope fixtures",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("nothing to do; pass --self-test")

    errors = validate_fixtures()
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"OK: {len(FIXTURES)} Codex review scope fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
