#!/usr/bin/env python3
"""Verify every UI test:* script is run by its CI checks workflow.

Each ``*-ui`` package declares ``test:*`` npm scripts. The matching workflow
``.github/workflows/<pkg>_checks.yml`` must invoke each of them with an explicit
``npm run test:<name>`` step. A declared test that no workflow runs is silently
outside CI -- the drift class from issue #1318, where 9 of 24 atlas-intel-ui
suites were never executed. This audit fails closed on that drift, and on a UI
that declares test:* scripts but has no checks workflow at all.

The UI -> workflow mapping is derived by convention (dir ``foo-ui`` ->
``foo_ui_checks.yml``) rather than a hand-maintained list, so a newly added UI
with tests is covered automatically instead of having to be remembered.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_SUBDIR = Path(".github") / "workflows"

# Match an explicit `npm run test:<name>` invocation. Names use letters, digits,
# underscores, colons, and hyphens; the set comparison in audit_root avoids any
# prefix-collision false positives (test:abc is not satisfied by test:abcd).
_RUN_TOKEN = re.compile(r"npm run (test:[\w:-]+)")


@dataclass(frozen=True)
class EnrollmentRow:
    ui: str
    workflow: str
    status: str  # OK | NO_TESTS | MISSING_WORKFLOW | UNENROLLED
    missing: tuple[str, ...] = ()


def parse_test_scripts(package_json_text: str) -> set[str]:
    """Return the set of ``test:*`` script names declared in a package.json."""
    try:
        data = json.loads(package_json_text)
    except json.JSONDecodeError:
        return set()
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if not isinstance(scripts, dict):
        return set()
    return {
        name
        for name in scripts
        if isinstance(name, str) and name.startswith("test:")
    }


def parse_workflow_runs(workflow_text: str) -> set[str]:
    """Return the set of ``test:*`` scripts a workflow invokes via ``npm run``."""
    return set(_RUN_TOKEN.findall(workflow_text))


def workflow_name_for(ui_dir_name: str) -> str:
    """Map a UI directory to its conventional checks-workflow filename."""
    return f"{ui_dir_name.replace('-', '_')}_checks.yml"


def discover_ui_dirs(root: Path) -> list[Path]:
    """Return ``*-ui`` directories that contain a package.json, sorted by name."""
    return sorted(
        (p for p in root.glob("*-ui") if (p / "package.json").is_file()),
        key=lambda p: p.name,
    )


def audit_root(root: Path) -> list[EnrollmentRow]:
    """Audit every discovered UI package's test:* CI enrollment under ``root``."""
    rows: list[EnrollmentRow] = []
    for ui_dir in discover_ui_dirs(root):
        declared = parse_test_scripts(
            (ui_dir / "package.json").read_text(encoding="utf-8")
        )
        workflow_name = workflow_name_for(ui_dir.name)
        if not declared:
            rows.append(EnrollmentRow(ui_dir.name, workflow_name, "NO_TESTS"))
            continue
        workflow_path = root / WORKFLOWS_SUBDIR / workflow_name
        if not workflow_path.is_file():
            rows.append(
                EnrollmentRow(
                    ui_dir.name,
                    workflow_name,
                    "MISSING_WORKFLOW",
                    tuple(sorted(declared)),
                )
            )
            continue
        run = parse_workflow_runs(workflow_path.read_text(encoding="utf-8"))
        missing = tuple(sorted(declared - run))
        status = "OK" if not missing else "UNENROLLED"
        rows.append(EnrollmentRow(ui_dir.name, workflow_name, status, missing))
    return rows


def row_is_drift(row: EnrollmentRow) -> bool:
    return row.status in {"MISSING_WORKFLOW", "UNENROLLED"}


def main() -> int:
    rows = audit_root(REPO_ROOT)
    print("UI test:* CI enrollment")
    print("-" * 60)
    if not rows:
        print("no *-ui packages found")
        return 0
    drift = False
    for row in rows:
        if row.status == "OK":
            print(f"OK               {row.ui} -> {row.workflow}")
        elif row.status == "NO_TESTS":
            print(f"SKIP (no tests)  {row.ui}")
        elif row.status == "MISSING_WORKFLOW":
            drift = True
            print(
                f"MISSING WORKFLOW {row.ui}: expected {row.workflow} to run "
                f"{len(row.missing)} declared test(s)"
            )
            for name in row.missing:
                print(f"    - {name}")
        else:  # UNENROLLED
            drift = True
            print(
                f"UNENROLLED       {row.ui} -> {row.workflow}: "
                f"{len(row.missing)} declared test(s) not run by CI"
            )
            for name in row.missing:
                print(f"    - {name}")
    if drift:
        print("-" * 60)
        print(
            "Wire each unenrolled test into its workflow with a "
            "`run: npm run <test>` step (or remove the dead script)."
        )
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
