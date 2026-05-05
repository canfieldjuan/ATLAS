#!/usr/bin/env python3
"""Run scoped hybrid reasoning checks and emit a machine-readable report."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "hybrid_reasoning_checks_report.json"


def _run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    steps = [
        _run(["./scripts/run_reasoning_provider_port_compat_checks.sh"]),
        _run(["./scripts/run_reasoning_provider_port_tests.sh"]),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "all_passed": all(int(step["returncode"]) == 0 for step in steps),
        "pytest_skipped": any(
            "SKIP: pytest_asyncio is not installed" in str(step.get("stdout") or "")
            for step in steps
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote report: {REPORT_PATH}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
