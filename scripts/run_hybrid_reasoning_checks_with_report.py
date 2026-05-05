#!/usr/bin/env python3
"""Run scoped hybrid reasoning checks and emit a machine-readable report."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "hybrid_reasoning_checks_report.json"


def _run(cmd: list[str]) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid reasoning checks and write JSON report")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output JSON report path (default: artifacts/hybrid_reasoning_checks_report.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    steps = [
        _run(["./scripts/run_reasoning_provider_port_compat_checks.sh"]),
        _run(["./scripts/run_reasoning_provider_port_tests.sh"]),
    ]
    report_path = args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
        "all_passed": all(int(step["returncode"]) == 0 for step in steps),
        "pytest_skipped": any(
            "SKIP: pytest_asyncio is not installed" in str(step.get("stdout") or "")
            for step in steps
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote report: {report_path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
