#!/usr/bin/env python3
"""Render a concise markdown summary from hybrid reasoning check report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "hybrid_reasoning_checks_report.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render markdown summary from hybrid checks JSON report")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to hybrid reasoning checks report JSON (relative paths resolve against the repo root)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = args.report if args.report.is_absolute() else (ROOT / args.report)
    if not report_path.exists():
        print(f"report not found: {report_path}")
        return 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    print("### Hybrid reasoning checks")
    print(f"- all_passed: `{data.get('all_passed')}`")
    print(f"- pytest_skipped: `{data.get('pytest_skipped')}`")
    for step in data.get("steps", []):
        cmd = step.get("command")
        rc = step.get("returncode")
        print(f"- `{cmd}` -> rc={rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
