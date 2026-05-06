#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPORT_PATH="${1:-artifacts/hybrid_reasoning_checks_report.json}"
SUMMARY_PATH="${2:-artifacts/hybrid_reasoning_checks_summary.md}"

./scripts/run_hybrid_reasoning_checks_with_report.py --output "$REPORT_PATH" >/dev/null
./scripts/render_hybrid_reasoning_report_summary.py --report "$REPORT_PATH" > "$SUMMARY_PATH"

echo "wrote report: $REPORT_PATH"
echo "wrote summary: $SUMMARY_PATH"
