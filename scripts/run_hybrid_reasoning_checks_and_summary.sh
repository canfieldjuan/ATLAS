#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPORT_PATH="${1:-artifacts/hybrid_reasoning_checks_report.json}"
SUMMARY_PATH="${2:-artifacts/hybrid_reasoning_checks_summary.md}"

# Ensure both parent dirs exist so positional overrides into nested
# paths don't fail at the shell-redirect layer (the JSON runner already
# creates its own parent, but the summary uses a shell redirect which
# does not).
mkdir -p "$(dirname "$REPORT_PATH")" "$(dirname "$SUMMARY_PATH")"

# Capture the checks runner's exit code instead of letting set -e abort
# the wrapper. The whole point of generating a JSON report + markdown
# summary is to explain failures, so the summary must render even when
# checks fail. Final exit code is propagated below.
checks_exit=0
./scripts/run_hybrid_reasoning_checks_with_report.py --output "$REPORT_PATH" >/dev/null || checks_exit=$?

./scripts/render_hybrid_reasoning_report_summary.py --report "$REPORT_PATH" > "$SUMMARY_PATH"

echo "wrote report: $REPORT_PATH"
echo "wrote summary: $SUMMARY_PATH"

exit "$checks_exit"
