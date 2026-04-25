#!/usr/bin/env bash
# Sleep, then run the skip-gate audit. Used as a one-shot replacement for
# `at` since at(1) is not installed on this host. Invoked by nohup+disown
# so it survives the launching shell.
#
# Usage:
#   _audit_skip_gates_at_runner.sh SLEEP_SECS WINDOW_HOURS TAG
#
# Outputs:
#   /tmp/skip_gate_audit_<TAG>_<UTC_TS>.md          markdown report
#   /tmp/skip_gate_audit_<TAG>_<UTC_TS>.md.stdout   raw stdout/stderr
#   /tmp/skip_gate_audit.log                        one-line completion record

set -uo pipefail

SLEEP_SECS="${1:?sleep_seconds required}"
WINDOW_HOURS="${2:?window_hours required}"
TAG="${3:?tag required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG=/tmp/skip_gate_audit.log

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [arm] tag=${TAG} sleep=${SLEEP_SECS}s window=${WINDOW_HOURS}h pid=$$" >> "${LOG}"

sleep "${SLEEP_SECS}"

UTC_TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/tmp/skip_gate_audit_${TAG}_${UTC_TS}.md"

cd "${REPO}"
# shellcheck disable=SC1091
source .venv/bin/activate

python scripts/audit_pre_scrape_skip_gates.py \
    --window-hours "${WINDOW_HOURS}" \
    --output "${OUT}" \
    > "${OUT}.stdout" 2>&1
EXIT=$?

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [done] tag=${TAG} exit=${EXIT} report=${OUT}" >> "${LOG}"
exit ${EXIT}
