#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DATE="${1:-$(date +%F)}"

cd "${REPO_ROOT}"
python scripts/generate_debt_register.py --date "${RUN_DATE}"
