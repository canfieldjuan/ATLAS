#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ATLAS_AUDIT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_ROOT="${ATLAS_AUDIT_SCRIPT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

bash "$SCRIPT_ROOT/extracted/_shared/scripts/check_ascii_python.sh" extracted_content_pipeline
