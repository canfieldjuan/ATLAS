#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/validate_extracted_content_pipeline.sh
bash scripts/check_ascii_python.sh
python scripts/check_extracted_imports.py

echo "All extracted_content_pipeline checks passed"
