#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/validate_extracted_llm_infrastructure.sh
bash scripts/check_ascii_python_llm_infrastructure.sh
python scripts/check_extracted_llm_infrastructure_imports.py
python scripts/smoke_extracted_llm_infrastructure_imports.py

echo "All extracted_llm_infrastructure checks passed"
