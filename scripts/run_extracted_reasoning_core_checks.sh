#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash extracted/_shared/scripts/validate_extracted.sh extracted_reasoning_core
bash extracted/_shared/scripts/check_ascii_python.sh extracted_reasoning_core
python extracted/_shared/scripts/check_extracted_imports.py --no-atlas-fallback extracted_reasoning_core
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_reasoning_core
python scripts/smoke_extracted_reasoning_core_standalone.py

python -m pytest -q \
  tests/test_extracted_reasoning_core_*.py \
  tests/test_forbid_atlas_reasoning_imports.py

echo "All extracted_reasoning_core checks passed"
