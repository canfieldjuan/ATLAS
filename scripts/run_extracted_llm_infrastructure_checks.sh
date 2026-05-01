#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/validate_extracted_llm_infrastructure.sh
bash scripts/check_ascii_python_llm_infrastructure.sh
python scripts/check_extracted_llm_infrastructure_imports.py

# Smoke imports run in two modes:
# - delegate (default): verifies the scaffold imports cleanly when
#   atlas_brain is on sys.path (Phase 1 contract).
# - standalone: sets EXTRACTED_LLM_INFRA_STANDALONE=1 and verifies the
#   five Phase 2 bridges load their local copies (no atlas_brain
#   dependency for the substrate).
python scripts/smoke_extracted_llm_infrastructure_imports.py
python scripts/smoke_extracted_llm_infrastructure_standalone.py

echo "All extracted_llm_infrastructure checks passed"
