#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash extracted/_shared/scripts/validate_extracted.sh extracted_evidence_to_story
bash extracted/_shared/scripts/check_ascii_python.sh extracted_evidence_to_story
python -m pytest \
  tests/test_extracted_evidence_to_story_sources.py \
  tests/test_extracted_evidence_to_story_claims.py

echo "All extracted_evidence_to_story checks completed"
