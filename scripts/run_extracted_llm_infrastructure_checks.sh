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

# Pytest contracts pinned by today's runtime-decoupling work:
#  - test_anthropic_batchable_protocol.py: PR-A5d Protocol satisfaction
#    (positive + negative cases against other LLM providers, companion
#    guard semantics, runtime_checkable presence).
#  - test_extracted_llm_infrastructure_standalone_config.py: PR-A6a
#    ProviderCostSubConfig defaults + ATLAS_PROVIDER_COST_* env-overrides.
#  - test_extracted_llm_infrastructure_skills.py: PR-A6b standalone
#    SkillRegistry substrate + EXTRACTED_LLM_INFRA_SKILLS_DIR override
#    + end-to-end llm_exact_cache.build_skill_messages.
python -m pytest \
  tests/test_anthropic_batchable_protocol.py \
  tests/test_extracted_llm_infrastructure_standalone_config.py \
  tests/test_extracted_llm_infrastructure_skills.py \
  -q

echo "All extracted_llm_infrastructure checks passed"
