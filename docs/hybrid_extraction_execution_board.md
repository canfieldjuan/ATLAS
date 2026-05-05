# Hybrid Extraction Execution Board

This board operationalizes `docs/hybrid_extraction_implementation_plan.md` into PR-sized work with owners, estimates, risks, and acceptance tests.

## Program constraints

- Preserve existing Atlas API/task behavior (no breaking contracts).
- Use additive adapters/ports over rewrites.
- Keep producer logic product-owned when ontology diverges.
- Reuse `extracted_llm_infrastructure` substrate for routing/tracing/cache/cost.

## Milestone overview

| Milestone | Focus | Duration target | Exit gate |
|---|---|---:|---|
| M1 | Interface standardization | 1 sprint | Reader + provider interfaces merged |
| M2 | Consumer contract adoption | 1-2 sprints | Two products consume typed contract |
| M3 | Producer-port isolation | 2 sprints | Producer injectable via host port |
| M4 | Competitive-intel decoupling | 1-2 sprints | Remaining phase-3 couplings removed |
| M5 | Hardening + migration runbooks | 1 sprint | Validation matrix green + runbooks complete |

## PR execution queue

### PR-1: Shared reasoning interface spec (docs + contracts)

- **Owner**: Platform Architecture
- **Estimate**: 2-3 days
- **Scope**:
  - Define canonical consumer contract fields (confidence bands, reference IDs, witness lineage).
  - Define provider port contract for producer-side handoff payloads.
  - Map compatibility envelope for v1/v2 synthesis consumers.
- **Primary files**:
  - `docs/hybrid_extraction_implementation_plan.md`
  - `docs/churn_reasoning_engine_map.md`
  - new: `docs/reasoning_interface_contract.md`
- **Risks**:
  - Over-specification before real adoption feedback.
- **Acceptance tests**:
  - Contract doc includes field-level invariants and backward-compat rules.
  - Sign-off from AI Content Ops + Competitive Intelligence owners.

### PR-2: Consumer adapter package (typed reader façade)

- **Owner**: Competitive Intelligence Team
- **Estimate**: 4-6 days
- **Scope**:
  - Add adapter module that wraps existing synthesis-reader outputs into stable consumer DTOs.
  - Integrate adapter in one existing read path without changing response contract.
- **Primary files**:
  - `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py`
  - new: `atlas_brain/autonomous/tasks/_b2b_reasoning_consumer_adapter.py`
  - `atlas_brain/mcp/b2b/signals.py`
- **Risks**:
  - Hidden downstream assumptions on raw dict shape.
- **Acceptance tests**:
  - Existing MCP response schema unchanged.
  - Adapter path includes `metric_ids`/`witness_ids` lineage when available.
  - Smoke import and MCP tool tests pass.

### PR-3: Host provider port for reasoning producer input

- **Owner**: AI Content Ops Team
- **Estimate**: 5-8 days
- **Scope**:
  - Introduce explicit provider interface for producer input/output handoff.
  - Wire one product flow to consume producer payload via port instead of direct internal calls.
- **Primary files**:
  - `extracted_content_pipeline/services/campaign_reasoning_context.py`
  - `extracted_content_pipeline/campaign_reasoning_data.py`
  - `extracted_content_pipeline/STATUS.md`
  - new: `extracted_content_pipeline/services/reasoning_provider_port.py`
- **Risks**:
  - Missing fields in handoff payload for edge campaign cases.
- **Acceptance tests**:
  - Campaign generation succeeds with file-backed provider and postgres-backed provider.
  - No direct import of Atlas synthesis internals in extracted content runtime path.

### PR-4: Shared substrate enforcement (LLM infra)

- **Owner**: Platform Runtime Team
- **Estimate**: 3-5 days
- **Scope**:
  - Audit and enforce all new reasoning paths use `extracted_llm_infrastructure` services.
  - Add guardrails/checks to block direct atlas-core LLM service coupling in extracted products.
- **Primary files**:
  - `extracted_llm_infrastructure/STATUS.md`
  - `scripts/validate_extracted_llm_infrastructure.sh`
  - `scripts/validate_extracted_content_pipeline.sh`
  - `scripts/validate_extracted_competitive_intelligence.sh`
- **Risks**:
  - Validation scripts may miss dynamic imports.
- **Acceptance tests**:
  - Standalone smoke scripts pass for both extracted products.
  - New guardrails fail closed on forbidden import patterns.

### PR-5: Competitive-intel phase-3 decoupling slice

- **Owner**: Competitive Intelligence Team
- **Estimate**: 1-2 weeks
- **Scope**:
  - Remove one high-impact remaining phase-3 coupling path per PR (iterative).
  - Start with deep-builder access behind explicit host adapter protocols.
- **Primary files**:
  - `extracted_competitive_intelligence/autonomous/tasks/b2b_battle_cards.py`
  - `extracted_competitive_intelligence/autonomous/tasks/b2b_vendor_briefing.py`
  - `extracted_competitive_intelligence/autonomous/tasks/_b2b_cross_vendor_synthesis.py`
  - `extracted_competitive_intelligence/STATUS.md`
- **Risks**:
  - Runtime regressions in battle-card generation quality.
- **Acceptance tests**:
  - Standalone mode smoke check passes.
  - Core battle-card outputs preserve baseline contract fields.

### PR-6: Hybrid migration runbook + compatibility matrix

- **Owner**: Platform Architecture + DX
- **Estimate**: 3-4 days
- **Scope**:
  - Create runbook for “reuse vs rebuild producer” decisions by product.
  - Add compatibility matrix for ontology/evidence/governance fit checks.
- **Primary files**:
  - new: `docs/hybrid_reasoning_migration_runbook.md`
  - new: `docs/hybrid_reasoning_compatibility_matrix.md`
- **Risks**:
  - Teams bypassing decision process under deadline pressure.
- **Acceptance tests**:
  - At least two real product scenarios mapped through matrix and reviewed.

## Dependency graph

- PR-1 blocks PR-2 and PR-3.
- PR-2 and PR-3 can run in parallel after PR-1.
- PR-4 can start after PR-1 and should complete before PR-5 merge.
- PR-5 should start after PR-2 adapter conventions stabilize.
- PR-6 closes program after PR-2/PR-3/PR-5 learnings are captured.

## Validation matrix (per PR)

| Check | PR-1 | PR-2 | PR-3 | PR-4 | PR-5 | PR-6 |
|---|---|---|---|---|---|---|
| Import smoke (atlas core) | optional | required | optional | required | required | optional |
| Import smoke (extracted package) | optional | required | required | required | required | optional |
| API/MCP schema diff check | optional | required | optional | optional | required | optional |
| Runtime standalone check | optional | optional | required | required | required | optional |
| Hard-coded value scan | required | required | required | required | required | required |
| Unicode scan (py/tests) | n/a | required | required | required | required | n/a |

## Risk register

1. **Contract drift across products**
   - Mitigation: single contract owner + schema diff checks in CI.
2. **Hidden runtime coupling to atlas_brain internals**
   - Mitigation: standalone smoke + forbidden-import validation.
3. **Quality regressions in reasoning outputs**
   - Mitigation: baseline fixtures and before/after contract comparison.
4. **Scope creep into full producer rewrite**
   - Mitigation: enforce PR atomicity and milestone exit gates.

## Ready-to-start checklist

- [ ] Engineering owners assigned for PR-1 through PR-6.
- [ ] CI jobs mapped to acceptance tests for each PR.
- [ ] Product leads aligned on reuse-vs-rebuild decision criteria.
- [ ] Baseline output fixtures captured for affected reasoning surfaces.


## Progress ledger

### Completed slices

- [x] PR-1 contract foundation
  - Added `docs/reasoning_interface_contract.md`.
- [x] PR-2 consumer adapter seam
  - Added `atlas_brain/autonomous/tasks/_b2b_reasoning_consumer_adapter.py`.
  - Wired MCP overlays in `atlas_brain/mcp/b2b/signals.py`.
  - Added adapter + overlay regression tests.
- [x] PR-3 provider-port groundwork
  - Added `extracted_content_pipeline/services/reasoning_provider_port.py`.
  - Added `load_reasoning_provider_port(...)` wrapper.
  - Wired example/postgres generation entrypoints and CLI runners.
  - Added compatibility tests and migration docs.

### Remaining slices (current scope)

- [x] Add one consolidated compatibility test matrix run target for provider-port paths (`scripts/run_reasoning_provider_port_compat_checks.sh`).
- [x] Add execution-board CI checklist links to each acceptance test command.
- [x] Keep contract-impact annotations in every new PR body (scope guard compliance).

### Deferred (explicitly out of current slice)

- [ ] Producer internals rewrite (`b2b_reasoning_synthesis`, pool compression).
- [ ] Contract-breaking schema changes.
- [ ] New persistence artifacts for reasoning.

- `scripts/run_reasoning_provider_port_tests.sh` runs scoped pytest checks when `pytest_asyncio` is available, and prints a deterministic skip message otherwise.


### CI checklist links

Use these commands as the scoped compatibility checklist for the current wave:

- Provider-port compatibility matrix:
  - `./scripts/run_reasoning_provider_port_compat_checks.sh`
- Provider-port scoped pytest matrix (env-aware skip if `pytest_asyncio` missing):
  - `./scripts/run_reasoning_provider_port_tests.sh`
- Targeted compile checks (fast local fallback):
  - `python -m py_compile extracted_content_pipeline/campaign_reasoning_data.py`
  - `python -m py_compile atlas_brain/autonomous/tasks/_b2b_reasoning_consumer_adapter.py`

- PR body template for scope + contract metadata:
  - `docs/hybrid_pr_body_template.md`

- Unified scoped runner:
  - `./scripts/run_hybrid_reasoning_checks.sh`

- Unified runner with machine-readable report:
  - `./scripts/run_hybrid_reasoning_checks_with_report.py`
  - writes `artifacts/hybrid_reasoning_checks_report.json` including skip/pass state
