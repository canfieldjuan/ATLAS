# Hybrid Extraction Plan (Reasoning Stack)

This plan follows the required four-phase workflow and is tailored to Atlas churn reasoning plus the already-extracted products.

## Goal
Build a hybrid extraction path that:
1. Reuses stable shared substrate from extracted packages.
2. Keeps product-specific reasoning producers behind explicit host ports.
3. Avoids churn-specific coupling leaking into non-churn products.

---

## Phase 1: Planning & Discovery

### 1) Review implementation plan before executing

We will implement in three tracks:

- **Track A (Shared substrate reuse):** standardize all new reasoning work on `extracted_llm_infrastructure` runtime-decoupled surfaces.
- **Track B (Consumer contract extraction):** promote typed reasoning readers/contracts as reusable consumers.
- **Track C (Producer isolation):** keep synthesis/pool-generation logic product-owned and accessed via provider ports.

Why:
- LLM infra is the most mature extracted boundary.
- Competitive-intel extraction is partially decoupled; deep task builders remain coupled.
- Content pipeline already uses host-owned reasoning handoff pattern.

### 2) Locate exact files needing updates

#### Architecture and planning docs
- `docs/churn_reasoning_engine_map.md`
- `docs/hybrid_extraction_implementation_plan.md` (this file)

#### Maturity references (used for guardrails and acceptance)
- `extracted_llm_infrastructure/STATUS.md`
- `extracted_competitive_intelligence/STATUS.md`
- `extracted_content_pipeline/STATUS.md`

#### Atlas reasoning producer/consumer boundaries
- `atlas_brain/autonomous/tasks/b2b_enrichment.py`
- `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py`
- `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py`
- `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py`
- `atlas_brain/autonomous/tasks/_b2b_reasoning_contracts.py`
- `atlas_brain/autonomous/tasks/_b2b_reasoning_atoms.py`
- `atlas_brain/autonomous/tasks/_b2b_cross_vendor_synthesis.py`

### 3) Identify precise insertion points (line-anchored)

- `extracted_llm_infrastructure/STATUS.md:44` — runtime decoupling complete marker.
- `extracted_competitive_intelligence/STATUS.md:73` — decoupling still pending.
- `extracted_content_pipeline/STATUS.md:98` and `:119` — host-owned reasoning boundary and handoff contract.
- `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py:11` — reasoning deferred to synthesis task.
- `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py:3` — synthesis orchestration entry point.
- `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py:563` — Phase 3 accessor section for consumer contract evolution.

### 4) Verify code blocks exist

Before each code PR:
- Re-run `rg -n` for target symbols/functions.
- Open line ranges with `nl -ba ... | sed -n 'start,endp'`.
- Confirm existing function signatures and expected call chains.

### 5) Impact analysis (dependencies/imports)

High-risk dependency surfaces:
- `b2b_reasoning_synthesis` imports `_b2b_reasoning_atoms`, `_b2b_reasoning_contracts`, and cross-vendor synthesis helpers.
- `_b2b_synthesis_reader` performs direct DB reads from `b2b_reasoning_synthesis` table and maps both v1/v2 forms.
- `b2b_churn_intelligence` is deterministic upstream and hands off to synthesis-first consumers.

Dependency rule:
- Do not alter existing public function signatures in these modules.
- Introduce new adapter functions/interfaces in new files where possible.

---

## Phase 2: Pre-Modification Validation

### 6) No assumptions - verify everything

Per file before edits:
1. Confirm symbol exists.
2. Confirm call sites.
3. Confirm migration/table references.
4. Confirm runtime import path behavior (especially extracted packages).

### 7) Check for hard-coded values

Run focused scans before and after each code PR:
- Search for inline constants in changed files (thresholds, model names, env names, table names).
- Keep defaults centralized in existing config layers; do not introduce new inline literals unless already pattern-consistent.

### 8) Type preservation

- Preserve existing `Any` usage when touching legacy code paths.
- Only add stricter typing in newly introduced adapter modules where clearly safe and non-breaking.

### 9) Unicode compliance (Python/tests)

- Python and test files must remain ASCII-only.
- If any copied text contains typographic unicode, normalize before commit.

---

## Phase 3: Implementation Rules (Execution Plan)

### 10) Atomic changes only

Planned PR sequence (one logical change each):

1. **PR-1 (Design contracts only):** add shared reasoning provider/consumer interface docs and acceptance criteria.
2. **PR-2 (Consumer boundary extraction):** add adapter layer so products read reasoning via typed readers/contracts, not raw synthesis dicts.
3. **PR-3 (Producer boundary port):** introduce host-provider port for reasoning producer inputs/outputs (similar to content pipeline pattern).
4. **PR-4 (Competitive-intel decoupling slice):** reduce remaining deep-builder coupling called out in status.
5. **PR-5 (Verification + migration guide):** add runbooks/checklists and cross-package compatibility matrix.

### 11) Block size limit

- Keep edits in small blocks (<=30 lines) unless completing a single cohesive logic unit.
- Prefer additive wrappers over broad rewrites.

### 12) No placeholders

- No TODO/stub/mock logic in production paths.
- If an implementation cannot be completed safely in one PR, defer it entirely and document explicitly in plan status.

### 13) No hard-coded values

- New thresholds or toggles must live in existing config structures or env-backed configuration modules.

### 14) Preserve breaking changes

- Do not change existing function signatures, DB schemas, or API response contracts in existing Atlas churn endpoints.
- New behavior must be opt-in via adapters/ports.

---

## Phase 4: Post-Modification Validation

### 15) Test each file after modification

For docs-only PRs:
- Markdown lint/build checks as available.

For code PRs:
- Run package-specific smoke/import checks already present in repo scripts.

### 16) Confirm no breaking changes

Validation matrix per PR:
- Atlas core import smoke.
- Extracted package import smoke.
- Existing MCP/API call shapes unchanged.

### 17) Remove hard-coded values

Post-change scans:
- Search changed files for introduced literals and ensure they are config-driven.

### 18) Type safety verification

- Confirm any newly added code uses the narrowest safe types.
- Preserve existing legacy `Any` where tightening would risk behavior drift.

---

## Concrete hybrid extraction rollout

### Stage A (now)
- Treat `extracted_llm_infrastructure` as the canonical shared runtime substrate.
- Do not duplicate routing/tracing/cache/cost logic.

### Stage B
- Standardize a **reasoning consumer contract** around typed reader outputs (`_b2b_synthesis_reader` pattern).
- Keep downstream products consuming contract objects only.

### Stage C
- Standardize a **reasoning producer port** (host-owned implementation) for pool/synthesis generation.
- Reuse deterministic utilities (hashing, lineage, quality gates) where semantics match.

### Stage D
- For each new product domain, decide:
  - **Reuse** if ontology/evidence semantics align.
  - **Rebuild producer** if ontology diverges.

### Exit criteria
- No direct Atlas-core imports from extracted products for reasoning generation paths.
- Producer logic interchangeable via explicit host port.
- Consumer products rely on typed contracts, not raw schema-specific payloads.
