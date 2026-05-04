# Per-Product State

Last updated: 2026-05-04T19:50Z by claude-2026-05-03-b

Cross-product state-of-the-world for the extraction effort. Update when a PR merges or a product's phase advances. See [`../COORDINATION.md`](../COORDINATION.md) for the protocol that governs edits to this file.

| Product | Phase | Most recent merged PR | Active PRs | Next milestone | Active hot zone |
|---|---|---|---|---|---|
| `extracted_llm_infrastructure` | 3 (runtime-decoupled + 100% standalone-operational; no OSS publish — internal refactor only) | #171 | — | Extraction effort terminal. PR-A6a #169 carved `ProviderCostSubConfig` into `_standalone/config.py`; PR-A6b #171 ported `SkillRegistry` substrate. Customer-facing API/SaaS work tracks under product roadmap (P1/P5/P6), not this scaffold. | none |
| `extracted_competitive_intelligence` | 2 in progress (standalone toggle surfaces landing) | #160 | — | Continue Phase 2 ownership of standalone-ready product surfaces | none |
| `extracted_content_pipeline` | 1 -> 2 (productization seams) | #167 | — | Continue remaining campaign orchestration/API seams after DB-backed review/export/send/progression/analytics paths landed | none |
| `extracted_reasoning_core` | 1 (scaffold + archetypes/evidence_map moved; PR-C1 series merged through #163) | #163 | — | Continue temporal/types/evidence_engine/API/wrapper follow-up slices per merged PR #82 audit | none |
| `extracted_quality_gate` | 1 (scaffold + 7 deterministic packs landed: product_claim core #85; safety-gate split #114; blog quality pack #118; campaign quality pack #120; witness specificity pack #125; evidence coverage gate #130; source-quality pack #132) | #154 | — | Decoupling work effectively complete; no OSS publish. Future quality-gate features land here as new packs when needed. | none |

Phase legend: 0 = pre-extraction (audit doc only). 1 = byte-for-byte scaffold, still imports from `atlas_brain`. 2 = standalone toggle loads local substrate (per-product env var: `EXTRACTED_LLM_INFRA_STANDALONE`, `EXTRACTED_COMP_INTEL_STANDALONE`, `EXTRACTED_PIPELINE_STANDALONE`, etc.; see `extracted/METHODOLOGY.md` for the canonical list). 3 = full Protocol-based decoupling, no `atlas_brain` runtime imports.

**Note:** `extracted_*` packages are internal decoupling refactors, not OSS libraries. Per 2026-05-04 product strategy, customer-facing surfaces are paid hosted APIs (P1 Amazon Seller Intelligence, P5/P6 B2B vendor retention/lead gen). Phase 3 is the terminal extraction milestone — no LICENSE / pyproject / PyPI publish work follows.
