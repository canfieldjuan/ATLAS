# Per-Product State

Last updated: 2026-05-04T09:11Z by codex-2026-05-04

Cross-product state-of-the-world for the extraction effort. Update when a PR merges or a product's phase advances. See [`../COORDINATION.md`](../COORDINATION.md) for the protocol that governs edits to this file.

| Product | Phase | Most recent merged PR | Active PRs | Next milestone | Active hot zone |
|---|---|---|---|---|---|
| `extracted_llm_infrastructure` | 2 (standalone toggle landed; Phase 3 decoupling pending) | #89 | — (PR-A1.5 queued by claude-2026-05-03-b) | PR-A1.5 Copilot-fix replay, then cost-closure additions (PR-A3 -> A4) | `extracted_llm_infrastructure/{skills/__init__.py,_standalone/config.py,STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_{imports,standalone}.py` |
| `extracted_competitive_intelligence` | 2 in progress (standalone toggle surfaces landing) | #152 | #155 | Continue Phase 2 ownership of standalone-ready product surfaces | `extracted_competitive_intelligence/services/b2b/challenger_dashboard_claims.py` |
| `extracted_content_pipeline` | 1 -> 2 (productization seams) | #124 | — | Continue remaining campaign orchestration/API seams after the DB-backed review/export path landed | none |
| `extracted_reasoning_core` | 1 (scaffold + archetypes/evidence_map moved; PR-C1 follow-ups claimed) | #94 | — (PR-C1 follow-up slices claimed by claude-2026-05-03) | Continue temporal/types/evidence_engine/API/wrapper follow-up slices per merged PR #82 audit | `extracted_reasoning_core/**` (api/types/archetypes/evidence_engine/evidence_map.yaml/temporal); `atlas_brain/reasoning/{evidence_engine.py, review_enrichment.py}`; `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py`; `tests/test_extracted_reasoning_*.py` |
| `extracted_quality_gate` | 1 (scaffold + product_claim core landed via #85) | #130 | — (PR-B5c queued by claude-2026-05-03-b) | Source-quality pack (PR-B5c) | none |

Phase legend: 0 = pre-extraction (audit doc only). 1 = byte-for-byte scaffold, still imports from `atlas_brain`. 2 = standalone toggle loads local substrate (per-product env var: `EXTRACTED_LLM_INFRA_STANDALONE`, `EXTRACTED_COMP_INTEL_STANDALONE`, `EXTRACTED_PIPELINE_STANDALONE`, etc.; see `extracted/METHODOLOGY.md` for the canonical list). 3 = full Protocol-based decoupling, no `atlas_brain` runtime imports.
