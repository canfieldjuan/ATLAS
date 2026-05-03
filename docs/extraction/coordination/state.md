# Per-Product State

Last updated: 2026-05-03T22:10Z by codex-2026-05-03

Cross-product state-of-the-world for the extraction effort. Update when a PR merges or a product's phase advances. See [`../COORDINATION.md`](../COORDINATION.md) for the protocol that governs edits to this file.

| Product | Phase | Most recent merged PR | Active PRs | Next milestone | Active hot zone |
|---|---|---|---|---|---|
| `extracted_llm_infrastructure` | 2 (standalone toggle landed; Phase 3 decoupling pending) | #89 | — (PR-A1.5 queued by claude-2026-05-03-b) | PR-A1.5 Copilot-fix replay, then cost-closure additions (PR-A3 -> A4) | `extracted_llm_infrastructure/{skills/__init__.py,_standalone/config.py,STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_{imports,standalone}.py` |
| `extracted_competitive_intelligence` | 1 (scaffold) | #80 | — | Phase 2 standalone toggle | none |
| `extracted_content_pipeline` | 1 -> 2 (productization seams) | #99 | — | Continue product-owned generation flows now that file-backed reasoning works on offline and DB-backed paths | none |
| `extracted_reasoning_core` | 1 (scaffold + archetypes/evidence_map moved; PR-C1 follow-ups claimed) | #94 | — (PR-C1 follow-up slices claimed by claude-2026-05-03) | Continue temporal/types/evidence_engine/API/wrapper follow-up slices per merged PR #82 audit | `extracted_reasoning_core/**` (api/types/archetypes/evidence_engine/evidence_map.yaml/temporal); `atlas_brain/reasoning/{evidence_engine.py, review_enrichment.py}`; `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py`; `tests/test_extracted_reasoning_*.py` |
| `extracted_quality_gate` | 1 (scaffold + product_claim core landed via #85) | #85 | — | Safety-gate split (PR-B3); blog + campaign packs (PR-B4) | none |

Phase legend: 0 = pre-extraction (audit doc only). 1 = byte-for-byte scaffold, still imports from `atlas_brain`. 2 = standalone toggle loads local substrate (per-product env var: `EXTRACTED_LLM_INFRA_STANDALONE`, `EXTRACTED_COMP_INTEL_STANDALONE`, `EXTRACTED_PIPELINE_STANDALONE`, etc.; see `extracted/METHODOLOGY.md` for the canonical list). 3 = full Protocol-based decoupling, no `atlas_brain` runtime imports.
