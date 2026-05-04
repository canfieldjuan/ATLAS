# Per-Product State

Last updated: 2026-05-04T09:15Z by claude-2026-05-03-b

Cross-product state-of-the-world for the extraction effort. Update when a PR merges or a product's phase advances. See [`../COORDINATION.md`](../COORDINATION.md) for the protocol that governs edits to this file.

| Product | Phase | Most recent merged PR | Active PRs | Next milestone | Active hot zone |
|---|---|---|---|---|---|
| `extracted_llm_infrastructure` | 3 (4 of 5 sub-tasks landed; only LICENSE+pyproject+PyPI publish remaining) | #150 | — | LICENSE + `pyproject.toml` + PyPI publish (needs license decision + maintainer email) | `extracted_llm_infrastructure/{LICENSE,pyproject.toml,README.md}` |
| `extracted_competitive_intelligence` | 2 in progress (standalone toggle surfaces landing) | #152 | #149 (PR-C2.1 follow-up to PR-C2 by claude-2026-05-03) | Continue Phase 2 ownership of standalone-ready product surfaces | `extracted_competitive_intelligence/reasoning/semantic_cache.py` (PR-C2.1 in flight) |
| `extracted_content_pipeline` | 1 -> 2 (productization seams) | #129 | — | Continue remaining campaign orchestration/API seams after the DB-backed review/export path landed | none |
| `extracted_reasoning_core` | 1 (scaffold + archetypes/evidence_map moved; PR-C1 series merged through #144) | #144 | #149 (PR-C2.1 follow-up by claude-2026-05-03) | Continue temporal/types/evidence_engine/API/wrapper follow-up slices per merged PR #82 audit | `extracted_reasoning_core/**` (api/types/archetypes/evidence_engine/evidence_map.yaml/temporal); `atlas_brain/reasoning/{evidence_engine.py, review_enrichment.py}`; `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py`; `tests/test_extracted_reasoning_*.py` |
| `extracted_quality_gate` | 1 (scaffold + 7 deterministic packs landed: product_claim core #85; safety-gate split #114; blog quality pack #118; campaign quality pack #120; witness specificity pack #125; evidence coverage gate #130; source-quality pack #132) | #132 | — | Open-source-grade README + LICENSE + pyproject.toml | none |

Phase legend: 0 = pre-extraction (audit doc only). 1 = byte-for-byte scaffold, still imports from `atlas_brain`. 2 = standalone toggle loads local substrate (per-product env var: `EXTRACTED_LLM_INFRA_STANDALONE`, `EXTRACTED_COMP_INTEL_STANDALONE`, `EXTRACTED_PIPELINE_STANDALONE`, etc.; see `extracted/METHODOLOGY.md` for the canonical list). 3 = full Protocol-based decoupling, no `atlas_brain` runtime imports.
