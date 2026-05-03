# Upcoming Queue

Last updated: 2026-05-03T23:31Z by codex-2026-05-03

Sequence reflects dependencies. Claim a slice (set Owner) before starting code so a parallel session does not pick the same one. See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-A1.5 | `extracted_llm_infrastructure` | claude-2026-05-03-b | none | Re-apply Copilot fixes that did not land in PR #87 merge: skills bridge stub, standalone `llm_exact_cache_enabled` flag, smoke script entries, STATUS detail rows. Opening immediately after PR-A2. |
| PR-A3 | `extracted_llm_infrastructure` | unclaimed | PR-A1 / #87 (merged) | New code: cache-savings persistence layer + migration. Closes the "$ saved by cache" telemetry gap. |
| PR-A4 | `extracted_llm_infrastructure` | unclaimed | PR-A2, PR-A3 | New code: drift report (local vs invoiced), budget gate, OpenAI provider adapter. May split if too large. |
| PR-B3 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | Safety-gate split: deterministic content/risk scan to core; approvals + audit log + DB to ports + Atlas adapter wrapper. |
| PR-B4 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | Blog + campaign quality packs over the core gate contract. |
| PR-B5 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | B2B evidence + witness + source-quality packs. |
| PR-C1 | `extracted_reasoning_core` | claude-2026-05-03 | PR #80, PR #82 (both merged) | Consolidate evidence/temporal/archetypes per merged PR #82 audit. NEW in core: `archetypes.py`, `evidence_engine.py` (slim conclusions+suppression surface), `evidence_map.yaml`, `temporal.py` (with `_numeric_value` / `_row_get` helpers + parameterized `MIN_DAYS_FOR_PERCENTILES`). Atlas-side: NEW `atlas_brain/reasoning/review_enrichment.py`; slim `atlas_brain/reasoning/evidence_engine.py`. Convert `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py` to re-export wrappers. EDIT `extracted_reasoning_core/api.py` (impl 3 stubs) and `extracted_reasoning_core/types.py` (rich `TemporalEvidence` + 4 sub-types + `ConclusionResult` + `SuppressionResult`). Rename + redirect `tests/test_extracted_reasoning_*.py`. PR #79 contract amendment lands in the same commit. |
| PR-D4 | `extracted_content_pipeline` | codex-2026-05-03 | PR-D3 / #99 (merged); avoid PR-C1 files | Make the markdown skill registry host-configurable and wire campaign CLIs to accept customer prompt roots without importing Atlas skills. |
