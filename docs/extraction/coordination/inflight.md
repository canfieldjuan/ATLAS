# In-Flight PRs

Last updated: 2026-05-04T00:50Z by claude-2026-05-03-b

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1d, in flight) | PR-C1d: Slim `EvidenceEngine` core (conclusions + suppression) | NEW: `extracted_reasoning_core/evidence_engine.py` (conclusions + suppression surface only; per-review enrichment stays atlas-side until PR-C1e). EDIT: `extracted_reasoning_core/api.py` (wire `evaluate_evidence` stub). NEW: `tests/test_extracted_reasoning_core_evidence_engine.py`. | claude-2026-05-03 | `extracted_reasoning_core/evidence_engine.py`; `extracted_reasoning_core/api.py`; the new evidence-engine test file |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
