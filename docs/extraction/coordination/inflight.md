# In-Flight PRs

Last updated: 2026-05-04T02:15Z by claude-2026-05-03-b

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1i, in flight) | PR-C1i: Route `extracted_content_pipeline/reasoning/evidence_engine.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/evidence_engine.py` (drop ~338-line drifted fork; replace with wrapper carrying rules as a Python dict + `from_rules(...)` route). EDIT: `extracted_reasoning_core/evidence_engine.py` (add `from_rules` classmethod, lazy yaml import + JSON suffix detection, drift-forward `_numeric_value` helper into numeric checks, `min_count`/`exists` operator parity, dual-form suppression). Existing `tests/test_extracted_reasoning_evidence_engine.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/evidence_engine.py`; `extracted_reasoning_core/evidence_engine.py`; `tests/test_extracted_reasoning_evidence_engine.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
