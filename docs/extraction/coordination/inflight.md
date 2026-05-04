# In-Flight PRs

Last updated: 2026-05-04T04:26Z by codex-2026-05-04

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1i, in flight) | PR-C1i: Route `extracted_content_pipeline/reasoning/evidence_engine.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/evidence_engine.py` (drop ~338-line drifted fork; replace with wrapper carrying rules as a Python dict + `from_rules(...)` route). EDIT: `extracted_reasoning_core/evidence_engine.py` (add `from_rules` classmethod, lazy yaml import + JSON suffix detection, drift-forward `_numeric_value` helper into numeric checks, `min_count`/`exists` operator parity, dual-form suppression). Existing `tests/test_extracted_reasoning_evidence_engine.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/evidence_engine.py`; `extracted_reasoning_core/evidence_engine.py`; `tests/test_extracted_reasoning_evidence_engine.py` |
| (pending) | Own competitive intelligence write/source impact surfaces | EDIT: `extracted_competitive_intelligence/{README.md,STATUS.md,manifest.json}`. EDIT: `extracted_competitive_intelligence/mcp/b2b/write_intelligence.py` and NEW `write_ports.py`. EDIT: `extracted_competitive_intelligence/services/scraping/capabilities.py`. EDIT: competitive extraction checks/workflow and NEW `tests/test_extracted_competitive_manifest.py`. | codex-2026-05-04 | `extracted_competitive_intelligence/**`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `scripts/smoke_extracted_competitive_intelligence_standalone.py`; `.github/workflows/extracted_competitive_intelligence_checks.yml`; `tests/test_extracted_competitive_manifest.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
