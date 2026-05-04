# In-Flight PRs

Last updated: 2026-05-04T09:46Z by codex-2026-05-04

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C2.1, in flight) | PR-C2.1: Address late Copilot review on PR-C2 + fix mirror import regression | NEW: `extracted_content_pipeline/reasoning/semantic_cache.py` (wrapper -- the bundled PR-A5c sync on PR-C2 left `_b2b_cross_vendor_synthesis.py` mirror with a relative import to a missing module; this wrapper resolves it; standalone-mode env-gate added). EDIT: `atlas_brain/reasoning/semantic_cache.py` + `extracted_llm_infrastructure/reasoning/semantic_cache.py` (mirror -- make the new STALE_THRESHOLD comment module-path-agnostic since both copies share the validate-byte-identical invariant). EDIT: `scripts/run_extracted_pipeline_checks.sh` (wire all 7 `test_extracted_reasoning_core_*.py` files into standalone CI; coverage gap Copilot flagged). EDIT: `.github/workflows/extracted_pipeline_checks.yml` (add pyyaml to the standalone CI's pip install so the core evidence-engine YAML-load tests can run; expand path filter to include `extracted_reasoning_core/**` + the new test globs). | claude-2026-05-03 | `extracted_content_pipeline/reasoning/semantic_cache.py`; `atlas_brain/reasoning/semantic_cache.py`; `extracted_llm_infrastructure/reasoning/semantic_cache.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
