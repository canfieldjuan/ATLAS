# In-Flight PRs

Last updated: 2026-05-04T04:56Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1k, in flight) | PR-C1k: Rename `tests/test_extracted_reasoning_*.py` wrapper tests to clearer prefix | RENAME (3 files): `tests/test_extracted_reasoning_{archetypes,evidence_engine,temporal}.py` → `tests/test_extracted_content_pipeline_reasoning_{archetypes,evidence_engine,temporal}.py`. EDIT: `scripts/run_extracted_pipeline_checks.sh` (test paths). EDIT: `docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md` (amend the "Test Migration" section -- the original "redirect imports to core" plan would have broken the evidence_engine wrapper tests since the wrapper carries content-pipeline-specific `_DEFAULT_RULES`). No code changes. | claude-2026-05-03 | `tests/test_extracted_reasoning_*.py`; `tests/test_extracted_content_pipeline_reasoning_*.py`; `scripts/run_extracted_pipeline_checks.sh`; `docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md` |
| (PR #129, in flight) | Own competitive intelligence write/source impact surfaces | EDIT: `extracted_competitive_intelligence/{README.md,STATUS.md,manifest.json}`. EDIT: `extracted_competitive_intelligence/mcp/b2b/write_intelligence.py` and NEW `write_ports.py`. EDIT: `extracted_competitive_intelligence/services/scraping/capabilities.py`. EDIT: competitive extraction checks/workflow and NEW `tests/test_extracted_competitive_manifest.py`. | codex-2026-05-04 | `extracted_competitive_intelligence/**`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `scripts/smoke_extracted_competitive_intelligence_standalone.py`; `.github/workflows/extracted_competitive_intelligence_checks.yml`; `tests/test_extracted_competitive_manifest.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
