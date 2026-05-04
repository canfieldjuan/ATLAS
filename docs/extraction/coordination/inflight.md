# In-Flight PRs

Last updated: 2026-05-04T06:21Z by codex-2026-05-04-content

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1k, in flight) | PR-C1k: Rename `tests/test_extracted_reasoning_*.py` wrapper tests to clearer prefix | RENAME (3 files): `tests/test_extracted_reasoning_{archetypes,evidence_engine,temporal}.py` → `tests/test_extracted_content_pipeline_reasoning_{archetypes,evidence_engine,temporal}.py`. EDIT: `scripts/run_extracted_pipeline_checks.sh` (test paths). EDIT: `docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md` (amend the "Test Migration" section -- the original "redirect imports to core" plan would have broken the evidence_engine wrapper tests since the wrapper carries content-pipeline-specific `_DEFAULT_RULES`). No code changes. | claude-2026-05-03 | `tests/test_extracted_reasoning_*.py`; `tests/test_extracted_content_pipeline_reasoning_*.py`; `scripts/run_extracted_pipeline_checks.sh`; `docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md` |
| (PR-D10, in flight) | Add AI Content Ops queued send worker CLI | NEW: `extracted_content_pipeline/campaign_postgres_send.py`; NEW: `scripts/send_extracted_campaigns.py`; NEW: `tests/test_extracted_campaign_postgres_send.py`. EDIT: `extracted_content_pipeline/{README.md,STATUS.md,manifest.json}`; `extracted_content_pipeline/docs/{host_install_runbook.md,standalone_productization.md}`; `scripts/run_extracted_pipeline_checks.sh`; `tests/test_extracted_campaign_manifest.py`. | codex-2026-05-04-content | `extracted_content_pipeline/campaign_postgres_send.py`; `scripts/send_extracted_campaigns.py`; `tests/test_extracted_campaign_postgres_send.py`; listed content-pipeline docs and manifest files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
