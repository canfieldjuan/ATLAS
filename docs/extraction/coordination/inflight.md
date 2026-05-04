# In-Flight PRs

Last updated: 2026-05-04T06:52Z by codex-2026-05-04-content

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1l, in flight) | PR-C1l: Document PR-C1 implementation outcomes in reasoning boundary audit | EDIT: `docs/extraction/reasoning_boundary_audit_2026-05-03.md` (append "PR-C1 Implementation Outcomes" section recording PR-C1a through PR-C1k slices, architectural deviations from the original plan, and the drift-forward pattern; mark which acceptance criteria from PR 2/PR 3 are now satisfied vs deferred to PR 4/PR 5/PR 6/PR 7). Doc-only change; no code touched. Closes the PR-C1 sequence. | claude-2026-05-03 | `docs/extraction/reasoning_boundary_audit_2026-05-03.md` |
| (PR-D10, in flight) | Add AI Content Ops queued send worker CLI | NEW: `extracted_content_pipeline/campaign_postgres_send.py`; NEW: `scripts/send_extracted_campaigns.py`; NEW: `tests/test_extracted_campaign_postgres_send.py`. EDIT: `extracted_content_pipeline/{README.md,STATUS.md,manifest.json}`; `extracted_content_pipeline/docs/{host_install_runbook.md,standalone_productization.md}`; `scripts/run_extracted_pipeline_checks.sh`; `tests/test_extracted_campaign_manifest.py`. | codex-2026-05-04-content | `extracted_content_pipeline/campaign_postgres_send.py`; `scripts/send_extracted_campaigns.py`; `tests/test_extracted_campaign_postgres_send.py`; listed content-pipeline docs and manifest files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
