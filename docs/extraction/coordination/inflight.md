# In-Flight PRs

Last updated: 2026-05-04T10:07Z by codex-2026-05-04

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #159 | Add AI Content Ops sequence progression worker CLI | `extracted_content_pipeline/campaign_postgres_sequence_progression.py`; `scripts/progress_extracted_campaign_sequences.py`; `tests/test_extracted_campaign_postgres_sequence_progression.py`; content-pipeline docs/status/manifest/check wiring | codex-content-sequence-worker | Avoid content-pipeline sequence progression runner/CLI/doc wiring until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
