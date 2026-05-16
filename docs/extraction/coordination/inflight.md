# In-Flight PRs

Last updated: 2026-05-16T20:26Z by codex-2026-05-16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | Content Ops strict validation observability | `extracted_content_pipeline/content_ops_execution.py`, `extracted_content_pipeline/reasoning_signals.py`, `tests/test_extracted_content_ops_execution.py`, `extracted_content_pipeline/STATUS.md`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `plans/PR-Content-Ops-Strict-Validation-Observability.md` | codex-2026-05-16 | Avoid strict validation logging / reason-signal serialization edits. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
