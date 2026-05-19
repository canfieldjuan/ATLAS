# In-Flight PRs

Last updated: 2026-05-19T01:52Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| pending | Content Ops review-source quote-grade row prefilter | `scripts/export_content_ops_review_sources.py`, `tests/test_export_content_ops_review_sources.py`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`, `extracted_content_pipeline/STATUS.md`, `plans/PR-Content-Ops-Review-Source-Quote-Grade-Prefilter.md` | codex-2026-05-18 | Avoid editing the review-source exporter SQL row query or review-source readiness/backlog docs |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
