# In-Flight PRs

Last updated: 2026-05-19T18:55Z by codex-2026-05-19

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-Content-Selling-Url-Source-Cli | `extracted_content_pipeline/campaign_source_adapters.py`; source-row CLIs; source adapter/generation/smoke tests | codex-2026-05-19 | Avoid concurrent edits to source-row default-field parsing or `--booking-url` CLI flags. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
