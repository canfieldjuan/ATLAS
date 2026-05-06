# In-Flight PRs

Last updated: 2026-05-06T02:05Z by codex-2026-05-05-d20

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #317 | PR-D20: AI Content Ops host install readiness checker | `scripts/check_extracted_content_install.py`, `extracted_content_pipeline/campaign_install_check.py`, extracted content docs/tests/check runner | codex-2026-05-05-d20 | Avoid editing reasoning-core PR-D21 work or long-form backlog docs. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
