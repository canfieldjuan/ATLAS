# In-Flight PRs

Last updated: 2026-05-05T16:58Z by codex-2026-05-05-d16

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #276 | Own Competitive Intelligence vendor briefing API ports | `extracted_competitive_intelligence/api/b2b_vendor_briefing.py`, `extracted_competitive_intelligence/services/b2b/vendor_briefing_api_ports.py`, `extracted_competitive_intelligence/templates/email/vendor_report_delivery.py`, `extracted_competitive_intelligence/templates/email/vendor_checkout_confirmation.py`, `extracted_competitive_intelligence/manifest.json`, `extracted_competitive_intelligence/STATUS.md`, competitive API tests | codex-2026-05-05 | Avoid vendor briefing API ownership and template bridge work until this lands |
| #283 | D16 campaign operations visibility telemetry | `extracted_content_pipeline/api/campaign_operations.py`, campaign operations API tests, content pipeline README/runbook/status | codex-2026-05-05-d16 | Avoid editing hosted campaign operations visibility/readiness wiring until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
