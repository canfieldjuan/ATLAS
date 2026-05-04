# In-Flight PRs

Last updated: 2026-05-04T09:18Z by codex-2026-05-04

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #155 | Add competitive challenger claims host port | EDIT: `extracted_competitive_intelligence/services/b2b/challenger_dashboard_claims.py`; EDIT: `extracted_competitive_intelligence/{README.md,STATUS.md,manifest.json}`; EDIT: `scripts/{smoke_extracted_competitive_intelligence_imports.py,smoke_extracted_competitive_intelligence_standalone.py,run_extracted_competitive_intelligence_checks.sh}`; EDIT: `tests/test_extracted_competitive_manifest.py`; ADD: `tests/test_extracted_competitive_challenger_claims_port.py` | codex-2026-05-04 | `extracted_competitive_intelligence/services/b2b/challenger_dashboard_claims.py`; competitive ProductClaim aggregation port |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
