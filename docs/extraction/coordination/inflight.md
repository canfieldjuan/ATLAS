# In-Flight PRs

Last updated: 2026-05-04T06:56Z by codex-2026-05-04

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #145 | Own competitive prompt contract surface | EDIT: `extracted_competitive_intelligence/{README.md,STATUS.md,manifest.json}`; EDIT: `scripts/smoke_extracted_competitive_intelligence_standalone.py`; EDIT: `scripts/run_extracted_competitive_intelligence_checks.sh`; EDIT: `.github/workflows/extracted_competitive_intelligence_checks.yml`; EDIT: `tests/test_extracted_competitive_manifest.py`; ADD: `tests/test_extracted_competitive_prompt_contracts.py` | codex-2026-05-04 | `extracted_competitive_intelligence/reasoning/single_pass_prompts/*`; competitive manifest/check wiring |
| (PR-C1l, in flight) | PR-C1l: Document PR-C1 implementation outcomes in reasoning boundary audit | EDIT: `docs/extraction/reasoning_boundary_audit_2026-05-03.md` (append "PR-C1 Implementation Outcomes" section recording PR-C1a through PR-C1k slices, architectural deviations from the original plan, and the drift-forward pattern; mark which acceptance criteria from PR 2/PR 3 are now satisfied vs deferred to PR 4/PR 5/PR 6/PR 7). Doc-only change; no code touched. Closes the PR-C1 sequence. | claude-2026-05-03 | `docs/extraction/reasoning_boundary_audit_2026-05-03.md` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
