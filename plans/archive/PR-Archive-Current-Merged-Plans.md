# PR-Archive-Current-Merged-Plans

## Why this slice exists

`AGENTS.md` says root `plans/` should hold in-flight slices and merged plans
should move under `plans/archive/` after merge. The backlog warning is now over
the threshold again after the recent workflow/process and Content Ops merges:
current `main` has 71 already-merged PR plan docs in root.

This slice is a dedicated archive sweep from current `origin/main`. It moves
only plan docs already present on `main`; the new active plan for this PR stays
in root. The diff is intentionally over the normal 400 LOC soft cap because the
change is a mechanical lifecycle move of 71 docs plus an index refresh. Splitting
the archive sweep would leave the same orientation tax and root backlog behind.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Move current merged root plan docs from `plans/` to `plans/archive/`.
2. Keep this slice's active plan in root while the PR is in flight.
3. Regenerate `plans/INDEX.md` from the archive.
4. Add no product code, workflow logic, or test behavior changes.

### Review Contract

- Acceptance criteria:
  - [ ] Root `plans/` keeps `PR-Archive-Current-Merged-Plans.md` as the only
        active PR plan introduced by this branch.
  - [ ] Already-merged root plans present on `origin/main` are renamed into
        `plans/archive/` without content edits.
  - [ ] `plans/INDEX.md` reflects the refreshed archive count and entries.
  - [ ] No non-plan source, workflow, or test files change.
- Affected surfaces: plan-doc layout and archive index only.
- Risk areas: accidentally archiving this active plan, overwriting an existing
  archived plan with a reused name, or sweeping another session's unmerged plan.
- Reviewer rules triggered: R2, R14.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Archive-Current-Merged-Plans.md`
- `plans/archive/PR-Archive-Remaining-Merged-Plans.md`
- `plans/archive/PR-Brand-Voice-Severity-Gate.md`
- `plans/archive/PR-Brand-Voice-Strict-Mixed-Label.md`
- `plans/archive/PR-Brand-Voice-Strict-Mode.md`
- `plans/archive/PR-Brand-Voice-Structured-Findings.md`
- `plans/archive/PR-Brand-Voice-Suggested-Fixes.md`
- `plans/archive/PR-Content-Marketing-Brand-Voice-Checks.md`
- `plans/archive/PR-Content-Ops-Adapter-Contract-Example.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Benchmark-Core.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Fixture-CLI.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Fixture-Contract.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Fixture-Loader.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Prompt-Schema.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Result-Artifact.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Runner-Harness.md`
- `plans/archive/PR-Content-Ops-Claim-Registry-Persistence.md`
- `plans/archive/PR-Content-Ops-MCP-ChatGPT-Adapter-Help-Text.md`
- `plans/archive/PR-Content-Ops-MCP-ChatGPT-Adapter-OAuth-Rollout.md`
- `plans/archive/PR-Content-Ops-MCP-ChatGPT-Adapter-Port-Env.md`
- `plans/archive/PR-Content-Ops-MCP-ChatGPT-Search-Fetch-Adapter.md`
- `plans/archive/PR-Content-Ops-MCP-Claude-Hosted-OAuth-Compatibility.md`
- `plans/archive/PR-Content-Ops-MCP-Claude-Hosted-PKCE-Smoke.md`
- `plans/archive/PR-Content-Ops-MCP-Claude-Public-Client-Metadata.md`
- `plans/archive/PR-Content-Ops-MCP-Dual-Client-Rollout-Guidance.md`
- `plans/archive/PR-Content-Ops-MCP-Dual-Client-Smoke.md`
- `plans/archive/PR-Content-Ops-MCP-Launcher-Contract-Guard.md`
- `plans/archive/PR-Content-Ops-MCP-Live-Dual-Client-Rollout.md`
- `plans/archive/PR-Content-Ops-MCP-Live-Run-Artifact-Template.md`
- `plans/archive/PR-Content-Ops-MCP-OAuth-Discovery-Smoke.md`
- `plans/archive/PR-Content-Ops-MCP-OAuth-E2E.md`
- `plans/archive/PR-Content-Ops-MCP-OAuth-Launcher.md`
- `plans/archive/PR-Content-Ops-MCP-OAuth-Transport.md`
- `plans/archive/PR-Content-Ops-MCP-Token-Tenant-Binding.md`
- `plans/archive/PR-Content-Ops-Marketer-Verify-MCP-Shell.md`
- `plans/archive/PR-Content-Ops-Output-Variations.md`
- `plans/archive/PR-Content-Ops-Quality-Gate-Coverage-Rows.md`
- `plans/archive/PR-Content-Ops-Report-Claim-Id-Ledger-Validation.md`
- `plans/archive/PR-Content-Ops-Report-Claim-Ids.md`
- `plans/archive/PR-Content-Ops-Review-Service-Gate-Rows.md`
- `plans/archive/PR-Content-Ops-Review-Status-Mapping.md`
- `plans/archive/PR-Content-Ops-Tenant-Binding-Bridge.md`
- `plans/archive/PR-Content-Ops-Verify-Draft-Rich-Schema.md`
- `plans/archive/PR-Deflection-Async-Payment-Fulfillment.md`
- `plans/archive/PR-Deflection-Badge-Deterministic-Clustering.md`
- `plans/archive/PR-Deflection-CSV-Ingestion-Hardening.md`
- `plans/archive/PR-Deflection-Checkout-Authorization.md`
- `plans/archive/PR-Deflection-Delivery-Reconciliation.md`
- `plans/archive/PR-Deflection-Full-Thread-Export-Guidance.md`
- `plans/archive/PR-Deflection-Full-Volume-Submit-Limit.md`
- `plans/archive/PR-Deflection-Inline-Html-Strip-Before-Clustering.md`
- `plans/archive/PR-Deflection-Inspect-Preview-Gate.md`
- `plans/archive/PR-Deflection-Live-Proof-Report-Golden.md`
- `plans/archive/PR-Deflection-Paid-Funnel-Alert-Sink.md`
- `plans/archive/PR-Deflection-Paid-Funnel-Incidents.md`
- `plans/archive/PR-Deflection-Paid-Report-PDF-Attachment.md`
- `plans/archive/PR-Deflection-Provider-Export-Fixtures.md`
- `plans/archive/PR-Deflection-Repeat-Volume-Preview.md`
- `plans/archive/PR-Deflection-Resolution-Evidence-Live-Proof.md`
- `plans/archive/PR-Deflection-Synonym-Clustering-Recall.md`
- `plans/archive/PR-Dev-Workflow-Open-PR-Stdin-Wrapper.md`
- `plans/archive/PR-Gate-A-Live-Convergence-Proof.md`
- `plans/archive/PR-Gate-A-Live-Output-Quality-Proof.md`
- `plans/archive/PR-Gate-A-Messy-Ticket-Grounding-Rerun.md`
- `plans/archive/PR-Gate-A-Report-Live-Coverage.md`
- `plans/archive/PR-Maturity-Sweep-Advisory-CI.md`
- `plans/archive/PR-Review-Rule-Fix-Class-Not-Example.md`
- `plans/archive/PR-Reviewer-Codebase-Verification-Rule.md`
- `plans/archive/PR-Reviewer-R14-Linter-CI-Enrollment.md`
- `plans/archive/PR-Reviewer-Reconciliation-Live-CI.md`
- `plans/archive/PR-Semantic-Diff-Advisor-CI.md`
- `plans/archive/PR-Stale-Base-Push-Guard.md`

## Mechanism

The sweep starts from fresh `origin/main`, where root plan docs are already
merged by definition. Because this PR also needs its own active root plan, it
does not call `archive_plans.py archive` directly; that command would move this
in-flight plan too. Instead, the mechanical move list excludes
`plans/PR-Archive-Current-Merged-Plans.md`, moves every other root PR plan doc
to `plans/archive/`, and then runs:

```bash
python scripts/archive_plans.py index
```

The index command reads `plans/archive/` and rewrites `plans/INDEX.md` with the
new archive count and links.

## Intentional

- This is archive housekeeping only. No code, workflow, test, or product files
  are in scope.
- The active plan is deliberately not archived in this PR. It remains the
  contract for review and will be archived by a future sweep after merge.
- This does not reuse the stale local `claude/pr-archive-remaining-merged-plans`
  branch from merged PR #1339. This slice starts clean from current
  `origin/main`.

## Deferred

- None.

Parked hardening: none.

## Verification

- Confirmed no archive filename collisions before moving files.
- `python scripts/archive_plans.py index` passed and wrote an index with 993
  archived plans.
- `find plans -maxdepth 1 -type f -name 'PR-*.md'` shows only
  `PR-Archive-Current-Merged-Plans.md` in the root.
- `git diff --name-only origin/main...HEAD | rg -v '^plans/'` returned no
  non-plan files.
- `python scripts/archive_plans.py check` passed with one root plan doc, below
  threshold.
- `python scripts/sync_pr_plan.py plans/PR-Archive-Current-Merged-Plans.md origin/main --check` passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Archive-Current-Merged-Plans.md` passed.
- `python scripts/audit_review_rules_triggered.py --plan plans/PR-Archive-Current-Merged-Plans.md` passed.
- `git diff --check origin/main...HEAD` passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 73 |
| `plans/PR-Archive-Current-Merged-Plans.md` | 241 |
| `plans/archive/PR-Archive-Remaining-Merged-Plans.md` | 0 |
| `plans/archive/PR-Brand-Voice-Severity-Gate.md` | 0 |
| `plans/archive/PR-Brand-Voice-Strict-Mixed-Label.md` | 0 |
| `plans/archive/PR-Brand-Voice-Strict-Mode.md` | 0 |
| `plans/archive/PR-Brand-Voice-Structured-Findings.md` | 0 |
| `plans/archive/PR-Brand-Voice-Suggested-Fixes.md` | 0 |
| `plans/archive/PR-Content-Marketing-Brand-Voice-Checks.md` | 0 |
| `plans/archive/PR-Content-Ops-Adapter-Contract-Example.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Benchmark-Core.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Fixture-CLI.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Fixture-Contract.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Fixture-Loader.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Prompt-Schema.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Result-Artifact.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Runner-Harness.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Registry-Persistence.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-ChatGPT-Adapter-Help-Text.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-ChatGPT-Adapter-OAuth-Rollout.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-ChatGPT-Adapter-Port-Env.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-ChatGPT-Search-Fetch-Adapter.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Claude-Hosted-OAuth-Compatibility.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Claude-Hosted-PKCE-Smoke.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Claude-Public-Client-Metadata.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Dual-Client-Rollout-Guidance.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Dual-Client-Smoke.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Launcher-Contract-Guard.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Live-Dual-Client-Rollout.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Live-Run-Artifact-Template.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-OAuth-Discovery-Smoke.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-OAuth-E2E.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-OAuth-Launcher.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-OAuth-Transport.md` | 0 |
| `plans/archive/PR-Content-Ops-MCP-Token-Tenant-Binding.md` | 0 |
| `plans/archive/PR-Content-Ops-Marketer-Verify-MCP-Shell.md` | 0 |
| `plans/archive/PR-Content-Ops-Output-Variations.md` | 0 |
| `plans/archive/PR-Content-Ops-Quality-Gate-Coverage-Rows.md` | 0 |
| `plans/archive/PR-Content-Ops-Report-Claim-Id-Ledger-Validation.md` | 0 |
| `plans/archive/PR-Content-Ops-Report-Claim-Ids.md` | 0 |
| `plans/archive/PR-Content-Ops-Review-Service-Gate-Rows.md` | 0 |
| `plans/archive/PR-Content-Ops-Review-Status-Mapping.md` | 0 |
| `plans/archive/PR-Content-Ops-Tenant-Binding-Bridge.md` | 0 |
| `plans/archive/PR-Content-Ops-Verify-Draft-Rich-Schema.md` | 0 |
| `plans/archive/PR-Deflection-Async-Payment-Fulfillment.md` | 0 |
| `plans/archive/PR-Deflection-Badge-Deterministic-Clustering.md` | 0 |
| `plans/archive/PR-Deflection-CSV-Ingestion-Hardening.md` | 0 |
| `plans/archive/PR-Deflection-Checkout-Authorization.md` | 0 |
| `plans/archive/PR-Deflection-Delivery-Reconciliation.md` | 0 |
| `plans/archive/PR-Deflection-Full-Thread-Export-Guidance.md` | 0 |
| `plans/archive/PR-Deflection-Full-Volume-Submit-Limit.md` | 0 |
| `plans/archive/PR-Deflection-Inline-Html-Strip-Before-Clustering.md` | 0 |
| `plans/archive/PR-Deflection-Inspect-Preview-Gate.md` | 0 |
| `plans/archive/PR-Deflection-Live-Proof-Report-Golden.md` | 0 |
| `plans/archive/PR-Deflection-Paid-Funnel-Alert-Sink.md` | 0 |
| `plans/archive/PR-Deflection-Paid-Funnel-Incidents.md` | 0 |
| `plans/archive/PR-Deflection-Paid-Report-PDF-Attachment.md` | 0 |
| `plans/archive/PR-Deflection-Provider-Export-Fixtures.md` | 0 |
| `plans/archive/PR-Deflection-Repeat-Volume-Preview.md` | 0 |
| `plans/archive/PR-Deflection-Resolution-Evidence-Live-Proof.md` | 0 |
| `plans/archive/PR-Deflection-Synonym-Clustering-Recall.md` | 0 |
| `plans/archive/PR-Dev-Workflow-Open-PR-Stdin-Wrapper.md` | 0 |
| `plans/archive/PR-Gate-A-Live-Convergence-Proof.md` | 0 |
| `plans/archive/PR-Gate-A-Live-Output-Quality-Proof.md` | 0 |
| `plans/archive/PR-Gate-A-Messy-Ticket-Grounding-Rerun.md` | 0 |
| `plans/archive/PR-Gate-A-Report-Live-Coverage.md` | 0 |
| `plans/archive/PR-Maturity-Sweep-Advisory-CI.md` | 0 |
| `plans/archive/PR-Review-Rule-Fix-Class-Not-Example.md` | 0 |
| `plans/archive/PR-Reviewer-Codebase-Verification-Rule.md` | 0 |
| `plans/archive/PR-Reviewer-R14-Linter-CI-Enrollment.md` | 0 |
| `plans/archive/PR-Reviewer-Reconciliation-Live-CI.md` | 0 |
| `plans/archive/PR-Semantic-Diff-Advisor-CI.md` | 0 |
| `plans/archive/PR-Stale-Base-Push-Guard.md` | 0 |
| **Total** | **314** |
