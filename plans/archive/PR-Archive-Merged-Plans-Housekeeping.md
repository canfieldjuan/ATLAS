# PR-Archive-Merged-Plans-Housekeeping

## Why this slice exists

The operator asked to continue the plan-archive housekeeping after #1985
merged. `origin/main` still had 94 merged PR plan docs in the root,
including the just-merged content-ops CI cache plan and the latest Resolution
Audit parsing/clustering plan. That recreates the orientation tax the archive
lifecycle is meant to avoid: new sessions list `plans/` and see historical
merged plans mixed with active slice contracts.

Root cause: merged plan docs were landing on `main` faster than the teardown
ritual archived them. This fixes the root for the current backlog by running
the existing archive lifecycle on a dedicated housekeeping branch and leaving
only this active housekeeping plan in `plans/`.

## Scope (this PR)

Ownership lane: workflow/plan-archive-housekeeping
Slice phase: Workflow/process

1. Move every merged root PR plan doc present on `origin/main` into
   `plans/archive/`.
2. Regenerate `plans/INDEX.md` from the archive after the moves.
3. Keep this active housekeeping plan in the root so concurrent sessions can
   still tell which slice is in flight.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Archive-Merged-Plans-Housekeeping.md`
- `plans/archive/PR-ASR-Pip-Audit-Egg-Fragment.md`
- `plans/archive/PR-Archive-Current-Merged-Plans.md`
- `plans/archive/PR-Archive-Resolution-Audit-Zendesk-Plans.md`
- `plans/archive/PR-Asyncpg-Parser-Fake-Cleanup.md`
- `plans/archive/PR-Asyncpg-Small-Test-Fake-Cleanup.md`
- `plans/archive/PR-Atlas-Churn-UI-Tailwind-v4-Migration.md`
- `plans/archive/PR-Atlas-Intel-UI-Tailwind-v4-Migration.md`
- `plans/archive/PR-Atlas-UI-Tailwind-v4-Migration.md`
- `plans/archive/PR-Atlas-Video-Kafka-Cluster-ID-Env.md`
- `plans/archive/PR-Atlas-Video-Processing-Kafka-8.md`
- `plans/archive/PR-Atlas-Video-Processing-Postgres-18.md`
- `plans/archive/PR-Atlas-Vision-Python-3-14-Base-Image.md`
- `plans/archive/PR-Autonomous-Coding-Repo-Playbook.md`
- `plans/archive/PR-Backstop-Green-Follow-Up.md`
- `plans/archive/PR-Billing-Delta-Invoice-Payment-Failed-Live-Adapter.md`
- `plans/archive/PR-CI-CD-Runtime-Audit.md`
- `plans/archive/PR-CI-Hygiene-Stale-Test-Expectations.md`
- `plans/archive/PR-CI-Repo-Wide-Unit-Backstop.md`
- `plans/archive/PR-CI-Trigger-Coverage-Hardening.md`
- `plans/archive/PR-CSV-Owner-Lane-Vertical.md`
- `plans/archive/PR-Checkov-SARIF-Upload.md`
- `plans/archive/PR-Claude-Workflow-SHA-Pin-Test-Robust.md`
- `plans/archive/PR-CodeQL-Upload-SARIF-Action-Bump.md`
- `plans/archive/PR-Codex-Max-Files-Budget.md`
- `plans/archive/PR-Codex-Wake-Bridge.md`
- `plans/archive/PR-Content-Ops-Adversarial-Pass-Id-Contract.md`
- `plans/archive/PR-Content-Ops-CI-Cache-Install.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Artifact-Bundle.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Result-Writeup.md`
- `plans/archive/PR-Content-Ops-Claim-Evidence-Strictness-Guard.md`
- `plans/archive/PR-Content-Ops-Claim-Registry-Admin-Write.md`
- `plans/archive/PR-Content-Ops-StrEnum-Shim-Harmonization.md`
- `plans/archive/PR-Content-Ops-Support-Ticket-Date-Window-Diagnostics.md`
- `plans/archive/PR-Content-Ops-Verdict-Render-Evidence.md`
- `plans/archive/PR-Deflection-Billing-Reconciliation.md`
- `plans/archive/PR-Deflection-Cluster-Preview-Skip.md`
- `plans/archive/PR-Deflection-Date-Marker-Out-Of-Band.md`
- `plans/archive/PR-Deflection-Delivery-Idempotency.md`
- `plans/archive/PR-Deflection-Hybrid-Question-Clustering.md`
- `plans/archive/PR-Deflection-Landing-Demo-Contract-Example.md`
- `plans/archive/PR-Deflection-Landing-Demo-Example.md`
- `plans/archive/PR-Deflection-Landing-Demo-Full-Example.md`
- `plans/archive/PR-Deflection-Landing-Demo-High-Volume-Example.md`
- `plans/archive/PR-Deflection-Measured-Repetition.md`
- `plans/archive/PR-Deflection-Migration-Apply-Check.md`
- `plans/archive/PR-Deflection-PDF-Model-Renderer.md`
- `plans/archive/PR-Deflection-PII-Source-Decision-Preflight.md`
- `plans/archive/PR-Deflection-Paid-PDF-Required.md`
- `plans/archive/PR-Deflection-Proven-Answer-Gate.md`
- `plans/archive/PR-Deflection-Reconciliation-Null-Session-Dedup.md`
- `plans/archive/PR-Deflection-Report-Density-Navigation.md`
- `plans/archive/PR-Deflection-Snapshot-Visible-Rank-Partition.md`
- `plans/archive/PR-Deflection-Standard-Price-Terms.md`
- `plans/archive/PR-Deflection-Status-CSAT-Ingestion.md`
- `plans/archive/PR-Deflection-Suppressed-Repeat-Review-Queue.md`
- `plans/archive/PR-Deflection-Suppressed-Review-Key.md`
- `plans/archive/PR-Deflection-Synthetic-Ticket-Generator.md`
- `plans/archive/PR-Deflection-Zendesk-API-Export-Import.md`
- `plans/archive/PR-Dependabot-Body-Contract-Exemption.md`
- `plans/archive/PR-Dependabot-Package-Maintenance-Wave.md`
- `plans/archive/PR-Docker-Baseimage-Security-Bumps.md`
- `plans/archive/PR-Fable5-Lessons-Index.md`
- `plans/archive/PR-Fix-Mode-Claude-Code-Enforcement.md`
- `plans/archive/PR-Fix-Mode-Doc-Layer.md`
- `plans/archive/PR-Invoicing-MCP-OAuth-Test-Enrollment.md`
- `plans/archive/PR-LinkedIn-Marketing-Kit.md`
- `plans/archive/PR-Local-MCP-Eval-Live-Runbook.md`
- `plans/archive/PR-Maturity-Sweep-Deflection-AI-Content-Ops-Lanes.md`
- `plans/archive/PR-Negatives-Presence-Gate.md`
- `plans/archive/PR-PR-Body-Contract-Gate.md`
- `plans/archive/PR-Producer-Fidelity-Fixture-Factory.md`
- `plans/archive/PR-Product-Gap-Company-Organization-Alias.md`
- `plans/archive/PR-Product-Gap-Owner-Lane-Precedence.md`
- `plans/archive/PR-Product-Gap-Platform-Provenance.md`
- `plans/archive/PR-Product-Gaps-Action-Context.md`
- `plans/archive/PR-Product-Gaps-CSV-QA-Proof.md`
- `plans/archive/PR-Product-Gaps-Jira-Copy-Action.md`
- `plans/archive/PR-Reachability-Proof-Review-Rule.md`
- `plans/archive/PR-Reddit-Fit-Runner.md`
- `plans/archive/PR-Resolution-Audit-S1-Pipeline-Map.md`
- `plans/archive/PR-Resolution-Audit-S2-Parsing-Clustering.md`
- `plans/archive/PR-Resolution-Audit-S3-Performance.md`
- `plans/archive/PR-Resolution-Audit-S4-Synthesis.md`
- `plans/archive/PR-Retired-Failure-Detector-Workflow.md`
- `plans/archive/PR-Retired-Failure-Mode-Detector.md`
- `plans/archive/PR-Retired-Failure-Mode-Detectors.md`
- `plans/archive/PR-Reviewer-Boundary-Probe-Contract.md`
- `plans/archive/PR-Security-Blog-Admin-Authz.md`
- `plans/archive/PR-Security-Full-Sweep.md`
- `plans/archive/PR-Security-Gitleaks-Required-Checks.md`
- `plans/archive/PR-Security-Scanner-Ratchet.md`
- `plans/archive/PR-Test-Collection-Fixes.md`
- `plans/archive/PR-Trusted-Base-Gate-Execution.md`
- `plans/archive/PR-Watcher-Safety-Handoff.md`

### Review Contract

Acceptance criteria:

- `python scripts/archive_plans.py check` reports the root plan count below the
  threshold and shows only the active housekeeping plan remains.
- `plans/INDEX.md` is regenerated from the archive and includes the newly
  archived plan docs.
- Open PR plans are not touched: #1737 uses a branch-local plan doc that was
  not present in the `origin/main` root archive input. #1953 merged while this
  PR was in review, so its plan is now included in the archive batch.
- No product code, workflow logic, or test behavior changes.

Affected surfaces: plan docs and the generated plan archive index only.

Risk areas: accidentally archiving an active in-flight plan, stale index output,
or hiding a non-plan code change inside housekeeping.

Reviewer rules: R1 requirements match, R2 test evidence, R11 plan/code
consistency, R14 codebase verification.

## Mechanism

The existing archive tool is the mechanism:

```bash
python scripts/archive_plans.py archive
```

It moves root PR plan docs into `plans/archive/`, refuses filename
collisions before moving anything, and rewrites `plans/INDEX.md` from the
archived files. The active plan was scaffolded after the archive command so the
housekeeping PR's own plan stays in the root as the only in-flight plan.

## Intentional

- This is a broad mechanical rename slice by design. Splitting 94 historical
  plan moves into smaller batches would keep the root noisy and create repeated
  index conflicts.
- The active housekeeping plan is not archived in this PR; it should be moved
  by the normal teardown ritual after this PR merges.
- No attempt is made to clean unrelated worktrees or close unrelated open PRs.

## Deferred

- Archive this plan after this PR merges.

Parked hardening: none.

## Verification

- `python scripts/archive_plans.py archive` - archived 94 root plan docs and
  regenerated `plans/INDEX.md`.
- `python scripts/archive_plans.py check` - OK: 1 plan doc in root (threshold
  50).
- `git diff --cached --name-only | awk 'BEGIN{bad=0} !/^plans\// {print; bad=1}
  END{exit bad}'` - confirmed the staged diff is plan-only.
- `git ls-files 'plans/PR-*.md' | sort` - confirmed the active housekeeping plan
  is the only root plan tracked.
- `python scripts/sync_pr_plan.py plans/PR-Archive-Merged-Plans-Housekeeping.md
  --check` - plan table is in sync.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.archive-merged-plans-housekeeping-1988.local.md
  bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/pr-body-archive-merged-plans-housekeeping.md` - local PR review passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 96 |
| `plans/PR-Archive-Merged-Plans-Housekeeping.md` | 292 |
| `plans/archive/PR-ASR-Pip-Audit-Egg-Fragment.md` | 0 |
| `plans/archive/PR-Archive-Current-Merged-Plans.md` | 0 |
| `plans/archive/PR-Archive-Resolution-Audit-Zendesk-Plans.md` | 0 |
| `plans/archive/PR-Asyncpg-Parser-Fake-Cleanup.md` | 0 |
| `plans/archive/PR-Asyncpg-Small-Test-Fake-Cleanup.md` | 0 |
| `plans/archive/PR-Atlas-Churn-UI-Tailwind-v4-Migration.md` | 0 |
| `plans/archive/PR-Atlas-Intel-UI-Tailwind-v4-Migration.md` | 0 |
| `plans/archive/PR-Atlas-UI-Tailwind-v4-Migration.md` | 0 |
| `plans/archive/PR-Atlas-Video-Kafka-Cluster-ID-Env.md` | 0 |
| `plans/archive/PR-Atlas-Video-Processing-Kafka-8.md` | 0 |
| `plans/archive/PR-Atlas-Video-Processing-Postgres-18.md` | 0 |
| `plans/archive/PR-Atlas-Vision-Python-3-14-Base-Image.md` | 0 |
| `plans/archive/PR-Autonomous-Coding-Repo-Playbook.md` | 0 |
| `plans/archive/PR-Backstop-Green-Follow-Up.md` | 0 |
| `plans/archive/PR-Billing-Delta-Invoice-Payment-Failed-Live-Adapter.md` | 0 |
| `plans/archive/PR-CI-CD-Runtime-Audit.md` | 0 |
| `plans/archive/PR-CI-Hygiene-Stale-Test-Expectations.md` | 0 |
| `plans/archive/PR-CI-Repo-Wide-Unit-Backstop.md` | 0 |
| `plans/archive/PR-CI-Trigger-Coverage-Hardening.md` | 0 |
| `plans/archive/PR-CSV-Owner-Lane-Vertical.md` | 0 |
| `plans/archive/PR-Checkov-SARIF-Upload.md` | 0 |
| `plans/archive/PR-Claude-Workflow-SHA-Pin-Test-Robust.md` | 0 |
| `plans/archive/PR-CodeQL-Upload-SARIF-Action-Bump.md` | 0 |
| `plans/archive/PR-Codex-Max-Files-Budget.md` | 0 |
| `plans/archive/PR-Codex-Wake-Bridge.md` | 0 |
| `plans/archive/PR-Content-Ops-Adversarial-Pass-Id-Contract.md` | 0 |
| `plans/archive/PR-Content-Ops-CI-Cache-Install.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Artifact-Bundle.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Result-Writeup.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Evidence-Strictness-Guard.md` | 0 |
| `plans/archive/PR-Content-Ops-Claim-Registry-Admin-Write.md` | 0 |
| `plans/archive/PR-Content-Ops-StrEnum-Shim-Harmonization.md` | 0 |
| `plans/archive/PR-Content-Ops-Support-Ticket-Date-Window-Diagnostics.md` | 0 |
| `plans/archive/PR-Content-Ops-Verdict-Render-Evidence.md` | 0 |
| `plans/archive/PR-Deflection-Billing-Reconciliation.md` | 0 |
| `plans/archive/PR-Deflection-Cluster-Preview-Skip.md` | 0 |
| `plans/archive/PR-Deflection-Date-Marker-Out-Of-Band.md` | 0 |
| `plans/archive/PR-Deflection-Delivery-Idempotency.md` | 0 |
| `plans/archive/PR-Deflection-Hybrid-Question-Clustering.md` | 0 |
| `plans/archive/PR-Deflection-Landing-Demo-Contract-Example.md` | 0 |
| `plans/archive/PR-Deflection-Landing-Demo-Example.md` | 0 |
| `plans/archive/PR-Deflection-Landing-Demo-Full-Example.md` | 0 |
| `plans/archive/PR-Deflection-Landing-Demo-High-Volume-Example.md` | 0 |
| `plans/archive/PR-Deflection-Measured-Repetition.md` | 0 |
| `plans/archive/PR-Deflection-Migration-Apply-Check.md` | 0 |
| `plans/archive/PR-Deflection-PDF-Model-Renderer.md` | 0 |
| `plans/archive/PR-Deflection-PII-Source-Decision-Preflight.md` | 0 |
| `plans/archive/PR-Deflection-Paid-PDF-Required.md` | 0 |
| `plans/archive/PR-Deflection-Proven-Answer-Gate.md` | 0 |
| `plans/archive/PR-Deflection-Reconciliation-Null-Session-Dedup.md` | 0 |
| `plans/archive/PR-Deflection-Report-Density-Navigation.md` | 0 |
| `plans/archive/PR-Deflection-Snapshot-Visible-Rank-Partition.md` | 0 |
| `plans/archive/PR-Deflection-Standard-Price-Terms.md` | 0 |
| `plans/archive/PR-Deflection-Status-CSAT-Ingestion.md` | 0 |
| `plans/archive/PR-Deflection-Suppressed-Repeat-Review-Queue.md` | 0 |
| `plans/archive/PR-Deflection-Suppressed-Review-Key.md` | 0 |
| `plans/archive/PR-Deflection-Synthetic-Ticket-Generator.md` | 0 |
| `plans/archive/PR-Deflection-Zendesk-API-Export-Import.md` | 0 |
| `plans/archive/PR-Dependabot-Body-Contract-Exemption.md` | 0 |
| `plans/archive/PR-Dependabot-Package-Maintenance-Wave.md` | 0 |
| `plans/archive/PR-Docker-Baseimage-Security-Bumps.md` | 0 |
| `plans/archive/PR-Fable5-Lessons-Index.md` | 0 |
| `plans/archive/PR-Fix-Mode-Claude-Code-Enforcement.md` | 0 |
| `plans/archive/PR-Fix-Mode-Doc-Layer.md` | 0 |
| `plans/archive/PR-Invoicing-MCP-OAuth-Test-Enrollment.md` | 0 |
| `plans/archive/PR-LinkedIn-Marketing-Kit.md` | 0 |
| `plans/archive/PR-Local-MCP-Eval-Live-Runbook.md` | 0 |
| `plans/archive/PR-Maturity-Sweep-Deflection-AI-Content-Ops-Lanes.md` | 0 |
| `plans/archive/PR-Negatives-Presence-Gate.md` | 0 |
| `plans/archive/PR-PR-Body-Contract-Gate.md` | 0 |
| `plans/archive/PR-Producer-Fidelity-Fixture-Factory.md` | 0 |
| `plans/archive/PR-Product-Gap-Company-Organization-Alias.md` | 0 |
| `plans/archive/PR-Product-Gap-Owner-Lane-Precedence.md` | 0 |
| `plans/archive/PR-Product-Gap-Platform-Provenance.md` | 0 |
| `plans/archive/PR-Product-Gaps-Action-Context.md` | 0 |
| `plans/archive/PR-Product-Gaps-CSV-QA-Proof.md` | 0 |
| `plans/archive/PR-Product-Gaps-Jira-Copy-Action.md` | 0 |
| `plans/archive/PR-Reachability-Proof-Review-Rule.md` | 0 |
| `plans/archive/PR-Reddit-Fit-Runner.md` | 0 |
| `plans/archive/PR-Resolution-Audit-S1-Pipeline-Map.md` | 0 |
| `plans/archive/PR-Resolution-Audit-S2-Parsing-Clustering.md` | 0 |
| `plans/archive/PR-Resolution-Audit-S3-Performance.md` | 0 |
| `plans/archive/PR-Resolution-Audit-S4-Synthesis.md` | 0 |
| `plans/archive/PR-Retired-Failure-Detector-Workflow.md` | 0 |
| `plans/archive/PR-Retired-Failure-Mode-Detector.md` | 0 |
| `plans/archive/PR-Retired-Failure-Mode-Detectors.md` | 0 |
| `plans/archive/PR-Reviewer-Boundary-Probe-Contract.md` | 0 |
| `plans/archive/PR-Security-Blog-Admin-Authz.md` | 0 |
| `plans/archive/PR-Security-Full-Sweep.md` | 0 |
| `plans/archive/PR-Security-Gitleaks-Required-Checks.md` | 0 |
| `plans/archive/PR-Security-Scanner-Ratchet.md` | 0 |
| `plans/archive/PR-Test-Collection-Fixes.md` | 0 |
| `plans/archive/PR-Trusted-Base-Gate-Execution.md` | 0 |
| `plans/archive/PR-Watcher-Safety-Handoff.md` | 0 |
| **Total** | **388** |
