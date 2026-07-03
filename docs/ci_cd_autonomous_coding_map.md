# Atlas CI/CD Map for Long-Running Autonomous Coding

Issue: #1962

This document maps the current Atlas coding loop as it exists today. It is the
S1 inventory slice: describe the machine before changing it.

## Operating Loop

Atlas has two related workflows:

1. Ordinary interactive slices stop after the PR is opened or updated. The
   builder reports the PR URL, local checks, and current status, then waits for
   the operator signal.
2. Explicit long-running coding tasks separate push/review-event attention from
   scheduled green confirmation. A push/review-event hook is immediate only when
   the operator environment provides an external bridge; otherwise the session
   records that hook as unavailable and the local 30-minute watcher is the
   autonomous fallback. The builder does not actively poll GitHub between those
   wake-ups.

The production loop is:

```text
issue / operator request
  -> builder reads AGENTS.md + session state
  -> plan doc first
  -> implementation
  -> focused tests
  -> sync plan from real diff
  -> local PR review / push wrapper
  -> PR body contract
  -> GitHub Actions checks
  -> Codex/Copilot/human review
  -> push/review-event attention wakes the builder only via external bridge/operator signal
     and never authorizes merge
  -> live AI reconciliation
  -> scheduled watcher confirms green
  -> merge only when owned PR is clean and the arc has explicit merge authorization
  -> worktree teardown + plan archive
  -> next approved slice
```

The important property is that model memory is not trusted. The durable state
lives in the repo, PR body, CI checks, review threads, and
`SESSION_STATE.local.md`.

## Local Gates

| Gate | Entry point | What it protects |
|---|---|---|
| Plan-first scaffold | `scripts/new_pr_plan.sh` | Required seven-section plan shape before code |
| Plan sync | `scripts/sync_pr_plan.py` | Files touched and LOC estimate match the real diff |
| Local mechanical review | `scripts/local_pr_review.sh` | Plan shape, file claims, diff size, session drift, AI reconciliation record, whitespace |
| Pre-push audit core | `scripts/pre_push_audit.sh` | MCP docs, extracted manifest sync, UI CI enrollment, plan audits, ASCII Python |
| Push wrapper | `scripts/push_pr.sh` | Runs the managed local review path with the PR body context |
| PR open wrapper | `scripts/open_pr.sh` | Opens or updates the PR body through the repo-approved stdin shape |
| Package gauntlets | `scripts/run_extracted_pipeline_checks.sh` and package-specific wrappers | Package-specific import, sync, test, and standalone checks |

`local_pr_review.sh` is the local funnel. It calls `pre_push_audit.sh`, then
adds PR-specific checks such as extracted test enrollment, cross-session drift,
cross-layer caller hints, AI reconciliation body audit, plan/code consistency,
review-rule detection, and `git diff --check`.

## GitHub Gate Layers

### Trusted-base PR meta gates

These workflows use `pull_request_target` but execute trusted base-branch code
and inspect PR content as data. They exist because a PR can edit the gate it is
trying to pass.

| Workflow | Required job/context | Purpose |
|---|---|---|
| `ai_reconciliation_live.yml` | `live-reconciliation` | Fails when PR body claims AI findings are fixed/waived while Codex/Copilot threads remain open |
| `pr_body_contract.yml` | `pr-body-contract` | Ensures PR body names the plan and follows AGENTS.md body shape |
| `pre_push_audit.yml` | `pre-push-audit` | Runs local PR review bundle from trusted base against PR worktree data |
| `diff_budget.yml` | `diff-budget` | Enforces the 400 LOC soft cap or explicit diff-budget override |
| `gitleaks_baseline_growth_guard.yml` | `Gitleaks baseline growth guard` | Prevents PR-side poisoning of the historical secret baseline |

`scripts/check_required_status_checks.py` currently audits branch protection for
these high-risk required contexts: `live-reconciliation`, `diff-budget`,
`Gitleaks PR secret scan`, and `Gitleaks baseline growth guard`.

### PR secret and product gates

`security_guardrails.yml` runs `Gitleaks PR secret scan` on pull requests. The
heavy security jobs in that workflow are intentionally skipped on PRs and run on
push, manual dispatch, or schedule.

Product and package workflows use path filters so only relevant checks run for
most PRs. Examples include extracted packages, Content Ops lanes, deflection
delivery/Stripe/report checks, Atlas Intel UI, portfolio UI, invoicing, npm
packages, migrations, Reddit listening, and voice startup.

### Advisory and scheduled gates

Advisory checks surface quality signals without always blocking the PR. Scheduled
or manual workflows run heavier checks that would slow every small PR, such as
full security sweeps, DAST, repo-wide unit backstops, label sync, TTL purge, and
branch-protection audits.

## Workflow Inventory

| Workflow | Events | Jobs |
|---|---|---|
| Admin Costs Checks | pull_request, push | `admin-costs-checks` |
| AI Reconciliation (live) | pull_request_target, pull_request_review, pull_request_review_comment | `live-reconciliation`, `live-reconciliation-review-events` |
| AI Reconciliation (review retrigger) | workflow_run | `retrigger-required-context` |
| Atlas B2B Campaign Migration Checks | pull_request, push | `atlas-b2b-campaign-migration-checks` |
| Atlas Blog Public Checks | pull_request, push | `atlas-blog-public-checks` |
| Atlas Content Ops Auth Checks | pull_request, push | `atlas-content-ops-auth-checks` |
| Atlas Content Ops Claim Registry Checks | pull_request, push | `atlas-content-ops-claim-registry-checks` |
| Atlas Content Ops Deflection Delivery Checks | pull_request, push | `atlas-content-ops-deflection-delivery-checks` |
| Atlas Content Ops Deflection Report Checks | pull_request, push | `atlas-content-ops-deflection-report-checks` |
| Atlas Content Ops Deflection Stripe Paid Checks | pull_request, push | `atlas-content-ops-deflection-stripe-paid-checks` |
| Atlas Content Ops Generated Assets Checks | pull_request, push | `atlas-content-ops-generated-assets-checks` |
| Atlas Content Ops Input Provider Checks | pull_request, push | `atlas-content-ops-input-provider-checks` |
| Atlas Content Ops Macro Writeback Checks | pull_request, push | `atlas-content-ops-macro-writeback-checks` |
| Atlas Content Ops Review Workflow Checks | pull_request, push | `atlas-content-ops-review-workflow-checks` |
| Atlas Deflection Migration Apply Checks | pull_request, push | `atlas-deflection-migration-apply-checks` |
| Atlas Intel UI Checks | pull_request, push | `atlas-intel-ui-checks` |
| Atlas Invoicing Checks | pull_request, push | `atlas-invoicing-checks` |
| Atlas Main Voice Startup Checks | pull_request, push | `atlas-main-voice-startup-checks` |
| Atlas Migrations Runner Checks | pull_request, push | `atlas-migrations-runner-checks` |
| Atlas Reddit Listening Checks | pull_request | `atlas-reddit-tests` |
| Atlas Security Policy Docs Checks | pull_request, push | `atlas-security-policy-docs-checks` |
| Branch Protection Required Checks | workflow_dispatch, schedule, push | `required-status-checks` |
| Brand Voice Checks | pull_request, push | `brand-voice-checks` |
| Claude Code | issue_comment, pull_request_review_comment, issues, pull_request_review | `claude` |
| Content Ops Deflection Report TTL Purge | schedule, workflow_dispatch | `purge` |
| Diff Budget | pull_request_target | `diff-budget` |
| Extracted Competitive Intelligence Checks | pull_request, push | `extracted-competitive-intelligence-checks` |
| Extracted LLM Infrastructure Checks | pull_request, push | `extracted-llm-infra-checks` |
| Extracted Pipeline Checks | pull_request, push | `extracted-checks` |
| Extracted Umbrella Checks | pull_request, push | `extracted-umbrella-checks` |
| Gitleaks Baseline Growth Guard | pull_request_target | `gitleaks-baseline-guard` |
| Marketing Content Voice Check | pull_request | `validate-voice` |
| Maturity Sweep | pull_request, push | `maturity-sweep` |
| Maturity Sweep Competitive Intelligence Surface | pull_request, push | `competitive-intelligence-product-surface` |
| Maturity Sweep Deflection Content Ops | pull_request, push | `maturity-sweep-deflection-content-ops` |
| NPM Package Checks | pull_request, push | `npm-package-checks` |
| Portfolio UI Checks | pull_request, push | `portfolio-ui-checks` |
| PR Body Contract | pull_request_target | `pr-body-contract` |
| Pre-push Audit | pull_request_target, push | `pre-push-audit`, `pre-push-audit-main` |
| Repo Labels | push, workflow_dispatch, schedule | `sync-repo-labels` |
| Repo-Wide Unit Backstop | schedule, workflow_dispatch, pull_request | `repo-wide-unit-backstop` |
| Security DAST ZAP | workflow_dispatch, schedule | `zap-baseline` |
| Security Full Sweep | schedule, workflow_dispatch | `semgrep`, `gitleaks`, `sca` |
| Security Guardrails | pull_request, push, workflow_dispatch, schedule | `secrets-pr`, `secrets-full-history`, `pip-audit`, `osv-full`, `semgrep`, `trivy-config`, `checkov` |
| Semantic Diff Advisor (advisory) | pull_request | `semantic-diff-advisor` |

## Why This Works

The loop works because every repeated model failure is turned into a durable
artifact:

- Plan docs make intent reviewable before code is trusted.
- PR bodies mirror the plan so CI can validate the story the builder tells.
- Trusted-base workflows prevent a PR from weakening the gate that judges it.
- Secret scans block the highest-severity, lowest-effort failure mode early.
- Live AI reconciliation prevents stale "fixed" claims while review threads are
  still open.
- Session ownership state prevents one long-running agent from touching another
  lane's PR.
- Fix-mode batons keep red-check loops narrow after compaction.
- Push/review-event signals reduce red-review latency when an operator or
  external bridge wakes the builder, without becoming merge signals; even green
  event wakes wait for scheduled confirmation.
- Long-running scheduled watchers remove operator babysitting while preserving
  ownership and merge-safety rules.
- Head-SHA pins catch unexpected remote branch movement before a builder can
  overwrite or merge another actor's push.

The practical result is less token burn: agents spend less time re-orienting,
humans spend less time asking "is it green yet?", and reviewers spend more time
on judgment instead of bookkeeping.

## S1 Boundaries

This slice deliberately does not change workflows, branch protection, required
checks, path filters, or runtime. Follow-up slices should classify required vs
advisory checks, measure runtime, and propose speedups only after this inventory
is reviewed.
