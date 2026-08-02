# Atlas CI Runtime and Duplication Audit

Issue: #1962

This is the S3 measurement slice after the CI/CD map and long-running watcher
handoff. It identifies where Atlas PR checks spend time today, where local and
GitHub gates intentionally overlap, and which speedups look safe enough to
consider next.

## Sample Window

Measured on 2026-07-03 from the latest 200 GitHub Actions runs using:

```bash
gh run list --repo canfieldjuan/ATLAS --limit 200 \
  --json databaseId,workflowName,displayTitle,event,status,conclusion,createdAt,updatedAt,headBranch,headSha
gh run view <run-id> --repo canfieldjuan/ATLAS --json jobs
```

The workflow table filters to completed, non-skipped `pull_request` and
`pull_request_target` runs. That produced 100 measured PR/PR-target runs.
Workflow duration is `updatedAt - createdAt`, so it includes runner setup and
GitHub envelope time. Job and step durations come from `gh run view --json jobs`
and are better for optimization decisions.

## Non-Negotiable Gates

Do not weaken these to save time. Branch protection requires only the four
contexts in the first table; the process guardrails in the second table are
still non-negotiable for Atlas PR work, but they are not current
branch-protection required contexts.

### Branch-Protection Gate Set

| Gate | Why it stays expected PR-blocking |
|---|---|
| `live-reconciliation` | Prevents stale "fixed/waived" AI-review claims while automated review threads remain open. |
| `Gitleaks PR secret scan` | Blocks leaked secrets before merge. |
| `Gitleaks baseline growth guard` | Prevents PR-side poisoning or expansion of the secret baseline. |
| `diff-budget` | Keeps slices small or forces an explicit override. Repo code expects this context, but live branch protection may need alignment. |

### Process Guardrails

| Gate | Why it stays non-negotiable |
|---|---|
| `pr-body-contract` | Keeps PR bodies tied to the plan contract, even though it is not in the current branch-protection required-context set. |
| `pre-push-audit` trusted-base local review | Re-runs the local mechanical bundle from trusted base so a PR cannot weaken its own audit. |
| Session-scoped state file ownership guard | Prevents long-running sessions from touching another lane's PR before review, push, comment handling, or merge. |

The optimization target is redundant runtime and broad triggering, not removing
these gates.

## Check Classes

| Class | Examples | Optimization posture |
|---|---|---|
| Branch-required meta gates | Live today: `live-reconciliation`, `Gitleaks PR secret scan`, `Gitleaks baseline growth guard`. Expected by `ci/gates.yml` / repo checker: those plus `diff-budget`, `plan-admission`, `session-lane`, `review-contract`, `pr-body-contract`. | Keep the intended gate set explicit; optimize only by making implementation faster without weakening coverage. |
| Process/meta CI contexts | `pr-body-contract`, `pre-push-audit`, branch-protection audit workflows | Keep as workflow guardrails; distinguish their latency from branch-required merge latency. |
| Product/package gates | Content-ops checks, extracted pipeline checks, Reddit listening checks, deflection package checks | Optimize with caching, narrower triggers, or safe decomposition while preserving package coverage. |
| Advisory checks | Advisory maturity sweeps, non-blocking detector runs, informational audits | Keep visible; reduce noise and runtime after required gates and product checks are stable. |
| Local-only gates | `scripts/local_pr_review.sh`, `scripts/push_pr.sh` hook path, package gauntlets run before push, session ownership guard | Treat as developer-loop cost and trusted-source duplication inputs; do not misread them as GitHub branch-protection contexts. |

## Runtime Snapshot

Median workflow duration from the sample. This table is a decision-focused
subset of the 100 measured runs and accounts for 91 of them. The remaining nine
were one-run, low-frequency workflows that did not change the ranked speedup
decision: Admin Costs Checks; Atlas Content Ops Auth Checks; Atlas Content Ops
Claim Registry Checks; Atlas Content Ops Deflection Delivery Checks; Atlas
Content Ops Generated Assets Checks; Atlas Content Ops Review Workflow Checks;
Atlas Main Voice Startup Checks; Extracted Umbrella Checks; and Semantic Diff
Advisor (advisory).

| Workflow | Event | Count | Median | Max | Notes |
|---|---:|---:|---:|---:|---|
| Atlas Content Ops Deflection Stripe Paid Checks | `pull_request` | 1 | 365s | 365s | Dependency install dominates. |
| Maturity Sweep | `pull_request` | 5 | 305s | 363s | Broad serial ratchet job. |
| Extracted Pipeline Checks | `pull_request` | 1 | 302s | 302s | Main check script dominates. |
| Atlas Content Ops Macro Writeback Checks | `pull_request` | 1 | 275s | 275s | Dependency install dominates. |
| Atlas Content Ops Input Provider Checks | `pull_request` | 1 | 248s | 248s | Install plus test runtime. |
| Atlas Content Ops Deflection Report Checks | `pull_request` | 1 | 240s | 240s | Dependency install dominates. |
| Maturity Sweep Deflection Content Ops | `pull_request` | 6 | 35s | 41s | Cheap lane-specific ratchet. |
| Pre-push Audit | `pull_request_target` | 14 | 33s | 511s | Typical process/meta job is small; outliers are event/review churn. |
| Atlas Reddit Listening Checks | `pull_request` | 1 | 18s | 18s | Fast targeted package check. |
| AI Reconciliation (live) | `pull_request_target` | 14 | 17s | 633s | Typical job is ~10s; review-event outliers affect run envelope. |
| Security Guardrails | `pull_request` | 9 | 15s | 168s | PR secret scan job is fast; skipped heavy jobs still add envelope. |
| PR Body Contract | `pull_request_target` | 14 | 14s | 155s | Fast process/meta context, not currently branch-required. |
| Diff Budget | `pull_request_target` | 14 | 12s | 143s | Fast expected meta gate; repo code expects branch protection to require it, but live protection may need alignment. |
| Gitleaks Baseline Growth Guard | `pull_request_target` | 9 | 10s | 164s | Fast required meta gate. |

## Step-Level Findings

### Maturity Sweep

The slow path is not runner setup. It is serial ratchet work in one job.

| Step | Median |
|---|---:|
| `Maturity sweep atlas_brain B2a support ratchet gates` | 61s |
| `Maturity sweep Phase C4 scripts ratchet gate` | 43s |
| `Maturity sweep atlas_brain B2b service/comms ratchet gates` | 42s |
| `Maturity sweep (advisory, non-blocking)` | 18s |
| `Maturity sweep atlas_brain B2c core-risk ratchet gates` | 18s |
| `Maturity sweep atlas_brain/autonomous ratchet gate` | 18s |
| `Maturity sweep ratchet gate` | 18s |
| `Maturity sweep atlas_brain B2d runtime-control ratchet gates` | 15s |
| `Maturity sweep Phase C1 extracted core ratchet gates` | 14s |

The workflow runs all of these in sequence. Parallelizing independent lane
groups would preserve the ratchets while reducing the wall clock toward the
slowest group instead of the sum.

### Product and Package Checks

Several checks spend more time installing dependencies than running their
targeted tests.

| Workflow | Dominant install/setup | Targeted test/check |
|---|---:|---:|
| Atlas Content Ops Deflection Stripe Paid Checks | 123s install + 22s containers + 22s setup | 15s tests |
| Atlas Content Ops Macro Writeback Checks | 129s install + 29s setup | 15s tests |
| Atlas Content Ops Input Provider Checks | 130s install + 27s setup | 73s tests |
| Atlas Content Ops Deflection Report Checks | 126s install + 22s containers + 21s setup | 15s tests |
| Extracted Pipeline Checks | 13s install | 264s `run_extracted_pipeline_checks.sh` |

For the content-ops checks, dependency caching is likely a safer first speedup
than reducing coverage. The extracted pipeline check is different: its runtime
is the check script itself, so speeding it up needs a more careful package-level
decomposition.

### Pre-Push Audit

The trusted-base `pre-push-audit` workflow is not the first place to optimize.
Measured job steps were small:

| Step | Median |
|---|---:|
| `Run local PR review bundle` | 6s |
| `PR-review tooling unit tests` | 6s |
| `Checkout trusted base` | 5s |
| `Materialize PR head as data` | 1s |
| `Workflow security posture audit` | 0-1s |

This is intentional duplication: the builder runs local review before push, and
CI re-runs it from trusted base so a PR cannot weaken the gate that judges it.
The time cost is low relative to the protection.

## Duplication Map

| Area | Duplicate shape | Keep, reduce, or split |
|---|---|---|
| Local review vs CI pre-push audit | `scripts/local_pr_review.sh` runs locally and again in `pre-push-audit.yml` from trusted base. | Keep. This is the core trusted-base safety pattern. |
| `pre_push_audit.sh` inside local review | Local review includes `pre_push_audit.sh`, then adds session drift, AI record, cross-layer hints, plan/code consistency, reviewer rules, and diff check. | Keep. These are different layers over the same PR contract. |
| Pre-push audit tooling tests | CI adds PR-review tooling unit tests after the trusted local-review bundle. | Keep for now. Median is ~6s. |
| Maturity Sweep broad ratchets | One workflow serially runs many independent lane ratchets. | Split or matrix. Same ratchets, less wall time. |
| General Maturity Sweep and lane-specific maturity checks | General sweep and deflection-specific sweep can both fire for content-ops/deflection changes. | Keep initially; consider path ownership rules after more samples. The lane check is cheap. |
| Product package installs | Several product workflows independently reinstall similar Python test dependencies even though their `actions/setup-python` steps already enable `cache: "pip"`. | Investigate cache misses, cache-key shape, dependency constraints, or a different dependency strategy. Do not recommend simply adding pip caching; it already exists on the sampled heavy checks. |
| `tests/**` path filters | Broad workflows fire on any test change, including unrelated tests. | Narrow only after proving the workflow does not consume the full test tree. Explicitly exclude Maturity Sweep for now because `scripts/maturity_sweep.py` indexes every test file through `--tests-root tests`. |

## Ranked Speedup Candidates

| Rank | Candidate | Expected savings | Risk | Safety boundary |
|---:|---|---:|---|---|
| 1 | Matrix or split `Maturity Sweep` independent ratchet groups. | 90-160s on PRs that trigger the broad sweep. | Medium | Run the same ratchet commands and require every matrix leg or an aggregate gate to pass. Do not delete ratchets. |
| 2 | Investigate why existing pip caches are not reducing heavy content-ops install time, then adjust cache keys, constraints, wheel reuse, or dependency split strategy. | 30-120s per affected product check if cache misses or reinstall churn are reduced. | Medium | Keep the same tests and dependency set. Treat current `cache: "pip"` usage as baseline evidence, not future work. |
| 3 | Narrow broad `tests/**` triggers only for workflows whose checks do not read the full test tree. | Case-by-case avoided workflow runs on unrelated test-only PRs. | Medium | Do not narrow Maturity Sweep while it uses `--tests-root tests` and `scripts/maturity_sweep.py` indexes every test file for coverage findings such as `NO_TEST_FILE`. Do not narrow security or ownership gates. |
| 4 | Add `concurrency.cancel-in-progress` to PR workflows with repeated pushes. | Reduces runner waste; latest-run wall clock unchanged. | Low | Use PR branch/ref grouping. Do not cancel push-to-main or scheduled security sweeps. |
| 5 | Split extracted pipeline check into sub-jobs only after package ownership is mapped. | Potentially 60-120s, but unproven. | Medium/High | Preserve the full extracted package gauntlet; split by independent modules only after proving equivalent coverage. |

## Recommended Next PR

Start with the broad `Maturity Sweep` job. It is slow, frequently triggered by
repo-wide paths, and structurally parallelizable without weakening the detector.
The safest implementation shape is:

1. Keep the existing unit test and advisory sweep.
2. Move independent ratchet command groups into a matrix or split jobs.
3. Preserve a single required/review-visible pass condition by requiring all
   matrix legs, or adding a small aggregate job if branch protection needs a
   stable context.
4. Prove equivalence by comparing the old command list to the new matrix list
   in a fixture test or audit script.

Do not start by removing `pre-push-audit`, `live-reconciliation`, PR body
contract, diff budget, or secret scanning. Those are cheap enough and are the
gates that keep the autonomous loop honest.
