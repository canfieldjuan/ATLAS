# PR-CI-CD-Runtime-Audit

## Why this slice exists

Issue #1962 S1 mapped the current Atlas CI/CD machine, and S2 turned the
long-running watcher handoff into repo-visible operating docs. The next useful
slice is S3: measure what the current PR checks cost, identify where the local
and GitHub gates overlap, and name safe speedup candidates without weakening the
non-negotiable safety boundaries.

The root cause is that "CI feels slow" is not actionable by itself. Without a
measured runtime and duplication map, speedup work risks weakening the same
guards that make long-running autonomous sessions safe. This slice fixes the
analysis layer: it produces the evidence and ranked candidates, but it does not
change workflows or branch protection.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Add a CI runtime and duplication audit document for issue #1962 S3.
2. Measure recent PR workflow/job durations from GitHub Actions and separate
   required/meta gates from product/package/advisory checks.
3. Compare GitHub Actions gates against the local `local_pr_review.sh`,
   `pre_push_audit.sh`, and package gauntlet layers to identify repeated work.
4. Rank safe speedup candidates by expected time savings and risk, while
   preserving the non-negotiable gates named in issue #1962.

### Review Contract

Acceptance criteria:
- The audit cites the data source and sampling window used for runtime
  measurement.
- The audit separates required/meta gates, product/package gates, advisory
  checks, and local-only gates.
- The audit identifies duplication across local review, pre-push audit, package
  gauntlets, and GitHub Actions.
- Each speedup candidate includes expected savings, risk rating, and the safety
  boundary it preserves.
- The PR is documentation/plan only; it must not change workflow behavior,
  branch protection, watcher code, or CI scripts.

Affected surfaces:
- Developer workflow docs for long-running Atlas builder sessions.

Risk areas:
- Recommending speedups that weaken required safety gates.
- Treating one flaky or unusual run as representative.
- Mixing local developer cost with GitHub required-check latency.

Triggered reviewer rules:
- R1 Requirements match
- R2 Test evidence
- R6 Workflow/process
- R14 Codebase verification

### Files touched

- `docs/ci_cd_runtime_duplication_audit.md`
- `plans/INDEX.md`
- `plans/PR-CI-CD-Runtime-Audit.md`
- `plans/archive/PR-Long-Running-Watcher-Handoff.md`

## Mechanism

The new audit doc uses GitHub Actions run/job metadata as the runtime source,
then cross-checks that against the workflow inventory from
`docs/ci_cd_autonomous_coding_map.md` and the local gate entry points in
`scripts/local_pr_review.sh`, `scripts/pre_push_audit.sh`, and package gauntlet
scripts.

The output is intentionally decision-oriented: measured slow spots, duplicated
work, non-negotiable gates, and a ranked backlog of optimization candidates.

## Intentional

- No CI, branch-protection, watcher, or script behavior changes in this slice.
  The point is to make the next optimization PR evidence-backed.
- Runtime numbers are approximate and sampled from recent PR runs rather than a
  full historical warehouse export.
- Security, ownership, PR-body/plan, and live reconciliation gates stay
  non-negotiable even when they appear in multiple layers.

## Deferred

- Applying any speedup candidate to workflows or local scripts.
- #1962 S4-S7: monitoring spec, reusable playbook, and Reddit/public story
  draft.

Parked hardening: none.

## Verification

- `python scripts/sync_pr_plan.py plans/PR-CI-CD-Runtime-Audit.md --check` - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/ci_cd_runtime_audit_pr_body.md` - passed; non-blocking warning only: #1967 and #1953 also edit `plans/INDEX.md`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/ci_cd_runtime_duplication_audit.md` | 186 |
| `plans/INDEX.md` | 3 |
| `plans/PR-CI-CD-Runtime-Audit.md` | 107 |
| `plans/archive/PR-Long-Running-Watcher-Handoff.md` | 0 |
| **Total** | **296** |
