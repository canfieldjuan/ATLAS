# Persistent Detection Layer for Retired Failure Modes

Date captured: 2026-07-02

## Problem framing

Atlas already has blocking guards for current known-bad behavior: plan-as-contract,
open-thread reconciliation, in-lane/scope ordering, brittle-code checks, and
root-cause tracing. Those guards reduced the rate of retired failures such as
plan-weakening, test-weakening, scope drift, and symptom patching. They do not
prove those modes are gone forever.

This investigation is about **detectors**, not new guards. A detector is cheap,
always-on, non-blocking, and labels a likely recurrence. It should notice and
record the recurrence shape so the team can decide later whether to build a
blocking guard from real data.

## Current hooks worth using

- `scripts/local_pr_review.sh` is the local mechanical review bundle. It already
  runs before PR open/update and invokes the pre-push audit wrapper,
  extracted-test enrollment, PR/session drift, cross-layer caller hints,
  plan/code consistency, and whitespace checks.
- `.github/workflows/pre_push_audit.yml` runs that same bundle on every PR, so a
  cheap detector added near this seam gets PR coverage without touching
  product-specific workflows. The workflow also triggers on pushes to `main`,
  but that path is not usable for detection as-is: the wrapper diffs from
  `merge-base HEAD origin/main`, and after checkout of the pushed `main`
  commit `origin/main` is that same commit, so the changed-file and plan-diff
  signatures see an empty diff. Main-push detection must pass an explicit
  non-HEAD base such as the push event's `before` SHA (`github.event.before`).
- `scripts/pre_push_audit.sh` is the lower-level wrapper for plan shape, plan
  files touched, plan diff-size, MCP docs, extracted manifest sync, and ASCII
  policy.
- `scripts/audit_pr_session_drift.py` already knows how to compare branch files,
  open PR files, ownership lanes, and PR-body slice phases.
- `scripts/audit_cross_layer_callers.py` already emits advisory call-site hints;
  this is a good precedent for signal-only output that does not block merges.
- `docs/ai_dev_operating_model.md` names the existing enforcement funnel:
  local bundle, pre-push hook, separate local reviewer, and GitHub Actions. The
  detector layer should reuse that funnel but stay non-blocking.

## Retired failure modes and cheap signatures

| Retired mode | Cheap recurrence signature | Why it is cheap | Expected false positives |
|---|---|---|---|
| Plan-weakening | A plan doc and implementation files change together **and** the plan diff removes or relaxes acceptance language such as `must`, `required`, `fails`, `blocks`, `verification`, named tests, or file claims. | Text diff over `plans/PR-*.md` plus changed-file list. | Legitimate plan updates after discovery; okay if labeled as detector-only. |
| Test-weakening | Test files and covered product files change together while assertions, expected errors, fixtures, or parametrized negative cases are deleted/loosened. | Text diff over `tests/**`, `*.spec.*`, and package test scripts. | Refactors that move assertions or replace brittle tests with better coverage. |
| Scope drift | Changed files fall outside the plan's `Files touched`, declared lane, or issue-declared arc paths; or a branch touches files owned by another open PR/lane. | Existing plan file list and PR/session drift machinery already computes most inputs. | Cross-cutting workflow docs and mechanical rename PRs. |
| Symptom patching | The PR body/plan claims a root cause or traced source, but changed files are localized only to a caller/test/output layer far from the named root-cause file; or no root-cause artifact is referenced while the diff changes only a narrow symptom surface. | Heuristic over plan text, PR body text, changed paths, and cross-layer caller hints. | Small legitimate adapter fixes where the root cause is actually local. |

## Candidate architecture 1: Advisory CI summary appended to the existing pre-push workflow

### Hook

Add a detector script, for example `scripts/detect_retired_failure_modes.py`, and
call it from `.github/workflows/pre_push_audit.yml` after `bash
scripts/local_pr_review.sh`. The step should use `continue-on-error: true`,
write Markdown to `$GITHUB_STEP_SUMMARY`, and carry an explicit status
condition such as `if: ${{ always() }}` (or `success() || failure()`).
Without one, GitHub Actions applies the default `success()` condition and
skips the detector whenever the review bundle step fails -- which is exactly
when a recurrence signal is most likely. The alternative is to run the
detector inside the bundle script itself so it cannot be skipped.

Local developers can run the same script through `scripts/local_pr_review.sh`, but
its exit code should be ignored or normalized to zero so it cannot block.

### Coverage and signatures

- **Plan-weakening:** inspect plan diffs for removed or softened acceptance
  language and simultaneous non-plan changes.
- **Test-weakening:** inspect test diffs for removed assertions, loosened expected
  exceptions, deleted negative fixtures, or snapshot churn in the same PR as
  product code.
- **Scope drift:** reuse changed files and plan `Files touched` claims to emit
  `retired-mode/scope-drift-candidate` when touched paths are not claimed.
- **Symptom patching:** emit a low-confidence signal when the plan says
  `root cause` but changed files do not include the named path/function, or when
  cross-layer caller hints show non-diff callers but no caller-layer tests changed.

### Maintenance cost and false-positive profile

Maintenance is low: one script plus one workflow step. The false-positive rate is
moderate because text diffs cannot understand intent. That is acceptable because
this architecture is deliberately non-blocking and only records candidates.

### Recording

Record detector output in the Actions job summary and upload a JSON artifact such
as `retired_failure_mode_signals.json`. The JSON should include:

- `mode`: `plan_weakening`, `test_weakening`, `scope_drift`, or `symptom_patching`.
- `confidence`: `low`, `medium`, or `high`.
- `files`: paths involved.
- `evidence`: short string snippets or diff-line counts.
- `category`: always `retired_failure_mode_recurrence`, so reviewers can
  distinguish it from quality misses and normal scope misses.

### What it does not catch

- Multi-commit behavior where weakening happened earlier and code later.
- Review-thread reconciliation problems unless the detector can read PR comments.
- Semantic test weakening that preserves assertion count but changes meaning.

## Candidate architecture 2: PR comment bot with sticky detector annotations

### Hook

Use the same detector script, but run it in a dedicated GitHub Action step. A
sticky timeline comment is created through the issue-comments API (every pull
request is also an issue), so the workflow must grant `issues: write`;
`pull-requests: write` alone covers review-comment and PR-mutation endpoints,
not timeline comments. Grant `issues: write` for this variant, or switch to PR
review comments under `pull-requests: write`. The step posts or updates one
sticky PR comment bounded by markers such as:

```markdown
<!-- retired-failure-mode-detector:start -->
...
<!-- retired-failure-mode-detector:end -->
```

The detector still exits zero. It should never request changes and should not
apply labels by default.

### Coverage and signatures

Same signatures as Architecture 1, plus better reviewability because the signal
sits where reviewers already work:

- **Plan-weakening:** PR comment includes removed/softened plan lines.
- **Test-weakening:** PR comment groups assertion/fixture deletions by test file.
- **Scope drift:** PR comment separates detector-level scope recurrence from the
  existing blocking scope guard.
- **Symptom patching:** PR comment shows the root-cause phrase/path it found and
  the actual changed files.

### Maintenance cost and false-positive profile

Maintenance is medium. Sticky comments require GitHub API handling, permissions,
and idempotent update logic. False positives are still moderate, but easier to
triage because the evidence is visible in the PR conversation.

### Recording

Record as a sticky PR comment with stable headings:

- `Retired failure mode signal: plan-weakening candidate`
- `Retired failure mode signal: test-weakening candidate`
- `Retired failure mode signal: scope-drift recurrence candidate`
- `Retired failure mode signal: symptom-patching candidate`

Each item should include `Why this is only a detector` so reviewers do not treat
it as a new blocking gate.

### What it does not catch

- Runs without PR write permissions, fork PRs, or GitHub API outages.
- Patterns that require historical comparison across many PRs.
- Cases where the signal should remain private until triaged; comments are noisy.

## Candidate architecture 3: Non-blocking labels plus issue ledger

### Hook

Run a detector workflow on `pull_request` and, when signals appear, apply labels
such as:

- `detector:retired-mode`
- `retired:plan-weakening-candidate`
- `retired:test-weakening-candidate`
- `retired:scope-drift-candidate`
- `retired:symptom-patching-candidate`

In parallel, append or update a central issue comment or issue body section that
acts as a ledger of detector events. The workflow exits zero even if labels or
issue updates fail.

This variant needs raised workflow permissions: `pull-requests: write` to apply
PR labels and `issues: write` to mutate the ledger issue comment/body. The
existing pre-push workflow is read-only (`contents: read`,
`pull-requests: read`), so implementing this variant under those defaults would
exit zero while silently recording no labels and no ledger entries.

### Coverage and signatures

This architecture can use all signatures from Architecture 1. Its advantage is
not stronger detection; its advantage is longitudinal tracking. Labels make it
cheap to query whether retired patterns are resurfacing under longer arcs, merge
pressure, or new model/vendor sessions.

### Maintenance cost and false-positive profile

Maintenance is medium-high because label creation, issue mutation, and idempotent
ledger updates need care. False positives are visible in repository metadata, so
labels must be explicitly named as candidates rather than findings.

### Recording

Record two layers:

1. PR labels for lightweight filtering.
2. A GitHub issue ledger row with PR number, commit SHA, detector mode,
   confidence, evidence summary, and eventual human disposition if available.

This cleanly separates retired-mode recurrence data from normal quality misses
or scope misses because all labels and ledger rows share the `retired:` namespace.

### What it does not catch

- Any signal on PRs where Actions cannot write labels or issues.
- Fine-grained evidence unless paired with an artifact or PR comment.
- Teams may overreact to labels and accidentally treat them as blockers.

## Candidate architecture 4: Offline detector ledger committed or uploaded by scheduled job

### Hook

Run a scheduled or manual workflow that scans recently merged PRs and open PRs,
then writes detector results to an artifact or a checked-in/generated ledger such
as `docs/audits/retired_failure_mode_detector_log.md`. It does not run on the PR
critical path at all.

### Coverage and signatures

- **Plan-weakening:** compare merged PR plan diffs against code diffs after the
  fact.
- **Test-weakening:** compare test and product diffs after merge and accumulate
  recurring files/modes.
- **Scope drift:** compare merged touched files against plan files and lane.
- **Symptom patching:** correlate root-cause claims, changed files, and later
  follow-up fixes in the same lane.

### Maintenance cost and false-positive profile

Maintenance is low-medium. It avoids PR noise and critical-path latency, but it
requires a small amount of GitHub history access. False positives are less
harmful because the ledger is reviewed asynchronously.

### Recording

Record a Markdown or JSON ledger with stable fields:

- `detected_at`
- `pr`
- `merge_sha`
- `mode`
- `confidence`
- `evidence`
- `human_disposition`
- `promoted_to_guard`: yes/no/link

### What it does not catch

- It is not immediate. A recurrence may merge before being labeled.
- It cannot help reviewers during the PR where the pattern recurs.
- It is weaker for survivability if the goal is to notice before merge.

## Tradeoff summary

| Architecture | Latency impact | Review visibility | Longitudinal data | Maintenance | Best fit |
|---|---:|---:|---:|---:|---|
| Advisory CI summary/artifact | Low | Medium | Medium | Low | Cheapest always-on detector. |
| Sticky PR comment | Low-medium | High | Medium | Medium | Best reviewer ergonomics. |
| Labels plus issue ledger | Low-medium | High | High | Medium-high | Best for trend analysis and recurrence accounting. |
| Scheduled/offline ledger | None on PR path | Low during PR | High | Low-medium | Best for zero critical-path cost. |

## Recommended next decision points

Do not build a guard yet. First choose where the team wants the signal to live:

1. If the priority is cheapest signal with almost no workflow churn, choose the
   advisory CI summary/artifact.
2. If the priority is reviewer actionability, choose the sticky PR comment.
3. If the priority is tracking model/vendor regressions over time, choose labels
   plus an issue ledger.
4. If the priority is zero critical-path impact, choose the scheduled/offline
   ledger.

The first implementation should keep the detector script pure and output JSON so
any of the recording backends can be swapped later without rewriting signatures.
