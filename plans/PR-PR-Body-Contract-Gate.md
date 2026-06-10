# PR-PR-Body-Contract-Gate

## Why this slice exists

The AGENTS.md section 1b PR-body contract (Plan lead line, Slice phase line,
then Intentional / Deferred / Parked hardening / Verification / Diff size) is
enforced today only when the builder routes through `scripts/open_pr.sh` /
`scripts/push_pr.sh` with a body file. A hand-rolled `gh pr create` bypasses
all of it, and human review is the only catch - which is exactly how
atlas-portfolio PR #303 shipped with a non-conforming body on 2026-06-10 and
drew a reviewer BLOCKER. The existing CI drift audit
(`scripts/audit_pr_session_drift.py`) validates only the Slice phase line of
the PR body, not the section contract.

This slice adds a mechanical CI gate: every `pull_request` event audits the
PR body itself against the section 1b contract, so a non-conforming body
fails checks regardless of which tool or session opened the PR. Dogfooding
during development immediately caught a live violation (PR #1476's body was
missing Parked hardening and Diff size).

The total lands a hair over the 400-LOC soft cap (401) because more than
half is this plan doc plus failure-mode unit tests; the runtime surface is
one ~110-line script and a 23-line workflow.

## Scope (this PR)

Ownership lane: pr-workflow-tooling
Slice phase: Production hardening

1. Add `scripts/audit_pr_body.py`: validates a PR-body file for the Plan
   lead line (and that the named plan doc exists in the checkout), a Slice
   phase line before the first section, and the five required sections in
   order.
2. Add `.github/workflows/pr_body_contract.yml`: on pull_request
   opened/edited/reopened/synchronize, write `github.event.pull_request.body`
   to a file (no GitHub API call - immune to the token-401 flake class) and
   run the audit.
3. Unit-cover the audit and enroll the test file in the existing PR-review
   tooling test step in `.github/workflows/pre_push_audit.yml`.

### Review Contract

- Acceptance criteria:
  - [ ] A PR body missing any required section, the Plan lead line, the
        Slice phase line, or naming a nonexistent plan doc fails the audit
        with the misses named.
  - [ ] A conforming body passes; extra non-required sections are allowed
        anywhere.
  - [ ] The workflow re-runs when the PR body is edited.
  - [ ] The workflow reads the body from the event payload, not the API.
- Affected surfaces: CI workflows, audit scripts, tests.
- Risk areas: false-positive blocking of PRs, workflow injection safety.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `.github/workflows/pr_body_contract.yml`
- `.github/workflows/pre_push_audit.yml`
- `plans/PR-PR-Body-Contract-Gate.md`
- `scripts/audit_pr_body.py`
- `tests/test_audit_pr_body.py`

## Mechanism

`audit_pr_body(body, root)` is a pure function over the body text: the first
non-empty line must be a Plan lead line naming a doc under the plans
directory, and that doc must exist under the repo root; a `Slice phase:`
line must appear before the first `##` heading; the five required `##`
sections must all be present and in relative order (other sections may be
interleaved - the contract fixes the order of the required ones, not
exclusivity). Failures are returned as a list and printed one per line;
exit 1 on any failure, 2 on usage errors.

The workflow materializes the PR body via an `env:` binding of
`github.event.pull_request.body` written with `printf '%s\n'` - the body
never passes through shell interpolation, and no GitHub API call is made, so
the gate cannot fail for auth reasons (the failure mode that flaked
pre-push-audit twice on 2026-06-10). The `edited` trigger means fixing the
body re-runs the gate without a new push.

## Intentional

- The gate validates section presence and order, not section content; the
  reviewer owns judgment about content quality. Diff-size inner format is
  deliberately not validated (the plan-doc audit already enforces the table
  shape on the plan side).
- No waiver mechanism. Every recent PR in this repo follows the contract,
  and a waiver marker would reintroduce the silent-bypass this gate exists
  to close. If real friction appears, remove or relax the gate deliberately
  per the guardrail-review convention, not by special-casing.
- The audit is a new script rather than an extension of
  `audit_pr_session_drift.py`: the drift audit needs git context and runs in
  a different lifecycle (pre-push, before the PR exists); this gate is
  body-only and runs on the PR event.
- Heading matching is case-sensitive and exact, mirroring AGENTS.md; a
  misspelled heading should fail, not fuzzy-match.

## Deferred

- The same gate for `atlas-portfolio` ships as its own PR in that repo
  (same script shape, `web/plans/` prefix).
- Bringing existing open PR bodies into conformance is per-PR cleanup
  (PR #1476's body gets fixed alongside this slice landing).

Parked hardening: none.

## Verification

- Passed: pytest tests/test_audit_pr_body.py
  - 9 passed.
- Passed: dogfood against live PR bodies - merged PR #1452's body passes;
  PR #1476's pre-fix body fails with the two genuinely missing sections
  named (Parked hardening, Diff size).
- Passed: bash scripts/check_ascii_python.sh
- Pending before push:
  - python scripts/sync_pr_plan.py plans/PR-PR-Body-Contract-Gate.md --check
  - bash scripts/local_pr_review.sh

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pr_body_contract.yml` | 23 |
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/PR-PR-Body-Contract-Gate.md` | 128 |
| `scripts/audit_pr_body.py` | 109 |
| `tests/test_audit_pr_body.py` | 144 |
| **Total** | **406** |
