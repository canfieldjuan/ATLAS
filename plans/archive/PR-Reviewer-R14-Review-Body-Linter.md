# PR-Reviewer-R14-Review-Body-Linter

## Why this slice exists

#1468 added R14: reviewers cannot LGTM from a PR story alone, and must record
the reviewed head SHA, changed code inspected, caller/test/artifact
spot-checks, and "not verified" disclosure. That PR intentionally deferred
mechanical enforcement.

This slice adds the first local enforcement tool: a review-body linter that
fails LGTM review text when the R14 evidence sections are missing or
placeholder-only. It turns the rule from a pure instruction into a reusable
checker a reviewer can run before posting an LGTM.

Diff budget note: this is slightly over the 400 LOC soft cap because the
checker and its failure-branch fixtures need to land together for the rule to
be trustworthy.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add `scripts/check_review_body_r14.py`, a stdlib-only CLI that reads a
   review body from a file or stdin and validates R14 evidence for LGTM
   verdicts.
2. Add focused fixture tests for valid LGTM text, non-LGTM bypass behavior, and
   each missing/placeholder failure branch.
3. Accept bounded reviewer label variants that preserve the same R14 evidence
   contract (`at HEAD` qualifiers and `Spot-checks:` shorthand), while reporting
   exact expected labels on failures.
4. Wire the pre-push audit workflow to pass the live pull-request body into the
   existing local review bundle, so CI enforces the same body contract the local
   `push_pr.sh` wrapper enforces.
5. Keep R14 review-body enforcement local/ad hoc in this slice; do not wire it
   to live GitHub review APIs yet.

### Review Contract

- Acceptance criteria:
  - [ ] LGTM review text fails if it omits the reviewed head SHA.
  - [ ] LGTM review text fails if it omits the `Codebase verification (R14)`
        section.
  - [ ] LGTM review text fails if changed-code, caller/test/artifact, or
        `Not verified` evidence lines are missing or placeholder-only.
  - [ ] LGTM review text accepts bounded evidence-label variants (`Changed code
        inspected at HEAD:`, `Caller/test/artifact spot-checks at HEAD:`, and
        `Spot-checks:`) without accepting placeholder evidence.
  - [ ] Missing-evidence errors name the exact expected label forms.
  - [ ] CI `pre-push-audit` passes the pull-request body to
        `local_pr_review.sh --current-pr-body-file` for PR events, while
        push-to-main keeps the no-body invocation.
  - [ ] LGTM review text fails if the R14 rule result is absent or non-passing.
  - [ ] Non-LGTM review text is allowed unless the caller explicitly passes
        `--verdict lgtm`.
  - [ ] Tests cover every detection branch and false-positive bypass path.
- Affected surfaces: local reviewer workflow script and tests only.
- Risk areas: false-positive LGTM detection / placeholder text passing as
  evidence / future live-review integration expecting a stable CLI contract.
- Reviewer rules triggered: R2, R10, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Reviewer-R14-Review-Body-Linter.md`
- `scripts/check_review_body_r14.py`
- `tests/test_check_review_body_r14.py`
- `tests/test_pre_push_audit_workflow.py`

## Mechanism

`check_review_body_r14.py` parses Markdown text and determines whether R14 is
required by explicit `--verdict lgtm` or auto-detected LGTM verdict language.
When R14 is required, it checks:

- a `Reviewed head:` line with a hex commit-like value;
- a `Codebase verification (R14)` section;
- non-placeholder `Changed code inspected`, `Caller/test/artifact
  spot-checks`, and `Not verified` lines inside that section;
- a passing R14 rule-results mention. `R14 Fail`, `R14 N/A`, and
  `R14 Not applicable` fail in LGTM mode.

For reviewer ergonomics, the changed-code and spot-check labels accept the
bounded `at HEAD` qualifier, and the spot-check line also accepts `Spot-checks:`
as a shorthand. `Not verified:` remains exact because it is a specific
disclosure field. Missing-evidence errors name the accepted labels so a reviewer
can fix a label mismatch without guessing.

The checker reports all missing evidence in one run and exits non-zero on
failure. It reads stdin when the path is `-`, matching common reviewer tooling
flows.

The GitHub Actions pre-push audit writes `github.event.pull_request.body` from
the event payload to a runner-temp body file and invokes
`local_pr_review.sh --current-pr-body-file` for pull-request events. Push events
continue to run the existing no-body command because there is no current PR body
to validate.

## Intentional

- No GitHub API integration in this PR. This is the local review-body contract
  checker; live review enforcement can come later once the body shape is proven.
- Auto-detection is conservative: it looks for verdict-style LGTM language
  rather than every incidental "LGTM" mention in discussion prose. Callers that
  know the verdict can force enforcement with `--verdict lgtm`.
- The checker validates that required evidence is present and non-placeholder;
  it does not judge whether the evidence is substantively correct. R14 remains a
  reviewer judgment rule.

## Deferred

- Wire the linter into a future live-review workflow that reads submitted LGTM
  reviews from GitHub and fails when R14 evidence is missing.
- Optionally add an AGENTS.md one-liner once the command is adopted by the
  reviewer session.

Parked hardening: none.

## Verification

- Focused pytest for `tests/test_check_review_body_r14.py`
  - Passed, 19 tests.
- Focused pytest for the linter plus workflow wiring tests
  (`tests/test_check_review_body_r14.py` and
  `tests/test_pre_push_audit_workflow.py`)
  - Passed, 21 tests.
- PR-review tooling unit-test list from `.github/workflows/pre_push_audit.yml`
  - Passed, 63 tests.
- Py-compile for `scripts/check_review_body_r14.py` and
  `tests/test_check_review_body_r14.py`
  - Passed.
- CLI stdin smoke for `scripts/check_review_body_r14.py`
  - Passed; forced-LGTM input with R14 evidence exits 0.
- Plan sync check for `plans/PR-Reviewer-R14-Review-Body-Linter.md`
  - Passed.
- Pending before push: local PR review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 26 |
| `plans/PR-Reviewer-R14-Review-Body-Linter.md` | 147 |
| `scripts/check_review_body_r14.py` | 202 |
| `tests/test_check_review_body_r14.py` | 195 |
| `tests/test_pre_push_audit_workflow.py` | 23 |
| **Total** | **593** |
