# Checker Detection Branch Discipline

## Why this slice exists

The checker-failure-detection gap has now repeated across multiple slices:
#940, #958, and #961. In each case, the product or workflow checker existed,
but some branch whose job was to detect a bad input was not pinned by a focused
negative fixture. Reviewers had to mutation-test branches by hand to prove
whether the checker actually failed when it should.

#954 closed #940's branch-coverage gap and explicitly named generic
checker-testing discipline as the follow-up if the pattern recurred. #958 and
#961 are that recurrence signal, so this rule belongs in `AGENTS.md` instead
of being rediscovered PR by PR.

## Scope (this PR)

Ownership lane: workflow/process

Slice phase: Workflow/process

1. Add a builder rule for validators, checkers, evaluators, and gate
   predicates: each detection branch or OR-clause needs a focused fixture that
   proves it bites.
2. Add guidance for false-positive fixtures, transport-level I/O tests, and
   fail-closed return-shape drift.
3. Update the reviewer checklist so the rule is checked before LGTM when a PR
   changes a checker/evaluator/gate.

### Files touched

- `plans/PR-Checker-Detection-Branch-Discipline.md`
- `AGENTS.md`

## Mechanism

The new `AGENTS.md` subsection sits after the existing auditor fixture-test
rule. It generalizes the lesson beyond audit-script-only coverage:

- branch-level negative fixtures for each detection rule,
- one-marker fixtures for OR predicates,
- false-positive fixtures for broad type or parser matches,
- transport-level tests for network/file/database checker I/O,
- fail-closed tests for checker result-envelope drift.

The reviewer checklist gains one item requiring the verdict to call out this
coverage, or explicitly explain why the PR does not touch a checker-like
surface.

## Intentional

- No code scanner or mutation-test runner in this slice. This is the written
  process rule that should stop the immediate drift; a mechanical mutation
  gate can follow if the written rule still does not internalize.
- No product code changes. This is independent of #961 and does not alter the
  support-ticket generated-content gate.

## Deferred

- Future PR: add a mechanical checker/mutation harness if future PRs keep
  missing validator detection branches after this documented rule lands.
- Parked hardening: none added by this slice.

## Verification

- `bash scripts/local_pr_review.sh --allow-dirty`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| AGENTS workflow rule | ~55 |
| **Total** | **~120** |
