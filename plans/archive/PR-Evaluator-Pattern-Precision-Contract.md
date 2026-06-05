# PR: Evaluator Pattern Precision Contract

## Why this slice exists

Recent support-ticket generated-content evaluator slices repeatedly added
narrow denylist/regex patterns with negative fixtures, but review kept catching
the same missing precision half: no allowed near-miss fixture proving the new
pattern does not reject neutral language.

AGENTS.md already says false-positive surfaces need rejection fixtures, but the
rule is broad enough that evaluator-pattern slices keep treating it as
optional. This workflow/process slice makes the requirement explicit for
denylist, regex, phrase-matcher, and pattern-list evaluator changes.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Update the checker/evaluator coverage contract to require at least one
   allowed near-miss fixture whenever a PR adds or changes denylist, regex,
   phrase-matcher, or pattern-list detection.
2. Update the reviewer checklist so reviewers can enforce that precision
   requirement directly instead of re-raising it as a recurring NIT.
3. Keep this to workflow documentation only; no product code, evaluator code,
   or test behavior changes.

### Files touched

- `AGENTS.md` - codify the evaluator-pattern precision-test requirement.
- `plans/PR-Evaluator-Pattern-Precision-Contract.md` - this plan.

## Mechanism

The existing `3i. Checkers prove their failure detection` section already
requires negative fixtures and false-positive rejection fixtures. This slice
adds a specific sub-rule for evaluator pattern work:

- new or changed denylist/regex/phrase-matcher/pattern-list detectors need one
  bad fixture that must fail and one allowed near-miss fixture that must pass
- the plan must explicitly defer the near-miss only when it names why that is
  safe and what future PR owns it

The reviewer checklist is updated to mirror the builder-side rule so both
sessions share the same expected coverage shape.

## Intentional

- This is documentation-only because the recurring gap is a workflow contract
  gap, not a runtime behavior gap after PR #1092.
- No mechanical audit is added. A CI rule that detects evaluator-pattern diffs
  and verifies paired tests would be larger and more brittle than this slice.
- `AUDITOR_PROMPT.md` is left unchanged because AGENTS.md is the active PR
  contract both builder and reviewer sessions are instructed to read.

## Deferred

- Future PR: add a mechanical audit if this explicit AGENTS.md rule still does
  not stop evaluator-pattern slices from shipping without near-miss coverage.
- Parked hardening: none.

## Verification

- Command: bash scripts/local_pr_review.sh --current-pr-body-file <PR body file>
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| AGENTS.md contract update | ~20 |
| Plan doc | ~70 |
| **Total** | **~90** |
