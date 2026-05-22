# PR-Audit-Cross-Layer-Caller-Hints

## Why this slice exists

Recent reviews found the same class of issue across multiple PRs: a shared
function changed correctly in the diff, but a caller layer outside the obvious
review path exposed the real break. Diff-only review is too narrow for logic and
shared-function PRs because CLI, API, repository, UI, and service adapters can
drift even when their files are not changed.

This slice makes that risk visible during local review by surfacing non-diff
caller references for changed Python symbols. It is mitigation, not a full proof
of correctness: the output tells the builder and reviewer which outside files
deserve focused tests or inspection.

This is over the 400 LOC soft budget because an audit script must ship with its
own parser and fixture tests; splitting the tests from the script would weaken
the guardrail this slice is meant to add.

## Scope (this PR)

Ownership lane: audit/local-review

1. Add an advisory Python caller-hint audit for changed Python functions and
   classes.
2. Wire the advisory into the local PR review bundle without making caller
   presence itself a blocking failure.
3. Document the builder and reviewer expectation for shared-function PRs.
4. Add fixture tests for changed-symbol detection, non-diff caller reporting,
   and pathological path handling.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Audit-Cross-Layer-Caller-Hints.md` | Plan doc for this mitigation slice. |
| `AGENTS.md` | Documents the cross-layer caller-hint expectation for builders and reviewers. |
| `scripts/audit_cross_layer_callers.py` | New advisory audit that lists non-diff references to changed Python symbols. |
| `scripts/local_pr_review.sh` | Runs the advisory audit as part of local PR review. |
| `tests/test_audit_cross_layer_callers.py` | Fixture coverage for the new audit. |

## Mechanism

The audit compares the branch against the selected base ref, gathers modified
Python files, reads changed line numbers from `git diff --unified=0`, and maps
those lines to top-level or nested Python `FunctionDef`, `AsyncFunctionDef`, and
`ClassDef` spans using `ast`. Added Python files are skipped because they cannot
have real pre-existing callers and mostly produce generic-name false positives.
For each changed symbol, the audit searches tracked code files outside the
branch diff and prints a compact advisory list of references. Top-level
functions require call-shaped matches, methods require attribute-call matches,
and Python files are tokenized so comments and string literals do not count as
caller hints.

`scripts/local_pr_review.sh` runs the audit after cross-session drift. The audit
returns success when it finds caller hints because hints are review input, not a
proof of failure. It returns non-zero only for invalid paths, missing refs, or
unexpected audit errors.

## Intentional

- Advisory, not blocking: many shared helpers legitimately have outside
  references, and the reviewer still needs judgment.
- Token-aware text search over full call-graph construction: this keeps the
  check fast and dependency-free while still surfacing the non-diff code files a
  reviewer should inspect.
- Python-only in this slice because the repeated failures were in Python
  shared logic and CLI/library seams.

## Deferred

- TypeScript caller hints for frontend contract/helper changes.
- A future stricter mode that requires a plan-doc cross-layer verification note
  when high-risk symbols have outside callers.
- A richer import/call graph if token-aware text-search hints become too noisy.

## Verification

- Cross-layer caller audit fixture tests - passed, 10 tests.
- Python compile for the new audit script - passed.
- Local PR review against origin/main - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan and AGENTS docs | ~115 |
| Audit script | ~255 |
| Local review wiring | ~10 |
| Tests | ~250 |
| **Total** | ~630 |
