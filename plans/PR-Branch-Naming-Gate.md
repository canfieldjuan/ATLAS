# PR-Branch-Naming-Gate

## Why this slice exists

The mechanical-enforcement audit found that Atlas documents builder PR branch
names as `claude/pr-<slice-name>`, but no wrapper or CI artifact enforces that
rule before a PR branch is pushed or opened. That leaves branch/lane drift as a
manual convention at the same point where `push_pr.sh` and `open_pr.sh` already
have the PR body and current branch available.

Audit finding:

- `docs/audits/agents-mechanical-enforcement-audit-2026-07-29.md` classified
  "Builder branch names use `claude/pr-<slice-name>`" as `PROSE_ONLY` because
  `AGENTS.md` defines the convention but no branch-name gate was found in
  `open_pr.sh`, `push_pr.sh`, or CI.

### Problem-derived contract

- Root cause: PR publication wrappers accept any current branch name, so a
  builder can push/open a human PR from `main`, `claude/<topic>`, or a
  wrong-slice branch even when the PR body names a PR plan file.
- Correct fix must touch/change: add a branch-name checker that reads the PR body
  and current branch; call it from `scripts/push_pr.sh` and `scripts/open_pr.sh`
  before fetch/review/GitHub mutation; add direct and wrapper tests for matching,
  mismatched, and docs-only bodies.
- Must not change: do not change product code, branch protection, PR body audit
  semantics, draft consent semantics, PR ownership semantics, docs-only body
  admission, Dependabot behavior, or any EOM / dependency / non-owned PR lane.

## Scope (this PR)

Ownership lane: dev-workflow/branch-naming
Slice phase: Workflow/process

1. Add `scripts/check_pr_branch_name.py` to validate PR branches against the PR
   body contract.
2. Call that checker from `scripts/push_pr.sh` and `scripts/open_pr.sh` before
   network/ref/GitHub mutation side effects.
3. Add direct checker tests plus wrapper tests proving bad branch names fail
   before fetch or GitHub mutation.

### Review Contract

- Acceptance criteria:
  1. A planned PR body for this slice only
     admits branch `claude/pr-branch-naming-gate`; settled by
     `tests/test_check_pr_branch_name.py::test_branch_name_accepts_matching_plan_branch`
     and `tests/test_check_pr_branch_name.py::test_branch_name_rejects_plan_slug_mismatch`.
  2. `scripts/push_pr.sh` runs the branch-name check before `git fetch`, body
     audit, local review, or push; settled by
     `tests/test_push_pr_wrapper.py::test_push_pr_rejects_branch_that_does_not_match_plan_before_fetch`.
  3. `scripts/open_pr.sh` runs the branch-name check before origin refresh or
     GitHub mutation; settled by
     `tests/test_open_pr_wrapper.py::test_open_pr_rejects_branch_that_does_not_match_plan_before_fetch`.
  4. Docs-only bodies keep the narrower exemption but still require a
     `claude/pr-*` PR branch; settled by
     `tests/test_check_pr_branch_name.py::test_docs_only_body_requires_pr_prefix_only`.
- Reachability proof: the real entrypoints `bash scripts/push_pr.sh BODY_FILE`
  and `bash scripts/open_pr.sh BODY_FILE` are exercised by wrapper tests; the
  observable effect is early failure before fetch/GitHub logs on mismatched
  branches.
- Affected surfaces: `scripts/check_pr_branch_name.py`, `scripts/push_pr.sh`,
  `scripts/open_pr.sh`, wrapper tests, and this plan.
- Risk areas: shell side-effect ordering, docs-only exemption, plan-to-branch
  slug normalization, wrapper fixture parity.
- Reviewer rules triggered: R1, R2, R6, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: PR publication admission through `scripts/push_pr.sh` and
  `scripts/open_pr.sh`.
- Replaced-path behaviors: wrappers previously accepted any current branch and
  deferred failure to later GitHub/session behavior.
- Guard-relevant fields: current branch name, first structural PR body marker
  (`Plan:` or `Docs-only: true`), and plan filename slug.
- Caller x input shape: wrapper invocations with a planned PR body, docs-only
  body, mismatched branch, non-PR branch, and detached branch.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: no deployed config; branch names and PR body
  files are local wrapper inputs.
- Explicit value probe: planned PR body plus matching
  `claude/pr-branch-naming-gate` branch passes direct checker tests.
- Absent value probe: missing plan/docs-only marker and detached branch both
  fail direct checker tests.
- Default-session/default-context probe: docs-only body has no plan slug and is
  probed separately as prefix-only admission.
- Side-effect ordering: push and open wrapper tests prove branch mismatch fails
  before fetch/review/GitHub mutation logs.

### Files touched

- `plans/PR-Branch-Naming-Gate.md`
- `scripts/check_pr_branch_name.py`
- `scripts/open_pr.sh`
- `scripts/push_pr.sh`
- `tests/test_check_pr_branch_name.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_push_pr_wrapper.py`

## Mechanism

`scripts/check_pr_branch_name.py` reads the PR body and current branch supplied
by the wrapper:

- A plan-backed PR body computes an expected lowercase slug branch:
  `claude/pr-<slice-slug>`.
- `Docs-only: true` has no plan slug, so it keeps the docs-only exemption while
  still requiring the `claude/pr-*` PR-branch prefix.
- missing structural markers, detached branches, non-PR branches, and plan/branch
  mismatches fail with exit code 2.

`push_pr.sh` and `open_pr.sh` invoke the checker before `git fetch`, body audit,
local review, push, or GitHub mutation. That makes the branch-name convention an
admission gate rather than a review-memory rule.

## Intentional

- Docs-only bodies require only `claude/pr-*` because there is intentionally no
  plan filename to compare against.
- The checker normalizes plan filenames by lowercasing and collapsing non
  alphanumeric separators to `-`, matching the branch style Atlas already uses
  for plan-backed slices.
- No CI-only branch-protection change in this slice; the wrappers are the point
  where the local branch exists and can be rejected cheapest.

## Deferred

- Commit-message contract enforcement remains a separate follow-up slice.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_pr_branch_name.py tests/test_push_pr_wrapper.py tests/test_open_pr_wrapper.py`
  - 74 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Branch-Naming-Gate.md` | 158 |
| `scripts/check_pr_branch_name.py` | 85 |
| `scripts/open_pr.sh` | 17 |
| `scripts/push_pr.sh` | 8 |
| `tests/test_check_pr_branch_name.py` | 74 |
| `tests/test_open_pr_wrapper.py` | 20 |
| `tests/test_push_pr_wrapper.py` | 31 |
| **Total** | **393** |
