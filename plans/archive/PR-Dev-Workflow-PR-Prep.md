# PR-Dev-Workflow-PR-Prep

## Why this slice exists

Issue #1306 asks which developer-workflow friction is burning time when
opening PRs. The concrete failure just reproduced in #1305: the branch passed
manual local review, but the pre-push hook failed because it was not given
`ATLAS_CURRENT_PR_BODY_FILE`; before that, the plan failed on parser-specific
shape details (`### Files touched` and a `| **Total** | **N** |` diff-size
table).

This slice fixes the highest-leverage ergonomic gap with a small wrapper pair:
one command updates the plan's machine-checked file list and diff-size total
from git, and one command runs local review/push with the PR body env wired
correctly. That removes the trial-and-error loop without weakening the audit
checks.

The slice is over the 400 LOC soft cap because the plan-sync helper and push
wrapper need focused failure-branch tests in the same PR; otherwise this would
ship more process machinery without proving it catches the exact drift that
caused the failed push.

## Scope (this PR)

Ownership lane: dev-workflow/pr-prep-ergonomics
Slice phase: Workflow/process

1. Add a plan-sync helper that rewrites `### Files touched` and
   `## Estimated diff size` from the current git diff.
2. Add a push wrapper that validates a PR body file exists, exports
   `ATLAS_CURRENT_PR_BODY_FILE`, runs `scripts/local_pr_review.sh` with the
   body file, and then pushes with the same env so the pre-push hook sees it.
3. Add focused tests for plan-section rewriting and wrapper command/env
   construction.
4. Document the intended operator command sequence in the plan.

### Files touched

- `plans/PR-Dev-Workflow-PR-Prep.md`
- `scripts/push_pr.sh`
- `scripts/sync_pr_plan.py`
- `tests/test_push_pr_wrapper.py`
- `tests/test_sync_pr_plan.py`

## Mechanism

`scripts/sync_pr_plan.py PLAN [BASE_REF]` computes the merge-base against the
base ref, reads `git diff --numstat -z`, normalizes renamed/copied entries to
their destination paths, then rewrites the plan's machine-checked subsections:

```bash
python scripts/sync_pr_plan.py plans/PR-Example.md
```

`scripts/push_pr.sh BODY_FILE [git push args...]` is the one push path for
builder sessions:

```bash
bash scripts/push_pr.sh tmp/pr-body-example.md -u origin HEAD
```

It fails early with a clear message if the body file is missing. Otherwise it
runs:

```bash
ATLAS_CURRENT_PR_BODY_FILE=... bash scripts/local_pr_review.sh --current-pr-body-file ...
ATLAS_CURRENT_PR_BODY_FILE=... git push ...
```

The wrapper carries the same env var into the installed pre-push hook, so the
hook does not bounce a push that already passed manual local review.

## Intentional

- The helpers do not relax any existing audit. They make the audited inputs
  easier to produce and pass the same data to the hook.
- The plan-sync helper updates only the machine-checked file list and diff-size
  section. Why/Scope/Mechanism/Intentional/Deferred/Verification remain human
  authored.
- The push wrapper requires an explicit body file instead of generating PR copy.
  That keeps PR judgment text human-authored while fixing the missing-env
  failure.
- #1305 and #1268 are not modified.

## Deferred

- `PR-Dev-Workflow-Plan-Scaffold`: optional `new_pr_plan.sh <slice>` template
  if the plan-sync helper does not remove enough friction.
- `PR-Dev-Workflow-PR-Body-Generator`: optional body generation from the plan if
  PR body churn remains a real bottleneck.

Parked hardening: none.

## Verification

- `pytest tests/test_sync_pr_plan.py tests/test_push_pr_wrapper.py -q` -- 7
  passed.
- `python -m py_compile scripts/sync_pr_plan.py` -- passed.
- `python scripts/audit_plan_doc.py plans/PR-Dev-Workflow-PR-Prep.md` --
  passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Dev-Workflow-PR-Prep.md`
  -- passed.
- `git diff --check` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-dev-workflow-pr-prep.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Dev-Workflow-PR-Prep.md` | 116 |
| `scripts/push_pr.sh` | 50 |
| `scripts/sync_pr_plan.py` | 223 |
| `tests/test_push_pr_wrapper.py` | 45 |
| `tests/test_sync_pr_plan.py` | 150 |
| **Total** | **584** |
