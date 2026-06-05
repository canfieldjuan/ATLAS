# PR-Dev-Workflow-Plan-Scaffold

## Why this slice exists

PR #1307 removed two PR-prep friction points: it syncs the plan's
machine-checked file/diff-size sections from git and pushes with
`ATLAS_CURRENT_PR_BODY_FILE` wired through the local review and pre-push hook.
The remaining item from that review is the initial plan-shape scaffold. Today a
builder still has to hand-type the seven-section `AGENTS.md` structure before
`scripts/sync_pr_plan.py` can help; missing or misspelled headings then waste a
review cycle.

This slice adds a narrow skeleton emitter for new plans. It should create a
valid starting shape, refuse risky or ambiguous writes, and hand off cleanly to
the existing sync/push helpers without weakening any audit.

## Scope (this PR)

Ownership lane: dev-workflow/pr-prep-ergonomics
Slice phase: Workflow/process

1. Add `scripts/new_pr_plan.sh` to create a PR-prefixed plan file with the seven
   required plan sections, ownership-lane and slice-phase lines, a `### Files
   touched` placeholder, and a zero-row diff-size table.
2. Refuse missing/unsafe slice names and existing plan files unless `--force`
   is explicit.
3. Add focused tests that exercise creation, default `PR-` prefixing,
   overwrite refusal, invalid-name rejection, and audit compatibility.
4. Keep PR body generation deferred; this helper emits only the plan skeleton.

### Files touched

- `plans/PR-Dev-Workflow-Plan-Scaffold.md`
- `scripts/new_pr_plan.sh`
- `tests/test_new_pr_plan.py`

## Mechanism

The helper runs from any directory inside a Git worktree:

```bash
bash scripts/new_pr_plan.sh Dev-Workflow-Plan-Scaffold \
  --lane dev-workflow/pr-prep-ergonomics \
  --phase Workflow/process
```

It resolves the repository root with `git rev-parse --show-toplevel`,
normalizes `Dev-Workflow-Plan-Scaffold` to
`plans/PR-Dev-Workflow-Plan-Scaffold.md`, creates `plans/` when needed, and
writes the fixed AGENTS seven-section scaffold. The output includes an empty
files-touched list and an initial `| **Total** | **0** |` diff-size table, so
the next command can be `python scripts/sync_pr_plan.py <plan>`.

Safety checks fail before writing when the slice name is missing, includes path
separators/traversal, or the target plan already exists without `--force`.

## Intentional

- The script is Bash to match `scripts/push_pr.sh` and keep the operator
  command lightweight.
- The helper does not infer lane, phase, deferred work, or verification from
  git state. It creates the contract shape; the builder still supplies
  judgment content.
- The generated plan is an editable scaffold, not a guarantee that local review
  will pass before the builder fills in the human-authored sections and runs
  `sync_pr_plan.py`.
- #1268 remains read-only and is not modified.

## Deferred

- `PR-Dev-Workflow-PR-Body-Generator`: optional body generation from the plan if
  PR body churn remains a real bottleneck after this scaffold lands.
- Brand voice storage/API/UI resumes after this workflow slice is opened and
  handed back.

Parked hardening: none.

## Verification

- `pytest tests/test_new_pr_plan.py tests/test_sync_pr_plan.py tests/test_push_pr_wrapper.py -q` --
  13 passed.
- `bash -n scripts/new_pr_plan.sh` -- passed.
- Temp-repo E2E smoke:
  `bash scripts/new_pr_plan.sh Sample-Workflow --lane workflow/test --phase Workflow/process`,
  then `python scripts/audit_plan_doc.py <temp>/plans/PR-Sample-Workflow.md`
  -- passed.
- `python scripts/audit_plan_doc.py plans/PR-Dev-Workflow-Plan-Scaffold.md`
  -- passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Dev-Workflow-Plan-Scaffold.md`
  -- passed.
- `git diff --check` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-dev-workflow-plan-scaffold.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Dev-Workflow-Plan-Scaffold.md` | 102 |
| `scripts/new_pr_plan.sh` | 145 |
| `tests/test_new_pr_plan.py` | 120 |
| **Total** | **367** |
