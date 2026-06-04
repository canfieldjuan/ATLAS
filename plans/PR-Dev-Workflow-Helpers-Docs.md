# PR-Dev-Workflow-Helpers-Docs

## Why this slice exists

The PR-prep helper scripts landed in #1307 (`push_pr.sh`, `sync_pr_plan.py`)
and #1308 (`new_pr_plan.sh`) and are merged to `main`, but no onboarding doc
references them. A fresh builder session reads `docs/SESSION_BOOTSTRAP.md` and
`AGENTS.md` to learn the PR workflow, finds no pointer to the helpers, and so
hand-reconstructs the plan shape and pushes raw -- hitting the same
plan-formatting churn and missing-`ATLAS_CURRENT_PR_BODY_FILE` failed-push that
the scripts were built to remove. The tooling fix only propagates to other
sessions once it is documented; this slice adds that pointer.

## Scope (this PR)

Ownership lane: dev-workflow/pr-prep-ergonomics
Slice phase: Workflow/process

1. Add the scaffold/sync/push flow (`scripts/new_pr_plan.sh`,
   `scripts/sync_pr_plan.py`, `scripts/push_pr.sh`) to
   `docs/SESSION_BOOTSTRAP.md` step 5 (the highest-leverage spot -- fresh
   sessions read it first).
2. Add a "PR-prep helpers" subsection to `AGENTS.md` under the builder
   workflow, with each script's purpose and the no-hook-bypass note.

### Files touched

- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Dev-Workflow-Helpers-Docs.md`

## Mechanism

Docs-only. The SESSION_BOOTSTRAP bullet gives the three commands in order with
the one-line rationale (scaffold the 7 sections; sync Files-touched + diff-size
from the real diff; push with the body env exported so the pre-push hook
validates the same body). The AGENTS.md subsection documents each script's
contract (including that `push_pr.sh` does not bypass the hook) and links the
two docs together. No script or hook behavior changes.

## Intentional

- Docs-only: no change to the scripts (#1307/#1308) or the pre-push hook.
- This PR was itself prepared with the documented flow (`new_pr_plan.sh` ->
  edits -> `sync_pr_plan.py` -> `push_pr.sh`), as a live end-to-end check that
  the chain works from a clean worktree.
- SESSION_BOOTSTRAP gets the imperative one-liner (read by fresh sessions);
  AGENTS.md gets the reference detail. Avoids duplicating the full contract in
  two places.

## Deferred

- The pre-push hook itself is per-checkout (installed via
  `scripts/install_local_pr_hook.sh`, not version-controlled); surfacing that
  install step in onboarding is a separate slice.
- Wiring `sync_pr_plan.py --check` as a CI gate.
- The `.gitignore`/scratch-file guidance friction item from #1306 remains
  unaddressed here.

Parked hardening: none.

## Verification

- The scaffold from `scripts/new_pr_plan.sh` passes
  `scripts/audit_plan_doc.py` (all 7 headers) -- confirmed during prep.
- After `scripts/sync_pr_plan.py`, the `scripts/audit_plan_doc_diff_size.py`
  check reports status OK.
- `scripts/local_pr_review.sh` passed after the comment fixes with the PR body
  file supplied.
- `scripts/push_pr.sh` ran `scripts/local_pr_review.sh` clean before the
  initial push.
- Both edited docs render (markdown headings intact).

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 23 |
| `docs/SESSION_BOOTSTRAP.md` | 10 |
| `plans/PR-Dev-Workflow-Helpers-Docs.md` | 81 |
| **Total** | **114** |
