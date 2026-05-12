# PR-Audit-Local-Review-Runner

## Why this slice exists

GitHub now runs the mechanical audit wrapper, but the preferred workflow
is faster and cheaper: run the mechanical checks locally, hand the branch
to a second local reviewer session, fix issues, and only then open the PR.
This slice adds a local runner and documents that handoff.

## Scope (this PR)

1. Add `scripts/local_pr_review.sh`.
2. Update `AGENTS.md` with the local builder/reviewer handoff.
3. Refresh the in-flight coordination row for this slice.

### Files touched

- `scripts/local_pr_review.sh`
- `plans/PR-Audit-Local-Review-Runner.md`
- `AGENTS.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The local runner accepts an optional base ref, defaulting to
`origin/main`, prints the branch diff footprint, then runs:

```bash
bash scripts/pre_push_audit.sh
python scripts/audit_plan_code_consistency.py <changed plan doc>
git diff --check
```

The script remains mechanical. It does not replace the second reviewer
session's judgment review; it gives that session a cleaner branch and
fewer predictable failures.

## Intentional

- No git hook installation. Developers can call the runner explicitly
  before opening or updating a PR.
- No model calls are made by this script. Model-based review remains a
  separate local reviewer session.
- No GitHub API calls are made. GitHub CI remains the final enforcement
  layer after the local loop is clean.

## Deferred

- Optional hook installer for teams that want `local_pr_review.sh` to run
  automatically before push.
- A separate reviewer prompt/template update if we want a standardized
  local model-review transcript.

## Verification

```bash
bash scripts/local_pr_review.sh
python scripts/audit_plan_doc.py plans/PR-Audit-Local-Review-Runner.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Local-Review-Runner.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Local-Review-Runner.md origin/main
bash -n scripts/local_pr_review.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/local_pr_review.sh` | 73 |
| `plans/PR-Audit-Local-Review-Runner.md` | 73 |
| `AGENTS.md` | 29 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~179** |
