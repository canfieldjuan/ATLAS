# PR-Audit-Pre-Push-CI

## Why this slice exists

The audit-kit wrapper exists on main, but it is still manual. This slice
adds CI enforcement so every PR gets the same mechanical review-shape
checks before merge.

## Scope (this PR)

1. Add a GitHub Actions workflow that runs `scripts/pre_push_audit.sh`.
2. Trigger the workflow for every pull request and every push to `main`.
3. Refresh the in-flight coordination row for this slice.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Audit-Pre-Push-CI.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The workflow checks out full git history with `fetch-depth: 0`, sets up
Python 3.11, resolves `origin/HEAD`, and runs the same wrapper developers
can run locally:

```bash
bash scripts/pre_push_audit.sh
```

Full history matters because the wrapper compares branch changes against
the trunk merge-base for plan files.

## Intentional

- This is CI-only. No local git hook installer is added in this slice.
- The workflow runs on every pull request, not only audit-kit paths. Scope
  and plan drift are cross-cutting PR risks, so path-filtering would leave
  gaps.
- Newer auditors that currently expose known drift remain outside the
  wrapper; this workflow enforces only the wrapper's current contract.

## Deferred

- Local hook installer for developers who want pre-push feedback before CI.
- Wiring additional auditors into `scripts/pre_push_audit.sh` after known
  drift is reconciled or explicitly accepted.

## Verification

```bash
bash scripts/pre_push_audit.sh
python scripts/audit_plan_doc.py plans/PR-Audit-Pre-Push-CI.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Pre-Push-CI.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Pre-Push-CI.md origin/main
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 28 |
| `plans/PR-Audit-Pre-Push-CI.md` | 66 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~98** |
