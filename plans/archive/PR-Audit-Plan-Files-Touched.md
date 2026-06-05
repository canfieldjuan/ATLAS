# PR-Audit-Plan-Files-Touched

## Why this slice exists

The reviewer framework depends on plan docs being a real scope source.
The shape auditor verifies required sections exist, but it does not
check whether the plan's declared files match the actual branch diff.

This split lands only the files-touched auditor from the oversized
audit-kit PRs. It catches drive-by files and stale plan claims before a
reviewer has to discover scope drift manually.

## Scope (this PR)

1. Add `scripts/audit_plan_doc_files_touched.py`.
2. Add focused tests for extraction, mismatch classification, and real
   CLI behavior against a temporary git repo.
3. Refresh the coordination row for this split slice.

### Files touched

- `scripts/audit_plan_doc_files_touched.py`
- `tests/test_audit_plan_doc_files_touched.py`
- `plans/PR-Audit-Plan-Files-Touched.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The auditor reads a plan doc, extracts backticked paths only from the
`### Files touched` subsection, and compares those paths against
`git diff --name-only BASE_REF...HEAD`.

It reports:

- `MISSING` for files in the diff but absent from the plan.
- `EXTRA` for files claimed in the plan but absent from the diff.
- `OK` when both sets match.

The default base ref is `origin/main`; callers can pass a different
base ref for stacked PRs or tests.

## Intentional

- The script checks committed git diff state, not untracked working-tree
  files. It is meant for PR/review validation after a slice is staged or
  committed.
- It parses only the explicit `### Files touched` subsection. Backticks
  elsewhere in the plan do not count as file claims.
- No wrapper integration in this PR. The wrapper comes after individual
  auditors land.

## Deferred

- Untracked-file support for local pre-push usage.
- Diff-size estimate auditing.
- Wrapper and CI integration.
- Cross-checking PR description file claims.

## Verification

```bash
python -m pytest tests/test_audit_plan_doc_files_touched.py
python -m py_compile scripts/audit_plan_doc_files_touched.py tests/test_audit_plan_doc_files_touched.py
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Plan-Files-Touched.md origin/main
git diff --check
```

## Estimated diff size

| File | LOC (approx) |
|---|---:|
| `scripts/audit_plan_doc_files_touched.py` | 119 |
| `tests/test_audit_plan_doc_files_touched.py` | 164 |
| `plans/PR-Audit-Plan-Files-Touched.md` | 76 |
| `docs/extraction/coordination/inflight.md` | 2 |
| **Total** | **~361** |
