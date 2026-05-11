# PR-Audit-Plan-Diff-Size

## Why this slice exists

The reviewer framework now verifies that plan docs have the required
sections and that their file list matches the branch diff. The remaining
size drift gap is that a PR can claim a small slice while the actual
diff grows far past the review-size budget.

This split lands only the diff-size auditor from the oversized audit-kit
PRs. It gives reviewers a deterministic signal when the estimate is
still useful, merely stale, or no longer a credible scope contract.

## Scope (this PR)

1. Add `scripts/audit_plan_doc_diff_size.py`.
2. Add focused tests for total parsing, scoped parsing, threshold
   behavior, missing totals, and real CLI behavior against a temporary
   git repo.
3. Refresh the coordination row for this split slice.

### Files touched

- `scripts/audit_plan_doc_diff_size.py`
- `tests/test_audit_plan_doc_diff_size.py`
- `plans/PR-Audit-Plan-Diff-Size.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The auditor reads the plan doc's `## Estimated diff size` section,
parses the `Total` table row, and compares it against
`git diff --numstat BASE_REF...HEAD`.

It reports:

- `OK` when drift is at or below 25%.
- `WARN` when drift is above 25% and at or below 50%.
- `FAIL` when drift is above 50%.

Only `FAIL` exits non-zero. `WARN` is a review signal but still permits
the PR to move if the author documented the size tradeoff.

## Intentional

- The script counts added plus deleted lines from git's numstat output.
  Binary files are skipped because they do not have line counts.
- It parses only the `## Estimated diff size` section. Example totals
  elsewhere in the plan do not count.
- No wrapper integration in this PR. The wrapper comes after individual
  auditors land.

## Deferred

- Wrapper and CI integration.
- Generated-file exemptions.
- PR-description diff-size claim checks.
- Combining shape, file-list, and diff-size outputs into one command.

## Verification

```bash
python -m pytest tests/test_audit_plan_doc_diff_size.py
python -m py_compile scripts/audit_plan_doc_diff_size.py tests/test_audit_plan_doc_diff_size.py
python scripts/audit_plan_doc.py plans/PR-Audit-Plan-Diff-Size.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Plan-Diff-Size.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Plan-Diff-Size.md origin/main
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/audit_plan_doc_diff_size.py` | 125 |
| `tests/test_audit_plan_doc_diff_size.py` | 179 |
| `plans/PR-Audit-Plan-Diff-Size.md` | 83 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~391** |
