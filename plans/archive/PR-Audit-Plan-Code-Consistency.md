# PR-Audit-Plan-Code-Consistency

## Why this slice exists

Oversized PR #486 bundled this auditor with AGENTS.md policy and a shell
hygiene script. This split lands only the plan/code consistency auditor,
with the old review comments fixed: exact section matching, backticked
path claims, root-level and hyphenated paths, and bounded bare-filename
resolution.

## Scope (this PR)

1. Add `scripts/audit_plan_code_consistency.py`.
2. Add fixture tests for parser behavior and missing-claim reporting.
3. Refresh the in-flight coordination row for this split.

### Files touched

- `plans/PR-Audit-Plan-Code-Consistency.md`
- `scripts/audit_plan_code_consistency.py`
- `tests/test_audit_plan_code_consistency.py`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The auditor reads a plan doc and extracts enforceable backticked claims
from exact `Scope`, `Mechanism`, and `Verification` sections. Optional
parentheticals like `Scope (this PR)` are allowed, but headings like
`Out of scope` do not match.

Path claims must be backticked tokens ending in a known file extension.
Function claims must be backticked snake_case calls with at least four
characters before `()`. Paths are resolved directly from the repo root,
with bare filenames searched only under bounded roots (`scripts`,
`plans`, `docs`, `tests`, `atlas_brain`, and `extracted_*` packages).

## Intentional

- The script reports missing path/function claims only. There is no
  meaningful "extra" check because plans are claims about expected code,
  not an exhaustive inventory of all code.
- Code-fence contents are not parsed as function claims unless the claim
  itself is backticked. That keeps prose examples from becoming accidental
  enforcement.
- The shell hygiene auditor from #486 is deferred because its old version
  has separate review findings.

## Deferred

- Split `scripts/audit_script_hygiene.sh` from #486 after fixing its shell
  review findings.
- Wire this auditor into the pre-push wrapper after we decide which plan
  docs should be checked automatically.
- Close or replace oversized #486 after its useful pieces are harvested.

## Verification

```bash
python -m pytest tests/test_audit_plan_code_consistency.py
python -m py_compile scripts/audit_plan_code_consistency.py tests/test_audit_plan_code_consistency.py
python scripts/audit_plan_doc.py plans/PR-Audit-Plan-Code-Consistency.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Plan-Code-Consistency.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Plan-Code-Consistency.md origin/main
python scripts/audit_plan_code_consistency.py plans/PR-Audit-Plan-Code-Consistency.md
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/audit_plan_code_consistency.py` | 148 |
| `tests/test_audit_plan_code_consistency.py` | 115 |
| `plans/PR-Audit-Plan-Code-Consistency.md` | 76 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~343** |
