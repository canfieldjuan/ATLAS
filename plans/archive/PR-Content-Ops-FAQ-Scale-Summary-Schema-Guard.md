# PR-Content-Ops-FAQ-Scale-Summary-Schema-Guard

## Why this slice exists

PR-Content-Ops-FAQ-Scale-Summary-Examples refreshed the checked-in large-run
examples with top-level FAQ run-summary blocks. The reviewer manually verified
that those blocks match the runtime summary shape. That manual check is useful
enough to lock in because future summary-field changes should fail the example
test instead of leaving stale operator examples.

This slice turns that manual schema check into a focused regression guard.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add a focused test helper that extracts nested dictionary shape.
2. Compare both checked-in FAQ scale-smoke example summaries to a runtime-built
   FAQ run-summary shape.
3. Wire the guard into extracted pipeline CI.
4. Keep runtime code and example values unchanged.

### Files touched

| File | Change |
|---|---|
| `.github/workflows/extracted_pipeline_checks.yml` | Triggers extracted checks when the FAQ CLI schema source or scale-smoke guard changes. |
| `plans/PR-Content-Ops-FAQ-Scale-Summary-Schema-Guard.md` | Plan doc for this guard slice. |
| `scripts/run_extracted_pipeline_checks.sh` | Runs the scale-smoke guard in the extracted pipeline check suite. |
| `tests/test_smoke_content_ops_faq_scale_run.py` | Adds schema-shape coverage for checked-in FAQ scale examples. |

## Mechanism

The existing scale-smoke example test loads both static example artifacts. This
slice imports the FAQ CLI module in the same spec-loader style used for the
scale-smoke script, builds a minimal runtime FAQ run-summary, and compares
nested dictionary keys against each example's top-level FAQ run-summary block.

The comparison is shape-only. Existing assertions still cover representative
values and internal consistency.

The extracted pipeline workflow now triggers when the FAQ CLI schema source or
the scale-smoke guard changes, and the extracted pipeline check script runs this
test file alongside the FAQ generator tests.

## Intentional

- Test-only slice. No runtime behavior or example data changes.
- The guard compares nested keys, not exact values, because the examples are
  representative artifacts.
- It imports a private helper from the CLI module only inside tests; this is a
  schema-regression guard, not production coupling.
- CI wiring is included because the guard must run in the pipeline to replace a
  manual review check.

## Deferred

- A public schema helper can be extracted later if more surfaces need the same
  guard.
- Parked hardening for this slice: none. The hardening tracker has no entries
  for this lane yet.

## Verification

- Scale-smoke FAQ pytest for `tests/test_smoke_content_ops_faq_scale_run.py` -
  passed, 18 tests.
- Py compile for affected test file - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Reviewer update: guard was wired into extracted pipeline CI.
- Full extracted pipeline checks - passed, 1,753 tests.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| CI wiring | ~10 |
| Test guard | ~35 |
| **Total** | ~120 |
