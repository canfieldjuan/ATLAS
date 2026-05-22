# PR-Content-Ops-FAQ-Scale-Smoke-Run-Summary

## Why this slice exists

PR-Content-Ops-FAQ-Scale-Run-Summary-Diagnostics added a compact FAQ CLI
`diagnostics.run_summary` block for large-upload triage. The scale-smoke wrapper
still buries that block inside the full CLI `result` payload, so operators have
to know the nested path before they can compare volume, generated count,
output-check status, and score shape.

This slice surfaces the FAQ run summary at the scale-smoke layer without
changing FAQ generation behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Copy `result.diagnostics.run_summary` into top-level
   `faq_run_summary` in the scale-smoke summary artifact.
2. Include a compact FAQ health fragment in the scale-smoke console line.
3. Preserve fail-closed exit behavior and the existing full `result` payload.
4. Add focused scale-smoke regression coverage for success and output-check
   failure summaries.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Scale-Smoke-Run-Summary.md` | Plan doc for this scale-smoke summary slice. |
| `scripts/smoke_content_ops_faq_scale_run.py` | Surfaces the FAQ CLI run summary in scale-smoke JSON and console output. |
| `tests/test_smoke_content_ops_faq_scale_run.py` | Covers top-level run summary passthrough and console visibility. |

## Mechanism

`run_scale_smoke(...)` already reads the FAQ CLI result JSON. A small helper will
extract `diagnostics.run_summary` when present and copy it to a top-level
`faq_run_summary` key before writing the scale-smoke summary artifact.

The console printer will use the same top-level block, not reparse the nested
result, to append a short fragment such as generated item count, weighted source
volume, failed output-check count, and max opportunity score.

## Intentional

- No FAQ CLI, generator, Markdown, or scoring changes.
- The full CLI `result` payload remains unchanged; this only adds a convenience
  alias and console visibility at the scale-smoke layer.
- Missing or older result payloads degrade to `faq=unavailable` rather than
  failing the wrapper.

## Deferred

- Hosted UI display for `faq_run_summary` remains separate.
- Updating checked-in historical failure examples can be a later docs/data slice
  if we want those static artifacts to include the new convenience alias.

## Verification

- Scale-smoke FAQ pytest for `tests/test_smoke_content_ops_faq_scale_run.py` -
  passed, 18 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| Scale-smoke summary/console | ~55 |
| Tests | ~35 |
| **Total** | ~170 |
