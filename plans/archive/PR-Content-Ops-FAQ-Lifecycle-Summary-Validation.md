# PR-Content-Ops-FAQ-Lifecycle-Summary-Validation

## Why this slice exists

The FAQ lifecycle smoke now supports compact `--summary-json` output, and the
docs are being updated to recommend that operator pattern. The existing
1,000-row database validation note still records the older full `--json` stdout
run, so it does not prove the current large-run triage flow.

This slice reruns the real local database lifecycle proof with 1,000 CFPB-derived
support-ticket rows using `--summary-json` plus `--output-result`, then records
the observed result and any surfaced issues.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-validation

1. Run the existing FAQ lifecycle smoke against the local Atlas Postgres DB with
   1,000 source rows.
2. Record the compact summary JSON behavior in the validation note.
3. Acknowledge whether the run surfaced new issues.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Summary-Validation.md`
- `docs/extraction/validation/content_ops_faq_lifecycle_db_1000_row_run_2026-05-23.md`

## Mechanism

The validation command uses the existing lifecycle smoke only, with an
`--output-result` path and `--summary-json`. The full payload stays in the
result artifact while stdout is redirected to a separate compact summary JSON
file. The validation note records parsed summary fields, output-check status,
artifact paths, and any new issue observed during the run.

## Intentional

- Documentation/validation record only; no generator, repository, migration, or
  lifecycle code changes.
- This slice does not depend on the unmerged artifact-runner PR.

## Deferred

- Full artifact-runner validation remains deferred until the artifact runner
  lands on main.
- Root `HARDENING.md` was scanned; no same-lane parked item is required before
  this validation run.

## Verification

- Passed: real local DB 1,000-row lifecycle smoke with `--summary-json`.
- Passed: parsed saved stdout/result JSON artifacts; stdout exactly matched
  `result.lifecycle_summary`, full result retained `reviewed_export`, and the
  compact summary reported `status=ok`, `source_rows=1000`,
  `saved_faq_count=1`, `error_count=0`.
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~64 |
| Validation note update | ~83 |
| **Total** | **~147** |
