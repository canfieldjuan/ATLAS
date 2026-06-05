# PR-Content-Ops-FAQ-Lifecycle-500-Validation

## Why this slice exists

The FAQ product needs confidence signals at common SMB and mid-market upload
sizes, not only at the 1,000-row upper proof point. The current main branch has
1,000-row FAQ generator and lifecycle coverage, but no separate validation note
for a 500-row database lifecycle run.

This slice runs the real local database FAQ lifecycle against 500 CFPB-derived
support-ticket rows and records the observed result. The goal is evidence, not a
new code path.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-validation

1. Build a 500-row local fixture from the existing CFPB-derived 1,000-row JSONL
   source artifact.
2. Run the existing FAQ lifecycle smoke against the local Atlas Postgres DB with
   500 source rows.
3. Add a validation note with the observed lifecycle summary, timing, and issue
   status.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-500-Validation.md`
- `docs/extraction/validation/content_ops_faq_lifecycle_db_500_row_run_2026-05-23.md`

## Mechanism

The run uses the existing lifecycle smoke with the documented compact-output
pattern: `--output-result` writes the full lifecycle payload and
`--summary-json` writes compact stdout. The validation note records parsed
summary fields, artifact paths, timing, and any issues surfaced.

## Intentional

- Documentation/validation record only; no generator, repository, migration, or
  lifecycle code changes.
- The 500-row source fixture is generated under `tmp/` and is not checked in.
- This slice does not depend on the unmerged artifact-runner PR.

## Deferred

- Full artifact-runner validation remains deferred until the artifact runner
  lands on main.
- Root `HARDENING.md` was scanned; no same-lane parked item is required before
  this validation run.

## Verification

- Passed: real local DB 500-row lifecycle smoke with `--summary-json`.
- Passed: parsed saved stdout/result JSON artifacts; stdout exactly matched
  `result.lifecycle_summary`, full result retained `reviewed_export`, and the
  compact summary reported `status=ok`, `source_rows=500`,
  `saved_faq_count=1`, `error_count=0`.
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~66 |
| Validation note | ~124 |
| **Total** | **~190** |
