# PR-Content-Ops-FAQ-Lifecycle-Summary-Docs

## Why this slice exists

PR-Content-Ops-FAQ-Lifecycle-Summary-JSON added `--summary-json` so large FAQ
lifecycle database runs can keep stdout compact while writing the full payload
to disk. The operator docs still show `--json`, which prints the full exported
Markdown payload to stdout and recreates the noisy-output problem the code slice
fixed.

This slice updates the existing lifecycle smoke examples to use the compact
summary mode that is already on main.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-docs

1. Update the extracted package README lifecycle smoke examples.
2. Update the host install runbook lifecycle smoke examples.
3. Keep the docs focused on the already-shipped `--summary-json` flow.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Summary-Docs.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`

## Mechanism

Lifecycle smoke examples now pass an `--output-result` file path together with
`--summary-json`. This preserves the full result payload on disk and prints only the compact
`lifecycle_summary` object to stdout.

## Intentional

- Docs only; no lifecycle code, database code, or source-ingestion behavior
  changes.
- This does not document the unmerged artifact runner from PR #850.

## Deferred

- Full artifact-runner runbook snippets remain deferred until the artifact
  runner lands on main.
- Root `HARDENING.md` was scanned; no same-lane parked item is required for this
  docs slice.

## Verification

- Passed: grep sweep confirmed the lifecycle smoke examples now use
  `--summary-json` with `--output-result`.
- Passed: live local DB smoke using the documented `--output-result` +
  `--summary-json` pattern; stdout equaled `lifecycle_summary`, reported
  `status=ok`, `source_rows=4`, `saved_faq_count=1`, and the full result file
  retained `reviewed_export`.
- Passed: `scripts/run_extracted_pipeline_checks.sh` (1839 passed, 1 skipped)
- Pending: `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~65 |
| README examples | ~7 |
| Host runbook examples | ~4 |
| **Total** | **~76** |
