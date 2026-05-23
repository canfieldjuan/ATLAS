# PR-Content-Ops-FAQ-Lifecycle-Artifact-Runner

## Why this slice exists

The FAQ lifecycle smoke can now print compact `--summary-json`, but a real
500-row or 1,000-row database run still requires a long hand-built command and
manual redirection to preserve the stdout, stderr, full payload, and compact
summary as separate artifacts.

This slice adds the thinnest repeatable operator wrapper around the existing
database lifecycle smoke. It does not create a second lifecycle path; it runs
the product smoke command and writes a small artifact bundle that makes input
density, output checks, persistence, export, review status, and failure type
visible after each run.

This is slightly over the 400 LOC target because it introduces a new executable
and its artifact/failure contract tests together; splitting the tests from the
runner would leave the operator command without reviewable behavior proof.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add `scripts/smoke_content_ops_faq_lifecycle_run.py`.
2. Have the wrapper invoke `scripts/smoke_content_ops_faq_lifecycle.py` with
   `--output-result` and `--summary-json`.
3. Capture stdout, stderr, lifecycle result JSON, compact summary JSON, and
   run summary JSON artifacts under an operator-provided
   artifact directory.
4. Add focused tests for success, lifecycle failure, and hard CLI failure
   artifact behavior.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Artifact-Runner.md`
- `scripts/smoke_content_ops_faq_lifecycle_run.py`
- `tests/test_smoke_content_ops_faq_lifecycle_run.py`

## Mechanism

The new wrapper is a subprocess runner, matching the established shape of
`scripts/smoke_content_ops_faq_scale_run.py`.

It builds the existing lifecycle command with:

```bash
--output-result <artifact-dir>/lifecycle_result.json --summary-json
```

Then it captures process stdout and stderr, parses the compact summary stdout
when available, parses the full result payload when available, and writes a
small run summary JSON file with:

- exit code and `ok`
- source path and source format
- artifact paths and sizes
- parsed `lifecycle_summary`
- parsed full lifecycle result
- failure classification for missing result, lifecycle errors, or raw CLI
  errors

## Intentional

- No lifecycle generation, repository, export, review, or source-ingestion logic
  changes.
- The wrapper does not bypass the existing smoke; the existing script remains
  the single execution path.
- No database fixture is checked in; live DB verification stays in the manual
  smoke lane.

## Deferred

- Documentation snippets for every 500/1,000-row operator command are deferred
  until the next real DB run confirms the final artifact names.
- Root `HARDENING.md` was scanned; no same-lane parked item is required for
  this slice.

## Verification

- Passed: `pytest tests/test_smoke_content_ops_faq_lifecycle_run.py -q` (4 passed)
- Passed: live local DB smoke with the checked-in support-ticket CSV; the runner
  wrote the run summary, lifecycle result, compact summary stdout, raw stdout,
  and raw stderr artifacts, and the parsed summary reported
  `status=ok`, `source_rows=4`, `saved_faq_count=1`.
- Passed: `scripts/run_extracted_pipeline_checks.sh` (1839 passed, 1 skipped)
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| Wrapper script | ~226 |
| Tests | ~135 |
| **Total** | **~456** |
