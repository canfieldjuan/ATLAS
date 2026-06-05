# PR-Content-Ops-FAQ-Lifecycle-Summary-JSON

## Why this slice exists

The real 1,000-row FAQ database lifecycle run passed, but `--json` printed the
entire lifecycle payload to stdout, including exported rows and full Markdown.
That made the operator output thousands of lines long even though the useful
readout was already available in `lifecycle_summary`.

This slice adds a compact machine-readable stdout mode for lifecycle smokes so
large runs can write the full payload to `--output-result` while printing only
the summary needed for triage.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Add a `--summary-json` CLI flag to `scripts/smoke_content_ops_faq_lifecycle.py`.
2. Make `--summary-json` print `payload["lifecycle_summary"]` only.
3. Keep existing `--json` full-payload behavior unchanged.
4. Add focused print tests for summary JSON success and failure output.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Summary-JSON.md`
- `scripts/smoke_content_ops_faq_lifecycle.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism

The parser adds `--summary-json` beside the existing `--json` flag. `_main`
passes both flags to `_print_payload(...)`.

`_print_payload(...)` keeps the current ordering:

1. `--json`: print the full payload exactly as today.
2. `--summary-json`: print only the `lifecycle_summary` mapping.
3. no JSON flag: print the compact human console line.

`--json` wins if both flags are passed, preserving the existing full-payload
contract for callers that already use it.

## Intentional

- No lifecycle generation, persistence, export, review, or result payload shape
  changes.
- `--output-result` still writes the full payload; `--summary-json` only changes
  stdout.
- `--json` remains backward compatible and still prints the full payload.

## Deferred

- A similar summary-only stdout mode for other Postgres smoke scripts remains
  deferred until those scripts show the same large-output problem.
- Root `HARDENING.md` was scanned; no parked FAQ-lane item is required for this
  slice.

## Verification

- Passed: `pytest tests/test_smoke_content_ops_faq_lifecycle.py -q` (12 passed)
- Passed: `scripts/run_extracted_pipeline_checks.sh` (1839 passed, 1 skipped)
- Passed: live local DB smoke with `--summary-json`; stdout equaled `lifecycle_summary` and omitted full export payloads while `--output-result` kept the full result.
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| CLI flag and printer | ~25 |
| Tests | ~85 |
| **Total** | **~180** |
