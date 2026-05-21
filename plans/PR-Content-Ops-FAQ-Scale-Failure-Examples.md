# PR-Content-Ops-FAQ-Scale-Failure-Examples

## Why this slice exists

The FAQ scale smoke now reports input density in JSON and on the console, but
operators still need concrete examples for the two failure modes we care about
in 1,000-row validation: sparse source material versus FAQ output checks
failing after enough usable rows were loaded.

This slice adds static run-summary examples so the next large-row investigation
can compare its summary artifact against known shapes instead of treating every failure
as a generic generator problem.

## Scope (this PR)

1. Add a density-limited run-summary example.
2. Add a healthy-input output-check failure example.
3. Document how to compare a real 1,000-row run summary against the
   examples.
4. Add a fixture test that keeps the examples valid JSON with the expected
   failure-profile fields.

### Files touched

- `plans/PR-Content-Ops-FAQ-Scale-Failure-Examples.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/examples/faq_scale_density_limited_summary.json`
- `extracted_content_pipeline/examples/faq_scale_output_check_failure_summary.json`
- `tests/test_smoke_content_ops_faq_scale_run.py`

## Mechanism

The two example JSON files are compact run-summary-shaped payloads. The
density-limited example shows a high raw row count but low
`usable_source_count`, high `skipped_row_count`, and missing source text. The
output-check example shows healthy input density with
`failure.type="output_checks"` and failed checks in the FAQ result.

The README points operators at both files after running the large scale smoke.
The test loads the checked-in examples and asserts the distinction that matters:
source-density examples have poor usable-source ratio, while output-check
examples have healthy source density and failed output checks.

## Intentional

- No runtime behavior changes; this slice only adds reference artifacts and
  documentation.
- The examples are compact summaries, not full generated Markdown bodies.
- Counts are illustrative, not claims about a specific live CFPB run.

## Deferred

- Running and archiving a fresh live 1,000-row CFPB artifact remains a separate
  operator action because CFPB availability is external.
- Additional source-specific examples can be added when another real upload
  exposes a new failure mode.

## Verification

- Focused pytest passed, 18 tests: pytest tests/test_smoke_content_ops_faq_scale_run.py
- Local PR review passed: bash scripts/local_pr_review.sh origin/main

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 71 |
| README | 15 |
| Example JSON | 82 |
| Tests | 24 |
| **Total** | **192** |
