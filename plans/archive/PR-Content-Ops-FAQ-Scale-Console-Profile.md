# PR-Content-Ops-FAQ-Scale-Console-Profile

## Why this slice exists

The FAQ scale smoke writes a detailed `run_summary.json`, but the CLI itself
returns silently. During large upload testing, that forces operators to open the
JSON artifact before they can tell whether a failure was caused by sparse input
rows, source adapter warnings, or FAQ output checks.

This slice adds a compact console summary so failures are acknowledged at the
command line while keeping the JSON artifact as the source of truth.

## Scope (this PR)

1. Print a one-line pass/fail summary from `smoke_content_ops_faq_scale_run.py`.
2. Include source-density fields from `input_profile` in that line.
3. Include failure type and summary artifact path when the run fails.
4. Add focused tests for the console output.

### Files touched

- `plans/PR-Content-Ops-FAQ-Scale-Console-Profile.md`
- `scripts/smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_scale_run.py`

## Mechanism

`main(...)` will keep calling `run_scale_smoke(...)`, then pass the returned
summary into a small printer. The printer derives a compact profile string from
`input_profile`, such as `source_rows=2/3 skipped_rows=1
missing_source_text=1`, and writes success summaries to stdout and failure
summaries to stderr.

The JSON summary, artifact paths, exit-code behavior, and
`--allow-output-check-failures` semantics remain unchanged.

## Intentional

- This is visibility only; no new fail-closed rule is added.
- The console line is compact and does not duplicate the full JSON summary.
- Programmatic callers of `run_scale_smoke(...)` still receive the same tuple
  and do not print.

## Deferred

- Rich terminal formatting remains out of scope.
- Hosted UI surfacing for input profiles remains a separate UI/dashboard slice.

## Verification

- Focused pytest passed, 17 tests: pytest tests/test_smoke_content_ops_faq_scale_run.py
- Full extracted pipeline checks passed, including 295 reasoning-core tests and
  1,597 extracted Content Ops tests: bash scripts/run_extracted_pipeline_checks.sh
- Local PR review passed: bash scripts/local_pr_review.sh origin/main

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 63 |
| Scale-smoke wrapper | 45 |
| Tests | 28 |
| **Total** | **136** |
