# PR-Content-Ops-FAQ-Lifecycle-Setup-Failure-Result

## Why this slice exists

The FAQ scale stress probe found that `scripts/smoke_content_ops_faq_lifecycle.py`
does not write `--output-result` when database pool creation fails. In the
100-way lifecycle pressure run, `TooManyConnectionsError` failures left only
stderr, which is too weak for automated survivability probes.

This slice makes that failure mode visible without changing FAQ generation,
repository behavior, migrations, or hosted runtime limits.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-validation

1. Keep the existing lifecycle success path unchanged.
2. Route database pool acquisition failures through the same lifecycle payload
   and result-file writing path used by in-flow failures.
3. Add focused regression coverage for a setup failure with `--output-result`.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Setup-Failure-Result.md`
- `scripts/smoke_content_ops_faq_lifecycle.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism

`run_faq_lifecycle_smoke()` will initialize its default failure state before it
attempts `_create_pool()`. If `_create_pool()` raises, the function records the
exception in `errors`, builds the normal compact `lifecycle_summary`, writes
`args.output_result` when requested, and returns exit code `1`.

The success path still closes the pool and still writes the same payload shape.

## Intentional

- No retry, backoff, semaphore, queue, or connection-pool sizing change in this
  slice; those are runtime hardening decisions, not result visibility.
- CLI argument validation failures remain command-line validation failures.
  This slice targets runtime setup failures after arguments have validated.
- The result payload stays compact and does not add traceback bodies.

## Deferred

- Bounded hosted FAQ concurrency and async/background job execution stay
  deferred to hardening slices.
- A reusable concurrent FAQ load-test runner is deferred; the current need is
  making existing smoke failures machine-readable.

## Verification

- Passed: pytest focused lifecycle smoke tests:
  `tests/test_smoke_content_ops_faq_lifecycle.py` (`13 passed`).
- Passed: extracted pipeline validation bundle:
  `scripts/run_extracted_pipeline_checks.sh`
  (`1850 passed, 1 skipped` for extracted_content_pipeline; extracted
  reasoning checks also passed).
- Pending: local PR review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~69 |
| Lifecycle smoke script | ~14 |
| Lifecycle smoke tests | ~32 |
| **Total** | **~115** |
