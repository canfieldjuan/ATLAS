# PR-Content-Ops-FAQ-Lifecycle-Console-Profile

## Why this slice exists

The FAQ lifecycle smoke now writes a structured `input_profile`, but the
human-readable console output still hides it. Manual lifecycle runs, including
future real-DB 1,000-row runs, should show the same compact row-count and warning
summary that the scale smoke already shows. This slice shares the console
formatter and uses it in lifecycle output.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Move the FAQ input-profile console formatter into the shared smoke profile
   helper.
2. Keep the scale smoke console output unchanged while using the shared
   formatter.
3. Add lifecycle console output for success and failure paths.
4. Add tests for lifecycle console success and failure summaries.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-Console-Profile.md`
- `scripts/content_ops_faq_smoke_profile.py`
- `scripts/smoke_content_ops_faq_lifecycle.py`
- `scripts/smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism

The shared helper exposes `console_input_profile(profile)`. The scale smoke
delegates its existing private wrapper to the shared helper, and the lifecycle
smoke includes that same string in `_print_payload(...)` for both pass and fail
output.

## Intentional

- This is console visibility only. It does not change lifecycle pass/fail
  behavior or JSON payload structure.
- The scale smoke keeps its private wrapper for compatibility with existing
  tests while moving implementation into the shared helper.

## Deferred

- Live database execution remains deferred because this environment has neither
  `EXTRACTED_DATABASE_URL` nor `DATABASE_URL` set.
- Browser upload coverage remains deferred until the UI upload path is active.

## Verification

- Passed: `pytest tests/test_smoke_content_ops_faq_scale_run.py tests/test_smoke_content_ops_faq_lifecycle.py -q` (28 passed)
- Passed: `scripts/run_extracted_pipeline_checks.sh` (1830 passed, 1 skipped)
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Shared console formatter | ~30 |
| Lifecycle console tests | ~35 |
| Smoke integrations | ~15 |
| **Total** | **~150** |
