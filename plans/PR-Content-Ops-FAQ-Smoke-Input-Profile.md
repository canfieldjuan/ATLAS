# PR-Content-Ops-FAQ-Smoke-Input-Profile

## Why this slice exists

After the lifecycle warning visibility slice, the lifecycle smoke reports warning
codes but still lacks the raw/usable row profile that the FAQ scale smoke already
prints and stores. For large uploads, reviewers need both pieces: how many rows
were in the file, how many normalized into usable source rows, and which warning
codes explain any drop. This slice adds that profile to the lifecycle smoke by
sharing the existing scale-smoke profiling logic instead of copying it.

The final diff is above the 400 LOC soft cap because the review fix adds a
load-failure regression path while the slice also extracts duplicated scale
smoke profiling into one shared helper. Splitting the helper extraction from the
lifecycle profile would leave either duplicated profiling semantics or a
half-wired lifecycle profile in review, so this remains one cohesive IO
visibility slice.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Extract FAQ smoke input profiling helpers into a shared script module.
2. Keep the scale smoke's existing `input_profile` output shape unchanged.
3. Add the same `input_profile` payload to the FAQ lifecycle smoke.
4. Add lifecycle assertions for clean, 1,000-row, and warning-failure profiles.
5. Add the shared helper to the extracted-checks workflow path filters.

### Files touched

- `plans/PR-Content-Ops-FAQ-Smoke-Input-Profile.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/content_ops_faq_smoke_profile.py`
- `scripts/smoke_content_ops_faq_lifecycle.py`
- `scripts/smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`

## Mechanism

The new helper owns raw-row counting and normalized-row profiling for FAQ source
files. The scale smoke calls `load_source_input_profile(...)` as before, while
the lifecycle smoke uses `raw_row_profile(...)` plus
`input_profile_from_loaded(...)` around the `loaded` result it already needs for
generation. That keeps lifecycle from parsing the source file twice.

## Intentional

- This is a small refactor plus lifecycle visibility. The scale smoke payload
  shape stays compatible with existing tests.
- The lifecycle smoke keeps `normalization_warnings` for the compact
  warning-focused field added in the prior slice; `input_profile` is the broader
  row-count profile.

## Deferred

- Live database execution remains deferred because this environment has neither
  `EXTRACTED_DATABASE_URL` nor `DATABASE_URL` set.
- Browser upload coverage remains deferred until the UI upload path is active.

## Verification

- `pytest tests/test_smoke_content_ops_faq_scale_run.py tests/test_smoke_content_ops_faq_lifecycle.py -q` - 26 passed.
- `scripts/run_extracted_pipeline_checks.sh` - 1828 passed, 1 skipped.
- `scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Shared profile helper | ~175 |
| Smoke integrations/tests | ~180 |
| Workflow path filters | ~5 |
| **Total** | **~440** |

Actual diff: +332 / -110.
