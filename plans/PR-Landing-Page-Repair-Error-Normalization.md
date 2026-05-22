# PR-Landing-Page-Repair-Error-Normalization

## Why this slice exists

The saved landing-page repair UI reads `repair_result.errors[].blockers` when
a repair fails. The service already returned that key for quality-blocked
failures, but the unparseable-response and provider-exception paths only
returned alternate or generic error details. That made the UI contract depend
on which failure path happened first.

## Scope (this PR)

Ownership lane: content-ops/landing-page-repair-error-normalization

1. Normalize saved-draft repair failure payloads so blocker-aware paths expose
   `errors[].blockers`.
2. Keep the existing `quality_blockers` alias on unparseable repair responses
   for compatibility with existing generation-path callers.
3. Add focused service tests for saved-draft unparseable-response and provider
   exception repair failures.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Repair-Error-Normalization.md` | Plan doc for this contract cleanup. |
| `extracted_content_pipeline/landing_page_generation.py` | Add normalized blockers to saved-draft repair failure payloads. |
| `tests/test_extracted_landing_page_generation.py` | Cover blocker payloads for saved-draft repair parse and provider failures. |

## Mechanism

Saved-draft repair already computes the current readiness repair issues before
it calls the LLM. This slice threads those issues into the two early failure
responses that happen before a repaired draft can be quality checked:
provider exceptions and unparseable repair responses. Both paths now expose the
same `errors[].blockers` key that final quality-blocked failures already use.

The unparseable-response path also keeps `quality_blockers` as an alias so
existing generation-path callers keep their current behavior while the saved
repair UI gets one stable key to read.

## Intentional

- This slice does not change the public API route shape beyond the normalized
  service payload it already returns.
- This slice does not change the generation-path `quality_blockers` behavior.
- `repair_update_missed` remains blocker-free because that path means the
  repaired content passed readiness but the repository update missed the row.

## Deferred

- `PR-Landing-Page-Repair-Cost-Guard` can add explicit cost/idempotency
  controls around repeated saved-draft repair attempts if needed.
- `PR-Landing-Page-Repair-Error-Schema` can centralize all repair error payload
  construction if another repair surface starts returning the same shape.

## Verification

- Python compile for `extracted_content_pipeline/landing_page_generation.py`
  -> passed.
- Focused pytest for `tests/test_extracted_landing_page_generation.py` and
  `tests/test_extracted_content_asset_api.py` -> 84 passed.
- Extracted content pipeline validation -> passed.
- Extracted reasoning import guard -> passed.
- Extracted standalone audit -> passed with 0 findings.
- ASCII Python policy -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Repair error payloads | ~5 |
| Service tests | ~55 |
| Total | ~135 |
