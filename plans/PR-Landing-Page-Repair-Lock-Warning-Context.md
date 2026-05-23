# PR-Landing-Page-Repair-Lock-Warning-Context

## Why this slice exists

`PR-Landing-Page-Repair-Lock-Skip-Warning` made skipped repair locks visible
through a warning log. The warning attached `account_id` and `landing_page_id`
as structured `extra` fields, but plain-text log handlers do not show those
fields unless their formatter is configured for them.

This slice keeps the structured fields and also includes the identifiers in the
warning message itself.

## Scope (this PR)

Ownership lane: content-ops/landing-page-repair-lock-warning-context

1. Include account id and landing-page id in the skipped-lock warning message.
2. Keep the existing structured `extra` fields.
3. Add focused route-level test assertions for the plain-text message.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Repair-Lock-Warning-Context.md` | Plan doc for this warning-context slice. |
| `extracted_content_pipeline/api/generated_assets.py` | Include ids in the skipped-lock warning message. |
| `tests/test_extracted_content_asset_api.py` | Assert ids are present in both log message and structured record fields. |

## Mechanism

The no-`acquire()` fallback already computes the tenant scope and landing-page
id before warning. This slice formats those values into the warning text while
preserving the structured `extra` fields.

## Intentional

- No behavior change to the repair endpoint.
- No lock lifecycle changes.
- No parked repair-lock items are drained in this slice.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock removal after
  rollout.
- `HARDENING.md` still tracks repair lock connection hold time across LLM
  latency.

## Parked hardening

- None added.

## Verification

- Python compile for `extracted_content_pipeline/api/generated_assets.py` ->
  passed.
- Focused pytest for `tests/test_extracted_content_asset_api.py` -> 52 passed.
- Extracted content pipeline validation -> passed.
- Extracted reasoning import guard -> passed.
- Extracted standalone audit -> passed with 0 findings.
- ASCII Python policy -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| API warning | ~10 |
| API test | ~10 |
| Total | ~75 |
