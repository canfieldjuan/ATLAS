# PR: Support Ticket CSV Shape Coverage

## Why this slice exists

The support-ticket provider has live validation on the packaged SaaS demo CSV
and scale tests, but the deferred acceptance backlog still calls out broader
customer CSV shape coverage. Real small-company exports will not all use our
canonical `ticket_id`, `subject`, `description`, `created_at`, and `email`
headers.

This slice adds a narrow source-side compatibility pass for common support
platform export names so customer uploads are less brittle before they reach
landing/blog generation.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Robust testing

1. Add focused normalization coverage for Zendesk-, Freshdesk-, and
   Intercom-shaped ticket rows.
2. Add only the alias keys needed for those shapes.
3. Keep the normalized Content Ops package shape unchanged.

### Files touched

- `plans/PR-Support-Ticket-CSV-Shape-Coverage.md` - Plan doc for this slice.
- `extracted_content_pipeline/support_ticket_input_package.py` - Add common
  export header aliases to the existing normalizer.
- `tests/test_extracted_support_ticket_input_package.py` - Pin mixed support
  platform CSV-shaped rows.

## Mechanism

The support-ticket input package already normalizes keys by stripping
punctuation/case, so this only extends the existing alias tuples. The new test
feeds three representative dict rows through `build_support_ticket_input_package`
and asserts the normalized `source_material`, FAQ questions, dated-window flag,
and contact emails are preserved. A focused precedence test also keeps
customer-authored `message` fields ahead of reply-style `latest_message` fields
when both appear in the same export.

## Intentional

- This does not add platform-specific parser classes. The existing normalized
  key lookup is enough for these small header differences.
- This does not change generated content prompts or FAQ-owned behavior.
- This does not claim exhaustive support for every export column a platform can
  emit. It covers the fields the generation path actually consumes.

## Deferred

- Future PR: add a file-based smoke around real customer exports when anonymized
  samples are available.
- Future PR: add UI copy that tells users which columns are useful once the
  upload screen is revisited.
- Parked hardening: none.

## Verification

- Focused support-ticket CSV shape pytest - 1 passed.
- Python compile over the support-ticket input package and test file - passed.
- Full support-ticket input package/provider pytest - 35 passed.
- Whitespace check - passed.
- Existing deferred references to broader customer CSV/live validation remain
  intentionally because this slice adds deterministic normalization coverage,
  not live runs against anonymized customer exports.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Alias keys | ~15 |
| Tests | ~75 |
| **Total** | **~160** |

This stays below the 400 LOC soft cap.
