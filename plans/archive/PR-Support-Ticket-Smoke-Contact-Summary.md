# PR: Support Ticket Smoke Contact Summary

## Why this slice exists

The platform CSV smoke now proves common help desk export headers survive file
loading, and the docs say the smoke can verify contact-email fields before DB or
LLM work. The smoke summary, however, reports ticket counts and customer wording
but not whether contact email columns were actually recognized.

This slice closes that visibility gap at the source by adding a non-PII contact
email count to the existing support-ticket package smoke summary.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Product polish

1. Add `contact_email_count` to the support-ticket package smoke summary.
2. Count only included normalized source rows that have a non-empty
   `contact_email`.
3. Pin the existing platform CSV fixture so the smoke proves all three platform
   contact-email aliases are recognized without printing the email addresses.

### Files touched

- `plans/PR-Support-Ticket-Smoke-Contact-Summary.md` - Plan doc for this slice.
- `scripts/smoke_content_ops_support_ticket_package.py` - Add a non-PII contact email count to the summary.
- `tests/test_smoke_content_ops_support_ticket_package.py` - Pin the count for the platform-shaped fixture and existing no-email scenarios.

## Mechanism

`build_support_ticket_package_smoke_summary(...)` already has access to the
normalized `source_material` rows emitted by the support-ticket package. This
slice counts included rows where `contact_email` is present after normalization
and adds only the integer count to the JSON summary.

## Intentional

- The smoke does not print contact email values. Customer exports can contain
  PII, so the operator-facing diagnostic stays count-only.
- This does not change package generation inputs, prompts, FAQ generation, live
  LLM behavior, or persisted draft shape.
- This does not add a new parser; it observes the existing normalized package.

## Deferred

- Future PR: run the platform smoke against anonymized real customer exports
  when samples are available.
- Future PR: add upload-screen copy for useful export columns when that UI is
  revisited.
- Parked hardening: none.

## Verification

- Platform CSV package smoke command - passed; summary reported
  `contact_email_count: 3` without printing email values.
- Support-ticket package smoke pytest - 9 passed.
- Python compile over the smoke script and smoke test - passed.
- Whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Smoke summary | ~10 |
| Tests | ~10 |
| **Total** | **~90** |

This stays below the 400 LOC soft cap.
