# PR: Support Ticket Platform CSV Smoke

## Why this slice exists

PR-Support-Ticket-CSV-Shape-Coverage broadened support-ticket header aliases
with dict-row tests. The remaining gap is the file boundary: the operator-facing
package smoke should prove those platform-shaped headers survive real CSV
loading before any DB or LLM work starts.

This slice adds a tiny committed export-shaped fixture and runs it through the
existing support-ticket package smoke path.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Robust testing

1. Add a small synthetic CSV fixture with Zendesk-, Freshdesk-, and
   Intercom-shaped headers.
2. Pin the existing smoke summary against that fixture.
3. Keep package semantics, prompts, FAQ generation, and live LLM paths unchanged.

### Files touched

- `plans/PR-Support-Ticket-Platform-CSV-Smoke.md` - Plan doc for this slice.
- `extracted_content_pipeline/examples/support_ticket_platform_export_shapes.csv` - Synthetic platform-shaped CSV fixture.
- `tests/test_smoke_content_ops_support_ticket_package.py` - Smoke test that loads the fixture through the file path.

## Mechanism

The fixture uses common support-platform export names such as `Requester email`,
`Ticket Subject`, `Message`, `Latest message`, `Conversation title`, and
`Conversation body`. The smoke test calls
`build_support_ticket_package_smoke_summary(...)` on the committed CSV and
asserts the included row count, dated-window label, FAQ questions, contact
examples, and the customer-message-before-latest-reply precedence.

## Intentional

- This is not a real customer export. It is a deterministic fixture for the
  file-loader boundary.
- This does not add another parser or provider branch. The existing CSV loader
  and support-ticket normalizer remain the source of truth.
- This does not touch FAQ output generation, which remains owned by the FAQ
  session.

## Deferred

- Future PR: run the same smoke against anonymized real customer exports when
  samples are available.
- Future PR: add upload-screen copy for useful export columns when that UI is
  revisited.
- Parked hardening: none.

## Verification

- Fixture smoke command - passed and printed 3 included rows, 3 FAQ questions,
  dated-window metadata, and zero warnings.
- Focused fixture smoke pytest - passed.
- Support-ticket smoke + input-package pytest - 33 passed.
- Python compile over the smoke script and smoke test - passed.
- Whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~65 |
| CSV fixture | ~5 |
| Smoke test | ~40 |
| **Total** | **~110** |

This stays below the 400 LOC soft cap.
