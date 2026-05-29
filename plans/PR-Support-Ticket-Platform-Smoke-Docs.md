# PR: Support Ticket Platform Smoke Docs

## Why this slice exists

PR-Support-Ticket-Platform-CSV-Smoke added a checked platform-shaped CSV fixture
and pinned the operator-facing package smoke against it. The code path is now
covered, but the Content Ops docs still only point operators at the older
packaged support-ticket CSV examples.

This slice documents the cheap pre-LLM smoke so a host can verify common help
desk export headers before running live blog or landing-page generation.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Product polish

1. Add the platform-shaped CSV smoke command to the Content Ops README.
2. Update Content Ops status to mention the checked platform-export fixture.
3. Keep runtime code, fixtures, tests, prompts, and FAQ generation unchanged.

### Files touched

- `plans/PR-Support-Ticket-Platform-Smoke-Docs.md` - Plan doc for this slice.
- `extracted_content_pipeline/README.md` - Document the platform-shaped CSV smoke command.
- `extracted_content_pipeline/STATUS.md` - Record the platform-export fixture in current package status.

## Mechanism

The README now shows
`python scripts/smoke_content_ops_support_ticket_package.py extracted_content_pipeline/examples/support_ticket_platform_export_shapes.csv --require-included-rows --pretty`
as the pre-LLM check for common support-platform exports. The STATUS note
records that the package smoke covers a fixture with Zendesk-, Freshdesk-, and
Intercom-style headers.

## Intentional

- Docs-only. The smoke behavior was implemented and tested in the prior slice.
- This does not claim arbitrary customer exports are proven. Real anonymized
  customer exports remain the follow-up when samples are available.
- This does not update FAQ-owned docs or generation behavior.

## Deferred

- Future PR: run the same smoke against anonymized real customer exports when
  samples are available.
- Future PR: add upload-screen copy for useful export columns when that UI is
  revisited.
- Parked hardening: none.

## Verification

- Platform CSV package smoke command - passed; summary reported 3 included
  rows, 3 FAQ questions, and 0 warnings.
- Whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~65 |
| README | ~20 |
| STATUS | ~10 |
| **Total** | **~95** |

This stays below the 400 LOC soft cap.
