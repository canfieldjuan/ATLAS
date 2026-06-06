# PR: Support Ticket Provider Warning Route Proof

## Why this slice exists

PR #938 made input-provider diagnostics visible on Content Ops API responses.
The lower-level route tests used synthetic provider packages. The remaining
integration gap is proving a real `SupportTicketInputProvider` package emits
the support-ticket truncation warning through the preview route, while the
Atlas request-aware inline provider keeps its existing request-size cap.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add a real support-ticket provider preview-route test with 1,005 loaded
   support-ticket-shaped rows.
2. Assert the route remains runnable while surfacing the exact truncation
   warning and operational metadata.
3. Add a companion Atlas inline-provider test proving inline payloads over the
   request cap still fail as 422 before provider packaging.
4. Keep the tests offline: no DB, LLM, file-ingestion, or FAQ generator changes.

### Files touched

- `plans/PR-Atlas-Support-Ticket-Provider-Warning-Route-Proof.md`
- `tests/test_atlas_content_ops_input_provider.py`

## Mechanism

The first test mounts the extracted Content Ops router with
`SupportTicketInputProvider(source_material_loader=...)`, returns 1,005 loaded
rows, and asserts the response includes
`input_provider.warnings[0].code == "ticket_rows_truncated"` plus the
allowlisted source-row counts.

The companion Atlas-provider test sends the same row count inline and asserts
the request validator rejects it before provider packaging. That keeps the
contract honest: large files need a loader/import path to produce truncation
diagnostics; oversized inline JSON is not accepted.

## Intentional

- No production code changes. This is route proof for already-merged API
  behavior and request-size policy.
- No UI changes; PR #939 owns frontend display.
- No file-ingestion or persisted upload lookup changes. Those remain owned by
  the host ingestion lane.

## Deferred

- Future PR: hosted upload policy can decide whether large files should be
  rejected, warned, or queued before generation.
- Parked hardening: none. `HARDENING.md` was scanned; the current entry is FAQ
  scale/backpressure work owned by the FAQ generation lane.

## Verification

- Focused route proof tests - 4 passed.
- Full host provider suite - 13 passed, 1 warning from the environment's
  deprecated `pynvml` import.
- Python compile for `tests/test_atlas_content_ops_input_provider.py` - passed.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| Route tests | ~75 |
| **Total** | **~135** |
