# PR-Deflection-Zendesk-Full-Thread-Importer

## Why this slice exists

The deflection report can ingest flat CSV rows, but a real Zendesk export/API
thread is shaped as `{ticket, comments}`. For launch-readiness we need to prove
that full-ticket conversation history can feed the existing deflection package:
customer wording from requester/public comments, public agent replies as answer
evidence, private notes dropped before ingestion, and Zendesk status/CSAT
feeding the #1510 metadata.

This is over the 400 LOC soft cap because the safe vertical slice needs all
three pieces together: runtime mode routing, a provider-shape normalizer, and
private-note/auto-ack/CSAT failure-path tests plus a compact real-shape fixture.
Splitting the tests or fixture out would leave the importer mergeable without
the evidence that it does not leak private notes or count auto-acks as answers.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a Zendesk full-thread JSON importer behind `importer_mode="full_thread"`
   on the existing deflection submit blob path. Omitted mode stays CSV.
2. Normalize Zendesk `{ticket, comments}` records into existing support-ticket
   rows; reuse `build_support_ticket_input_package` downstream.
3. Add CI-facing tests for role/public/private mapping, auto-ack rejection,
   malformed-shape handling, API routing, and default CSV compatibility.
4. Add local full-artifact proof against the generated Zendesk trial export.

### Review Contract
- Acceptance criteria:
  - [ ] Default deflection submit behavior remains CSV when `importer_mode` is
        omitted.
  - [ ] `importer_mode="full_thread"` parses Zendesk JSON blob bytes into rows
        with customer wording, public-agent `resolution_text`, status, and CSAT.
  - [ ] `public=false` comments never appear in row text, resolution evidence,
        package examples, or submit diagnostics.
  - [ ] Generic public auto-ack replies do not count as resolution evidence.
  - [ ] Bad mode and malformed JSON fail closed with clear 422/400 responses.
- Affected surfaces: extracted package importer, deflection submit API,
  extracted-checks tests.
- Risk areas: private-note leakage, false resolution evidence, API
  compatibility, CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/support_ticket_zendesk_thread.py`
- `plans/PR-Deflection-Zendesk-Full-Thread-Importer.md`
- `tests/fixtures/zendesk_full_thread_seed_sample.json`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

`extracted_content_pipeline/support_ticket_zendesk_thread.py` exposes a small
normalizer that accepts decoded JSON mappings/lists or JSON bytes and returns
rows plus warnings. The submit API chooses CSV vs full-thread parsing after the
existing bounded blob fetch, using `importer_mode` from the validated payload.

The normalizer uses `ticket.requester_id` to split comments: public comments
from the requester become customer wording; public comments from other authors
become `resolution_text` unless they match generic auto-ack wording; private
comments are skipped. Zendesk `satisfaction_rating.score` is copied only for
real ratings (`good`/`bad` or numeric-like values), so `unoffered` does not
inflate CSAT presence.

## Intentional

- No Zendesk API fetch or credential handling in this slice; the runtime input
  is an already-exported JSON blob.
- Multipart/browser upload remains CSV-only for now. Full-thread mode is wired
  to JSON `blob_url` submit first so the backend contract can land without a
  portfolio UI upload-type change.
- The importer emits ordinary support-ticket rows rather than adding a new
  downstream package contract; status, CSAT, clustering, truncation, and
  resolution diagnostics stay owned by `build_support_ticket_input_package`.

## Deferred

- Portfolio UI JSON/Zendesk upload UX and copy.
- Direct Zendesk API export/import using tenant-scoped credentials.
- Full 49-ticket generated artifact is local proof, not a checked-in fixture;
  CI uses a compact real-shape sample fixture.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_support_ticket_input_package.py::test_zendesk_full_thread_rows_suppress_private_first_description tests/test_extracted_support_ticket_input_package.py::test_zendesk_full_thread_rows_keep_substantive_agent_reply_after_boilerplate tests/test_extracted_support_ticket_input_package.py::test_zendesk_full_thread_rows_preserve_public_roles_and_drop_private_notes tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob -q` - passed, 4 tests.
- `python -m pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_content_deflection_submit.py -q` - passed, 100 tests.
- Local full-artifact proof against the generated 49-ticket Zendesk trial
  artifact kept outside the repo - passed: 49 rows, 0 warnings, resolution
  evidence count 38, status summary `{'resolved': 29, 'open': 20}`, CSAT
  `24 good / 5 bad`, no private-note leak, no auto-ack leak. CI uses the
  checked-in compact real-shape fixture.
- `scripts/validate_extracted_content_pipeline.sh` via bash - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` via bash - passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` - passed.
- `scripts/run_extracted_pipeline_checks.sh` via bash - passed: `extracted_reasoning_core` 295 passed; `extracted_content_pipeline` 4008 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 96 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/support_ticket_zendesk_thread.py` | 247 |
| `plans/PR-Deflection-Zendesk-Full-Thread-Importer.md` | 118 |
| `tests/fixtures/zendesk_full_thread_seed_sample.json` | 133 |
| `tests/test_extracted_content_deflection_submit.py` | 112 |
| `tests/test_extracted_support_ticket_input_package.py` | 182 |
| **Total** | **891** |
