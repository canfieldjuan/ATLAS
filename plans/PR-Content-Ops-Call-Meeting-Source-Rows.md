# Content Ops Call And Meeting Source Rows

## Why This Slice Exists

AI Content Ops already accepts reviews, tickets, surveys, transcripts, and
documents as customer source rows. A common host export shape is sales-call or
meeting data from tools such as call recorders and CRM activity logs. Without
explicit identifiers and collection keys, those rows fall back to generic
document behavior or require hosts to reshape data before import.

## Scope

- Add `calls`, `call_transcripts`, `meetings`, and `meeting_transcripts` as
  source-bundle collection keys.
- Add `call_id`, `meeting_id`, and `recording_id` as source identifiers.
- Infer `sales_call` and `meeting` source types when rows use call/meeting
  identifiers without an explicit `transcript` field.
- Document the new accepted bundle shape.

## Mechanism

The existing `campaign_source_adapters` source-row pipeline stays intact:
collection expansion, source text extraction, evidence construction, and final
opportunity normalization are reused. This slice only teaches that pipeline
about common call/meeting export keys.

## Intentional

- Rows with a `transcript` field still infer `source_type="transcript"` for
  backward compatibility.
- `recording_id` presence infers `sales_call`; hosts should rename ambiguous
  non-call recording identifiers before import.
- `meeting_id` presence infers `meeting`.

## Deferred

- No provider-specific schemas for Gong, Fireflies, Zoom, or HubSpot activity
  exports. Hosts can map those into the generic call/meeting keys first.
- No new packaged example file; the existing source-bundle example remains the
  starter fixture, and focused tests lock the new keys.

## Verification

- Focused source-adapter tests.
- Python compile check for the adapter and focused tests.
- Git diff whitespace check.
- Local PR review wrapper.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `plans/PR-Content-Ops-Call-Meeting-Source-Rows.md`
- `tests/test_extracted_campaign_source_adapters.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Adapter keys | ~20 |
| Tests | ~115 |
| Docs and coordination | ~35 |
| Plan doc | ~75 |
| **Total** | ~245 |
