# PR-FAQ-Macro-Writeback-Content-Hash-Update
## Why this slice exists
PR-FAQ-Macro-Writeback-Scheduled-Publish intentionally deferred body-only FAQ
edit detection. The scheduled publisher currently selects a previously
published macro for update when the mapping is missing, pending, unpublished,
or when title/category drift. If an approved FAQ answer changes but its
question and topic stay the same, the scheduled task can skip the draft and
leave Zendesk with stale customer-facing macro text.
## Scope (this PR)
Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening
1. Add a deterministic content hash for the platform-agnostic macro body
   contract.
2. Store that hash in Zendesk mapping metadata on create/update/reservation.
3. Select scheduled publish candidates when a published mapping is missing the
   hash or has a stale hash, so body-only edits update instead of no-op.
4. Add focused tests for hash stability, Zendesk metadata persistence, and
   scheduled candidate selection on body-only edits.
### Files touched
- `plans/PR-FAQ-Macro-Writeback-Content-Hash-Update.md`
- `extracted_content_pipeline/faq_macro_writeback.py`
- `extracted_content_pipeline/faq_macro_writeback_zendesk.py`
- `atlas_brain/autonomous/tasks/faq_macro_writeback_scheduled_publish.py`
- `tests/test_extracted_ticket_faq_macro_writeback.py`
- `tests/test_extracted_ticket_faq_macro_writeback_zendesk.py`
- `tests/test_autonomous_faq_macro_writeback_scheduled_publish.py`
## Mechanism
The macro core exposes a small helper that canonicalizes the publish-relevant
fields (`title`, `body`, `category`) and returns a stable SHA-256 digest.
Zendesk writes that digest into mapping metadata whenever it reserves or
persists a mapping. The scheduled selector continues to reuse the shared
preview/double gate, but `_has_unpublished_macro(...)` also compares the stored
`content_hash` with the current macro digest. A missing stored hash is treated
as needs-update so existing mappings backfill safely on their next scheduled
pass.
## Intentional
- No migration is needed; mapping metadata is already JSONB and optional.
- No live Zendesk API calls; CI uses existing fake transport/repository tests.
- No adapter work for Freshdesk, Help Scout, or Gorgias in this slice. Their
  write APIs are logged locally for future adapter slices, but this PR hardens
  the shipped Zendesk path.
- The selector still calls the existing publish service; the double gate and
  provider idempotency remain the source of truth.
## Deferred
- Future PR: operator-facing telemetry beyond the scheduled task result
  summary if operators need a dashboard view of update/backfill counts.
Parked hardening: none.
## Verification
- Python compile check for the changed modules and tests - passed.
- Focused pytest for macro core, Zendesk adapter, and scheduled selector - 25 passed.
- Extracted package validation - passed.
- Extracted reasoning-import guard - passed.
- Extracted standalone audit with fail-on-debt - passed.
- ASCII check for extracted package Python files - passed.
- Full extracted pipeline CI mirror - 2877 passed, 10 skipped, 1 existing torch/pynvml warning.
## Estimated diff size
| Area | LOC |
|---|---:|
| Plan | 64 |
| Core helper | 19 |
| Zendesk adapter | 24 |
| Scheduler selector | 2 |
| Tests | 80 |
| **Total** | **189** |
