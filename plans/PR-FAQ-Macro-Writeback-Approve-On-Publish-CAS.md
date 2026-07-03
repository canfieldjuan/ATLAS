# PR-FAQ-Macro-Writeback-Approve-On-Publish-CAS

## Why this slice exists

The paid Resolution Audit -> Zendesk macro publish route (PR #1955, lane
content-ops/resolution-audit-zendesk-writeback) needs the shared
`FAQMacroWritebackPublishService` to support a new behavior: an explicit paid
publish action is the *approval* for a generated (`status='draft'`) FAQ, and
that promotion plus the publish-mark must be race-safe against concurrent
review decisions. This foundation slice lands that service/port behavior on its
own so the route slice can delegate to it.

Splitting the service layer out of #1955 also keeps each PR's diff small enough
that the `audit_cross_layer_callers.py` hint audit (which re-scans the repo per
changed symbol and is O(changed_symbols x tracked_files)) stays under the
`pre_push_audit.yml` 10-minute job cap. The combined change was ~78 changed
symbols and timed the audit out; this foundation half is ~40, the route half
~38 -- both back under the budget that passed earlier in the arc.

## Scope (this PR)

Ownership lane: content-ops/resolution-audit-zendesk-writeback
Slice phase: Vertical slice

1. Add an opt-in `approve_draft` to `FAQMacroWritebackPublishService.publish_faq_draft`
   with race-safe terminal-state semantics (promotion + publish-mark are
   compare-and-set; the summary reflects the true end state).
2. Add an optional compare-and-set `expected_status` to the
   `TicketFAQRepository.update_status` port and its Postgres implementation.
3. Prove the state machine's truth table and update every in-repo repository
   fake used with this service to the widened port.

### Review Contract

- Acceptance criteria:
  - [ ] `publish_faq_draft(..., approve_draft=True)` promotes only an exact
        `'draft'` status, and only when the draft is *fully* publishable
        (`publishable>0 AND skipped==0`); a partial or empty generated draft is
        never promoted (so it cannot enter the approved-filtered FAQ search
        projection).
  - [ ] Promotion (`draft->approved`) and the publish-mark
        (`approved->published`) are compare-and-set; a concurrent review
        decision landing after `get_draft` or mid-publish wins, and the summary
        reports `ok=false` / `status_conflict=true` rather than a false success.
  - [ ] An already-`published` draft is a retry through the provider's
        idempotent mapping: no status write, honest `draft_status='published'`.
  - [ ] `rejected`/`archived`/`expired` drafts are never revived;
        `approve_draft=False` keeps the AI Content Station and scheduled paths
        byte-for-byte unchanged.
  - [ ] `update_status(..., expected_status=...)` adds `AND status = $N` in
        Postgres; omitting it keeps the unconditional SQL shape.
- Reachability proof: service-level tests exercise the real
  `FAQMacroWritebackPublishService` against compare-and-set-honoring fakes;
  the Postgres compare-and-set is proven with SQL-shape tests plus a gated
  real-Postgres race integration test (`@pytest.mark.integration`).
- Affected surfaces: shared publish service, `TicketFAQRepository` port + its
  Postgres adapter, FAQ macro publish attempt semantics, the FAQ search
  projection side effect of an approval.
- Risk areas: concurrency (TOCTOU on status), backward compatibility of the
  widened port, the never-revive contract, idempotent retries.
- Reviewer rules triggered: R1, R2, R4, R6, R10.

### Files touched

- `extracted_content_pipeline/faq_macro_writeback.py`
- `extracted_content_pipeline/faq_macro_writeback_postgres.py`
- `extracted_content_pipeline/faq_macro_writeback_publish.py`
- `extracted_content_pipeline/storage/migrations/335_ticket_faq_macro_publish_attempt_conflict.sql`
- `extracted_content_pipeline/ticket_faq_ports.py`
- `extracted_content_pipeline/ticket_faq_postgres.py`
- `plans/PR-FAQ-Macro-Writeback-Approve-On-Publish-CAS.md`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_content_ops_faq_macro_writeback_flow.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_extracted_ticket_faq_macro_writeback_postgres.py`
- `tests/test_extracted_ticket_faq_macro_writeback_publish.py`
- `tests/test_extracted_ticket_faq_postgres.py`
- `tests/test_faq_macro_writeback_live_zendesk_smoke.py`
- `tests/test_seed_faq_macro_writeback_live_smoke_draft.py`

## Mechanism

`publish_faq_draft` is one three-phase transaction with explicit terminal-state
semantics:

1. **Eligibility** -- build the macro preview under approved-semantics. A
   generated draft is promoted only when fully publishable (`publishable>0 AND
   skipped==0`); otherwise no status write and `ok=false`.
2. **Promotion** -- for an observed `'draft'`, compare-and-set `draft->approved`
   (`expected_status='draft'`); if it loses, a concurrent review decision won,
   so it fails closed with no publish.
3. **Publish + mark** -- publish the macros, then compare-and-set
   `approved->published` (`expected_status='approved'`). `draft_status` reports
   the *effective* post-promotion status, so an already-`published` retry does
   no mark (idempotent) while a real promotion still marks. A new
   `status_conflict` flag, folded into `ok`, is set when a needed mark loses to
   a mid-flight review decision: the macro is published externally but the
   caller sees a conflict, not a false success.

The port gains an optional `expected_status`; the Postgres adapter adds
`AND status = $N` only when provided. A compare-and-set *miss* is
disambiguated by re-reading the stored status: an already-`published` row is
an idempotent success, a review-decided row (reject/archive) is a real
conflict. A `published` draft is an idempotent retry for every caller, not
only `approve_draft=True`. `status_conflict` is persisted to the append-only
attempt history (migration adds the column). Every in-repo fake used with the
service accepts the optional argument.

## Intentional

- `approve_draft` defaults `False`, so the Generated Asset Review and scheduled
  publish paths are unchanged; this is a backward-compatible port + service
  extension.
- The route + host wiring that consume this (the paid Resolution Audit
  `publish-macros` endpoint) land in the follow-up route PR on
  `claude/pr-content-ops-resolution-audit-zendesk-writeback` (#1955), rebased
  onto this foundation once merged.
- No live Zendesk call in CI: the macro publish provider is the true external
  boundary; the real `ZendeskMacroPublishProvider` is exercised only for its
  no-credentials fail-closed short-circuit.

## Deferred

- The paid Resolution Audit `POST /deflection-reports/{request_id}/publish-macros`
  route, its host factory (`build_content_ops_faq_macro_publish_service`), and
  the Content Ops router wiring -- follow-up route PR #1955 (depends on this).

Parked hardening: none.

## Verification

- `py_compile` of the four changed service/port modules -- pass.
- `python -m pytest` foundation set (service + CAS + consumers) -- 189 passed.
- `python -m pytest tests/test_extracted_ticket_faq_postgres.py -m integration -q` against real Postgres -- compare-and-set race passes.
- `scripts/run_extracted_pipeline_checks.sh` -- 5083 passed, 21 skipped.
- `python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --min-score 8` -- ratchet passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_macro_writeback.py` | 1 |
| `extracted_content_pipeline/faq_macro_writeback_postgres.py` | 8 |
| `extracted_content_pipeline/faq_macro_writeback_publish.py` | 145 |
| `extracted_content_pipeline/storage/migrations/335_ticket_faq_macro_publish_attempt_conflict.sql` | 8 |
| `extracted_content_pipeline/ticket_faq_ports.py` | 9 |
| `extracted_content_pipeline/ticket_faq_postgres.py` | 11 |
| `plans/PR-FAQ-Macro-Writeback-Approve-On-Publish-CAS.md` | 151 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 1 |
| `tests/test_content_ops_faq_macro_writeback_flow.py` | 1 |
| `tests/test_extracted_content_asset_api.py` | 12 |
| `tests/test_extracted_ticket_faq_macro_writeback_postgres.py` | 4 |
| `tests/test_extracted_ticket_faq_macro_writeback_publish.py` | 428 |
| `tests/test_extracted_ticket_faq_postgres.py` | 101 |
| `tests/test_faq_macro_writeback_live_zendesk_smoke.py` | 1 |
| `tests/test_seed_faq_macro_writeback_live_smoke_draft.py` | 1 |
| **Total** | **882** |
