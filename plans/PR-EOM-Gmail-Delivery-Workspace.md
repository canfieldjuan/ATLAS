# PR-EOM-Gmail-Delivery-Workspace

## Why this slice exists

The Billing & Payments workspace coordination issue [#2362](https://github.com/canfieldjuan/ATLAS/issues/2362) requires an operator to recover an approved Gmail-PDF invoice after a browser reload or a later stale-source reconciliation.  ATLAS already persists the approval, immutable PDF, no-send Gmail draft, and proof-gated sent-mail outcome, but the durable billing-run read returns only candidate snapshots.  The Website therefore loses the approval/PDF/draft identifiers held in page-local maps and cannot safely resume an existing delivery workflow from a saved run.

This slice is above the 400-LOC soft target because one safe recovery read must compose five existing durable surfaces and prove their no-send, false-sent, cross-run-reuse, pagination, schema-unavailable, and real-route boundaries together.  Splitting the join/state projection from its PostgreSQL and ASGI proof would create a reusable financial-delivery read with no evidence that it preserves the lifecycle distinction this slice exists to protect.

### Problem-derived contract

- Root cause: Durable Gmail delivery evidence is split across the existing approval, PDF-artifact, Gmail-draft, and sent-reconciliation records, while the existing run reader intentionally projects only immutable pre-approval candidates.  A client reload consequently has no canonical read path to discover a prior exact approval and its delivery lifecycle, especially when an identical candidate is reused by a later run.
- Correct fix must touch/change: Add one authenticated, bounded read projection under the existing receivables API.  It must join an explicitly requested run's exact candidate fingerprint to any matching immutable approval, then project the existing PDF, Gmail-draft, reconciliation, and invoice lifecycle evidence without calling Gmail or changing ATLAS state.  Tests must prove pagination, cross-run exact-approval reuse, authentication, no-send behavior, false-sent resistance, and real FastAPI reachability.
- Must not change: No invoice, payment, allocation, PDF, Gmail draft, Gmail send, sent-mail reconciliation, manual Square, candidate generation, run reconciliation, migration, MCP behavior, or customer-visible email/PDF content changes.  ATLAS remains the only financial ledger; tracker and Website consumers are deferred until this provider is deployed.

## Scope (this PR)

Ownership lane: eom/billing-gmail-delivery-state
Slice phase: Vertical slice
Max files: 6

1. Add `GET /api/v1/receivables/commercial-billing-runs/{billing_run_id}/gmail-delivery-state`, protected by the existing receivables service token and bounded by `limit` plus a PostgreSQL signed-`bigint` `offset`.
2. Reuse the existing commercial Gmail sent-reconciliation service as the durable read owner.  Its projection identifies approvals by the requested run candidate's exact `candidateKey` plus `sourceFingerprint`, so a safely reused approval remains reopenable from a later equivalent review run.
3. Return only persistent evidence: approval/candidate identity, draft-invoice lifecycle fields, immutable PDF metadata (never bytes), Gmail draft intent/identity, and persisted reconciliation observation.  The computed `deliveryState` is a closed read-model vocabulary: `needs_pdf`, `needs_gmail_draft`, `gmail_draft_creating`, `gmail_draft_retryable`, `gmail_draft_recovery_required`, `gmail_draft_not_reconciled`, `gmail_draft_present`, `gmail_draft_missing`, `gmail_sent_confirmed`, or `lifecycle_conflict`.
4. Add real PostgreSQL and full-application tests that prove the endpoint is pagination-safe, does not initialize or call Gmail, and never mistakes a disappeared draft for sent.

### Closure Declaration

- Set: `deliveryState` values returned by this new read projection.
- Membership: CLOSED.  The finite vocabulary is authored in this slice from the closed PDF-artifact state, migration-374 Gmail-draft states, migration-375 reconciliation states, and the established invoice sent lifecycle.
- Source: DERIVED at read time from those retained ATLAS records; no caller supplies a state and no browser/Gmail observation is added to the set.
- Out-of-set behavior: a known but inconsistent lifecycle is `lifecycle_conflict`; malformed/unrecognized durable data fails closed through the existing conflict/unavailable boundary and is never promoted to sent.

### Review Contract

- Acceptance criteria:
  - [ ] An authenticated request for an existing billing run returns a bounded page of only `gmail_pdf` approvals whose candidate key and source fingerprint exactly match that requested run; an approval originally created from an equivalent earlier run is still returned, and an empty beyond-end page retains the collection total.  Its page and total derive from one PostgreSQL statement snapshot, so concurrent approvals cannot make an item and total disagree.  Settled by `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_is_bounded_and_never_calls_gmail`, `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_total_and_page_share_one_snapshot`, and `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_reopens_exact_prior_approval_from_a_later_equivalent_run`.
  - [ ] The response returns no PDF bytes and makes no Gmail gateway/load call; it exposes only retained metadata and persisted state.  Settled by `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_is_bounded_and_never_calls_gmail`.
  - [ ] A `draft_missing` observation remains `gmail_draft_missing`, while only retained `sent_confirmed` evidence with matching ATLAS invoice lifecycle becomes `gmail_sent_confirmed`; inconsistent durable rows are surfaced as `lifecycle_conflict`, never inferred sent.  A ready artifact whose canonical PDF render fingerprint no longer matches a still-draft invoice, or a draft whose independent artifact reference does not match its approval's immutable PDF, is also `lifecycle_conflict`; unfinished or conflicting Gmail drafts expose no impossible sent-mail reconciliation action.  Settled by `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_never_infers_sent_from_a_missing_draft`, `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_marks_a_stale_pdf_as_a_lifecycle_conflict`, `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_projects_cross_linked_gmail_drafts_as_lifecycle_conflicts`, and `tests/test_commercial_billing_gmail_drafts.py::test_real_postgres_delivery_state_omits_sent_reconciliation_until_draft_is_ready`.
  - [ ] Missing runs return the existing 404 error boundary; invalid limits and offsets outside PostgreSQL's signed-`bigint` range are rejected before a database/Gmail action; unavailable delivery schema maps to 503.  Settled by `tests/test_commercial_billing_gmail_drafts.py::test_delivery_state_rejects_postgres_out_of_range_offsets_before_opening_a_database_transaction` and the mounted ASGI route test.
  - [ ] The real mounted `/api/v1/receivables` route rejects missing/wrong service bearer tokens and returns the paged durable projection for valid auth without an actor header, matching existing read-only receivables routes.  Settled by `tests/test_commercial_billing_gmail_drafts.py::test_full_atlas_app_gmail_delivery_state_route_requires_existing_auth_and_returns_no_delivery_side_effect`.
  - [ ] Existing Gmail-draft creation and sent-mail reconciliation semantics remain unchanged.  Settled by the existing `tests/test_commercial_billing_gmail_drafts.py` durable draft/reconciliation suite.
- Reachability proof: `TestClient(create_app())` calls the mounted authenticated endpoint against the real receivables router and observes a bounded state page plus zero calls to a failing Gmail gateway.
- Affected surfaces: `CommercialBillingInvoiceGmailSentReconciliationService`, existing PDF/Gmail durable response projections, receivables route registration/error mapping, EOM commercial Gmail PostgreSQL/ASGI contract tests, and the enrolled invoicing workflow.
- Risk areas: financial-state truthfulness, customer-recipient metadata exposure behind service auth, backward compatibility, pagination, schema/mixed-deployment unavailability, delivery retry/recovery state, and accidental third-party Gmail I/O.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R7, R8, R10, R12, R14.  R4 is N/A: this slice adds no migration or schema mutation.  R9/R11/R13 are N/A: no frontend/config/open-input guard is changed.

### Boundary-change enumeration

The new route is an authenticated read boundary, not an open-input classifier.

- Boundary path/seam: Receivables token guard -> UUID run identifier plus bounded pagination -> durable delivery-state projection.
- Replaced-path behaviors: None.  Existing run, PDF, Gmail-draft, and sent-mail endpoints keep their request/response semantics.
- Guard-relevant fields: `billing_run_id`, `limit`, and `offset`; the route validates query bounds, including the PostgreSQL signed-`bigint` maximum for `offset`, and the service repeats bounds for direct callers.
- Caller x input shape: tracker/Website consumers are deferred; ASGI coverage sends absent/wrong/correct bearer headers and valid/invalid bounded query shapes through the mounted application.

### Deployed-config probing

N/A - this slice changes no environment/config fallback.  It uses the already-deployed receivables API enablement and token digest boundary.

### Files touched

- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py`
- `atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py`
- `atlas_brain/services/commercial_billing_invoice_pdfs.py`
- `plans/PR-EOM-Gmail-Delivery-Workspace.md`
- `tests/test_commercial_billing_gmail_drafts.py`

## Mechanism

`CommercialBillingInvoiceGmailSentReconciliationService.list_delivery_state_for_run()` executes one bounded PostgreSQL statement that proves the requested immutable run exists and derives the matching collection, bounded page, and total from the same statement snapshot.  The materialized matching set is rooted at `commercial_billing_run_candidates`; its join matches `commercial_billing_candidate_approvals` on the candidate's globally exact `(candidate_key, source_fingerprint)` pair rather than only the approval's original `billing_run_id`.  This preserves the existing idempotent reuse rule across equivalent regenerated runs while preventing the default `READ COMMITTED` transaction isolation from splitting a page and total across committed states.  It filters the linked invoice to the canonical EOM business context and explicit `deliveryMethod: gmail_pdf`, and left-joins the one immutable PDF artifact and one Gmail draft record.

The response reuses the existing PDF and Gmail-draft metadata projections and the reconciliation service's persistent evidence projection.  It never selects PDF bytes, never calls `get_gmail_transport`, and never invokes a Gmail `drafts` or `messages` endpoint.  `deliveryState` is derived only from persisted artifact/draft/reconciliation/invoice lifecycle facts.  In particular, the reader uses the PDF service's canonical current-invoice fingerprint parser before offering a draft recovery state, so a still-draft invoice edited after PDF generation becomes `lifecycle_conflict`; it preserves an already verified sent lifecycle instead of reinterpreting its normal status transition as stale.  A draft that is not yet `draft_created` omits the sent-mail reconciliation action because that writer rejects unfinished draft states.  `draft_missing` stays a recovery observation, while `gmail_sent_confirmed` requires the pre-existing persisted sent proof and matching invoice sent lifecycle.  Because the draft table has independent approval and artifact foreign keys, a cross-linked durable draft is still projected for recovery inspection but is marked `lifecycle_conflict` and has no reconciliation action; the reader does not turn that data-integrity conflict into a page-level 409.

The existing receivables router adds one `GET` route with the same service-token dependency as all other financial reads.  It accepts no idempotency key or actor because it creates no operation/audit/financial record.  It returns `items`, `limit`, `offset`, and `total`, so later tracker and Website consumers reuse a single canonical query instead of reimplementing a browser ledger.  No migration is needed: the query consumes the restrictive, additive records created by migrations 370, 372, 373, 374, and 375, all of which the earlier provider slices deploy before this reader.

## Intentional

- The run-rooted query is deliberate: it reopens the exact candidates Juan reviewed and also finds an idempotently reused approval by fingerprint; it is not a global Gmail mailbox/reporting queue.
- Durable state is read from ATLAS only.  Gmail is neither queried nor treated as a financial ledger by this endpoint.
- The projection returns immutable PDF metadata but never PDF bytes, and retains existing Gmail identifiers/recipient metadata only behind the established service-to-service authorization boundary.
- `lifecycle_conflict` is visible rather than repaired automatically; corrections remain explicit audited actions in their existing writers, including a cross-linked Gmail draft/PDF record.

## Deferred

- #2362: eom-timetracker proxy for this canonical read, Website rehydration/recovery UI, and explicit operator-triggered sent-mail reconciliation controls.
- #2362: operating documentation and end-to-end production-safe verification after tracker and Website consumers deploy.
- #2363: background Gmail mailbox sweeps, push/webhook delivery observation, and broader delivery reporting remain deferred; none is required to reopen an operator-selected immutable run.
- #2363 (discovered by #2386): database-level prevention and audit of Gmail drafts cross-linked to an approval's different PDF artifact require a separate production-data assessment and additive migration/rollback plan.  This read-only slice already exposes those durable rows fail-closed as `lifecycle_conflict`.

Parking predicate: park any new delivery writer, Gmail mailbox scanner, report/export, migration, customer-visible copy/PDF/email redesign, or generic multi-channel queue unless it directly prevents this read-only durable recovery path from being truthful.

Parked hardening: #2363 database-level cross-linked Gmail-draft/PDF prevention; it is not required for this reader to report existing evidence truthfully.

## Verification

- Focused durable delivery-state service/route proof: `22 passed` after the beyond-end offset-total, same-snapshot pagination, stale-PDF, cross-linked draft, oversized-offset, and unfinished-draft recovery regression repairs.
- Existing commercial Gmail draft and sent-reconciliation suite: `108 passed`.
- Adjacent commercial approval/run/Gmail regression suite: `181 passed`.
- Local Atlas Invoicing Checks mirror: approval blockers `2 passed`; EOM render, receivables, billing-recipient, payment-receipt, and commercial-candidate suite `240 passed`; manual-Square/invoice/PDF suite `117 passed`; MCP/OAuth surface `43 passed`.
- Changed-path compilation, `ruff`, `git diff --check`, strict guard-class closure lint, and the file maturity lane all passed with zero findings. Plan synchronization and the managed local PR review remain to run immediately before publish.
- All database coverage uses disposable schemas in a local test PostgreSQL instance.  No real customer, Gmail account, production database, or live financial record is contacted.  The Gmail boundary is a failing fake in tests, which proves the read path never invokes it.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/invoicing/receivables.py` | 27 |
| `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py` | 8 |
| `atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py` | 512 |
| `atlas_brain/services/commercial_billing_invoice_pdfs.py` | 32 |
| `plans/PR-EOM-Gmail-Delivery-Workspace.md` | 114 |
| `tests/test_commercial_billing_gmail_drafts.py` | 625 |
| **Total** | **1318** |
