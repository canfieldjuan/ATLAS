# PR-Resolution-Audit-Zendesk-Writeback

## Why this slice exists

Juan asked to resume the AI Content Ops lane and make macro writeback work for
articles created by the Resolution Audit, not only content created in the AI
Content Station.

Root cause: the Zendesk macro writeback path is coupled to the generic
`faq_markdown` generated-asset route. Resolution Audit generation already uses
the same FAQ Markdown producer internally and, when DB-backed, stores the
source FAQ draft id in the paid report artifact's `faq_result.saved_ids`, but
the report route replaces the execute response with a locked Snapshot and has
no paid-report entrypoint that delegates that saved FAQ draft to the existing
macro publish service. The Zendesk adapter is not missing; the reachability
bridge from paid Resolution Audit report to the existing writeback service is.

This change fixes that root for the backend path by adding a paid-report
publish endpoint that resolves the saved FAQ draft id from the persisted
Resolution Audit artifact and publishes through the existing
`FAQMacroWritebackPublishService`.

This slice is over the 400 LOC soft cap (1383 total) because the vertical
slice ships the route, host wiring, and both-sides tests in one reviewable
unit: ~148 LOC of production code, with the remainder being this plan doc
and tests proving the paid/locked/missing/cross-tenant/stale-id/
missing-credentials/unwired-provider paths fail closed.

## Scope (this PR)

Ownership lane: content-ops/resolution-audit-zendesk-writeback
Slice phase: Vertical slice

1. Add a backend route for publishing the saved FAQ draft behind a paid
   Resolution Audit report to the configured macro provider.
2. Wire Atlas' Content Ops router to build the route with the existing Zendesk
   macro publish provider and Postgres FAQ/macro-attempt repositories.
3. Add route-level proof that a paid report artifact with a saved FAQ draft id
   reaches the existing publish service and updates observable state.
4. Add negative proof for locked reports, missing artifacts, and artifacts that
   do not carry a saved FAQ draft id.

### Review Contract

- Acceptance criteria:
  - [ ] `POST /content-ops/deflection-reports/{request_id}/publish-macros`
        requires the report to exist and be paid before publishing.
  - [ ] The route extracts the saved FAQ draft id from
        `artifact.faq_result.saved_ids` and delegates to
        `FAQMacroWritebackPublishService` rather than reimplementing macro
        eligibility, Zendesk publishing, idempotency, or attempt-history logic.
  - [ ] A successful call returns the request id, account id, FAQ draft id,
        publish summary, and observable successful publish counts.
  - [ ] Locked reports, malformed/missing paid artifacts, missing saved FAQ ids,
        missing provider wiring, and stale saved FAQ ids fail closed with no
        provider call.
  - [ ] Existing `faq_markdown` Generated Asset Review macro publish route
        remains unchanged.
  - [ ] A producer-real generated draft (`status='draft'`, as
        `save_drafts` persists) publishes through the paid route via the
        explicit `approve_draft` promotion; non-`draft` statuses
        (rejected/archived/published/expired) are never revived.
- Reachability proof: route test exercises the real control-surface endpoint
  function, `InMemoryDeflectionReportArtifactStore`, a real
  `FAQMacroWritebackPublishService`, and an in-memory FAQ repository; only the
  external Zendesk publish provider is mocked.
- Affected surfaces: API, auth/paid gate, DB-backed service wiring,
  third-party Zendesk boundary, attempt history.
- Risk areas: tenant isolation, paid/unpaid access control, duplicate
  external writes, stale artifact references, backward compatibility.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R14.

### Files touched

- `atlas_brain/_content_ops_macro_writeback.py`
- `atlas_brain/api/__init__.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/faq_macro_writeback.py`
- `extracted_content_pipeline/faq_macro_writeback_publish.py`
- `extracted_content_pipeline/ticket_faq_ports.py`
- `extracted_content_pipeline/ticket_faq_postgres.py`
- `plans/PR-Resolution-Audit-Zendesk-Writeback.md`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_atlas_content_ops_macro_writeback.py`
- `tests/test_content_ops_faq_macro_writeback_flow.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_ticket_faq_macro_writeback_publish.py`
- `tests/test_extracted_ticket_faq_postgres.py`
- `tests/test_faq_macro_writeback_live_zendesk_smoke.py`
- `tests/test_seed_faq_macro_writeback_live_smoke_draft.py`

## Mechanism

The control-surface router gets an optional `faq_macro_publish_service_provider`
factory. Hosts that pass it unlock a new paid-report action:

```text
POST /deflection-reports/{request_id}/publish-macros
```

The route:

1. resolves tenant scope and fetches the persisted deflection report record;
2. rejects missing or unpaid reports before reading the saved FAQ id;
3. extracts the first UUID-like id from `record.artifact.faq_result.saved_ids`;
4. resolves the publish service factory;
5. calls `publish_faq_draft(saved_faq_id, scope=tenant, approve_draft=True)`
   -- the tenant's explicit publish action on their paid report is the
   approval for the generated draft. The opt-in only promotes the exact
   `'draft'` status (what `save_drafts` persists for real Resolution Audit
   runs); rejected/archived/published drafts are never revived, the promotion is
   compare-and-set on the stored status (`update_status(...,
   expected_status='draft')` adds `AND status = $4` in Postgres) so a
   concurrent review decision landing after `get_draft` wins and the publish
   fails closed, and a refused promotion falls through to the existing
   `draft_not_approved` skip. The
   default (`approve_draft=False`) keeps the AI Content Station and scheduled
   publish paths byte-for-byte unchanged;
6. returns the existing `FAQMacroPublishSummary` payload plus `request_id`,
   `account_id`, and `faq_id`.

Atlas wires the provider using the existing building blocks:

- `PostgresTicketFAQRepository(get_db_pool())`
- `build_content_ops_macro_publish_provider(pool_provider=get_db_pool)`
- `PostgresFAQMacroPublishAttemptRepository(get_db_pool())`

That keeps Zendesk credentials, idempotency mappings, pending-reconcile
behavior, attempt history, and draft status updates centralized in the existing
macro-writeback path.

## Intentional

- No frontend button in this slice. The first proof is backend reachability and
  authorization; UI affordance can land once the route shape is reviewed.
- No direct Zendesk call in CI. The Zendesk provider is the true external
  boundary; tests use a provider fake and assert service-level observable
  status/summary instead of network calls.
- No `faq_markdown` catalog exposure for the hosted Resolution Audit flow. The
  report remains the customer-facing product surface; this route uses the saved
  internal FAQ id as implementation identity.
- The route uses the same authenticated + rate-limited dependency set
  (`deflection_report_public_dependencies`) as the other paid-report routes
  (`/artifact`, `/report-model`, `/delta`), because publishing acts only on the
  tenant's own paid report, own saved FAQ draft, and own stored Zendesk
  credentials, all of which fail closed on scope. Operator-only gating
  (`deflection_report_paid_dependencies`, used by the billing `/paid` mutation)
  was considered and not chosen; it can be tightened later without changing the
  route shape.

## Deferred

- Frontend action on the paid Resolution Audit page to invoke the route and
  render publish history.
- Live Zendesk smoke using a paid test report artifact and explicit operator
  confirmation flags.
- Optional endpoint to list macro publish attempts by report request id instead
  of by underlying FAQ draft id.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_content_deflection_submit.py -k "publish_macros" -q` - 8 passed.
- `python -m pytest tests/test_atlas_content_ops_macro_writeback.py -q` - 17 passed (3 new host-factory tests).
- `python -m pytest tests/test_extracted_ticket_faq_macro_writeback_publish.py tests/test_atlas_content_ops_generated_assets_api.py -q` - 31 passed.
- `python -m py_compile atlas_brain/_content_ops_macro_writeback.py atlas_brain/api/__init__.py extracted_content_pipeline/api/control_surfaces.py` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed.
- `python scripts/audit_plan_doc.py plans/PR-Resolution-Audit-Zendesk-Writeback.md` - passed.
- Local PR review runs via `scripts/push_pr.sh` (body-aware pre-push hook) at push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_macro_writeback.py` | 39 |
| `atlas_brain/api/__init__.py` | 4 |
| `extracted_content_pipeline/api/control_surfaces.py` | 112 |
| `extracted_content_pipeline/faq_macro_writeback.py` | 1 |
| `extracted_content_pipeline/faq_macro_writeback_publish.py` | 113 |
| `extracted_content_pipeline/ticket_faq_ports.py` | 9 |
| `extracted_content_pipeline/ticket_faq_postgres.py` | 11 |
| `plans/PR-Resolution-Audit-Zendesk-Writeback.md` | 192 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 1 |
| `tests/test_atlas_content_ops_macro_writeback.py` | 70 |
| `tests/test_content_ops_faq_macro_writeback_flow.py` | 1 |
| `tests/test_extracted_content_asset_api.py` | 10 |
| `tests/test_extracted_content_deflection_submit.py` | 405 |
| `tests/test_extracted_ticket_faq_macro_writeback_publish.py` | 317 |
| `tests/test_extracted_ticket_faq_postgres.py` | 96 |
| `tests/test_faq_macro_writeback_live_zendesk_smoke.py` | 1 |
| `tests/test_seed_faq_macro_writeback_live_smoke_draft.py` | 1 |
| **Total** | **1383** |
