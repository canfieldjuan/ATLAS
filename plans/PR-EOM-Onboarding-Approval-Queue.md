# PR-EOM-Onboarding-Approval-Queue

## Why this slice exists

Arc #2275 names A3 as the next Atlas funnel slice after A2 merged:
onboarding-email drafts are now enqueued when the office books the first clean,
but they still have no real approval/send operation. That leaves Juan with no
code-owned way to approve the queued welcome email and no durable state proving
that exactly one send happened after approval.

Diff-budget override: this A3 slice is intentionally over the 400 LOC soft cap
because the smallest safe vertical proof has to ship the private route, approval
service state machine, CRM-provider SQL, email transport trace header, handoff
race fence, index predicate parity, render-profile route reachability, and
behavioral tests together. Splitting the endpoint away from the
claim/send/confirm proof would expose a callable approval surface
without the single-send or lead-stage evidence the slice exists to provide.

### Problem-derived contract

- Root cause: A2 created the source-of-truth draft queue, but stopped before
  the approval surface. The existing `eom_onboarding_email_drafts` table can
  represent `pending -> sending -> sent`, but no Atlas endpoint/service/provider
  currently lists those drafts for the office, atomically claims one draft, sends
  it through the existing email transport, or confirms delivery after the
  transport accepts. Without that code path, the funnel cannot move from `won`
  to an approved/sent onboarding-email state without a manual or duplicate-prone
  side channel.
- Correct fix must touch/change: Add a narrow Atlas EOM funnel approval API over
  the existing private `/api/v1/eom-funnel` surface; add a service function that
  claims one pending/unblocked draft with the migration-360 predicate, sends
  outside the claim transaction via the existing email provider with draft-id
  provider-log evidence, and confirms `sent` only after transport acceptance;
  add CRM-provider methods for listing drafts, claiming drafts, and confirming
  sent while moving the lead from `won` to `onboarding_sent` with lifecycle
  evidence; fence customer handoff while an onboarding draft is `sending`; add
  tests for list shape, blocked/blank-recipient refusal, duplicate-approval
  single-send behavior, send-failure non-delivery, handoff fencing, and endpoint
  reachability.
- Must not change: Do not create Customer/Site records, tracker handoffs,
  tokenized onboarding links/pages, Stripe/card-on-file behavior, Calendar
  booking behavior, estimate/first-clean booking admission rules, email template
  wording, lead intake, Render config, or receivables/invoicing behavior. Do not
  auto-send at first-clean booking time; Juan approval remains the only send
  trigger in this slice.

## Scope (this PR)

Ownership lane: eom/funnel-go-live
Slice phase: Vertical slice

1. Add office-only EOM funnel endpoints for listing onboarding-email drafts and
   approving/sending one draft.
2. Add the provider/service state machine that enforces exactly-one claim,
   sends through the existing email provider, and confirms sent only after
   transport acceptance.
3. Add behavioral tests over provider/service/API seams plus the existing A2
   enqueue integration path.

### Review Contract

- Acceptance criteria:
  - [ ] `GET /api/v1/eom-funnel/onboarding-email-drafts` returns the office
        projection for pending/sending/blocked/sent onboarding drafts without
        exposing credentials or private runtime URLs; settled by API model tests
        and provider list tests.
  - [ ] `POST /api/v1/eom-funnel/onboarding-email-drafts/{draft_id}/approve-and-send`
        rejects blocked/no-recipient/non-pending drafts before calling email
        transport; settled by service/provider tests that assert zero sends.
  - [ ] The approval execution model is claim-first: provider claim uses a
        row-locked sendable CTE whose predicate requires `status = 'pending'`,
        `blocker IS NULL`, and a nonblank `recipient_email`, so every admitted
        interleaving has at most one caller receive sendable content; settled by
        provider SQL inspection and duplicate/blank-recipient tests.
  - [ ] The service calls `email_provider.send(...)` outside the claim update and
        passes the draft id in transport-supported message headers, then
        confirms `sent` only after the send call returns; settled by service
        tests for success and raised send failure.
  - [ ] Confirming sent moves the lead from `won` to `onboarding_sent` and
        appends lifecycle evidence; settled by provider/integration tests.
  - [ ] Customer handoff rejects a lead while any onboarding email draft for the
        contact is `sending`; settled by provider handoff tests.
  - [ ] Existing estimate and first-clean booking behavior stays unchanged;
        settled by the existing EOM lead conversion tests in local verification.
- Reachability proof: Real FastAPI routes under `/api/v1/eom-funnel` exercised
  through the route handler/test client with the CRM provider and email provider
  faked at the adapter boundary, asserting response state and persisted provider
  side effects.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`, CRM provider EOM draft
  methods, new EOM onboarding approval service, existing email provider adapter
  and transport header passthrough, EOM lead conversion tests.
- Risk areas: duplicate send, send/confirm ordering, blocked recipient handling,
  lead-stage regression, API/auth contract, backcompat for A1/A2 bookings.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: Private EOM funnel API gains onboarding draft list and
  approve/send endpoints; provider claim predicate admits only pending,
  unblocked, nonblank-recipient draft rows while row-locking the draft/contact
  decision; customer handoff rejects contacts with an in-flight `sending`
  onboarding draft.
- Replaced-path behaviors: N/A - A3 adds the first approval/send path; it does
  not replace an existing endpoint.
- Guard-relevant fields: draft `status`, `blocker`, `recipient_email`,
  `contact_id`, actor id/name, and draft UUID path parameter.
- Caller x input shape: Tracker/office service bearer plus actor header calls
  list or approves a single draft UUID; no browser-direct token or public
  endpoint is introduced.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: No new config. Existing EOM funnel
  `api_enabled` and service token gates continue to protect the route.
- Explicit value probe: Existing funnel auth tests cover the enabled/token path;
  this slice adds route-level dependency tests through the same private router
  seam.
- Absent value probe: Existing funnel disabled/token-missing startup behavior is
  unchanged.
- Default-session/default-context probe: N/A - no default tenant or fallback
  context is added.
- Side-effect ordering: Claim draft first; send email second with draft-id
  transport evidence; confirm sent and advance lead stage only after transport
  acceptance. Customer handoff is fenced while the draft is `sending`.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/email_provider.py`
- `atlas_brain/services/eom_onboarding_email.py`
- `atlas_brain/storage/migrations/361_eom_lead_review_queue_onboarding_sent_stage.sql`
- `atlas_brain/tools/email.py`
- `atlas_brain/tools/gmail.py`
- `plans/PR-EOM-Onboarding-Approval-Queue.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migrations_runner.py`

## Mechanism

The service asks the CRM provider to claim a draft. Claiming is a row-locked
sendable CTE plus `UPDATE` whose predicate requires `status='pending'`,
`blocker IS NULL`, a nonblank recipient, an active EOM lead, and
`lead_stage='won'`, so only one approver receives sendable
subject/body/recipient content. The service then calls the existing email
provider with the snapshot subject/body and an
`X-Atlas-EOM-Onboarding-Draft-ID` header for provider-log reconciliation. Only
after that call returns does it ask the provider to confirm the draft as `sent`,
set `sent_at`, move the contact from `won` to `onboarding_sent`, and append an
EOM lifecycle event. A send exception bubbles to the API and leaves the row in
`sending`, which is the migration-360 recovery state for operator
reconciliation. The customer handoff path rejects a contact while any
onboarding draft for that contact is still `sending`, so a handoff cannot
convert the lead out from under a confirming send.

## Intentional

- No template wording changes: this slice wires the queue/approval mechanics,
  not the customer-facing email copy.
- No automatic retry from `sending` back to `pending`: migration 360 defines a
  stuck `sending` row as operator reconciliation evidence, because retrying an
  uncertain send could duplicate customer email.
- No tokenized onboarding link yet: A4 owns the customer-facing onboarding page
  and token generation.

## Deferred

- A4: tokenized onboarding link/page and the `onboarding_sent -> onboarded`
  transition.
- Tracker/Website UI slices: surface the pending/sending/sent draft queue to the
  office; this PR provides the private Atlas API.
- Sending recovery action: confirm/revoke a stuck `sending` row after transport
  log reconciliation if production use shows it is needed.

Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/eom_onboarding_email.py atlas_brain/services/crm_provider.py atlas_brain/services/email_provider.py atlas_brain/tools/email.py atlas_brain/tools/gmail.py tests/test_eom_lead_conversion.py tests/test_migrations_runner.py tests/test_eom_render_profile.py`
  - Passed.
- `python -m pytest tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_migrations_runner.py::test_eom_lead_review_queue_onboarding_sent_stage_matches_provider_filter tests/test_eom_render_profile.py::test_eom_profile_import_does_not_load_full_api_package -q`
  - Passed: 134 passed, 36 skipped, 1 warning.
  - Skipped: DB-gated EOM lead conversion integration cases that require a
    local asyncpg migration test database not configured in this worktree.
- `python -m ruff check atlas_brain/eom_api/funnel.py atlas_brain/services/eom_onboarding_email.py atlas_brain/services/crm_provider.py atlas_brain/services/email_provider.py atlas_brain/tools/email.py atlas_brain/tools/gmail.py tests/test_eom_lead_conversion.py tests/test_migrations_runner.py tests/test_eom_render_profile.py`
  - Passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 118 |
| `atlas_brain/services/crm_provider.py` | 233 |
| `atlas_brain/services/email_provider.py` | 4 |
| `atlas_brain/services/eom_onboarding_email.py` | 98 |
| `atlas_brain/storage/migrations/361_eom_lead_review_queue_onboarding_sent_stage.sql` | 23 |
| `atlas_brain/tools/email.py` | 5 |
| `atlas_brain/tools/gmail.py` | 11 |
| `plans/PR-EOM-Onboarding-Approval-Queue.md` | 211 |
| `tests/test_eom_lead_conversion.py` | 490 |
| `tests/test_eom_render_profile.py` | 6 |
| `tests/test_migrations_runner.py` | 38 |
| **Total** | **1237** |
