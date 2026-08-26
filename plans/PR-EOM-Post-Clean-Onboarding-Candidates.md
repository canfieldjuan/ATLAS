# PR-EOM-Post-Clean-Onboarding-Candidates

## Why this slice exists

Issue #2156 requires post-clean onboarding to begin only after Atlas receives
durable evidence that a residential customer's first cleaning completed. PR
#2493 made that evidence authoritative but intentionally stops before creating
the distinct post-clean candidate, leaving no Atlas-owned queue for the next
approval/UI slices to consume. Reusing `eom_onboarding_email_drafts` would join
the first-clean-booked lead welcome lifecycle to a post-clean customer/card
lifecycle and would expose an existing customer-send action before terms and
Stripe are ready.

### Problem-derived contract

- Root cause: the canonical completion transaction persists only its immutable
  receipt; no distinct, idempotent post-clean candidate is derived from that
  authoritative trigger.
- Correct fix must touch/change: add an Atlas-owned candidate relation; make a
  valid new completion commit its receipt and candidate in one PostgreSQL
  transaction; heal an old receipt on exact replay; add an authenticated,
  bounded read projection and capability declarations; prove the real POST and
  GET entrypoints, migration/backfill, retry, concurrency, and fail-closed
  eligibility behavior with focused PostgreSQL tests; enroll the migration in
  the EOM pipeline workflow.
- Must not change: the first-clean-booked welcome draft table/copy/approve-send
  path, public onboarding tokens, terms text, Stripe, email delivery, Tracker,
  Website, calendar/booking inference, or the immutable receipt identity and
  existing response fields.

## Scope (this PR)

Ownership lane: eom/onboarding-post-clean-candidate
Slice phase: Vertical slice

Max files: 6

Diff-budget override: The required plan artifact plus the indivisible migration,
atomic provider write/read boundary, registered API seam, and real PostgreSQL
retry/concurrency/catalog proof must deploy together; splitting them would ship
either unconsumable state or an advertised route without at-most-once evidence.

1. Create/reuse one non-sendable `pending` post-clean candidate for every
   admitted residential first-clean completion.
2. Expose the tenant-scoped pending queue through one read-only, paginated
   funnel route with current-contact blockers.
3. Prove the trigger-to-row-to-list path without customer delivery side effects.

### Review Contract

- Acceptance criteria:
  - The completion service inserts one receipt and one linked candidate in the
    existing PostgreSQL transaction; an unchanged retry returns both original
    IDs, settled by focused real-Postgres service tests.
  - The execution model is one database transaction fenced by the existing
    contact/operation/service advisory locks plus candidate receipt/contact
    uniqueness; every admitted interleaving yields at most one candidate, while
    transaction failure commits neither new row, settled by the concurrent and
    schema-failure integration tests (R8/AGENTS 3k.4).
  - `POST .../first-clean-completions` followed by authenticated
    `GET /eom-funnel/post-clean-onboarding-candidates` exposes the persisted
    candidate and current blocker through real ASGI routes.
  - Booking/calendar evidence alone still has no candidate writer, and the
    welcome-draft, token, email, and Stripe modules remain outside the diff.
  - Existing receipt fields/status codes remain unchanged and the two new
    fields are additive, settled by route contract tests.
  - The migration is additive, conditionally backfills existing receipts, and
    rollback retains candidate evidence while old code ignores the table.
- Reachability proof: authenticated completion POST -> committed receipt and
  candidate rows -> authenticated candidate-list GET returns that candidate.
- Affected surfaces: EOM funnel API, first-clean completion service, additive
  PostgreSQL migration, focused EOM pipeline tests/workflow.
- Risk areas: migration ordering, orphan/duplicate candidates, retries and
  concurrent delivery, tenant leakage, response compatibility, premature send.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R7, R8, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: first-clean completion admission and candidate-list
  projection.
- Replaced-path behaviors: valid new completion changes from receipt-only to
  atomic receipt+candidate; exact replay changes from receipt-only to returning
  or healing the same candidate; invalid/ineligible completion remains a
  zero-write rejection; list is new and read-only.
- Guard-relevant fields: authoritative contact context/type/status/customer
  type and email; receipt/contact/handoff identities; operation fingerprint;
  list limit/cursor.
- Caller x input shape: Tracker valid completion, unchanged retry, changed-key
  conflict, concurrent same/different operations, invalid/ineligible request,
  authenticated bounded list, and malformed list bounds/cursor.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no environment/config fallback changes.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: authenticated EOM context is enforced
  by the existing funnel dependency and candidate query.
- Side-effect ordering: every candidate admission check and row write occurs
  before the completion transaction commits; the list performs no writes.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/eom_first_clean_completion.py`
- `atlas_brain/storage/migrations/395_eom_post_clean_onboarding_candidates.sql`
- `plans/PR-EOM-Post-Clean-Onboarding-Candidates.md`
- `tests/test_eom_first_clean_completion.py`

## Mechanism

The next additive migration creates a runtime-owned candidate table keyed
uniquely by immutable completion receipt and canonical contact. It has no
foreign key to the guard-owned receipt/handoff relations because the runtime
role intentionally lacks `REFERENCES`; the completion service inserts and
validates the association inside the same locked transaction. A conditional
`INSERT ... SELECT` backfills deployed receipts when migration 394 is present;
exact replay heals any receipt created before the candidate schema existed.

The queue joins the candidate to current EOM contact and immutable receipt
state. It reports `inactive_customer`, `not_residential`, or `no_email` as a
current blocker and never advances state or sends. Pagination reuses the
existing opaque cursor envelope.

Closure declarations:

- Candidate status/blocker vocabulary is CLOSED and ENUMERATED here from the
  migration/API contract; PostgreSQL/Pydantic reject out-of-set stored or wire
  values, and current-contact ambiguity defaults to a blocker rather than
  sendability.
- Capability membership is CLOSED and ENUMERATED at the canonical
  `_CAPABILITY_ROUTES` map; unregistered method/path pairs are omitted, which
  keeps older consumers disabled by default.
- The caller/input inventory is CLOSED and DERIVED from the one registered POST,
  one registered GET, their Pydantic models, and the direct service caller;
  out-of-contract inputs are rejected by the existing validation/auth boundary.

## Intentional

- Candidate creation is atomic with a new receipt. Candidate-schema failure
  returns 503 and rolls that Atlas transaction back; Tracker's already-merged
  durable pending report remains the execution evidence and exact retry source.
- The candidate is deliberately non-sendable. This is detection-quality proof,
  not a second onboarding email implementation.
- Candidate rows remain on code rollback; old code ignores the additive table,
  preserving audit/recovery evidence.

## Deferred

- Tracker proxy/status UI and manager recovery action: #2156 next consumer slice.
- Signed approval/customer delivery, public versioned terms, and Stripe card
  vault: #2156 slices C/D after their customer-facing contracts are ready.

Parking predicate: adjacent queue polish, additional blocker taxonomy, and
recovery/reporting beyond the one trigger/list proof are parked unless they
break idempotency, tenant isolation, migration safety, or the real route.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_eom_first_clean_completion.py -q` against an
  isolated PostgreSQL 16 runtime/DBA pair: 88 passed.
- Targeted `compileall`, Ruff, and `git diff --check`: passed.
- Boundary probe: the complete migrated catalog passes readiness; dropping the
  candidate receipt-identity UNIQUE constraint fails readiness closed.
- Pending before push: plan sync and the mechanical local PR review.
- Full Unit Gate is GitHub-only per repository policy.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 2 |
| `atlas_brain/eom_api/funnel.py` | 97 |
| `atlas_brain/services/eom_first_clean_completion.py` | 205 |
| `atlas_brain/storage/migrations/395_eom_post_clean_onboarding_candidates.sql` | 40 |
| `plans/PR-EOM-Post-Clean-Onboarding-Candidates.md` | 188 |
| `tests/test_eom_first_clean_completion.py` | 160 |
| **Total** | **692** |
