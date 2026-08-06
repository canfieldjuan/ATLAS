# PR-EOM-Lost-Replay-Generation

## Why this slice exists

Issue #2295 split a real but deferred lost/reopen idempotency bug out of the
#2289 / #2291 lost-lead slice. PR #2296 made reopen choose the latest
`lead_lost` row by database-owned `lifecycle_sequence`, but it did not validate
that an idempotent replay key still describes the latest lost/reopen lifecycle
generation.

Ground truth on current `origin/main`:

- `mark_eom_lead_lost` finds a replay row by `(contact_id, event_type,
  operation_key)` and, if the contact is currently `lead/lost`, returns that
  row's `from_stage` and reason as an idempotent success without comparing the
  replay row to newer `lead_lost` / `lead_reopened` events
  (`atlas_brain/services/crm_provider.py:3491-3550`).
- `reopen_eom_lead` finds a replay row by `(contact_id, event_type,
  operation_key)` and, if the contact is currently active at the replay row's
  `to_stage`, returns an idempotent success without comparing the replay row to
  newer `lead_lost` / `lead_reopened` events
  (`atlas_brain/services/crm_provider.py:3700-3755`).
- The lifecycle table now has a database-owned `lifecycle_sequence` through
  migration 363, and reopen already depends on it for latest-loss ordering
  (`atlas_brain/services/crm_provider.py:3771-3811`).

This slice is over the 400-LOC target because the production guard and the CI
proof are not independently shippable. The replay fix changed the real
lost/reopen lifecycle surface and therefore had to ship with real-Postgres
coverage proving stale lost replay, stale reopen replay, immediate same-key
retry, legacy unsequenced replay ownership, and the held-out transition-shape
matrix requested by review in the same PR. After the current Unit Gate exercised
`tests/test_b2b_reviews_import.py`, that test module failed only because its
imports crossed eager package-level API/autonomous side effects unrelated to
the B2B assertions. Leaving that isolation for a separate PR would keep this PR
red and unmergeable. The B2B edits are therefore limited to test import
isolation needed to satisfy the gate for this exact head; no B2B runtime code or
behavior moves.

### Problem-derived contract

- Root cause: lost/reopen idempotency keys are scoped only to a matching
  lifecycle row plus the contact's current coarse state. They are not scoped to
  the lifecycle generation that row created. After
  `lost(K1) -> reopen(R1) -> lost(K2)`, a delayed retry of `K1` can return K1's
  stale `from_stage` / `reason_code` even though K2 owns the current loss. After
  `reopen(R1) -> lost(K2) -> reopen(R2)` back to the same stage, a delayed retry
  of `R1` can similarly claim R1 is still the successful restore even though R2
  owns the current active generation.
- Correct fix must touch/change: add one generation check in
  `DatabaseCRMProvider` that uses the database-owned lifecycle order to decide
  whether a replay row is still the latest lost/reopen disposition for that
  contact; apply it to both `mark_eom_lead_lost` replay and `reopen_eom_lead`
  replay before returning idempotent success; return a 409 conflict when the
  replay row was superseded; add real-Postgres integration tests for stale lost
  replay, stale reopen replay, immediate same-key replay still succeeding, and
  legacy unsequenced replay rows succeeding only for an unambiguous current
  generation.
- Must not change: no public route shape, request body, auth, reason-code
  taxonomy, email/onboarding draft behavior, estimate/first-clean booking,
  customer/site handoff, `won` admission, calendar side effects, migration
  schema, provider insert/create semantics, tracker/website code, customer-
  visible copy, or unrelated CI behavior outside the B2B test import-isolation
  repair needed for this head's Unit Gate.

## Scope (this PR)

Ownership lane: eom/funnel-go-live
Slice phase: production hardening

1. Reject stale idempotency replays for EOM `lead_lost` and `lead_reopened`
   operation keys when a later lost/reopen lifecycle event exists for the same
   contact.
2. Preserve normal same-key immediate retries: if no later lost/reopen
   lifecycle event supersedes the replay row, the existing idempotent 200 shape
   stays intact.
3. Add integration coverage through the real provider methods and real
   lifecycle table ordering.

### Review Contract

- Acceptance criteria:
  - [ ] `mark_eom_lead_lost` replay fetches the replay row's lifecycle order and
        rejects the replay before returning 200 when any later
        `lead_lost`/`lead_reopened` disposition exists for the same contact;
        settled by `atlas_brain/services/crm_provider.py` and
        `tests/test_eom_lead_conversion_integration.py`.
  - [ ] `reopen_eom_lead` replay fetches the replay row's lifecycle order and
        rejects the replay before returning 200 when any later
        `lead_lost`/`lead_reopened` disposition exists for the same contact;
        settled by `atlas_brain/services/crm_provider.py` and
        `tests/test_eom_lead_conversion_integration.py`.
  - [ ] Immediate same-key retries with no later disposition still return the
        existing idempotent success shapes for lost and reopen; settled by
        `test_mark_lead_lost_records_reason_is_idempotent_and_reopens`.
  - [ ] The generation comparison uses `lifecycle_sequence` for sequenced rows,
        validates replay row transition shape before returning idempotent
        success, admits a legacy unsequenced `lead_reopened` replay only when it
        has exactly its required `lead_lost` predecessor, the reopen row is
        `lost -> <restorable stage>`, that predecessor's `from_stage` matches the
        replay row's `to_stage`, and no additional lost/reopen disposition row
        exists; keeps legacy `lead_lost` strict because any other disposition
        makes lost ownership ambiguous; settled by the helper query and
        real-Postgres integration fixtures.
  - [ ] `lead_lost` and `lead_reopened` are the closed event set for replay
        ownership because these are the only lifecycle events that directly
        write the lost/active disposition toggled by the provider methods in
        this slice; out-of-set lifecycle events do not supersede lost/reopen
        replay ownership.
- Reachability proof: the real `DatabaseCRMProvider.mark_eom_lead_lost` and
  `DatabaseCRMProvider.reopen_eom_lead` methods are exercised against a
  disposable PostgreSQL schema. Observable effects are 409 conflicts for
  superseded replay keys and unchanged contact/lifecycle state after rejection.
- Affected surfaces: `DatabaseCRMProvider` lost/reopen replay branches, the
  EOM lead conversion integration tests, B2B review import test isolation for
  the Unit Gate, this plan, and PR-body evidence.
- Risk areas: stale idempotency success, lifecycle order correctness across
  mixed legacy/sequenced rows, idempotent retry compatibility, and avoiding new
  customer/site/email/calendar side effects.
- Reviewer rules triggered: R1, R2, R4, R5, R8, R10, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: EOM lost/reopen idempotent replay admission in
  `DatabaseCRMProvider`.
- Replaced-path behaviors: stale same-key replay after a later lost/reopen
  disposition changes from idempotent 200 to 409 conflict; immediate retry of
  the latest disposition remains idempotent 200.
- Guard-relevant fields: `contact_id`, `event_type`, `operation_key`,
  `lifecycle_sequence`, and `id` on `eom_lead_lifecycle_events`, plus the
  current locked contact row.
- Caller x input shape: private/office callers that retry the same lost or
  reopen operation key after the lead has moved through a newer lifecycle
  generation.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no env/config/default value change.
- Explicit value probe: operation-key replay tests provide explicit K1/R1 stale
  keys and K2/R2 current keys.
- Absent value probe: N/A - the private routes already require an
  idempotency/operation key; this PR does not change request validation.
- Default-session/default-context probe: N/A - no session/default-context
  behavior changes.
- Side-effect ordering: the stale replay guard must run after the contact row is
  locked and before returning an idempotent success; it must not update contacts
  or append lifecycle rows on rejected stale replay.

### Files touched

- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Lost-Replay-Generation.md`
- `tests/test_b2b_reviews_import.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

On replay, load the replay lifecycle row with its database-owned
`lifecycle_sequence` and immutable row `id`. Before returning idempotent success,
ask whether any `lead_lost` or `lead_reopened` row for the same contact
supersedes that replay row. Sequenced replay rows compare by higher
`lifecycle_sequence`. Every replay row must also match the provider-produced
transition shape before it can return idempotent success: `lead_lost` is
`<restorable stage> -> lost`, and `lead_reopened` is
`lost -> <restorable stage>`. Legacy unsequenced `lead_lost` replay rows are
accepted only when no other lost/reopen disposition row exists for the contact,
because no committed ordering key can prove which unsequenced row owns a later
lifecycle generation. Legacy unsequenced `lead_reopened` replay rows are
accepted for the one real provider-produced current-generation shape: exactly
this reopen row plus one preceding unsequenced `lead_lost` row that moved the
lead to `lost` and whose `from_stage` matches the reopen replay row's `to_stage`.
Any additional lost/reopen disposition, malformed replay row, or legacy
loss/reopen stage mismatch that the provider could not have produced for that
operation remains ambiguous and returns 409. If no later or ambiguous disposition
exists, keep the existing replay response.

This is a response-truth guard, not a new transition. It only prevents an old
operation key from claiming ownership of a newer lead disposition.

The disposition-event set is closed for this ownership check:

- `lead_lost` is the only event that moves an EOM lead into the lost
  disposition through `DatabaseCRMProvider.mark_eom_lead_lost`.
- `lead_reopened` is the only event that restores a lost EOM lead to an active
  restorable stage through `DatabaseCRMProvider.reopen_eom_lead`.
- Other lifecycle events (`lead_created`, estimate booking/completion events,
  handoff/customer conversion events, and future unrelated annotations) may
  describe the funnel, but they do not own the lost-vs-active replay truth for
  these two operation keys. If a future event type directly changes this same
  disposition, that is a new boundary change and must update this closed set and
  its tests in the same PR.

Execution model:

- Both replay branches execute inside the provider transaction created by
  `_transaction_connection`, acquire deterministic transaction-scoped advisory
  locks for the contact and operation key, and then lock the contact row with
  `FOR UPDATE` before returning replay success.
- New lost/reopen writes in these provider methods use the same contact lock and
  append the lifecycle row in the same transaction as the contact-stage update;
  rollback cancels both the row append and the contact mutation, so no committed
  partial disposition can be observed by a later replay.
- The `lifecycle_sequence` comparison is a committed-row ordering invariant for
  admitted provider writes after migration 363. A row with a greater sequence is
  a later committed lost/reopen disposition for the same contact.
- Legacy unsequenced rows do not carry that ordering invariant. Lost replay
  succeeds only when the lost row is shaped as `<restorable stage> -> lost` and
  is the sole lost/reopen disposition row for the contact. Reopen replay succeeds
  for the provider-produced legacy pair of one lost predecessor plus the replay
  reopen row when the reopen row is shaped as `lost -> <restorable stage>` and
  the predecessor's `from_stage` matches the replay row's `to_stage`. Any
  additional lost/reopen row, malformed replay row, or mismatched loss/reopen
  pair makes the replay owner ambiguous and returns 409.
- Direct database writers or legacy applications that bypass the provider
  transaction/advisory/contact-lock path are excluded assumptions for admitted
  runtime behavior; their existing records are treated by the legacy
  fail-closed branch when they lack sequence evidence.

## Intentional

- No new migration: migration 363 already supplies the database-owned
  `lifecycle_sequence` required for new rows, and #2296 already made the enabled
  funnel readiness guard require that column.
- No change to `won` loss admission: losing a won lead still belongs to #2292
  because it must coordinate onboarding draft revoke and first-clean/calendar
  teardown.
- No client/API shape change: callers still receive 200 for current-generation
  same-key retries and 409 for conflicts.

## Deferred

- #2292: admit losing `won` only with atomic onboarding-draft revoke and
  first-clean/calendar teardown.
- Any manual reconciliation of multiple unsequenced legacy loss rows remains
  covered by #2296's fail-closed chronology-reconciliation behavior.

Parking predicate: this slice parks lifecycle work that changes a different
side-effect owner, such as draft/email/calendar teardown, customer/site
handoff, or live legacy data repair. It does not park stale lost/reopen replay
truthfulness for the current provider methods.

Parked hardening: none against the predicate above.

## Verification

- Passed:
  `python -m py_compile atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py`.
- Passed:
  `python -m pytest tests/test_eom_lead_conversion.py::test_disposition_replay_supersession_uses_lifecycle_sequence tests/test_eom_lead_conversion.py::test_disposition_replay_supersession_checks_legacy_rows_by_other_id tests/test_eom_lead_conversion.py::test_disposition_replay_supersession_admits_legacy_reopen_pair -q` — 3 passed.
- Skipped locally:
  `python -m pytest tests/test_eom_lead_conversion_integration.py::test_legacy_disposition_replay_rejects_held_out_transition_shapes -q -rs` — skipped because `ATLAS_MIGRATION_TEST_DATABASE_URL` is not configured in this shell; CI supplies the real-Postgres lane.
- Passed:
  `python -m ruff check atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py`.
- Passed:
  `python -m pytest tests/test_b2b_reviews_import.py -m "not integration and not e2e" --continue-on-collection-errors -rfE --tb=no -q -p no:cacheprovider` — 6 passed.
- Passed:
  `<python3.11 temp env>/bin/python -m pytest tests/test_b2b_reviews_import.py -m "not integration and not e2e" --continue-on-collection-errors -rfE --tb=short -q -p no:cacheprovider` — 6 passed.
- Passed:
  `python scripts/sync_pr_plan.py plans/PR-EOM-Lost-Replay-Generation.md 9629361 --check`.
- Passed: `git diff --check`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/crm_provider.py` | 128 |
| `plans/PR-EOM-Lost-Replay-Generation.md` | 279 |
| `tests/test_b2b_reviews_import.py` | 188 |
| `tests/test_eom_lead_conversion.py` | 145 |
| `tests/test_eom_lead_conversion_integration.py` | 619 |
| **Total** | **1359** |
