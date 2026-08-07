# PR-D2-Email-Backfill-Boundary

## Why this slice exists

`email_backfill` writes EOM contacts from sent mail while bypassing the EOM
boundary: it calls `crm.find_or_create_contact` directly with the tenant as a
string literal. Website #125 (D2), the lowest-risk child of Slice 0D.

### Correction to the plan this came from

The prior intent was "build 0B's domain function and migrate D2 as its first
caller". D2 does not need 0B. #125 says route it through ingress, and ingress
already owns EOM identity matching. What ingress cannot do is create a
**customer**: it hardcodes `contact_type="lead"` / `lead_stage="new"`, correct
for a web enquiry and wrong for a backfill of people who already bought.

So this is a sibling that shares ingress's matching, not the operator-mutation
tier. 0B's authenticated endpoint, idempotency ledger and receipts stay deferred
until 0C has a remote caller for them -- building an endpoint with no consumer
is what produced the unarbitrable review loops earlier in this arc.

### Problem-derived contract

- Root cause: an EOM-tenant write outside the EOM boundary, so it inherits none
  of the boundary's rules. Concretely it defaulted to `preserve_existing=False`,
  which merges non-null fields into an existing record -- and this task derives
  display names from email local parts, so a backfill could overwrite a curated
  customer name with a guess. Dormant only because the task is disabled.
- Correct fix must touch: the shared resolver (lifted so matching is
  single-sourced), a sibling resolve-or-create that stamps the tenant and
  preserves existing records, and the `email_backfill` call site.
- Must not change: `resolve_or_create_eom_inbound_lead` behaviour, the provider,
  the scheduler's disabled seed, `contacts.source` values, or D1/D3/D4/D5.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: production hardening

1. Lift `_resolve_readonly` out of `resolve_or_create_eom_inbound_lead` to
   module-level `resolve_eom_contact_readonly`. Pure move; ingress keeps using
   it so EOM identity matching stays single-sourced.
2. Add `resolve_or_create_eom_contact` -- resolve scoped-then-legacy, return an
   existing contact untouched, else create with the tenant from the boundary,
   `preserve_existing=True`, and a caller-specified `contact_type`.
3. Migrate `email_backfill` to it; the tenant literal disappears from the caller.

### Files touched

- `atlas_brain/autonomous/tasks/email_backfill.py`
- `atlas_brain/services/eom_lead_ingress.py`
- `plans/PR-D2-Email-Backfill-Boundary.md`
- `tests/test_eom_lead_ingress.py`
- `tests/test_tenant_stamping.py`

### Review Contract

1. `resolve_or_create_eom_inbound_lead` behaves identically -- the live
   website-form intake path runs through it. The extraction is a move, proven by
   `tests/test_eom_lead_ingress.py` and `tests/test_leads_intake.py` passing
   unchanged.
2. An existing contact is returned untouched: derived names never overwrite
   curated ones.
3. The tenant comes from `EOM_BUSINESS_CONTEXT_ID`, never a caller literal.
4. A legacy null-tenant contact is claimed, not duplicated, and a correctly
   tenanted row wins over one merely unclassified.
5. The scheduler seed and the live disabled state are untouched. This migrates
   the code path; it does not turn the task on.

Affected surfaces: `eom_lead_ingress` (one extraction, one new sibling) and one
call site in `email_backfill`. No provider change -- the two `INSERT INTO
contacts` sites remain the only persistence points, which 0A's guard enforces.

Risk areas: the extraction touches a revenue-carrying path. Probed by running
the ingress and intake suites unchanged, and by a mutation that removes the
legacy lookup -- which fails an existing intake test, proving the shared
resolver is genuinely shared rather than shadowed.

- Reviewer rules triggered: R1, R2, R5, R6, R8, R10, R14.

R6/R8 are the path triggers for `atlas_brain/autonomous/tasks/**` (scheduled
jobs, retry/durability). Dispositioned: this changes which function the task
calls, not when it runs, how it retries, or its enabled state. The scheduler
seed and the live `enabled=false` row are untouched, and the task remains
disabled, so there is no runtime or durability behaviour to regress. The write
it performs becomes strictly less destructive -- `preserve_existing=True`
replaces a merge that could overwrite curated fields.

**boundary-probe:** existing-untouched, create-path stamping, legacy claim, and
scoped-over-legacy precedence are each asserted separately.

**Mutation-probe (run, not asserted), three:** `preserve_existing=False` fails 1;
a literal tenant instead of the constant fails 1; removing the legacy lookup
fails 2, one of them an existing intake test.

## Mechanism

One closure lifted to module scope, one sibling function beside it, one call site
rewired.

## Intentional

- **Sibling, not a parameter on ingress.** Widening the website-form intake path
  to serve a dormant backfill puts revenue at risk for no gain.
- **`preserve_existing=True` is not optional.** Callers of this function derive
  their fields; derived data must not overwrite curated data.
- **`email_backfill` moved from `WRITER_SITES` to a new `EOM_BOUNDARY_DELEGATES`
  registry rather than being deleted from the test.** Deleting it would let a
  direct provider call quietly reappear. The new assertion additionally forbids
  the caller passing `business_context_id` at all.

## Deferred

- 0B proper: authenticated endpoint, idempotency receipts, lifecycle actor
  attribution. Deferred until 0C needs a remote caller.
- D1, D3, D4, D5 (website #124, #126, #127, #128).

Parking predicate: this slice parks everything except D2's write path and the
resolver it shares with ingress.

Parked hardening: none.

## Merge-order dependency with ATLAS #2313: RESOLVED

#2313 (EOM operator mutation contract, opened in a parallel session) renames
`_normalised_phone` to `normalise_eom_phone_digits` in this same file.
#2313 landed first; this branch is rebased onto it and the call site now uses
`normalise_eom_phone_digits`. Zero stale references remain. The edits are otherwise disjoint -- #2313
touches the phone normalizer, this touches the resolver closure.

The two are complementary rather than competing: #2313's
`mutate_eom_operator_contact` requires `operation_key`, `actor_id` and
`actor_name`, so it is an operator boundary. D2 is an unattended backfill with
no operator and no idempotency key; routing it through the operator tier would
mean inventing a fake actor. It belongs on the ingress sibling.

## Verification

```
$ python -m pytest tests/test_leads_intake.py tests/test_eom_lead_ingress.py \
    tests/test_tenant_stamping.py tests/test_eom_lead_pipeline_integration.py -q
81 passed, 15 skipped

$ python scripts/check_contact_write_boundary.py --baseline ... --inventory-baseline ...
(exit 0 -- no new contact write site)

$ bash scripts/pre_push_audit.sh
all checks passed
```

Live state confirmed unchanged: `email_backfill enabled=false` in
`scheduled_tasks`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/email_backfill.py` | 11 |
| `atlas_brain/services/eom_lead_ingress.py` | 88 |
| `plans/PR-D2-Email-Backfill-Boundary.md` | 163 |
| `tests/test_eom_lead_ingress.py` | 118 |
| `tests/test_tenant_stamping.py` | 29 |
| **Total** | **409** |
