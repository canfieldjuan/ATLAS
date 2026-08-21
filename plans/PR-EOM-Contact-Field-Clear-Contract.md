# PR-EOM-Contact-Field-Clear-Contract

## Why this slice exists

Website #254 (Slice 5 of the CRM arc, child of website #105) locks a tri-state
field-clearing contract for contact editing: an operator must be able to CLEAR
an obsolete email/phone (explicit JSON null), distinct from OMITTING an
untouched field (key absent) and from REPLACING it (non-empty value), with
audit evidence that distinguishes all three. This is the Juan-directed
follow-up to the shipped Slice 4 edit boundary (tracker #229/#231, website
#249): editing that cannot remove an obsolete value is incomplete editing.

The Slice 5 investigation (recorded on website #254) proved the Atlas
mutation ALREADY implements the wire mechanics: `_operator_contact_fields`
gates on `model_fields_set` (present-null kept as an explicit clear, absent
key dropped as an omit), the domain normalizer stores `None` under the key,
`_update_existing` writes SQL NULL, and `request_fingerprint` hashes
present-null distinctly from key-absent. Two gaps remain, and they are this
PR:

1. **The audit cannot tell a clear from a re-point.** `_write_lifecycle_event`
   records `changed_fields` + `previous_values` (old values only), so
   `email -> null` and `email -> new@addr` produce identical event shapes.
2. **No capability names the clear semantics.** `contact.operator_mutation`
   proves the route exists, not that its null semantics + audit are supported;
   an older Atlas serves the same route. Downstream (tracker, website) must
   gate clearing on a versioned capability or they cannot deploy fail-closed.

Diff-budget override: 592 added lines are 21 production LOC (one
capability-map entry + one additive audit metadata key) plus the mandated
plan doc and the tri-state proof tests. The slice is genuinely indivisible:
the #254 contract requires the tri-state PROVEN at the write authority before
any downstream leg ships, so the capability (the advertisement), the audit
key (the semantics being advertised), and the tests (the proof) must land
atomically -- a capability advertised without its proven audit contract is
exactly the over-advertising failure the manifest exists to prevent, and
tests split into a follow-up would leave a live, advertised semantics
unproven in the window between merges.

### Problem-derived contract

- Root cause: the operator-mutation audit under-describes intent (old values
  only, no cleared-vs-changed distinction), and the capability manifest has no
  name for the null-clear semantics, so no downstream caller can safely enable
  clearing against a deployed Atlas.
- Correct fix must touch/change:
  - `_write_lifecycle_event` gains an additive, PII-free `cleared_fields`
    metadata key (field NAMES whose post-update value is NULL) on the update
    path, making OMITTED (absent from `changed_fields`) / CLEARED (listed in
    `cleared_fields`) / CHANGED (`changed_fields` minus `cleared_fields`)
    distinguishable from the event alone;
  - `_CAPABILITY_ROUTES` gains `contact.field_clear` mapped to the SAME
    `POST /eom-funnel/operator-contacts` signature, versioning the route's
    semantics: the dict entry ships only with builds that carry the audited
    clear contract;
  - tests pinning the tri-state at both tiers (HTTP boundary present-null vs
    absent; DB-backed clear/omit/change with audit assertions), idempotent
    clear replay, the omit-vs-clear fingerprint conflict, the clearable-set
    edges (full_name/customer_type refuse null), and the shared-route
    capability pairing.
- Must not change:
  - no request/response schema change (`EOMOperatorContactRequest` and
    `_operator_contact_item` already carry the contract);
  - no change to `_operator_contact_fields`, `_normalize_fields`,
    `_blank_to_none`, or `_update_existing` -- the clear mechanics are live,
    documented behavior (see `_normalize_customer_type`'s docstring);
  - no migration (contacts.email/phone already nullable; event metadata is
    jsonb);
  - no tracker/website work in this PR (they are the next two legs, gated on
    this deploy);
  - no widening of the clearable set: full_name and customer_type keep
    refusing blank/null with 422; address-family fields keep their existing
    generic-boundary behavior and are out of the #254 capability contract.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. Advertise `contact.field_clear` as a semantics-versioning capability on the
   existing operator-mutation route.
2. Record `cleared_fields` (names only) in the `contact_updated` lifecycle
   event so cleared/changed/omitted are distinguishable without duplicating
   PII.
3. Prove the locked tri-state contract with route-tier, manifest, and
   DB-integration tests.

Max files: 6

### Review Contract

- Acceptance criteria:
  - [ ] `contact.field_clear` maps to `("POST", "/eom-funnel/operator-contacts")`
        and is advertised iff `contact.operator_mutation` is -- settled by
        `tests/test_eom_funnel_capability_manifest.py::test_field_clear_capability_versions_the_operator_mutation_semantics`.
  - [ ] A request carrying a present-null optional field reaches the command
        with `fields[<field>] is None` while an absent sibling key never
        appears -- proven for BOTH advertised fields (email-cleared/phone-absent
        and phone-cleared/email-absent) by
        `tests/test_eom_lead_conversion.py::test_operator_contact_route_keeps_present_null_distinct_from_absent`.
  - [ ] A clear persists as SQL NULL, the mutation result echoes the null, an
        omitted sibling field is preserved, and the event carries
        `changed_fields == ["email"]`, `cleared_fields == ["email"]`,
        `previous_values == {"email": <old>}` -- settled by
        `tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_clear_vs_omit_vs_change_tri_state`.
  - [ ] A re-point (value -> new value) lists the field in `changed_fields`
        with `cleared_fields == []` -- settled by the same test's CHANGED
        control.
  - [ ] The phone sibling clears end-to-end through its own normalizer and
        column: row NULL, result echo null, `changed_fields == ["phone"]`,
        `cleared_fields == ["phone"]`, `previous_values == {"phone": <old>}`
        -- settled by the phone-clear leg of the same tri-state test.
  - [ ] A retried clear with the same Idempotency-Key replays (one lifecycle
        row), and the same key with the null dropped (omit) conflicts 409 --
        settled by `tests/test_eom_lead_conversion_integration.py::test_operator_contact_clear_replay_is_idempotent_and_omit_conflicts`.
  - [ ] `full_name: null` and `customer_type: null` are refused 422, and a
        repeat clear of an already-null field is a no-op update with a uniform
        empty tri-state event -- settled by
        `tests/test_eom_lead_conversion_integration.py::test_operator_contact_clear_is_scoped_to_nullable_optional_fields`.
  - [ ] The event carries field NAMES only in the new key (no `new_values`
        map) -- settled by the `"new_values" not in metadata` assertion in the
        tri-state test plus the `cleared_fields` implementation at its
        `crm_provider.py` insertion point.
  - [ ] Creates are unaffected: the create path omits
        `changed_fields`, `previous_values`, AND the new `cleared_fields` --
        settled by the direct `"cleared_fields" not in metadata` assertion
        added to
        `tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_creates_replays_and_records_actor`.
- Reachability proof: real ASGI POST to `/eom-funnel/operator-contacts` with a
  present-null field (route-tier test above) plus DB-backed provider tests
  asserting the persisted contact row NULL and the lifecycle event metadata.
- Affected surfaces: `atlas_brain/eom_api/funnel.py` (capability map only),
  `atlas_brain/services/crm_provider.py` (`_write_lifecycle_event` only), EOM
  funnel manifest/route/integration tests.
- Risk areas: capability-manifest drift (two names sharing one route),
  lifecycle-event shape compatibility for existing consumers, idempotency
  fingerprint stability, clearable-set boundary (full_name/customer_type).
- Reviewer rules triggered: R2, R3, R5, R8, R12.
- Guard/set closure declaration: the clearable field set is CLOSED by the
  existing domain boundary, not by new code: `EOM_OPERATOR_CONTACT_FIELDS`
  enumerates operator fields; within it, `full_name` (422 on blank/null via
  `_normalize_text_field`) and `customer_type` (422 via
  `_normalize_customer_type`) refuse clearing, and the remaining optional
  fields map blank/null to SQL NULL. This PR adds no admission logic; the
  negative tests above pin the closed edges the #254 contract names.

### Boundary-change enumeration

N/A - no boundary change. No guard, validator, normalizer, resolver, or
admission boundary is modified; the diff adds one capability-map entry and one
additive audit metadata key, plus tests pinning existing boundary behavior.

### Deployed-config probing

N/A - no guard/config boundary change. No env/config read is added or altered;
`ATLAS_EOM_FUNNEL_API_ENABLED` gating is untouched (existing tests already
exercise enabled/disabled).

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Contact-Field-Clear-Contract.md`
- `tests/test_eom_funnel_capability_manifest.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

`contact.field_clear` is a second name for the already-registered
`POST /eom-funnel/operator-contacts` signature in `_CAPABILITY_ROUTES`.
`served_capabilities()` derives advertisement from registered routes, so the
name is served exactly when the mutation route is -- but only on builds whose
dict carries the entry, which is exactly the builds that ship this audited
clear contract. Callers therefore gate clearing on the NAME (+ exact route,
tracker-side strict proof), never on the route's existence.

`_write_lifecycle_event` already computes `changed = sorted(key for key in
command.fields if previous.get(key) != contact.get(key))` on the update path.
The new `cleared_fields = sorted(key for key in changed if contact.get(key) is
None)` derives the cleared subset from the post-update row already in scope --
no new queries, no new writes, one additive jsonb key. Field names only:
the overwritten value already lives in `previous_values` (the sole surviving
copy; there is no contacts history table), and the new value lives on the
contact row itself, so a `new_values` map would duplicate PII into the audit
stream for zero recovery benefit.

Tri-state, end to end: OMITTED = key absent from the request ->
`model_fields_set` drops it -> not in `command.fields` -> not in
`changed_fields`. CLEARED = present null -> normalizer stores `None` ->
`_update_existing` writes SQL NULL -> in `changed_fields` AND in
`cleared_fields`. CHANGED = present value -> normalized/replaced -> in
`changed_fields`, not in `cleared_fields`. Idempotency: the request
fingerprint serializes `fields` with present-nulls (`"email":null`), so a
replayed clear matches its receipt and a same-key omit conflicts 409.

## Intentional

- Two capability names deliberately share one route signature. The manifest's
  derivation stays honest (never advertise what this build cannot serve), and
  the pinned pairing test makes the sharing explicit so a future "dedupe"
  cannot silently drop the semantics version.
- No `new_values` in the event: names-only `cleared_fields` is the
  PII-conscious minimum that completes the tri-state (see Mechanism).
- `cleared_fields` is emitted on every update event (empty list when nothing
  cleared) rather than only-when-non-empty: consumers distinguish "new-format
  event, nothing cleared" from "pre-slice event" by key presence.
- Atlas boundary behavior for address/city/state/zip/notes (blank/null clears
  at the generic operator boundary) is pre-existing, documented behavior and
  is left untouched; the #254 capability contract scopes portal-reachable
  clearing to email/phone at the tracker (`extra="forbid"` admits no other
  field), not by narrowing this boundary under its existing callers.
- The route-tier present-null test asserts `status in (200, 201)` because the
  `_CRM` fake decides created-vs-updated; the assertion that matters there is
  the command's field presence shape, and the DB tests pin real status
  semantics.

## Deferred

- Tracker leg (next PR, gated on this deploy): strict
  `contact.field_clear` proof -> `contactFieldClearAvailable`, present-null
  forwarding on the edit path only, blank-on-edit 422, response projection
  widened to email/phone.
- Website leg (after the tracker is live): dirty-field tri-state body
  assembly, #252 refusal retained as the no-capability fallback, EN/ES copy.
- #247 ledger items are unchanged by this PR (notably item 7's
  Atlas-emitted `editable` flag -- separate slice).

Parking predicate: this slice parks only the downstream caller legs that the
deploy order (Atlas first) requires to land separately, plus previously
ledgered #247 hardening. It does not park correctness, security, idempotency,
or audit defects in the Atlas clear contract itself.

Parked hardening: none.

## Verification

- `python3 -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/crm_provider.py tests/test_eom_funnel_capability_manifest.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `python3 -m pytest tests/test_eom_funnel_capability_manifest.py -q`
  -- 9 passed.
- `python3 -m pytest tests/test_eom_lead_conversion.py::test_operator_contact_route_keeps_present_null_distinct_from_absent -q`
  -- 1 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python3 -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_clear_vs_omit_vs_change_tri_state tests/test_eom_lead_conversion_integration.py::test_operator_contact_clear_replay_is_idempotent_and_omit_conflicts tests/test_eom_lead_conversion_integration.py::test_operator_contact_clear_is_scoped_to_nullable_optional_fields -q`
  -- 3 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python3 -m pytest tests/test_eom_lead_conversion_integration.py -q -k "operator_contact"`
  -- 24 passed, 87 deselected.
- `python3 -m pytest tests/test_eom_lead_conversion.py -q`
  -- 225 passed, 1 warning.
- Pending before push: `python scripts/check_guard_class_closure.py --base origin/main --strict`,
  `python scripts/sync_pr_plan.py plans/PR-EOM-Contact-Field-Clear-Contract.md --check`,
  `git diff --check`, `bash scripts/local_pr_review.sh` / `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 8 |
| `atlas_brain/services/crm_provider.py` | 10 |
| `plans/PR-EOM-Contact-Field-Clear-Contract.md` | 264 |
| `tests/test_eom_funnel_capability_manifest.py` | 25 |
| `tests/test_eom_lead_conversion.py` | 32 |
| `tests/test_eom_lead_conversion_integration.py` | 315 |
| **Total** | **654** |
