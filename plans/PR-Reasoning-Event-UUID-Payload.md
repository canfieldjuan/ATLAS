# PR-Reasoning-Event-UUID-Payload

## Why this slice exists

Logging a CRM interaction on the live brain prints `Event emission failed
(non-fatal): Object of type UUID is not JSON serializable` from
`atlas_brain/reasoning/producers.py`. The warning is not cosmetic: the
interaction row commits, but its `crm.interaction_logged` event never reaches
`atlas_events`, so the reasoning agent never sees it.

Reproduced against real code on a throwaway Postgres (migrations 035, 256,
346, 366, 071) with reasoning enabled, as it is on the live brain: the real
`DatabaseCRMProvider.log_interaction` committed one interaction and wrote zero
events, and `find_or_create_contact` wrote zero `crm.contact_created` events,
both with the warning above. Isolation held the `crm.interaction_logged`
payload fixed and changed one value per run: a `uuid.UUID` `interaction_id`
drops the event, the same value as `str` lands it, and a `uuid.UUID`
`entity_id` drops it again with `invalid input for query argument $4 ...
(expected str, got UUID)`.

### Problem-derived contract

- Root cause: `emit_event` in `atlas_brain/reasoning/events.py` is the single
  sink for every reasoning event, and it hands producer values straight to
  stdlib `json.dumps` (payload) and to asyncpg's `TEXT` codec (`entity_id`).
  Producers build both from asyncpg rows, which decode `uuid` columns to
  `uuid.UUID`, and neither consumer accepts one. The CRM payload started
  carrying the row UUID in f72ae1553 (`"interaction_id": result.get("id")`
  from `RETURNING contact_interactions.*`); `crm.contact_created` passes the
  row UUID as both `contact_id` and `entity_id`. `emit_if_enabled` swallows
  the exception by design, so every such event has been dropped silently.
- Correct fix: normalize at the sink, once, for every producer. The payload is
  serialized with `json.dumps(payload, default=str)`, the convention the
  sibling sinks already use (`atlas_brain/autonomous/visibility.py`, and
  `atlas_brain/reasoning/consumer.py`, which writes `processing_result` to the
  same `atlas_events` table). `entity_id` is converted with `str()` when
  present and stays `NULL` when absent. Stringifying the id at each CRM call
  site would fix today's two producers and leave the class open for the next
  producer that passes a row value.
- Must not change: the `atlas_events` schema, event types, payload keys,
  `emit_if_enabled`'s never-raise contract, or any producer.

## Scope (this PR)

Ownership lane: reasoning/event-emission
Slice phase: production hardening
Max files: 4

1. Normalize the payload and `entity_id` in `emit_event` so row-decoded values
   persist instead of dropping the event.
2. Prove it through the real CRM producers and the real emit path on a
   disposable Postgres schema, reading back the persisted `atlas_events` row.
3. Enroll that test in the Postgres-backed CRM workflow and trigger it on the
   event sink, the producer helper, and migration 071, so it runs in CI
   instead of skipping.

### Review Contract

- A logged interaction persists its `crm.interaction_logged` event with
  `interaction_id` equal to the string form of the row's UUID and `entity_id`
  equal to the contact id; settled by
  `test_log_interaction_persists_interaction_logged_event`, which first
  asserts the producer really holds a `uuid.UUID`.
- A created contact persists its `crm.contact_created` event with the row UUID
  as a string in both `entity_id` and `payload.contact_id`; settled by
  `test_find_or_create_contact_persists_contact_created_event`.
- The class, not the two CRM examples: a UUID nested in a list and in an
  object, a `datetime`, and a UUID `entity_id` all persist through
  `emit_event` directly, and an absent `entity_id` stays `NULL` rather than
  becoming the string `None`; settled by
  `test_emit_event_normalizes_nested_row_values_and_keeps_null_entity`.
- Reachability proof: the tests drive the published provider methods and the
  real `emit_if_enabled` / `emit_event` path; only the global pool is bound to
  the test schema. Before the fix all three fail (two with "event was
  dropped", one with the `TypeError`); with only the JSON half applied, the
  contact-created and direct tests still fail on the `entity_id` codec.
- Affected surfaces: every producer of `atlas_events`, including
  `emit_if_enabled` callers and `AtlasEventSink`.
- Risk areas: consumers that read `payload` or `entity_id` now receive rows
  that were previously dropped; values that used to raise now arrive as their
  `str()` form (a `datetime` reads `2026-09-25 19:27:46+00:00`).
- Reviewer rules triggered: R1, R2, R5, R6, R12, R13, R14.
  - R5: no schema, key, or event-type change; ids arrive in canonical UUID
    string form, which is what a producer passing `str(id)` already sends.
  - R6: the swallow in `emit_if_enabled` hid this for months; making the
    failure louder is parked below rather than changed here.
  - R12: the test is enrolled in
    `.github/workflows/atlas_eom_lead_pipeline_checks.yml`, which
    provisions Postgres and already owns
    `atlas_brain/services/crm_provider.py`.
  - R13: fixed at the one sink every producer shares, with a nested-value
    test beyond the two reported producers.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/reasoning/events.py`
- `plans/PR-Reasoning-Event-UUID-Payload.md`
- `tests/test_reasoning_event_emission_integration.py`

## Mechanism

`emit_event` imports `json` at module level and passes
`json.dumps(payload, default=str)` and
`None if entity_id is None else str(entity_id)` as the fourth and fifth query
arguments. Nothing else in the insert, the returned id, or the NOTIFY trigger
changes.

## Intentional

- `default=str` rather than a type-by-type encoder: it matches the two sibling
  writers to the same kind of JSONB column, and an event that carries an
  unexpected object in its `str()` form is more useful than one that is
  dropped.
- No backfill of events dropped since f72ae1553. The reasoning agent reacts to
  live events; replaying months of stale CRM activity would trigger actions on
  it now.
- The CRM call sites keep passing row values; the sink owns the conversion.

## Deferred

Parking predicate: behavior not needed to stop events being dropped is parked.

Parked hardening: `emit_if_enabled` logs every emission failure at WARNING and
returns `None`, which is how this failure stayed invisible; surfacing repeated
emission failures (a counter or an alert) is recorded here, not changed here.

## Verification

`tests/test_reasoning_event_emission_integration.py` is marked `integration`,
skips unless `ATLAS_MIGRATION_TEST_DATABASE_URL` is set, and creates and drops
its own schema. Run locally against a throwaway Postgres 16 container: 3
passed with the fix; all 3 failed with the `origin/main` sink swapped back in;
2 failed with only the JSON half applied. The adjacent reasoning, call, MCP,
tracing, and task-trigger suites fail the same 10 tests with and without the
fix (semantic-router and MCP mock cases unrelated to this sink). CI runs the
test in `atlas-eom-lead-pipeline-checks` against its Postgres service; the
Unit Gate deselects it. Rollback is a revert; no schema or data changes.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 11 |
| `atlas_brain/reasoning/events.py` | 8 |
| `plans/PR-Reasoning-Event-UUID-Payload.md` | 148 |
| `tests/test_reasoning_event_emission_integration.py` | 154 |
| **Total** | **321** |
