# PR-EOM-Live-Calendar-Import

## Why this slice exists

Issue #2151 Phase 3 / issue #2156 slice A (operator-requested: the customer
onboarding epic's calendar watcher needs a populated, tenant-clean "is this an
existing EOM customer?" predicate). Phase 2 (PR #2155, merged) stamped every
contact writer and added the NULL-row provenance backfill — but the customer
master itself has still never durably entered Postgres. Verified at HEAD
67b0b16de (2026-07-23):

- The only customer importer, `scripts/import_calendar_contacts.py`, reads ICS
  snapshot files frozen on 2026-02-22 under a machine-local path — five
  months stale, and re-export is manual.
- The Google token store already holds `auth/calendar` scope, and
  `GoogleCalendarProvider.list_events` supports arbitrary calendar ids with
  pagination and `singleEvents` expansion — live reads need zero new auth.
- `create_contact` dedupes by phone then email only: a record with neither
  channel (common for calendar events) would insert a duplicate row on every
  re-run. Calendar-derived import needs an address-level pre-resolution to
  be idempotent.

### Problem-derived contract

- Root cause: the customer master lives in Google Calendar; no code path
  reads it live, so the CRM's customer population decays from the moment any
  snapshot import runs. The watcher slice (#2156 B) cannot exist on top of a
  decaying base.
- Correct fix must touch/change: an additive live-import script reading the
  three EOM booking calendars (One-Time, Residential, Commercial — the
  Estimates calendar is a lead surface owned by the #2153 intake and is
  excluded) via the existing `GoogleCalendarProvider`; calendar ids via
  environment only (public repo); reuse of the ICS importer's extraction and
  dedupe core so both import paths classify names/addresses identically;
  tenant stamp `effingham_maids` + `source='calendar_import'` + segment
  tags on every contact; address-level pre-resolution for phone-less/
  email-less records; a stable `source_ref` dedupe anchor (migration 256)
  on the logged interaction so re-runs never accumulate duplicates;
  lead-to-customer upgrade via `create_contact` merge semantics, never the
  reverse.
- Must NOT change: `scripts/import_calendar_contacts.py` (the ICS path stays
  runnable as-is); `atlas_brain/services/crm_provider.py` semantics (only
  existing params passed); `atlas_brain/api/leads.py`; schema (no
  migrations); `scripts/backfill_business_context.py`; B2B flows; money
  paths (billing, invoices, receivables, customer_services); read-path
  default filters (deferred to the watcher slice, which passes scope
  explicitly).

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. New `scripts/import_eom_customers_live.py`: live Google Calendar import
   of the three EOM booking calendars into tenant-stamped customer
   contacts, idempotent across re-runs, `--dry-run` supported, window
   configurable (`--months-back 24`, `--months-forward 12`).
2. New `tests/test_eom_live_calendar_import.py`: 12 behavioural tests (pure
   parse/mapping/config plus stubbed DB-edge routing).
3. This plan doc.

### Review Contract

- Acceptance criteria:
  1. The calendar list contains exactly commercial, residential, one_time —
     no Estimates surface (asserted).
  2. Missing calendar-id env vars fail fast, naming every missing variable
     (asserted); no calendar id string appears anywhere in the diff.
  3. Every imported contact carries `business_context_id='effingham_maids'`,
     `source='calendar_import'`, its segment tag, and
     `contact_type='customer'` (asserted).
  4. The latest-dated event decides the cancellation state — a newer
     CANCELLED booking re-flags a record inactive, a newer active booking
     reactivates it, in either input order (asserted; Codex round 3, R1);
     calendar events with `STATUS=cancelled` are skipped entirely
     (asserted).
  5. Records with a phone or email resolve via tenant-scoped channel
     search first (phone priority, extension-stripped) and only consult
     the address net on a channel miss; records with neither channel
     resolve by address and update rather than duplicate (asserted).
  6. Import interactions carry a stable `source_ref` anchor so re-runs
     dedupe (asserted).
  7. A matched contact receives the calendar-computed `status` through
     the diffed update path — the provider merge allowlist excludes it
     (asserted; Codex R1/R8; zero-write case covered by criterion 16).
  8. Single-calendar mode requires only the selected calendar's env var
     (asserted; Codex R1).
  9. The address resolver excludes archived rows exactly like the
     provider's search guard (asserted; Codex R4/R8).
  10. The script exits non-zero when any record errored (asserted;
      Codex R6).
  11. Calendar ids resolve through typed `ATLAS_TOOLS_EOM_CALENDAR_*`
      settings (which read .env/.env.local) with process-env override
      (asserted; Codex round 2, R11).
  12. Cancellation recency holds ACROSS calendars: the inherited
      cross-calendar merger's any-active-clears is corrected by a
      recency map keyed on phone/address, asserted in both input orders
      plus reactivation (Codex round 4, R1).
  13. Channel resolution never miss-creates: tenant phone -> email ->
      address net, in that order, BEFORE any create — a previously
      address-only row that gains a phone is enriched, not duplicated
      (asserted; Codex round 4, R8).
  14. Imported segment tags UNION with existing tags; recorded provenance
      like the intake's website/estimate_request is never erased
      (asserted; Codex round 4, R1).
  15. A NULL-context legacy row is claimed via a script-side CAS UPDATE
      whose guards live INSIDE the statement (context NULL-or-same,
      unarchived always, and `source='calendar_import'` additionally for
      weak address-only identity) — a raced archive/source-correction makes
      the CAS return no row and the record is created fresh, with the raced
      row never tenant-stamped; channel-matched (phone/email) legacy rows
      claim without the source predicate so a legacy web lead reconciles
      instead of duplicating (all asserted; Codex rounds 2-3, 7-8, R4/R8).
  16. A repeat import of an unchanged calendar performs ZERO writes on
      matched contacts -- updates are diffed field-by-field and skipped
      when identical, counted 'unchanged' (asserted; Codex round 5, P1).
  17. Trailing phone extensions are stripped before last-10 channel
      matching (asserted; Codex round 5).
  18. Matched updates never carry `source`; a returning website lead keeps
      `source='web'` -- only newly created contacts stamp
      `calendar_import` (asserted; Codex round 5).
  19. Cross-calendar dedupe is TRANSITIVE over phone/address keys
      (union-find), and same-day cancellation ordering is resolved at
      full event-timestamp granularity within a calendar (asserted;
      Codex round 5).
  20. The dedupe phone key uses extension-stripped digits, so ext-bearing
      and plain forms of one number are one customer (asserted; Codex
      round 6).
  21. Neither `source` nor `tags` ride a find-or-create that may
      race-merge: creates are stripped of both and stamped post-create on
      truly-new rows only; a race-merged result gets the full matched-path
      reconciliation (asserted; Codex rounds 6-7).
  22. (superseded by 26, round 9: matched rows never carry `source` --
      schema-default 'manual' rows are legitimate provenance; the
      failed-post-create-stamp window is an accepted provenance-label
      trade-off reported as the run error.)
  23. (superseded by 15's in-CAS guards, round 8)
  24. The address-net SELECTs return `source`, so an address-only matched
      web contact keeps its provenance instead of being treated as
      provider-default manual (asserted; Codex round 8, P1).
  25. Uppercase phone extensions (X42 / EXT 42) are normalized before
      extraction so the base number is never corrupted (asserted against
      the real extractor behavior; Codex round 8).
  26. Matched updates NEVER carry `source` (asserted incl. a 'manual' row),
      and every matched/stamp write goes through an UPDATE whose archived
      guard lives inside the statement -- a contact archived mid-run is
      skipped, never resurrected -- including its interaction timeline,
      via a distinct 'skipped' outcome AND a pre-log status re-check that
      covers archives landing after the contact write (asserted; Codex
      rounds 9-11).
  28. The guarded UPDATE carries a tenant predicate (NULL-or-EOM) and
      matched payloads never re-send the stamp, so a concurrently
      reassigned row is never stolen back across tenants (asserted;
      Codex round 11).
  29a. The pre-log re-check also confirms the row still belongs to
      `effingham_maids`; the current (latest) address leads the fallback
      candidate order; and the matched identity (address / phone last-10 /
      email) rides inside the claim CAS so a row corrected mid-race no
      longer matches the predicate (all asserted; Codex round 12).
  34. Same-calendar equal-start ties also resolve to cancelled (asserted;
      Codex round 18); the final-update identity-predicate TOCTOU is
      tracked in #2160 per the declared disposition rule.
  33. A failed post-create provenance stamp is a run FAILURE (non-zero
      exit), and equal-timestamp cancellation ties resolve to cancelled
      deterministically regardless of input order (asserted; Codex
      round 17).
  32. Email is a dedupe identity key alongside phone/address; merged
      groups propagate `latest_event_dt`; and logged interactions carry the
      real event time in `occurred_at` (all asserted; Codex round 16).
  31. Cross-calendar channel merges follow the same event recency as
      in-calendar, tracked PER FIELD (a newer email never makes an older
      phone look fresh); provider "Untitled" placeholders are rejected;
      interaction anchors are timestamp-scoped; and a failed post-create
      stamp is reported, not silently dropped (all asserted; Codex rounds
      14-15).
  30. The claim-CAS phone predicate is contains-match (provider parity, so
      ext-bearing stored contacts satisfy the CAS they matched by); same-day
      cross-calendar ties resolve by full timestamp; and the latest event's
      phone/email replaces stale channels within an address, in any input
      order (all asserted; Codex round 13).
  29. Interaction anchors are date-scoped: re-runs with the same latest
      booking dedupe, while each new latest booking advances the CRM
      timeline instead of colliding with the old anchor forever
      (asserted; Codex round 11, R1/R6).
  27. A merged group's surviving address is the latest-dated one and ALL
      group addresses ride into the fallback lookup, so an existing
      address-only contact at any of them is enriched, never duplicated
      (asserted; Codex round 9, P1).
- Reachability proof: operator script, invoked directly. The full CLI
  entrypoint (arg parsing, id resolution, `GoogleCalendarProvider.
  list_events` pagination, cross-calendar dedupe, exit code) was exercised
  by a real `--dry-run` invocation against the three production booking
  calendars: 7,163 events -> 93 unique customers, exit 0 (recorded in
  Verification; Codex round 3, R2). The extraction helpers additionally
  execute on every event via `parse_events`, which the tests drive with
  representative summaries/locations/descriptions.
- Reviewer rules triggered: R1, R2, R4, R6, R8, R11, R12, R14.
  - R1: behavior matches #2151 Phase 3 and the #2156 watcher contract;
    cancellation recency semantics asserted.
  - R2: every acceptance criterion has a named test; the entrypoint is
    evidenced by the recorded live dry-run.
  - R4: data safety — tenant stamp, archived guard, provenance-restricted
    CAS claim, no guessing.
  - R6: the operator script exits non-zero on partial failure; per-record
    errors are reported, never swallowed.
  - R8: idempotency — provider dedupe, address net, interaction anchor,
    second-run-zero-changes acceptance.
  - R11: dependencies & config — three optional typed
    `ATLAS_TOOLS_EOM_CALENDAR_*` fields on `ToolsConfig`, default None;
    absent config changes no behavior (reviewer-directed, round 2).
  - R12: env/config — calendar ids are deployment config read via
    pydantic-settings from `.env`/`.env.local` with process-env override;
    no ids or secrets enter the repo (grep-proven in Verification).
  - R14: this contract is the reviewer's checklist; every criterion
    states its assertion.

### Files touched

- `atlas_brain/config.py`
- `plans/PR-EOM-Live-Calendar-Import.md`
- `scripts/import_eom_customers_live.py`
- `tests/test_eom_live_calendar_import.py`

## Mechanism

`resolve_calendar_ids` maps the three env vars to calendar ids, failing fast
on any gap. `main` fetches each booking calendar through
`GoogleCalendarProvider.list_events` over the configured window, feeds the
events to `parse_events` — a mirror of `parse_ics`'s per-event pipeline
(summary cleaning, address normalization, phone/email/contact extraction,
address-keyed richest-record merge) operating on provider `CalendarEvent`
objects — then merges cross-calendar via `dedup_records` -- a union-find transitive
merger (phone/address keys) applying the inherited field-merge rules with
recency-based cancellation (the inherited `dedup_across_calendars` is not
transitive and clears cancellation on any active record; replaced at
reviewer direction, rounds 4-5). `record_to_contact_data` builds the stamped payload. `import_one`
routes records with a phone/email through `create_contact` (tenant-scoped
dedupe, lead-to-customer upgrade via merge) and address-only records through
`resolve_by_address` -> `update_contact`, closing the duplicate-on-re-run
hole; every dated record logs one `appointment` interaction anchored by a
stable `source_ref`.

## Intentional

- Estimates calendar excluded: it is a lead surface owned by the #2153
  real-time intake; importing it as customers would poison the watcher's
  predicate.
- Calendar ids via env only; this repo is public and the ids stay out of it.
- Extraction/dedupe core imported from the ICS script rather than copied so
  both paths classify identically forever; the ICS script itself is
  untouched.
- The address net applies only to channel-less records; anything with a
  phone/email uses the provider's reviewed dedupe untouched.
- `contact_type` is always sent as `customer`, so `create_contact`'s merge
  can upgrade an existing lead but this script can never downgrade one.
- Three optional typed fields on `ToolsConfig` (reviewer-directed, R11):
  deployment records ids in `.env` as `ATLAS_TOOLS_EOM_CALENDAR_*`, which
  pydantic-settings reads but `os.environ` never sees; the script resolves
  settings first, process env wins.

## Waived (with reason)

- Codex round 15, "Avoid unguarded provider merge on create races" (R4/R8):
  a concurrent writer creating a matching contact in the window between
  this script's own channel searches and `create_contact` routes through
  the provider's OWN reviewed merge path instead of `_update_matched`.
  Waived for the same recorded reason as the round-12 concurrency item:
  single-operator runbook-serialized execution; the "damage" is the
  provider's reviewed merge semantics (used by production intake) rather
  than data loss; and the next sequential run reconciles the row through
  the matched path (asserted idempotency).
- Codex round 12, "Serialize address-only creates" (R8, concurrent-run
  duplicate window): two SIMULTANEOUS runs of this operator script could
  each miss the address lookup and both insert. Waived: the script is a
  runbook-serialized, single-operator tool -- concurrent self-runs are not
  a supported execution mode; sequential re-run idempotency IS asserted
  (criterion 16); the failure mode is a non-destructive duplicate contact,
  repairable by the next run's fallback matching; and the fix (per-address
  advisory locks through the provider's non-transactional create path)
  carries more risk than the window it closes.

## Deferred

- Read-path tenant-scoping defaults (next #2151 slice; the watcher passes
  scope explicitly).
- The #2156 slice B watcher itself (this PR builds its data foundation).
- Operator runs post-merge, in order: the #2155 backfill `--apply` per its
  runbook, then this import with `--dry-run`, then live.

## Verification

- `tests/test_eom_live_calendar_import.py` — 55 passed.
- Live entrypoint verification: direct `--dry-run` against the three
  production booking calendars — 7,163 events -> 93 unique customers,
  correct segments, exit 0.
- Maturity ratchet note: touching `atlas_brain/config.py` wakes the b2d
  tools-lane ratchet (advisory, not merge-required), which flags
  pre-existing drift in `atlas_brain/tools/calendar.py` (12 -> 15) — a file
  this PR does not touch (reproduced locally on the branch base). Per
  review, no baseline change ships in this PR; the drift is tracked in
  #2159 for its own tools-lane slice, and the advisory check stays red
  here by design.
- `tests/test_tenant_stamping.py` — passed (adjacent, 8).
- `tests/test_leads_intake.py` — passed (adjacent, 38).
- `python -m py_compile` on both new Python files — clean.
- Grep proof: 0 occurrences of `group.calendar.google.com` in the diff.
- NOT run: live import against prod (operator-run; dry-run first, after the
  #2155 backfill `--apply`).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/config.py` | 13 |
| `plans/PR-EOM-Live-Calendar-Import.md` | 322 |
| `scripts/import_eom_customers_live.py` | 778 |
| `tests/test_eom_live_calendar_import.py` | 886 |
| **Total** | **1999** |
