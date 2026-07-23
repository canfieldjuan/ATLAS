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
  4. Cancelled-summary records import as `status='inactive'`; calendar
     events with `STATUS=cancelled` are skipped entirely (asserted).
  5. Records with a phone or email route through `create_contact`'s
     tenant-scoped dedupe and never consult the address net (asserted);
     records with neither channel resolve by address first and update
     rather than duplicate (both branches asserted).
  6. Import interactions carry a stable `source_ref` anchor so re-runs
     dedupe (asserted).
- Reachability proof: operator script, invoked directly; the extraction
  helpers execute on every event via `parse_events`, which the tests drive
  with representative summaries/locations/descriptions.

### Files touched

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
objects — then reuses `dedup_across_calendars` unchanged for cross-calendar
merge. `record_to_contact_data` builds the stamped payload. `import_one`
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

## Deferred

- Read-path tenant-scoping defaults (next #2151 slice; the watcher passes
  scope explicitly).
- The #2156 slice B watcher itself (this PR builds its data foundation).
- Operator runs post-merge, in order: the #2155 backfill `--apply` per its
  runbook, then this import with `--dry-run`, then live.

## Verification

- `tests/test_eom_live_calendar_import.py` — 12 passed.
- `tests/test_tenant_stamping.py` — passed (adjacent, 8).
- `tests/test_leads_intake.py` — passed (adjacent, 38).
- `python -m py_compile` on both new Python files — clean.
- Grep proof: 0 occurrences of `group.calendar.google.com` in the diff.
- NOT run: live import against prod (operator-run; dry-run first, after the
  #2155 backfill `--apply`).

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Live-Calendar-Import.md` | 150 |
| `scripts/import_eom_customers_live.py` | 339 |
| `tests/test_eom_live_calendar_import.py` | 210 |
| **Total** | **699** |
