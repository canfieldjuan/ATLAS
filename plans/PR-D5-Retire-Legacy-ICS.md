# PR-D5-Retire-Legacy-ICS

## Why this slice exists

Website #128 (D5), the last 0D child: `scripts/import_calendar_contacts.py` is a
strictly-worse duplicate of the live importer and can be run by accident.

### Problem-derived contract

- **Root cause.** The legacy importer writes through `crm_provider.create_contact`,
  which resolves an existing contact on **phone then email only, never address**
  (provider docstring: "returning an existing one if phone or email already
  matches. Dedup order: phone first, then email."). So an **address-only** record
  (no phone, no email) can never match an existing contact and is **re-created on
  every run** -- the documented rerun-duplicate defect. The replacement
  `scripts/import_eom_customers_live.py` added an address pre-resolver
  (`import_one`); the legacy script did not. Repairing the legacy one means
  maintaining two importers where the older is strictly worse.
- **Correct fix touches.** Retire the legacy script's **runnable command** behind a
  deprecation error naming the replacement -- WITHOUT breaking the module's
  importability, because the replacement does `import import_calendar_contacts as
  ics` (`import_eom_customers_live.py:38`) and reuses its `parse_ics` /
  `CustomerRecord` extraction core. Pin the defect by test **first**, or retiring
  the CLI makes the defect unobservable and its evidence is lost.
- **Must NOT change.** `import_records` / `parse_ics` / `dedup_across_calendars` /
  `CustomerRecord` (kept intact and importable); the replacement importer; the
  `"business_context_id": "effingham_maids"` line (`test_tenant_stamping` reads
  it); the defect itself (retire, do not repair). No deletion.

### Operator gate (satisfied)

#128 blocks on operator confirmation that no workflow depends on the ICS path.
**Received 2026-08-08:** Juan confirmed he does not run the manual
`python scripts/import_calendar_contacts.py` command, after I traced the ops-tab
Schedule import to a separate live Google Calendar path
(`service-schedule.js` -> `/admin/google-calendar/*`) with zero `.ics` references
in the portal, the schedule JS, or the tracker backend. Nothing automated invokes
the legacy script; its only code reference is the replacement's library reuse.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: production hardening

1. `scripts/import_calendar_contacts.py`: the `if __name__ == "__main__"` entry
   raises `SystemExit` with a deprecation message naming the replacement, instead
   of `asyncio.run(main())`. A `RETIRED` banner is added to the module docstring.
   Nothing else in the module changes.
2. `tests/test_calendar_import_rerun.py`: pin the rerun-duplicate defect (address-
   only duplicates; a phone-bearing record does not) and pin that the module stays
   importable as a library.

### Files touched

- `scripts/import_calendar_contacts.py`
- `tests/test_calendar_import_rerun.py`
- `plans/PR-D5-Retire-Legacy-ICS.md`

### Review Contract

1. Running the script exits with a clear, non-silent deprecation error that names
   `scripts/import_eom_customers_live.py`.
2. The module remains importable: `import import_calendar_contacts` succeeds and
   `parse_ics` / `CustomerRecord` / `import_records` are still present, so the
   replacement importer's `import ... as ics` reuse is unaffected.
3. The defect is pinned by test BEFORE retirement: an address-only record
   duplicates on re-run; a phone-bearing record does not (the contrast proves the
   claim is class-specific, not a blanket "import_records is broken").
4. No behavior change to any path that is not the ICS importer's runnable command.
   `import_records`, `parse_ics`, dedup, and the tenant stamp are byte-unchanged.

Affected surfaces: the `__main__` entry only (one call swapped for a raise) plus a
docstring banner. No provider change, no schema change, no change to the
replacement importer.

Risk areas: (a) breaking the library reuse -> probed by
`test_module_stays_importable_as_a_library` and a live `import
import_eom_customers_live` check; (b) losing the defect evidence -> probed by the
two rerun tests, which exercise the real `import_records` against a faithful
phone/email-only provider stub.

- Reviewer rules triggered: R1, R14. (R1: retire the dup-generating path at its
  root -- the runnable entry -- rather than patching a symptom, and pin the root
  defect by test first. R14: reviewer verdict discipline.)

**boundary-probe:** both sides -- the runnable command refuses (exit 1), while the
importable-library path still works (`parse_ics`/`CustomerRecord` present).

**Mutation-probe (run, not asserted):** the pin test exercises the real
`import_records`; a stub that (wrongly) resolved on address would make
`test_address_only_record_duplicates_on_rerun` fail, so the test is bound to the
real defect, not the stub's convenience.

## Mechanism

One entry-point swap (`asyncio.run(main())` -> `raise SystemExit(...)`) plus a
docstring banner. The library surface is untouched.

## Intentional

- **Retire, do not repair.** #128 is explicit: the legacy path is strictly worse
  than the replacement, so repairing it would mean maintaining two importers.
- **CLI-only, not import-time.** The replacement reuses this module as a library,
  so the deprecation must fire on execution (`__main__`), never on import -- an
  import-time raise would break the dominant live importer.
- **Deprecation, not deletion.** The runnable path is recoverable by reverting one
  commit; the module and its extraction core stay in place.
- **Pin before retire.** Once `__main__` raises, the duplicate behavior is
  unobservable by execution; the test preserves the evidence it existed.

## Deferred

- Fixing the legacy `import_records` to resolve by address -- explicitly out of
  scope (the replacement already does this; the legacy path is being retired, not
  repaired).
- Routing the live importer through the canonical boundary -- separate concern,
  tracked as website #142 (the D3 remainder).

Parked hardening: none.

## Verification

```
$ python -m pytest tests/test_calendar_import_rerun.py -q
3 passed

$ python scripts/import_calendar_contacts.py --dry-run
scripts/import_calendar_contacts.py is retired: ... (exit 1)

$ python -c "import sys; sys.path.insert(0,'scripts'); import import_eom_customers_live"
(imports OK -- library reuse intact)
```

`tests/test_tenant_stamping.py`, `tests/test_contact_write_boundary.py`, and
`tests/test_eom_live_calendar_import.py` (which imports the legacy module) all
stay green.

## Estimated diff size

| File | LOC (added) |
|---|---:|
| `scripts/import_calendar_contacts.py` | 22 |
| `tests/test_calendar_import_rerun.py` | 133 |
| `plans/PR-D5-Retire-Legacy-ICS.md` | 145 |
| **Total** | **300** |
