# PR-D4-Portal-Merge-Workaround-Decision

## Why this slice exists

Website #127 (D4): re-evaluate the portal-sync `merge_existing=False` workaround
now that 0B (ATLAS #2313) has landed, and either retire it or keep it with
recorded evidence. **The outcome is keep** -- but not for the reason the
issue assumed: matcher strength was never the deciding factor (see below).

### The comparison (acceptance criterion 1), corrected after review

My first draft claimed the provider phone matcher is "weaker" than the portal
sync's resolver. **That is false, and Codex R1/R14 caught it.** The portal
resolver's phone rung (`resolve_contact` -> `live._search_channel` at
`scripts/import_eom_customers_live.py:529`) delegates to `crm.search_contacts`
-- the same substring predicate as the provider create path, mirroring
`create_contact._resolve`'s scoped-then-claimable order. Phone and email
matching are identical between the two.

The real differential is the merge MODE, which the code docstring already names
"the portal-reconciliation race seam":

- `merge_existing=False` (the seam): skips phone entirely, returns only a
  same-tenant email match **without claiming a NULL-context row or merging any
  field**, and otherwise creates fresh.
- `merge_existing=True` (default): re-runs phone+email over scoped and claimable
  pages, claims NULL-context legacy rows by CAS, and **merges non-null fields**
  into whatever it matches.

`resolve_contact` runs the full five-rung ladder (portal_customer_id,
atlasContactId, phone, email, address -- phone and email via the *same*
`_search_channel` as the create path, address via `resolve_by_address`) **before**
`create_contact` is ever reached. If it matched,
the portal sync is on the update path; `create_contact` is reached only when
resolution already found nothing. So in the common case the two modes are
equivalent -- both re-query, find nothing, insert. They diverge only in the race
window between resolve and create (a concurrent insert), where `merge_existing=
True` would re-find and merge/claim it -- potentially cross-linking to a row a
different portal customer just created, and overwriting fields the clean-create
path deliberately leaves alone.

**Therefore matcher strength -- the thing 0B changed -- was never the deciding
factor.** 0B is orthogonal to a race-window seam and cannot obsolete it. Both
the #127 premise and the provider docstring misattribute the reason to matcher
strength.

**The race window cuts both ways (correcting a second over-claim).** My first
correction only weighed the harmful side. Codex R1/R14 is right that there is a
benefit too:

- *Same* phone-only customer, two overlapping runs that both clear
  `resolve_contact` before either inserts: `merge_existing=False` skips phone and
  inserts a **duplicate**; `merge_existing=True` re-finds the first row and
  dedupes. Enabling merge helps here.
- *Different* customers whose phones collide under the substring predicate:
  `merge_existing=True` merges into the wrong row -- a silent **cross-link plus
  field overwrite**; `merge_existing=False` inserts clean.

So enabling merge is a real tradeoff, not "no benefit." It is kept anyway,
weighed:

1. **Asymmetric failure.** `merge_existing=False` fails toward a duplicate that
   carries the *same* `portal_customer_id` -- detectable by a `COUNT(*)>1` query
   and reconcilable. `merge_existing=True` fails toward a silent merge of two
   distinct customers with data overwrite -- no count signal, hard to undo. For a
   system of record, detectable-duplicate beats silent-wrong.
2. **The duplicate is already mitigated** within a run (cached resolutions and
   the shared-normalized-identity check, `sync_eom_portal_customers.py:249,263`);
   it requires two *overlapping* runs to surface at all.
3. **There is a better fix for the duplicate than flipping the seam**: a partial
   unique index on `(business_context_id, metadata->>'portal_customer_id')`
   closes the duplicate at the DB level without taking on the cross-link risk. If
   same-customer duplicates ever become real operationally, that index -- not
   `merge_existing=True` -- is the fix, filed as website #137 rather than done
   here. (The `create_contact` docstring does gesture at a DB-level partial
   unique index "for extra safety", but for phone/email dedup, and its "migration
   037" pointer is stale -- 037 is `037_plan_status.sql` -- so the index stands
   on its own merit, not on that reference.)

**Outcome: KEEP `merge_existing=False`**, as a weighed tradeoff. 0B does not
change either side of it.

### Problem-derived contract

- Root cause: `merge_existing=False` is a race-window seam, not a
  matcher-strength workaround. `resolve_contact` owns identity resolution
  upstream (same matcher as the create path), so `create_contact` is deliberately
  a race-safe clean insert. 0B changed the matcher, which is orthogonal to that.
- Correct resolution touches: the decision record (this PR + closing #127) and
  the durability of two currently-under-protected invariants -- the workaround
  flag and the demotion source list. No portal-sync behaviour changes.
- Must not change: `merge_existing=False`, the demotion filter, `DEMOTABLE_SOURCES`,
  the provider matchers.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: production hardening

1. Document, at the existing `merge_existing is False` assertion, *why* it is
   load-bearing -- the race-window merge behaviour (see the comparison above),
   NOT matcher strength -- so a future reader cannot flip the flag and "fix" the
   test in one motion.
2. Add `test_demotable_sources_are_pinned_to_calendar_and_portal_only`. The
   demotion source list is the issue's highest-blast-radius element -- widening
   it silently archives live customers -- and was pinned by no test. This
   enforces acceptance criterion 4 (byte-identical) going forward. It **guards**
   the filter; it does not change it.

### Files touched

- `plans/PR-D4-Portal-Merge-Workaround-Decision.md`
- `tests/test_sync_eom_portal_customers.py`

### Review Contract

1. Zero behaviour change: `scripts/sync_eom_portal_customers.py` is byte-identical
   to `origin/main`. `merge_existing=False` and `DEMOTABLE_SOURCES` are unchanged.
2. The workaround cannot silently regress: flipping the script flag fails the
   existing pin.
3. The demotion source list cannot silently widen: adding any source fails the
   new guard.

Affected surfaces: one test file. No script, no provider, no schema.

Risk areas: none material -- this adds guards to existing invariants.

- Reviewer rules triggered: R1, R2, R13, R14. (R1: this decides whether the workaround satisfies website #127. R13: the guard-class closure declaration and the exact source-set assertion.)

**Guard-class closure declaration -- `DEMOTABLE_SOURCES`**

- **Member set:** `DEMOTABLE_SOURCES` in `scripts/sync_eom_portal_customers.py`,
  the contact `source` values a portal run may auto-archive.
- **CLOSED and ENUMERATED**, a literal 2-tuple `("calendar_import",
  "portal_sync")`. Not derived from data, a query, or a naming convention.
- **Canonical basis: roster-authoritative provenance**, NOT "system-managed."
  The earlier draft said system-managed, which is wrong and self-contradictory:
  the parent commit (D2, ATLAS #2314) routes the automated `email_backfill` task
  through `resolve_or_create_eom_contact` to create EOM *customers*, so
  `email_backfill` IS system-managed -- yet it is correctly excluded here. The
  actual membership rule is narrower: a source belongs only if the portal roster
  is the authority on that customer's current membership, so *absence from the
  roster is a reliable churn signal*.
  - `portal_sync`: created from the roster; absent -> churned. In.
  - `calendar_import`: roster-correlated -- a live booking is meant to veto
    demotion (`sync_eom_portal_customers.py:564`), so absence-from-roster plus
    no-booking should be churn. **Known gap (website #138):** the veto only
    builds guard keys 4 months ahead while imports admit bookings 12 months
    ahead, so a customer with a booking 5-12 months out can still be wrongly
    demoted. In the set *as inherited from `main`*, not certified
    unconditionally safe -- whether it should stay is the separate review #138
    forces, and this PR does not change the set.
  - `email_backfill`: system-created, but a backfilled email address is not a
    roster-membership claim; an active customer can be absent from the portal
    system entirely. Absence is meaningless, so it must never be auto-archived.
    Out -- despite being system-managed.
  - `manual`, `web`, `phone_call`: human/intake-authored, no roster authority.
    Out.
- **Out-of-set default: NOT demotable (safe).** The demotion `UPDATE` filters
  `AND source = ANY($4)`, so any source outside the tuple -- system-managed or
  not -- simply does not match and is left active.
- **What the guard certifies.** The pin exists to make *widening* the set a
  visible, reviewed change (adding `manual`/`web` would mass-archive live
  customers). It does NOT certify that every current member is bug-free:
  `calendar_import`'s veto has the horizon gap above (#138). Widening is the
  catastrophe this prevents; per-member correctness is tracked separately.
- **Widening is the hazard, not narrowing.** Adding a value turns a routine sync
  into a mass archive of live customers, which is why the new test pins exact
  membership. Changing it is a deliberate, separately-reviewed decision.

**Mutation-probe (run, not asserted):** widening `DEMOTABLE_SOURCES` fails the
new guard; flipping `merge_existing` in the script fails the existing pin.

## Mechanism

One documentation comment and one assertion.

## Intentional

- **Keep, not retire.** The issue explicitly names "keep `merge_existing=False`
  with evidence" as a valid outcome, and the evidence points that way.
- **Guard the demotion list even though the filter is out of scope for changes.**
  Criterion 4 requires it byte-identical; a pin is the enforcement of that
  requirement, not a modification of the filter.

## Deferred

- Retiring the substring matcher in `search_contacts` -- a separate, larger
  change with its own blast radius. Not required for D4.
- D1, D3, D5 (website #124, #126, #128).

Parking predicate: this slice parks everything except the recorded decision and
the two guard tests.

Parked hardening: none.

## Verification

```
$ python -m pytest tests/test_sync_eom_portal_customers.py -q
54 passed

$ git diff --stat origin/main -- scripts/sync_eom_portal_customers.py
(empty -- script byte-identical)
```

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-D4-Portal-Merge-Workaround-Decision.md` | 213 |
| `tests/test_sync_eom_portal_customers.py` | 36 |
| **Total** | **249** |
