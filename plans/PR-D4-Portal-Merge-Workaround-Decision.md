# PR-D4-Portal-Merge-Workaround-Decision

## Why this slice exists

Website #127 (D4): re-evaluate the portal-sync `merge_existing=False` workaround
now that 0B (ATLAS #2313) has landed, and either retire it or keep it with
recorded evidence. **The outcome is keep** -- 0B did not close the gap.

### The comparison (acceptance criterion 1), from code + live data

`scripts/sync_eom_portal_customers.py` calls `crm.create_contact(..., merge_existing=False)`
only after its own `resolve_contact` returns no match. That resolver is a
five-rung ladder: `portal_customer_id` -> `atlasContactId` -> phone -> email ->
address. The provider's create-path matcher (`crm_provider.create_contact._resolve`)
sees only **phone + email**, and its phone predicate in `search_contacts` is
still substring: `... LIKE '%digits%' OR ... LIKE '%last10%' OR RIGHT(...)=RIGHT(...)`.
0B added the exact `RIGHT(...)` clause but left the two `LIKE` clauses ORed in,
so the matcher remains a superset that can false-positive, and it still ignores
the portal-id/atlas-id/address rungs.

Live check (706-row `contacts`): the substring predicate produces 0 false phone
matches *today*, but that is a property of the current data, not the matcher --
`LIKE '%last10%'` matches any two numbers sharing a 10-digit suffix, so the
decision must not rest on today's data staying collision-free.

Enabling `merge_existing=True` would therefore both **miss** three identity rungs
the portal resolver already handled and **reintroduce** substring false-positives.
Keep the workaround.

### Problem-derived contract

- Root cause: the provider create-path matcher is narrower (phone+email) and
  looser (substring phone) than the portal sync's own resolver; 0B did not
  change that.
- Correct resolution touches: the decision record (this PR + closing #127) and
  the durability of two currently-under-protected invariants -- the workaround
  flag and the demotion source list. No portal-sync behaviour changes.
- Must not change: `merge_existing=False`, the demotion filter, `DEMOTABLE_SOURCES`,
  the provider matchers.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: production hardening

1. Document, at the existing `merge_existing is False` assertion, *why* it is
   load-bearing (five-axis-vs-two + substring, re-evaluated against 0B), so a
   future reader cannot flip the flag and "fix" the test in one motion.
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

- Reviewer rules triggered: R2, R14.

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
| `plans/PR-D4-Portal-Merge-Workaround-Decision.md` | 117 |
| `tests/test_sync_eom_portal_customers.py` | 26 |
| **Total** | **143** |
