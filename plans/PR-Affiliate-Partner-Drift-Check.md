# PR-Affiliate-Partner-Drift-Check

## Why this slice exists

`affiliate-system-investigation.md` item #2 had two halves. Migration 326
(PR #664) did the first: version-control the existing partner rows. This
slice does the second -- "stop creating partners via raw API calls in prod"
-- by making drift *detectable*.

The `/b2b/tenant/affiliates` API still exposes full `POST`/`PATCH`/`DELETE`.
Nothing prevents a partner row from being created directly in the DB again,
re-introducing exactly the ungoverned data the migration just cleaned up: no
git history, no review, no disaster-recovery path. Capturing the current rows
without a way to catch the *next* drift would be patching the symptom, not
the source.

This adds an audit that reconciles the live `affiliate_partners` table against
the partner definitions seeded by migrations, and fails when they diverge.

## Scope (this PR)

1. Add `scripts/audit_affiliate_partner_drift.py` (matches the existing
   `scripts/audit_*` convention; same `_status`/`_exit_code`/JSON-report shape
   as `scripts/check_reasoning_rollout_readiness.py`).
2. Reconcile live rows against migration seeds with four checks:
   - `migration_seeds_parseable` -- **FAIL** if the parser cannot map a
     migration's `INSERT INTO affiliate_partners` (column/value mismatch or
     missing VALUES). A parse regression must not be silently dropped, since
     reconciliation would then run against an incomplete seed set.
   - `all_live_partners_versioned` -- **FAIL** if a live partner's
     `product_name` is seeded by no migration.
   - `no_seed_value_divergence` -- **WARN** if a seeded partner's migration
     definition differs from the live row on a business field
     (`name`, `category`, `affiliate_url`, `commission_type`,
     `commission_value`, `notes`, `product_aliases`).
   - `no_orphan_seeds` -- **WARN** if a migration seeds a `product_name` with
     no live row.
3. Add `tests/test_affiliate_partner_drift.py` covering the pure parser and
   reconciler (no DB), including both migration formats and each drift case.

### Files touched

- `scripts/audit_affiliate_partner_drift.py`
- `tests/test_affiliate_partner_drift.py`
- `plans/PR-Affiliate-Partner-Drift-Check.md`

## Mechanism

The parser and reconciler are pure functions (no DB), so they run and are
tested everywhere -- CI included -- while only `_run` touches the live
database (the same `:5433/atlas` the app uses, via `init_database()`).

`parse_seeded_partners` finds each `INSERT INTO affiliate_partners (...)
VALUES ...` block with a quote- and bracket-aware tokenizer (not a naive
regex), so it handles both seed formats already in the tree: the packed
7-column 088 Amazon row and the 9-column 326 rows with `ARRAY[...]::text[]`
and `'{...}'::text[]` aliases, `NULL`, and `true`/`false`. It reads **every**
value tuple in a multi-row `VALUES (...), (...)` insert, not just the first,
so a future batched seed is not silently truncated. A column/value count
mismatch (or a missing VALUES clause) is **not** silently skipped: it is
returned as an error and surfaced by the `migration_seeds_parseable` FAIL
check, because quietly dropping a seed would make the reconciliation compare
against stale data and report a misleading pass. Values are normalized to
Python types; later migrations override earlier ones for the same
`product_name` (apply-order semantics).

`reconcile` compares by `lower(product_name)`. Aliases are compared as sets
(the matcher is order-insensitive, so a reordered array is not drift). `notes`
normalizes `''` and `NULL` as equal. `enabled` is deliberately excluded from
divergence -- toggling a partner on/off is legitimate operational state, not
a version-control gap.

The script exits non-zero only on a `fail` (an unversioned live partner), so
it is safe to wire into a pre-deploy hook or run manually; warnings surface
post-seed edits and orphan seeds without blocking.

## Intentional

- **Operator/pre-deploy script, not a CI gate.** The self-hosted deployment's
  CI has no access to the live `:5433/atlas` DB, which is the only place real
  drift is visible. The DB-touching path is therefore run by the operator; the
  CI-runnable value (the parser + reconciler) is covered by unit tests.
- **WARN, not FAIL, on value divergence.** A URL/commission edit made in the
  DB may be a deliberate operational change; the audit surfaces it and lets
  the operator decide whether the DB or the migration is authoritative, rather
  than blocking deploys on a judgment call.
- **Hand-written tokenizer over a SQL library.** The seed format is small and
  fixed; a dependency-free tokenizer keeps the audit self-contained and its
  behavior fully covered by the fixture tests.
- **Larger than 400 LOC by design.** This is a net-new audit with its own
  parser and full test coverage, not a change to existing code; the bulk is
  the tokenizer and its tests.

## Deferred

- **Auto-generating a migration from a drifted DB row.** The audit reports
  drift; turning a live row into a migration stub is a separate convenience.
- **Restricting the partner API to read-only in prod.** A heavier alternative
  that removes the only partner-management path; not warranted while the audit
  makes drift visible.
- **Wiring the audit into a scheduled task / alert.** Routine observability,
  not blocking.

## Verification

- `python -m pytest tests/test_affiliate_partner_drift.py -q` -> `9 passed in
  0.06s` (parser on both 088 + 326 formats, multi-row VALUES,
  empty-array/quote-escape, column/value-mismatch surfacing, and the clean /
  unversioned-FAIL / value-divergence-WARN / unparseable-FAIL reconcile paths).
- `scripts/audit_affiliate_partner_drift.py` run against live `:5433/atlas`
  -> `summary` `{seeded_partners: 6, live_partners: 6, pass: 4, warn: 0,
  fail: 0}`, exit `0`. Confirms the parser reads the real 088 + 326 seeds and
  that migration 326 fully reconciles with the live table (every live partner
  versioned, no value divergence).
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `scripts/audit_affiliate_partner_drift.py` (parser + reconciler + DB run) | ~390 |
| `tests/test_affiliate_partner_drift.py` | ~270 |
| Plan doc | ~145 |
| **Total** | **~805** |
