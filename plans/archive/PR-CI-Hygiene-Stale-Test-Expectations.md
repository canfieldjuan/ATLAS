# PR: CI hygiene -- refresh two stale test expectations

## Why this slice exists

Two tests are red on `origin/main` because their pinned expectations were not
updated when earlier work merged. They are unrelated to each other and to any
in-flight feature; the failures pre-date this branch (verified by checking them
out on `origin/main`). The deflection-paid-flow one is what makes the
`atlas-content-ops-deflection-stripe-paid-checks` CI lane red on any PR that
touches the deflection billing surface (e.g. #1462), so it blocks unrelated
merges. This slice refreshes the expectations to match the already-merged,
already-reviewed behavior. No production code changes.

## Scope (this PR)

Ownership lane: testing/ci-hygiene
Slice phase: Robust testing

1. `test_repo_migration_prefix_collisions_are_only_historical_exceptions`: add
   the accepted historical prefix collisions 281/282/283/298 (parallel-session
   b2b migrations that already shipped on main with the same numeric prefix) to
   the allowlist, so the test again catches only NEW collisions.
2. `test_deflection_paid_flow_locks_snapshot_until_stripe_webhook_unlocks`:
   refresh the expected snapshot summary to the current counts produced by the
   merged measured-repetition (#1486: repeat vs non-repeat split) and
   proven-answer gate (#1466: stricter resolution evidence).

Out of scope: any production behavior change; the underlying migration-numbering
convention (the collisions are accepted as historical, not renumbered).

- Reviewer rules triggered: R1, R2.

### Files touched

- `.github/workflows/atlas_migrations_runner_checks.yml`
- `plans/PR-CI-Hygiene-Stale-Test-Expectations.md`
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py`
- `tests/test_migrations_runner.py`

## Mechanism

Both changes are assertion/fixture updates only.

- The collision allowlist gains four entries with the exact sorted file pairs
  `_find_duplicate_migration_prefixes` returns for prefixes 281/282/283/298.
- The paid-flow test's two summary checks (the pre-payment gated snapshot and
  the post-payment unlocked summary) move from the pre-#1486/#1466 counts
  (`generated=3`, `no_proven_answer_count=2`, `repeat_ticket_count=4`) to the
  current ones (`generated=1`, `no_proven_answer_count=0`, `repeat_ticket_count=2`,
  `non_repeat_ticket_count=2`). The markdown section assertions are unchanged --
  the "No Proven Answer Yet" section still renders for this fixture.

## Intentional

- The migration collisions are accepted as historical (already on main), so the
  allowlist is extended rather than the migrations renumbered. The test still
  fails on any genuinely new collision.
- The paid-flow expectations are updated to match merged/reviewed behavior, not
  to assert a new behavior; this is fixture drift, not a product change.

## Deferred

- A scoped real-DB apply test for the deflection migration chain (328/332/336)
  -> the #1462 reconciliation slice.
- Making the full migration set fresh-appliable (the missing `product_metadata`
  migration) -> separate migration-debt issue; out of scope here.

Parked hardening: none.

## Verification

- `tests/test_migrations_runner.py` -- passed (incl. the refreshed collision
  allowlist test).
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` -- passed
  (the refreshed gated + unlocked summary counts).
- Both files failed identically on `origin/main` before this change (verified by
  checkout/stash); they pass after.
- Non-ASCII scan of the two test files -- clean.
- `python -m py_compile` for the two test files -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_migrations_runner_checks.yml` | 45 |
| `plans/PR-CI-Hygiene-Stale-Test-Expectations.md` | 89 |
| `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` | 11 |
| `tests/test_migrations_runner.py` | 20 |
| **Total** | **165** |
