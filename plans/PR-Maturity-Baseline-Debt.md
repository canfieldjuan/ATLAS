# PR-Maturity-Baseline-Debt

## Why this slice exists

Two pre-existing, unbaselined maturity findings on main turn the advisory
ratchet red for EVERY PR that wakes their lanes (tracked in #2159; surfaced
again on #2158/#2161/#2163). Verified against current main:

- `atlas_brain/tools/calendar.py` (score 15): the flagged patterns are
  deliberate — `events[0]` sits behind explicit `if not events` /
  `len(events) == 1` guards (sweep false-positive), and both swallowed
  excepts are commented intentional fallbacks ("Already logged as
  CRITICAL").
- `scripts/import_eom_customers_live.py` (score 9): reviewed per-record
  operator patterns from the 18-round #2158 gauntlet, deliberately NOT
  baselined inside that PR per its round-4 review direction ("move the
  baseline acceptance to its owning PR") — this is that owning PR.

### Problem-derived contract

- Root cause: recorded-intentional patterns without baseline entries make
  the ratchet cry wolf on unrelated PRs.
- Correct fix must touch/change: exactly two baseline entries (tools lane:
  calendar.py; scripts lane: import script), accepted deliberately with the
  rationale above. No production code changes.
- Must NOT change: any Python source; any other baseline entry (surgically
  verified against origin/main).

## Scope (this PR)

Ownership lane: ci/maturity-baselines
Slice phase: vertical slice

1. `tests/maturity_sweep/baseline_atlas_brain_tools.json`: calendar.py entry.
2. `tests/maturity_sweep/baseline_scripts.json`: import script entry.
3. This plan doc.

### Review Contract

- Acceptance criteria:
  1. Both ratchet lanes pass locally against these baselines (verified:
     "ratchet gate passed" both).
  2. The diff contains ONLY the two entries + this doc (name-status
     verified).
- Reachability proof: CI ratchet jobs consume the baselines directly.
- Reviewer rules triggered: R12, R14.
  - R12: CI config/baseline change only; no behavior, no secrets.
  - R14: this contract is the reviewer checklist.

### Files touched

- `plans/PR-Maturity-Baseline-Debt.md`
- `tests/maturity_sweep/baseline_atlas_brain_tools.json`
- `tests/maturity_sweep/baseline_scripts.json`

## Mechanism

`--update-baseline` runs for both lanes, then a surgical pass restores every
entry except the two named files to origin/main's values.

## Intentional

- Baseline acceptance over code churn: the flagged patterns are correct as
  written; "fixing" them would add noise to reviewed code.

## Deferred

- Nothing; closes #2159.

## Verification

- Local ratchet runs: both lanes "ratchet gate passed".
- `git diff --stat`: 2 baseline files, +9/-1.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Maturity-Baseline-Debt.md` | 75 |
| `tests/maturity_sweep/baseline_atlas_brain_tools.json` | 3 |
| `tests/maturity_sweep/baseline_scripts.json` | 7 |
| **Total** | **85** |
