# PR-Negatives-Presence-Gate

## Why this slice exists

Lesson 5 of the arc index (`docs/fable5_pr_1935_1941_review_lessons.md`):
suites that only probe happy paths stayed green over 8 reviewer-caught
injection holes in a single slice (S3). The detection half already
exists -- `scripts/maturity_sweep.py` emits `NO_RAISES_TESTS` when a
module's test files never assert that anything raises -- but at weight 3
it cannot reach the blocking ratchet's min-score alone, so today it is a
whisper, not a gate. This slice makes it blocking where it matters:
inside the sensitive globs the ratchet workflows already declare.

## Scope (this PR)

Ownership lane: workflow/negatives-presence-gate
Slice phase: Workflow/process

1. `scripts/check_diff_budget.py`-style one-line root change: add
   `NO_RAISES_TESTS` to `SENSITIVE_ZERO_TOLERANCE` in
   `scripts/maturity_sweep.py`. The existing zero-tolerance ratchet
   already fails ANY new occurrence of listed codes inside
   `--sensitive-glob` paths regardless of min-score; the CI workflows
   already pass guard-lane globs (auth, billing, webhook, payment,
   deletion, extracted lanes, scripts).
2. Index maintenance mandated by the doc's own standing rule: rows 4, 5,
   and 7 of `docs/fable5_pr_1935_1941_review_lessons.md` flip to
   ENFORCED naming their gates -- 5 by this slice, 4 landed via #1947,
   7 landed via #1949/#1950; neither PR updated the index.

### Review Contract

- Acceptance criteria:
  - [ ] A NEW module on a sensitive glob whose tests (>= 3) never
        assert a raise fails the ratchet with a
        `sensitive-path NO_RAISES_TESTS` failure, and min-score cannot
        save it.
  - [ ] The same module off the sensitive globs still passes (the gate
        is scoped, not repo-wide noise).
  - [ ] The same module WITH a raises assertion passes even on the
        sensitive glob (second side).
  - [ ] All pre-existing maturity-sweep tests still pass (baselined
        files with stable counts are unaffected).
- Affected surfaces: `scripts/maturity_sweep.py` (one tuple),
  `tests/test_maturity_sweep.py`, the lessons index rows.
- Risk areas: gate false positives -- bounded by scoping to the
  sensitive globs the workflows already declare and by the existing
  `--update-baseline` acceptance path the failure message prints.
- Reviewer rules triggered: R2, R10 (gate predicate change: both sides
  probed), R12 (`tests/test_maturity_sweep.py` runs blocking in the
  maturity-sweep workflows and the pre-push audit), R14.

### Files touched

- `docs/fable5_pr_1935_1941_review_lessons.md`
- `plans/PR-Negatives-Presence-Gate.md`
- `scripts/maturity_sweep.py`
- `tests/test_maturity_sweep.py`

## Mechanism

`ratchet_failures` already iterates `SENSITIVE_ZERO_TOLERANCE` for every
swept file matching a sensitive glob and fails on any per-code count
increase over the baseline (new files compare against zero). Adding
`NO_RAISES_TESTS` to that tuple therefore needs no new plumbing: the
finding is emitted by `score_tests` for substantive modules with >= 3
matched tests and no `pytest.raises`/`assertRaises`, and the ratchet
does the rest.

## Intentional

- **`HAPPY_PATH_TESTS` deliberately stays out of zero tolerance.** It is
  a name-based heuristic (counts tests whose names contain failure
  hints), so gating on it teaches cosmetic renaming and punishes adding
  happy-path coverage to a module with a fixed number of negatives.
  `NO_RAISES_TESTS` is the crisp mechanical proxy for the actual S3
  failure mode: zero raises assertions on a boundary.
- **No new globs in this slice.** The workflows' existing sensitive
  globs (auth/billing/webhook/payment/deletion/extracted lanes) already
  cover the guard lanes the lesson names; widening coverage is a
  per-lane decision for the sessions that own those lanes.
- **Threat model** (per the index's own practice): the adversary is
  honest-but-hasty authors. The gate cannot prove negatives are
  adversarial; it makes "zero negative probes on a sensitive boundary"
  impossible to merge silently, which is the failure mode the arc
  actually had. A determined adversary can still write one trivial
  raises-test; the review contract, not this gate, owns that.

## Deferred

- Widening sensitive globs per lane (owners decide).
- The row-2 reconciliation gap stays tracked on #1942 (unrelated to
  this slice; listed to show it was considered).

Parked hardening: none.

## Verification

- `python -m pytest tests/test_maturity_sweep.py -q` -- 25 passed
  (23 pre-existing + the new both-sides probes:
  new-sensitive-module-without-raises fails with the
  `sensitive-path NO_RAISES_TESTS` reason and passes off-glob;
  with-raises passes on-glob).
- ASCII byte-scan on the two changed Python files: clean.
- Row artifacts verified on main: `tests/atlas_reddit_fixtures.py`,
  `tests/test_atlas_reddit_fixture_fidelity.py` (row 4);
  trusted-base steps in the merged #1949/#1950 workflows (row 7).

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/fable5_pr_1935_1941_review_lessons.md` | 6 |
| `plans/PR-Negatives-Presence-Gate.md` | 120 |
| `scripts/maturity_sweep.py` | 7 |
| `tests/test_maturity_sweep.py` | 100 |
| **Total** | **~233** |

Under the 400 cap; no override needed.
