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
3. Housekeeping (arc convention: rides the next slice): archive the
   merged #1948 plan to `plans/archive/PR-Reddit-Listening-Hardening.md`
   and regenerate `plans/INDEX.md`.

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
- `plans/INDEX.md`
- `plans/PR-Negatives-Presence-Gate.md`
- `plans/archive/PR-Reddit-Listening-Hardening.md`
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

Review-fix notes (Codex wave 1; all three verified real, fixed at root):

- **Raises detection was a suppressible text regex**: a comment or
  string mentioning pytest.raises satisfied the now-blocking signal --
  exactly an honest-but-hasty artifact. `_has_raises_assertion` now
  parses the AST for real `pytest.raises`/`raises`/`assertRaises*`
  calls (regex fallback only for unparseable sources). Probed with a
  comment+string-only file. All 35 CI ratchet gates re-run locally
  against the checked-in baselines with the stricter detector: zero
  regressions.
- **Testless sensitive modules escaped entirely**: `NO_TEST_FILE`
  scored below min-score, leaving the worst zero-negatives case
  mergeable. It joins `SENSITIVE_ZERO_TOLERANCE`; probed both ways
  (testless sensitive module fails; same module off-glob passes).
- **Row 7 overstated trusted-base coverage**: the maturity-sweep
  workflows still run `scripts/maturity_sweep.py` from the merge ref,
  so a PR can weaken this very gate and self-pass. Row 7 now reads
  "ENFORCED for the PR meta-gates; maturity sweeps TRACKED" with the
  gap logged on #1942 -- extending trusted-base execution to the sweep
  workflows is a named follow-up slice, not smuggled in here.

Review-fix notes (Codex wave 2; 2 fixed at root, 3 waived on the
written threat model):

- **FIXED -- arbitrary `.raises()` helpers satisfied the signal**: the
  AST check now accepts only the real assertion APIs
  (`pytest.raises(...)`, bare `raises(...)` from-import,
  `assertRaises*`). An aliased `import pytest as pt` is deliberately
  missed -- the gate errs toward demanding the standard idiom. Probed
  with a `client.raises()` helper. All ratchet gates re-run: zero
  regressions.
- **FIXED -- row 4 overstated fixture-fidelity enforcement**: the
  factory is opt-in outside the converted purge/tracker suites; row 4
  now scoped honestly with the residual (static seeding probe) logged
  on #1942.
- **WAIVED -- raises in skipped/unreachable tests count**: this is a
  static presence gate against honest-but-hasty authors; a skipped
  negative test is still an authored negative probe. Proving
  collection/execution is dynamic-coverage territory owned by the
  review contract, not a source scanner.
- **WAIVED -- sparse test files (1-2 tests, no raises) emit neither
  finding**: lowering the >= 3 emission threshold reprices 32 baselined
  modules across every lane (measured), which is a baseline-repricing
  slice of its own -- logged on #1942, not smuggled into this diff.
- **WAIVED -- a module merely mentioned by an unrelated test suppresses
  `NO_TEST_FILE`**: the stem matcher is heuristic by design; removing
  the mentioned-fallback sprays `NO_TEST_FILE` across legitimately
  indirectly-tested modules repo-wide. Named residual, tracked with the
  sparse-threshold item on #1942.

Review-fix note (Codex wave 3; verified real, fixed at root):

- **Bare `raises(...)` accepted without an import check**: a local
  helper or fixture named `raises` suppressed the blocking signal. The
  AST check now collects the names bound by `from pytest import raises`
  (including aliases) and accepts only those. Probed both sides (local
  helper fails the gate; aliased from-import satisfies it). All ratchet
  gates re-run: zero regressions.

## Deferred

- Widening sensitive globs per lane (owners decide).
- Trusted-base execution for the maturity-sweep workflows (the row-7
  residual surfaced in review; logged on #1942).
- The row-2 reconciliation gap stays tracked on #1942 (unrelated to
  this slice; listed to show it was considered).

Parked hardening: none.

## Verification

- `python -m pytest tests/test_maturity_sweep.py -q` -- 27 passed
  (23 pre-existing + both-sides probes: no-raises sensitive module
  fails with the `sensitive-path NO_RAISES_TESTS` reason and passes
  off-glob; with-raises passes on-glob; comment/string mention of
  pytest.raises does NOT satisfy the gate; testless sensitive module
  fails with `sensitive-path NO_TEST_FILE` and passes off-glob).
- All 35 blocking ratchet gates from both maturity-sweep workflows
  re-run locally against the checked-in baselines with the stricter
  detector: every gate passes (no baseline regressions).
- ASCII byte-scan on the two changed Python files: clean.
- Row artifacts verified on main: `tests/atlas_reddit_fixtures.py`,
  `tests/test_atlas_reddit_fixture_fidelity.py` (row 4);
  trusted-base steps in the merged #1949/#1950 workflows (row 7).

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/fable5_pr_1935_1941_review_lessons.md` | 6 |
| `plans/INDEX.md` | 2 |
| `plans/PR-Negatives-Presence-Gate.md` | 130 |
| `plans/archive/PR-Reddit-Listening-Hardening.md` | 0 |
| `scripts/maturity_sweep.py` | 60 |
| `tests/test_maturity_sweep.py` | 280 |
| **Total** | **~520** |

Over the 400 soft cap after two review-fix waves; the PR body
carries the diff-budget override.
