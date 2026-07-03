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
- `tests/maturity_sweep/baseline_scripts.json`
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

Review-fix notes (Codex wave 4; all three verified real, fixed at
root):

- **Dangling `pytest.raises(X)` statements counted as assertions**: the
  call builds a context manager and asserts nothing. The AST check now
  accepts a raises-API call only in an asserting position -- a with-item
  or the callable form (>= 2 args). Probed both sides.
- **Stub test files escaped as coverage**: a matched test file with zero
  collected tests returned no finding at all. It now emits NO_TEST_FILE
  ("matched test file(s) contain no tests"). Blast radius measured: the
  only baselined modules with zero-test matched files are 7 __init__.py
  files, which test scoring already excludes -- zero repricing.
- **Row 5 claimed the full adversarial-quality lesson**: the gate proves
  negatives EXIST; whether they probe the exact boundaries the lesson
  names is the review contract's job (R2). Row 5 now reads "ENFORCED for
  negatives PRESENCE; adversarial QUALITY review-owned" -- this is the
  gate's threat-model boundary written into the index itself.

Review-fix notes (Codex wave 5; all four verified real, fixed at
root):

- **Any `.assertRaises*` attribute call satisfied the signal**: a helper
  like `client.assertRaises(ValueError, fn)` suppressed the blocking
  code. assertRaises* now counts only on the unittest receivers
  self/cls. Probed both sides (client helper fails; real
  `self.assertRaises` passes).
- **Dangling `self.assertRaisesRegex(Exc, "regex")` counted**: with only
  the regex it builds a context object and asserts nothing, but the
  shared >= 2-args test accepted it. Each accepted API now carries its
  own callable-form arity (assertRaisesRegex* needs 3 positional args;
  assertRaises and pytest.raises need 2). Probed both sides.
- **Row 5 did not name the >= 3-test emission threshold**: the sparse
  1-2-test case is a waived residual (item 7), but the index row claimed
  the unqualified lesson. Row 5 now names the threshold and the #1942
  detector-fidelity follow-up in the row itself.
- **The plan's diff-size table drifted from the actual diff**:
  review-fix growth left the checked-in numbers stale against
  `scripts/sync_pr_plan.py --check`. Table resynced with the helper.

Review-fix note (Codex wave 6; verified real, fixed at root):

- **`self.assertRaises*` accepted without unittest ancestry**: a
  pytest-style class with a non-asserting helper named assertRaises
  suppressed the signal. The call now counts only inside a class whose
  bases statically name a *TestCase (unittest.TestCase,
  IsolatedAsyncioTestCase, project FooTestCase bases). A cross-file
  base not named *TestCase is deliberately missed -- the standard-idiom
  trade-off already named for aliased pytest imports. Blast radius
  measured: the repo's only real self.assertRaises user inherits
  unittest.TestCase directly -- zero repricing. Probed both sides.

Review-fix notes (Codex wave 7; 2 fixed at root, 1 refuted with a
disproof):

- **FIXED -- `import pytest as pt` assertions misreported as absent**:
  the module-receiver branch only accepted the literal name pytest, so
  a real `pt.raises(...)` failed an honestly-tested sensitive module.
  The check now tracks `import pytest as ...` aliases the same way the
  from-import aliases are tracked. This retires the wave-2 named
  trade-off outright. Probed (aliased import satisfies the gate).
- **FIXED -- unparseable matched test files failed open**: the
  SyntaxError fallback reused the old text regex, so a comment
  mentioning pytest.raises in a broken file suppressed the signal. A
  source that does not parse has no runnable tests; it now fails
  closed. Probed (syntax-error file with a comment mention fails the
  gate).
- **REFUTED -- async-only test files are NOT counted as testless**: the
  `def\s+(test_\w+)` counter is unanchored, so `async def test_x`
  already matches (verified by direct execution). A regression-locking
  probe now pins async-only suites as counted, non-stub, and
  raises-satisfying.

The fail-closed handler tripped the sweep's own SWALLOWED_EXCEPT
heuristic on the scripts lane (the gate caught itself). The single
falsy return IS the deliberate fail-closed policy, not silent
degradation, so it was accepted through the documented
`--update-baseline` path rather than restyled to dodge the detector --
the cosmetic-evasion move this plan itself rejects. The refreshed
`tests/maturity_sweep/baseline_scripts.json` also reconciles two
pre-existing drift entries (a deleted script removed, an existing
sub-min-score script added).

Review-fix notes (Codex wave 8; 4 fixed at root, 1 waived on the
written threat model):

- **FIXED -- the test counter was a text regex**: commented-out
  `def test_*` lines made a placeholder file look non-stub, dodging
  NO_TEST_FILE. `_collect_test_defs` now counts real def/async-def AST
  nodes; unparseable sources contribute no runnable tests (fail closed,
  matching the raises detector). Probed (comment-only stub fails).
- **FIXED -- any `self.assertRaises*` prefix matched**: a fabricated
  `self.assertRaisesLater(...)` helper satisfied the signal. Only the
  exact unittest APIs count now (assertRaises, assertRaisesRegex,
  assertRaisesRegexp). Probed both sides.
- **FIXED -- the literal pytest name was seeded without an import**:
  `with pytest.raises(...)` counted even in a file that never imports
  pytest (it would NameError). The name is now added only from a real
  `import pytest` binding. Probed (importless file fails; the existing
  import-carrying probes still pass).
- **FIXED -- the with-form skipped arity validation**: `with
  self.assertRaisesRegex(ValueError):` (missing the regex) errors
  before asserting but still counted. With-items now need everything
  but the callable (per-API arity minus one). Probed both sides.
- **WAIVED -- an imported `raises` shadowed by a local fake still
  counts**: importing the real API and then redefining it as a
  non-asserting fake is deliberate evasion, not haste -- tracking
  effective bindings is data-flow analysis a static presence gate does
  not claim (same boundary as the skipped-test waiver). The review
  contract owns adversarial evasion.

The fail-closed counter handler tripped SWALLOWED_EXCEPT on the
scripts lane again (the gate caught itself, second time); accepted
through the documented `--update-baseline` path as deliberate policy.
The refreshed baseline also absorbs new-from-main script entries after
the drift merge.

## Deferred

- Widening sensitive globs per lane (owners decide).
- Trusted-base execution for the maturity-sweep workflows (the row-7
  residual surfaced in review; logged on #1942).
- The row-2 reconciliation gap stays tracked on #1942 (unrelated to
  this slice; listed to show it was considered).

Parked hardening: none.

## Verification

- `python -m pytest tests/test_maturity_sweep.py -q` -- 41 passed
  (23 pre-existing + both-sides probes for every review-fix wave:
  no-raises sensitive module fails with the
  `sensitive-path NO_RAISES_TESTS` reason and passes off-glob;
  with-raises passes on-glob; comment/string mention of pytest.raises
  rejected; `client.raises()` and `client.assertRaises()` helpers
  rejected while `self.assertRaises` passes; local `raises` helper
  rejected while aliased from-import passes; dangling `pytest.raises(X)`
  and `self.assertRaisesRegex(X, "regex")` rejected while the callable
  forms pass; testless sensitive module fails with
  `sensitive-path NO_TEST_FILE` and passes off-glob; stub test file
  counts as testless).
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
| `plans/INDEX.md` | 1 |
| `plans/PR-Negatives-Presence-Gate.md` | 313 |
| `plans/archive/PR-Reddit-Listening-Hardening.md` | 0 |
| `scripts/maturity_sweep.py` | 134 |
| `tests/maturity_sweep/baseline_scripts.json` | 29 |
| `tests/test_maturity_sweep.py` | 859 |
| **Total** | **1342** |
