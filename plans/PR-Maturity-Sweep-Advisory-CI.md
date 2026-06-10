# PR-Maturity-Sweep-Advisory-CI

## Why this slice exists

Adds a brittleness-triage tool (`maturity_sweep`) to CI as an **advisory,
non-blocking** check. It ranks content-ops files by structural-fragility
signals (untested modules, happy-path-only tests, swallowed exceptions,
unguarded input) and prints a worst-first agenda for an adversarial pass --
systematizing the triage step the lane's review already relies on.

It is explicitly **not** a production-hardening gate. It catches *structural*
brittleness only; semantic / contract / logic bugs (over-broad matchers,
fail-open fields, asymmetric token handling -- the actual recent BLOCKERs)
remain the job of adversarial review + the live AI-reconciliation gate. Wired
advisory-first per the operator decision, to avoid false-positive blocking and
to avoid a green-sweep-equals-hardened false signal.

## Scope (this PR)

Ownership lane: review-workflow
Slice phase: workflow/process

1. Add `scripts/maturity_sweep.py` -- a stdlib-only AST brittleness scorer
   (operator-provided), plus a small `--tests-root` flag so it can index tests
   from a separate directory (the repo `tests/` dir lives outside the swept
   lane).
2. Add `.github/workflows/maturity_sweep_advisory.yml` -- runs the sweep on
   `extracted_content_pipeline` with `--tests-root tests`, top 25, non-blocking
   (`continue-on-error` + no `--min-score`). Triggers on PRs touching the
   lane / tool / its tests.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Advisory-CI.md`
- `scripts/maturity_sweep.py`

### Review Contract
- Acceptance: the sweep runs in CI on lane PRs, prints a worst-first agenda,
  and never fails the build. Test correlation uses `tests/` (no false
  `NO_TEST_FILE` on files actually tested there). ASCII-clean. stdlib-only
  (no dependency install).
- Affected surfaces: CI only -- a new advisory workflow + a new script. No
  runtime or product code is touched.
- Risk areas: a noisy or blocking gate (mitigated: advisory; no `--min-score`;
  `continue-on-error`); a false "hardening" signal (mitigated: explicit banner
  in the workflow + this plan).
- Reviewer rules triggered: R8 (determinism -- the tool is deterministic
  static analysis). R2 test-evidence is N/A (tooling, not product logic).

## Mechanism

`maturity_sweep` walks a lane for Python source files, AST-parses each, and scores
structural smells (`SWALLOWED_EXCEPT`, `BARE_EXCEPT`, `UNGUARDED_INPUT` on
boundary calls, `UNGUARDED_INDEX`, `ASSERT_AS_VALIDATION`, `MUTABLE_DEFAULT`,
`WEAK_CONTRACT`, `HEURISTIC_COMMENT`) plus test-coverage smells
(`NO_TEST_FILE`, `HAPPY_PATH_TESTS`, `NO_RAISES_TESTS`), with per-code caps so
one noisy category cannot dominate, sorted worst-first. The new `--tests-root`
points test indexing at the repo `tests/` dir so coverage signals are computed
against the real tests (which live outside the lane). Without `--min-score`
the tool exits 0, so the CI step is advisory by construction; `continue-on-error`
is a safety net.

## Intentional
- Advisory, not gating. A hard `--min-score` gate would block on false
  positives (legit `except ...: return` guards flagged as `SWALLOWED_EXCEPT`;
  safe `x[0]` as `UNGUARDED_INDEX`) and on the large pre-existing debt.
  Operator chose advisory-first.
- Scoped to `extracted_content_pipeline` (the active lane), not repo-wide: a
  repo-root scan hits ~53k vendored Python files and is too slow/noisy for CI.
- The workflow banner states plainly that a green sweep is not a hardening
  stamp and that semantic bugs are out of scope -- to head off the
  green-CI-equals-good misread.
- The tool is committed near-verbatim (operator-provided); the only logic
  delta is the ~6-line `--tests-root` wiring. stdlib-only, so no requirements
  install in CI.

## Deferred
- Per-module `HAPPY_PATH_TESTS` / `NO_RAISES_TESTS` matching is under-reported
  because Atlas tests are named test_extracted_&lt;module&gt; while the tool
  matches test_&lt;module&gt;; `NO_TEST_FILE` is already correct via the
  "mentioned anywhere" fallback. Tightening the stem match (handle the
  `extracted_` prefix) is a follow-up.
- Broadening scope beyond `extracted_content_pipeline` (e.g. `atlas_brain`
  content-ops surfaces) -- follow-up.
- Promoting to a regression-gate (fail only when a changed file's score
  increases vs main, ignoring pre-existing debt) once a baseline exists --
  follow-up.
- Reducing `SWALLOWED_EXCEPT` false positives (distinguish legit early-return
  guards) -- follow-up.
- Parked hardening: none.

## Verification
- `python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --top 25`
  runs in ~8s, exits 0, prints a worst-first agenda.
- Tested deflection files no longer show a false `NO_TEST_FILE`
  (`faq_deflection_report`, `support_ticket_input_package` -> `NO_TEST_FILE`
  absent).
- Running `scripts/check_ascii_python.sh` passes (`scripts/maturity_sweep.py`
  is ASCII).
- Workflow is non-blocking by construction: no `--min-score`, plus
  `continue-on-error: true`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 51 |
| `plans/PR-Maturity-Sweep-Advisory-CI.md` | 111 |
| `scripts/maturity_sweep.py` | 467 |
| **Total** | **629** |
