# PR-Retired-Failure-Mode-Detector

## Why this slice exists

Issue #1964 starts the retired failure-mode recurrence ledger. Before any
scheduled collector or issue-posting workflow exists, the arc needs a cheap
detector that can emit stable JSON for likely recurrences. This slice builds only
that detector and its tests, so later ledger slices record a real signal instead
of a placeholder.

This is over the 400 LOC soft cap because the schema, the four initial detector
modes, and their adversarial git-diff fixtures need to land together for the
JSON signal to be meaningful. The slice stays indivisible by deferring all
workflow, comment, label, and ledger-writing behavior.

## Scope (this PR)

Ownership lane: workflow/retired-failure-detectors
Slice phase: Workflow/process

1. Add a non-blocking CLI that reads a git diff and emits advisory recurrence
   signals for retired autonomous-coding failure modes.
2. Cover the initial retired modes from #1964: plan weakening, test weakening,
   scope drift, and symptom patching.
3. Keep GitHub writes, scheduled collection, sticky comments, labels, and merge
   blocking deferred.

### Review Contract

- Acceptance criteria:
  - [ ] CLI emits schema-versioned JSON with `signal_type`,
        `detector_version`, `head_sha`, and `signals`.
  - [ ] Signals include `mode`, `signature`, `confidence`, `paths`,
        `evidence`, and `explanation`.
  - [ ] Findings are advisory: signals do not make the CLI exit non-zero.
  - [ ] Each initial retired mode has at least one regression fixture.
  - [ ] A clean planned diff emits no signals.
- Affected surfaces: developer tooling / advisory detector scripts / tests.
- Risk areas: false positives, detector rot, future ledger noise.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `plans/PR-Retired-Failure-Mode-Detector.md`
- `scripts/detect_retired_failure_modes.py`
- `tests/test_detect_retired_failure_modes.py`

## Mechanism

`scripts/detect_retired_failure_modes.py` resolves the merge-base against a
caller-provided base ref, inspects changed paths and line-level diffs, and emits
a report with `schema_version: 1` and `signal_type:
retired_failure_recurrence`.

The detector signatures are intentionally cheap:

- `plan_weakening`: obligation-like plan lines are removed while production code
  changes.
- `test_weakening`: test assertions/cases are removed or skip-like markers are
  added while production code changes.
- `scope_drift`: code changes lack a visible slice plan, or changed files are
  outside the plan's `Files touched`.
- `symptom_patching`: fix-type plans lack root-cause language or name root cause
  while changing only downstream-looking files.

The CLI exits `0` when signals are found and can write JSON via `--json-out` for
the future workflow artifact/ledger collector.

## Intentional

- Advisory only. This PR does not add a required check, PR label, GitHub comment,
  or issue write.
- Heuristic signatures are allowed to be somewhat noisy because the ledger is
  for recurrence review, not merge blocking.
- `code_change_without_plan_doc` is low-confidence because trivial and
  Dependabot-style changes can be valid exceptions.

## Deferred

- Slice 2: advisory PR workflow that runs this detector and uploads the JSON
  artifact.
- Slice 3: scheduled/manual collector that posts grouped detector signals to
  #1964.
- Slice 4: disposition documentation for true recurrence / false positive /
  needs guard / needs detector tuning.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_detect_retired_failure_modes.py -q` — 6 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Retired-Failure-Mode-Detector.md` | 100 |
| `scripts/detect_retired_failure_modes.py` | 387 |
| `tests/test_detect_retired_failure_modes.py` | 283 |
| **Total** | **770** |
