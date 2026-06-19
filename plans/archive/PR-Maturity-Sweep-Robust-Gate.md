# PR-Maturity-Sweep-Robust-Gate

Slice phase: Production hardening (CI enforcement)
Ownership lane: ci/maturity-sweep

## Why this slice exists

`maturity_sweep.py` (#1470) already detects the right brittleness -- `SWALLOWED_EXCEPT` (weight 5), `BARE_EXCEPT` (4), `HAPPY_PATH_TESTS` (4), unguarded indexing, brittle/quick-fix comments -- and already has a `--min-score` CI gate and `--json` output. But two things make it advisory in practice:

1. **`.github/workflows/maturity_sweep_advisory.yml` runs it with `continue-on-error: true`** (`python scripts/maturity_sweep.py extracted_content_pipeline --top 25`). So brittleness is printed in logs and the PR still goes green. Reported, not prevented.
2. **`--min-score` is an absolute threshold** (`print_report`: `over = [r for r in results if r.score >= min_score]` -> exit 1). You cannot turn it on across a codebase with existing debt -- every PR touching an already-brittle file floods red. There is no baseline/ratchet, so it can never be made blocking as-is.

This slice adds the missing ratchet and a sensitive-path escalation, then flips the existing lane to blocking. Goal: a NEW swallowed-except can't merge; pre-existing debt doesn't block (it's tracked and burned down deliberately).

This PR is expected to exceed the 400 LOC soft cap because the initial extracted-content baseline is a generated JSON snapshot. The handwritten logic and workflow diff stay scoped to the ratchet mechanism; splitting the baseline away would make the gate red on arrival.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. **Baseline / ratchet mode in `scripts/maturity_sweep.py`.**
   - `--baseline <path>`: JSON baseline of `{relpath: {"score": int, "counts": {code: n}}}` for the swept lane.
   - `--update-baseline`: write/refresh the baseline from the current sweep (the deliberate "accept this debt" action).
   - Gate semantics when `--baseline` is given (replaces the flat `--min-score` behavior, keep `--min-score` as the new-file ceiling): exit 1 when **any** of:
     - a baselined file's `score` **increases** vs baseline, or
     - a **new** (non-baselined) file scores `>= --min-score`, or
     - the sensitive-path rule below fires.
   - On failure, print exactly which files/findings caused it and the one-liner to accept intentionally (`--update-baseline`). Ratchet, not big-bang.

2. **Sensitive-path zero-tolerance.**
   - `--sensitive-glob <glob>` (repeatable) -- e.g. `**/billing/**`, `**/paid*`, `**/auth/**`, `**/webhook*`, `**/payment*`, `**/*deletion*`.
   - For a file matching any sensitive glob, **any new** `SWALLOWED_EXCEPT` or `BARE_EXCEPT` finding (count increases vs baseline, or present on a new file) fails the gate **regardless** of `--min-score` or score delta. Silent failure in money/auth/delete paths is never acceptable as "below threshold."

3. **Commit the initial baseline** at `tests/maturity_sweep/baseline_extracted_content_pipeline.json`, snapshotting current state so the gate starts green and only catches new debt.

4. **Flip the workflow** (`.github/workflows/maturity_sweep_advisory.yml`):
   - keep the existing advisory top-N step (visibility), and
   - add a **blocking** gate step (NO `continue-on-error`) running `maturity_sweep.py extracted_content_pipeline --tests-root tests --baseline tests/maturity_sweep/baseline_extracted_content_pipeline.json --min-score <N> --sensitive-glob ...`.

5. **Tests** (`tests/test_maturity_sweep.py`):
   - `--update-baseline` writes the expected JSON shape.
   - **ratchet:** a pre-existing baselined finding does NOT fail the gate.
   - a NEW finding raising a baselined file's score DOES fail.
   - a new file `>= --min-score` fails; a new file below it passes.
   - **sensitive-path:** a new `SWALLOWED_EXCEPT` on a sensitive-glob file fails even when its score is below `--min-score` and even on an otherwise-passing baseline.
   - updating the baseline to include the new finding makes the gate pass again (deliberate acceptance works).

### Review Contract

Acceptance criteria:
- `scripts/maturity_sweep.py` supports `--baseline`, `--update-baseline`, and repeatable `--sensitive-glob`.
- Baseline mode fails on score increases, new files at or above `--min-score`, and new sensitive `SWALLOWED_EXCEPT` / `BARE_EXCEPT` findings.
- Existing baselined debt does not fail the gate.
- `tests/maturity_sweep/baseline_extracted_content_pipeline.json` is committed and makes the extracted content pipeline gate green on arrival.
- `.github/workflows/maturity_sweep_advisory.yml` keeps the top-N advisory report and adds a blocking baseline-backed gate.

Affected surfaces:
- CI maturity sweep workflow for `extracted_content_pipeline`.
- `scripts/maturity_sweep.py` gate semantics.
- `tests/test_maturity_sweep.py` failure-branch coverage.

Risk areas:
- False red CI from baseline path mismatches or unstable JSON output.
- False green if sensitive path matching or count deltas silently skip findings.
- Baseline churn if output ordering is nondeterministic.

Triggered reviewer rules:
- R1 Requirements match.
- R2 Test evidence.
- R9 Thin-slice scope.
- R10 Evaluator/checker precision.
- R11 Configuration/workflow changes.
- R13 Defect-class proof.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Robust-Gate.md`
- `scripts/maturity_sweep.py`
- `tests/maturity_sweep/baseline_extracted_content_pipeline.json`
- `tests/test_maturity_sweep.py`

## Mechanism

The existing `--min-score` compares an absolute score; the baseline makes the comparison **relative** (per-file delta vs a committed snapshot), which is what lets it run blocking without flooding on existing debt. Sensitive globs add an absolute floor (zero new silent-failure) exactly where it is most dangerous. No change to the detectors or weights -- this is purely the gate/enforcement layer plus a baseline artifact.

## Intentional

- Ratchet, not big-bang: existing debt is baselined and burned down deliberately; only NEW brittleness blocks.
- Sensitive paths get zero-tolerance for swallowed/bare except, independent of score.
- The advisory top-N report stays for visibility alongside the blocking gate.
- `--update-baseline` is the explicit, reviewable escape hatch (a baseline diff in a PR is a visible "we accepted this debt" decision).

## Deferred

- Extending the gate to `atlas_brain/**`, the other `extracted_*` packages, and `scripts/**` -- follow-on slice (depends on this mechanism + per-lane baselines).
- Auto-burndown reporting / trend tracking of baseline debt over time.

Parked hardening: none.

## Verification

- PASS: `python -m pytest tests/test_maturity_sweep.py --noconftest -q` -- 14 passed.
- PASS: `python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --baseline tests/maturity_sweep/baseline_extracted_content_pipeline.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/webhook*' --sensitive-glob '**/payment*' --sensitive-glob '**/*deletion*'`
- PASS: `python -c "import yaml,glob; [yaml.safe_load(open(f, encoding='utf-8')) for f in glob.glob('.github/workflows/*.yml')]"`.
- PASS: scratch sensitive-path proof: clean scratch baseline exits 0; after planting `try/except: pass` in a temporary paid probe file, the same gate exits 1 with `new sensitive-path SWALLOWED_EXCEPT`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 25 |
| `plans/PR-Maturity-Sweep-Robust-Gate.md` | 119 |
| `scripts/maturity_sweep.py` | 163 |
| `tests/maturity_sweep/baseline_extracted_content_pipeline.json` | 906 |
| `tests/test_maturity_sweep.py` | 170 |
| **Total** | **1383** |
