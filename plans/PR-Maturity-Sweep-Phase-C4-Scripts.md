# PR-Maturity-Sweep-Phase-C4-Scripts

## Why this slice exists

Issue #1689 is extending the maturity-sweep ratchet lane-by-lane. Phase C3
enrolled `extracted_llm_infrastructure`; its deferred follow-up was Phase C4:
enroll repo `scripts/**`, where most audit, CI, migration, backfill, and
operator tooling lives.

Root cause: `scripts/**` is not yet covered by the blocking maturity-sweep
baseline workflow. A PR can add a new brittle script, or new swallowed/bare
exception debt in operational tooling, without the ratchet failing. This fixes
the root for this lane by enrolling `scripts/**` directly and committing its
current baseline.

This slice is expected to exceed the normal 400 LOC target because the
committed baseline snapshots the current debt for hundreds of existing Python
scripts. The behavioral surface is still narrow: one workflow gate, one
baseline, and this plan.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add a blocking maturity-sweep ratchet gate for `scripts/**`.
2. Add the committed scripts baseline so current debt is tracked but new debt
   fails.
3. Add workflow path triggers so PRs touching repo scripts run the gate.
4. Prove the new gate fires for a new script crossing `--min-score 8`.

### Review Contract

Acceptance criteria:
- `pull_request` and `push` path filters include `scripts/**`.
- CI has one blocking Phase C4 step that runs `scripts/maturity_sweep.py` for
  `scripts` with the committed baseline and `--min-score 8`.
- The Phase C4 step marks the whole `scripts/**` lane sensitive for new
  bare/swallowed exceptions.
- The baseline is generated from the current tree with `--update-baseline`.
- Existing maturity-sweep tests still pass.
- A scratch negative proof shows the shipped C4 command fails when a new script
  crosses `--min-score 8`.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- `tests/maturity_sweep/baseline_scripts.json`
- `scripts/**`

Risk areas:
- CI enrollment/path-filter drift.
- Baseline churn that could accept unrelated script debt.
- False-green coverage for operational scripts that mutate data, workflows,
  reports, or CI state.

Reviewer rules triggered: R2, R10, R12, R14.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Phase-C4-Scripts.md`
- `tests/maturity_sweep/baseline_scripts.json`

## Mechanism

The maturity-sweep workflow adds `scripts/**` to PR and main-push path filters,
then runs:

```bash
python scripts/maturity_sweep.py scripts \
  --tests-root tests \
  --baseline tests/maturity_sweep/baseline_scripts.json \
  --min-score 8 \
  --sensitive-glob 'scripts/**'
```

The committed baseline snapshots current per-file scores/counts. Future PRs
fail if a script score increases, a new script crosses the threshold, or a new
bare/swallowed exception appears anywhere under `scripts/**`.

## Intentional

- This does not fix existing scripts brittleness; it ratchets the lane so new
  brittleness cannot land silently.
- Full-lane sensitivity is intentional because scripts include audits,
  migrations, backfills, CI gates, and one-off operational tools.
- This slice does not split script subdirectories into multiple baselines. The
  workflow path filter is repo-wide for `scripts/**`, and one baseline keeps
  Phase C4 reviewable as one enrollment.

## Deferred

- Future maturity-sweep phases, if any, should cover remaining Python lanes
  outside `scripts/**` and the already-enrolled extracted/atlas_brain surfaces.

Parked hardening: none.

## Verification

- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --update-baseline` - pass.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` - pass, 261 files scanned and no new brittleness above baseline.
- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` - pass, 14 passed.
- Scratch negative proof for a new script crossing `--min-score 8` - pass. A
  temporary `scripts/tmp_maturity_sweep_c4_probe.py` scored 11 and the shipped
  C4 command failed with `new file at or above min-score (score 11 >= 8)`.
  The scratch file was removed and the clean C4 command reran successfully.
- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` - pass.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Phase-C4-Scripts.md` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 11 |
| `plans/PR-Maturity-Sweep-Phase-C4-Scripts.md` | 117 |
| `tests/maturity_sweep/baseline_scripts.json` | 1855 |
| **Total** | **1983** |
