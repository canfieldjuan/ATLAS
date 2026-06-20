# PR-Maturity-Sweep-Phase-C1-Extracted-Core

## Why this slice exists

Issue #1689 has completed the `atlas_brain` maturity-sweep rollout through
B2e. Phase C now extends the same baseline ratchet to extracted packages and
repo scripts. This first Phase C slice enrolls the compact extracted
reasoning, quality, and evidence-schema packages before the larger extracted
infrastructure packages and `scripts/**`.

Root cause: `extracted_reasoning_core`, `extracted_quality_gate`, and
`extracted_evidence_to_story` are standalone safety/evidence surfaces, but they
are not yet included in the blocking maturity-sweep baseline workflow. A PR can
add new high-score files or new swallowed/bare exception debt there without the
ratchet firing. This fixes the root for these three packages by enrolling them
directly in the workflow and committing their current baselines.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add blocking maturity-sweep ratchet gates for
   `extracted_reasoning_core`, `extracted_quality_gate`, and
   `extracted_evidence_to_story`.
2. Add committed baselines for those packages so current debt is tracked but
   new debt fails.
3. Add path triggers so PRs touching those extracted packages run the workflow.

### Review Contract

Acceptance criteria:
- `pull_request` and `push` path filters include the three C1 extracted package
  directories.
- CI has one blocking Phase C1 loop that runs `scripts/maturity_sweep.py` for
  all three packages with committed baselines and `--min-score 8`.
- The C1 loop marks each enrolled package full-sensitive for new bare/swallowed
  exceptions.
- The baselines are generated from the current tree with `--update-baseline`.
- Existing maturity-sweep tests still pass.
- A scratch negative proof shows the shipped C1 command fails when new
  swallowed-exception debt is added under an enrolled package.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- `tests/maturity_sweep/baseline_extracted_reasoning_core.json`
- `tests/maturity_sweep/baseline_extracted_quality_gate.json`
- `tests/maturity_sweep/baseline_extracted_evidence_to_story.json`

Risk areas:
- CI enrollment/path-filter drift.
- False-green safety/evidence package coverage if new swallowed exceptions are
  not treated as sensitive.
- Baseline churn that could accept unrelated debt.

Reviewer rules triggered:
- R2 Test evidence.
- R12 Deployment safety and CI enrollment.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Phase-C1-Extracted-Core.md`
- `tests/maturity_sweep/baseline_extracted_evidence_to_story.json`
- `tests/maturity_sweep/baseline_extracted_quality_gate.json`
- `tests/maturity_sweep/baseline_extracted_reasoning_core.json`

## Mechanism

The workflow adds the three package directories to PR and main-push path
filters, then runs a grouped Bash loop:

```bash
for lane in extracted_reasoning_core extracted_quality_gate extracted_evidence_to_story; do
  python scripts/maturity_sweep.py "${lane}" \
    --tests-root tests \
    --baseline "tests/maturity_sweep/baseline_${lane}.json" \
    --min-score 8 \
    --sensitive-glob "${lane}/**"
done
```

The committed baselines snapshot current per-file counts. Future PRs fail if a
file score increases, a new file crosses the threshold, or a new bare/swallowed
exception appears anywhere in one of the C1 packages.

## Intentional

- This does not fix existing extracted package brittleness; it ratchets these
  packages so new brittleness cannot land silently.
- Full-lane sensitivity is intentional for these packages because they own
  reasoning, quality gates, and evidence schema boundaries.
- This leaves larger Phase C surfaces for follow-up slices instead of mixing
  all extracted packages and `scripts/**` into one oversized PR.

## Deferred

- Phase C2+: enroll `extracted_competitive_intelligence`,
  `extracted_llm_infrastructure`, and `scripts/**` with lane-specific
  baselines.

Parked hardening: none.

## Verification

- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` - pass.
- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` - pass, 14 passed.
- C1 ratchet loop over the three extracted packages - pass.
- Path-filter spot check for the three package triggers and retained `tests/**` - pass.
- Scratch negative proof in `extracted_quality_gate/evidence_pack.py` - pass. A temporary swallowed-exception probe failed with `score increased (5 -> 10)` and `new sensitive-path SWALLOWED_EXCEPT (1 -> 2)`; the scratch code was removed and the clean C1 loop reran successfully.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Phase-C1-Extracted-Core.md --check` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 19 |
| `plans/PR-Maturity-Sweep-Phase-C1-Extracted-Core.md` | 123 |
| `tests/maturity_sweep/baseline_extracted_evidence_to_story.json` | 8 |
| `tests/maturity_sweep/baseline_extracted_quality_gate.json` | 66 |
| `tests/maturity_sweep/baseline_extracted_reasoning_core.json` | 98 |
| **Total** | **314** |
