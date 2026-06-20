# PR-Maturity-Sweep-Atlas-Brain-B2d

## Why this slice exists

Issue #1689 is rolling the maturity-sweep ratchet across Atlas lane by lane so
new structural brittleness is blocked without making existing debt a big-bang
failure. B2a covered support lanes, B2b covered service/comms lanes, and B2c
covered reasoning/security/storage. This slice extends the same baseline
ratchet to the remaining `atlas_brain` runtime-control surfaces:
`agents`, `capabilities`, and `tools`.

Root cause: these runtime-control lanes contain orchestration, Home Assistant
actions, and tool execution code, but they are not yet included in the
blocking maturity-sweep baseline gate. That means a PR can add new high-score
or sensitive swallowed-exception debt there without tripping the ratchet. This
change fixes the root for these three lanes by enrolling them directly in the
workflow and committing their current baselines.

This approaches the 400 LOC soft cap because the workflow loop needs all three
current baselines to land green; splitting the baselines from the gate would
ship a red or ineffective CI change.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add blocking maturity-sweep ratchet gates for `atlas_brain/agents`,
   `atlas_brain/capabilities`, and `atlas_brain/tools`.
2. Add committed baselines for those lanes so CI fails only on new debt or
   explicit baseline changes.
3. Add path triggers so PRs touching those runtime-control source lanes run
   the maturity-sweep workflow.

### Review Contract

Acceptance criteria:
- `pull_request` and `push` path filters include `atlas_brain/agents/**`,
  `atlas_brain/capabilities/**`, and `atlas_brain/tools/**`.
- CI has one blocking B2d loop that runs `scripts/maturity_sweep.py` for all
  three lanes with the committed baselines and `--min-score 8`.
- The B2d loop carries the same common sensitive globs as prior B2 lanes and
  also marks security-named files/directories sensitive, because this scope
  includes `atlas_brain/agents/graphs/security.py` and
  `atlas_brain/tools/security.py`.
- The new baselines are generated from the current tree with
  `--update-baseline`.
- Existing maturity-sweep tests still pass.
- A scratch negative proof shows the shipped B2d command fails when new
  sensitive swallowed-exception debt is added under a security-named B2d file.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- `tests/maturity_sweep/baseline_atlas_brain_agents.json`
- `tests/maturity_sweep/baseline_atlas_brain_capabilities.json`
- `tests/maturity_sweep/baseline_atlas_brain_tools.json`

Risk areas:
- CI enrollment/path-filter drift.
- False-green sensitive-path coverage for security-named runtime-control code.
- Baseline churn that could accept unrelated debt.

Reviewer rules triggered:
- R2 Test evidence.
- R12 Deployment safety and CI enrollment.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Atlas-Brain-B2d.md`
- `tests/maturity_sweep/baseline_atlas_brain_agents.json`
- `tests/maturity_sweep/baseline_atlas_brain_capabilities.json`
- `tests/maturity_sweep/baseline_atlas_brain_tools.json`

## Mechanism

The workflow adds the three B2d directories to both PR and main-push path
filters, then runs a grouped Bash loop:

```bash
for lane in agents capabilities tools; do
  python scripts/maturity_sweep.py "atlas_brain/${lane}" \
    --tests-root tests \
    --baseline "tests/maturity_sweep/baseline_atlas_brain_${lane}.json" \
    --min-score 8 \
    "${common_sensitive_args[@]}"
done
```

The committed baselines snapshot the current per-file counts for each lane.
That keeps current debt visible but non-blocking, while any file whose score
increases, any new file over the threshold, or any new sensitive-path bare or
swallowed exception fails the gate.

## Intentional

- This does not fix existing brittle files in `agents`, `capabilities`, or
  `tools`; it ratchets them so new brittleness cannot land silently.
- This does not broaden `tests/**`; the broad test trigger already exists from
  earlier maturity-sweep phases so test-side baseline or detector changes keep
  running the workflow.
- Security-named files/directories are treated as sensitive for B2d. The common
  billing/auth/webhook/payment/delete globs do not catch
  `atlas_brain/agents/graphs/security.py` or `atlas_brain/tools/security.py`,
  and those are runtime control surfaces where silent failure should be
  zero-tolerance.

## Deferred

- B2e: enroll `atlas_brain/services/scraping`.
- Phase C: enroll the remaining `extracted_*` packages and `scripts/**` with
  lane-specific baselines.

Parked hardening: none.

## Verification

- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` - pass.
- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` - pass,
  14 passed.
- B2d ratchet loop over `agents capabilities tools` with the shipped
  sensitive globs - pass, all three lanes reported `ratchet gate passed: no
  new brittleness above baseline`.
- Path-filter spot check for `atlas_brain/agents/**`,
  `atlas_brain/capabilities/**`, `atlas_brain/tools/**`, and retained
  `tests/**` - pass.
- Scratch sensitive-path negative proof in `atlas_brain/tools/security.py` -
  pass. A temporary swallowed-exception probe made the shipped tools command
  fail with `score increased (8 -> 13)` and
  `new sensitive-path SWALLOWED_EXCEPT (0 -> 1)`; the scratch code was removed
  and the clean B2d loop was rerun successfully.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Atlas-Brain-B2d.md --check` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 36 |
| `plans/PR-Maturity-Sweep-Atlas-Brain-B2d.md` | 144 |
| `tests/maturity_sweep/baseline_atlas_brain_agents.json` | 83 |
| `tests/maturity_sweep/baseline_atlas_brain_capabilities.json` | 69 |
| `tests/maturity_sweep/baseline_atlas_brain_tools.json` | 80 |
| **Total** | **412** |
