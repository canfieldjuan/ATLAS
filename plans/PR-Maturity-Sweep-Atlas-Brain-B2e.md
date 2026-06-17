# PR-Maturity-Sweep-Atlas-Brain-B2e

## Why this slice exists

Issue #1689 is rolling the maturity-sweep ratchet across Atlas lane by lane so
new structural brittleness is blocked without turning existing debt into a
big-bang failure. B2d left one `atlas_brain` lane before Phase C:
`atlas_brain/services/scraping`.

Root cause: the scraping service is parser/admission heavy and currently has no
blocking maturity-sweep baseline gate. A PR can add new parser brittleness,
high-score files, or new swallowed/bare exception debt there without tripping
the ratchet. This fixes the root for this lane by enrolling it directly in the
workflow and committing its current baseline.

This exceeds the 400 LOC soft cap because the current scraping baseline is 269
lines by itself. The slice is still indivisible: landing the workflow without
the baseline would be red, and landing the baseline without the workflow would
not protect the lane.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add a blocking maturity-sweep ratchet gate for
   `atlas_brain/services/scraping`.
2. Add the committed scraping baseline so current debt is tracked but new debt
   fails.
3. Add path triggers so PRs touching scraping source code run the workflow.

### Review Contract

Acceptance criteria:
- `pull_request` and `push` path filters include
  `atlas_brain/services/scraping/**`.
- CI has one blocking B2e step that runs `scripts/maturity_sweep.py` for
  `atlas_brain/services/scraping` with the committed baseline and
  `--min-score 8`.
- The B2e step carries common billing/auth/webhook/payment/delete sensitivity
  and marks `atlas_brain/services/scraping/**` sensitive, because silent parser
  failures in this admission lane should be zero-tolerance for new files too.
- The baseline is generated from the current tree with `--update-baseline`.
- Existing maturity-sweep tests still pass.
- A scratch negative proof shows the shipped B2e command fails when new
  swallowed-exception debt is added under a scraping parser file.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- `tests/maturity_sweep/baseline_atlas_brain_services_scraping.json`

Risk areas:
- CI enrollment/path-filter drift.
- False-green parser/admission coverage if the lane is not full-sensitive.
- Baseline churn that could accept unrelated debt.

Reviewer rules triggered:
- R2 Test evidence.
- R12 Deployment safety and CI enrollment.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Atlas-Brain-B2e.md`
- `tests/maturity_sweep/baseline_atlas_brain_services_scraping.json`

## Mechanism

The workflow adds the scraping directory to both PR and main-push path filters,
then runs the ratchet directly:

```bash
python scripts/maturity_sweep.py atlas_brain/services/scraping \
  --tests-root tests \
  --baseline tests/maturity_sweep/baseline_atlas_brain_services_scraping.json \
  --min-score 8 \
  "${common_sensitive_args[@]}"
```

The committed baseline snapshots current per-file counts. Future PRs fail if a
file score increases, a new file crosses the threshold, or a new bare/swallowed
exception appears anywhere in the scraping lane.

## Intentional

- This does not fix existing scraping parser brittleness; it ratchets the lane
  so new brittleness cannot land silently.
- `atlas_brain/services/scraping/**` is full-sensitive for new bare/swallowed
  exceptions. Scraping is not auth/billing, but it is an evidence-admission
  path where silent parser failure can erase or distort source data.
- This keeps Phase C deferred rather than mixing `extracted_*` and `scripts/**`
  enrollment into the final `atlas_brain` lane.

## Deferred

- Phase C: enroll the remaining `extracted_*` packages and `scripts/**` with
  lane-specific baselines.

Parked hardening: none.

## Verification

- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` - pass.
- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` - pass, 14 passed.
- B2e ratchet command with the shipped sensitive globs - pass.
- Path-filter spot check for `atlas_brain/services/scraping/**` and retained `tests/**` - pass.
- Scratch negative proof in `atlas_brain/services/scraping/parsers/g2.py` - pass. A temporary swallowed-exception probe failed with `score increased (31 -> 36)` and `new sensitive-path SWALLOWED_EXCEPT (4 -> 5)`; the scratch code was removed and the clean B2e command reran successfully.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Atlas-Brain-B2e.md --check` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 27 |
| `plans/PR-Maturity-Sweep-Atlas-Brain-B2e.md` | 118 |
| `tests/maturity_sweep/baseline_atlas_brain_services_scraping.json` | 269 |
| **Total** | **414** |
