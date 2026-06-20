# PR-Maturity-Sweep-Phase-C3-LLM-Infrastructure

## Why this slice exists

Issue #1689 is extending the maturity-sweep ratchet through Phase C after the
`atlas_brain` rollout. Phase C1 enrolled compact extracted reasoning, quality,
and evidence packages. Phase C2 enrolled competitive intelligence. This slice
continues the same rollout by enrolling `extracted_llm_infrastructure`, the
shared LLM routing, provider, cache, tracing, and cost substrate.

Root cause: `extracted_llm_infrastructure` is not yet included in the blocking
maturity-sweep baseline workflow. A PR can add new high-score files or new
swallowed/bare exception debt in provider routing, cost, tracing, cache, or
storage code without the ratchet firing. This fixes the root for this package
by enrolling it directly in the workflow and committing its current baseline.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add a blocking maturity-sweep ratchet gate for
   `extracted_llm_infrastructure`.
2. Add the committed baseline so current debt is tracked but new debt fails.
3. Add path triggers so PRs touching the package run the workflow.

### Review Contract

Acceptance criteria:
- `pull_request` and `push` path filters include
  `extracted_llm_infrastructure/**`.
- CI has one blocking Phase C3 step that runs `scripts/maturity_sweep.py` for
  `extracted_llm_infrastructure` with the committed baseline and
  `--min-score 8`.
- The C3 step marks the whole package sensitive for new bare/swallowed
  exceptions.
- The baseline is generated from the current tree with `--update-baseline`.
- Existing maturity-sweep tests still pass.
- A scratch negative proof shows the shipped C3 command fails when new
  swallowed-exception debt is added under the enrolled package.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- `tests/maturity_sweep/baseline_extracted_llm_infrastructure.json`
- `extracted_llm_infrastructure/**`

Risk areas:
- CI enrollment/path-filter drift.
- False-green coverage for provider routing, cache, tracing, cost, and storage
  code if new swallowed exceptions are not treated as sensitive.
- Baseline churn that could accept unrelated debt.

- Reviewer rules triggered: R2, R10, R12, R14.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Phase-C3-LLM-Infrastructure.md`
- `tests/maturity_sweep/baseline_extracted_llm_infrastructure.json`

## Mechanism

The advisory workflow adds the package directory to PR and main-push path
filters, then runs:

```bash
python scripts/maturity_sweep.py extracted_llm_infrastructure \
  --tests-root tests \
  --baseline tests/maturity_sweep/baseline_extracted_llm_infrastructure.json \
  --min-score 8 \
  --sensitive-glob "extracted_llm_infrastructure/**"
```

The committed baseline snapshots current per-file counts. Future PRs fail if a
file score increases, a new file crosses the threshold, or a new bare/swallowed
exception appears anywhere in the package.

## Intentional

- This does not fix existing LLM infrastructure brittleness; it ratchets the
  package so new brittleness cannot land silently.
- Full-package sensitivity is intentional because the package owns provider
  routing, cache, tracing, cost, and storage surfaces where silent failures can
  hide spend, data, or runtime issues.
- No companion product-surface manifest is added in this slice. Unlike C2,
  this package is itself the Python substrate being guarded, not an extracted
  product with adjacent UI/media surfaces.

## Deferred

- Phase C4: enroll `scripts/**` with a script-specific baseline.

Parked hardening: none.

## Verification

- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` - pass.
- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` - pass, 14 passed.
- C3 ratchet command for `extracted_llm_infrastructure` - pass, 41 files scanned and no new brittleness above baseline.
- Path-filter spot check for `extracted_llm_infrastructure/**` and retained `tests/**` - pass.
- Scratch negative proof in `extracted_llm_infrastructure/services/llm/openrouter.py` - pass. A temporary swallowed-exception probe failed with `score increased (23 -> 28)` and `new sensitive-path SWALLOWED_EXCEPT (2 -> 3)`; the scratch code was removed and the clean C3 command reran successfully.
- Scratch negative proof for a new file crossing `--min-score 8` - pass. A
  temporary `extracted_llm_infrastructure/tmp_new_file_min_score_probe.py`
  scored 11 from `NO_TEST_FILE`, `UNGUARDED_INPUT`, and `WEAK_CONTRACT`; the
  shipped C3 command failed with `new file at or above min-score (score 11 >=
  8)`. The scratch file was removed and the clean C3 command reran
  successfully.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Phase-C3-LLM-Infrastructure.md --check` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 11 |
| `plans/PR-Maturity-Sweep-Phase-C3-LLM-Infrastructure.md` | 117 |
| `tests/maturity_sweep/baseline_extracted_llm_infrastructure.json` | 150 |
| **Total** | **278** |
