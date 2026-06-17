# PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence

## Why this slice exists

Issue #1689 is extending the maturity-sweep ratchet through Phase C after the
`atlas_brain` rollout. Phase C1 enrolled compact extracted reasoning, quality,
and evidence packages. This slice enrolls `extracted_competitive_intelligence`
by itself because its baseline is large enough that combining it with LLM
infrastructure would exceed the reviewable PR budget.

Root cause: `extracted_competitive_intelligence` contains B2B briefing,
vendor-registry, MCP, storage, and campaign surfaces, but it is not yet
included in the blocking maturity-sweep baseline workflow. A PR can add new
high-score files or new swallowed/bare exception debt there without the
ratchet firing. This fixes the root for this package by enrolling it directly
in the workflow and committing its current baseline.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add a blocking maturity-sweep ratchet gate for
   `extracted_competitive_intelligence`.
2. Add the committed baseline so current debt is tracked but new debt fails.
3. Add path triggers so PRs touching the package run the workflow.

### Review Contract

Acceptance criteria:
- `pull_request` and `push` path filters include
  `extracted_competitive_intelligence/**`.
- CI has one blocking Phase C2 step that runs `scripts/maturity_sweep.py` for
  `extracted_competitive_intelligence` with the committed baseline and
  `--min-score 8`.
- The C2 step marks the whole package sensitive for new bare/swallowed
  exceptions.
- The baseline is generated from the current tree with `--update-baseline`.
- Existing maturity-sweep tests still pass.
- A scratch negative proof shows the shipped C2 command fails when new
  swallowed-exception debt is added under the enrolled package.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json`

Risk areas:
- CI enrollment/path-filter drift.
- False-green coverage for B2B/MCP/storage/campaign surfaces if new swallowed
  exceptions are not treated as sensitive.
- Baseline churn that could accept unrelated debt.

Reviewer rules triggered:
- R2 Test evidence.
- R12 Deployment safety and CI enrollment.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence.md`
- `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json`

## Mechanism

The workflow adds the package directory to PR and main-push path filters, then
runs:

```bash
python scripts/maturity_sweep.py extracted_competitive_intelligence \
  --tests-root tests \
  --baseline tests/maturity_sweep/baseline_extracted_competitive_intelligence.json \
  --min-score 8 \
  --sensitive-glob 'extracted_competitive_intelligence/**'
```

The committed baseline snapshots current per-file counts. Future PRs fail if a
file score increases, a new file crosses the threshold, or a new bare/swallowed
exception appears anywhere in the package.

## Intentional

- This does not fix existing competitive-intelligence brittleness; it ratchets
  the package so new brittleness cannot land silently.
- Full-package sensitivity is intentional because this package includes B2B
  data, MCP, campaign, and storage surfaces.
- `extracted_llm_infrastructure` is deferred to keep this PR under the soft
  diff budget.

## Deferred

- Phase C3: enroll `extracted_llm_infrastructure`.
- Phase C4: enroll `scripts/**` with a script-specific baseline.

Parked hardening: none.

## Verification

- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"` - pass.
- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` - pass, 14 passed.
- C2 ratchet command for `extracted_competitive_intelligence` - pass.
- Path-filter spot check for `extracted_competitive_intelligence/**` and retained `tests/**` - pass.
- Scratch negative proof in `extracted_competitive_intelligence/services/vendor_registry.py` - pass. A temporary swallowed-exception probe failed with `score increased (8 -> 13)` and `new sensitive-path SWALLOWED_EXCEPT (1 -> 2)`; the scratch code was removed and the clean C2 command reran successfully.
- `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence.md --check` - pass.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 11 |
| `plans/PR-Maturity-Sweep-Phase-C2-Competitive-Intelligence.md` | 113 |
| `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json` | 220 |
| **Total** | **344** |
