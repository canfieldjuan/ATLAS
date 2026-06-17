# PR-Maturity-Sweep-Atlas-Brain-B2c

## Why this slice exists

Issue #1689 extends the robust maturity-sweep ratchet from
`extracted_content_pipeline` across the rest of `atlas_brain/**` and the
remaining packages. Phase B1 covered high-blast-radius API/auth/MCP/B2B lanes;
B2a covered support lanes; B2b covered service/comms edge lanes and replaced
fragile per-lane test triggers with `tests/**`.

The remaining `atlas_brain` lanes are the larger/riskier core runtime areas.
This slice takes the highest-severity-per-review-effort subset: reasoning,
security, and storage. These lanes hold decision logic, security monitoring, and
persistence boundaries, so a new swallowed exception or brittle parser in these
paths can hide production failures even though the workflow is green today.

This is intentionally narrower than "all remaining B2" because the
`services/scraping` baseline is parser-heavy and the
`agents`/`capabilities`/`tools` set is another coherent runtime-control group.
Splitting them keeps the generated baselines reviewable, as issue #1689 allows
when a phase gets large.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add baseline-backed blocking maturity-sweep gates for these Phase B2c lanes:
   `atlas_brain/reasoning`, `atlas_brain/security`, and
   `atlas_brain/storage`.
2. Add one committed ratchet baseline per lane under `tests/maturity_sweep/`
   so the workflow starts green and blocks only new debt.
3. Extend the maturity-sweep workflow source triggers for the three lanes.
   Existing `tests/**` coverage from B2b continues to trigger the sweep for any
   test-only change.
4. Prove the gates pass against their committed baselines and that a new
   sensitive-path swallowed exception fails in a scratch copy.

### Review Contract

Acceptance criteria:
- Each B2c lane has a committed lane-specific baseline under
  `tests/maturity_sweep/`.
- `.github/workflows/maturity_sweep_advisory.yml` runs blocking ratchet gates
  for reasoning, security, and storage with no `continue-on-error`.
- The workflow `pull_request` and `push` path filters include every B2c source
  lane; `tests/**` remains present for all test-only changes.
- Existing B1, B2a, and B2b gates and baselines remain intact.
- No `scripts/maturity_sweep.py` detector or ratchet logic changes in this
  slice.

Affected surfaces:
- Maturity sweep GitHub Actions workflow.
- Generated maturity-sweep baseline artifacts for reasoning, security, and
  storage.

Risk areas:
- Workflow path filters that omit a source change or remove the broad
  `tests/**` trigger.
- Baseline paths mismatched to lane paths.
- Sensitive globs missing obvious auth/webhook/billing/payment/invoice/delete
  filenames in reasoning/security/storage paths.

Triggered reviewer rules:
- R1 Requirements match.
- R2 Test evidence.
- R3 Security/auth for sensitive-path gate coverage.
- R6 CI/workflow correctness.
- R9 Generated artifacts/baseline review.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Atlas-Brain-B2c.md`
- `tests/maturity_sweep/baseline_atlas_brain_reasoning.json`
- `tests/maturity_sweep/baseline_atlas_brain_security.json`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`

## Mechanism

The workflow reuses the existing robust maturity-sweep gate from #1690/#1692
and the B2a/B2b rollout. Each B2c lane gets a committed baseline via
`--update-baseline`, then CI runs the same command without `--update-baseline`:

```bash
python scripts/maturity_sweep.py <lane> \
  --tests-root tests \
  --baseline <that lane's committed baseline> \
  --min-score 8 \
  --sensitive-glob ...
```

Existing debt is recorded in the baseline. New files above threshold, score
regressions, or new swallowed/bare exceptions in sensitive paths fail the gate.
The B2c workflow step follows the existing loop pattern with `set -euo pipefail`.
The `security` and `storage` lanes are whole-lane sensitive for new
swallowed/bare exceptions; reasoning keeps the shared sensitive filename globs.

## Intentional

- This is B2c core-risk, not all remaining `atlas_brain/**`. The
  `agents`/`capabilities`/`tools` runtime-control group and
  `services/scraping` parser-heavy group stay deferred so their baselines are
  readable and reviewable.
- No detector changes. This slice only expands workflow coverage and baseline
  artifacts for existing logic.
- Existing debt remains baselined. This PR prevents new brittleness; it does
  not burn down grandfathered entries.

## Deferred

- Issue #1689 B2d: remaining `atlas_brain` runtime-control lanes:
  `agents`, `capabilities`, and `tools`.
- Issue #1689 B2e: parser-heavy `atlas_brain/services/scraping`.
- Issue #1689 Phase C: remaining `extracted_*` packages and `scripts/**`.
- Baseline debt burndown/trend reporting across lanes remains a later
  observability slice.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` -- 14
  passed.
- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"`.
- B2c ratchet loop over `reasoning security storage` -- each lane printed
  `ratchet gate passed: no new brittleness above baseline`.
- Path-filter spot check: `atlas_brain/reasoning/context_aggregator.py`,
  `atlas_brain/security/monitor.py`,
  `atlas_brain/storage/repositories/session.py`, and
  `tests/test_reasoning_context_aggregator.py` are covered by both PR and push
  triggers.
- Scratch sensitive-path proof: temporarily added a swallowed exception to
  `atlas_brain/security/monitor.py`; the security gate exited 1 with
  `score increased (10 -> 15)` and `new sensitive-path SWALLOWED_EXCEPT
  (0 -> 1)`, then the scratch function was removed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 36 |
| `plans/PR-Maturity-Sweep-Atlas-Brain-B2c.md` | 148 |
| `tests/maturity_sweep/baseline_atlas_brain_reasoning.json` | 189 |
| `tests/maturity_sweep/baseline_atlas_brain_security.json` | 63 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 142 |
| **Total** | **578** |
