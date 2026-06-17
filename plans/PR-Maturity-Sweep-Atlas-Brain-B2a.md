# PR-Maturity-Sweep-Atlas-Brain-B2a

## Why this slice exists

Issue #1689 extends the robust maturity-sweep ratchet from the initial
`extracted_content_pipeline` lane across the rest of `atlas_brain/**` and the
remaining packages. Phase B1 already covered the high-blast-radius
`atlas_brain` lanes: `api`, `auth`, `autonomous`, `mcp`, and
`services/b2b`. The lower-risk support packages still have advisory-only
coverage, so a new brittle file in those lanes can merge without the baseline
ratchet noticing it.

This slice starts Phase B2 with the small/medium support lanes whose failures
affect runtime support behavior but are not the crown-jewel money/auth/MCP
paths covered in B1. It keeps the rollout reviewable by leaving the largest and
riskier remaining `atlas_brain` directories for B2b/C.

The diff is slightly above the 400 LOC soft cap because this slice adds 14
generated baseline JSON files. Splitting the same support-lane set further
would reduce line count but make the workflow coverage harder to review as one
coherent Phase B2a boundary.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add baseline-backed blocking maturity-sweep gates for these Phase B2a
   support lanes:
   `atlas_brain/alerts`, `atlas_brain/brand`, `atlas_brain/discovery`,
   `atlas_brain/escalation`, `atlas_brain/events`, `atlas_brain/jobs`,
   `atlas_brain/memory`, `atlas_brain/modes`, `atlas_brain/orchestration`,
   `atlas_brain/pipelines`, `atlas_brain/presence`, `atlas_brain/schemas`,
   `atlas_brain/templates`, and `atlas_brain/utils`.
2. Add one committed ratchet baseline per lane under `tests/maturity_sweep/`
   so the workflow starts green and blocks only new debt.
3. Extend the maturity-sweep workflow triggers to run when those lanes or their
   baselines change.
4. Prove the gates pass against their committed baselines and that a new
   sensitive-path swallowed exception fails in a scratch copy.

### Review Contract

Acceptance criteria:
- Each B2a lane has a committed lane-specific baseline under
  `tests/maturity_sweep/`.
- `.github/workflows/maturity_sweep_advisory.yml` runs blocking ratchet gates
  for every B2a lane with no `continue-on-error`.
- The workflow `pull_request` and `push` path filters include every B2a source
  lane and baseline so the gates fire on relevant changes.
- Existing B1 gates and baselines remain intact.
- No `scripts/maturity_sweep.py` detector or ratchet logic changes in this
  slice.

Affected surfaces:
- Maturity sweep GitHub Actions workflow.
- Generated maturity-sweep baseline artifacts for B2a support lanes.

Risk areas:
- Workflow path filters that accidentally omit a lane, turning a gate into a
  no-op on source changes.
- Baseline paths mismatched to lane paths.
- Over-broad sensitive globs that make the lane red on arrival, or under-broad
  globs that miss obvious auth/webhook/billing/delete filenames inside support
  lanes.

Triggered reviewer rules:
- R1 Requirements match.
- R2 Test evidence.
- R3 Security/auth for sensitive-path gate coverage.
- R6 CI/workflow correctness.
- R9 Generated artifacts/baseline review.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Atlas-Brain-B2a.md`
- `tests/maturity_sweep/baseline_atlas_brain_alerts.json`
- `tests/maturity_sweep/baseline_atlas_brain_brand.json`
- `tests/maturity_sweep/baseline_atlas_brain_discovery.json`
- `tests/maturity_sweep/baseline_atlas_brain_escalation.json`
- `tests/maturity_sweep/baseline_atlas_brain_events.json`
- `tests/maturity_sweep/baseline_atlas_brain_jobs.json`
- `tests/maturity_sweep/baseline_atlas_brain_memory.json`
- `tests/maturity_sweep/baseline_atlas_brain_modes.json`
- `tests/maturity_sweep/baseline_atlas_brain_orchestration.json`
- `tests/maturity_sweep/baseline_atlas_brain_pipelines.json`
- `tests/maturity_sweep/baseline_atlas_brain_presence.json`
- `tests/maturity_sweep/baseline_atlas_brain_schemas.json`
- `tests/maturity_sweep/baseline_atlas_brain_templates.json`
- `tests/maturity_sweep/baseline_atlas_brain_utils.json`

## Mechanism

The workflow reuses the existing robust maturity-sweep gate exactly as shipped
in #1690 and extended in #1692:

```bash
python scripts/maturity_sweep.py <lane> \
  --tests-root tests \
  --baseline <that lane's committed baseline> \
  --min-score 8 \
  --sensitive-glob ...
```

Each B2a lane gets its own baseline via `--update-baseline`, then CI runs the
same command without `--update-baseline`. Existing debt is tracked in the
baseline; new score regressions, new files above threshold, or new
swallowed/bare exceptions in sensitive paths fail the job.

Sensitive globs stay conservative but lane-appropriate. General support lanes
reuse the billing/auth/webhook/payment/invoice/deletion filename globs. The
support lanes are not marked full-lane sensitive because B1 already reserved
that treatment for the auth and MCP crown-jewel lanes.

## Intentional

- This is B2a, not all of B2. The largest/riskier remaining directories
  (`atlas_brain/reasoning`, `atlas_brain/storage`, `atlas_brain/tools`,
  `atlas_brain/capabilities`, `atlas_brain/security`, `atlas_brain/comms`,
  `atlas_brain/voice`, and non-B2a `atlas_brain/services/**`) are deferred so
  their baselines are reviewable.
- No detector changes. This slice only expands workflow coverage and baseline
  artifacts for existing logic.
- Existing debt remains baselined. This PR prevents new brittleness; it does
  not burn down grandfathered entries.

## Deferred

- Issue #1689 B2b: remaining larger/riskier `atlas_brain` lanes, including
  reasoning, storage, tools, capabilities, security, comms, voice, and service
  subpackages not covered here.
- Issue #1689 Phase C: remaining `extracted_*` packages and `scripts/**`.
- Baseline debt burndown/trend reporting across lanes remains a later
  observability slice.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` -- 14
  passed.
- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"`.
- B2a ratchet loop over `alerts brand discovery escalation events jobs memory
  modes orchestration pipelines presence schemas templates utils` -- each lane
  printed `ratchet gate passed: no new brittleness above baseline`.
- Scratch sensitive-path proof: temporarily added a swallowed exception to
  `atlas_brain/templates/email/invoice.py`; the templates gate exited 1 with
  `new sensitive-path SWALLOWED_EXCEPT (0 -> 1)`, then the scratch function was
  removed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 81 |
| `plans/PR-Maturity-Sweep-Atlas-Brain-B2a.md` | 173 |
| `tests/maturity_sweep/baseline_atlas_brain_alerts.json` | 21 |
| `tests/maturity_sweep/baseline_atlas_brain_brand.json` | 8 |
| `tests/maturity_sweep/baseline_atlas_brain_discovery.json` | 23 |
| `tests/maturity_sweep/baseline_atlas_brain_escalation.json` | 8 |
| `tests/maturity_sweep/baseline_atlas_brain_events.json` | 8 |
| `tests/maturity_sweep/baseline_atlas_brain_jobs.json` | 21 |
| `tests/maturity_sweep/baseline_atlas_brain_memory.json` | 34 |
| `tests/maturity_sweep/baseline_atlas_brain_modes.json` | 15 |
| `tests/maturity_sweep/baseline_atlas_brain_orchestration.json` | 8 |
| `tests/maturity_sweep/baseline_atlas_brain_pipelines.json` | 22 |
| `tests/maturity_sweep/baseline_atlas_brain_presence.json` | 15 |
| `tests/maturity_sweep/baseline_atlas_brain_schemas.json` | 1 |
| `tests/maturity_sweep/baseline_atlas_brain_templates.json` | 37 |
| `tests/maturity_sweep/baseline_atlas_brain_utils.json` | 21 |
| **Total** | **496** |
