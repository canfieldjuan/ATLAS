# PR-Maturity-Sweep-Atlas-Brain-B2b

## Why this slice exists

Issue #1689 extends the robust maturity-sweep ratchet from
`extracted_content_pipeline` across the rest of `atlas_brain/**` and the
remaining packages. Phase B1 covered the high-blast-radius API/auth/MCP/B2B
lanes, and Phase B2a covered smaller support lanes. The remaining
service/comms edge lanes still have no baseline-backed gate, so a new brittle
file or a new swallowed exception in those paths can merge without this ratchet
running.

This slice continues Phase B2 with the remaining communication/service edge
lanes that are cohesive enough to review together. It leaves the larger graph,
storage, security, and scraping-heavy lanes for the next slice so their
baselines are not buried in an oversized PR.

The diff is expected to be above the 400 LOC soft cap because this slice adds
generated baseline JSON files plus explicit source/test workflow triggers.
Keeping these edge lanes together gives the reviewer a coherent coverage
boundary while still deferring the largest remaining lanes.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add baseline-backed blocking maturity-sweep gates for these Phase B2b lanes:
   `atlas_brain/comms`, `atlas_brain/services/email_webhooks`,
   `atlas_brain/services/embedding`, `atlas_brain/services/llm`,
   `atlas_brain/services/personaplex`, `atlas_brain/services/speaker_id`,
   `atlas_brain/skills`, `atlas_brain/vision`, and `atlas_brain/voice`.
2. Add one committed ratchet baseline per lane under `tests/maturity_sweep/`
   so the workflow starts green and blocks only new debt.
3. Extend the maturity-sweep workflow triggers to run when those lanes or any
   `tests/**` file changes, so module/provider-named tests cannot bypass the
   gate.
4. Prove the gates pass against their committed baselines and that a new
   sensitive-path swallowed exception fails in a scratch copy.

### Review Contract

Acceptance criteria:
- Each B2b lane has a committed lane-specific baseline under
  `tests/maturity_sweep/`.
- `.github/workflows/maturity_sweep_advisory.yml` runs blocking ratchet gates
  for every B2b lane with no `continue-on-error`.
- The workflow `pull_request` and `push` path filters include every B2b source
  lane and `tests/**` so the gates fire on relevant source changes and all
  test-only changes.
- Existing B1 and B2a gates and baselines remain intact.
- No `scripts/maturity_sweep.py` detector or ratchet logic changes in this
  slice.

Affected surfaces:
- Maturity sweep GitHub Actions workflow.
- Generated maturity-sweep baseline artifacts for B2b edge lanes.

Risk areas:
- Workflow path filters that omit a source change or any test-only change.
- Baseline paths mismatched to lane paths, especially nested
  `atlas_brain/services/*` lanes.
- Sensitive globs missing obvious auth/webhook/billing/payment/invoice/delete
  filenames in service/comms paths.

Triggered reviewer rules:
- R1 Requirements match.
- R2 Test evidence.
- R3 Security/auth for sensitive-path gate coverage.
- R6 CI/workflow correctness.
- R9 Generated artifacts/baseline review.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Atlas-Brain-B2b.md`
- `tests/maturity_sweep/baseline_atlas_brain_comms.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_email_webhooks.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_embedding.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_llm.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_personaplex.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_speaker_id.json`
- `tests/maturity_sweep/baseline_atlas_brain_skills.json`
- `tests/maturity_sweep/baseline_atlas_brain_vision.json`
- `tests/maturity_sweep/baseline_atlas_brain_voice.json`

## Mechanism

The workflow reuses the existing robust maturity-sweep gate from #1690, #1692,
and #1694. Each B2b lane gets a committed baseline via `--update-baseline`, then
CI runs the same command without `--update-baseline`:

```bash
python scripts/maturity_sweep.py <lane> \
  --tests-root tests \
  --baseline <that lane's committed baseline> \
  --min-score 8 \
  --sensitive-glob ...
```

Existing debt is recorded in the baseline. New files above threshold, score
regressions, or new swallowed/bare exceptions in sensitive paths fail the gate.
The B2b workflow step uses the same explicit `set -euo pipefail` pattern added
in B2a.

## Intentional

- This is B2b, not all remaining `atlas_brain/**`. The largest/riskier
  directories (`agents`, `capabilities`, `security`, `storage`, `tools`,
  `reasoning`, and `services/scraping`) stay deferred so their baselines are
  readable and reviewable.
- No detector changes. This slice only expands workflow coverage and baseline
  artifacts for existing logic.
- Existing debt remains baselined. This PR prevents new brittleness; it does
  not burn down grandfathered entries.

## Deferred

- Issue #1689 B2c: remaining large/riskier `atlas_brain` lanes, including
  agents, capabilities, security, storage, tools, reasoning, and
  services/scraping.
- Issue #1689 Phase C: remaining `extracted_*` packages and `scripts/**`.
- Baseline debt burndown/trend reporting across lanes remains a later
  observability slice.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_maturity_sweep.py --noconftest -q` -- 14
  passed.
- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"`.
- B2b ratchet loop over `comms services/email_webhooks services/embedding
  services/llm services/personaplex services/speaker_id skills vision voice`
  -- each lane printed `ratchet gate passed: no new brittleness above
  baseline`.
- Path-filter spot checks: `tests/test_call_intelligence.py`,
  `tests/test_anthropic_convert_messages.py`,
  `tests/test_openrouter_structured_output.py`, `tests/test_cloud_latency.py`,
  `tests/test_email_webhooks.py`, `tests/test_voice.py`, and
  `tests/test_skill_registry.py` are covered by `tests/**` in both PR and push
  filters. This closes the recurring class where module/provider-named tests
  bypassed lane-name globs.
- Scratch sensitive-path proof: temporarily added a swallowed exception to
  `atlas_brain/services/email_webhooks/__init__.py`; the email-webhooks gate
  exited 1 with `new sensitive-path SWALLOWED_EXCEPT (0 -> 1)`, then the
  scratch function was removed. This proof also tightened the B2b sensitive
  globs with `**/*webhook*/**` so `email_webhooks` directory segments are
  actually sensitive.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 166 |
| `plans/PR-Maturity-Sweep-Atlas-Brain-B2b.md` | 167 |
| `tests/maturity_sweep/baseline_atlas_brain_comms.json` | 42 |
| `tests/maturity_sweep/baseline_atlas_brain_services_email_webhooks.json` | 9 |
| `tests/maturity_sweep/baseline_atlas_brain_services_embedding.json` | 8 |
| `tests/maturity_sweep/baseline_atlas_brain_services_llm.json` | 71 |
| `tests/maturity_sweep/baseline_atlas_brain_services_personaplex.json` | 18 |
| `tests/maturity_sweep/baseline_atlas_brain_services_speaker_id.json` | 16 |
| `tests/maturity_sweep/baseline_atlas_brain_skills.json` | 9 |
| `tests/maturity_sweep/baseline_atlas_brain_vision.json` | 17 |
| `tests/maturity_sweep/baseline_atlas_brain_voice.json` | 75 |
| **Total** | **598** |
