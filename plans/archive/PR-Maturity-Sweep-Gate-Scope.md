# PR-Maturity-Sweep-Gate-Scope

## Why this slice exists

Issue #1689 asks to extend the robust maturity-sweep ratchet from #1690 beyond
`extracted_content_pipeline`. The highest-risk uncovered surface is
`atlas_brain`: billing, auth, MCP servers, webhooks, autonomous tasks, and B2B
services can still add swallowed exceptions or other brittle patterns without a
blocking ratchet.

This PR implements Phase B1 only: high-blast-radius `atlas_brain` lanes first.
It intentionally does not change the detector logic from #1690; it adds
per-lane baselines, path triggers, and blocking workflow steps using the
existing baseline/sensitive-path gate.

This PR exceeds the 400 LOC soft cap because it commits five generated baseline
JSON files so the new gates are green on arrival. Splitting a baseline away from
its workflow step would make the gate fail immediately.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add committed maturity-sweep baselines for Phase B1 lanes:
   `atlas_brain/api`, `atlas_brain/auth`, `atlas_brain/autonomous`,
   `atlas_brain/mcp`, and `atlas_brain/services/b2b`.
2. Add path triggers for those lanes and their baseline files.
3. Add one blocking ratchet gate step per B1 lane.
4. Mark auth and MCP lanes fully sensitive, and mark billing/auth/webhook/
   payment/invoicing/deletion paths sensitive in API, autonomous, and B2B
   service lanes.

### Review Contract

Acceptance criteria:
- The workflow still runs the #1690 extracted-content gate.
- Each B1 lane has a committed baseline under `tests/maturity_sweep/`.
- Each B1 lane has a blocking workflow step with `--baseline` and
  `--min-score 8`.
- Sensitive globs cover auth, MCP, billing, paid, webhook, payment, invoicing,
  invoice, and deletion paths as appropriate to each lane.
- All baseline-backed gates exit 0 against the committed baselines.

Affected surfaces:
- `.github/workflows/maturity_sweep_advisory.yml`
- The five Phase B1 baseline files listed below.

Risk areas:
- False red CI from baseline path drift.
- False green if a high-risk path is omitted from sensitive globs.
- Workflow noise from path triggers that are too broad or too narrow.

Triggered reviewer rules:
- R1 Requirements match.
- R2 Test evidence.
- R9 Thin-slice scope.
- R11 Workflow/config changes.
- R14 Codebase verification.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `plans/PR-Maturity-Sweep-Gate-Scope.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/maturity_sweep/baseline_atlas_brain_auth.json`
- `tests/maturity_sweep/baseline_atlas_brain_autonomous.json`
- `tests/maturity_sweep/baseline_atlas_brain_mcp.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_b2b.json`

## Mechanism

This slice reuses the #1690 ratchet unchanged. For each B1 lane, the workflow
runs:

```text
python scripts/maturity_sweep.py <lane> --tests-root tests \
  --baseline tests/maturity_sweep/baseline_<lane>.json \
  --min-score 8 --sensitive-glob ...
```

The committed baseline accepts existing debt for that lane. Future PRs fail
when a baselined file's score increases, a new file crosses `--min-score 8`, or
a new `SWALLOWED_EXCEPT` / `BARE_EXCEPT` appears on a sensitive path.

## Intentional

- Phase B1 only. This keeps the PR reviewable and leaves lower-risk
  `atlas_brain` areas plus remaining packages/scripts for later phases.
- Auth and MCP use `--sensitive-glob '**/*'` because silent failure anywhere in
  those lanes is high risk.
- API, autonomous, and B2B services use targeted sensitive globs for billing,
  paid, auth, webhook, payment, invoicing, invoice, and deletion paths.
- No detector or gate-logic changes; #1690 owns the mechanism.

## Deferred

- Phase B2: remaining lower-risk `atlas_brain/**` lanes.
- Phase C: remaining `extracted_*` packages and `scripts/**`.
- Baseline debt burndown/trend reporting and future threshold tightening.

Parked hardening: none.

## Verification

- PASS: `python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --baseline tests/maturity_sweep/baseline_extracted_content_pipeline.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/webhook*' --sensitive-glob '**/payment*' --sensitive-glob '**/*deletion*'`
- PASS: `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --min-score 8 ...`
- PASS: `python scripts/maturity_sweep.py atlas_brain/auth --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_auth.json --min-score 8 --sensitive-glob '**/*'`
- PASS: `python scripts/maturity_sweep.py atlas_brain/autonomous --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_autonomous.json --min-score 8 ...`
- PASS: `python scripts/maturity_sweep.py atlas_brain/mcp --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_mcp.json --min-score 8 --sensitive-glob '**/*'`
- PASS: `python scripts/maturity_sweep.py atlas_brain/services/b2b --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_services_b2b.json --min-score 8 ...`
- PASS: `python -m pytest tests/test_maturity_sweep.py --noconftest -q` -- 14 passed.
- PASS: `python -c "import yaml,glob; [yaml.safe_load(open(f, encoding='utf-8')) for f in glob.glob('.github/workflows/*.yml')]"`.
- PASS: scratch sensitive-path proof: clean scratch `atlas_brain/api` baseline exits 0; after adding one swallowed exception to copied `billing.py`, the same gate exits 1 with `new sensitive-path SWALLOWED_EXCEPT (4 -> 5)`.
- PASS: `python scripts/sync_pr_plan.py plans/PR-Maturity-Sweep-Gate-Scope.md --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 92 |
| `plans/PR-Maturity-Sweep-Gate-Scope.md` | 128 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 313 |
| `tests/maturity_sweep/baseline_atlas_brain_auth.json` | 33 |
| `tests/maturity_sweep/baseline_atlas_brain_autonomous.json` | 652 |
| `tests/maturity_sweep/baseline_atlas_brain_mcp.json` | 178 |
| `tests/maturity_sweep/baseline_atlas_brain_services_b2b.json` | 279 |
| **Total** | **1675** |
