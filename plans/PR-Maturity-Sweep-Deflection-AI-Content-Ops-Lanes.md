# PR-Maturity-Sweep-Deflection-AI-Content-Ops-Lanes

## Why this slice exists

Issue #1689's Phase C1 extracted-core gate enrolled shared reasoning,
quality, and evidence packages. The existing maturity-sweep workflow already
ratchets `extracted_content_pipeline/**`, but root Atlas product wrappers and
scripted proof helpers for deflection and AI content ops can still change
without a focused lane baseline.

This slice widens the maturity-sweep blast radius to those product lanes
without sweeping all of `atlas_brain` or all repo scripts at once.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add `scripts/maturity_sweep_file_lane.py`, a thin wrapper that sweeps
   explicit Python files with the existing maturity-sweep internals.
2. Add a focused companion workflow with path triggers for Atlas-side
   `content_ops` and `deflection` Python files under `atlas_brain/**` and
   `scripts/**`.
3. Add a blocking deflection lane ratchet over deflection Python files in
   `atlas_brain`, `scripts`, and `extracted_content_pipeline`.
4. Add a blocking AI content-ops lane ratchet over non-deflection
   `content_ops` Python files in `atlas_brain` and `scripts`.
5. Add a blocking manifest check for the deployed/non-Python deflection
   product surface: portfolio UI pages/API routes, Vercel rewrite, frontend
   contracts/examples, atlas-intel report UI model/test, and deflection SQL
   migrations.
6. Commit baselines for both focused Python lanes.

### Review Contract

Acceptance criteria:
- The workflow runs when Atlas-side or script-side deflection/content-ops
  Python files change.
- Deflection-specific files are gated with payment/auth/webhook/billing
  sensitive globs plus a lane-wide deflection sensitive glob.
- Non-deflection AI content-ops files are gated as a focused content-ops lane.
- The new file-list wrapper deduplicates explicit files and leaves the
  existing directory sweep behavior untouched.
- Baseline paths are normalized so Windows-generated baselines compare cleanly
  in Ubuntu CI.
- The production product surface is explicitly listed and changes fail unless
  the manifest is updated intentionally.

Affected surfaces:
- `.github/workflows/maturity_sweep_deflection_content_ops.yml`
- `scripts/check_deflection_product_surface_manifest.py`
- `scripts/maturity_sweep_file_lane.py`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/maturity_sweep/baseline_deflection_lane.json`
- `tests/maturity_sweep/baseline_ai_content_ops_lane.json`

Risk areas:
- CI enrollment/path-filter drift.
- Accidental overreach into unrelated root Atlas modules.
- Platform-specific path separators in generated baselines.

Reviewer rules triggered:
- R2 Test evidence.
- R12 Deployment safety and CI enrollment.
- R14 Codebase verification.

## Intentional

- `extracted_content_pipeline/**` remains covered by its existing broad gate;
  the new deflection lane includes deflection-specific files there so the
  product slice can be reviewed as one lane.
- The non-Python production product surface is a manifest/enrollment check, not
  a structural Python maturity score.
- The AI content-ops lane excludes deflection-named files so it does not
  double-own the deflection-specific gate.
- This does not enroll every script or every root `atlas_brain` module; those
  remain separate slices if needed.

## Deferred

- Full `scripts/**` maturity-sweep enrollment remains a larger script-specific
  baseline slice.

Parked hardening: none.

## Verification

- `python scripts/maturity_sweep_file_lane.py <deflection files> --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --update-baseline` - pass, 21 files.
- `python scripts/maturity_sweep_file_lane.py <ai content ops files> --tests-root tests --baseline tests/maturity_sweep/baseline_ai_content_ops_lane.json --update-baseline` - pass, 68 files.
- `python scripts/check_deflection_product_surface_manifest.py` - pass, 30 files.
- `python -m py_compile scripts/maturity_sweep_file_lane.py` - pass.
- `python -m py_compile scripts/check_deflection_product_surface_manifest.py` - pass.
- Deflection lane ratchet command - pass.
- AI content-ops lane ratchet command - pass.
