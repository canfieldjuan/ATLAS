# PR-Maturity-Sweep-Deflection-AI-Content-Ops-Lanes

## Why this slice exists

Issue #1689's Phase C1 extracted-core gate enrolled shared reasoning,
quality, and evidence packages. The existing maturity-sweep workflow already
ratchets `extracted_content_pipeline/**`, but root Atlas product wrappers and
scripted proof helpers for deflection and AI content ops can still change
without a focused lane baseline.

This slice widens the maturity-sweep blast radius to those product lanes
without sweeping all of `atlas_brain` or all repo scripts at once.

The over-budget size is intentional and indivisible: the workflow, two checker
scripts, fixture tests, generated baselines, and explicit product-surface
manifest must land together so the new gates can run, fail closed, and avoid a
temporary enforcement/baseline gap.

This PR also carries a narrow CI unblock for `Security Guardrails`: the OSV
reusable workflow job requested `actions: read` but the caller job did not grant
it, causing startup failure before any job logs existed. The PR Gitleaks job is
also scoped to the PR head range (`origin/<base>..HEAD`) so this PR is not held
red by trusted-base findings that require a separate baseline-rotation PR.
Scheduled and main-push security scans still keep their full-history behavior.

## Scope (this PR)

Ownership lane: ci/maturity-sweep
Slice phase: Production hardening

1. Add `scripts/maturity_sweep_file_lane.py`, a thin wrapper that sweeps
   explicit Python files with the existing maturity-sweep internals.
2. Add a focused companion workflow with path triggers for Atlas-side
   `content_ops` and `deflection` Python files under `atlas_brain/**` and
   `scripts/**`.
3. Add a blocking deflection lane ratchet over deflection Python files in
   `atlas_brain`, `scripts`, and deflection-named `extracted_content_pipeline`
   files.
4. Add a blocking AI content-ops lane ratchet over non-deflection
   `content_ops` Python files in `atlas_brain` and `scripts`.
5. Add a blocking manifest check for the deployed/non-Python deflection
   product surface: portfolio UI pages/API routes, route wiring, service link,
   Vercel rewrite, frontend contracts/examples, atlas-intel report UI
   model/test, and deflection SQL migrations.
6. Commit baselines for both focused Python lanes.
7. Add fixture tests for the explicit file-list wrapper and product-surface
   manifest checker.
8. Add the missing read-only `actions: read` permission to the trusted OSV
   reusable workflow caller and scope pull-request Gitleaks scanning to PR
   commits from the PR head checkout.

### Review Contract

Acceptance criteria:
- The workflow runs when Atlas-side or script-side deflection/content-ops
  Python files change.
- Deflection-specific files are gated with payment/auth/webhook/billing
  sensitive globs plus a lane-wide deflection sensitive glob.
- Non-deflection AI content-ops files are gated as a focused content-ops lane.
- The new file-list wrapper deduplicates explicit files, rejects missing or
  non-Python explicit inputs, and leaves the existing directory sweep behavior
  untouched.
- Baseline paths are normalized so Windows-generated baselines compare cleanly
  in Ubuntu CI.
- The production product surface is explicitly listed and changes fail unless
  the manifest is updated intentionally.
- Deflection route wiring and service-page links are included in the workflow
  filters and manifest guard.
- Product-surface API discovery is recursive so nested route files cannot bypass
  the manifest guard.
- Fixture tests cover missing/untracked manifest files and explicit lane
  failure modes.
- Security Guardrails no longer fails at workflow startup on the OSV reusable
  workflow permission check.
- The pull-request Gitleaks scan checks PR commits from `origin/<base>..HEAD`
  instead of scanning trusted-base history already outside this PR.

### Files touched

- `.github/workflows/maturity_sweep_deflection_content_ops.yml`
- `.github/workflows/security_guardrails.yml`
- `plans/PR-Maturity-Sweep-Deflection-AI-Content-Ops-Lanes.md`
- `scripts/check_deflection_product_surface_manifest.py`
- `scripts/maturity_sweep_file_lane.py`
- `tests/test_deflection_product_surface_manifest.py`
- `tests/test_maturity_sweep_file_lane.py`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/maturity_sweep/baseline_deflection_lane.json`
- `tests/maturity_sweep/baseline_ai_content_ops_lane.json`

Affected surfaces:
- Deflection and AI content-ops Python files in `atlas_brain/**` and
  `scripts/**`.
- Deflection-named Python files in `extracted_content_pipeline/**`.
- Deployed/non-Python deflection product files in portfolio UI, atlas-intel UI,
  frontend docs/contracts, Vercel routing, and storage migrations.

Risk areas:
- CI enrollment/path-filter drift.
- Accidental overreach into unrelated root Atlas modules.
- Platform-specific path separators in generated baselines.
- Product route wiring changing without the manifest guard running.
- Security workflow startup failure if a reusable workflow permission is not
  passed through by the caller job.
- Pull-request secret scans failing on trusted-base history instead of only the
  PR's commits.

- Reviewer rules triggered: R2, R10, R12, R14.

## Mechanism

The companion workflow runs the existing maturity-sweep unit tests plus focused
fixture tests for the two new guard scripts, then checks the product-surface
manifest:

```bash
python scripts/check_deflection_product_surface_manifest.py
```

The manifest checker expands committed glob patterns and compares discovered
files with the explicit `files` list. Missing expected files or newly discovered
untracked files fail until the manifest is updated intentionally.

The deflection lane command collects deflection-named Python files from
`atlas_brain`, `scripts`, and `extracted_content_pipeline`, then runs:

```bash
python scripts/maturity_sweep_file_lane.py "${deflection_lane[@]}" \
  --tests-root tests \
  --baseline tests/maturity_sweep/baseline_deflection_lane.json \
  --min-score 8 \
  --sensitive-glob '*deflection*' \
  --sensitive-glob '**/billing/**' \
  --sensitive-glob '**/billing*' \
  --sensitive-glob '**/paid*' \
  --sensitive-glob '**/auth/**' \
  --sensitive-glob '**/auth*' \
  --sensitive-glob '**/webhook*' \
  --sensitive-glob '**/webhooks/**' \
  --sensitive-glob '**/payment*' \
  --sensitive-glob '**/invoicing/**' \
  --sensitive-glob '**/*invoice*' \
  --sensitive-glob '**/*deletion*'
```

The AI content-ops lane command collects non-deflection `content_ops` Python
files from `atlas_brain` and `scripts`, then runs the same wrapper with
`tests/maturity_sweep/baseline_ai_content_ops_lane.json` and a lane-wide
`*content_ops*` sensitive glob.

The Security Guardrails OSV reusable workflow caller now passes through the
read-only `actions: read` permission requested by the reusable workflow. The PR
Gitleaks job checks out the PR head SHA, fetches the trusted base ref, and scans
`origin/<base>..HEAD`; the trusted full-history scan remains reserved for main,
scheduled, and manual trusted-ref runs.

## Intentional

- `extracted_content_pipeline/**` remains covered by its existing broad gate;
  this workflow only triggers on deflection-named extracted Python files to keep
  this lane focused.
- The non-Python production product surface is a manifest/enrollment check, not
  a structural Python maturity score.
- The AI content-ops lane excludes deflection-named files so it does not
  double-own the deflection-specific gate.
- This does not enroll every script or every root `atlas_brain` module; those
  remain separate slices if needed.
- The Security Guardrails permission change is read-only and limited to the
  trusted OSV reusable workflow caller.
- The PR Gitleaks range is intentionally limited to PR commits; trusted-base
  baseline growth still belongs in a labeled security-rotation PR.

## Deferred

- Full `scripts/**` maturity-sweep enrollment remains a larger script-specific
  baseline slice.

## Parked hardening

- None.

## Verification

- `python scripts/maturity_sweep_file_lane.py <deflection files> --tests-root tests --baseline tests/maturity_sweep/baseline_deflection_lane.json --update-baseline` - pass, 21 files before adding the manifest checker baseline entry.
- `python scripts/maturity_sweep_file_lane.py <ai content ops files> --tests-root tests --baseline tests/maturity_sweep/baseline_ai_content_ops_lane.json --update-baseline` - pass, 68 files.
- `python scripts/check_deflection_product_surface_manifest.py` - pass, 32 files after adding route wiring.
- `python -m py_compile scripts/maturity_sweep_file_lane.py scripts/check_deflection_product_surface_manifest.py tests/test_maturity_sweep_file_lane.py tests/test_deflection_product_surface_manifest.py` - pass.
- `python -m pytest tests/test_maturity_sweep_file_lane.py tests/test_deflection_product_surface_manifest.py -q` - pass.
- Deflection lane ratchet command - pass.
- AI content-ops lane ratchet command - pass.
- Security Guardrails prior startup failure matched the OSV reusable workflow permission issue fixed by granting `actions: read` to the OSV caller.
- Security Guardrails PR secret scan is scoped to the PR head range, matching the green #1704 follow-up.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_deflection_content_ops.yml` | 136 |
| `.github/workflows/security_guardrails.yml` | 11 |
| `plans/PR-Maturity-Sweep-Deflection-AI-Content-Ops-Lanes.md` | 196 |
| `scripts/check_deflection_product_surface_manifest.py` | 72 |
| `scripts/maturity_sweep_file_lane.py` | 126 |
| `tests/test_deflection_product_surface_manifest.py` | 73 |
| `tests/test_maturity_sweep_file_lane.py` | 102 |
| `tests/maturity_sweep/baseline_ai_content_ops_lane.json` | 394 |
| `tests/maturity_sweep/baseline_deflection_lane.json` | 148 |
| `tests/maturity_sweep/deflection_product_surface_manifest.json` | 52 |
| **Total** | **1310** |
