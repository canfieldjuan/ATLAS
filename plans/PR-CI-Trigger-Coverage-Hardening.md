# PR-CI-Trigger-Coverage-Hardening

## Why this slice exists

A 30-workflow audit found two systemic path-trigger gaps that let CI go green on
changes it should test:

1. **`requirements.txt` is not a trigger on 15 workflows that install it.** Each
   does `pip install -r requirements.txt`, but `requirements.txt` is absent from
   their `pull_request.paths` / `push.paths`, so a dependency bump that breaks
   those suites does not retrigger them. This is the exact class that bit #1556:
   FastAPI 0.137.0 (unpinned at the time) broke route introspection, and only
   `atlas_content_ops_generated_assets_checks.yml` had `requirements.txt` in its
   paths, so the rest stayed green on the dependency change. The pin (#1559)
   stopped that one release; this closes the detection gap repo-wide so the next
   drift is caught on the dep-change PR, not three PRs later.

2. **~7 content-ops workflows do not trigger on the `extracted_content_pipeline`
   source their own tests exercise.** Their tests import `control_surfaces.py`,
   `content_ops_execution.py`, the macro-writeback modules, etc., but the
   workflow `paths:` watch only a handful of brain-side host adapters. The change
   is still caught by `extracted_pipeline_checks.yml`'s `**` glob, but the
   *named* gate stays green, giving a false "the specific check passed" signal on
   the PR (the same misleading-green that hid the #1556 break).

This is a pure trigger-coverage fix: no test logic, no source logic, no new
checks. It only makes existing checks fire when the files they depend on change.

## Scope (this PR)

Ownership lane: ci/trigger-coverage
Slice phase: Production hardening

1. Add `requirements.txt` to `pull_request.paths` and `push.paths` of the 15
   workflows that install it but do not trigger on it:
   admin_costs, atlas_blog_public, atlas_brand_voice, atlas_content_ops_auth,
   atlas_content_ops_claim_registry, atlas_content_ops_deflection_delivery,
   atlas_content_ops_deflection_report, atlas_content_ops_deflection_stripe_paid,
   atlas_content_ops_input_provider, atlas_content_ops_macro_writeback,
   atlas_content_ops_review_workflow, atlas_invoicing, atlas_main_voice_startup,
   extracted_competitive_intelligence, extracted_llm_infrastructure.
2. Add `extracted_content_pipeline/**` to the `paths:` of the content-ops
   workflows whose tests import package source not currently triggered:
   atlas_content_ops_input_provider, atlas_content_ops_macro_writeback,
   atlas_content_ops_review_workflow, atlas_content_ops_deflection_report,
   atlas_content_ops_deflection_stripe_paid, atlas_content_ops_claim_registry,
   atlas_content_ops_generated_assets. Use the `**` glob (not a hand-list of
   modules) so the trigger does not drift as imports change.
3. Add `atlas_brain/api/__init__.py` + `atlas_brain/config.py` to
   atlas_main_voice_startup (the broad-import surface `main.py` loads).

### Review Contract

- Acceptance criteria:
  - [ ] Each of the 15 workflows has `requirements.txt` in BOTH `pull_request`
        and `push` path lists.
  - [ ] Each listed content-ops workflow triggers on
        `extracted_content_pipeline/**`.
  - [ ] No workflow `jobs:`/`steps:`/runner content changed -- triggers only.
  - [ ] All edited YAML parses (CI lints / GitHub accepts the workflows).
- Affected surfaces: the GitHub Actions workflow files listed under Files
  touched (path filters only).
- Risk areas: a too-broad trigger causes extra (not fewer) CI runs -- acceptable
  / fail-safe; YAML indentation under `paths:`; not removing any existing
  trigger.
- Reviewer rules triggered: R2 (test/CI evidence), R10 (maintainability), R12
  (CI enrollment), R14.

### Files touched

- `.github/workflows/admin_costs_checks.yml`
- `.github/workflows/atlas_blog_public_checks.yml`
- `.github/workflows/atlas_brand_voice_checks.yml`
- `.github/workflows/atlas_content_ops_auth_checks.yml`
- `.github/workflows/atlas_content_ops_claim_registry_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `.github/workflows/atlas_main_voice_startup_checks.yml`
- `.github/workflows/extracted_competitive_intelligence_checks.yml`
- `.github/workflows/extracted_llm_infrastructure_checks.yml`
- `plans/PR-CI-Trigger-Coverage-Hardening.md`

## Mechanism

Each change appends entries to existing `on.pull_request.paths` and
`on.push.paths` lists. `requirements.txt` makes a dependency bump retrigger the
suites that install it; `extracted_content_pipeline/**` makes a package-source
change retrigger the named content-ops gate that tests it. Triggers are additive
and fail-safe -- the worst case is an extra CI run, never a skipped one. No job,
step, install, or test content changes.

## Intentional

- Use `extracted_content_pipeline/**` rather than enumerating modules, so the
  trigger tracks the import graph automatically (the hand-list was the original
  source of drift).
- Leave `requirements.asr.txt` alone except where a workflow installs it
  (generated-assets already lists it; no other workflow installs it).
- Do not touch always-run workflows (no `paths:` filter) or jobs/steps -- this is
  strictly a trigger-surface change.

## Deferred

- **Redaction-guard enrollment** (`tests/test_docs_no_raw_deflection_request_ids.py`
  from #1578) -- owned by that PR's lane; not duplicated here to avoid a
  cross-session collision.
- **Orphan-gate enrollment** -- `audit_content_ops_marketing_claims.py` (the
  #1503 deflection-claim gate) and an `extracted_evidence_to_story` workflow
  (the package has zero CI) are real gaps but are *new lanes*, not trigger
  coverage; they belong in a separate `ci/orphan-gate-enrollment` slice.
- LOW path-trigger items (script-self-edit smokes, the config->reasoning seam)
  -- not worth the trigger churn.

Parked hardening: none.

## Verification

- `python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]"`
  -- all workflows parse.
- Per-workflow grep: `requirements.txt` present in pull_request + push paths for
  all 15; `extracted_content_pipeline/**` present for the 7 content-ops gates.
- Confirm `git diff` touches only `on:`/`paths:` lines -- no `jobs:`/`steps:`.

## Estimated diff size

| File group | LOC |
|---|---:|
| 15 workflows x `requirements.txt` (pull_request + push) | ~30 |
| 7 content-ops workflows x `extracted_content_pipeline/**` (x2) | ~15 |
| atlas_main_voice_startup api/config triggers | ~4 |
| `plans/PR-CI-Trigger-Coverage-Hardening.md` | ~115 |
| **Total** | **~164** |
