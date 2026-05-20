# PR-Atlas-UI-Lint-B2B-Reports

## Why this slice exists

PR-Blog-GEO-Publish-CI left lint out of the Atlas Intel UI workflow because existing no-explicit-any violations in atlas-intel-ui/src/pages/b2b/B2BReports.tsx blocked the lint command.

This slice removes those lint blockers and adds lint to the UI workflow so future UI changes get lint coverage alongside build and blog GEO publish verification.

## Scope (this PR)

1. Replace explicit any annotations in B2BReports.tsx with a local loose JSON alias.
2. Keep the dynamic report renderer behavior unchanged.
3. Add the UI lint step to the Atlas Intel UI GitHub Actions workflow.
4. Verify lint, build, and blog GEO prerender checks locally.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Atlas-UI-Lint-B2B-Reports.md` | Plan doc for this slice. |
| `atlas-intel-ui/src/pages/b2b/B2BReports.tsx` | Remove explicit any lint violations. |
| `.github/workflows/atlas_intel_ui_checks.yml` | Add the UI lint step. |

## Mechanism

B2BReports.tsx is a dynamic renderer over loosely-shaped report payloads. Instead of fully modeling every payload shape in this slice, it uses a local LooseJson alias and keeps the existing defensive runtime checks.

The workflow now runs the lint command after dependency installation and before the build and blog GEO verifier.

## Intentional

- No UI behavior changes.
- No attempt to fully model every B2B report payload shape.
- No changes to backend or extracted pipeline workflows.

## Deferred

- Add stronger typed report payload models if the B2B report UI gets a larger refactor.

## Verification

- Atlas UI lint passed.
- Atlas UI production build passed.
- Blog GEO prerender verification passed.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| B2BReports type cleanup | ~10 |
| Workflow lint step | ~5 |
| Total | ~70 |
