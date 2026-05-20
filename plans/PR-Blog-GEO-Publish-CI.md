# PR-Blog-GEO-Publish-CI

## Why this slice exists

PR-Blog-GEO-Publish-Check added a local verifier for public blog prerender
metadata. The next step is making that check run automatically when UI blog
publishing code changes, so crawler-visible GEO metadata cannot regress without
showing up in GitHub checks.

This slice wires the Atlas Intel UI build and blog GEO verifier into GitHub
Actions.

## Scope (this PR)

1. Add a focused GitHub Actions workflow for `atlas-intel-ui`.
2. Trigger it on UI package, source, public asset, and Vite config changes.
3. Install Node dependencies with `npm ci`.
4. Run the production build.
5. Run the blog GEO prerender verifier against the build output.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Publish-CI.md` | Plan doc for this slice. |
| `.github/workflows/atlas_intel_ui_checks.yml` | Add Atlas Intel UI build + blog GEO verifier workflow. |

## Mechanism

The workflow runs on pull requests and pushes to `main` when relevant
`atlas-intel-ui` files or the workflow itself change. It uses Node 20, installs
from `atlas-intel-ui/package-lock.json`, builds the Vite app, and then runs
`npm run verify:blog-geo`.

## Intentional

- No `npm run lint` in this workflow yet. Lint currently fails on existing
  `no-explicit-any` violations in `atlas-intel-ui/src/pages/b2b/B2BReports.tsx`,
  which is outside this slice.
- No changes to Vercel deployment behavior.
- No backend/extracted pipeline workflow changes.

## Deferred

- Add lint after the existing B2BReports TypeScript violations are cleaned up.
- Add frontend unit tests if/when the UI gets a test runner.

## Verification

- Workflow YAML parsed successfully.
- Atlas UI production build passed.
- Blog GEO prerender verification passed across 14 blog pages.
- Whitespace diff check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~55 |
| Workflow | ~45 |
| **Total** | **~100** |
