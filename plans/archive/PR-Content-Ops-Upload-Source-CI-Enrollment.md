# PR: Content Ops Upload Source CI Enrollment

## Why this slice exists

Reviewer feedback on #1228 found that the upload-source handoff frontend tests
were present in `atlas-intel-ui/package.json` but not enrolled in the Atlas
Intel UI workflow. That made `atlas-intel-ui-checks` green without running the
two scripts that prove the upload/import handoff contract:
`test:content-ops-upload-source-run-handoff` and
`test:content-ops-ingestion-routing`.

This slice closes only that CI enrollment gap.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Workflow/process

1. Add the missing ingestion routing test step to the Atlas Intel UI workflow.
2. Add the missing upload-source run handoff test step to the Atlas Intel UI
   workflow.
3. Verify the workflow run list names both scripts so future green checks
   execute the #1228 frontend coverage.

### Files touched

- `.github/workflows/atlas_intel_ui_checks.yml`
- `plans/PR-Content-Ops-Upload-Source-CI-Enrollment.md`

## Mechanism

The workflow already installs dependencies, lints, runs targeted frontend tests,
builds, and verifies prerender artifacts. This PR inserts two additional
focused test steps before the existing landing/blog review tests:

```yaml
- name: Test Content Ops ingestion routing
  run: npm run test:content-ops-ingestion-routing

- name: Test Content Ops upload source run handoff
  run: npm run test:content-ops-upload-source-run-handoff
```

Both scripts already exist in `atlas-intel-ui/package.json`; this PR makes CI
execute them.

## Intentional

- No product code changes. The #1228 behavior already merged; this is the CI
  coverage enrollment follow-up.
- No new frontend test script. The reviewer identified enrollment as the gap,
  not missing coverage.
- The steps stay in the Atlas Intel UI workflow instead of the extracted suite
  because these are Node/UI scripts under `atlas-intel-ui/scripts`.

## Deferred

- A general audit that fails when changed `atlas-intel-ui/package.json` test
  scripts are not enrolled in `.github/workflows/atlas_intel_ui_checks.yml`
  remains deferred; this slice fixes the immediate #1228 MAJOR.
- Parked hardening: none.

## Verification

- `cd atlas-intel-ui && npm ci` - passed; npm reports the existing 6 audit
  findings.
- `cd atlas-intel-ui && npm run test:content-ops-ingestion-routing` - 4
  passed.
- `cd atlas-intel-ui && npm run test:content-ops-upload-source-run-handoff` -
  3 passed.
- `rg -n
  "test:content-ops-ingestion-routing|test:content-ops-upload-source-run-handoff"
  .github/workflows/atlas_intel_ui_checks.yml atlas-intel-ui/package.json` -
  confirmed both scripts are named in the workflow and `package.json`.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/content-ops-upload-source-ci-enrollment-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Workflow enrollment | ~6 |
| **Total** | **~91** |
