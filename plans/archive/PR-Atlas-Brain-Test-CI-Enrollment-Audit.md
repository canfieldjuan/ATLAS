# PR-Atlas-Brain-Test-CI-Enrollment-Audit

## Why this slice exists

AI Content Ops / FAQ deflection work has repeatedly exposed the same CI shape
failure: a test importing `atlas_brain.*` gets treated like extracted-lane
coverage, then GitHub Actions fails because the extracted lane is not a host
runtime lane. This recurred around #1097, #1154, and #1158, and the standing
rule now needs a mechanical check instead of relying on session memory.

The existing #957 extracted CI enrollment auditor verifies extracted-runner
test enrollment. It does not verify the host-test counterpart: changed
`atlas_brain.*` importing tests need a dedicated atlas area checks workflow
that includes the file in both PR/push path filters and in the pytest run step.

This slice exceeds the 400 LOC target after review feedback because the
guardrail needs focused negative fixtures for the false-green split-workflow
case and the inline `- run:` parser shape. Splitting that follow-up out would
leave this PR with a known detector gap.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Workflow/process

1. Extend `scripts/audit_extracted_pipeline_ci_enrollment.py` with an optional
   changed-test scan for `atlas_brain.*` imports.
2. Have local PR review pass the current base ref into that scan so the guard
   only applies to tests changed by the current slice.
3. Add fixture coverage for happy path and each missing-enrollment branch:
   missing workflow, missing PR path filter, missing push path filter, and
   missing pytest runner entry.
4. Require one atlas workflow to carry all three enrollments for a changed
   `atlas_brain.*` importing test instead of accepting coverage split across
   multiple workflows.
5. Leave existing extracted enrollment behavior unchanged.

### Files touched

- `plans/PR-Atlas-Brain-Test-CI-Enrollment-Audit.md`
- `scripts/audit_extracted_pipeline_ci_enrollment.py`
- `tests/test_audit_extracted_pipeline_ci_enrollment.py`
- `scripts/local_pr_review.sh`

## Mechanism

The auditor keeps its current all-repo extracted enrollment scan. A new CLI
option asks git for changed Python test files under `tests/` versus the base ref.
For those changed tests only, the auditor reads the test file, detects top-level
`atlas_brain` imports, and checks dedicated atlas-prefixed workflow files
for all three required enrollments:

```text
pull_request paths contains the test
push paths contains the test
workflow run text contains the exact pytest test path
```

The three checks must pass in the same atlas-prefixed workflow. If the PR path,
push path, and run step are split across different workflows, the auditor emits
a `DRIFT` failure naming the missing single-workflow enrollment. The workflow
run parser accepts both named-step `run:` entries and inline `- run:` entries.

## Intentional

- This is changed-file scoped. A whole-repo scan would turn historical host test
  coverage into unrelated debt for this workflow/process slice.
- The dedicated workflow is intentionally identified by atlas-prefixed checks
  workflow naming, matching the standing rule and avoiding accidental credit
  from extracted or UI lanes.
- The pytest runner check requires the exact test path in workflow text instead
  of inferring from shell globs; that mirrors the "same PR add the file to the
  pytest run-step" rule.
- The workflow parser only reads `run:` step text for test paths. It is not a
  YAML validator; malformed workflows still fail through the existing Actions
  and local review gates.

## Deferred

- Parked hardening: none. This slice directly promotes the standing CI
  enrollment rule into local review.
- A future workflow/process slice can migrate historical host tests into
  dedicated atlas workflows. This guard prevents new regressions without
  rewriting old CI ownership in the same PR.

## Verification

- `python -m pytest tests/test_audit_extracted_pipeline_ci_enrollment.py -q` - 17 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed, 135 matching tests enrolled.
- `python -m py_compile scripts/audit_extracted_pipeline_ci_enrollment.py tests/test_audit_extracted_pipeline_ci_enrollment.py` - passed.
- `git diff --check -- scripts/audit_extracted_pipeline_ci_enrollment.py tests/test_audit_extracted_pipeline_ci_enrollment.py scripts/local_pr_review.sh plans/PR-Atlas-Brain-Test-CI-Enrollment-Audit.md` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-brain-ci-enrollment-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~102 |
| Auditor | ~168 |
| Tests | ~203 |
| Local review hook | ~2 |
| **Total** | **~475** |
