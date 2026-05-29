# FAQ Deflection Report Rule File Docs

## Why this slice exists

The deflection report CLI now supports output-check gating, result JSON,
custom intent rules, and reusable JSON rule files, but the operator docs still
only show the lower-level FAQ Markdown CLI. Operators need a copy-pasteable
report command for the $1,500 deliverable path, including the fail-closed
behavior added before this slice.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Product polish

1. Add README guidance for generating the customer-facing deflection report
   artifact from support-ticket rows.
2. Add host-runbook guidance with the same command shape.
3. Document rule-file reuse, result JSON, and output-check gating at the report
   CLI boundary.

### Files touched

| File | Purpose |
|---|---|
| `extracted_content_pipeline/README.md` | Add operator-facing deflection report CLI example and behavior notes. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Add host runbook command for the report deliverable. |
| `plans/PR-FAQ-Deflection-Report-Rule-File-Docs.md` | Plan and verification contract. |

## Mechanism

The docs place the report CLI immediately after the existing reusable FAQ rule
file section. Both examples use:

```bash
python scripts/build_content_ops_deflection_report.py ...
```

with `--rule-file`, `--result-output`, `--summary-output`,
`--require-output-checks`, and `--output`. The prose names the fail-closed
behavior: result JSON is written before the gate exits, while customer-facing
Markdown is withheld when output checks fail.

## Intentional

- Docs only. The CLI behavior was implemented and tested in the prior code
  slices.
- The examples reuse the checked synthetic SaaS support-ticket rows and checked
  JSON rule file so operators can run them locally without credentials.
- No hosted-route runbook changes here; this is the local deliverable artifact
  path, not the #1075 hosted search route.

## Deferred

- Parked hardening: none.
- Future production-hardening slice: add deployed-host report build/runbook
  proof only after the operator-provisioned host is available.

## Verification

- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-Rule-File-Docs.md
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-Rule-File-Docs.md
- Command: python scripts/audit_plan_doc_diff_size.py plans/PR-FAQ-Deflection-Report-Rule-File-Docs.md origin/main
- Command: rg -n "build_content_ops_deflection_report.py|deflection-report-result|Deflection report" extracted_content_pipeline/README.md extracted_content_pipeline/docs/host_install_runbook.md
- Command: git diff --check
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-report-rule-file-docs.md

## Estimated diff size

| Area | Estimate |
|---|---:|
| `extracted_content_pipeline/README.md` | 24 LOC |
| `extracted_content_pipeline/docs/host_install_runbook.md` | 20 LOC |
| `plans/PR-FAQ-Deflection-Report-Rule-File-Docs.md` | 75 LOC |
| **Total** | **119 LOC** |
