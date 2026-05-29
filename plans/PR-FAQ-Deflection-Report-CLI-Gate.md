# FAQ Deflection Report CLI Gate

## Why this slice exists

The customer-facing deflection report CLI can build the $1,500 report artifact,
but it does not yet expose the same fail-closed output-check gate as the
underlying FAQ generator CLI. That leaves operator runs able to write a
customer-facing report even when FAQ output checks fail.

This is a production-hardening slice because the end-to-end report path already
exists; the missing piece is making weak report output visible and blockable at
the operator boundary.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Production hardening

1. Add a deflection-report CLI `--require-output-checks` flag.
2. Add a compact `--result-output` JSON artifact that records status, failed
   checks, report summary, and item-level proof metadata without duplicating the
   full Markdown body.
3. Prove the CLI still succeeds for the SaaS demo source rows.
4. Prove weak output fails closed when the new gate is enabled and still writes
   the JSON result artifact before exiting.

### Files touched

| File | Purpose |
|---|---|
| `scripts/build_content_ops_deflection_report.py` | Add the CLI gate and compact result artifact. |
| `tests/test_content_ops_deflection_report.py` | Add success/failure coverage for the new CLI behavior. |
| `plans/PR-FAQ-Deflection-Report-CLI-Gate.md` | Plan and verification contract. |

## Mechanism

`main()` builds the existing `DeflectionReportArtifact`, computes failed output
checks from `artifact.faq_result.output_checks`, and writes `--result-output`
before enforcing `--require-output-checks`.

When required checks fail, the command raises `SystemExit` before writing the
customer-facing Markdown output. The result artifact remains compact: it records
the output status, summary counts, failed checks, and per-item counts/source ID
proofs, but not the full Markdown body or full answer text.

## Intentional

- No change to the default CLI behavior. Existing operator commands still write
  the report unless they opt into `--require-output-checks`.
- The fail-closed path writes the JSON result artifact but does not write the
  Markdown report. A failed gate should leave diagnostics, not a publishable
  customer artifact.
- The result artifact is smaller than the report body on purpose; the Markdown
  file remains the report artifact on success.

## Deferred

- Parked hardening: none.
- Future production-hardening slice: add a deployed-host report build runbook
  once the operator-provisioned host from issue #1075 is available. This slice
  stays on the local CLI boundary.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py -q
- Command: python -m py_compile scripts/build_content_ops_deflection_report.py tests/test_content_ops_deflection_report.py
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-CLI-Gate.md
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-CLI-Gate.md
- Command: python scripts/audit_plan_doc_diff_size.py plans/PR-FAQ-Deflection-Report-CLI-Gate.md origin/main
- Command: git diff --check
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-report-cli-gate.md

## Estimated diff size

| Area | Estimate |
|---|---:|
| `scripts/build_content_ops_deflection_report.py` | 121 LOC |
| `tests/test_content_ops_deflection_report.py` | 69 LOC |
| `plans/PR-FAQ-Deflection-Report-CLI-Gate.md` | 80 LOC |
| **Total** | **270 LOC** |
