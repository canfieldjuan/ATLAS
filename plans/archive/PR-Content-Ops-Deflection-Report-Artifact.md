# Content Ops Deflection Report Artifact

## Why this slice exists

The hosted FAQ search route is parked on operator-side provisioning, but the
$1,500 one-time deflection report does not depend on that route. The already
validated generation pipeline can rank repeated support questions, distinguish
resolution-backed answers from review-needed gaps, and emit source IDs.

This slice turns that pipeline output into the first customer-facing report
artifact shape: ranked question opportunities, drafted answers only when real
resolution evidence exists, and a clear "no proven answer yet" list for gaps
that still need a verified support answer.

## Scope (this PR)

Ownership lane: content-ops/deflection-report
Slice phase: Vertical slice

1. Add a deterministic deflection-report renderer for
   `TicketFAQMarkdownResult` objects.
2. Add a CLI that loads a support-ticket CSV/JSON/JSONL file, runs the existing
   FAQ generator, and writes the customer-facing report Markdown.
3. Add focused tests proving resolution-backed items land in the drafted-answer
   section, draft-needed items land in the no-proven-answer section, and the CLI
   runs end to end against the checked SaaS support-ticket corpus.

### Files touched

- `plans/PR-Content-Ops-Deflection-Report-Artifact.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/manifest.json`
- `scripts/build_content_ops_deflection_report.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

The renderer accepts the canonical FAQ result shape already produced by
`build_ticket_faq_markdown(...)`. It partitions items by
`answer_evidence_status`:

```python
resolution_evidence -> Drafted Answers With Proven Solutions
draft_needs_review -> No Proven Answer Yet
```

The CLI uses the same source loader as the FAQ Markdown CLI, calls
`build_ticket_faq_markdown(...)`, renders the report, and optionally writes a
compact JSON summary. The report stays deterministic and does not call a model.

## Intentional

- This does not use the hosted FAQ search route or persisted search projection.
  The report artifact is pipeline-only.
- This does not add UI/API selection for saved FAQ reports; another open PR is
  already working near that lane.
- The checked SaaS corpus is labeled synthetic. It proves the artifact flow, not
  a real customer acceptance run.

## Deferred

- Future PR: run this artifact against a real anonymized customer ticket export
  once one is available.
- Future PR: expose this report artifact through the hosted UI/API after the
  CLI shape is accepted.
- Future PR: add saved-FAQ selection only after the selection lane lands.
- Parked hardening: none. `HARDENING.md` was scanned; no relevant active
  `content-ops/deflection-report` item exists.

## Verification

Ran locally:

- `python -m pytest tests/test_content_ops_deflection_report.py -q` - 4
  passed.
- `python -m py_compile extracted_content_pipeline/faq_deflection_report.py scripts/build_content_ops_deflection_report.py tests/test_content_ops_deflection_report.py`
  - passed.
- `python scripts/build_content_ops_deflection_report.py extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --source-format csv --output /tmp/content-ops-deflection-report.md --summary-output /tmp/content-ops-deflection-report-summary.json`
  - passed; generated 7 ranked FAQ opportunities from 36 source rows, with 7
    no-proven-answer gaps and no drafted answers because the checked corpus has
    no resolution evidence. The focused pytest also covers a mixed
    resolution-backed/no-proven corpus end to end through the CLI.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py` - passed; 126
  matching tests are enrolled.
- `python -m pytest tests/test_content_ops_deflection_report.py tests/test_audit_extracted_pipeline_ci_enrollment.py -q`
  - 12 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline`
  - passed; synced mapped files only.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed; 2662 passed, 9
  skipped, 1 warning.
- `python scripts/audit_plan_doc.py plans/PR-Content-Ops-Deflection-Report-Artifact.md`
  - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-Deflection-Report-Artifact.md`
  - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-deflection-report-artifact.md`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Renderer | 302 |
| CLI | 141 |
| Tests | 210 |
| Manifest, runner, and plan | 119 |
| **Total** | **772** |

This is over the 400 LOC soft cap because the vertical slice needs a reusable
renderer, a runnable operator command, and end-to-end proof in one PR. Splitting
the CLI from the renderer would leave the artifact unable to demonstrate the
real flow.
