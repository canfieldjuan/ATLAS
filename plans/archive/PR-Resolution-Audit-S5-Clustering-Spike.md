# PR-Resolution-Audit-S5-Clustering-Spike

## Why this slice exists

Issue #1993 S5 tracks the central Resolution Audit CSV clustering concern:
current code can both undercount repeated customer questions and over-merge
distinct workflows. The older audit findings are useful history, but this slice
reconstructs the behavior from current code before any algorithm change.

### Problem-derived contract

Root cause: `build_support_ticket_input_package` assigns deterministic
token-set clusters, and `build_ticket_faq_markdown` treats
`support_ticket_cluster` as the FAQ topic before question similarity and
embedding logic runs. A correct implementation fix must change that upstream
partition contract or add a deterministic cross-partition merge while leaving
buyer-facing report/snapshot/email/PDF/landing shape untouched. This slice does
not implement the fix; it pins current behavior and enrolls the fixtures in CI.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Functional validation

1. Add current-code S5 spike tests using the real support-ticket package
   builder, FAQ markdown builder, and token-set clusterer.
2. Enroll those spike tests in the explicit extracted-pipeline CI list and
   workflow path triggers so the fixture gates follow-up clustering changes.
3. Document the observed current behavior and next-slice constraints in the
   Resolution Audit CSV audit folder.
4. Update the #1993 remediation tracker now that S4 has merged and S5 is in
   spike/proof mode.
5. Archive the merged S4 plan while keeping this S5 plan in-flight.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `docs/audits/resolution-audit-csv/S5_CLUSTERING_SPIKE.md`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-S5-Clustering-Spike.md`
- `plans/archive/PR-Resolution-Audit-S4-Money-Reconciliation.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_resolution_audit_s5_clustering_spike.py`

### Review Contract

- Acceptance criteria:
  - The S5 tests prove same-intent SSO rows currently drop below the repeat
    gate through real package + FAQ builders.
  - The S5 tests prove a fake embedding port cannot cross the hard
    `support_ticket_cluster` topic partition.
  - The S5 tests prove cancel-subscription and cancel-order rows can currently
    produce one mixed buyer-visible row.
  - The S5 tests prove current output/source ordering follows input order.
  - The S5 tests or existing scale coverage keep the token-set skip boundary
    visible.
  - The new S5 test file is enrolled in the extracted-pipeline CI runner and
    workflow path trigger.
- Affected surfaces: tests, audit docs, remediation tracker, plan archive only.
- Risk areas: do not turn current-behavior spike assertions into a product
  algorithm change; do not change buyer-visible report/snapshot/email/PDF
  shape.
- Triggered reviewer rules: R1 requirements match, R2 test evidence, R10
  documentation drift, R13 adversarial coverage, R14 codebase verification.

## Mechanism

The spike tests run small synthetic support-ticket sets through the public
Resolution Audit builders. They pin five current behaviors: SSO/SAML rows split
below the repeat gate; the embedding port only sees rows inside one hard topic;
cancel-subscription and cancel-order can produce one mixed buyer row; reversing
input reverses equal-score output/source order; and the token-set skip leaves
above-threshold preview rows uncategorized with diagnostics.

The audit note summarizes those observations and declares the next-slice
constraints. The tracker marks S4 complete and leaves S5 as a spike rather than
claiming a clustering fix. The workflow/script enrollment makes the spike
load-bearing for the follow-up implementation rather than a local-only proof.

## Intentional

- No clustering algorithm change in this PR. The spike is the contract for the
  implementation slice; changing thresholds here would mix proof and fix.
- No buyer-facing shape change. The report, snapshot, landing page, email, PDF,
  and copy are intentionally untouched.
- The tests assert current broken behavior by design. The implementation PR
  should replace those expectations with corrected acceptance behavior.
- The tokenizer `es` suffix hardening entry in `HARDENING.md` was considered
  but left parked; it is not load-bearing for the hard-partition root cause.

## Deferred

- S5 implementation: change the upstream clustering/partition contract using
  this spike's fixtures as acceptance coverage.
- F7/service embedding-port forwarding remains a separate implementation
  concern if the accepted S5 design needs a runtime embedding port.
- S6 text/comment/outcome hygiene and S7 date-window policy remain separate
  slices in the #1993 queue.

Parked hardening: none.

## Verification

- Passed:
  focused pytest suite for `tests/test_resolution_audit_s5_clustering_spike.py`
  and `tests/test_extracted_support_ticket_clustering_scale.py`.
- Passed: extracted pipeline CI enrollment audit.
- Passed: local PR review bundle (`scripts/local_pr_review.sh`) with the
  planned PR body file.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 14 |
| `docs/audits/resolution-audit-csv/S5_CLUSTERING_SPIKE.md` | 43 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Resolution-Audit-S5-Clustering-Spike.md` | 124 |
| `plans/archive/PR-Resolution-Audit-S4-Money-Reconciliation.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_resolution_audit_s5_clustering_spike.py` | 204 |
| **Total** | **391** |
