# PR - Content-Ops Review Service Gate Rows

## Why this slice exists

PR #1374 landed pure coverage-row adapters for deterministic quality reports
and brand-voice audits. The host review workflow service still only consumes
caller-supplied `CoverageRow` values, so service callers would have to run the
adapter themselves. This slice threads the adapter into the existing service
boundary while keeping the service the single wrapper around `review_verdict`.
Review feedback on #1377 found one silent-approval edge in the adapter: decoded
quality-gate envelopes with `passed=True` plus a blocking `decision`, `verdict`,
or `outcome` field could approve. This update keeps that rejection at the pure
adapter boundary so future callers fail closed too.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

Wire deterministic gate evidence through the host review service:

1. Add optional quality-report and brand-voice evidence fields to
   `ContentOpsReviewRequest`.
2. Append adapter-built rows to caller-supplied coverage rows before building
   the `ContentPR`.
3. Preserve existing caller-supplied coverage behavior and registry handling.
4. Prove pass, failure, malformed-evidence, and public brand-voice metadata
   paths through the service.
5. Reject contradictory quality-gate envelopes before they can emit the sole
   required pass row.

### Review Contract

- Acceptance criteria:
  - [x] A passing quality report can satisfy required coverage without a
        hand-written row.
  - [x] Caller-supplied coverage rows are preserved.
  - [x] Blocker quality findings require revision through service verdicts.
  - [x] Malformed supplied quality evidence blocks as unresolved coverage.
  - [x] Contradictory decoded quality-gate evidence blocks as unresolved
        coverage.
  - [x] Public `brand_voice_audit` metadata produces a pass row through the
        service.
  - [x] Brand-voice warnings require revision through service verdicts.
  - [x] Existing tenant scope and registry-reader behavior is unchanged.
  - [x] No MCP transport, LLM, DB migration, or generated-asset mutation is
        added.
- Affected surfaces: host review service, coverage-row adapter, CI-covered
  service and adapter tests.
- Risk areas: silent approval, decoded input robustness, backcompat.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `atlas_brain/_content_ops_review_workflow.py`
- `extracted_content_pipeline/coverage_rows.py`
- `tests/test_atlas_content_ops_review_workflow.py`
- `tests/test_extracted_content_coverage_rows.py`
- `plans/PR-Content-Ops-Review-Service-Gate-Rows.md`

## Mechanism

`ContentOpsReviewRequest` gains two optional deterministic evidence inputs:
quality reports as a tuple and a brand-voice payload. The service converts those
inputs with `quality_gate_coverage_rows` and `brand_voice_coverage_rows`, appends
the resulting rows after `request.coverage`, and passes the merged coverage into
`ContentPR`.

Absent optional evidence does not add rows, preserving existing callers.
Supplying malformed evidence deliberately adds unresolved rows through the
adapter so the verdict blocks instead of approving silently.
The quality-gate adapter also checks decoded blocking-signal fields on otherwise
passing reports and emits an unresolved `contradictory-*` row when the envelope
claims both pass and block.

## Intentional

- This does not run quality packs; callers still choose the asset-specific pack.
- This does not persist `ContentPR` state or mutate generated assets.
- Caller-supplied coverage rows remain first in the merged matrix for stable
  output order.
- MCP transport and tenant/OAuth binding stay deferred.
- Blocking-signal detection is limited to explicit quality-gate envelope fields;
  `decision=warn` remains a nonblocking near-miss because warning findings are
  already represented as optional resolved rows.

## Deferred

- `PR-Content-Ops-Tenant-Binding-Bridge`: reconcile connector tenant binding
  with `TenantScope`.
- `PR-Marketer-Verification-MCP`: expose verify-only marketer tools after the
  remaining service and tenant seams are wired.
- Parked hardening: none expected.

## Verification

- Command: python -m pytest tests/test_atlas_content_ops_review_workflow.py -q
  - Passed: 17 tests.
- Command: python -m pytest tests/test_extracted_content_coverage_rows.py tests/test_atlas_content_ops_review_workflow.py -q
  - Passed: 28 tests.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py
  - Passed: 156 matching tests are enrolled.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed: mapped files match Atlas sources; hard Atlas imports clean.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed: clean.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed: Atlas runtime import findings 0.
- Command: bash scripts/check_ascii_python.sh
  - Passed: extracted content pipeline Python files are ASCII.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Passed: 3338 tests, 10 skipped.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content_ops_review_service_gate_rows_pr_body.md
  - Passed: local PR review passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_review_workflow.py` | 32 |
| `extracted_content_pipeline/coverage_rows.py` | 41 |
| `tests/test_atlas_content_ops_review_workflow.py` | 140 |
| `tests/test_extracted_content_coverage_rows.py` | 27 |
| `plans/PR-Content-Ops-Review-Service-Gate-Rows.md` | 125 |
| **Total** | **365** |
